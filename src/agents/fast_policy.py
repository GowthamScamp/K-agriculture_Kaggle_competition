"""Cheap rollout policy for search. Operates on the raw observation dict.

`BaselineAgent` costs ~750 us per call, which caps a 700 ms decision at about 19
simulated turns - far too few to search with. This policy plays the same *shape*
of game (water, harvest, deliver, sell, hire, plant, keep animals) for a small
fraction of the cost.

Three deliberate departures from the real agent, each measured:

* **No `View`.** Building one allocates a `Tile` object for all 100 board
  squares; at two views per simulated turn that alone dominated the profile.
  This policy reads the raw dicts the environment already holds.
* **One board scan per turn**, hoisted out of the per-unit loop. An earlier
  version rescanned the board inside the loop looking for free animal housing,
  costing ~85,000 property lookups per 300 turns.
* **No global unit/task matching and no replanning** - each unit greedily claims
  the nearest job, and macro-strategy is fixed by `FastParams`.

It is a *rollout* policy, not a competitor. Its job is to give comparable value
estimates across candidate macro-actions, so uniform mild pessimism is harmless.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..game_constants import (
    ANIMALS,
    CROPS,
    LAND_PRICES,
    MARKET_PARAMS,
    PRODUCTS,
    hire_cost,
    market_price,
    sell_revenue,
)

SHED_CAPACITY = 100
SEASON_DAYS = 30
MAX_MARKET_ORDERS = 10
Pos = tuple[int, int]


@dataclass(slots=True)
class FastParams:
    """Macro-strategy knobs. These are exactly what the search chooses between."""

    crop: str = "WHEAT"
    target_hands: int = 10
    buy_land: bool = True
    buy_animals: bool = True
    animal: str = "COW"
    sell_price_frac: float = 0.90
    sell_max_drop_frac: float = 0.88
    plants_per_unit: float = 6.5
    animal_share: float = 0.22
    max_animals: int = 14
    land_reserve: float = 400.0
    animal_reserve: float = 900.0

    def key(self) -> tuple:
        return (
            self.crop, self.target_hands, self.buy_land, self.buy_animals,
            self.animal, round(self.sell_price_frac, 3),
            round(self.plants_per_unit, 2), round(self.animal_share, 2),
            self.max_animals,
        )

    def label(self) -> str:
        bits = [f"{self.crop.lower()}", f"h{self.target_hands}"]
        if self.buy_animals:
            bits.append(self.animal.lower())
        else:
            bits.append("noanimal")
        if not self.buy_land:
            bits.append("noland")
        bits.append(f"p{self.plants_per_unit:g}")
        return "/".join(bits)


class FastPolicy:
    """Callable on a raw observation dict, returning a legal action."""

    __slots__ = ("p",)

    def __init__(self, params: FastParams | None = None) -> None:
        self.p = params or FastParams()

    def __call__(self, obs: Any, config: Any = None) -> dict[str, Any]:
        # `config` is accepted and ignored so this is usable both as a rollout
        # policy (called with one argument) and as a real agent, which the
        # environment always calls with two.
        try:
            return self._decide(obs)
        except Exception:
            farms = obs.get("farms") or []
            player = obs.get("player", 0)
            n = len(farms[player].get("hands") or []) if player < len(farms) else 0
            return {"farmer": ["PASS"], "hands": [["PASS"]] * n, "market": []}

    # ------------------------------------------------------------------
    def _decide(self, obs: Any) -> dict[str, Any]:
        p = self.p
        player = obs["player"]
        farm = obs["farms"][player]
        private = obs["private"]
        tiles = farm["tiles"]
        board = len(tiles)
        day = obs["day"]
        hour = obs["hour"]
        money = farm["money"]
        shed = private["shed"]
        seeds = private["seeds"]
        inventories = private["inventories"]
        market_inv = obs["market"]["inventory"]
        liquidating = day >= SEASON_DAYS - 2

        animal_data = ANIMALS[p.animal]
        want_structure_kind = animal_data["structure"]

        # --- single board scan -------------------------------------------
        water: list[Pos] = []
        harvest: list[Pos] = []
        empty: list[Pos] = []
        weeds: list[Pos] = []
        feed: list[Pos] = []
        free_housing: list[Pos] = []
        n_plants = 0
        n_animals = 0
        for y in range(board):
            row = tiles[y]
            for x in range(board):
                raw = row[x]
                if raw is None:
                    empty.append((x, y))
                    continue
                if not isinstance(raw, dict):
                    continue  # "LOCKED"

                kind = raw.get("kind")
                if kind == "PLANT":
                    n_plants += 1
                    if not raw["watered_today"]:
                        water.append((x, y))
                    if raw["yield_units"] > 0 and (
                        day - raw["planted_day"] >= CROPS[raw["crop"]]["first_yield_day"]
                    ):
                        harvest.append((x, y))
                elif "animal" in raw:
                    n_animals += 1
                    if not raw["fed_today"]:
                        feed.append((x, y))
                    if raw["yield_units"] > 0:
                        harvest.append((x, y))
                elif kind == "WEED":
                    weeds.append((x, y))
                elif kind == want_structure_kind:
                    free_housing.append((x, y))

        half = board // 2
        shed_tiles = [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]
        n_units = 1 + len(farm["hands"])
        shed_used = 0
        for val in shed.values():
            if val > 0:
                shed_used += val

        # --- market -------------------------------------------------------
        # Order slots are capped at 10 per turn and excess is silently dropped,
        # so they are budgeted by category rather than appended first-come.
        # Hiring eight hands used to crowd out the seed order entirely.
        sells: list[list[Any]] = []
        hires: list[list[Any]] = []
        buys: list[list[Any]] = []
        budget = money
        fill = shed_used / SHED_CAPACITY
        wheat_guard = n_animals * 3

        if day >= 2 or fill > 0.5:
            for item in PRODUCTS:
                held = shed.get(item, 0)
                if held <= 0:
                    continue
                if item == "WHEAT" and n_animals and not liquidating:
                    held -= wheat_guard
                    if held <= 0:
                        continue
                inv = market_inv.get(item, MARKET_PARAMS[item]["I0"])
                base = MARKET_PARAMS[item]["base"]
                spot = market_price(item, inv)
                if liquidating:
                    n = max(1, held // max(1, SEASON_DAYS - day))
                elif fill > 0.85:
                    n = held
                elif spot < base * p.sell_price_frac:
                    continue
                else:
                    floor = base * 0.70
                    rel = spot * p.sell_max_drop_frac
                    if rel > floor:
                        floor = rel
                    n = 0
                    cur = inv
                    while n < held and n < 30:
                        if market_price(item, cur) < floor:
                            break
                        n += 1
                        cur += 1
                if n > 0:
                    sells.append(["SELL", item, n if n < held else held])
                    if len(sells) >= 4:
                        break

        if not liquidating:
            if hour <= 3 and day < SEASON_DAYS - 1:
                want = min(p.target_hands, 13) - len(farm["hands"])
                done = farm["hires_today"]
                for i in range(max(0, min(want, 4))):
                    c = hire_cost(done + i)
                    if budget - c < 120:
                        break
                    budget -= c
                    hires.append(["HIRE"])

            extra = len(farm["unlocked_quadrants"]) - 1
            # Expand once most of the land we already hold is in use. The old
            # guard compared idle tiles against `n_plants`, which is zero on day
            # zero, so the policy could never buy its first quadrant.
            worked = n_plants + n_animals + len(free_housing)
            idle = len(empty) + len(weeds)
            if (
                p.buy_land
                and extra < len(LAND_PRICES)
                and day <= 21
                and budget >= LAND_PRICES[extra] + p.land_reserve
                and idle <= max(4, worked // 2)
            ):
                budget -= LAND_PRICES[extra]
                buys.append(["BUY_LAND"])

            capacity = int(n_units * p.plants_per_unit)
            if n_plants < capacity and empty:
                crop = p.crop
                have = seeds.get(crop, 0)
                want = min(len(empty), capacity - n_plants, 6) - have
                cost = CROPS[crop]["seed"]
                n = min(want, int((budget - 40) // cost) if budget > 40 else 0)
                if n > 0:
                    budget -= n * cost
                    buys.append(["BUY_SEED", crop, n])

            # Animals eat one wheat a day and escape after two hungry days, so
            # buying them before a wheat supply exists just burns 400 coins per
            # cow. An earlier version bought four on day zero and starved them all.
            wheat_supply = shed.get("WHEAT", 0) >= 2 + n_animals
            if (
                p.buy_animals
                and 3 <= day <= 20
                and free_housing
                and wheat_supply
                and n_animals < p.max_animals
            ):
                in_transit = shed.get(p.animal, 0)
                for inv_d in inventories:
                    in_transit += inv_d.get(p.animal, 0)
                if len(free_housing) > in_transit and budget >= animal_data["cost"] + p.animal_reserve:
                    buys.append(["BUY_ANIMAL", p.animal, 1])

        # --- units --------------------------------------------------------
        animal_cap = min(
            p.max_animals,
            max(1, int(p.animal_share * (n_plants + n_animals + len(free_housing) + 1))),
        )
        want_structure = (
            p.buy_animals and not liquidating and 2 <= day <= 20
            and (n_animals + len(free_housing)) < animal_cap
            and len(free_housing) < 2        # never more than a small buffer
            and money >= animal_data["cost"] + p.animal_reserve
        )
        structure_op = ["BUILD_PASTURE"] if want_structure_kind == "PASTURE" else ["BUILD_COOP"]

        carried_animal = shed.get(p.animal, 0) > 0
        any_carrying = any(iv.get(p.animal, 0) > 0 for iv in inventories)

        capacity = int(n_units * p.plants_per_unit)
        plant_budget = 0 if liquidating else min(
            seeds.get(p.crop, 0), capacity - n_plants if capacity > n_plants else 0
        )
        shed_room = SHED_CAPACITY - shed_used
        built = 0
        ops: list[list[Any]] = []

        for idx in range(n_units):
            pos = tuple(farm["farmer"]) if idx == 0 else tuple(farm["hands"][idx - 1])
            inv = inventories[idx] if idx < len(inventories) else {}
            carried = 0
            for val in inv.values():
                carried += val

            # Place a held animal.
            if inv.get(p.animal, 0) > 0 and free_housing:
                tgt = _nearest(pos, free_housing)
                if pos == tgt:
                    free_housing.remove(tgt)
                    ops.append(["PLACE", p.animal])
                else:
                    ops.append(_step(pos, tgt, board))
                continue

            # Deliver a full load, or anything at all late in the day.
            if carried and (carried >= 6 or hour >= 20) and shed_room > 0:
                tgt = _nearest(pos, shed_tiles)
                ops.append(["DROP"] if pos == tgt else _step(pos, tgt, board))
                continue

            # Feed, if holding wheat.
            if feed and inv.get("WHEAT", 0) > 0:
                tgt = _nearest(pos, feed)
                if pos == tgt:
                    feed.remove(tgt)
                    ops.append(["FEED"])
                else:
                    ops.append(_step(pos, tgt, board))
                continue

            # Shed errands: collect feed wheat, or collect a bought animal.
            if not carried:
                if carried_animal and not any_carrying and free_housing:
                    tgt = _nearest(pos, shed_tiles)
                    if pos == tgt:
                        any_carrying = True
                        ops.append(["PICKUP", p.animal, 1])
                        continue
                    ops.append(_step(pos, tgt, board))
                    continue
                if feed and shed.get("WHEAT", 0) > 0:
                    tgt = _nearest(pos, shed_tiles)
                    if pos == tgt:
                        ops.append(["PICKUP", "WHEAT", min(len(feed), shed["WHEAT"])])
                        continue
                    ops.append(_step(pos, tgt, board))
                    continue

            if water:
                tgt = _nearest(pos, water)
                if pos == tgt:
                    water.remove(tgt)
                    ops.append(["WATER"])
                else:
                    ops.append(_step(pos, tgt, board))
                continue

            if harvest:
                tgt = _nearest(pos, harvest)
                if pos == tgt:
                    harvest.remove(tgt)
                    ops.append(["HARVEST"])
                else:
                    ops.append(_step(pos, tgt, board))
                continue

            if want_structure and built < 1 and empty:
                tgt = _nearest(pos, empty)
                if pos == tgt:
                    empty.remove(tgt)
                    built += 1
                    ops.append(structure_op)
                else:
                    ops.append(_step(pos, tgt, board))
                continue

            if plant_budget > 0 and empty:
                tgt = _nearest(pos, empty)
                if pos == tgt:
                    empty.remove(tgt)
                    plant_budget -= 1
                    ops.append(["PLANT", p.crop])
                else:
                    ops.append(_step(pos, tgt, board))
                continue

            if weeds:
                tgt = _nearest(pos, weeds)
                if pos == tgt:
                    weeds.remove(tgt)
                    ops.append(["DIG"])
                else:
                    ops.append(_step(pos, tgt, board))
                continue

            ops.append(["PASS"])

        market = (sells + buys + hires)[:MAX_MARKET_ORDERS]
        return {"farmer": ops[0], "hands": ops[1:], "market": market}


# --------------------------------------------------------------------------
def _nearest(src: Pos, targets: list[Pos]) -> Pos:
    sx, sy = src
    best = targets[0]
    bd = abs(sx - best[0]) + abs(sy - best[1])
    for t in targets:
        d = abs(sx - t[0]) + abs(sy - t[1])
        if d < bd:
            best, bd = t, d
    return best


def _step(src: Pos, dst: Pos, board: int) -> list[Any]:
    dx = dst[0] - src[0]
    dy = dst[1] - src[1]
    if dx and (dx if dx > 0 else -dx) >= (dy if dy > 0 else -dy):
        return ["EAST"] if dx > 0 else ["WEST"]
    if dy:
        return ["SOUTH"] if dy > 0 else ["NORTH"]
    return ["PASS"]


# --------------------------------------------------------------------------
def fast_net_worth(obs: Any, liquidation: bool = False) -> float:
    """Shaped value of a position, computed straight off the raw dicts.

    Raw cash is the true objective, but over a horizon of a few days it rates
    every investment as a pure loss - seeds, land and animals all leave the bank
    before returning anything. This adds a conservative liquidation value so
    short rollouts can still tell a growing farm from a liquidating one.

    `liquidation` changes how *sellable stock* is priced, and only that:

    - `False` (default): every unit at the current market price. Simple, but it
      makes holding stock look free, because it ignores that dumping N units
      walks the price down. A search offered "sell less" then takes it - which is
      why `sell-patient` scored 1% over 100 seeds at max_candidates=6.
    - `True`: through `sell_revenue`, which re-quotes each unit as inventory
      rises, so stock is worth what it would actually fetch. Standing crops,
      animals, seeds and land keep their shaped treatment, because the reason
      shaping exists - letting a 2-day rollout see an investment - still applies.
      Measured: raw-bank scoring (no shaping at all) collapses to 14% at
      max_candidates=2, so removing shaping wholesale is not the answer.
    """
    player = obs["player"]
    farm = obs["farms"][player]
    private = obs["private"]
    market_inv = obs["market"]["inventory"]

    price: dict[str, int] = {}
    for item in PRODUCTS:
        price[item] = market_price(item, market_inv.get(item, MARKET_PARAMS[item]["I0"]))

    day = obs["day"]
    total = float(farm["money"])
    for item, n in private["shed"].items():
        if n <= 0:
            continue
        if item in price:
            total += (sell_revenue(item, int(n), market_inv.get(item, MARKET_PARAMS[item]["I0"]))
                      if liquidation else n * price[item])
        elif item in ANIMALS:
            total += n * ANIMALS[item]["cost"] * 0.5
    for inv in private["inventories"]:
        for item, n in inv.items():
            if n > 0 and item in price:
                total += (sell_revenue(item, int(n), market_inv.get(item, MARKET_PARAMS[item]["I0"]))
                          if liquidation else n * price[item])
    for crop, n in private["seeds"].items():
        if n > 0:
            total += n * CROPS[crop]["seed"] * 0.5

    for row in farm["tiles"]:
        for raw in row:
            # NB isinstance, not `type(...) is dict`: the live environment hands
            # out `Struct`, a dict subclass, and an identity check silently
            # classified every real tile as locked.
            if not isinstance(raw, dict):
                continue
            if raw.get("kind") == "PLANT":
                total += _standing_crop_value(raw, day, price)
            elif "animal" in raw:
                a = ANIMALS[raw["animal"]]
                total += _standing_animal_value(raw, a, day, price)
    return total


def _standing_crop_value(raw: Any, day: int, price: dict[str, int]) -> float:
    """What a growing plant is worth: what it will *harvest*, not what it cost.

    Valuing a standing crop at its seed price made short rollouts read every
    investment as a loss, so the search preferred hiring nobody and planting
    nothing - it could not see past its own horizon. Pricing the expected final
    yield instead makes labour pay for itself inside a two-day rollout, which is
    what lets the search evaluate strategy rather than thrift.
    """
    crop = raw["crop"]
    d = CROPS[crop]
    units = raw["yield_units"]
    age = day - raw["planted_day"]

    if d["ongoing"]:
        remaining_days = SEASON_DAYS - day
        future = max(0, min(d["max_yield"], remaining_days // max(1, d["interval"])))
        expected = units + future
    else:
        days_to_ripe = d["max_yield_day"] - age
        if days_to_ripe > SEASON_DAYS - 1 - day:
            return units * price[crop]  # cannot mature before the bell
        window_start = (d["max_yield_day"] + 1) // 2
        first_bonus = window_start if age < window_start else age + 1
        remaining_waters = max(0, d["max_yield_day"] - first_bonus + 1)
        expected = min(d["max_yield"], units + remaining_waters)
    return expected * price[crop]


def _standing_animal_value(
    raw: Any, data: dict[str, Any], day: int, price: dict[str, int]
) -> float:
    """Held product plus what the animal should still produce this season."""
    product = data["product"]
    value = raw["yield_units"] * price[product]
    age = day - raw["placed_day"]
    remaining = SEASON_DAYS - 1 - day
    if remaining > 0:
        start_gap = max(0, data["first_yield_day"] - age)
        producing = max(0, remaining - start_gap)
        value += (producing // max(1, data["interval"])) * price[product]
    return value
