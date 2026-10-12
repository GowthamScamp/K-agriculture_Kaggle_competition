"""Phase 2: hand-coded heuristic agent. The reference every later agent must beat.

Design notes worth reading before tuning anything. These are conclusions from
measured episodes, not guesses:

* **Labour is the whole game.** A hand costs `fib(n)` coins on the n-th hire of
  the day and the counter resets daily, so ten hands cost 143 coins total and
  each is a full extra action every turn. The first version of this agent hired
  almost none and spent 76% of its unit-turns walking; fixing that was worth more
  than every market heuristic combined.
* **Never plant what you cannot water.** A plant is created with
  `consecutive_unwatered = 1`, so it must be watered on its planting day and
  every day after, or it becomes a weed that *blocks the tile*. An over-planted
  farm is strictly worse than a small one: it wastes the seed, the action, and
  then the ground. Planting is therefore capped by watering capacity, not by
  available land.
* **Distance is the hidden cost.** Every unit respawns at the shed each morning,
  so cultivation is kept compact around the centre and units are matched to tasks
  by a global nearest-first assignment.
* **The market fights back, but slowly.** Town demand drains inventory all season,
  so prices drift *up* - observed strawberry 120 -> 293 by day 25. Selling is
  sized by marginal price rather than dumping the shed. MELON is the exception:
  no shop buys it and its glut curve is quadratic, so a melon dump is permanent.
* **Animals beat crops per action.** A cow needs one FEED plus an occasional
  HARVEST and returns ~1,760 coins; a wheat tile needs a watering every single
  day. Animals are how labour converts into money once the crop engine funds them.
* **Unsold goods score zero**, so the last days are a controlled liquidation.

Every tunable lives in `Params`. There are no bare numbers in the logic below.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..env_adapter import (
    Action,
    Pos,
    Tile,
    View,
    distance,
    safe_action,
    step_toward,
)
from ..game_constants import (
    ANIMALS,
    CROPS,
    LAND_PRICES,
    MARKET_I0,
    MARKET_PARAMS,
    PRODUCTS,
    hire_cost,
    marginal_sell_price,
    projected_sell_revenue,
    sell_revenue,
)

SEASON_DAYS = 30
TURNS_PER_DAY = 24
MAX_MARKET_ORDERS = 10
SHED_CAPACITY = 100


# ==========================================================================
# Tunables
# ==========================================================================
@dataclass(frozen=True, slots=True)
class Params:
    """Every heuristic threshold, named and in one place."""

    # -- labour ------------------------------------------------------------
    MAX_HANDS: int = 13
    """Hard cap on hands per day. fib(13) = 377 for the 14th alone, and the
    cumulative cost turns steep immediately after."""

    HIRE_BEFORE_HOUR: int = 4
    """A hand hired at dusk does nothing - it vanishes at end of day."""

    MAX_HIRES_PER_TURN: int = 8
    """Hiring competes with selling for the 10 market-order slots."""

    HIRE_CASH_RESERVE: float = 150.0
    FEED_WHEAT_CASH_RESERVE: float = 200.0
    """Never spend the last of the cash on feed; hiring and seed come first."""
    MIN_HANDS: int = 3
    """Always worth having, even on day 0 - the first three cost 4 coins total."""

    HIRE_FLOOR_IGNORES_RESERVE: bool = False
    """Let the first `MIN_HANDS` hires bypass `HIRE_CASH_RESERVE`.

    The reserve is compared against the bank *after* paying, so a reserve larger
    than the whole bank blocks hiring outright rather than merely throttling it.
    Measured on seed 7: the bank sits at 46 coins from day 1 to day 9, where a
    150 reserve permits **zero** hands - nine days farmed by the farmer alone -
    while the seven cheapest hands cost 33 coins in total. `MIN_HANDS` already
    documents the intent above; this is what makes it bind."""

    LIQUIDATION_HIRES: bool = False
    """Emit hire orders during liquidation.

    `_market_orders` returns early once liquidating, before hires are appended,
    so the closing days run on one unit exactly when harvesting and hauling to
    market need labour most."""

    LIQUIDATION_HAND_TARGET: int | None = None
    """Hand target while liquidating. `None` keeps the `MIN_HANDS` cap."""

    WORK_PER_UNIT_PER_DAY: float = 7.5
    """Useful (non-walking) actions one unit completes per day. Measured, not
    assumed: this is what sets both the hire target and the planting cap."""

    WORK_PER_PLANT_EXTRA: float = 0.45
    """Amortised harvest/plant cost per standing plant, on top of the one
    watering per plant per day that dominates the workload."""

    WORK_PER_ANIMAL: float = 1.4
    """Feed plus amortised produce collection. Higher than a plant's extra
    because an animal that misses two feeds escapes entirely."""

    WORK_PER_IDLE_TILE: float = 0.5
    """Digging and weeding an idle tile, amortised. Counted only up to the
    number of tiles actually owned."""

    # -- land --------------------------------------------------------------
    LAND_CASH_RESERVE: float = 400.0
    LAND_LAST_USEFUL_DAY: int = 21
    """Past this, a new quadrant cannot pay for itself before the season ends."""

    LAND_IDLE_TILE_LIMIT: float = 0.30
    """Do not buy more land while this fraction of what we own is still idle."""

    # -- planting ----------------------------------------------------------
    SEED_CASH_RESERVE: float = 40.0
    MIN_DAYS_MARGIN: int = 1
    """Only plant a crop whose full cycle finishes this many days before the end."""

    SEED_BUY_AHEAD: int = 8
    """Seeds bought per turn, sized to what is plantable soon."""

    CROP_VALUE_BLEND: float = 0.55
    """0 = rank crops purely by coins per tile-day, 1 = purely by coins per
    action. Land is scarce early and labour is scarce throughout, so this leans
    toward coins per action."""

    MELON_MAX_TILES: int = 8
    """MELON has no shop demand and a quadratic glut curve. Cap the acreage no
    matter how good the headline price looks."""

    REALIZED_SHOP_DEMAND: bool = False
    """Price future harvests against the shops actually open, not an average one.

    `unlocked_shops` is in every observation, and the town consumes from that
    exact list - single-product shops at double rate. Modelling it as an average
    shop is wrong by up to 8x (a town of YARN_STOREs pulls WOOL at 97 units/day
    where the average model says 13), and crop and animal choice are both ranked
    on that number.

    Defaulted **off** deliberately. `BaselineAgent` is this project's reference
    opponent as well as its executor, so turning this on unconditionally would
    move the yardstick every stored result is measured against. The candidate
    enables it explicitly; the default flips only if measurement says so."""

    PLANT_CAPACITY_SLACK: float = 0.85
    """Fraction of watering capacity actually committed to plants, leaving room
    for harvesting, planting and feeding in the same day."""

    # -- task priorities ---------------------------------------------------
    P_WATER_CRITICAL: float = 100.0
    P_FEED: float = 92.0
    P_HARVEST_URGENT: float = 84.0
    P_WATER_BONUS: float = 76.0
    P_HARVEST: float = 64.0
    P_PLACE_ANIMAL: float = 60.0
    P_WATER_PLAIN: float = 56.0
    P_PLANT: float = 48.0
    P_DIG_WEED: float = 44.0
    """Above planting: a weed blocks the tile, so clearing precedes filling."""
    P_DROP: float = 40.0
    P_BUILD: float = 36.0
    P_COLLECT_FERT: float = 30.0
    P_CARE: float = 26.0
    P_FERTILIZE: float = 22.0

    DISTANCE_WEIGHT: float = 0.34
    """How sharply priority decays with travel distance. Raised sharply from the
    first version, which sent units across the farm for marginal gains."""

    WATER_URGENCY_HOUR: int = 12
    """After this hour, unwatered plants escalate to critical - there may not be
    another chance before the day rolls over."""

    # -- selling -----------------------------------------------------------
    SELL_PRICE_FRAC: float = 0.90
    """Sell only while the marginal price is at least this fraction of base."""

    SELL_MARGINAL_FLOOR_FRAC: float = 0.70
    """Stop a sale once the *next* unit would fetch less than this fraction of
    base. This is what stops a big harvest cratering its own price."""

    SELL_MAX_DROP_FRAC: float = 0.88
    """A single sale may not push the marginal price below this fraction of the
    price it started at."""

    SELL_RELAXED_FLOOR_MULT: float = 0.55
    """How far the price floor drops during liquidation and shed-panic sales.
    Unsold goods score zero, so late in the season a bad price beats no sale."""

    SHED_PRESSURE_FRAC: float = 0.60
    SHED_PANIC_FRAC: float = 0.85
    MAX_SELL_BATCH: int = 30
    LIQUIDATION_DAYS: int = 3
    HOLD_EARLY_DAYS: int = 3
    """Town demand lifts prices all season, so the first sales can wait."""

    # -- carrying / shed logistics ----------------------------------------
    DROP_THRESHOLD: int = 6
    DROP_HOUR: int = 20

    # -- animals -----------------------------------------------------------
    ENABLE_ANIMALS: bool = True
    ANIMAL_CASH_RESERVE: float = 500.0
    ANIMAL_LAST_USEFUL_DAY: int = 20
    MAX_ANIMALS: int = 18
    ANIMAL_TILE_SHARE: float = 0.30
    """Ceiling on the share of owned land given over to coops and pastures."""

    FEED_WHEAT_BUFFER: int = 3
    """Wheat kept in the shed per animal rather than sold."""

    ANIMAL_PRIORITY: tuple[str, ...] = ("COW", "SHEEP", "GOOSE")
    """Ranked by coins per unit of labour, cross-checked against how much of each
    product the town actually absorbs."""

    # -- fertilizer --------------------------------------------------------
    USE_FERTILIZER: bool = True

    # -- search overrides --------------------------------------------------
    # Left as None in normal play. The Phase 4 search sets them to turn this
    # agent into a family of distinguishable macro-strategies without needing a
    # second implementation of the execution logic.
    FORCE_CROP: str | None = None
    HAND_TARGET: int | None = None
    FORCE_ANIMAL: str | None = None


DEFAULT_PARAMS = Params()


# ==========================================================================
# Task model
# ==========================================================================
@dataclass(slots=True)
class Task:
    """A unit-executable intent at a board position."""

    priority: float
    pos: Pos
    op: list[Any]
    kind: str
    crop: str | None = None
    requires_item: str | None = None
    """Item the acting unit must already be carrying. FEED, FERTILIZE and
    PLACE all consume from the unit's own inventory, and the environment turns
    a missing item into a silent no-op rather than an error - so an unsatisfied
    task is a wasted turn, not a warning."""

    def score_for(self, unit_pos: Pos, weight: float) -> float:
        """Priority discounted by travel time."""
        return self.priority / (1.0 + weight * distance(unit_pos, self.pos))


# ==========================================================================
# Agent
# ==========================================================================
class BaselineAgent:
    """Rule-based agent. Stateless across episodes; safe to reuse per seat."""

    def __init__(
        self,
        params: Params | None = None,
        log_path: str | Path | None = None,
    ) -> None:
        self.p = params or DEFAULT_PARAMS
        self._log_path = Path(log_path) if log_path else None
        self._log_handle = None
        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = self._log_path.open("w", encoding="utf-8")

    # -- entry point -------------------------------------------------------
    def __call__(self, obs: Any, config: Any = None) -> Action:
        """Never raises. A crash mid-episode forfeits the game, so the whole
        decision is wrapped and degrades to a legal PASS."""
        n_hands = 0
        try:
            farms = obs.get("farms") or []
            player = int(obs.get("player", 0))
            if player < len(farms):
                n_hands = len(farms[player].get("hands") or [])
            return self.decide(View.from_obs(obs))
        except Exception as exc:  # pragma: no cover - defensive by design
            self._log_error(exc)
            return safe_action(n_hands)

    def close(self) -> None:
        if self._log_handle:
            self._log_handle.close()
            self._log_handle = None

    # -- the decision ------------------------------------------------------
    def decide(self, v: View) -> Action:
        plan = FarmPlan.build(v, self.p)
        market = self._market_orders(v, plan)
        tasks = self._build_tasks(v, plan)
        unit_ops, chosen = self._assign_units(v, tasks)
        self._log(v, plan, market, chosen)
        return {
            "farmer": unit_ops[0],
            "hands": unit_ops[1:],
            "market": market[:MAX_MARKET_ORDERS],
        }

    # ------------------------------------------------------------------
    # Market
    # ------------------------------------------------------------------
    def _market_orders(self, v: View, plan: "FarmPlan") -> list[list[Any]]:
        budget = v.money

        # The environment processes at most `maxMarketOrdersPerTurn` orders and
        # silently drops the rest, so slots are budgeted by category. Appending
        # first-come let a batch of sells plus eight hires crowd out the seed
        # order completely - the farm kept its labour and ran out of crops.
        sells: list[list[Any]] = []
        hires: list[list[Any]] = []
        buys: list[list[Any]] = []

        sells.extend(self._sell_orders(v, plan))

        # Labour. Cheap, compounding, and the binding constraint. Costed before
        # the liquidation branch so the closing days can keep their hands.
        for i in range(plan.hires_wanted):
            cost = hire_cost(v.hires_today + i)
            # The cheapest few hands are near-free (fib: 1,1,2 for the first
            # three) and a flat reserve above the bank would veto them outright.
            at_floor = (
                self.p.HIRE_FLOOR_IGNORES_RESERVE
                and v.n_hands + i < self.p.MIN_HANDS
            )
            reserve = 0.0 if at_floor else self.p.HIRE_CASH_RESERVE
            if budget - cost < reserve:
                break
            budget -= cost
            hires.append(["HIRE"])

        if plan.liquidating:
            if not self.p.LIQUIDATION_HIRES:
                return sells[:MAX_MARKET_ORDERS]
            # Selling is what liquidation is for, so hires yield the slots.
            n_sell = min(len(sells), MAX_MARKET_ORDERS - min(len(hires), 3))
            return (sells[:n_sell] + hires)[:MAX_MARKET_ORDERS]

        # 3. Land, while there is still season left to farm it.
        if plan.buy_land:
            budget -= LAND_PRICES[len(v.unlocked_quadrants) - 1]
            buys.append(["BUY_LAND"])

        # 4. Seeds for what we intend to plant next.
        if plan.plant_crop and plan.seeds_to_buy > 0:
            crop = plan.plant_crop
            unit = CROPS[crop]["seed"]
            n = min(plan.seeds_to_buy, int(max(0.0, budget - self.p.SEED_CASH_RESERVE) // unit))
            if n > 0:
                budget -= unit * n
                buys.append(["BUY_SEED", crop, n])

        # 5. Animals, once the crop engine funds them.
        if plan.animal_to_buy:
            animal = plan.animal_to_buy
            if budget - ANIMALS[animal]["cost"] >= self.p.ANIMAL_CASH_RESERVE:
                budget -= ANIMALS[animal]["cost"]
                buys.append(["BUY_ANIMAL", animal, 1])

        # 6. Feed wheat, if we keep animals but grow none.
        if plan.feed_wheat_to_buy > 0:
            price = marginal_sell_price("WHEAT", v.market_inventory.get("WHEAT", MARKET_I0))
            n = min(
                plan.feed_wheat_to_buy,
                int(max(0.0, budget - self.p.FEED_WHEAT_CASH_RESERVE) // max(1, price)),
            )
            if n > 0:
                buys.append(["BUY_PRODUCT", "WHEAT", n])

        # Sells fund everything, purchases compound, hires are cheap and can
        # simply be repeated next turn - so that is the order slots are given up.
        n_sell = min(len(sells), MAX_MARKET_ORDERS - min(len(buys), 4))
        return (sells[:n_sell] + buys + hires)[:MAX_MARKET_ORDERS]

    def _sell_orders(self, v: View, plan: "FarmPlan") -> list[list[Any]]:
        """Size each sale by marginal price so a harvest cannot crater itself."""
        fill = v.shed_used / SHED_CAPACITY
        pressured = fill >= self.p.SHED_PRESSURE_FRAC
        panicking = fill >= self.p.SHED_PANIC_FRAC
        if v.day < self.p.HOLD_EARLY_DAYS and not pressured:
            return []

        candidates: list[tuple[float, str, int]] = []
        for item in PRODUCTS:
            held = int(v.shed.get(item, 0))
            if item == "WHEAT" and not plan.liquidating:
                held -= plan.wheat_reserve
            if held <= 0:
                continue
            inv = v.market_inventory.get(item, MARKET_PARAMS[item]["I0"])
            base = MARKET_PARAMS[item]["base"]

            if plan.liquidating:
                # Spread the dump over the remaining days so the final sale is
                # not the one that hits the price floor.
                days_left = max(1, SEASON_DAYS - v.day)
                n = held if days_left <= 1 else max(1, math.ceil(held / days_left))
            elif panicking:
                n = held
            else:
                if marginal_sell_price(item, inv) < base * self.p.SELL_PRICE_FRAC and not pressured:
                    continue
                n = self._batch_size(item, held, inv, relaxed=pressured)
            n = min(n, self.p.MAX_SELL_BATCH, held)
            if n > 0:
                candidates.append((sell_revenue(item, n, inv), item, n))

        # Highest-value sales first: order slots are limited to 10 per turn.
        candidates.sort(reverse=True)
        return [["SELL", item, n] for _, item, n in candidates]

    def _batch_size(self, item: str, held: int, inv: float, relaxed: bool) -> int:
        """Largest n whose *last* unit still clears the marginal floor."""
        base = MARKET_PARAMS[item]["base"]
        # Two floors: an absolute one against base, and a relative one so a sale
        # into an elevated market cannot walk its own price down more than
        # SELL_MAX_DROP_FRAC. Without the second, a strawberry price of 316
        # would happily be sold down toward 84.
        spot = marginal_sell_price(item, inv)
        floor = max(
            base * self.p.SELL_MARGINAL_FLOOR_FRAC,
            spot * self.p.SELL_MAX_DROP_FRAC,
        ) * (self.p.SELL_RELAXED_FLOOR_MULT if relaxed else 1.0)
        n = 0
        cur = inv
        while n < held and n < self.p.MAX_SELL_BATCH:
            if marginal_sell_price(item, cur) < floor:
                break
            n += 1
            cur += 1
        return n

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------
    def _build_tasks(self, v: View, plan: "FarmPlan") -> list[Task]:
        p = self.p
        tasks: list[Task] = []
        late = v.hour >= p.WATER_URGENCY_HOUR
        carried_fert = sum(v.unit_inventory(i).get("FERTILIZER", 0) for i in range(v.n_units))
        plants_planned = 0

        for t in v.owned_tiles():
            if t.is_plant:
                if t.needs_water(v.day):
                    if late:
                        pri = p.P_WATER_CRITICAL
                    elif t.in_water_bonus_window(v.day):
                        pri = p.P_WATER_BONUS
                    else:
                        pri = p.P_WATER_PLAIN
                    tasks.append(Task(pri, t.pos, ["WATER"], "WATER", t.crop))

                if t.harvestable(v.day):
                    ripe = self._fully_ripe(t, v.day)
                    pri = p.P_HARVEST_URGENT if (ripe or plan.liquidating) else p.P_HARVEST
                    tasks.append(Task(pri, t.pos, ["HARVEST"], "HARVEST", t.crop))

                # Fertilizing is a silent no-op without fertilizer in hand - the
                # first version burned 135 unit-turns on exactly that.
                if (
                    p.USE_FERTILIZER
                    and carried_fert > 0
                    and t.fertilized_until_day < v.day
                    and t.in_water_bonus_window(v.day)
                ):
                    tasks.append(Task(p.P_FERTILIZE, t.pos, ["FERTILIZE"], "FERTILIZE",
                                      t.crop, requires_item="FERTILIZER"))

            elif t.has_animal:
                if not t.fed_today:
                    tasks.append(Task(p.P_FEED, t.pos, ["FEED"], "FEED",
                                      requires_item="WHEAT"))
                if t.yield_units > 0:
                    urgent = t.yield_units >= ANIMALS[t.animal]["max_held"]
                    pri = p.P_HARVEST_URGENT if (urgent or plan.liquidating) else p.P_HARVEST
                    tasks.append(Task(pri, t.pos, ["HARVEST"], "HARVEST"))
                if t.fertilizer_available and p.USE_FERTILIZER:
                    tasks.append(
                        Task(p.P_COLLECT_FERT, t.pos, ["COLLECT_FERTILIZER"], "COLLECT_FERT")
                    )
                if not t.cared_today:
                    tasks.append(Task(p.P_CARE, t.pos, ["CARE"], "CARE"))

            elif t.is_structure:
                # PLACE consumes the animal from the *unit's* inventory, not the
                # shed, so this task is only legal for a unit already carrying
                # one. Emitting it otherwise burns the turn on a silent no-op.
                carried = plan.carried_animal_for(t.kind, v)
                if carried:
                    tasks.append(
                        Task(p.P_PLACE_ANIMAL, t.pos, ["PLACE", carried], "PLACE_ANIMAL",
                             requires_item=carried)
                    )

            elif t.is_weed:
                tasks.append(Task(p.P_DIG_WEED, t.pos, ["DIG"], "DIG"))

            elif t.empty:
                if t.pos in plan.build_spots:
                    op = ["BUILD_PASTURE"] if plan.build_kind == "PASTURE" else ["BUILD_COOP"]
                    tasks.append(Task(p.P_BUILD, t.pos, op, "BUILD"))
                elif (
                    plan.plant_crop
                    and t.pos in plan.plant_spots
                    and plants_planned < plan.plant_budget
                ):
                    plants_planned += 1
                    tasks.append(
                        Task(p.P_PLANT, t.pos, ["PLANT", plan.plant_crop], "PLANT", plan.plant_crop)
                    )

        return tasks

    @staticmethod
    def _fully_ripe(t: Tile, day: int) -> bool:
        """True when waiting longer cannot add yield."""
        d = CROPS[t.crop]
        if d["ongoing"]:
            return t.yield_units >= d["max_yield"]
        return t.age_days(day) >= d["max_yield_day"] or t.yield_units >= d["max_yield"]

    # ------------------------------------------------------------------
    # Unit assignment
    # ------------------------------------------------------------------
    def _assign_units(
        self, v: View, tasks: list[Task]
    ) -> tuple[list[list[Any]], list[dict[str, Any]]]:
        """Globally greedy nearest-first matching of units to tasks.

        Repeatedly takes the best remaining (unit, task) pair rather than walking
        the task list in priority order. With <= 14 units and ~120 tasks this is
        a couple of thousand comparisons - cheap, and it stops a distant unit
        claiming a task that a neighbouring one should take.
        """
        n_units = v.n_units
        ops: list[list[Any]] = [["PASS"] for _ in range(n_units)]
        chosen: list[dict[str, Any]] = []
        free = set(range(n_units))

        # Logistics first: carrying goods blocks selling, and feeding needs wheat.
        reserved = self._reserved_items(v, tasks)
        for idx in list(free):
            drop = self._drop_task(v, idx, reserved)
            if drop is not None:
                ops[idx] = self._execute(v, idx, drop)
                chosen.append(self._record(idx, drop, ops[idx]))
                free.discard(idx)

        for getter in (self._assign_animal_pickup, self._assign_feed_pickup):
            got = getter(v, tasks, free)
            if got is not None:
                idx, task = got
                ops[idx] = self._execute(v, idx, task)
                chosen.append(self._record(idx, task, ops[idx]))
                free.discard(idx)

        # Only pair a task with a unit that can actually carry it out.
        carrying: dict[str, set[int]] = {}
        for ui in free:
            for item, n in v.unit_inventory(ui).items():
                if n > 0:
                    carrying.setdefault(item, set()).add(ui)
        pairs: list[tuple[float, int, int]] = []
        for ti, task in enumerate(tasks):
            eligible = free if task.requires_item is None else (
                free & carrying.get(task.requires_item, set())
            )
            for ui in eligible:
                pairs.append((task.score_for(v.unit_pos(ui), self.p.DISTANCE_WEIGHT), ui, ti))
        pairs.sort(key=lambda x: x[0], reverse=True)

        used_tasks: set[int] = set()
        used_tiles: set[Pos] = set()
        for _score, ui, ti in pairs:
            if ui not in free or ti in used_tasks:
                continue
            task = tasks[ti]
            if task.pos in used_tiles:
                continue
            ops[ui] = self._execute(v, ui, task)
            chosen.append(self._record(ui, task, ops[ui]))
            free.discard(ui)
            used_tasks.add(ti)
            used_tiles.add(task.pos)
            if not free:
                break

        for idx in free:
            ops[idx] = self._idle_op(v, idx)
        return ops, chosen

    @staticmethod
    def _reserved_items(v: View, tasks: list[Task]) -> dict[str, int]:
        """Items a unit is carrying because a pending task needs them."""
        reserved: dict[str, int] = {}
        for t in tasks:
            if t.requires_item:
                reserved[t.requires_item] = reserved.get(t.requires_item, 0) + 1
        return reserved

    @staticmethod
    def _record(idx: int, task: Task, op: list[Any]) -> dict[str, Any]:
        return {"unit": idx, "kind": task.kind, "pos": list(task.pos), "op": op}

    def _execute(self, v: View, idx: int, task: Task) -> list[Any]:
        """Do the task if standing on it, otherwise take one step toward it."""
        pos = v.unit_pos(idx)
        if pos == task.pos:
            return list(task.op)
        return step_toward(pos, task.pos, v.board_size)

    def _drop_task(self, v: View, idx: int, reserved: dict[str, int] | None = None) -> Task | None:
        """Send a loaded unit to the shed - but only for goods it is not about to use.

        Wheat destined for FEED and an animal destined for PLACE must stay in
        hand. Counting them here produced a pickup/drop cycle that burned 383
        unit-turns against only 76 actual feeds.
        """
        inv = dict(v.unit_inventory(idx))
        for item, n in (reserved or {}).items():
            if item in inv:
                inv[item] = max(0, inv[item] - n)
        carried = sum(inv.values())
        if carried <= 0 or v.shed_room(SHED_CAPACITY) <= 0:
            return None
        if carried < self.p.DROP_THRESHOLD and v.hour < self.p.DROP_HOUR:
            return None
        pos = v.unit_pos(idx)
        target = min(v.owned_shed_tiles(), key=lambda q: distance(pos, q))
        return Task(self.p.P_DROP, target, ["DROP"], "DROP")

    def _assign_animal_pickup(
        self, v: View, tasks: list[Task], free: set[int]
    ) -> tuple[int, Task] | None:
        """Collect a bought animal from the shed so it can be placed.

        BUY_ANIMAL deposits into the shed, but PLACE consumes from the unit's
        own inventory, so this hop is mandatory. Missing it is what made the
        first animal-enabled version spend 93% of its turns on a no-op.
        """
        if not self.p.ENABLE_ANIMALS or not free:
            return None
        plan_animal = FarmPlan.shed_animal_for_pickup(None, v)  # type: ignore[arg-type]
        if not plan_animal:
            return None
        shed_tiles = v.owned_shed_tiles()
        idx = min(free, key=lambda i: min(distance(v.unit_pos(i), q) for q in shed_tiles))
        target = min(shed_tiles, key=lambda q: distance(v.unit_pos(idx), q))
        return idx, Task(
            self.p.P_PLACE_ANIMAL, target, ["PICKUP", plan_animal, 1], "PICKUP_ANIMAL"
        )

    def _assign_feed_pickup(
        self, v: View, tasks: list[Task], free: set[int]
    ) -> tuple[int, Task] | None:
        """Send one unit to fetch wheat when animals need feeding and nobody has any."""
        if not self.p.ENABLE_ANIMALS or not free:
            return None
        hungry = sum(1 for t in tasks if t.kind == "FEED")
        if hungry <= 0:
            return None
        in_hand = sum(v.unit_inventory(i).get("WHEAT", 0) for i in range(v.n_units))
        if in_hand >= hungry:
            return None
        stock = int(v.shed.get("WHEAT", 0))
        if stock <= 0:
            return None
        shed_tiles = v.owned_shed_tiles()
        idx = min(free, key=lambda i: min(distance(v.unit_pos(i), q) for q in shed_tiles))
        target = min(shed_tiles, key=lambda q: distance(v.unit_pos(idx), q))
        n = max(1, min(hungry - in_hand, stock))
        return idx, Task(self.p.P_FEED, target, ["PICKUP", "WHEAT", n], "PICKUP_FEED")

    def _idle_op(self, v: View, idx: int) -> list[Any]:
        pos = v.unit_pos(idx)
        target = min(v.owned_shed_tiles(), key=lambda q: distance(pos, q))
        return ["PASS"] if pos == target else step_toward(pos, target, v.board_size)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _log(
        self, v: View, plan: "FarmPlan", market: list[list[Any]], chosen: list[dict[str, Any]]
    ) -> None:
        if not self._log_handle:
            return
        self._log_handle.write(json.dumps({
            "step": v.step,
            "day": v.day,
            "hour": v.hour,
            "money": round(v.money, 1),
            "net_worth": round(v.net_worth(), 1),
            "hands": v.n_hands,
            "shed_used": v.shed_used,
            "quadrants": v.unlocked_quadrants,
            "plan": {
                "crop": plan.plant_crop,
                "plant_budget": plan.plant_budget,
                "hires": plan.hires_wanted,
                "animal": plan.animal_to_buy,
                "capacity": round(plan.work_capacity, 1),
            },
            "market": market,
            "tasks": chosen,
        }) + "\n")

    def _log_error(self, exc: Exception) -> None:
        if self._log_handle:
            self._log_handle.write(json.dumps({"error": repr(exc)}) + "\n")


# ==========================================================================
# Per-turn plan
# ==========================================================================
@dataclass(slots=True)
class FarmPlan:
    """Whole-farm decisions computed once per turn and shared by market + tasks.

    Keeping these together is what enforces the central invariant: the number of
    plants we commit to is derived from the same labour estimate that decides how
    many hands to hire, so the two can never disagree.
    """

    liquidating: bool
    work_capacity: float
    hires_wanted: int
    buy_land: bool
    plant_crop: str | None
    plant_budget: int
    plant_spots: set[Pos]
    seeds_to_buy: int
    build_spots: set[Pos]
    build_kind: str
    animal_to_buy: str | None
    wheat_reserve: int
    feed_wheat_to_buy: int

    @classmethod
    def build(cls, v: View, p: Params) -> "FarmPlan":
        days_left = SEASON_DAYS - v.day
        liquidating = days_left <= p.LIQUIDATION_DAYS

        g = v.groups()
        n_owned = g.n_owned
        plants, animals = g.plants, g.animals
        structures, empties, weeds = g.structures, g.empties, g.weeds

        # --- labour ---------------------------------------------------------
        # Daily work = one water per plant, plus amortised harvest/plant/feed.
        demand = (
            len(plants)
            + p.WORK_PER_PLANT_EXTRA * len(plants)
            + p.WORK_PER_ANIMAL * len(animals)
            + p.WORK_PER_IDLE_TILE * min(len(empties) + len(weeds), n_owned)
        )
        if p.HAND_TARGET is not None:
            want_hands = max(0, min(p.MAX_HANDS, p.HAND_TARGET))
        else:
            want_hands = min(
                p.MAX_HANDS,
                max(p.MIN_HANDS, math.ceil(demand / p.WORK_PER_UNIT_PER_DAY) - 1),
            )
        if liquidating:
            cap = p.MIN_HANDS if p.LIQUIDATION_HAND_TARGET is None else p.LIQUIDATION_HAND_TARGET
            want_hands = min(want_hands, cap)
        hires_wanted = 0
        if v.hour <= p.HIRE_BEFORE_HOUR and v.day < SEASON_DAYS - 1:
            hires_wanted = min(max(0, want_hands - v.n_hands), p.MAX_HIRES_PER_TURN)

        # Capacity is what we will *have* today, not what we have this instant.
        units_today = max(v.n_units, want_hands + 1)
        work_capacity = units_today * p.WORK_PER_UNIT_PER_DAY

        # --- planting -------------------------------------------------------
        # The cap that matters: never commit to more plants than can be watered
        # every single day for the rest of their cycle.
        sustainable = int(work_capacity * p.PLANT_CAPACITY_SLACK)
        plant_budget = max(0, sustainable - len(plants))
        if liquidating:
            plant_budget = 0

        crop = cls._best_crop(v, p, plants) if plant_budget > 0 else None
        seeds_held = int(v.seeds.get(crop, 0)) if crop else 0
        plant_budget = min(plant_budget, seeds_held)

        # --- animals --------------------------------------------------------
        animal_cap = min(p.MAX_ANIMALS, int(n_owned * p.ANIMAL_TILE_SHARE))
        pending = sum(int(v.shed.get(a, 0)) for a in ANIMALS)
        want_animals = (
            p.ENABLE_ANIMALS
            and v.day <= p.ANIMAL_LAST_USEFUL_DAY
            and len(animals) + pending < animal_cap
        )
        animal_to_buy = None
        build_kind = "PASTURE"
        build_spots: set[Pos] = set()
        if want_animals:
            candidate = cls._best_animal(v, p, len(structures), pending)
            if candidate:
                build_kind = ANIMALS[candidate]["structure"]
                # Housing first, animal second. Buying an animal with nowhere to
                # put it strands 300-500 coins in the shed and blocks the slot.
                in_transit = pending + sum(
                    v.unit_inventory(i).get(candidate, 0) for i in range(v.n_units)
                )
                free_housing = sum(1 for t in structures if t.kind == build_kind)
                if free_housing > in_transit:
                    animal_to_buy = candidate
                planned_housing = free_housing + len(animals)
                if planned_housing < animal_cap and free_housing <= in_transit + 1:
                    shed_tiles = v.shed_tiles()
                    near = sorted(
                        empties, key=lambda t: min(distance(t.pos, q) for q in shed_tiles)
                    )
                    build_spots = {t.pos for t in near[:1]}

        # --- planting locations: compact, near the shed ---------------------
        shed_tiles = v.shed_tiles()
        plant_candidates = sorted(
            (t for t in empties if t.pos not in build_spots),
            key=lambda t: min(distance(t.pos, q) for q in shed_tiles),
        )
        plant_spots = {t.pos for t in plant_candidates[: max(0, plant_budget)]}

        seeds_to_buy = 0
        if crop and not liquidating:
            room = max(0, sustainable - len(plants))
            seeds_to_buy = max(0, min(room, p.SEED_BUY_AHEAD, len(empties)) - seeds_held)

        # --- land -----------------------------------------------------------
        buy_land = cls._should_buy_land(v, p, n_owned, len(empties) + len(weeds))

        # --- feed -----------------------------------------------------------
        n_animals = len(animals)
        wheat_reserve = n_animals * p.FEED_WHEAT_BUFFER
        growing_wheat = any(t.crop == "WHEAT" for t in plants)
        feed_wheat_to_buy = 0
        if n_animals and not growing_wheat and not liquidating:
            deficit = wheat_reserve - int(v.shed.get("WHEAT", 0))
            feed_wheat_to_buy = max(0, deficit)

        return cls(
            liquidating=liquidating,
            work_capacity=work_capacity,
            hires_wanted=hires_wanted,
            buy_land=buy_land,
            plant_crop=crop,
            plant_budget=plant_budget,
            plant_spots=plant_spots,
            seeds_to_buy=seeds_to_buy,
            build_spots=build_spots,
            build_kind=build_kind,
            animal_to_buy=animal_to_buy,
            wheat_reserve=wheat_reserve,
            feed_wheat_to_buy=feed_wheat_to_buy,
        )

    def carried_animal_for(self, structure_kind: str | None, v: View) -> str | None:
        """An animal some unit is holding that this empty structure could house."""
        for animal, data in ANIMALS.items():
            if data["structure"] != structure_kind:
                continue
            if any(v.unit_inventory(i).get(animal, 0) > 0 for i in range(v.n_units)):
                return animal
        return None

    @staticmethod
    def shed_animal_for_pickup(_unused: Any, v: View) -> str | None:
        """An animal waiting in the shed that has somewhere to go."""
        free = {}
        for t in v.owned_tiles():
            if t.is_structure:
                free[t.kind] = free.get(t.kind, 0) + 1
        for animal, data in ANIMALS.items():
            if int(v.shed.get(animal, 0)) <= 0:
                continue
            housed = free.get(data["structure"], 0)
            carried = sum(v.unit_inventory(i).get(animal, 0) for i in range(v.n_units))
            if housed > carried:
                return animal
        return None

    # -- selection helpers -------------------------------------------------
    @staticmethod
    def _best_crop(v: View, p: Params, plants: list[Tile]) -> str | None:
        """Pick what to plant from current marginal prices, not a fixed rotation.

        Reading live market inventory means the agent rotates away from anything
        it has already glutted, without needing an explicit diversification rule.
        """
        days_left = SEASON_DAYS - v.day - p.MIN_DAYS_MARGIN
        melons = sum(1 for t in plants if t.crop == "MELON")

        if p.FORCE_CROP is not None:
            # The search has already chosen; only veto it if it cannot finish.
            if _cycle_days(p.FORCE_CROP) <= days_left:
                return p.FORCE_CROP

        best: tuple[float, str] | None = None
        for crop, data in CROPS.items():
            cycle_days = _cycle_days(crop)
            if cycle_days > days_left:
                continue
            if crop == "MELON" and melons >= p.MELON_MAX_TILES:
                continue
            if data["seed"] > v.money - p.SEED_CASH_RESERVE:
                continue
            units = _cycle_units(crop)
            inv = v.market_inventory.get(crop, MARKET_PARAMS[crop]["I0"])
            # Price the harvest at the market it will meet, not today's. Town
            # demand lifts prices all season, and valuing at spot systematically
            # underrates the long-cycle, high-margin crops.
            harvest_day = v.day + cycle_days
            revenue = projected_sell_revenue(
                crop, units, inv, v.day, harvest_day,
                v.unlocked_shops if p.REALIZED_SHOP_DEMAND else None,
            ) - data["seed"]
            per_tile_day = revenue / max(1, cycle_days)
            per_action = revenue / max(1, _cycle_actions(crop))
            score = (1.0 - p.CROP_VALUE_BLEND) * per_tile_day + p.CROP_VALUE_BLEND * per_action
            if best is None or score > best[0]:
                best = (score, crop)
        return best[1] if best else None

    @staticmethod
    def _best_animal(v: View, p: Params, n_structures: int, pending: int) -> str | None:
        """Rank animals by expected coins per unit of labour at current prices."""
        days_left = SEASON_DAYS - v.day
        if p.FORCE_ANIMAL is not None:
            data = ANIMALS[p.FORCE_ANIMAL]
            if data["cost"] <= v.money - p.ANIMAL_CASH_RESERVE:
                return p.FORCE_ANIMAL
            return None

        best: tuple[float, str] | None = None
        for animal in p.ANIMAL_PRIORITY:
            data = ANIMALS[animal]
            if data["cost"] > v.money - p.ANIMAL_CASH_RESERVE:
                continue
            productive_days = days_left - data["first_yield_day"]
            if productive_days <= data["interval"]:
                continue
            yields = productive_days // data["interval"]
            product = data["product"]
            inv = v.market_inventory.get(product, MARKET_PARAMS[product]["I0"])
            mid_day = v.day + data["first_yield_day"] + (productive_days // 2)
            revenue = projected_sell_revenue(
                product, yields, inv, v.day, mid_day,
                v.unlocked_shops if p.REALIZED_SHOP_DEMAND else None,
            ) - data["cost"]
            actions = days_left + yields  # one feed per day plus harvests
            score = revenue / max(1, actions)
            if score > 0 and (best is None or score > best[0]):
                best = (score, animal)
        return best[1] if best else None

    @staticmethod
    def _should_buy_land(v: View, p: Params, n_owned: int, idle: int) -> bool:
        owned_extra = len(v.unlocked_quadrants) - 1
        if owned_extra >= len(LAND_PRICES) or v.day > p.LAND_LAST_USEFUL_DAY:
            return False
        price = LAND_PRICES[owned_extra]
        if v.money < price + p.LAND_CASH_RESERVE:
            return False
        # Land we cannot work is worse than the cash: buying it does nothing and
        # the idle tiles just sprout weeds.
        return n_owned == 0 or idle / n_owned < p.LAND_IDLE_TILE_LIMIT


# --------------------------------------------------------------------------
# Crop cycle arithmetic (shared with the MCTS rollout policy)
# --------------------------------------------------------------------------
def _cycle_days(crop: str) -> int:
    d = CROPS[crop]
    if d["ongoing"]:
        return d["first_yield_day"] + d["interval"] * (d["max_yield"] - 1) + 1
    return d["max_yield_day"] + 1


def _cycle_units(crop: str) -> int:
    d = CROPS[crop]
    if d["ongoing"]:
        return d["max_yield"]
    window_start = (d["max_yield_day"] + 1) // 2
    return min(d["max_yield"], 1 + (d["max_yield_day"] - window_start + 1))


def _cycle_actions(crop: str) -> int:
    d = CROPS[crop]
    harvests = d["max_yield"] if d["ongoing"] else 1
    return 1 + _cycle_days(crop) + harvests  # plant + a water per day + harvests


# Module-level default instance for convenient import.
_DEFAULT = BaselineAgent()


def baseline_agent(obs: Any, config: Any = None) -> Action:
    """Stateless entry point using default parameters."""
    return _DEFAULT(obs, config)
