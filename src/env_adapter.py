"""Adapter between the raw kaggriculture observation and agent-friendly state.

Every agent reads state through `View`, so if the environment's schema ever
changes this is the one file that needs fixing. Nothing below imports
`kaggle_environments` at module scope: `View` works on a plain observation dict,
which keeps it usable inside `submission/main.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

from .game_constants import (
    ANIMALS,
    CROPS,
    MOVE_BY_DELTA,
    PRODUCTS,
    is_shed_adjacent,
    market_price,
    quadrant_of,
    shed_access_tiles,
)

Action = dict[str, Any]
Pos = tuple[int, int]

PASS_ACTION: Action = {"farmer": ["PASS"], "hands": [], "market": []}


def safe_action(n_hands: int = 0) -> Action:
    """A legal, always-valid no-op action. The universal fallback."""
    return {"farmer": ["PASS"], "hands": [["PASS"] for _ in range(n_hands)], "market": []}


# --------------------------------------------------------------------------
# Tile classification
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Tile:
    """A normalised view of one board tile."""

    x: int
    y: int
    raw: Any

    @property
    def locked(self) -> bool:
        return self.raw == "LOCKED"

    @property
    def empty(self) -> bool:
        return self.raw is None

    @property
    def kind(self) -> str | None:
        return self.raw.get("kind") if isinstance(self.raw, dict) else None

    @property
    def is_plant(self) -> bool:
        return self.kind == "PLANT"

    @property
    def is_weed(self) -> bool:
        return self.kind == "WEED"

    @property
    def is_structure(self) -> bool:
        """An empty coop/pasture with no animal in it yet."""
        return self.kind in ("COOP", "PASTURE") and not self.has_animal

    @property
    def has_animal(self) -> bool:
        return isinstance(self.raw, dict) and "animal" in self.raw

    @property
    def crop(self) -> str | None:
        return self.raw.get("crop") if self.is_plant else None

    @property
    def animal(self) -> str | None:
        return self.raw.get("animal") if self.has_animal else None

    @property
    def yield_units(self) -> int:
        return int(self.raw.get("yield_units", 0)) if isinstance(self.raw, dict) else 0

    @property
    def watered_today(self) -> bool:
        return bool(self.raw.get("watered_today", False)) if isinstance(self.raw, dict) else False

    @property
    def fed_today(self) -> bool:
        return bool(self.raw.get("fed_today", False)) if isinstance(self.raw, dict) else False

    @property
    def cared_today(self) -> bool:
        return bool(self.raw.get("cared_today", False)) if isinstance(self.raw, dict) else False

    @property
    def fertilizer_available(self) -> bool:
        return bool(self.raw.get("fertilizer_available", False)) if isinstance(self.raw, dict) else False

    @property
    def planted_day(self) -> int:
        return int(self.raw.get("planted_day", 0)) if isinstance(self.raw, dict) else 0

    @property
    def fertilized_until_day(self) -> int:
        return int(self.raw.get("fertilized_until_day", -1)) if isinstance(self.raw, dict) else -1

    @property
    def pos(self) -> Pos:
        return (self.x, self.y)

    # -- derived agronomy ---------------------------------------------------
    def age_days(self, day: int) -> int:
        return day - self.planted_day

    def harvestable(self, day: int) -> bool:
        """True when a HARVEST here would actually yield something."""
        if not isinstance(self.raw, dict) or self.yield_units <= 0:
            return False
        if self.is_plant:
            return self.age_days(day) >= CROPS[self.crop]["first_yield_day"]
        return self.has_animal

    def in_water_bonus_window(self, day: int) -> bool:
        """Watering inside this window increases a one-shot crop's yield."""
        if not self.is_plant:
            return False
        data = CROPS[self.crop]
        if data["ongoing"]:
            return True  # ongoing crops need water every day to produce at all
        age = self.age_days(day)
        return (data["max_yield_day"] + 1) // 2 <= age <= data["max_yield_day"]

    def needs_water(self, day: int) -> bool:
        return self.is_plant and not self.watered_today


# --------------------------------------------------------------------------
# The observation view
# --------------------------------------------------------------------------
@dataclass(slots=True)
class View:
    """Agent-facing projection of one turn's observation."""

    obs: dict[str, Any]
    player: int
    step: int
    day: int
    hour: int
    board_size: int
    farm: dict[str, Any]
    opponent: dict[str, Any] | None
    shed: dict[str, int]
    seeds: dict[str, int]
    inventories: list[dict[str, int]]
    prices: dict[str, int]
    market_inventory: dict[str, float]
    unlocked_shops: list[str]
    _tiles: list[list[Tile]] = field(default_factory=list, repr=False)
    _classified: "TileGroups | None" = field(default=None, repr=False)

    # -- construction -------------------------------------------------------
    @classmethod
    def from_obs(cls, obs: Any) -> "View":
        o = dict(obs)
        player = int(o.get("player") or 0)
        farms = o.get("farms") or []
        farm = farms[player] if player < len(farms) else {}
        opponent = farms[1 - player] if len(farms) > 1 else None
        private = o.get("private") or {}
        market = o.get("market") or {}
        town = o.get("town") or {}
        day = int(o.get("day") or 0)
        hour = int(o.get("hour") or 0)
        # Some ladder/replay wrappers have emitted ``step=None`` for seat 1.
        # Day and hour are the authoritative clock and are present in every
        # observation, so derive the step instead of letting ``int(None)``
        # degrade the whole agent to its emergency PASS path.
        raw_step = o.get("step")
        step = day * 24 + hour if raw_step is None else int(raw_step)
        tiles_raw = farm.get("tiles") or []
        board_size = len(tiles_raw)
        grid = [
            [Tile(x, y, tiles_raw[y][x]) for x in range(board_size)]
            for y in range(board_size)
        ]
        return cls(
            obs=o,
            player=player,
            step=step,
            day=day,
            hour=hour,
            board_size=board_size,
            farm=farm,
            opponent=opponent,
            shed=dict(private.get("shed") or {}),
            seeds=dict(private.get("seeds") or {}),
            inventories=[dict(i) for i in (private.get("inventories") or [{}])],
            prices=dict(market.get("prices") or {}),
            market_inventory=dict(market.get("inventory") or {}),
            unlocked_shops=list(town.get("unlocked_shops") or []),
            _tiles=grid,
        )

    # -- basic accessors ----------------------------------------------------
    @property
    def money(self) -> float:
        return float(self.farm.get("money", 0.0))

    @property
    def farmer_pos(self) -> Pos:
        p = self.farm.get("farmer") or [0, 0]
        return (int(p[0]), int(p[1]))

    @property
    def hand_positions(self) -> list[Pos]:
        return [(int(p[0]), int(p[1])) for p in (self.farm.get("hands") or [])]

    @property
    def n_hands(self) -> int:
        return len(self.farm.get("hands") or [])

    @property
    def n_units(self) -> int:
        return 1 + self.n_hands

    @property
    def hires_today(self) -> int:
        return int(self.farm.get("hires_today", 0))

    @property
    def unlocked_quadrants(self) -> list[str]:
        return list(self.farm.get("unlocked_quadrants") or ["NW"])

    @property
    def turns_left_today(self) -> int:
        return 24 - self.hour

    def unit_pos(self, idx: int) -> Pos:
        """idx 0 is the farmer, 1+ are hands."""
        return self.farmer_pos if idx == 0 else self.hand_positions[idx - 1]

    def unit_inventory(self, idx: int) -> dict[str, int]:
        return dict(self.inventories[idx]) if idx < len(self.inventories) else {}

    def tile(self, pos: Pos) -> Tile:
        x, y = pos
        return self._tiles[y][x]

    def tiles(self) -> Iterator[Tile]:
        for row in self._tiles:
            yield from row

    def owned_tiles(self) -> Iterator[Tile]:
        for t in self.tiles():
            if not t.locked:
                yield t

    def groups(self) -> "TileGroups":
        """Single-pass tile classification, computed once per turn.

        Callers used to walk `owned_tiles()` half a dozen times to count plants,
        animals, structures, empties and weeds separately; that showed up as the
        single largest cost in the agent's profile.
        """
        if self._classified is None:
            plants: list[Tile] = []
            animals: list[Tile] = []
            structures: list[Tile] = []
            empties: list[Tile] = []
            weeds: list[Tile] = []
            owned = 0
            for row in self._tiles:
                for t in row:
                    raw = t.raw
                    if raw == "LOCKED":
                        continue
                    owned += 1
                    if raw is None:
                        empties.append(t)
                    elif isinstance(raw, dict):
                        kind = raw.get("kind")
                        if kind == "PLANT":
                            plants.append(t)
                        elif "animal" in raw:
                            animals.append(t)
                        elif kind == "WEED":
                            weeds.append(t)
                        elif kind in ("COOP", "PASTURE"):
                            structures.append(t)
            self._classified = TileGroups(
                plants=plants, animals=animals, structures=structures,
                empties=empties, weeds=weeds, n_owned=owned,
            )
        return self._classified

    # -- aggregates ---------------------------------------------------------
    @property
    def shed_used(self) -> int:
        return sum(v for v in self.shed.values() if v > 0)

    def shed_room(self, capacity: int = 100) -> int:
        return max(0, capacity - self.shed_used)

    def carried_total(self) -> int:
        return sum(sum(inv.values()) for inv in self.inventories)

    def price(self, item: str) -> int:
        if item in self.prices:
            return int(self.prices[item])
        return market_price(item, self.market_inventory.get(item, 10000))

    def is_shed_tile(self, pos: Pos) -> bool:
        return is_shed_adjacent(pos, self.board_size)

    def shed_tiles(self) -> list[Pos]:
        return shed_access_tiles(self.board_size)

    def owned_shed_tiles(self) -> list[Pos]:
        """Shed-access tiles we can actually stand on and use (any tile works,
        but preferring unlocked ones keeps units inside owned land)."""
        return [p for p in self.shed_tiles() if not self.tile(p).locked] or self.shed_tiles()

    def quadrant(self, pos: Pos) -> str:
        return quadrant_of(pos[0], pos[1], self.board_size)

    # -- valuation ----------------------------------------------------------
    def net_worth(self) -> float:
        """Cash plus a conservative liquidation value of everything we hold.

        Used as the shaped reward for short rollouts, where raw cash would
        punish any investment that has not paid off yet. Deliberately values
        goods at the *marginal* price rather than simulating a full dump, which
        overstates a large hoard slightly but is far cheaper to compute.
        """
        total = self.money
        for item, n in self.shed.items():
            if n > 0 and item in PRODUCTS:
                total += n * self.price(item)
            elif n > 0 and item in ANIMALS:
                total += n * ANIMALS[item]["cost"] * 0.5
        for inv in self.inventories:
            for item, n in inv.items():
                if n > 0 and item in PRODUCTS:
                    total += n * self.price(item)
        for crop, n in self.seeds.items():
            total += n * CROPS[crop]["seed"] * 0.5
        for t in self.owned_tiles():
            if t.is_plant:
                total += t.yield_units * self.price(t.crop)
                total += CROPS[t.crop]["seed"] * 0.5  # growth already paid for
            elif t.has_animal:
                total += ANIMALS[t.animal]["cost"] * 0.6
                total += t.yield_units * self.price(ANIMALS[t.animal]["product"])
        return total


@dataclass(frozen=True, slots=True)
class TileGroups:
    """Owned tiles bucketed by what they are."""

    plants: list[Tile]
    animals: list[Tile]
    structures: list[Tile]
    empties: list[Tile]
    weeds: list[Tile]
    n_owned: int


# --------------------------------------------------------------------------
# Pathfinding
# --------------------------------------------------------------------------
def step_toward(src: Pos, dst: Pos, board_size: int) -> list[str]:
    """One movement op that reduces Manhattan distance from `src` to `dst`.

    All tiles are passable - the environment allows walking over LOCKED tiles
    precisely so hands cannot be stranded - so there is no obstacle avoidance to
    do and greedy stepping is optimal. Returns `["PASS"]` when already there.
    """
    if src == dst:
        return ["PASS"]
    dx = dst[0] - src[0]
    dy = dst[1] - src[1]
    # Move along the longer axis first; ties break horizontally for stability.
    if abs(dx) >= abs(dy) and dx != 0:
        delta = (1 if dx > 0 else -1, 0)
    elif dy != 0:
        delta = (0, 1 if dy > 0 else -1)
    else:
        delta = (1 if dx > 0 else -1, 0)
    nx, ny = src[0] + delta[0], src[1] + delta[1]
    if not (0 <= nx < board_size and 0 <= ny < board_size):
        return ["PASS"]
    return [MOVE_BY_DELTA[delta]]


def distance(a: Pos, b: Pos) -> int:
    """Manhattan distance; exact travel time given orthogonal single steps."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def nearest(src: Pos, targets: Sequence[Pos]) -> Pos | None:
    return min(targets, key=lambda t: distance(src, t)) if targets else None
