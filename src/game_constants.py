"""Game constants, vendored verbatim from the kaggriculture environment.

Why vendored rather than imported: `submission/main.py` has to be readable and
self-contained, and the heuristics need these numbers on every turn. Importing
them from `kaggle_environments` would work at submission time (the env is what
runs the agent) but couples plain arithmetic to a package import.

The copy is *checked*, not trusted: `tests/test_constants_drift.py` asserts every
table here still equals the installed environment's. If Kaggle rebalances the
game, that test fails loudly instead of the agent quietly mispricing everything.

Source of truth: kaggle_environments/envs/kaggriculture/kaggriculture.py
"""

from __future__ import annotations

import math
from typing import Any, Final, Sequence

# --------------------------------------------------------------------------
# Crops
# --------------------------------------------------------------------------
CROPS: Final[dict[str, dict[str, Any]]] = {
    "WHEAT":      {"seed": 10,  "first_yield_day": 2,  "max_yield_day": 4,  "interval": 0, "max_yield": 6, "ongoing": False},
    "CARROT":     {"seed": 20,  "first_yield_day": 2,  "max_yield_day": 3,  "interval": 0, "max_yield": 4, "ongoing": False},
    "TOMATO":     {"seed": 50,  "first_yield_day": 8,  "max_yield_day": 8,  "interval": 1, "max_yield": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "interval": 2, "max_yield": 4, "ongoing": True},
    "MELON":      {"seed": 80,  "first_yield_day": 10, "max_yield_day": 12, "interval": 0, "max_yield": 6, "ongoing": False},
}

ANIMALS: Final[dict[str, dict[str, Any]]] = {
    "GOOSE": {"cost": 300, "structure": "COOP",    "first_yield_day": 4, "interval": 1, "max_held": 4, "product": "EGG"},
    "COW":   {"cost": 400, "structure": "PASTURE", "first_yield_day": 8, "interval": 2, "max_held": 6, "product": "MILK"},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "first_yield_day": 6, "interval": 3, "max_held": 6, "product": "WOOL"},
}

PRODUCTS: Final[list[str]] = [
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "EGG", "MILK", "WOOL", "FERTILIZER",
]

# --------------------------------------------------------------------------
# Market
# --------------------------------------------------------------------------
MARKET_I0: Final[int] = 10000
PRICE_FLOOR: Final[int] = 1
HINGE_GAIN: Final[float] = 8.0

MARKET_PARAMS: Final[dict[str, dict[str, Any]]] = {
    "WHEAT":      {"base":  25, "I0": MARKET_I0, "T": 400, "below_func": "sqrt",   "below_target": 0.80, "above_func": "log",    "above_target": 0.20},
    "CARROT":     {"base":  35, "I0": MARKET_I0, "T": 450, "below_func": "hinge",  "below_target": 1.00, "above_func": "sqrt",   "above_target": 0.70},
    "TOMATO":     {"base":  60, "I0": MARKET_I0, "T": 200, "below_func": "hinge",  "below_target": 0.40, "above_func": "sqrt",   "above_target": 0.60},
    "STRAWBERRY": {"base": 120, "I0": MARKET_I0, "T": 100, "below_func": "sqrt",   "below_target": 0.70, "above_func": "linear", "above_target": 1.60},
    "MELON":      {"base": 250, "I0": MARKET_I0, "T": 300, "below_func": "log",    "below_target": 0.20, "above_func": "sq",     "above_target": 3.60},
    "EGG":        {"base":  50, "I0": MARKET_I0, "T": 332, "below_func": "hinge",  "below_target": 0.40, "above_func": "log",    "above_target": 0.20},
    "MILK":       {"base": 160, "I0": MARKET_I0, "T": 122, "below_func": "sqrt",   "below_target": 0.60, "above_func": "linear", "above_target": 1.60},
    "WOOL":       {"base": 200, "I0": MARKET_I0, "T": 105, "below_func": "log",    "below_target": 0.20, "above_func": "sq",     "above_target": 3.20},
    "FERTILIZER": {"base": 100, "I0": MARKET_I0, "T": 200, "below_func": "linear", "below_target": 0.40, "above_func": "linear", "above_target": 0.40},
}

SHOPS: Final[dict[str, list[str]]] = {
    "BAKERY":         ["EGG", "WHEAT"],
    "PIZZA_SHOP":     ["MILK", "TOMATO", "WHEAT"],
    "BRUNCH_SPOT":    ["EGG", "WHEAT", "STRAWBERRY"],
    "YARN_STORE":     ["WOOL"],
    "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"],
    "PET_CAFE":       ["CARROT"],
    "SMOOTHIE_SHOP":  ["STRAWBERRY", "MILK"],
    "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
}

TOWN_CENTER_PRODUCTS: Final[list[str]] = [p for p in PRODUCTS if p != "FERTILIZER"]
MAX_SHOP_INSTANCES: Final[int] = 8

# --------------------------------------------------------------------------
# Board / land / labour
# --------------------------------------------------------------------------
FARMER_MOVES: Final[dict[str, tuple[int, int]]] = {
    "NORTH": (0, -1),
    "SOUTH": (0, 1),
    "EAST":  (1, 0),
    "WEST":  (-1, 0),
}
MOVE_BY_DELTA: Final[dict[tuple[int, int], str]] = {v: k for k, v in FARMER_MOVES.items()}

LAND_ORDER: Final[list[str]] = ["NE", "SW", "SE"]
LAND_PRICES: Final[list[int]] = [1000, 2000, 4000]
FARM_HAND_COST_MULT: Final[int] = 1


# --------------------------------------------------------------------------
# Derived helpers (ports of the environment's own maths)
# --------------------------------------------------------------------------
def shape(func: str, x: float, t: float | None = None) -> float:
    """Price-curve shape function. Mirrors `kaggriculture._shape`."""
    x = max(0.0, x)
    if func == "linear":
        return x
    if func == "sq":
        return x * x
    if func == "sqrt":
        return math.sqrt(x)
    if func == "log":
        return math.log(1.0 + x)
    if func == "log10":
        return math.log10(1.0 + x)
    if func == "hinge":
        if not t or t <= 0:
            return x
        u = x / t
        return u + HINGE_GAIN * max(0.0, u - 1.0) ** 2
    return x


def market_price(item: str, inventory: float) -> int:
    """Price of `item` at a given market inventory. Mirrors `kaggriculture.market_price`."""
    p = MARKET_PARAMS[item]
    base, i0, t = p["base"], p["I0"], p["T"]
    if inventory < i0:
        f = p["below_func"]
        amp = p["below_target"] * base / shape(f, t, t)
        price = base + amp * shape(f, i0 - inventory, t)
    else:
        f = p["above_func"]
        amp = p["above_target"] * base / shape(f, t, t)
        price = base - amp * shape(f, inventory - i0, t)
    return max(PRICE_FLOOR, int(round(price)))


def fib(n: int) -> int:
    """Hire-cost sequence: fib(0)=1, fib(1)=1, fib(2)=2, fib(3)=3, fib(4)=5..."""
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def hire_cost(hires_today: int, mult: int = FARM_HAND_COST_MULT) -> int:
    """Cost of the next hire given how many were already hired today."""
    return mult * fib(hires_today)


def cumulative_hire_cost(n: int, mult: int = FARM_HAND_COST_MULT) -> int:
    """Total cost of hiring `n` hands from a fresh day."""
    return mult * sum(fib(i) for i in range(n))


def sell_revenue(item: str, units: int, inventory: float) -> int:
    """Revenue from selling `units` one at a time, each re-quoted as inventory rises.

    The environment prices every unit of a multi-unit SELL separately, so a big
    sell walks its own price down. This is the function the agent uses to decide
    how much is worth dumping right now.
    """
    total = 0
    inv = inventory
    for _ in range(units):
        price = market_price(item, inv)
        total += price
        if price > PRICE_FLOOR:  # sales at the floor do not add supply
            inv += 1
    return total


def marginal_sell_price(item: str, inventory: float) -> int:
    """Price the next single unit would fetch."""
    return market_price(item, inventory)


def quadrant_of(x: int, y: int, board_size: int) -> str:
    half = board_size // 2
    return ("N" if y < half else "S") + ("W" if x < half else "E")


def shed_access_tiles(board_size: int) -> list[tuple[int, int]]:
    """The four inner-corner tiles around the shed, in NWSE order."""
    half = board_size // 2
    return [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]


def is_shed_adjacent(pos: tuple[int, int] | list[int], board_size: int) -> bool:
    return tuple(pos) in set(shed_access_tiles(board_size))


# --------------------------------------------------------------------------
# Forward price projection
# --------------------------------------------------------------------------
# Town demand drains market inventory all season, which lifts prices well above
# base (observed: strawberry 120 -> 316 by day 25). A crop planted today is sold
# one to three weeks from now, so valuing it at today's price systematically
# undervalues the long-cycle, high-margin crops. These constants let the agent
# price a harvest at the market it will actually meet.

_TURNS_PER_DAY = 24
_SHOP_SELL_INTERVAL = 4
_SHOP_TICKS_PER_DAY = _TURNS_PER_DAY // _SHOP_SELL_INTERVAL

#: Expected units of each product pulled per shop instance per tick. Shops are
#: drawn uniformly with replacement, and a single-product shop pulls double.
SHOP_PULL_RATE: Final[dict[str, float]] = {
    product: sum(
        (2.0 if len(prods) == 1 else 1.0)
        for prods in SHOPS.values()
        if product in prods
    ) / len(SHOPS)
    for product in PRODUCTS
}

_SHOP_UNLOCK_INTERVAL = 3


def shops_unlocked_at(day: int) -> int:
    """Expected number of shop instances open on a given day."""
    return max(0, min(MAX_SHOP_INSTANCES, day // _SHOP_UNLOCK_INTERVAL))


def daily_town_drain(
    item: str, day: int, unlocked_shops: Sequence[str] | None = None
) -> float:
    """Units of `item` the town removes from the market on `day`.

    Two models, and which one you get depends on whether you pass what you can
    actually see:

    - `unlocked_shops is None` - every open instance is assumed to be an
      *average* shop, via `SHOP_PULL_RATE`. Correct only in expectation.
    - a list - the observed instances are priced **exactly**, the way the
      environment consumes them, and only the instances not yet drawn fall back
      to the average.

    The averaged model can be badly wrong, because single-product shops consume
    double and the realized draw is small (at most `MAX_SHOP_INSTANCES`). A town
    of YARN_STOREs pulls WOOL at 2.0 per instance per tick where the average
    says 0.250 - a factor of **eight**. CARROT via PET_CAFE is 5.3x, EGG via
    BAKERY 4x. Crop and animal choice are ranked on this number, so the error
    goes straight into the most strategic decision the agent makes.

    `unlocked_shops` is in every observation and is already parsed onto
    `View.unlocked_shops`; passing it costs nothing. It may legitimately contain
    duplicates - shops are drawn with replacement and each instance consumes
    independently.
    """
    if unlocked_shops is None:
        shops = shops_unlocked_at(day) * _SHOP_TICKS_PER_DAY * SHOP_PULL_RATE.get(item, 0.0)
    else:
        # Exactly what `_town_consume` does, per instance, per tick.
        known = sum(
            (2.0 if len(SHOPS[name]) == 1 else 1.0)
            for name in unlocked_shops
            if name in SHOPS and item in SHOPS[name]
        )
        # Instances that will open later are genuinely unknown, so they keep the
        # averaged rate. The fix is to use the facts we have, not to invent more.
        unknown = max(0, shops_unlocked_at(day) - len(unlocked_shops))
        shops = _SHOP_TICKS_PER_DAY * (
            known + unknown * SHOP_PULL_RATE.get(item, 0.0)
        )
    center = 1.0 if item in TOWN_CENTER_PRODUCTS else 0.0
    return shops + center


def project_inventory(
    item: str,
    inventory: float,
    from_day: int,
    to_day: int,
    unlocked_shops: Sequence[str] | None = None,
) -> float:
    """Market inventory of `item` at `to_day`, ignoring player sales.

    Deliberately optimistic about scarcity - it does not model the opponent
    dumping supply - so it is used only to *rank* options against each other,
    never as a promised price.
    """
    if to_day <= from_day:
        return inventory
    drain = sum(
        daily_town_drain(item, d, unlocked_shops) for d in range(from_day, to_day)
    )
    return inventory - drain


def projected_price(
    item: str,
    inventory: float,
    from_day: int,
    to_day: int,
    unlocked_shops: Sequence[str] | None = None,
) -> int:
    """Price `item` is expected to fetch at `to_day`."""
    return market_price(
        item, project_inventory(item, inventory, from_day, to_day, unlocked_shops)
    )


def projected_sell_revenue(
    item: str,
    units: int,
    inventory: float,
    from_day: int,
    to_day: int,
    unlocked_shops: Sequence[str] | None = None,
) -> int:
    """Revenue from selling `units` at the projected future market."""
    return sell_revenue(
        item, units, project_inventory(item, inventory, from_day, to_day, unlocked_shops)
    )
