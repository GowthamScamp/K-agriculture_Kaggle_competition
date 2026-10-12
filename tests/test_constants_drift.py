"""Guard: the vendored constants must still match the installed environment.

`src/game_constants.py` holds a hand-copied version of the game tables so the
submission stays self-contained. A copy that silently goes stale is worse than no
copy at all - every price, yield and cost the agent reasons about would be wrong
while the code kept running. If Kaggle rebalances the game, this is the test that
says so.
"""

from __future__ import annotations

import pytest

from src import game_constants as gc
from src.simulator import simulator_available

pytestmark = pytest.mark.skipif(
    not simulator_available(), reason="kaggle_environments not installed"
)


@pytest.fixture(scope="module")
def kg():
    from kaggle_environments.envs.kaggriculture import kaggriculture as mod

    return mod


@pytest.mark.parametrize("table", ["CROPS", "ANIMALS", "MARKET_PARAMS", "SHOPS"])
def test_tables_match(kg, table):
    assert getattr(gc, table) == getattr(kg, table), (
        f"{table} has drifted from the installed environment; "
        "regenerate docs/ENVIRONMENT.md and update src/game_constants.py"
    )


@pytest.mark.parametrize(
    "name",
    ["PRODUCTS", "MARKET_I0", "PRICE_FLOOR", "HINGE_GAIN", "FARMER_MOVES",
     "LAND_ORDER", "LAND_PRICES", "MAX_SHOP_INSTANCES", "TOWN_CENTER_PRODUCTS",
     "FARM_HAND_COST_MULT"],
)
def test_scalars_match(kg, name):
    assert getattr(gc, name) == getattr(kg, name), f"{name} has drifted"


def test_market_price_matches_environment(kg):
    """Our price function must agree with the environment at every scale."""
    for item in gc.PRODUCTS:
        for delta in (-5000, -900, -450, -100, -1, 0, 1, 100, 450, 900, 5000, 20000):
            inv = gc.MARKET_I0 + delta
            assert gc.market_price(item, inv) == kg.market_price(item, inv), (
                f"price mismatch for {item} at inventory offset {delta}"
            )


def test_hire_cost_matches_environment(kg):
    for n in range(20):
        assert gc.hire_cost(n) == kg._hire_cost(n), f"hire cost mismatch at n={n}"
    assert gc.cumulative_hire_cost(10) == sum(kg._fib(i) for i in range(10))


def test_shed_geometry_matches_environment(kg):
    for board in (4, 6, 10):
        assert gc.shed_access_tiles(board) == kg._shed_access_tiles(board)
        for x in range(board):
            for y in range(board):
                assert gc.quadrant_of(x, y, board) == kg._quadrant_of(x, y, board)


def test_sell_revenue_matches_a_real_sequence(kg):
    """Selling n units one at a time must price exactly as the market loop does."""
    for item in ("WHEAT", "MELON", "STRAWBERRY"):
        inv = float(gc.MARKET_I0)
        expected = 0
        for _ in range(25):
            price = kg.market_price(item, inv)
            expected += price
            if price > kg.PRICE_FLOOR:
                inv += 1
        assert gc.sell_revenue(item, 25, gc.MARKET_I0) == expected, item


def test_town_drain_matches_shop_tables(kg):
    """The forward price projection must be built on the real shop tables."""
    for product in gc.PRODUCTS:
        expected = sum(
            (2.0 if len(prods) == 1 else 1.0)
            for prods in kg.SHOPS.values()
            if product in prods
        ) / len(kg.SHOPS)
        assert gc.SHOP_PULL_RATE[product] == pytest.approx(expected), product
    assert gc.daily_town_drain("MELON", 0) == pytest.approx(1.0), (
        "no shop buys MELON, so day-0 demand is the town centre alone"
    )


# ==========================================================================
# Town demand: the realized-shop model
# ==========================================================================
def test_realized_town_drain_matches_the_environment_exactly():
    """Our realized-shop demand must reproduce the env's own consumption.

    This is a parity test against ground truth, not a restatement of the
    formula. Both seats play `pass`, so nobody sells and *every* change in
    market inventory is town consumption - which makes the observed delta over
    one in-game day directly comparable to `daily_town_drain`.

    It is the test that earns the feature. Asserting the arithmetic would only
    prove our formula equals itself; comparing against the real interpreter
    proves the model is right.
    """
    from src.runner import _quiet, make_env

    env = make_env(seed=5)
    with _quiet():
        env.run(["pass", "pass"])
    steps = env.steps

    checked = 0
    for day in range(29):
        a, b = day * 24, (day + 1) * 24
        if b >= len(steps):
            break
        before = steps[a][0]["observation"]
        after = steps[b][0]["observation"]
        shops = list(before["town"]["unlocked_shops"])
        for item in ("WHEAT", "WOOL", "CARROT", "EGG", "MILK"):
            observed = (before["market"]["inventory"][item]
                        - after["market"]["inventory"][item])
            predicted = gc.daily_town_drain(item, day, shops)
            assert predicted == pytest.approx(observed), (
                f"day {day} {item}: predicted {predicted} but the town actually "
                f"removed {observed} with shops {shops}"
            )
            checked += 1
    assert checked >= 100, f"only {checked} day/item pairs compared"


def test_averaged_model_is_the_one_that_is_wrong():
    """Pins *why* the realized model exists, so nobody 'simplifies' it away.

    Same ground truth as above. The averaged model - every open instance treated
    as an average shop - is off by whole units per day and, on a town that draws
    several copies of one single-product shop, by more than an order of
    magnitude. Crop and animal ranking are both built on this number.
    """
    from src.runner import _quiet, make_env

    env = make_env(seed=5)
    with _quiet():
        env.run(["pass", "pass"])
    steps = env.steps

    worst_ratio = 1.0
    for day in range(29):
        a, b = day * 24, (day + 1) * 24
        if b >= len(steps):
            break
        before, after = steps[a][0]["observation"], steps[b][0]["observation"]
        shops = list(before["town"]["unlocked_shops"])
        for item in ("CARROT", "WOOL"):
            truth = (before["market"]["inventory"][item]
                     - after["market"]["inventory"][item])
            if truth <= 0:
                continue
            averaged = gc.daily_town_drain(item, day)
            assert gc.daily_town_drain(item, day, shops) == pytest.approx(truth)
            worst_ratio = max(worst_ratio, averaged / truth, truth / averaged)

    assert worst_ratio > 5.0, (
        "the averaged model used to be wrong by ~19x on this seed; if that gap "
        "has closed, either the env's shop rules changed or the fixture no "
        "longer exercises a lopsided town"
    )


def test_town_drain_without_shops_keeps_the_old_behaviour():
    """`unlocked_shops=None` must stay bit-identical to the pre-change model.

    `baseline` is this project's reference opponent and deliberately keeps the
    averaged model, so every stored comparison depends on this path not moving.
    """
    for item in gc.PRODUCTS:
        for day in (0, 1, 3, 7, 12, 24, 29):
            expected = (
                gc.shops_unlocked_at(day)
                * (24 // 4)
                * gc.SHOP_PULL_RATE.get(item, 0.0)
                + (1.0 if item in gc.TOWN_CENTER_PRODUCTS else 0.0)
            )
            assert gc.daily_town_drain(item, day) == pytest.approx(expected), (
                f"{item} day {day}"
            )
