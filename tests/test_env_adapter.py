"""Unit tests for the observation adapter - the one place that knows the schema."""

from __future__ import annotations

from src.env_adapter import Tile, View, distance, step_toward
from src.game_constants import CROPS

MOVES = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}


def _grid(n: int = 10, fill=None):
    return [[fill for _ in range(n)] for _ in range(n)]


def _nw_only(tiles):
    """Lock everything outside the starting NW quadrant, as a real board is."""
    n = len(tiles)
    half = n // 2
    for y in range(n):
        for x in range(n):
            if not (x < half and y < half):
                tiles[y][x] = "LOCKED"
    return tiles


def _obs(tiles, **over):
    base = {
        "player": 0,
        "step": 0,
        "day": 0,
        "hour": 0,
        "farms": [{
            "money": 1000.0, "tiles": tiles, "farmer": [4, 4], "hands": [],
            "unlocked_quadrants": ["NW"], "hires_today": 0,
        }],
        "private": {"shed": {"WHEAT": 3}, "seeds": {"WHEAT": 2},
                    "inventories": [{"CARROT": 1}]},
        "market": {"inventory": {"WHEAT": 10000}, "prices": {"WHEAT": 25}},
        "town": {"unlocked_shops": []},
    }
    base.update(over)
    return base


def _plant(crop="WHEAT", **over):
    raw = {"kind": "PLANT", "crop": crop, "planted_day": 0, "watered_today": False,
           "yield_units": 1, "consecutive_unwatered": 0, "fertilized_until_day": -1,
           "max_lifespan_step": 999}
    raw.update(over)
    return raw


def test_tile_classification():
    t = Tile(1, 2, _plant(yield_units=2))
    assert t.is_plant and not t.locked and not t.empty
    assert t.crop == "WHEAT" and t.yield_units == 2 and t.needs_water(0)

    assert Tile(0, 0, "LOCKED").locked
    assert Tile(0, 0, None).empty
    assert Tile(0, 0, {"kind": "WEED"}).is_weed
    assert Tile(0, 0, {"kind": "PASTURE"}).is_structure

    cow = {"kind": "PASTURE", "animal": "COW", "yield_units": 1, "fed_today": False,
           "cared_today": False, "fertilizer_available": False, "placed_day": 0,
           "consecutive_unfed": 0}
    assert Tile(0, 0, cow).has_animal
    assert not Tile(0, 0, cow).is_structure, "an occupied pasture is not free housing"


def test_harvestable_respects_first_yield_day():
    d = CROPS["WHEAT"]
    t = Tile(0, 0, _plant(watered_today=True))
    assert not t.harvestable(d["first_yield_day"] - 1)
    assert t.harvestable(d["first_yield_day"])
    assert not Tile(0, 0, _plant(yield_units=0)).harvestable(10)


def test_water_bonus_window():
    t = Tile(0, 0, _plant())
    d = CROPS["WHEAT"]
    start = (d["max_yield_day"] + 1) // 2
    assert not t.in_water_bonus_window(start - 1)
    assert t.in_water_bonus_window(start)
    assert t.in_water_bonus_window(d["max_yield_day"])
    assert not t.in_water_bonus_window(d["max_yield_day"] + 1)


def test_groups_is_cached_and_correct():
    tiles = _nw_only(_grid())
    tiles[0][0] = {"kind": "WEED"}
    tiles[1][1] = _plant()
    tiles[2][2] = {"kind": "COOP"}

    v = View.from_obs(_obs(tiles))
    g = v.groups()
    assert len(g.plants) == 1
    assert len(g.weeds) == 1
    assert len(g.structures) == 1
    assert g.n_owned == 25, "only the NW quadrant is unlocked"
    assert len(g.empties) == 22
    assert v.groups() is g, "groups() should be computed once per view"


def test_step_toward_is_a_single_legal_move():
    assert step_toward((4, 4), (4, 4), 10) == ["PASS"]
    assert step_toward((4, 4), (9, 4), 10) == ["EAST"]
    assert step_toward((4, 4), (0, 4), 10) == ["WEST"]
    assert step_toward((4, 4), (4, 9), 10) == ["SOUTH"]
    assert step_toward((4, 4), (4, 0), 10) == ["NORTH"]


def test_step_toward_reaches_target_and_stays_on_board():
    for dst in [(0, 0), (9, 9), (0, 9), (9, 0), (5, 3)]:
        pos = (4, 4)
        for _ in range(64):
            if pos == dst:
                break
            dx, dy = MOVES[step_toward(pos, dst, 10)[0]]
            pos = (pos[0] + dx, pos[1] + dy)
            assert 0 <= pos[0] < 10 and 0 <= pos[1] < 10
        assert pos == dst, f"never reached {dst}"


def test_net_worth_counts_more_than_cash():
    tiles = _nw_only(_grid())
    tiles[1][1] = _plant(yield_units=3, watered_today=True)
    v = View.from_obs(_obs(tiles))
    assert v.net_worth() > v.money, "standing crops and shed stock must carry value"


def test_shed_accounting_and_distance():
    v = View.from_obs(_obs(_nw_only(_grid())))
    assert v.shed_used == 3
    assert v.shed_room(100) == 97
    assert v.carried_total() == 1
    assert distance((0, 0), (3, 4)) == 7
    assert v.is_shed_tile((4, 4)) and not v.is_shed_tile((0, 0))


def test_view_handles_struct_like_tiles():
    """Tiles may arrive as a dict subclass; classification must still work.

    Regression guard for an identity check that treated every `Struct` tile as
    locked and so hid the entire board from the agent.
    """

    class Struct(dict):
        pass

    tiles = _nw_only(_grid())
    tiles[1][1] = Struct(_plant())
    v = View.from_obs(_obs(tiles))
    assert len(v.groups().plants) == 1


def test_view_derives_step_when_wrapper_supplies_none():
    """Seat wrappers may omit the redundant step while retaining day/hour."""

    v = View.from_obs(_obs(_nw_only(_grid()), step=None, day=3, hour=5))
    assert v.step == 3 * 24 + 5


def test_view_selects_seat_one_public_farm_but_keeps_its_private_state():
    tiles_a = _nw_only(_grid())
    tiles_b = _nw_only(_grid())
    obs = _obs(tiles_a)
    obs["farms"].append({
        "money": 4321.0,
        "tiles": tiles_b,
        "farmer": [1, 2],
        "hands": [[2, 2]],
        "unlocked_quadrants": ["NW"],
        "hires_today": 1,
    })
    obs["player"] = 1
    obs["private"] = {
        "shed": {"MILK": 7},
        "seeds": {"CARROT": 3},
        "inventories": [{}, {"WOOL": 2}],
    }
    view = View.from_obs(obs)
    assert view.player == 1
    assert view.money == 4321.0
    assert view.farmer_pos == (1, 2)
    assert view.shed["MILK"] == 7
    assert view.unit_inventory(1)["WOOL"] == 2
