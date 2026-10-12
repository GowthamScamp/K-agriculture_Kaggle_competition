"""Ordered semantics and loss-accounting regressions, not guessed preconditions."""

import copy

import pytest

from src.agents import competitive_meta as meta
from src.diagnostics import error_counters, preview_units
from src.runner import EpisodeTelemetry, make_env, run_episode


def observation(seat=0):
    env = make_env(seed=3)
    env.reset(2)
    obs = copy.deepcopy(dict(env.state[seat].observation))
    farm = obs["farms"][seat]
    farm["hands"] = [[4, 4], [4, 4]]
    obs["private"]["inventories"] = [{}, {}, {}]
    return obs


@pytest.mark.parametrize("seat", [0, 1])
def test_pickup_then_drop_frees_capacity_and_preserves_observation(seat):
    obs = observation(seat)
    obs["private"]["shed"] = {"WHEAT": 100}
    obs["private"]["inventories"][1] = {"MILK": 3}
    before = copy.deepcopy(obs)
    action = {"farmer": ["PICKUP", "WHEAT", 3], "hands": [["DROP"], ["PASS"]], "market": []}
    _, private, effects = preview_units(obs, action)
    assert private["shed"] == {"WHEAT": 97, "MILK": 3}
    assert effects[1].deposited == 3 and effects[1].discarded == 0
    assert obs == before
    assert meta._protect_drops_and_shadow_shed(obs, action) == private["shed"]
    assert action["hands"][0] == ["DROP"]


def test_shared_pickup_and_water_are_not_counted_twice():
    obs = observation()
    obs["private"]["shed"] = {"WHEAT": 1}
    telemetry = EpisodeTelemetry()
    telemetry.record(obs, {"farmer": ["PICKUP", "WHEAT"],
                           "hands": [["PICKUP", "WHEAT"], ["WEST"]], "market": []})
    assert telemetry.successful_unit_ops["PICKUP"] == 1
    assert telemetry.noop_unit_ops["PICKUP"] == 1


def test_atomic_seed_rejection_and_same_tile_ordered_chain():
    obs = observation()
    obs["private"]["seeds"] = {"WHEAT": 1}
    _, private, effects = preview_units(obs, {"farmer": ["PLANT", "WHEAT"],
                                            "hands": [["PLANT", "WHEAT"]], "market": []})
    assert private["seeds"]["WHEAT"] == 1
    assert sum(effect.blocked_seed for effect in effects) == 2
    farm, _, effects = preview_units(obs, {"farmer": ["PLANT", "WHEAT"],
                                          "hands": [["WATER"], ["WATER"]], "market": []})
    assert farm["tiles"][4][4]["watered_today"]
    assert [effect.changed for effect in effects] == [True, True, False]


def test_place_animal_at_shed_access_has_priority_over_deposit():
    obs = observation()
    obs["farms"][0]["tiles"][4][4] = {"kind": "PASTURE"}
    obs["private"]["inventories"][0] = {"COW": 1}
    action = {"farmer": ["PLACE", "COW"], "hands": [], "market": []}
    farm, private, _ = preview_units(obs, action)
    assert farm["tiles"][4][4]["animal"] == "COW"
    assert meta._protect_drops_and_shadow_shed(obs, action).get("COW", 0) == 0
    assert private["shed"] == obs["private"]["shed"]


def test_out_of_bounds_move_is_a_real_noop():
    obs = observation()
    obs["farms"][0]["farmer"] = [0, 0]
    _, _, effects = preview_units(obs, {"farmer": ["NORTH"], "hands": [], "market": []})
    assert not effects[0].changed


def test_meta_error_is_observable_and_resets_for_a_new_episode(monkeypatch):
    obs = observation()
    obs.update(step=8, day=0, hour=8)
    original = meta._upstream_agent
    def boom(_obs):
        raise RuntimeError("forced test exception")
    monkeypatch.setattr(meta, "_upstream_agent", boom)
    assert meta.agent(obs)["farmer"] == ["PASS"]
    assert error_counters(meta.agent)["_POLICY_DIAGNOSTICS.errors"] == 1
    monkeypatch.setattr(meta, "_upstream_agent", original)
    obs.update(step=None, day=0, hour=0)
    meta.agent(obs)
    assert error_counters(meta.agent)["_POLICY_DIAGNOSTICS.errors"] == 0


def test_night_loss_and_executed_market_units_use_actual_engine():
    def policy(obs, config=None):
        step = int(obs["step"])
        if step == 0:
            return {"farmer": ["PASS"], "hands": [], "market": [["BUY_PRODUCT", "WHEAT", 3]]}
        if step == 1:
            return {"farmer": ["PICKUP", "WHEAT", 3], "hands": [], "market": [["BUY_PRODUCT", "WHEAT", 3]]}
        return {"farmer": ["PASS"], "hands": [], "market": []}
    result = run_episode([policy, "pass"], seed=2, telemetry_seat=0, episodeSteps=27, shedCapacity=3)
    assert result.telemetry.discarded_nightly_units == 3
    assert result.telemetry.executed_market_units["BUY_PRODUCT:WHEAT"] == 6
