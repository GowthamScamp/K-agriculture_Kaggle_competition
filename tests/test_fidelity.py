"""Simulator fidelity: does `simulate_step` reproduce the real environment?

Because `src/simulator.py` reuses the environment's own `interpreter()`, the
expected answer is *exact* agreement, not agreement within a tolerance. These
tests are written to fail loudly the moment that stops being true - which is the
signal that our state reconstruction has drifted from the real schema, and that
everything the search believes is now suspect.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.registry import build_agent
from src.runner import _quiet, make_env
from src.simulator import (
    EnvShim,
    clone_state,
    simulate_step,
    simulator_available,
    state_from_obs,
)

pytestmark = pytest.mark.skipif(
    not simulator_available(), reason="kaggle_environments not installed"
)

SEED = 11
#: Deliberately shorter than a full episode: `tools/bench_sim.py` compares all
#: 719 steps and is run as a release gate, so this keeps the unit suite fast
#: while still covering a full in-game day boundary many times over.
STEPS_TO_CHECK = 240


def _tile_key(t: Any) -> Any:
    if t is None or isinstance(t, str):
        return t
    return {k: t[k] for k in sorted(t)}


def _canonical(state: list[Any], seat: int) -> dict[str, Any]:
    obs = state[0].observation
    return {
        "farms": [
            {
                "money": round(float(f["money"]), 6),
                "farmer": list(f["farmer"]),
                "hands": [list(h) for h in f["hands"]],
                "quads": sorted(f["unlocked_quadrants"]),
                "hires": f["hires_today"],
                "tiles": [[_tile_key(t) for t in row] for row in f["tiles"]],
            }
            for f in obs.farms
        ],
        "market_inv": {k: round(float(v), 6) for k, v in obs.market["inventory"].items()},
        "prices": dict(obs.market["prices"]),
        "town": list(obs.town["unlocked_shops"]),
        "shed": {k: v for k, v in state[seat].observation.private["shed"].items() if v},
        "seeds": {k: v for k, v in state[seat].observation.private["seeds"].items() if v},
        "day": obs.day,
        "hour": obs.hour,
    }


@pytest.fixture(scope="module")
def recorded():
    """A real episode, with every observation and both seats' actions kept."""
    agent = build_agent("baseline")
    from kaggle_environments.envs.kaggriculture import kaggriculture as kg

    mine: list[dict[str, Any]] = []
    theirs: list[dict[str, Any]] = []

    def seat0(obs, config=None):
        act = agent(obs, config)
        mine.append(act)
        return act

    def seat1(obs, config=None):
        act = kg.starter_agent(obs)
        theirs.append(act)
        return act

    env = make_env(seed=SEED)
    with _quiet():
        env.run([seat0, seat1])
    if hasattr(agent, "close"):
        agent.close()

    truth = [json.loads(json.dumps(dict(s[0].observation))) for s in env.steps]
    privates = [
        json.loads(json.dumps(dict(s[1]["observation"]["private"]))) for s in env.steps
    ]
    return {
        "config": dict(env.configuration),
        "seed": env.info.get("seed", SEED),
        "mine": mine,
        "theirs": theirs,
        "truth": truth,
        "opp_private": privates,
    }


def test_simulator_matches_environment_exactly(recorded):
    """Replay real actions; every simulated state must equal the recorded one."""
    shim = EnvShim(config=recorded["config"], seed=recorded["seed"])
    state = state_from_obs(
        recorded["truth"][0], opponent_private=recorded["opp_private"][0], step=0
    )

    n = min(STEPS_TO_CHECK, len(recorded["mine"]), len(recorded["theirs"]),
            len(recorded["truth"]) - 1)
    assert n > 100, "episode too short to be a meaningful fidelity check"

    for i in range(n):
        simulate_step(state, recorded["mine"][i], recorded["theirs"][i],
                      shim, seat=0, in_place=True)
        expected = _canonical(
            state_from_obs(recorded["truth"][i + 1],
                           opponent_private=recorded["opp_private"][i + 1], step=i + 1),
            0,
        )
        got = _canonical(state, 0)
        if got != expected:
            diverged = [k for k in expected if got[k] != expected[k]]
            pytest.fail(
                f"simulator diverged from the environment at step {i + 1} "
                f"in {diverged}. The simulator reuses interpreter(), so this means "
                f"state reconstruction is wrong, not that the model is approximate."
            )


def test_clone_is_independent(recorded):
    """A cloned state must not share mutable structure with its source."""
    state = state_from_obs(recorded["truth"][40], step=40)
    copy = clone_state(state)

    # Snapshot before mutating, so the shop assertion tests *independence*
    # rather than the incidental fact that this step has no shops open. It used
    # to read `... or True`, which made it unconditionally pass; presumably
    # defensive, since a fixture at a later step legitimately could have BAKERY
    # unlocked. Comparing against the snapshot is correct at any step.
    shops_before = list(state[0].observation.town["unlocked_shops"])

    copy[0].observation.farms[0]["money"] = -12345.0
    copy[0].observation.farms[0]["tiles"][0][0] = {"kind": "WEED"}
    copy[0].observation.market["inventory"]["WHEAT"] = 1
    copy[0].observation.private["shed"]["WHEAT"] = 999
    copy[0].observation.town["unlocked_shops"].append("BAKERY")

    assert state[0].observation.farms[0]["money"] != -12345.0
    assert state[0].observation.farms[0]["tiles"][0][0] != {"kind": "WEED"}
    assert state[0].observation.market["inventory"]["WHEAT"] != 1
    assert state[0].observation.private["shed"].get("WHEAT") != 999
    assert state[0].observation.town["unlocked_shops"] == shops_before, (
        "appending to the clone's shop list mutated the original - the town "
        "dict is aliased, not copied"
    )


def test_clone_preserves_struct_tiles(recorded):
    """Tiles arriving as `Struct` (a dict subclass) must still be deep-copied.

    Regression guard: an identity check (`type(t) is dict`) silently failed for
    `Struct`, which both aliased cloned tiles and made the agent treat every
    structured tile as locked.
    """
    from kaggle_environments.utils import structify

    obs = structify(recorded["truth"][60])
    state = state_from_obs(obs, step=60)
    copy = clone_state(state)
    src_tiles = state[0].observation.farms[0]["tiles"]
    cpy_tiles = copy[0].observation.farms[0]["tiles"]
    for y, row in enumerate(src_tiles):
        for x, tile in enumerate(row):
            if isinstance(tile, dict):
                assert cpy_tiles[y][x] is not tile, f"tile ({x},{y}) is aliased, not copied"


def test_seats_share_public_state(recorded):
    """Both seats must observe the same farms/market/town objects, as in the env."""
    state = state_from_obs(recorded["truth"][10], step=10)
    assert state[0].observation.farms is state[1].observation.farms
    assert state[0].observation.market is state[1].observation.market
    assert state[0].observation.private is not state[1].observation.private
