"""Agent safety and timing.

These are the tests that protect the leaderboard run rather than the score: an
agent that raises, emits a malformed action, or overruns the 1-second
`actTimeout` forfeits an episode no matter how good its strategy is.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from src.env_adapter import View, safe_action
from src.game_constants import ANIMALS, CROPS, FARMER_MOVES, PRODUCTS
from src.registry import build_agent
from src.runner import _quiet, make_env, run_episode
from src.simulator import simulator_available

UNIT_OPS = set(FARMER_MOVES) | {
    "PASS", "PICKUP", "PLACE", "DROP", "PLANT", "WATER", "HARVEST", "FERTILIZE",
    "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER", "CARE", "DIG",
}
MARKET_OPS = {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL", "HIRE", "BUY_LAND"}
ACT_TIMEOUT_S = 1.0


def _assert_legal(action: Any, n_hands: int) -> None:
    assert isinstance(action, dict), f"action must be a dict, got {type(action)}"
    assert set(action) <= {"farmer", "hands", "market"}, f"unexpected keys: {set(action)}"

    farmer = action.get("farmer", ["PASS"])
    assert isinstance(farmer, list) and farmer, "farmer op must be a non-empty list"
    assert farmer[0] in UNIT_OPS, f"illegal farmer op {farmer[0]!r}"

    hands = action.get("hands", [])
    assert isinstance(hands, list)
    assert len(hands) <= n_hands, f"{len(hands)} hand ops for {n_hands} hands"
    for h in hands:
        assert isinstance(h, list) and h, "hand op must be a non-empty list"
        assert h[0] in UNIT_OPS, f"illegal hand op {h[0]!r}"

    market = action.get("market", [])
    assert isinstance(market, list)
    assert len(market) <= 10, f"{len(market)} market orders exceeds the per-turn cap"
    for order in market:
        assert isinstance(order, list) and order, "market order must be a non-empty list"
        op = order[0]
        assert op in MARKET_OPS, f"illegal market op {op!r}"
        if op in ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"):
            assert len(order) >= 3, f"{op} needs an item and a count: {order}"
            assert int(order[2]) > 0, f"{op} count must be positive: {order}"
            if op == "BUY_SEED":
                assert order[1] in CROPS, f"unknown crop {order[1]!r}"
            elif op == "BUY_ANIMAL":
                assert order[1] in ANIMALS, f"unknown animal {order[1]!r}"
            elif op == "SELL":
                assert order[1] in PRODUCTS or order[1] in ANIMALS, f"unsellable {order[1]!r}"
            elif op == "BUY_PRODUCT":
                assert order[1] in ("WHEAT", "FERTILIZER"), (
                    f"only WHEAT and FERTILIZER are purchasable, got {order[1]!r}"
                )


@pytest.mark.parametrize(
    "name", ["do_nothing", "baseline", "fast", "reactive_route", "competitive_meta"]
)
def test_actions_are_legal_all_episode(name: str) -> None:
    """Every action of a full 720-turn episode must be well-formed."""
    agent = build_agent(name)
    env = make_env(seed=5)
    checked = 0

    def probe(obs, config=None):
        nonlocal checked
        action = agent(obs, config) if callable(agent) else {}
        n_hands = len(obs["farms"][obs["player"]]["hands"])
        _assert_legal(action, n_hands)
        checked += 1
        return action

    with _quiet():
        env.run([probe, "starter"])
    assert checked >= 700, f"only {checked} turns were checked"
    assert env.steps[-1][0]["status"] == "DONE"


def test_do_nothing_keeps_exactly_starting_money() -> None:
    result = run_episode([build_agent("do_nothing"), "pass"], seed=2)
    assert result.steps == 720
    assert result.bank == 3000.0, "a passing agent should neither earn nor spend"


def test_baseline_beats_starter_on_several_seeds() -> None:
    """A sanity floor, not a performance claim - that lives in evaluate.py."""
    wins = 0
    for seed in range(4):
        r = run_episode([build_agent("baseline"), "starter"], seed=seed)
        assert r.statuses[0] == "DONE", f"seed {seed} ended {r.statuses[0]}"
        wins += r.won
    assert wins >= 3, f"baseline only beat starter on {wins}/4 seeds"


def test_agent_never_raises_on_malformed_observation() -> None:
    """The top-level guard must swallow anything and still return a legal action."""
    agent = build_agent("baseline")
    for bad in (
        {},
        {"player": 0, "farms": []},
        {"player": 0, "farms": [{}], "private": {}, "market": {}, "town": {}},
        {"player": 5, "farms": [{"tiles": [], "money": 0}], "step": 0},
        {"player": 0, "farms": [{"tiles": [[None]], "money": "x", "farmer": [0, 0],
                                 "hands": [], "unlocked_quadrants": ["NW"],
                                 "hires_today": 0}],
         "private": {"shed": {}, "seeds": {}, "inventories": [{}]},
         "market": {"inventory": {}, "prices": {}}, "town": {}, "step": 0,
         "day": 0, "hour": 0},
    ):
        action = agent(bad)
        assert isinstance(action, dict)
        assert action.get("farmer", ["PASS"])[0] in UNIT_OPS


def test_episode_survives_an_agent_that_always_raises() -> None:
    """A broken decide must degrade to a legal action, never forfeit the episode."""
    from src.agents.baseline_agent import BaselineAgent

    class Exploding(BaselineAgent):
        def decide(self, v: View):
            raise RuntimeError("injected failure")

    agent = Exploding()
    env = make_env(seed=1)
    with _quiet():
        env.run([agent, "starter"])
    final = env.steps[-1]
    assert len(env.steps) == 720
    assert final[0]["status"] == "DONE", "episode did not complete"
    assert final[0]["reward"] == 3000.0, "a fully failing agent should sit on its cash"


def test_safe_action_shape() -> None:
    for n in (0, 1, 7):
        a = safe_action(n)
        _assert_legal(a, n)
        assert len(a["hands"]) == n


@pytest.mark.skipif(not simulator_available(), reason="kaggle_environments not installed")
def test_search_agent_respects_the_turn_limit() -> None:
    """No single turn may exceed the environment's 1s actTimeout.

    This is the failure mode that silently kills a ladder run: the agent plays
    well, overruns on one turn late in the season, and forfeits.
    """
    from src.mcts_agent import MCTSAgent

    agent = MCTSAgent(budget_ms=450)
    env = make_env(seed=4)
    times: list[float] = []

    def probe(obs, config=None):
        t0 = time.perf_counter()
        action = agent(obs, config)
        times.append(time.perf_counter() - t0)
        return action

    with _quiet():
        env.run([probe, "starter"])

    worst = max(times)
    assert worst < ACT_TIMEOUT_S, (
        f"slowest turn took {worst * 1000:.0f} ms, over the {ACT_TIMEOUT_S * 1000:.0f} ms limit"
    )
    assert env.steps[-1][0]["status"] == "DONE"


@pytest.mark.skipif(not simulator_available(), reason="kaggle_environments not installed")
def test_search_falls_back_when_simulation_is_unavailable() -> None:
    """With search disabled the agent must still play, as plain baseline."""
    from src.mcts_agent import MCTSAgent

    agent = MCTSAgent(budget_ms=200)
    agent._available = False
    r = run_episode([agent, "starter"], seed=6)
    assert r.statuses[0] == "DONE"
    assert r.bank > 3000.0, "fallback play should still make money"


@pytest.mark.skipif(not simulator_available(), reason="kaggle_environments not installed")
def test_search_is_deterministic_when_the_budget_is_never_binding() -> None:
    """Two runs with a never-binding budget must choose identically.

    Historically this test carried the whole reproducibility story, and it was
    too weak for that job. It only ever exercised the *non-binding* regime, so
    it passed happily throughout the period when the candidate list was being
    truncated by machine speed - see
    `test_candidate_set_does_not_depend_on_machine_speed`, which covers the case
    that actually broke, and docs/NOTES.md (30 Aug 2026) for what it cost.

    What this still usefully pins: with the deadline out of the picture there is
    no *other* source of nondeterminism in the search. If it ever fails, one has
    been introduced.
    """
    from src.mcts_agent import MCTSAgent

    def run() -> list[str]:
        agent = MCTSAgent(budget_ms=60_000, rollout_days=1, max_candidates=4)

        def play(obs, config=None):
            return agent(obs, config)

        env = make_env(seed=11)
        with _quiet():
            env.run([play, "starter"])
        assert not any(h.timed_out for h in agent.history), (
            "budget bound after all - the test is not measuring what it claims"
        )
        return [h.chosen.label for h in agent.history]

    first, second = run(), run()
    assert first, "the agent never reached a decision point"
    assert first == second, (
        "identical configuration chose different macro-actions; "
        f"run 1 chose {first} but run 2 chose {second}"
    )


def test_candidate_set_does_not_depend_on_machine_speed() -> None:
    """The agent's *strategy* must not be a function of how fast the box is.

    Until 30 Aug 2026 the candidate list was truncated to whatever the measured
    rollout cost said the wall-clock budget could afford:

        affordable = int(self.budget_s / self._est_rollout_s(horizon0))
        candidates = candidates[:max(2, affordable)]

    so a loaded machine searched a smaller problem than an idle one. That is not
    a rounding error. Over 100 identical seeds the candidate count decides the
    entire result - 1 candidate is the plain heuristic at 50.5%, 2 is the best
    agent we have at 86%, 5 is far worse than not searching at all at 30%. Every
    historical win rate in this repo was really a measurement of CPU load, which
    is why three separate verification passes each "confirmed" a different one.

    A punishing budget may still stop the sweep *finishing* - that is the
    actTimeout safety net, and a partial sweep is discarded in favour of the
    heuristic. What it must never do is silently search a different problem.

    `test_search_is_deterministic_when_the_budget_is_never_binding` covers the
    non-binding regime and passed throughout the era of this bug; the damage
    lived in the transition between regimes, which nothing tested.
    """
    from src.mcts_agent import MCTSAgent

    def candidate_counts(budget_ms: float) -> list[int]:
        agent = MCTSAgent(budget_ms=budget_ms, rollout_days=2, max_candidates=5)

        def play(obs, config=None):
            return agent(obs, config)

        env = make_env(seed=11)
        with _quiet():
            env.run([play, "starter"])
        return [h.n_candidates for h in agent.history]

    generous = candidate_counts(60_000)
    punishing = candidate_counts(1)

    assert generous, "the agent never reached a decision point"
    assert generous == punishing, (
        "the candidate set changed with the time budget, so the agent plays a "
        "different strategy on a slow machine than a fast one. "
        f"generous budget considered {generous}, punishing budget {punishing}"
    )


# ==========================================================================
# Determinization (src/determinization.py)
# ==========================================================================
# `mcts_det` is reachable from `src/registry.py`, so it is shipped-adjacent code
# that any experiment or a future FOOTER change could put in front of Kaggle -
# and until now nothing tested it at all. These are deliberately cheap: the point
# is that the path runs, degrades safely, and inherits the machine-speed fix, not
# to re-measure its strength (that is `tools/compare_agents.py`).


@pytest.mark.skipif(not simulator_available(), reason="needs kaggle_environments")
@pytest.mark.parametrize("aggregation", ["mean", "worst"])
def test_determinized_agent_plays_a_legal_episode(aggregation: str) -> None:
    """Both aggregation rules must survive a real episode and stay legal."""
    agent = build_agent("mcts_det", aggregation=aggregation, n_determinizations=2)
    env = make_env(seed=5)
    checked = 0
    worst_turn = 0.0

    def probe(obs, config=None):
        nonlocal checked, worst_turn
        t0 = time.perf_counter()
        action = agent(obs, config)
        worst_turn = max(worst_turn, time.perf_counter() - t0)
        _assert_legal(action, len(obs["farms"][obs["player"]]["hands"]))
        checked += 1
        return action

    with _quiet():
        env.run([probe, "starter"])

    assert checked >= 700, f"only {checked} turns were checked"
    assert env.steps[-1][0]["status"] == "DONE"
    assert worst_turn < 1.0, (
        f"slowest turn {worst_turn * 1000:.0f} ms exceeds the 1 s actTimeout; "
        "determinization multiplies rollout cost by n_determinizations"
    )


def test_determinized_agent_rejects_an_unknown_aggregation() -> None:
    """The guard exists in `__init__` but nothing exercised it.

    Worth a test because the value arrives from `--agent-args` JSON and a typo
    would otherwise be discovered only as strange results, not as an error.
    """
    with pytest.raises(ValueError, match="aggregation"):
        build_agent("mcts_det", aggregation="bogus")


@pytest.mark.skipif(not simulator_available(), reason="needs kaggle_environments")
def test_determinized_candidate_set_also_ignores_machine_speed() -> None:
    """`DeterminizedAgent` subclasses `MCTSAgent`, so it inherits the fix.

    Checked rather than assumed: the subclass overrides `__init__` and passes its
    own `max_candidates` default through, so it is exactly the kind of place a
    reintroduced 5 - or a reintroduced speed-based truncation - would hide.
    """
    def candidate_counts(budget_ms: float) -> list[int]:
        agent = build_agent("mcts_det", budget_ms=budget_ms, n_determinizations=2)
        env = make_env(seed=11)
        with _quiet():
            env.run([lambda obs, config=None: agent(obs, config), "starter"])
        return [h.n_candidates for h in agent.history]

    generous = candidate_counts(60_000)
    punishing = candidate_counts(1)
    assert generous, "the agent never reached a decision point"
    assert generous == punishing, (
        "determinized search considered a different candidate set under a "
        f"different budget: {generous} vs {punishing}"
    )
