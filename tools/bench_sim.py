"""Measure simulator throughput and fidelity. Both gate whether MCTS is viable.

    python -m tools.bench_sim

Fidelity: replay a real episode's recorded actions through `simulate_step` and
compare state to what the environment itself produced. Because the simulator
reuses `interpreter`, the expected answer is *exact* agreement, and any drift
means our state reconstruction is wrong.

Throughput: the 1s per-turn budget divided by the cost of one simulated step is
the entire search budget. Measured here rather than assumed.
"""

from __future__ import annotations

import json
import time
from typing import Any

from src.env_adapter import Action, View, safe_action
from src.registry import build_agent
from src.runner import _quiet, make_env
from src.simulator import (
    EnvShim,
    bank_of,
    clone_state,
    simulate_step,
    state_from_obs,
    view_of,
)

SEED = 7


def _canonical(state: list[Any], seat: int) -> dict[str, Any]:
    """The parts of a state that must match, in a comparable form."""
    obs = state[0].observation
    farms = []
    for f in obs.farms:
        farms.append({
            "money": round(float(f["money"]), 6),
            "farmer": list(f["farmer"]),
            "hands": [list(h) for h in f["hands"]],
            "quads": sorted(f["unlocked_quadrants"]),
            "hires": f["hires_today"],
            "tiles": [[_tile_key(t) for t in row] for row in f["tiles"]],
        })
    priv = state[seat].observation.private
    return {
        "farms": farms,
        "market_inv": {k: round(float(v), 6) for k, v in obs.market["inventory"].items()},
        "prices": dict(obs.market["prices"]),
        "town": list(obs.town["unlocked_shops"]),
        "shed": {k: v for k, v in priv["shed"].items() if v},
        "seeds": {k: v for k, v in priv["seeds"].items() if v},
        "day": obs.day,
        "hour": obs.hour,
    }


def _tile_key(t: Any) -> Any:
    if t is None or isinstance(t, str):
        return t
    return {k: t[k] for k in sorted(t)}


def fidelity() -> tuple[int, int, list[str]]:
    """Replay recorded actions through the simulator; report first divergence."""
    agent = build_agent("baseline")
    opponent = build_agent("starter")

    env = make_env(seed=SEED)
    recorded: list[dict[str, Any]] = []

    def rec_a(obs: Any, config: Any = None) -> dict[str, Any]:
        act = agent(obs, config)
        recorded.append({"obs": json.loads(json.dumps(dict(obs))), "a": act})
        return act

    opp_actions: list[Any] = []

    def rec_b(obs: Any, config: Any = None) -> dict[str, Any]:
        from kaggle_environments.envs.kaggriculture import kaggriculture as kg

        act = kg.starter_agent(obs)
        opp_actions.append(act)
        return act

    with _quiet():
        env.run([rec_a, rec_b])
    if hasattr(agent, "close"):
        agent.close()

    # Ground truth: the environment's own recorded observations, one per step.
    truth = [json.loads(json.dumps(dict(s[0].observation))) for s in env.steps]

    shim = EnvShim(config=dict(env.configuration), seed=env.info.get("seed", SEED))
    # Seat 1's private state is genuinely hidden from seat 0, so seed the
    # simulator from the recorded full state to isolate *turn logic* fidelity
    # from the separate question of opponent-state estimation.
    state = state_from_obs(recorded[0]["obs"], opponent_private=_seat1_private(env, 0), step=0)

    mismatches: list[str] = []
    compared = 0
    n = min(len(recorded), len(opp_actions), len(truth) - 1)
    for i in range(n):
        simulate_step(state, recorded[i]["a"], opp_actions[i], shim, seat=0, in_place=True)
        expected_obs = truth[i + 1]
        expected = _canonical(
            state_from_obs(expected_obs, opponent_private=_seat1_private(env, i + 1),
                           step=i + 1),
            0,
        )
        got = _canonical(state, 0)
        compared += 1
        if got != expected:
            for key in expected:
                if got[key] != expected[key]:
                    mismatches.append(f"step {i + 1}: field {key} diverged")
                    break
            if len(mismatches) >= 5:
                break
    return compared, len(mismatches), mismatches


def _seat1_private(env: Any, step_index: int) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(dict(env.steps[step_index][1]["observation"]["private"])))
    except Exception:
        return {}


def throughput() -> None:
    env = make_env(seed=SEED)
    agent = build_agent("baseline")
    captured: dict[str, Any] = {}

    def probe(obs: Any, config: Any = None) -> dict[str, Any]:
        if obs["step"] == 120:
            captured["obs"] = json.loads(json.dumps(dict(obs)))
        return agent(obs, config)

    with _quiet():
        env.run([probe, "starter"])
    if hasattr(agent, "close"):
        agent.close()

    obs = captured["obs"]
    shim = EnvShim(config=dict(env.configuration), seed=SEED)
    base = state_from_obs(obs, step=120)
    pass_action = safe_action(0)

    # -- clone -------------------------------------------------------------
    n = 3000
    t0 = time.perf_counter()
    for _ in range(n):
        clone_state(base)
    clone_us = (time.perf_counter() - t0) / n * 1e6

    import copy

    t0 = time.perf_counter()
    for _ in range(200):
        copy.deepcopy(base)
    deep_us = (time.perf_counter() - t0) / 200 * 1e6

    # -- step --------------------------------------------------------------
    n = 2000
    st = clone_state(base)
    t0 = time.perf_counter()
    for _ in range(n):
        simulate_step(st, pass_action, pass_action, shim, seat=0, in_place=True)
    step_us = (time.perf_counter() - t0) / n * 1e6

    # -- step under a real policy -----------------------------------------
    policy = build_agent("baseline")

    def pol(v: View) -> Action:
        return policy(v.obs)

    n = 400
    st = clone_state(base)
    t0 = time.perf_counter()
    for _ in range(n):
        a = pol(view_of(st, 0))
        b = pol(view_of(st, 1))
        simulate_step(st, a, b, shim, seat=0, in_place=True)
    policy_us = (time.perf_counter() - t0) / n * 1e6

    budget_ms = 700.0
    print("\n  simulator throughput")
    print("  " + "-" * 58)
    print(f"  clone_state            {clone_us:>9.1f} us   ({1e6 / clone_us:>9,.0f}/s)")
    print(f"  copy.deepcopy          {deep_us:>9.1f} us   ({1e6 / deep_us:>9,.0f}/s)"
          f"   -> {deep_us / clone_us:.0f}x slower")
    print(f"  simulate_step (PASS)   {step_us:>9.1f} us   ({1e6 / step_us:>9,.0f}/s)")
    print(f"  step + baseline policy {policy_us:>9.1f} us   ({1e6 / policy_us:>9,.0f}/s)")
    print()
    print(f"  within a {budget_ms:.0f} ms decision budget:")
    print(f"    raw steps                    {budget_ms * 1000 / step_us:>10,.0f}")
    print(f"    policy-driven rollout steps  {budget_ms * 1000 / policy_us:>10,.0f}")
    print(f"    -> e.g. {budget_ms * 1000 / policy_us / 24:>4,.0f} rollouts of 24 turns "
          f"(one in-game day)")
    print()


def main() -> int:
    compared, bad, details = fidelity()
    print("\n  simulator fidelity vs the real environment")
    print("  " + "-" * 58)
    print(f"  steps compared         {compared}")
    print(f"  divergent steps        {bad}")
    if bad == 0:
        print("  result                 EXACT - the simulator reuses interpreter(),")
        print("                         so agreement is by construction, not tolerance")
    else:
        for d in details:
            print(f"    {d}")
    throughput()
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
