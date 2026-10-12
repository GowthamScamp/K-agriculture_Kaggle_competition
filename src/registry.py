"""Name -> agent factory. One place that knows how to build every agent."""

from __future__ import annotations

from typing import Any, Callable

Agent = Callable[..., dict[str, Any]]

BUILTIN = {"pass", "random", "starter"}

AGENT_NAMES = [
    "do_nothing",
    "baseline",
    "fast",
    "baseline_noisy",
    "reactive_route",
    "competitive_meta",
    "mcts",
    "mcts_det",
    "pass",
    "random",
    "starter",
]


def build_agent(name: str, **kwargs: Any) -> Agent | str:
    """Build an agent by name. Built-in names pass straight through to the env."""
    if name in BUILTIN:
        return name

    if name == "do_nothing":
        from .agents.do_nothing import do_nothing_agent

        return do_nothing_agent

    if name == "baseline":
        from .agents.baseline_agent import BaselineAgent

        return BaselineAgent(**kwargs)

    if name == "fast":
        from .agents.fast_policy import FastParams, FastPolicy

        return FastPolicy(FastParams(**kwargs)) if kwargs else FastPolicy()

    if name == "baseline_noisy":
        from .agents.noisy_baseline import NoisyBaselineAgent

        return NoisyBaselineAgent(**kwargs)

    if name == "reactive_route":
        from .agents.reactive_route import ReactiveRouteAgent

        return ReactiveRouteAgent(**kwargs)

    if name == "competitive_meta":
        if kwargs:
            raise ValueError("competitive_meta has no runtime-tuned parameters")
        from .agents.competitive_meta import agent as competitive_meta_agent

        return competitive_meta_agent

    if name == "mcts":
        from .mcts_agent import MCTSAgent

        return MCTSAgent(**kwargs)

    if name == "mcts_det":
        from .determinization import DeterminizedAgent

        return DeterminizedAgent(**kwargs)

    raise ValueError(f"unknown agent {name!r}; expected one of {AGENT_NAMES}")
