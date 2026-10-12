"""Phase 5: determinization over the hidden opponent.

## What is actually uncertain

It is worth being precise, because it changes the design. The kaggriculture
environment is **fully deterministic** given both players' actions: weeds and
shop unlocks come from `Random(seed * 1_000_003 ^ day)` and everything else is a
pure function of state. So the market is not a stochastic process we need to
integrate over. The only thing we cannot see is:

* the opponent's **policy** - what they will do next, and
* the opponent's **private state** - shed, seeds and unit inventories, which
  bound how much they can dump on the shared market.

Both are opponent uncertainty. That makes this a classic imperfect-information
problem and determinization the right tool: sample several complete opponents,
solve each resulting perfect-information game, and combine.

Because each determinization is deterministic, one rollout per (candidate,
opponent) pair is *exact* - repeated rollouts would return identical numbers.
The budget therefore buys breadth (more opponents, more candidates) rather than
repetition, which is why this is a flat evaluation rather than a UCT tree.

## Aggregation

`aggregation="mean"` picks the strategy with the best average across sampled
opponents; `"worst"` picks the best worst case, which is the risk-averse choice
`task.md` asks for. Ladder scoring is your own final bank rather than a margin,
so mean is the theoretically better fit - but a strategy that only wins against
passive opponents is a real failure mode, and `tools/compare_agents.py` reports
both rather than assuming.

Opponent samples currently come from the built-in agents plus seeded
perturbations of our own baseline. `replay_loader.build_replay_models()` is
wired in and takes precedence automatically once real replays are downloaded -
see `docs/REPLAYS.md`.
"""

from __future__ import annotations

import random
from typing import Any, Sequence

from .agents.baseline_agent import BaselineAgent, DEFAULT_PARAMS, Params
from .agents.noisy_baseline import NoisyBaselineAgent
from .env_adapter import Action
from .mcts_agent import MCTSAgent, OpponentModel

AGGREGATIONS = ("mean", "worst", "best")


# ==========================================================================
# Opponent pool
# ==========================================================================
def _builtin_policy(name: str):
    """Wrap one of the environment's own agents as a plain callable."""
    from .simulator import _load_env_module

    kg = _load_env_module()
    if not kg:
        return None
    fn = {"pass": kg.pass_agent, "random": kg.random_agent, "starter": kg.starter_agent}[name]

    def policy(obs: Any, config: Any = None) -> Action:
        try:
            return fn(obs)
        except Exception:
            return {"farmer": ["PASS"], "hands": [], "market": []}

    return policy


def build_opponent_pool(
    n: int = 3,
    seed: int = 0,
    use_replays: bool = True,
) -> list[OpponentModel]:
    """Sample `n` plausible opponents, preferring real replays when present.

    The pool deliberately spans a range of market aggression rather than several
    strong-but-similar agents: what we need to be robust to is how much supply
    the opponent dumps, because that is the only channel through which their
    play reaches our score.
    """
    models: list[OpponentModel] = []

    if use_replays:
        try:
            from .replay_loader import build_replay_models, replays_available

            if replays_available():
                for name, policy in build_replay_models(limit=n):
                    models.append(OpponentModel(name, policy, None, 1.0))
        except Exception:
            pass

    if len(models) < n:
        rng = random.Random(seed)
        # A mirror of ourselves is the most informative single hypothesis: the
        # ladder pairs submissions, and strong opponents move the market most.
        # Cheap policies here on purpose: the opponent seat is half of every
        # rollout's cost, and their shed is unobservable anyway, so precision
        # buys nothing. Breadth of *behaviour* is what matters.
        from .agents.fast_policy import FastParams, FastPolicy

        mirror = FastPolicy(FastParams(crop="WHEAT", target_hands=10))
        models.append(OpponentModel("mirror-fast", lambda o, c=None: mirror(o), None, 1.0))

        while len(models) < n:
            i = len(models)
            if i == 1:
                # A markedly more aggressive seller - the case that hurts us most,
                # because their supply is what drives our prices down.
                aggressive = FastPolicy(FastParams(
                    crop="MELON", target_hands=12, sell_price_frac=0.55,
                    sell_max_drop_frac=0.70))
                models.append(OpponentModel(
                    "aggressive", lambda o, c=None, a=aggressive: a(o), None, 1.0))
            elif i == 2:
                starter = _builtin_policy("starter")
                if starter is not None:
                    models.append(OpponentModel("starter", starter, None, 1.0))
                else:  # pragma: no cover
                    noisy = NoisyBaselineAgent(seed=rng.randrange(1 << 30))
                    models.append(OpponentModel(f"noisy{i}", lambda o, c=None, a=noisy: a(o), None, 1.0))
            else:
                noisy = NoisyBaselineAgent(seed=rng.randrange(1 << 30))
                models.append(OpponentModel(f"noisy{i}", lambda o, c=None, a=noisy: a(o), None, 1.0))

    return models[:n]


# ==========================================================================
# Agent
# ==========================================================================
class DeterminizedAgent(MCTSAgent):
    """`MCTSAgent` scored across several sampled opponents instead of one.

    Everything structural is inherited; determinization is entirely a question of
    which opponents the rollouts face and how their results are combined.
    """

    def __init__(
        self,
        n_determinizations: int = 2,
        aggregation: str = "mean",
        budget_ms: float = 450.0,
        rollout_days: int = 2,
        max_candidates: int = 2,
        seed: int = 0,
        params: Params | None = None,
        use_replays: bool = True,
        **kwargs: Any,
    ) -> None:
        if aggregation not in AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {aggregation!r}")
        pool = build_opponent_pool(n=n_determinizations, seed=seed, use_replays=use_replays)
        super().__init__(
            budget_ms=budget_ms,
            rollout_days=rollout_days,
            max_candidates=max_candidates,
            params=params or DEFAULT_PARAMS,
            opponent_models=pool,
            aggregation=aggregation,
            seed=seed,
            **kwargs,
        )
        self.n_determinizations = n_determinizations

    def describe(self) -> str:
        names = ", ".join(m.name for m in (self._models or []))
        return (f"DeterminizedAgent(k={self.n_determinizations}, "
                f"aggregation={self.aggregation!r}, opponents=[{names}])")
