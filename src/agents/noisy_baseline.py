"""A baseline that plays slightly differently every game.

Two uses, both about not fooling ourselves:

* **Opponent diversity.** Evaluating only against the deterministic baseline
  invites overfitting to one opponent's exact habits. A perturbed variant is a
  cheap stand-in for "someone who plays roughly like this but not identically".
* **Determinization samples.** Phase 5 needs a spread of plausible opponents.
  Until real replay data is available (see `docs/REPLAYS.md`), sampling
  perturbed baselines is the honest substitute - it captures uncertainty about
  *how aggressively* an opponent farms and sells, which is what actually moves
  the shared market.

Perturbation is seeded, so a given seed reproduces exactly.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Any

from ..env_adapter import Action
from ..game_constants import CROPS
from .baseline_agent import DEFAULT_PARAMS, BaselineAgent, Params


def perturb(params: Params, rng: random.Random) -> Params:
    """Jitter the strategic knobs that most change observable market behaviour."""
    return replace(
        params,
        MAX_HANDS=max(3, min(13, params.MAX_HANDS + rng.randint(-4, 2))),
        SELL_PRICE_FRAC=round(
            min(1.15, max(0.60, params.SELL_PRICE_FRAC + rng.uniform(-0.22, 0.18))), 3
        ),
        SELL_MARGINAL_FLOOR_FRAC=round(
            min(0.95, max(0.45, params.SELL_MARGINAL_FLOOR_FRAC + rng.uniform(-0.15, 0.15))), 3
        ),
        HOLD_EARLY_DAYS=max(0, min(8, params.HOLD_EARLY_DAYS + rng.randint(-2, 4))),
        ENABLE_ANIMALS=rng.random() < 0.75,
        PLANT_CAPACITY_SLACK=round(
            min(1.10, max(0.55, params.PLANT_CAPACITY_SLACK + rng.uniform(-0.2, 0.15))), 3
        ),
        FORCE_CROP=rng.choice([None, None, *CROPS]),
        LAND_CASH_RESERVE=max(100.0, params.LAND_CASH_RESERVE + rng.uniform(-250, 600)),
    )


class NoisyBaselineAgent:
    """`BaselineAgent` with seeded parameter jitter, fixed at construction.

    The jitter is drawn once from `seed`, so a single instance plays one fixed
    perturbation for its whole life - it is not re-rolled per episode. That is
    what `determinization.py` wants (it builds a fresh one per opponent model
    with `rng.randrange(1 << 30)`, giving a genuinely varied pool), but it means
    `build_agent("baseline_noisy")` with no seed is the *same* opponent every
    time. Worth knowing before using it as an evaluation opponent and expecting
    diversity.
    """

    def __init__(self, seed: int = 0, params: Params | None = None) -> None:
        rng = random.Random(seed)
        self.params = perturb(params or DEFAULT_PARAMS, rng)
        self._agent = BaselineAgent(self.params)

    def __call__(self, obs: Any, config: Any = None) -> Action:
        return self._agent(obs, config)

    def close(self) -> None:
        self._agent.close()
