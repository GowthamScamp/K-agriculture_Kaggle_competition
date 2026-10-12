"""Phase 4: search-based agent over macro-actions.

## Why macro-actions, and why not a deep UCT tree

`task.md` sketches the search as "for each candidate action, clone the state,
apply it, roll out, keep the best". Two measured facts from Phases 1-3 decide how
that gets built here:

1. **The raw action space cannot be enumerated.** One turn is a farmer op *plus*
   an op for every hired hand *plus* up to ten market orders. With a dozen hands
   the joint space is astronomically large, and almost all of it is noise -
   whether hand #7 walks north or south this turn is not a strategic decision.
   So the search ranges over **macro-actions**: whole-day strategies (which crop,
   how many hands, whether to expand, which animal, how hard to sell), each
   executed by the Phase 2 agent. `BaselineAgent` is both the rollout policy the
   task asks for *and* the executor, so the thing being searched over is exactly
   the thing that will be played.

2. **The simulator is deterministic.** Weed spawns and shop unlocks are drawn
   from `Random(seed * 1_000_003 ^ day)`, and every other rule is a pure
   function. Given a fixed opponent, a rollout returns the *same* number every
   time. Averaging repeated identical rollouts would buy nothing, so the budget
   goes into breadth - more candidates and more opponent determinizations -
   rather than depth of repetition. This is why the search is a one-ply
   evaluation over macro-actions rather than a UCT tree: with no stochasticity to
   integrate over, extra visits to the same node are pure waste.

The remaining uncertainty is entirely the *opponent*, which is what
`determinization.py` handles.

## Cost, and why the budget is deliberately slack

A `BaselineAgent` decision costs ~450 us and a simulated turn ~45 us, so one
two-day rollout driving both seats runs ~28 ms. The shipped configuration scores
`max_candidates=2` against one opponent model, so a full sweep costs ~56 ms
inside a 450 ms budget - roughly **8x headroom**. Searching once per in-game day
gives 30 decisions per episode.

That slack is the point, not waste. The budget used to double as a throttle: the
candidate list was truncated to whatever the measured rollout cost said was
affordable, which made the agent's *strategy* a function of machine load. It is
now sized so the sweep completes even on hardware several times slower than the
box it was tuned on, and the deadline is a pure `actTimeout` safety net that
should never fire.

## How much search, honestly

More is not better here, and the effect is not subtle. Over 100 identical seeds
against `baseline`:

    max_candidates:   1      2      3      4      5      6
    win rate:       52%    93%    89%    77%    30%     1%

`1` is the bare heuristic - the search has only "auto" to pick. Past `3` the
evaluator starts choosing actively harmful strategies, because `_value` scores a
2-day rollout in a 30-day season and counts unsold inventory as wealth, so
hoarding always looks profitable inside the window and is worth zero at the end.
`6` adds `sell-patient`, which raises the selling threshold, and loses 99 of 100.
Fixing the evaluator is the open work; capping the candidate list is the
measured mitigation. See docs/NOTES.md, 30 Aug 2026.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, fields, replace
from typing import Any, Callable, Sequence

from .agents.baseline_agent import BaselineAgent, DEFAULT_PARAMS, Params
from .env_adapter import Action, View, safe_action
from .game_constants import ANIMALS, CROPS
from .simulator import (
    EnvShim,
    bank_of,
    clone_state,
    simulate_step,
    simulator_available,
    state_from_obs,
)

TURNS_PER_DAY = 24
SEASON_DAYS = 30


# ==========================================================================
# Macro-actions
# ==========================================================================
@dataclass(frozen=True, slots=True)
class MacroAction:
    """A whole-day strategy, expressed as an override of the baseline's params."""

    label: str
    params: Params

    def __str__(self) -> str:
        return self.label


def generate_candidates(v: View, base: Params, max_candidates: int = 8) -> list[MacroAction]:
    """Propose plausible strategies for the coming day.

    Deliberately small and state-dependent: options that cannot pay off before
    the season ends, or that we cannot afford, are never generated, so the whole
    budget is spent on choices that are actually live.
    """
    days_left = SEASON_DAYS - v.day
    out: list[MacroAction] = [MacroAction("auto", base)]

    # -- crop choices that can still finish -------------------------------
    viable = [
        c for c in CROPS
        if _cycle_days(c) <= days_left - 1 and CROPS[c]["seed"] <= v.money
    ]
    for crop in viable:
        out.append(MacroAction(f"crop={crop.lower()}", replace(base, FORCE_CROP=crop)))

    # -- labour: the single most valuable lever, so probe both directions --
    g = v.groups()
    demand = len(g.plants) + 1.4 * len(g.animals) + 0.5 * len(g.empties)
    natural = max(3, min(base.MAX_HANDS, int(demand / base.WORK_PER_UNIT_PER_DAY)))
    for delta in (-3, 3):
        target = natural + delta
        if 0 <= target <= base.MAX_HANDS:
            out.append(MacroAction(f"hands={target}", replace(base, HAND_TARGET=target)))

    # -- animals, while they still have time to produce -------------------
    if days_left > 8:
        for animal in ("COW", "SHEEP", "GOOSE"):
            if ANIMALS[animal]["cost"] <= v.money:
                out.append(MacroAction(
                    f"animal={animal.lower()}",
                    replace(base, ENABLE_ANIMALS=True, FORCE_ANIMAL=animal),
                ))
    out.append(MacroAction("no-animals", replace(base, ENABLE_ANIMALS=False)))

    # -- market posture ----------------------------------------------------
    out.append(MacroAction("sell-hard", replace(
        base, SELL_PRICE_FRAC=0.70, SELL_MAX_DROP_FRAC=0.78)))
    out.append(MacroAction("sell-patient", replace(
        base, SELL_PRICE_FRAC=1.05, SELL_MAX_DROP_FRAC=0.94)))

    # -- expansion ---------------------------------------------------------
    if len(v.unlocked_quadrants) < 4 and v.day <= base.LAND_LAST_USEFUL_DAY:
        out.append(MacroAction("expand", replace(base, LAND_CASH_RESERVE=100.0)))

    # Keep "auto" first: it is the fallback if the budget runs out immediately.
    seen: set[Any] = set()
    unique: list[MacroAction] = []
    for m in out:
        key = m.params
        if key in seen:
            continue
        seen.add(key)
        unique.append(m)
    return unique[:max_candidates]


def _cycle_days(crop: str) -> int:
    d = CROPS[crop]
    if d["ongoing"]:
        return d["first_yield_day"] + d["interval"] * (d["max_yield"] - 1) + 1
    return d["max_yield_day"] + 1


# ==========================================================================
# Opponent models
# ==========================================================================
@dataclass(slots=True)
class OpponentModel:
    """One hypothesis about how the opponent will play, plus its hidden state.

    `private` is the opponent's shed/seeds/inventories, which are never
    observable. Getting it wrong only mis-predicts how much they can dump on the
    market, so the search stays useful under a bad guess - it just misprices.
    """

    name: str
    policy: Callable[[Any], Action]
    private: dict[str, Any] | None = None
    weight: float = 1.0


def default_opponent_models() -> list[OpponentModel]:
    """A single opponent hypothesis: someone farming and selling competently.

    Driven by `FastPolicy` rather than a full `BaselineAgent`. The opponent seat
    is half of every rollout's cost, and modelling them precisely is pointless
    anyway - we cannot see their shed, so the prediction is approximate no matter
    how good the policy is. Spending that budget on more candidates instead is
    the better trade, and it roughly doubles how many sweeps finish.
    """
    from .agents.fast_policy import FastParams, FastPolicy

    mirror = FastPolicy(FastParams(crop="WHEAT", target_hands=10))
    return [OpponentModel("mirror-fast", lambda obs: mirror(obs), None, 1.0)]


# ==========================================================================
# Search
# ==========================================================================
@dataclass(slots=True)
class SearchResult:
    chosen: MacroAction
    scores: dict[str, float]
    rollouts: int
    elapsed: float
    timed_out: bool
    sweeps: int = 0
    n_candidates: int = 0


class MCTSAgent:
    """Macro-action search on top of the exact simulator, executed by the baseline.

    Anytime and unconditionally safe: a legal fallback action is computed before
    any searching happens, the wall clock is checked before every rollout, and
    the whole decision is wrapped so that any failure degrades to plain baseline
    play rather than forfeiting the episode.
    """

    def __init__(
        self,
        budget_ms: float = 450.0,
        rollout_days: int = 2,
        # 2, chosen on measurement rather than intuition. Over 100 identical
        # seeds this parameter spans the whole quality range: 1 is the bare
        # heuristic (52%), 2 is 93%, 3 is 89%, 4 is 77%, 5 is 30% and 6 is 1%.
        # More search is emphatically not better here - see docs/NOTES.md.
        max_candidates: int = 2,
        decide_every_hours: int = 24,
        params: Params | None = None,
        params_kwargs: dict[str, Any] | None = None,
        opponent_models: Sequence[OpponentModel] | None = None,
        aggregation: str = "mean",
        shaped_reward: bool = True,
        objective: str = "absolute",
        valuation: str = "market",
        seed: int = 0,
        verbose: bool = False,
    ) -> None:
        self.budget_s = budget_ms / 1000.0
        self.rollout_days = rollout_days
        self.max_candidates = max_candidates
        self.decide_every_hours = max(1, decide_every_hours)
        base = params or DEFAULT_PARAMS
        # Lets a single heuristic threshold be overridden from `--agent-args`
        # without constructing a whole `Params` in JSON, so any one of them can
        # be A/B-tested from the command line the way `objective` and
        # `valuation` already can.
        if params_kwargs:
            unknown = set(params_kwargs) - {f.name for f in fields(Params)}
            if unknown:
                raise ValueError(f"unknown Params field(s): {sorted(unknown)}")
            base = replace(base, **params_kwargs)
        self.base_params = base
        self.aggregation = aggregation
        self.shaped_reward = shaped_reward
        if objective not in ("absolute", "relative"):
            raise ValueError(
                f"objective must be 'absolute' or 'relative', got {objective!r}"
            )
        self.objective = objective
        if valuation not in ("market", "liquidation"):
            raise ValueError(
                f"valuation must be 'market' or 'liquidation', got {valuation!r}"
            )
        self.valuation = valuation
        self.seed = seed
        self.verbose = verbose
        self._models = list(opponent_models) if opponent_models else None

        self._executor = BaselineAgent(self.base_params)
        self._committed: Params = self.base_params
        self._committed_label = "auto"
        self.history: list[SearchResult] = []
        self._available = simulator_available()
        # Rollout cost varies a lot over a season - a day-28 farm has ten times
        # the tiles and units of a day-1 one - so the cutoff uses a measured
        # running estimate rather than a fixed guess.
        self._rollout_cost: float | None = None

    # -- entry point -------------------------------------------------------
    def __call__(self, obs: Any, config: Any = None) -> Action:
        n_hands = 0
        try:
            farms = obs.get("farms") or []
            player = int(obs.get("player", 0))
            if player < len(farms):
                n_hands = len(farms[player].get("hands") or [])
        except Exception:
            pass

        try:
            v = View.from_obs(obs)
            if self._available and self._is_decision_point(v):
                self._search(v, obs, config)
            self._executor.p = self._committed
            return self._executor.decide(v)
        except Exception:
            # Last line of defence: never forfeit a turn.
            try:
                self._executor.p = self.base_params
                return self._executor(obs, config)
            except Exception:
                return safe_action(n_hands)

    def _is_decision_point(self, v: View) -> bool:
        if v.day >= SEASON_DAYS - 1:
            return False  # liquidation is not a strategic choice
        return v.hour % self.decide_every_hours == 0

    # -- the search --------------------------------------------------------
    def _search(self, v: View, obs: Any, config: Any) -> None:
        t0 = time.perf_counter()
        deadline = t0 + self.budget_s

        candidates = generate_candidates(v, self.base_params, self.max_candidates)
        models = self._models or default_opponent_models()

        # The candidate list is fixed by `max_candidates` and nothing else.
        #
        # It used to be truncated to whatever the measured rollout cost said the
        # wall-clock budget could afford:
        #
        #     affordable = int(self.budget_s / self._est_rollout_s(horizon0))
        #     candidates = candidates[:max(2, affordable)]
        #
        # which made the agent's *strategy* a function of how fast the machine
        # happened to be that second. It was not a small effect. Measured over
        # 100 identical seeds, the number of candidates scored decides the whole
        # result: 1 candidate is the plain heuristic (50.5%), 2 is the best agent
        # we have (86%), 5 is far worse than doing nothing (30%). So a busy box
        # truncated to 2 and looked excellent, an idle one scored all 5 and
        # looked broken, and every historical number in this repo was really a
        # measurement of machine load. See docs/NOTES.md, 30 Aug 2026.
        #
        # The deadline below is now a pure safety net against `actTimeout`. If it
        # ever binds, the partial sweep is discarded and we fall back to the
        # heuristic - a safe, and loudly recorded, degradation rather than a
        # silent change of strategy.
        horizon0 = self.rollout_days * TURNS_PER_DAY

        cfg = dict(config) if config else {}
        env = EnvShim(config=cfg, seed=self.seed)
        horizon = horizon0

        scores: dict[str, list[float]] = {c.label: [] for c in candidates}
        rollouts = 0
        sweeps = 0
        timed_out = False

        # One opponent model at a time, and a model's results are only kept if
        # *every* candidate was scored against it. A partial sweep would compare
        # some candidates against a tough opponent and others against a soft one,
        # which is worse than not searching at all - the ranking would reflect
        # which opponent a candidate happened to draw rather than its merit.
        for model in models:
            try:
                base_state = state_from_obs(obs, opponent_private=model.private)
            except Exception:
                continue
            sweep: dict[str, float] = {}
            for cand in candidates:
                if time.perf_counter() + self._est_rollout_s(horizon) > deadline:
                    timed_out = True
                    break
                t_roll = time.perf_counter()
                sweep[cand.label] = self._evaluate(
                    base_state, cand, model, env, horizon, v.player, deadline=deadline
                )
                self._observe_cost(time.perf_counter() - t_roll)
                rollouts += 1
            if len(sweep) == len(candidates):
                sweeps += 1
                for label, value in sweep.items():
                    scores[label].append(value)
            if timed_out:
                break

        best = self._pick(candidates, scores)
        self._committed = best.params
        self._committed_label = best.label
        result = SearchResult(
            chosen=best,
            scores={k: self._aggregate(vs) for k, vs in scores.items() if vs},
            rollouts=rollouts,
            elapsed=time.perf_counter() - t0,
            timed_out=timed_out,
            sweeps=sweeps,
            n_candidates=len(candidates),
        )
        self.history.append(result)
        if self.verbose:
            ranked = sorted(result.scores.items(), key=lambda kv: -kv[1])[:4]
            print(f"  day {v.day:>2} -> {best.label:<16} "
                  f"({rollouts} rollouts, {result.elapsed * 1000:.0f} ms)  "
                  + "  ".join(f"{k}={s:,.0f}" for k, s in ranked))

    def _est_rollout_s(self, horizon: int) -> float:
        """Per-rollout cost estimate used to stop *before* overrunning the budget.

        Seeded pessimistically and then replaced by measurement, with a safety
        factor. Overrunning the 1 s `actTimeout` costs the whole episode, so this
        errs high.
        """
        if self._rollout_cost is None:
            return horizon * 1.0e-3
        return self._rollout_cost * 1.25

    def _observe_cost(self, seconds: float) -> None:
        if self._rollout_cost is None:
            self._rollout_cost = seconds
        else:
            self._rollout_cost = 0.7 * self._rollout_cost + 0.3 * seconds

    def _evaluate(
        self,
        base_state: list[Any],
        cand: MacroAction,
        model: OpponentModel,
        env: EnvShim,
        horizon: int,
        seat: int,
        deadline: float | None = None,
    ) -> float:
        """One deterministic rollout, abortable mid-flight.

        Checking the clock only *between* rollouts was not enough: a single
        late-season rollout on a hundred-tile farm ran long enough to push a turn
        to 1,191 ms and breach the 1 s `actTimeout`, which forfeits the episode.
        The horizon is therefore also cut from inside the loop, and a truncated
        rollout is still scored - shorter, but on the same shaped value as every
        other candidate.
        """
        state = clone_state(base_state)
        me = BaselineAgent(cand.params)
        them = model.policy
        limit = int(env.configuration["episodeSteps"]) - state[0].observation.step - 1
        steps = max(0, min(horizon, limit))
        for i in range(steps):
            if deadline is not None and (i & 7) == 0 and time.perf_counter() >= deadline:
                break
            try:
                a = me(state[seat].observation.__dict__)
                b = them(state[1 - seat].observation.__dict__)
            except Exception:
                break
            simulate_step(state, a, b, env, seat=seat, in_place=True)
        return self._value(state, seat)

    def _value(self, state: list[Any], seat: int) -> float:
        """Score a rolled-out position.

        `absolute` - our own final bank, shaped with net worth so short rollouts
        can see investment. Raw cash over three days would score every seed, hand
        and animal as a pure loss and drive the agent to hoard, which is
        precisely the failure `task.md` warns about.

        `relative` - the same quantity minus the opponent's bank. The ladder pays
        only for *beating* the opponent and explicitly ignores the size of the
        win, so the gap is the thing worth maximising. It uses their **bank**
        rather than their net worth deliberately: their money is public and
        exact, whereas their shed is an assumption - and unsold inventory scores
        zero at season end anyway, so bank is both the knowable term and the
        scoring one.
        """
        if self.shaped_reward:
            from .agents.fast_policy import fast_net_worth

            mine = fast_net_worth(
                state[seat].observation.__dict__,
                liquidation=self.valuation == "liquidation",
            )
        else:
            mine = bank_of(state, seat)

        if self.objective == "relative":
            return mine - bank_of(state, 1 - seat)
        return mine

    def _aggregate(self, values: list[float]) -> float:
        if not values:
            return float("-inf")
        if self.aggregation == "worst":
            return min(values)
        if self.aggregation == "best":
            return max(values)
        return sum(values) / len(values)

    def _pick(
        self, candidates: list[MacroAction], scores: dict[str, list[float]]
    ) -> MacroAction:
        best = candidates[0]  # "auto" - the anytime fallback
        best_score = float("-inf")
        for cand in candidates:
            vals = scores.get(cand.label) or []
            if not vals:
                continue
            score = self._aggregate(vals)
            if score > best_score:
                best, best_score = cand, score
        return best

    def close(self) -> None:
        self._executor.close()
