"""Head-to-head evaluation over many seeded episodes.

    python -m src.evaluate --agent mcts --opponent baseline --episodes 100

Farming and market outcomes are noisy, so a single episode says nothing. This
module always reports a distribution and a paired significance test, and refuses
to describe a difference as an improvement unless it survives that test.

Pairing matters here: both agents play the *same* seeds, so per-seed differences
cancel most of the shared randomness (weed spawns, shop unlock order) and the
test is far more sensitive than comparing two independent means.

**Which statistic decides.** `task.md` describes a game's score as the final
coin balance, and it is - but the competition's ladder does not rate on it.
Kaggle rates each episode by outcome alone ("the actual coin difference in a
match does not affect the rating change"), and the final leaderboard is a
Bradley-Terry tournament over those win/loss/tie results. So the verdict here is
driven by **win rate with a Wilson interval**, and the coin margin is reported
underneath it as effect size - useful for understanding *how* an agent wins, but
not what it is paid for. An earlier version of this module had that backwards.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Sequence

from .runner import EpisodeResult, run_episode

DEFAULT_EPISODES = 30


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def _mean(xs: Sequence[float]) -> float:
    return statistics.fmean(xs) if xs else 0.0


def _stdev(xs: Sequence[float]) -> float:
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def _student_t_critical(df: int) -> float:
    """Two-sided 95% critical value. Table for small df, normal limit beyond.

    Avoids a scipy dependency, which would be one more thing to justify for the
    offline submission environment.

    Off-table degrees of freedom round **down** to the nearest tabulated df,
    which yields a slightly larger critical value than the truth (df=99 gives
    t(60)=2.000 against a real t(99)=1.984). Erring wide is deliberate: the
    alternative - rounding up - narrows every confidence interval and makes the
    test more willing to call noise an improvement, which is exactly the failure
    `task.md` warns about.
    """
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086,
        25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980,
    }
    if df <= 0:
        return float("inf")
    if df in table:
        return table[df]
    critical = 1.960  # normal limit, used once df runs past the table
    for k in sorted(table):
        if k <= df:
            critical = table[k]
    return critical


Z_95: Final[float] = 1.959963984540054
#: Chi-square critical value, 1 df, alpha = 0.05. Kept for reference; the
#: paired test below uses an exact binomial instead, which needs no table.
CHI2_1DF_95: Final[float] = 3.841


def _binomial_two_sided_p(k: int, n: int) -> float:
    """Exact two-sided p-value for `k` successes in `n` trials under p = 0.5.

    Exact rather than a normal or chi-square approximation because the counts
    that matter here are small: a paired comparison between two strong agents
    turns on the handful of seeds where they disagree, often fewer than twenty.
    """
    if n <= 0:
        return 1.0
    tail = min(k, n - k)
    cumulative = sum(math.comb(n, i) for i in range(tail + 1))
    return min(1.0, 2.0 * cumulative / (2.0 ** n))


@dataclass(slots=True)
class WinRateResult:
    """Win / tie / loss record with a Wilson interval on win probability.

    This is the statistic the competition actually pays for. Kaggle's ladder
    rates an episode by its outcome alone - "the actual coin difference in a
    match does not affect the rating change" - and the final leaderboard is a
    Bradley-Terry tournament over those outcomes. An agent that wins more games
    by smaller margins outranks one that wins fewer by blowouts, which is the
    opposite of what a t-test on coin margin rewards.

    The interval is Wilson rather than the textbook Wald one because the rates
    in question sit at 85-95%, where Wald intervals are badly skewed and can
    even reach past 1.0. Ties count as half a win, matching the Bradley-Terry
    convention.
    """

    n: int
    wins: int
    ties: int
    losses: int
    rate: float
    ci_low: float
    ci_high: float
    significant: bool

    @classmethod
    def of(cls, margins: Sequence[float]) -> "WinRateResult":
        n = len(margins)
        wins = sum(1 for m in margins if m > 0)
        ties = sum(1 for m in margins if m == 0)
        losses = n - wins - ties
        if n == 0:
            return cls(0, 0, 0, 0, 0.0, 0.0, 1.0, False)

        p = (wins + 0.5 * ties) / n
        z2 = Z_95 * Z_95
        denom = 1.0 + z2 / n
        centre = (p + z2 / (2 * n)) / denom
        half = Z_95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
        # Clamped to bracket the estimate: at p = 0 or 1 the arithmetic lands a
        # few ulps the wrong side of the bound, which would report an interval
        # that excludes the value it is an interval for.
        low = min(p, max(0.0, centre - half))
        high = max(p, min(1.0, centre + half))
        return cls(n, wins, ties, losses, p, low, high, low > 0.5 or high < 0.5)

    @property
    def record(self) -> str:
        return f"{self.wins}W-{self.ties}T-{self.losses}L"


@dataclass(slots=True)
class McNemarResult:
    """Paired win/loss comparison of two agents over the same seeds.

    The right test when the outcome is binary and the seeds are shared. Seeds
    where both agents won, or both lost, carry no information about which is
    better; only the discordant pairs do. Reported as an exact sign test on
    those pairs.
    """

    n: int
    both: int
    neither: int
    a_only: int
    b_only: int
    p_value: float
    significant: bool

    @classmethod
    def of(cls, a_wins: Sequence[bool], b_wins: Sequence[bool]) -> "McNemarResult":
        pairs = list(zip(a_wins, b_wins))
        both = sum(1 for a, b in pairs if a and b)
        neither = sum(1 for a, b in pairs if not a and not b)
        a_only = sum(1 for a, b in pairs if a and not b)
        b_only = sum(1 for a, b in pairs if b and not a)
        discordant = a_only + b_only
        p = _binomial_two_sided_p(a_only, discordant)
        return cls(len(pairs), both, neither, a_only, b_only, p, p < 0.05)


@dataclass(slots=True)
class PairedComparison:
    """Paired t-test on per-seed margins."""

    n: int
    mean_diff: float
    stdev_diff: float
    ci_low: float
    ci_high: float
    t_stat: float
    significant: bool

    @classmethod
    def of(cls, diffs: Sequence[float]) -> "PairedComparison":
        n = len(diffs)
        if n < 2:
            return cls(n, _mean(diffs), 0.0, float("-inf"), float("inf"), 0.0, False)
        m = _mean(diffs)
        sd = _stdev(diffs)
        se = sd / math.sqrt(n) if sd > 0 else 0.0
        crit = _student_t_critical(n - 1)
        half = crit * se
        t = m / se if se > 0 else (math.inf if m > 0 else (-math.inf if m < 0 else 0.0))
        return cls(n, m, sd, m - half, m + half, t, (m - half) > 0 or (m + half) < 0)


@dataclass(slots=True)
class EvalReport:
    """Everything one evaluation run produced."""

    agent: str
    opponent: str
    episodes: int
    banks: list[float] = field(default_factory=list)
    opponent_banks: list[float] = field(default_factory=list)
    margins: list[float] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)
    played_seats: list[int] = field(default_factory=list)
    wins: int = 0
    ties: int = 0
    max_turn_time: float = 0.0
    turn_time_means: list[float] = field(default_factory=list)
    wall_time: float = 0.0
    failures: list[str] = field(default_factory=list)

    def seat_split(self) -> dict[int, WinRateResult]:
        """Win rate per seat, so a seat bias shows up instead of averaging out.

        Pooling the two seats hides exactly the asymmetry the swap was run to
        find - this agent scores 93% from seat 0 and 87% from seat 1, which is
        worth seeing rather than reporting a flat 90%.
        """
        out: dict[int, WinRateResult] = {}
        for seat in sorted(set(self.played_seats)):
            margins = [
                b - o
                for b, o, s in zip(self.banks, self.opponent_banks, self.played_seats)
                if s == seat
            ]
            out[seat] = WinRateResult.of(margins)
        return out

    @property
    def mean_bank(self) -> float:
        return _mean(self.banks)

    @property
    def stdev_bank(self) -> float:
        return _stdev(self.banks)

    @property
    def win_rate(self) -> float:
        """Strict wins as a fraction of episodes. See `outcome` for ties."""
        return self.wins / len(self.banks) if self.banks else 0.0

    @property
    def mean_turn_time(self) -> float:
        """Mean seconds per decision across every episode.

        Every episode has the same turn count, so averaging the per-episode
        means is the overall mean. This used to accumulate with `max()`, which
        reported the worst episode's mean under a label that said 'mean'.
        """
        return _mean(self.turn_time_means)

    @property
    def outcome(self) -> WinRateResult:
        """The ladder-relevant statistic: win/tie/loss with a Wilson interval."""
        return WinRateResult.of(self.margins)

    @property
    def win_flags(self) -> list[bool]:
        """Per-seed win booleans, for a paired McNemar test against another run."""
        return [m > 0 for m in self.margins]

    @property
    def comparison(self) -> PairedComparison:
        return PairedComparison.of(self.margins)

    def summary(self) -> str:
        c = self.comparison
        w = self.outcome
        lines = [
            "",
            f"  {self.agent}  vs  {self.opponent}      ({len(self.banks)} episodes)",
            "  " + "-" * 62,
            "  LADDER METRIC - Kaggle rates win/loss/tie only, margin is ignored",
            f"  {'record':<22}  {w.record:>15}",
            f"  {'win rate':<22}  {w.rate:>15.1%}",
            f"  {'95% CI (Wilson)':<22}  [{w.ci_low:>7.1%}, {w.ci_high:>7.1%}]",
        ]
        split = self.seat_split()
        if len(split) > 1:
            # Pooling hides seat asymmetry, which is the reason for playing both.
            for seat, sw in sorted(split.items()):
                lines.append(
                    f"  {'  as seat ' + str(seat):<22}  {sw.record:>15}  {sw.rate:>7.1%}"
                )
        if len(self.banks) < 2:
            lines.append("  verdict                 too few episodes to say anything")
        elif w.significant and w.rate > 0.5:
            lines.append("  verdict                 WINS  (CI excludes 50%)")
        elif w.significant and w.rate < 0.5:
            lines.append("  verdict                 LOSES (CI excludes 50%)")
        else:
            lines.append("  verdict                 NOT SIGNIFICANT - the CI includes 50%,")
            lines.append("                          so this is noise, not an improvement")
        lines += [
            "",
            "  EFFECT SIZE - how big the wins are, not what the ladder pays for",
            f"  {'own bank':<22}  mean {self.mean_bank:>12,.0f}   sd {self.stdev_bank:>11,.0f}",
            f"  {'opponent bank':<22}  mean {_mean(self.opponent_banks):>12,.0f}   "
            f"sd {_stdev(self.opponent_banks):>11,.0f}",
            f"  paired margin           mean {c.mean_diff:>12,.0f}   sd {c.stdev_diff:>11,.0f}",
            f"  95% CI                  [{c.ci_low:>12,.0f}, {c.ci_high:>12,.0f}]"
            + (f"   t = {c.t_stat:.2f}" if len(self.banks) >= 2 else ""),
        ]
        if self.banks:
            lo, hi = min(self.banks), max(self.banks)
            lines.append(f"  bank range              [{lo:,.0f} .. {hi:,.0f}]")
        if self.max_turn_time:
            flag = "  ** OVER 1s LIMIT **" if self.max_turn_time > 1.0 else ""
            lines.append(
                f"  turn time               mean {self.mean_turn_time * 1000:.1f} ms, "
                f"max {self.max_turn_time * 1000:.1f} ms{flag}"
            )
        lines.append(f"  wall time               {self.wall_time:.1f}s")
        if self.failures:
            lines.append(f"  FAILURES                {len(self.failures)}: {self.failures[:3]}")
        lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["mean_bank"] = self.mean_bank
        d["stdev_bank"] = self.stdev_bank
        d["win_rate"] = self.win_rate
        d["mean_turn_time"] = self.mean_turn_time
        d["comparison"] = asdict(self.comparison)
        d["outcome"] = asdict(self.outcome)
        return d


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------
def _play(spec: tuple[str, str, int, dict[str, Any], dict[str, Any], int]) -> dict[str, Any]:
    """Run one episode in a worker process. Returns a plain dict (picklable).

    `seat` is where the agent under test sits. Everything this returns is phrased
    from that agent's point of view regardless, so a swapped episode is directly
    comparable to an unswapped one - the caller never has to remember which way
    round a row was played.
    """
    agent_name, opponent_name, seed, agent_kwargs, opp_kwargs, seat = spec
    from .registry import build_agent

    try:
        a = build_agent(agent_name, **agent_kwargs)
        b = build_agent(opponent_name, **opp_kwargs)
        order = [a, b] if seat == 0 else [b, a]
        result: EpisodeResult = run_episode([*order], seed=seed, time_seat=seat)
        bank = result.bank if seat == 0 else result.opponent_bank
        opp = result.opponent_bank if seat == 0 else result.bank
        return {
            "ok": True,
            "seed": seed,
            "seat": seat,
            "bank": bank,
            "opponent_bank": opp,
            "max_turn_time": result.max_turn_time,
            "mean_turn_time": result.mean_turn_time,
            "statuses": result.statuses,
        }
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "seed": seed, "seat": seat,
                "error": f"{type(exc).__name__}: {exc}"}


def evaluate(
    agent: str,
    opponent: str,
    episodes: int = DEFAULT_EPISODES,
    seed_start: int = 0,
    workers: int | None = None,
    agent_kwargs: dict[str, Any] | None = None,
    opponent_kwargs: dict[str, Any] | None = None,
    progress: bool = True,
    seats: str = "a",
) -> EvalReport:
    """Play `episodes` seeded games and report the distribution.

    `seats` decides where the agent under test sits: "a" (seat 0, the historical
    default), "b" (seat 1), or "both" - each seed played twice, once from each
    side. Kaggle's ladder gives you both seats, so "both" is the honest headline;
    "a" stays the default only so stored results remain comparable.
    """
    if seats not in ("a", "b", "both"):
        raise ValueError(f"seats must be 'a', 'b' or 'both', got {seats!r}")
    seeds = list(range(seed_start, seed_start + episodes))
    wanted = (0, 1) if seats == "both" else ((0,) if seats == "a" else (1,))
    specs = [
        (agent, opponent, s, agent_kwargs or {}, opponent_kwargs or {}, seat)
        for s in seeds
        for seat in wanted
    ]
    report = EvalReport(agent=agent, opponent=opponent, episodes=len(specs))
    t0 = time.perf_counter()

    if workers is None:
        workers = max(1, min(len(seeds), (os.cpu_count() or 2) - 1))

    results: list[dict[str, Any]] = []
    if workers <= 1:
        for i, spec in enumerate(specs, 1):
            results.append(_play(spec))
            if progress:
                _tick(i, len(specs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_play, s): s[2] for s in specs}
            for i, fut in enumerate(as_completed(futures), 1):
                results.append(fut.result())
                if progress:
                    _tick(i, len(specs))
    if progress:
        sys.stderr.write("\n")

    results.sort(key=lambda r: (r["seed"], r.get("seat", 0)))
    for r in results:
        if not r.get("ok"):
            report.failures.append(f"seed {r['seed']}: {r.get('error')}")
            continue
        report.seeds.append(r["seed"])
        report.played_seats.append(r.get("seat", 0))
        report.banks.append(r["bank"])
        report.opponent_banks.append(r["opponent_bank"])
        report.margins.append(r["bank"] - r["opponent_bank"])
        report.wins += 1 if r["bank"] > r["opponent_bank"] else 0
        report.ties += 1 if r["bank"] == r["opponent_bank"] else 0
        report.max_turn_time = max(report.max_turn_time, r["max_turn_time"])
        report.turn_time_means.append(r["mean_turn_time"])

    report.wall_time = time.perf_counter() - t0
    return report


def _tick(done: int, total: int) -> None:
    sys.stderr.write(f"\r  running episodes: {done}/{total}")
    sys.stderr.flush()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    from .registry import AGENT_NAMES

    parser = argparse.ArgumentParser(
        description="Evaluate two kaggriculture agents head to head over seeded episodes."
    )
    parser.add_argument("--agent", default="baseline", help=f"one of {AGENT_NAMES}")
    parser.add_argument("--opponent", default="starter", help=f"one of {AGENT_NAMES}")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--agent-args", type=str, default=None,
                        help="JSON kwargs for the agent, e.g. {\"aggregation\":\"worst\"}")
    parser.add_argument("--opponent-args", type=str, default=None,
                        help="JSON kwargs for the opponent")
    parser.add_argument("--label", type=str, default=None,
                        help="name to show in the report instead of the agent key")
    parser.add_argument("--json", type=str, default=None, help="write the full report here")
    parser.add_argument("--seats", choices=("a", "b", "both"), default="a",
                        help="which seat the agent under test plays. 'both' runs "
                             "every seed from each side, which is what the ladder "
                             "actually does; 'a' is the default only so older "
                             "results stay comparable")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    report = evaluate(
        agent=args.agent,
        opponent=args.opponent,
        episodes=args.episodes,
        seed_start=args.seed_start,
        workers=args.workers,
        agent_kwargs=json.loads(args.agent_args) if args.agent_args else None,
        opponent_kwargs=json.loads(args.opponent_args) if args.opponent_args else None,
        progress=not args.quiet,
        seats=args.seats,
    )
    if args.label:
        report.agent = args.label
    print(report.summary())
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"  report written to {args.json}\n")
    return 0 if not report.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
