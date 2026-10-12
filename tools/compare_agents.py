"""Run the agent comparison matrix and report it honestly.

    python -m tools.compare_agents --episodes 100 --workers 8

Every row plays the *same* seeds against the same opponent, so the results are
directly comparable and every pairwise test below is a paired one.

**What decides a row.** Kaggle's ladder rates an episode by its outcome alone -
the coin difference does not affect the rating change - and the final
leaderboard is a Bradley-Terry tournament over those outcomes. So rows are
ranked by **win rate**, tested against a coin flip with a Wilson interval, and
compared against each other with an exact McNemar sign test on the seeds where
they disagree. Mean bank is still printed, as effect size, because it explains
*how* a row wins - but it is not what the competition pays for.

This is what answers the Phase 5 question `task.md` asks: whether average-case
or worst-case (risk-averse) aggregation actually scores better. Note that the
answer can differ by metric, and the ladder's metric is the one that counts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.evaluate import McNemarResult, evaluate

ROOT = Path(__file__).resolve().parents[1]

MATRIX: list[tuple[str, str, dict[str, Any]]] = [
    ("baseline",      "baseline", {}),
    ("mcts",          "mcts",     {}),
    ("det-mean",      "mcts_det", {"aggregation": "mean", "n_determinizations": 2}),
    ("det-worst",     "mcts_det", {"aggregation": "worst", "n_determinizations": 2}),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--opponent", default="baseline")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel episodes. The search budget is wall-clock, so "
                         "over-subscribing the box starves each decision of rollouts "
                         "and understates every search agent. Default is cpu_count-1.")
    ap.add_argument("--json", default=str(ROOT / "logs" / "comparison.json"))
    args = ap.parse_args()

    print()
    print(f"  Each row plays {args.episodes} identical seeds against '{args.opponent}'.")
    print(f"  Workers: {args.workers if args.workers else 'auto (cpu_count-1)'}.")
    print("  Ranked by win rate - the ladder ignores coin margin.")
    print()
    header = (f"  {'agent':<12} {'record':>10} {'win%':>6} {'95% CI (Wilson)':>18} "
              f"{'mean bank':>11} {'margin':>10} {'max turn':>9}  verdict")
    print(header)
    print("  " + "-" * (len(header) + 4))

    rows: list[dict[str, Any]] = []
    for label, key, kwargs in MATRIX:
        report = evaluate(
            agent=key, opponent=args.opponent, episodes=args.episodes,
            seed_start=args.seed_start, agent_kwargs=kwargs, progress=False,
            workers=args.workers,
        )
        c = report.comparison
        w = report.outcome
        if len(report.banks) < 2:
            verdict = "too few"
        elif not w.significant:
            verdict = "noise"
        elif w.rate > 0.5:
            verdict = "WINS"
        else:
            verdict = "LOSES"
        over = "!" if report.max_turn_time > 1.0 else " "
        print(f"  {label:<12} {w.record:>10} {w.rate:>5.0%} "
              f"[{w.ci_low:>7.1%},{w.ci_high:>7.1%}] "
              f"{report.mean_bank:>11,.0f} {c.mean_diff:>10,.0f} "
              f"{report.max_turn_time * 1000:>8.0f}ms{over} {verdict}")
        rows.append({
            "label": label, "agent": key, "kwargs": kwargs,
            "episodes": len(report.banks), "seeds": report.seeds,
            "wins": report.wins, "ties": report.ties,
            "win_rate": w.rate, "win_ci": [w.ci_low, w.ci_high],
            "win_significant": w.significant,
            "mean_bank": report.mean_bank, "stdev_bank": report.stdev_bank,
            "margin": c.mean_diff, "ci": [c.ci_low, c.ci_high],
            "significant": c.significant,
            "max_turn_time": report.max_turn_time,
            "mean_turn_time": report.mean_turn_time,
            "banks": report.banks, "opponent_banks": report.opponent_banks,
            "margins": report.margins,
        })

    print()
    ranked = sorted(rows, key=lambda r: -r["win_rate"])
    best = ranked[0]
    print(f"  Best win rate:  {best['label']} at {best['win_rate']:.0%}")
    by_bank = max(rows, key=lambda r: r["mean_bank"])
    print(f"  Best mean bank: {by_bank['label']} at {by_bank['mean_bank']:,.0f}"
          + ("" if by_bank["label"] == best["label"]
             else "   <- differs from the ladder metric"))
    over = [r["label"] for r in rows if r["max_turn_time"] > 1.0]
    if over:
        joined = ", ".join(over)
        print(f"  ** OVER THE 1s TURN LIMIT: {joined} - not submittable **")

    # -- pairwise, on the metric the ladder uses ----------------------------
    print()
    print("  Pairwise McNemar (paired on seeds; only disagreements carry signal)")
    print(f"  {'comparison':<26} {'A only':>7} {'B only':>7} {'p':>8}  verdict")
    print("  " + "-" * 62)
    comparisons: list[dict[str, Any]] = []
    for i, a in enumerate(ranked):
        for b in ranked[i + 1:]:
            m = McNemarResult.of([x > 0 for x in a["margins"]],
                                 [x > 0 for x in b["margins"]])
            verdict = ("no difference" if not m.significant
                       else (f"{a['label']} better" if m.a_only > m.b_only
                             else f"{b['label']} better"))
            name = f"{a['label']} vs {b['label']}"
            print(f"  {name:<26} {m.a_only:>7} {m.b_only:>7} {m.p_value:>8.3f}  {verdict}")
            comparisons.append({
                "a": a["label"], "b": b["label"], "a_only": m.a_only,
                "b_only": m.b_only, "both": m.both, "neither": m.neither,
                "p_value": m.p_value, "significant": m.significant,
            })

    mean_row = next((r for r in rows if r["label"] == "det-mean"), None)
    worst_row = next((r for r in rows if r["label"] == "det-worst"), None)
    if mean_row and worst_row:
        print()
        print(f"  Aggregation: mean {mean_row['win_rate']:.0%} win vs "
              f"worst-case {worst_row['win_rate']:.0%} win "
              f"(bank {mean_row['mean_bank']:,.0f} vs {worst_row['mean_bank']:,.0f})")
        pair = next((c for c in comparisons
                     if {c["a"], c["b"]} == {"det-mean", "det-worst"}), None)
        if pair and not pair["significant"]:
            print("  The two aggregation rules are statistically indistinguishable at")
            print(f"  this episode count (McNemar p = {pair['p_value']:.3f}); report it")
            print("  as unresolved rather than picking the larger number.")
    print()

    payload = {"rows": rows, "comparisons": comparisons,
               "opponent": args.opponent, "workers": args.workers}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  full results written to {args.json}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
