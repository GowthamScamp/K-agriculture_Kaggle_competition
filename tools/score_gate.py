"""Resumable, paired, seed-clustered promotion gate over real trading agents.

Each game loads fresh modules. JSON checkpoints are atomic and keyed by the
source/engine/seed fingerprint. No replay file or account credential is read.
This is a local comparison, never a prediction of an absolute Kaggle rating.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
from pathlib import Path

from src.evaluate import WinRateResult
from src.runner import run_episode
from src.simulator import _load_env_module
from tools.competitive_pool import ROOT, _competitor


def fingerprint(specification: str) -> dict:
    if "=" not in specification:
        raise ValueError("Gate competitors must be explicit frozen LABEL=FILE specifications")
    label, location = specification.split("=", 1)
    head, separator, tail = location.rpartition(":")
    path = head if separator and tail.isidentifier() else location
    path = (ROOT / path).resolve()
    return {"label": label, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "symbol": tail if separator and tail.isidentifier() else "agent"}


def game(task: tuple[str, str, int, int]) -> dict:
    own_spec, rival_spec, seed, seat = task
    started = time.perf_counter()
    own = _competitor(own_spec).factory()
    rival = _competitor(rival_spec).factory()
    load_ms = (time.perf_counter() - started) * 1000
    agents = [own, rival] if seat == 0 else [rival, own]
    result = run_episode(agents, seed=seed, time_seat=seat)
    return {
        "agent": own_spec.split("=", 1)[0], "opponent": rival_spec.split("=", 1)[0],
        "seed": seed, "seat": seat, "bank": result.rewards[seat],
        "opponent_bank": result.rewards[1 - seat], "statuses": result.statuses,
        "steps": result.steps, "load_ms": load_ms,
        "max_turn_ms": result.max_turn_time * 1000, "wall_seconds": result.wall_time,
        "diagnostic_errors": result.diagnostic_errors,
        "overlay_activity": dict(getattr(own, "__globals__", {}).get("_OVERLAY_DIAGNOSTICS") or {}),
    }


def outcome_score(row: dict) -> float:
    margin = row["bank"] - row["opponent_bank"]
    return 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5


def paired_comparison(records: list[dict], candidate: str, control: str,
                      opponents: list[str], expected_seeds: list[int], samples: int = 10000) -> dict:
    """Compare identical triples; resample entire seeds across seats/opponents."""
    lookup = {(r["agent"], r["opponent"], r["seed"], r["seat"]): r for r in records}
    clusters = []
    by_opponent = {o: [] for o in opponents}
    for seed in expected_seeds:
        deltas = []
        complete = True
        for opponent in opponents:
            pair = []
            for seat in (0, 1):
                a = lookup.get((candidate, opponent, seed, seat))
                b = lookup.get((control, opponent, seed, seat))
                if a is None or b is None:
                    complete = False
                    break
                pair.append(outcome_score(a) - outcome_score(b))
            if len(pair) == 2:
                mean = statistics.fmean(pair)
                deltas.extend(pair)
                by_opponent[opponent].append(mean)
        if complete and deltas:
            clusters.append(statistics.fmean(deltas))
    full = len(clusters) == len(expected_seeds)
    mean = statistics.fmean(clusters) if clusters else 0.0
    rng = random.Random(20261002)
    draws = sorted(statistics.fmean(rng.choices(clusters, k=len(clusters)))
                   for _ in range(samples)) if clusters else [0.0]
    lo = draws[int(0.025 * (len(draws) - 1))]
    hi = draws[int(0.975 * (len(draws) - 1))]
    regressions = {o: statistics.fmean(xs) for o, xs in by_opponent.items() if xs}
    relevant = [r for r in records if r["agent"] in (candidate, control)]
    healthy = all(r["steps"] == 720 and all(s == "DONE" for s in r["statuses"])
                  and not any(v for counters in r["diagnostic_errors"] for v in counters.values())
                  for r in relevant)
    reasons = []
    if not full or len(clusters) < 96:
        reasons.append("requires all 96 held-out seeds, both seats, every frozen opponent")
    if lo <= 0:
        reasons.append("paired 95% lower confidence bound is not positive")
    if any(delta < -0.05 for delta in regressions.values()):
        reasons.append("an opponent regressed by more than five percentage points")
    if not healthy:
        reasons.append("an episode failed or reported an internal fallback/error")
    return {"candidate": candidate, "control": control, "seed_clusters": len(clusters),
            "complete": full, "mean_outcome_improvement": mean,
            "paired_cluster_bootstrap_95_ci": [lo, hi], "by_opponent_improvement": regressions,
            "economic_gate_passed": not reasons, "blockers": reasons,
            "serial_artifact_verification_still_required": True}


def summary(records: list[dict]) -> list[dict]:
    rows = []
    for pair in sorted({(r["agent"], r["opponent"]) for r in records}):
        games = [r for r in records if (r["agent"], r["opponent"]) == pair]
        margins = [r["bank"] - r["opponent_bank"] for r in games]
        outcome = WinRateResult.of(margins)
        rows.append({"agent": pair[0], "opponent": pair[1], "games": len(games),
                     "record": outcome.record, "outcome_score": outcome.rate,
                     "mean_bank": statistics.fmean(r["bank"] for r in games),
                     "mean_opponent_bank": statistics.fmean(r["opponent_bank"] for r in games),
                     "max_turn_ms": max(r["max_turn_ms"] for r in games)})
    return rows


def save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--opponent", action="append", required=True)
    parser.add_argument("--seed-start", type=int, default=5000)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.seeds <= 0 or args.workers <= 0:
        parser.error("seeds and workers must be positive")
    specs = [args.control, *args.candidate]
    labels = [s.split("=", 1)[0] for s in specs]
    if len(labels) != len(set(labels)):
        parser.error("candidate and control labels must be unique")
    opponent_labels = [s.split("=", 1)[0] for s in args.opponent]
    if len(opponent_labels) != len(set(opponent_labels)):
        parser.error("opponent labels must be unique")
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    identity = {"agents": [fingerprint(s) for s in specs],
                "opponents": [fingerprint(s) for s in args.opponent], "seeds": seeds,
                "engine_version": version("kaggle-environments"),
                "engine_sha256": hashlib.sha256(Path(_load_env_module().__file__).read_bytes()).hexdigest(),
                "runner_sha256": hashlib.sha256((ROOT / "src/runner.py").read_bytes()).hexdigest(),
                "gate_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    identity["diagnostics_sha256"] = hashlib.sha256((ROOT / "src/diagnostics.py").read_bytes()).hexdigest()
    identity["loader_sha256"] = hashlib.sha256((ROOT / "tools/competitive_pool.py").read_bytes()).hexdigest()
    if identity["engine_version"] != "1.32.7":
        parser.error("re-measure explicitly if the pinned engine version changes")
    records = []
    if args.out.exists():
        if not args.resume:
            parser.error("output already exists; use --resume rather than overwrite evidence")
        old = json.loads(args.out.read_bytes())
        if old["identity"] != identity:
            parser.error("sources, engine or seeds changed; choose a new output checkpoint")
        records = old["games"]
    complete = {(r["agent"], r["opponent"], r["seed"], r["seat"]) for r in records}
    tasks = [(a, o, seed, seat) for a in specs for o in args.opponent for seed in seeds for seat in (0, 1)
             if (a.split("=", 1)[0], o.split("=", 1)[0], seed, seat) not in complete]
    total = len(specs) * len(args.opponent) * len(seeds) * 2
    payload = {"identity": identity, "games": records, "expected_games": total, "complete": False}
    save(args.out, payload)
    print(f"{len(records)}/{total} recorded; {args.workers} worker processes", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(game, task): task for task in tasks}
        for future in as_completed(futures):
            row = future.result()
            records.append(row)
            save(args.out, payload)
            print(f"[{len(records)}/{total}] {row['agent']} vs {row['opponent']} seed={row['seed']} seat={row['seat']} "
                  f"{row['bank']:,.0f}/{row['opponent_bank']:,.0f} max={row['max_turn_ms']:.0f}ms", flush=True)
    payload["complete"] = len(records) == total
    payload["summary"] = summary(records)
    payload["comparisons"] = [paired_comparison(records, label, labels[0], opponent_labels, seeds)
                              for label in labels[1:]]
    save(args.out, payload)
    print(json.dumps({k: payload[k] for k in ("complete", "summary", "comparisons")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
