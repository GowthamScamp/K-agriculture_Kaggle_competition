"""Screen controlled ``RouteParams`` variants on fixed seeds.

This is deliberately a small economic diagnostic, not a candidate selector for
the competition entrypoint.  It runs both seats and reports the distribution and
execution telemetry needed to reject fragile parameter changes.

Example::

    python -m tools.screen_route --seeds 0,1,3,7 --set herd_commit_day=6
"""

from __future__ import annotations

import argparse
import ast
import statistics
from dataclasses import fields
from typing import Any, Sequence

from src.agents.reactive_route import ReactiveRouteAgent, RouteParams
from src.runner import run_episode


def _route_overrides(raw_values: Sequence[str]) -> dict[str, Any]:
    known = {field.name for field in fields(RouteParams)}
    overrides: dict[str, Any] = {}
    for raw in raw_values:
        name, separator, value = raw.partition("=")
        if not separator or name not in known:
            raise ValueError(f"expected RouteParams name=value, got {raw!r}")
        try:
            overrides[name] = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            overrides[name] = value
    return overrides


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--opponent", default="starter")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args(argv)

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    overrides = _route_overrides(args.overrides)
    banks: list[float] = []
    moves: list[float] = []
    productive: list[float] = []
    passes: list[float] = []
    feeds: list[float] = []
    cares: list[float] = []
    waters: list[float] = []
    noops: list[int] = []
    discarded: list[int] = []
    unsold: list[int] = []
    max_turn_ms = 0.0

    for seed in seeds:
        for seat in (0, 1):
            candidate = ReactiveRouteAgent(route_kwargs=overrides)
            agents = [candidate, args.opponent] if seat == 0 else [args.opponent, candidate]
            result = run_episode(
                agents,
                seed=seed,
                time_seat=seat,
                telemetry_seat=seat,
            )
            if any(status != "DONE" for status in result.statuses):
                raise RuntimeError(
                    f"non-DONE status at seed={seed}, seat={seat}: {result.statuses}"
                )
            telemetry = result.telemetry
            if telemetry is None:  # pragma: no cover - runner contract guard
                raise RuntimeError("telemetry was not returned")
            bank = result.rewards[seat]
            banks.append(bank)
            moves.append(telemetry.movement_rate)
            productive.append(telemetry.productive_rate)
            passes.append(telemetry.pass_rate)
            feeds.append(telemetry.feed_completion)
            cares.append(telemetry.care_completion)
            waters.append(telemetry.water_completion)
            noops.append(telemetry.avoidable_noops)
            discarded.append(telemetry.discarded_drop_units)
            unsold.append(telemetry.final_unsold_units)
            max_turn_ms = max(max_turn_ms, result.max_turn_time * 1000.0)
            print(
                f"seed={seed:>3} seat={seat} bank={bank:>9.0f} "
                f"herd={telemetry.final_animal_types} q={telemetry.max_quadrants} "
                f"move={telemetry.movement_rate:6.2%} feed={telemetry.feed_completion:6.2%}"
            )

    ordered = sorted(banks)
    percentile_10 = ordered[max(0, int(0.10 * (len(ordered) - 1)))]
    print(
        "summary "
        f"games={len(banks)} mean={statistics.fmean(banks):.1f} "
        f"p10={percentile_10:.1f} min={min(banks):.1f} max={max(banks):.1f} "
        f"move={statistics.fmean(moves):.2%} feed={statistics.fmean(feeds):.2%} "
        f"productive={statistics.fmean(productive):.2%} pass={statistics.fmean(passes):.2%} "
        f"care={statistics.fmean(cares):.2%} water={statistics.fmean(waters):.2%} "
        f"noops={sum(noops)} max_unsold={max(unsold)} "
        f"discarded_drop_units={sum(discarded)} "
        f"max_turn_ms={max_turn_ms:.1f} overrides={overrides}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
