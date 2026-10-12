"""Run a seat-balanced offline pool and fit a Bradley--Terry ranking.

Examples::

    python -m tools.competitive_pool --agent reactive_route \
        --opponent starter \
        --opponent lonespear=benchmarks/public/lonespear/main.py \
        --seeds 8 --round-robin

Opponent files are loaded as ordinary Python modules and must expose ``agent``.
The harness never reads replay logs and writes only the optional JSON summary.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from src.evaluate import WinRateResult
from src.registry import AGENT_NAMES, build_agent
from src.runner import run_episode

ROOT = Path(__file__).resolve().parents[1]
AgentFactory = Callable[[], Any]


@dataclass(slots=True)
class Competitor:
    name: str
    factory: AgentFactory


@dataclass(slots=True)
class PairResult:
    a: str
    b: str
    banks_a: list[float]
    banks_b: list[float]
    seats_a: list[int]
    max_turn_a_ms: float

    @property
    def margins(self) -> list[float]:
        return [a - b for a, b in zip(self.banks_a, self.banks_b)]

    @property
    def outcome(self) -> WinRateResult:
        return WinRateResult.of(self.margins)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["outcome"] = asdict(self.outcome)
        payload["mean_bank_a"] = statistics.fmean(self.banks_a)
        payload["mean_bank_b"] = statistics.fmean(self.banks_b)
        return payload


def _competitor(
    specification: str, build_kwargs: dict[str, Any] | None = None
) -> Competitor:
    """Parse ``registry_name`` or ``label=path/to/main.py[:symbol]``."""

    if "=" not in specification:
        if specification not in AGENT_NAMES and specification not in ("starter", "pass", "random"):
            raise ValueError(f"unknown registry agent {specification!r}")
        return Competitor(
            specification,
            lambda name=specification, kwargs=build_kwargs or {}: build_agent(name, **kwargs),
        )
    if build_kwargs:
        raise ValueError("build kwargs are only supported for registry agents")

    label, raw_location = specification.split("=", 1)
    # rpartition + identifier test, so a Windows drive colon ("C:\...") is
    # never mistaken for the ":symbol" suffix.
    head, separator, tail = raw_location.rpartition(":")
    if separator and tail.isidentifier():
        raw_path, symbol = head, tail
    else:
        raw_path, symbol = raw_location, "agent"
    path = (ROOT / raw_path).resolve() if not Path(raw_path).is_absolute() else Path(raw_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    def factory() -> Any:
        # A fresh module per game prevents stateful public agents leaking memory
        # from one seed or seat into the next.
        module_name = f"kaggriculture_benchmark_{label}_{time.time_ns()}"
        module_spec = importlib.util.spec_from_file_location(module_name, path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"cannot load {path}")
        module = importlib.util.module_from_spec(module_spec)
        # Dataclasses resolve annotations through sys.modules during import.
        # Remove the temporary registration afterward so a large league does
        # not retain hundreds of decoded route portfolios indefinitely.
        sys.modules[module_name] = module
        try:
            module_spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)
        candidate = getattr(module, symbol)
        if not callable(candidate):
            raise TypeError(f"{path}:{symbol} is not callable")
        return candidate

    return Competitor(label, factory)


def play_pair(a: Competitor, b: Competitor, seeds: Sequence[int]) -> PairResult:
    banks_a: list[float] = []
    banks_b: list[float] = []
    seats_a: list[int] = []
    max_a = 0.0
    for seed in seeds:
        for seat_a in (0, 1):
            first = a.factory() if seat_a == 0 else b.factory()
            second = b.factory() if seat_a == 0 else a.factory()
            result = run_episode(
                [first, second],
                seed=seed,
                time_seat=seat_a,
            )
            own = result.rewards[seat_a]
            rival = result.rewards[1 - seat_a]
            banks_a.append(own)
            banks_b.append(rival)
            seats_a.append(seat_a)
            max_a = max(max_a, result.max_turn_time)

            if any(status != "DONE" for status in result.statuses):
                raise RuntimeError(
                    f"non-DONE status for {a.name} vs {b.name}, seed={seed}, "
                    f"seat={seat_a}: {result.statuses}"
                )
    return PairResult(a.name, b.name, banks_a, banks_b, seats_a, max_a * 1000.0)


def bradley_terry(competitors: Sequence[Competitor], rows: Sequence[PairResult]) -> dict[str, float]:
    """Fit regularized strengths by minorization-maximization.

    A half-win/half-loss pseudo-game per pair prevents the infinite maximum
    likelihood estimate produced when a small screen contains a clean sweep.
    """

    names = [c.name for c in competitors]
    wins = {name: 0.5 * (len(names) - 1) for name in names}
    games = {(a, b): 1 for a in names for b in names if a != b}
    for row in rows:
        for margin in row.margins:
            games[row.a, row.b] += 1
            games[row.b, row.a] += 1
            if margin > 0:
                wins[row.a] += 1.0
            elif margin < 0:
                wins[row.b] += 1.0
            else:
                wins[row.a] += 0.5
                wins[row.b] += 0.5

    strength = {name: 1.0 for name in names}
    for _ in range(500):
        updated: dict[str, float] = {}
        for name in names:
            denominator = sum(
                games.get((name, other), 0) / (strength[name] + strength[other])
                for other in names
                if other != name
            )
            updated[name] = max(1e-12, wins[name]) / max(1e-12, denominator)
        geometric = math.exp(sum(math.log(value) for value in updated.values()) / len(updated))
        updated = {name: value / geometric for name, value in updated.items()}
        delta = max(abs(math.log(updated[name] / strength[name])) for name in names)
        strength = updated
        if delta < 1e-10:
            break
    return {name: 1500.0 + 400.0 * math.log10(value) for name, value in strength.items()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", default="reactive_route")
    parser.add_argument(
        "--agent-kwargs",
        default="{}",
        help='JSON kwargs passed to the registry factory, e.g. {"route_kwargs":{"flex_slots":6}}',
    )
    parser.add_argument(
        "--agent-set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="RouteParams override for the primary registry agent; repeat as needed",
    )
    parser.add_argument("--opponent", action="append", required=True)
    parser.add_argument("--seeds", type=int, default=8, help="seeds per seat and matchup")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--round-robin", action="store_true")
    parser.add_argument("--json", default=str(ROOT / "logs" / "competitive_pool.json"))
    args = parser.parse_args(argv)

    agent_kwargs = json.loads(args.agent_kwargs)
    if not isinstance(agent_kwargs, dict):
        raise ValueError("--agent-kwargs must decode to a JSON object")
    if args.agent_set:
        route_kwargs = dict(agent_kwargs.get("route_kwargs") or {})
        for raw in args.agent_set:
            name, separator, value = raw.partition("=")
            if not separator:
                raise ValueError(f"expected --agent-set NAME=VALUE, got {raw!r}")
            try:
                route_kwargs[name] = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                route_kwargs[name] = value
        agent_kwargs["route_kwargs"] = route_kwargs
    competitors = [
        _competitor(args.agent, agent_kwargs),
        *(_competitor(spec) for spec in args.opponent),
    ]
    names = [competitor.name for competitor in competitors]
    if len(names) != len(set(names)):
        raise ValueError(f"competitor labels must be unique: {names}")
    seeds = range(args.seed_start, args.seed_start + args.seeds)
    pairs = (
        [(i, j) for i in range(len(competitors)) for j in range(i + 1, len(competitors))]
        if args.round_robin
        else [(0, j) for j in range(1, len(competitors))]
    )

    rows: list[PairResult] = []
    for index, (i, j) in enumerate(pairs, 1):
        row = play_pair(competitors[i], competitors[j], seeds)
        rows.append(row)
        outcome = row.outcome
        print(
            f"[{index}/{len(pairs)}] {row.a} vs {row.b}: {outcome.record}, "
            f"win={outcome.rate:.1%} [{outcome.ci_low:.1%}, {outcome.ci_high:.1%}], "
            f"bank={statistics.fmean(row.banks_a):,.0f}/{statistics.fmean(row.banks_b):,.0f}"
        )

    ratings = bradley_terry(competitors, rows)
    print("\nBradley--Terry local rating")
    for name, rating in sorted(ratings.items(), key=lambda item: -item[1]):
        print(f"  {name:<24} {rating:8.1f}")

    payload = {
        "seeds": list(seeds),
        "both_seats": True,
        "round_robin": args.round_robin,
        "rows": [row.to_dict() for row in rows],
        "bradley_terry": ratings,
    }
    output = Path(args.json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
