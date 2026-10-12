"""Compare in-memory telemetry for both agents in one deterministic matchup.

The environment currently instruments one seat at a time, so the same seeded
match is replayed twice with fresh agent instances.  No replay file is written
or read.
"""

from __future__ import annotations

import argparse
from typing import Any, Sequence

from src.runner import EpisodeTelemetry, run_episode
from tools.competitive_pool import _competitor


def _summary(name: str, bank: float, telemetry: EpisodeTelemetry) -> dict[str, Any]:
    checkpoints = {3, 6, 9, 12, 18, 24, 29}
    bank_by_day = dict(telemetry.daily_bank)
    hands_by_day = dict(telemetry.daily_hands)
    crops_by_day = dict(telemetry.daily_crops)
    animals_by_day = dict(telemetry.daily_animals)
    return {
        "agent": name,
        "bank": bank,
        "total_ops": telemetry.total_unit_ops,
        "movement": round(telemetry.movement_rate, 4),
        "productive": round(telemetry.productive_rate, 4),
        "pass": round(telemetry.pass_rate, 4),
        "water": round(telemetry.water_completion, 4),
        "feed": round(telemetry.feed_completion, 4),
        "care": round(telemetry.care_completion, 4),
        "noops": telemetry.avoidable_noops,
        "discarded_drop_units": telemetry.discarded_drop_units,
        "input_drop_units": telemetry.input_drop_units,
        "unit_ops": dict(telemetry.unit_ops),
        "market_units": dict(telemetry.market_units),
        "crop_plants": dict(telemetry.crop_plants),
        "daily_farm": [
            {
                "day": day,
                "bank": bank_by_day.get(day),
                "hands": hands_by_day.get(day),
                "crops": crops_by_day.get(day, {}),
                "animals": animals_by_day.get(day, {}),
            }
            for day in sorted(checkpoints)
        ],
        "final_crops": dict(telemetry.final_crops),
        "final_animals": dict(telemetry.final_animal_types),
        "final_unsold_units": telemetry.final_unsold_units,
        "final_unsold_value": round(telemetry.final_unsold_value, 1),
        "shops": telemetry.final_shops,
        "prices": telemetry.final_prices,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", default="reactive_route")
    parser.add_argument("--right", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    competitors = (_competitor(args.left), _competitor(args.right))
    expected_rewards: list[float] | None = None
    for seat, competitor in enumerate(competitors):
        agents = [entry.factory() for entry in competitors]
        result = run_episode(
            agents,
            seed=args.seed,
            telemetry_seat=seat,
            time_seat=seat,
        )
        if expected_rewards is None:
            expected_rewards = result.rewards
        elif result.rewards != expected_rewards:
            raise RuntimeError(
                "repeated seeded match was not deterministic: "
                f"{expected_rewards!r} != {result.rewards!r}"
            )
        if result.telemetry is None:  # pragma: no cover - runner contract guard
            raise RuntimeError("telemetry was not returned")
        print(_summary(competitor.name, result.rewards[seat], result.telemetry))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
