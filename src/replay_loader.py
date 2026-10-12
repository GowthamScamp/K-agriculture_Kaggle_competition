"""Load public Kaggriculture replays, when they are available.

No Kaggle credentials exist on this machine, so nothing here has real data to
read yet. It is written and wired anyway so that dropping `kaggle.json` in place
and downloading the dataset changes *no call sites* - `build_replay_models()`
simply starts returning replay-derived opponents instead of an empty list.
See `docs/REPLAYS.md` for the exact commands.

What a replay gives us that self-play cannot: the real distribution of opponent
selling behaviour. Our own market impact is only half the picture, and modelling
the other half from our own agent risks assuming everyone plays like us.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
REPLAY_DIR = ROOT / "data" / "replays"


@dataclass(slots=True)
class ReplayEpisode:
    """One recorded game, reduced to what an opponent model needs."""

    path: Path
    final_banks: list[float]
    steps: int

    @property
    def winner(self) -> int:
        return 0 if self.final_banks[0] >= self.final_banks[1] else 1


def replays_available() -> bool:
    return REPLAY_DIR.is_dir() and any(REPLAY_DIR.glob("*.json"))


def iter_replays(limit: int | None = None) -> Iterator[ReplayEpisode]:
    """Yield parsed replays from `data/replays/`, skipping anything unreadable."""
    if not REPLAY_DIR.is_dir():
        return
    for i, path in enumerate(sorted(REPLAY_DIR.glob("*.json"))):
        if limit is not None and i >= limit:
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            steps = raw.get("steps") or []
            if not steps:
                continue
            final = steps[-1]
            banks = [float(s.get("reward") or 0.0) for s in final]
            yield ReplayEpisode(path=path, final_banks=banks, steps=len(steps))
        except Exception:
            continue


def replay_action_sequence(path: Path, seat: int) -> list[dict[str, Any]]:
    """Every action one seat played, in order. The basis for a replay policy."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for step in raw.get("steps") or []:
        if seat < len(step):
            action = step[seat].get("action")
            if isinstance(action, dict):
                out.append(action)
    return out


class ReplayPolicy:
    """Replays a recorded action sequence, falling back to PASS when exhausted.

    Deliberately open-loop: it ignores the state it is shown. That is wrong in
    detail - the recorded actions were chosen against a different board - but it
    reproduces the opponent's *market* behaviour, which is the part that couples
    to our decisions through shared prices.
    """

    def __init__(self, actions: list[dict[str, Any]]) -> None:
        self.actions = actions

    def __call__(self, obs: Any, config: Any = None) -> dict[str, Any]:
        try:
            step = int(obs["step"])
            farms = obs.get("farms") or []
            player = int(obs.get("player", 0))
            n_hands = len(farms[player].get("hands") or []) if player < len(farms) else 0
            if step < len(self.actions):
                a = self.actions[step]
                hands = list(a.get("hands") or [])[:n_hands]
                hands += [["PASS"]] * (n_hands - len(hands))
                return {
                    "farmer": list(a.get("farmer") or ["PASS"]),
                    "hands": hands,
                    "market": list(a.get("market") or []),
                }
        except Exception:
            pass
        return {"farmer": ["PASS"], "hands": [], "market": []}


def build_replay_models(limit: int = 4) -> list[tuple[str, ReplayPolicy]]:
    """Opponent policies derived from real games. Empty until replays exist."""
    models: list[tuple[str, ReplayPolicy]] = []
    for ep in iter_replays(limit=limit):
        try:
            seat = ep.winner  # learn from whoever actually won
            models.append((f"replay:{ep.path.stem}:{seat}",
                           ReplayPolicy(replay_action_sequence(ep.path, seat))))
        except Exception:
            continue
    return models


def _cli() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Inspect downloaded replays.")
    ap.add_argument("--stat", action="store_true")
    ap.add_argument("--limit", type=int, default=20)
    ap.parse_args()

    if not replays_available():
        print(f"\n  no replays found in {REPLAY_DIR}")
        print("  see docs/REPLAYS.md for how to download them")
        print("  (the agent works without them; opponent models fall back to self-play)\n")
        return 0
    eps = list(iter_replays(limit=None))
    print(f"\n  {len(eps)} replays in {REPLAY_DIR}")
    banks = [b for e in eps for b in e.final_banks]
    if banks:
        print(f"  final bank: min {min(banks):,.0f}  max {max(banks):,.0f}  "
              f"mean {sum(banks) / len(banks):,.0f}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
