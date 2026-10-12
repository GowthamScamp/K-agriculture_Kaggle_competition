"""Prove the *bundle* survives with `kaggle_environments` absent.

    python -m tools.offline_check

`submission/main.py` imports `kaggle_environments` lazily inside the search, to
reuse the real turn logic, and is supposed to fall back to pure heuristic play
if that import fails. `tests/test_agents.py` checks that property for `src/`,
which is not the same claim: the bundle is a different file, assembled by a
build step, and it is the one Kaggle runs.

The check has to happen in a child process, because blocking an import means
mutating `sys.meta_path` and purging `sys.modules` - not something to do inside
a test runner that still needs the package afterwards. So this module runs in
two modes:

* parent (default) - play a short seeded episode to capture real observations,
  then re-invoke itself in `--replay` mode;
* `--replay PATH`  - block the import, load the bundle, and feed it those
  observations, asserting each returned action is well formed.
"""

from __future__ import annotations

import argparse
import importlib.abc
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "submission" / "main.py"
BLOCKED = "kaggle_environments"
CAPTURE_AT = (0, 200, 500)
SEED = 5


# --------------------------------------------------------------------------
# parent mode: capture real observations
# --------------------------------------------------------------------------
def capture(path: Path) -> list[int]:
    """Play a seeded episode and save observations from a few representative turns."""
    sys.path.insert(0, str(ROOT))
    from src.runner import _quiet, make_env

    grabbed: dict[int, Any] = {}

    def probe(obs: Any, config: Any = None) -> dict[str, Any]:
        step = obs["step"]
        if step in CAPTURE_AT and step not in grabbed:
            grabbed[step] = json.loads(json.dumps(dict(obs)))
        return {"farmer": ["PASS"], "hands": [], "market": []}

    env = make_env(seed=SEED)
    with _quiet():
        env.run([probe, "starter"])
    path.write_text(json.dumps(grabbed), encoding="utf-8")
    return sorted(grabbed)


# --------------------------------------------------------------------------
# child mode: block the import and exercise the bundle
# --------------------------------------------------------------------------
class _Blocker(importlib.abc.MetaPathFinder):
    """Make `kaggle_environments` unimportable, the way an offline runner would."""

    def find_spec(self, name: str, path: Any = None, target: Any = None) -> None:
        if name == BLOCKED or name.startswith(BLOCKED + "."):
            raise ImportError(f"offline check: {name} is unavailable")
        return None


def replay(samples_path: Path) -> int:
    for mod in [k for k in sys.modules if k == BLOCKED or k.startswith(BLOCKED + ".")]:
        del sys.modules[mod]
    sys.meta_path.insert(0, _Blocker())
    # Hide src/ too, so the bundle cannot quietly satisfy an import from it.
    sys.path = [p for p in sys.path if p and Path(p).resolve() != ROOT]

    try:
        import kaggle_environments  # noqa: F401
    except ImportError:
        print(f"  {BLOCKED:<24} blocked, as on an offline runner")
    else:
        print("  SETUP FAILED: the import was not actually blocked")
        return 1

    name = "kaggriculture_offline"
    spec = importlib.util.spec_from_file_location(name, MAIN)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    print("  bundle import            OK")

    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    ok = True
    for key in sorted(samples, key=int):
        obs = samples[key]
        t0 = time.perf_counter()
        action = module.agent(obs, None)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        hands = obs["farms"][obs["player"]].get("hands") or []
        legal = (
            isinstance(action, dict)
            and isinstance(action.get("farmer"), list)
            and action["farmer"]
            and isinstance(action.get("hands"), list)
            and len(action["hands"]) == len(hands)
            and isinstance(action.get("market"), list)
        )
        ok = ok and bool(legal)
        print(f"  step {key:>3}  {elapsed_ms:7.1f} ms  farmer={action['farmer'][0]:<14} "
              f"hands={len(action['hands'])}/{len(hands)}  market={len(action['market'])}  "
              f"{'legal' if legal else 'MALFORMED'}")

    print()
    print("  RESULT: " + ("offline-safe - heuristic play with no environment"
                          if ok else "FAILED - the bundle needs kaggle_environments"))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check the bundle runs without kaggle_environments.")
    ap.add_argument("--replay", type=Path, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--main", type=Path, default=None, help="bundle to check (default submission/main.py)")
    args = ap.parse_args(argv)

    global MAIN
    if args.main is not None:
        MAIN = args.main.resolve()

    if args.replay is not None:
        return replay(args.replay)

    if not MAIN.exists():
        print(f"  BLOCKER: {MAIN} does not exist; run `tools.build_submission`")
        return 1

    with tempfile.TemporaryDirectory() as td:
        samples = Path(td) / "obs.json"
        steps = capture(samples)
        print(f"\n  captured observations at steps {steps} (seed {SEED})")
        proc = subprocess.run(
            [sys.executable, "-m", "tools.offline_check", "--replay", str(samples),
             "--main", str(MAIN)],
            cwd=str(ROOT), capture_output=True, text=True,
        )
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0 and proc.stderr:
        sys.stderr.write(proc.stderr)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
