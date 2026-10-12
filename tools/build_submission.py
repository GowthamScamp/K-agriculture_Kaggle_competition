"""Build the historical heuristic control and guarded tape candidate.

    python -m tools.build_submission

The checked-in ``submission/main.py`` is the historical heuristic control,
maintained at ``src/agents/competitive_meta.py``. The ``meta`` target rebuilds
it byte-for-byte. The separately built tape artifact is only promoted to
``submission/main.py`` by ``primary`` when the recorded competitive and serial
verification gates pass.

Two artifacts are written, because Kaggle resolves the entry point differently
for each and the difference is a real safety property - see `write_archive`:

- `submission/main.py` - selected policy; bound by **the last `def`**
- `submission/kaggriculture.zip` - same bytes; bound by **`main.py` at top level**

The artifact entry point is checked by `tools/verify_submission.py`. The old
reactive-route flattener is retained as research infrastructure and is not part
of the current rendered policy.

Dependencies in the generated file: the standard library only. The submitted
policies do not require `kaggle_environments`, so the entry point remains
usable in Kaggle's isolated agent process.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import zipfile
import zlib
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUT = ROOT / "submission" / "main.py"
ARCHIVE = ROOT / "submission" / "kaggriculture.zip"
META_SOURCE = SRC / "agents" / "competitive_meta.py"
META_OUT = OUT
META_ARCHIVE = ARCHIVE
MANIFEST = ROOT / "submission" / "manifest.json"

TAPE_OUT = ROOT / "submission" / "tape" / "main.py"
TAPE_ARCHIVE = ROOT / "submission" / "tape" / "kaggriculture_tape.zip"
_UPSTREAM = (
    "https://github.com/warlord-2227/kaggle/blob/"
    "f8ddefbef8393011363746b158692ab2ca89d9aa/refagents/public/"
)
#: Apache-2.0 public chassis eligible for slot 1, pinned by content hash so a
#: silently edited benchmark file cannot reach the upload.
TAPE_SOURCES = {
    "farm2945": (
        ROOT / "benchmarks" / "public" / "farm2945" / "main.py",
        "BFEE70E9DAAEBEAE0737A880F1DF8F1C60D0783C59AF620136CC0D28EF482BC7",
        _UPSTREAM + "farm2945_v9_4.py",
    ),
    "tetsutani_dp": (
        ROOT / "benchmarks" / "public" / "tetsutani_dp" / "main.py",
        "9C0A8921AA49FBA080D53DEEE3C9FE8DB7A8E2B02881E9ADDA4090AF73E9C187",
        _UPSTREAM + "tetsutani_demand_preserving.py",
    ),
}
#: 16-0 over farm2945 in the seeds 0-7 round-robin (docs/tape_chassis.md).
TAPE_SELECTED = "tetsutani_dp"

#: Fixed timestamp for the zip entry. Zip records a mtime per member, so without
#: this the archive's bytes change on every rebuild and the "reproducible from
#: src/" claim would hold for the .py but quietly fail for the .zip.
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)

# Dependency order: each module may only reference names defined above it.
MODULES = [
    SRC / "game_constants.py",
    SRC / "env_adapter.py",
    SRC / "agents" / "baseline_agent.py",
    SRC / "agents" / "reactive_route.py",
]

# Lines importing from inside our own package: dropped, since the flattened file
# shares one namespace.
INTRA_IMPORT = re.compile(r"^\s*from\s+\.{1,2}[\w.]*\s+import\s+", re.MULTILINE)
INTRA_IMPORT_BLOCK = re.compile(
    r"^from\s+\.{1,2}[\w.]*\s+import\s+\([^)]*\)\n", re.MULTILINE
)
FUTURE = re.compile(r"^from __future__ import .*\n", re.MULTILINE)

HEADER = '''"""Kaggriculture submission - generated, do not edit by hand.

Built from src/ by `python -m tools.build_submission` on {today}.
Edit the modules under src/ and rebuild; changes made here will be overwritten.

Entry point: `agent(obs)` at the bottom of this file.

Strategy, in one paragraph: operate a compact two-quadrant cow/sheep and
wheat/strawberry economy; keep workers on stable daily quadrant routes; preserve
feeding, watering, and care before optional work; choose the herd mix from the
first durable shop-demand evidence; meter sales against visible town and opponent
supply; reuse expired long-crop slots for short endgame crops; and liquidate all
bankable inventory by the real final executable turn, step 718.

Safety: every decision is wrapped so a failure degrades to a legal PASS rather
than forfeiting the episode. The route policy is deterministic and uses only
the public observation and standard-library code.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

'''

FOOTER = '''

# ==========================================================================
# Kaggle entry point
# ==========================================================================
# One instance for the whole episode so public market estimates, daily worker
# zones, and unfinished route commitments persist between observations.
_AGENT = ReactiveRouteAgent()


def agent(obs: Any, config: Any = None) -> dict[str, Any]:
    """The function Kaggle calls once per turn.

    `ReactiveRouteAgent` already has an internal safety boundary; this outer
    guard also protects the generated entry point itself.
    """
    try:
        return _AGENT(obs, config)
    except Exception:
        try:
            farms = obs.get("farms") or []
            player = int(obs.get("player", 0))
            n_hands = len(farms[player].get("hands") or []) if player < len(farms) else 0
        except Exception:
            n_hands = 0
        return {"farmer": ["PASS"], "hands": [["PASS"]] * n_hands, "market": []}
'''


def strip_module(path: Path) -> str:
    """Return a module's source with its docstring header and local imports removed."""
    text = path.read_text(encoding="utf-8")

    # Drop the module docstring: the combined file has its own header, and the
    # per-module design notes stay where they belong, in src/.
    if text.lstrip().startswith('"""'):
        start = text.index('"""')
        end = text.index('"""', start + 3) + 3
        text = text[end:]

    text = FUTURE.sub("", text)
    text = INTRA_IMPORT_BLOCK.sub("", text)
    text = "\n".join(
        line for line in text.splitlines()
        if not INTRA_IMPORT.match(line + "\n")
    )
    return text.strip("\n")


def write_archive(out: Path = OUT, archive: Path = ARCHIVE) -> None:
    """Package the same bytes as a zip whose single member is `main.py`.

    The two artifacts differ in how Kaggle finds the entry point, which is the
    whole point of shipping both. A bare `.py` is bound to **the last `def` in
    the file** - positional, and therefore quietly breakable by reordering. An
    archive is bound to **`main.py` at the top level** - by name, so no amount
    of reordering inside the file can misdirect it.

    `.zip` rather than `.gz`: a bare gzip stream carries one nameless blob, so
    "main.py at the top level" cannot be expressed in it at all.
    """
    archive.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo("main.py", date_time=ZIP_EPOCH)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(info, out.read_bytes())


def render_meta_submission() -> str:
    """Return the attributed heuristic control, not the recommended upload."""

    text = META_SOURCE.read_text(encoding="utf-8")
    if "Copyright (c) 2026 Jonathan Day" not in text:
        raise ValueError("competitive source is missing its required MIT notice")
    if not text.endswith("\n"):
        text += "\n"
    return text


TAPE_FOOTER = '''

# ==========================================================================
# Submitter modifications (Apache License 2.0, section 4(b))
# ==========================================================================
# Everything above this banner is the public Apache-2.0 chassis `{name}`,
# reproduced byte-for-byte from {url}
# (SHA-256 {sha}); its notices are retained unchanged.
#
# Everything below was added by the submitter. It does not alter chassis
# behaviour: it wraps the chassis entry point and, only when an exception
# escapes it, answers that one turn with the fallback policy embedded below.
#
# The fallback is `competitive_meta`, derived from lonespear/kaggriculture
# main_v18.py and distributed under this notice:
#
{mit_notice}

import base64 as _kgw_base64
import zlib as _kgw_zlib

_KGW_CHASSIS_AGENT = agent
# Kaggle binds the last *callable inserted* into the module namespace, and a
# redefined name keeps its original slot, so the old binding must be deleted.
del agent

_KGW_FALLBACK_SOURCE = (
{fallback_b64}
)
_KGW_STATE = {{"fallback": None, "chassis_errors": 0, "fallback_errors": 0,
               "last_step": -1, "player": None}}


def _kgw_fallback(observation, configuration):
    fallback = _KGW_STATE["fallback"]
    if fallback is None:
        namespace = {{"__name__": "kaggriculture_fallback"}}
        source = _kgw_zlib.decompress(_kgw_base64.b64decode(_KGW_FALLBACK_SOURCE))
        exec(compile(source.decode("utf-8"), "competitive_meta.py", "exec"), namespace)
        fallback = _KGW_STATE["fallback"] = namespace["agent"]
    return fallback(observation, configuration)


def agent(observation, configuration=None):
    try:
        raw_step = observation.get("step")
        step = int(raw_step) if raw_step is not None else int(observation.get("day") or 0) * 24 + int(observation.get("hour") or 0)
        player = int(observation.get("player") or 0)
        if step < _KGW_STATE["last_step"] or player != _KGW_STATE["player"]:
            _KGW_STATE.update(fallback=None, chassis_errors=0, fallback_errors=0)
        _KGW_STATE.update(last_step=step, player=player)
        if raw_step is None:
            observation = dict(observation, step=step)
        return _KGW_CHASSIS_AGENT(observation, configuration)
    except Exception:
        _KGW_STATE["chassis_errors"] += 1
    try:
        return _kgw_fallback(observation, configuration)
    except Exception:
        _KGW_STATE["fallback_errors"] += 1
        return {{"farmer": ["PASS"], "hands": [], "market": []}}
'''


def _mit_notice(meta_text: str) -> str:
    """The MIT header block of `competitive_meta.py`, as comment lines."""
    lines = []
    for line in meta_text.splitlines():
        if not line.startswith("#"):
            break
        lines.append(line)
    notice = "\n".join(lines)
    if "Copyright (c) 2026 Jonathan Day" not in notice or "MIT License" not in notice:
        raise ValueError("competitive_meta.py no longer starts with its MIT notice")
    return notice


def render_tape_submission(name: str = TAPE_SELECTED) -> bytes:
    """The pinned Apache chassis plus the fallback wrapper, as exact bytes."""
    path, expected_sha, url = TAPE_SOURCES[name]
    chassis = path.read_bytes()
    actual_sha = hashlib.sha256(chassis).hexdigest().upper()
    if actual_sha != expected_sha:
        raise ValueError(f"{path} changed: SHA-256 {actual_sha}, pinned {expected_sha}")
    text = chassis.decode("utf-8")
    if "Apache" not in text:
        raise ValueError(f"{path} is missing its Apache-2.0 notice")
    if not text.endswith("\n"):
        text += "\n"

    meta = META_SOURCE.read_text(encoding="utf-8")
    packed = base64.b64encode(zlib.compress(meta.encode("utf-8"), 9)).decode("ascii")
    chunks = "\n".join(f'    "{packed[i:i + 96]}"' for i in range(0, len(packed), 96))
    footer = TAPE_FOOTER.format(
        name=name, url=url, sha=expected_sha,
        mit_notice=_mit_notice(meta), fallback_b64=chunks,
    )
    return (text + footer).encode("utf-8")


def build_tape(name: str = TAPE_SELECTED) -> int:
    TAPE_OUT.parent.mkdir(parents=True, exist_ok=True)
    TAPE_OUT.write_bytes(render_tape_submission(name))
    print(f"  wrote {TAPE_OUT}  ({TAPE_OUT.stat().st_size / 1024:.0f} KB, chassis {name})")
    info = zipfile.ZipInfo("main.py", date_time=ZIP_EPOCH)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(TAPE_ARCHIVE, "w") as zf:
        zf.writestr(info, TAPE_OUT.read_bytes())
    print(f"  wrote {TAPE_ARCHIVE}  ({TAPE_ARCHIVE.stat().st_size / 1024:.0f} KB, main.py at top level)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build submission artifacts.")
    parser.add_argument("--target", choices=("primary", "meta", "tape", "all"), default="primary")
    parser.add_argument("--chassis", choices=sorted(TAPE_SOURCES), default=TAPE_SELECTED)
    parser.add_argument("--evidence", type=Path, default=ROOT / "logs/score_holdout_20261002.json")
    parser.add_argument("--verification", type=Path, default=ROOT / "logs/t0_contract_20261002.json")
    args = parser.parse_args(argv)
    status = 0
    if args.target in ("meta", "all"):
        status |= build_meta()
    if args.target in ("tape", "all"):
        status |= build_tape(args.chassis)
    if args.target in ("primary", "all"):
        status |= build_primary(args.evidence, args.verification, args.chassis)
    return status


def build_meta() -> int:
    META_OUT.parent.mkdir(parents=True, exist_ok=True)
    # Encode explicitly so Windows text-mode newline translation cannot make
    # the generated artifact differ byte-for-byte from the measured source.
    META_OUT.write_bytes(render_meta_submission().encode("utf-8"))

    text = META_OUT.read_text(encoding="utf-8")
    size_kb = len(text.encode("utf-8")) / 1024
    leftover = INTRA_IMPORT.findall(text)
    print(f"  wrote {META_OUT}  ({size_kb:.0f} KB, {text.count(chr(10)):,} lines)")
    if leftover:
        print(f"  WARNING: {len(leftover)} intra-package imports survived: {leftover[:3]}")
        return 1

    write_archive(META_OUT, META_ARCHIVE)
    print(f"  wrote {META_ARCHIVE}  ({META_ARCHIVE.stat().st_size / 1024:.0f} KB, main.py at top level)")
    return 0


def render_submission() -> str:
    """Render the checked-in historical heuristic control for ``main.py``."""
    return render_meta_submission()


def build_primary(evidence: Path, verification: Path, chassis: str = TAPE_SELECTED) -> int:
    """Promote only complete held-out evidence plus exact-artifact verification."""
    if not evidence.exists() or not verification.exists():
        print("  BLOCKED: complete held-out evidence and artifact verification are required; main.py unchanged")
        return 1
    from tools.score_gate import paired_comparison

    measured = json.loads(evidence.read_bytes())
    checked = json.loads(verification.read_bytes())
    data = render_tape_submission(chassis)
    digest = hashlib.sha256(data).hexdigest()
    identity = measured["identity"]
    own = next((r for r in identity["agents"][1:] if r["sha256"] == digest), None)
    rivals = [r["label"] for r in identity["opponents"]]
    if own is None or not measured.get("complete") or len(rivals) < 6:
        print("  BLOCKED: evidence must cover this exact artifact and the six-family frozen league")
        return 1
    comparison = paired_comparison(measured["games"], own["label"], identity["agents"][0]["label"], rivals, identity["seeds"])
    if not comparison["economic_gate_passed"] or identity["seeds"] != list(range(6000, 6096)):
        print("  BLOCKED:", comparison["blockers"], "; main.py unchanged")
        return 1
    if checked.get("sha256") != digest or not checked.get("passed") or checked.get("max_turn_ms", 1000) >= 500:
        print("  BLOCKED: exact artifact must pass serial verification below 500 ms; main.py unchanged")
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(data)
    write_archive()
    manifest = {"schema_version": 1, "policy": chassis + " + guarded heuristic fallback",
                "main": "submission/main.py", "sha256": digest,
                "chassis_sha256": TAPE_SOURCES[chassis][1],
                "fallback_sha256": hashlib.sha256(META_SOURCE.read_bytes()).hexdigest(),
                "evidence": str(evidence.relative_to(ROOT)), "verification": str(verification.relative_to(ROOT)),
                "comparison": comparison, "summary": measured["summary"],
                "kaggle_uploaded": False, "live_rating_guarantee": None,
                "competition_deadline_utc": "2026-09-30T23:59:00Z"}
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  PROMOTED {chassis}: {OUT}, SHA-256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
