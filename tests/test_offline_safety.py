"""The submitted bundle must not need `kaggle_environments` to play.

`test_agents.py` already proves the *source* agent degrades gracefully when the
simulator is unavailable. That is a different claim from the one that matters on
submission day: `submission/main.py` is a generated file, and it is the artefact
Kaggle actually runs. If the build ever hoisted the lazy import to module scope,
every existing gate would still pass and the agent would fail to import offline.

Delegated to `tools/offline_check.py`, which does the work in a child process -
blocking an import means mutating `sys.meta_path` and purging `sys.modules`, and
doing that inside the test session would break every later test.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "submission" / "main.py"


@pytest.mark.skipif(not MAIN.exists(), reason="no bundle built yet")
def test_bundle_plays_without_kaggle_environments() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "tools.offline_check"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    assert proc.returncode == 0, (
        "the bundle does not survive without kaggle_environments:\n"
        + proc.stdout + proc.stderr
    )
    assert "offline-safe" in proc.stdout
