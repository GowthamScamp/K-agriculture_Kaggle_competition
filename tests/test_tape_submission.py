"""Contract for the slot-1 tape artifact, `submission/tape/main.py`.

The artifact is a public Apache-2.0 chassis copied byte-for-byte, followed by a
wrapper that only acts when an exception escapes the chassis. These tests pin
the properties an upload depends on: Kaggle binds the wrapper, both licence
notices survive, nothing outside the stdlib is imported, the file is exactly
what the builder emits, and a chassis failure degrades to the fallback policy.
"""

from __future__ import annotations

import ast
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TAPE = ROOT / "submission" / "tape" / "main.py"
TAPE_ZIP = ROOT / "submission" / "tape" / "kaggriculture_tape.zip"

needs_tape = pytest.mark.skipif(not TAPE.exists(), reason="no tape artifact built yet")


def _namespace() -> dict:
    ns: dict = {}
    exec(compile(TAPE.read_text(encoding="utf-8"), str(TAPE), "exec"), ns)
    return ns


@needs_tape
def test_kaggle_binds_the_wrapper_agent() -> None:
    from kaggle_environments.agent import get_last_callable

    raw = TAPE.read_text(encoding="utf-8")
    bound = get_last_callable(raw, path=str(TAPE))
    last_def = [n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef)][-1]
    assert bound.__name__ == "agent"
    assert bound.__code__.co_firstlineno == last_def.lineno


@needs_tape
def test_licence_notices_are_retained() -> None:
    text = TAPE.read_text(encoding="utf-8")
    assert "Apache" in text.split("Submitter modifications")[0], "chassis notice missing"
    assert "Submitter modifications (Apache License 2.0, section 4(b))" in text
    assert "Copyright (c) 2026 Jonathan Day" in text, "fallback MIT notice missing"


@needs_tape
def test_tape_imports_only_the_standard_library() -> None:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(TAPE.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    assert not (roots - set(sys.stdlib_module_names)), sorted(roots - set(sys.stdlib_module_names))


@needs_tape
def test_tape_is_exactly_what_the_builder_emits() -> None:
    from tools.build_submission import render_tape_submission

    assert render_tape_submission() == TAPE.read_bytes(), "rebuild with --target tape"


@needs_tape
def test_primary_verifier_accepts_a_promoted_tape(monkeypatch) -> None:
    import tools.verify_submission as verifier

    monkeypatch.setattr(verifier, "MAIN", TAPE)
    monkeypatch.setattr(verifier, "ARCHIVE", TAPE_ZIP)
    monkeypatch.setattr(verifier, "TARGET", "primary")
    assert verifier.check_reproducible() == []


@needs_tape
def test_tape_archive_matches_the_py() -> None:
    with zipfile.ZipFile(TAPE_ZIP) as zf:
        assert zf.namelist() == ["main.py"]
        assert zf.read("main.py") == TAPE.read_bytes()


@needs_tape
def test_chassis_exception_falls_back_to_competitive_meta() -> None:
    from kaggle_environments import make

    env = make("kaggriculture", configuration={"seed": 3})
    env.reset(2)
    obs = env.state[0].observation

    def boom(*_args, **_kwargs):
        raise RuntimeError("forced chassis failure")

    ns = _namespace()
    ns["_KGW_CHASSIS_AGENT"] = boom
    action = ns["agent"](obs, env.configuration)
    assert ns["_KGW_STATE"]["chassis_errors"] == 1
    assert action["market"], "fallback produced no market orders; it is not competitive_meta"

    ns["_kgw_fallback"] = boom
    assert ns["agent"](obs, env.configuration) == {"farmer": ["PASS"], "hands": [], "market": []}


@needs_tape
def test_binding_check_catches_a_callable_assigned_after_agent(tmp_path, monkeypatch) -> None:
    """The static last-def check passes this mutation; Kaggle's real rule does not."""
    import tools.verify_submission as vs

    mutated = tmp_path / "main.py"
    mutated.write_bytes(TAPE.read_bytes() + b"\n_late_alias = len\n")
    monkeypatch.setattr(vs, "MAIN", mutated)
    problems = vs.check_kaggle_binding()
    assert problems and "len" in problems[0]
