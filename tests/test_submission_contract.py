"""How Kaggle finds our entry point, pinned so a rebuild cannot break it.

The submission dialog states two different binding rules, one per upload shape:

    "Your submission should be a python file with the last 'def' accepting an
    observation and returning an action. You can also upload multiple files in
    a zip/gz/7z archive with a main.py at the top level."

The first is **positional**, and that is the dangerous one. `submission/main.py`
is a generated 2,800-line flattening of six modules with ~45 top-level `def`s,
and `agent` sits last only because `build_submission` appends its footer last.
Nothing about that is structural. Add a module to `MODULES` after the footer, or
a helper below `agent`, and Kaggle binds the wrong function: every episode is
forfeited, and no other gate notices, because `hasattr(module, "agent")` is still
perfectly true.

So these tests exist to fail on a reordered build. They parse the artefact rather
than importing it, which keeps them meaningful even when an import would fail for
some unrelated reason.
"""

from __future__ import annotations

import ast
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "submission" / "main.py"
ARCHIVE = ROOT / "submission" / "kaggriculture.zip"

needs_bundle = pytest.mark.skipif(not MAIN.exists(), reason="no bundle built yet")
needs_archive = pytest.mark.skipif(not ARCHIVE.exists(), reason="no archive built yet")


def _module() -> ast.Module:
    return ast.parse(MAIN.read_text(encoding="utf-8"))


def _top_level_functions(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


@needs_bundle
def test_agent_is_the_last_top_level_def() -> None:
    """The rule Kaggle actually applies to a bare `.py` upload."""
    funcs = _top_level_functions(_module())
    assert funcs, "bundle defines no top-level functions"
    assert funcs[-1].name == "agent", (
        f"the last top-level def is `{funcs[-1].name}`, not `agent`. Kaggle binds "
        "the LAST def in the file, so it would run that instead. Check the order "
        "of MODULES and FOOTER in tools/build_submission.py."
    )


@needs_bundle
def test_nothing_is_defined_after_the_entry_point() -> None:
    """`agent` must also be the final statement, not merely the final function.

    A trailing `class` would not steal the binding, but it would mean the footer
    is no longer last - which is the single assumption the rule above rests on.
    Catching it here keeps the invariant honest rather than incidentally true.
    """
    tree = _module()
    last_stmt = tree.body[-1]
    assert isinstance(last_stmt, (ast.FunctionDef, ast.AsyncFunctionDef)), (
        f"a top-level {type(last_stmt).__name__} trails the entry point at line "
        f"{last_stmt.lineno}"
    )
    assert last_stmt.name == "agent"


@needs_bundle
def test_agent_is_callable_with_only_an_observation() -> None:
    """Kaggle says "accepting an observation" - singular.

    `kaggle_environments` passes `(obs, config)`, but the dialog describes a
    one-argument call, so every parameter past the first needs a default. Getting
    this wrong raises TypeError on turn one and forfeits.
    """
    agent = _top_level_functions(_module())[-1]
    positional = agent.args.posonlyargs + agent.args.args
    required = len(positional) - len(agent.args.defaults)
    assert positional, "`agent` takes no positional argument for the observation"
    assert required == 1, (
        f"`agent` requires {required} positional arguments; it must be callable "
        "as agent(obs) as well as agent(obs, config)"
    )


@needs_bundle
def test_the_bundle_declares_no_third_party_imports() -> None:
    """Kaggle runs the agent offline; only the stdlib is guaranteed present."""
    allowed = {
        "kaggle_environments",
        "json", "math", "random", "time", "dataclasses", "pathlib", "typing",
        "contextlib", "io", "os", "sys", "collections", "itertools", "functools",
        "__future__", "copy", "abc", "enum", "heapq", "bisect", "statistics",
    }
    roots: set[str] = set()
    for node in ast.walk(_module()):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    assert not (roots - allowed), f"third-party imports in the bundle: {sorted(roots - allowed)}"


@needs_bundle
@needs_archive
def test_archive_holds_exactly_main_py_at_the_top_level() -> None:
    """The fallback artefact, bound by filename instead of by position.

    A nested path (`submission/main.py`) would not be found at the top level, and
    a second member would make which file is the entry point ambiguous.
    """
    with zipfile.ZipFile(ARCHIVE) as zf:
        assert zf.namelist() == ["main.py"], (
            f"archive contains {zf.namelist()}, expected exactly ['main.py'] "
            "at the top level"
        )
        packed = zf.read("main.py")
    assert packed == MAIN.read_bytes(), (
        "kaggriculture.zip is stale - its main.py differs from submission/main.py. "
        "Rebuild with `python -m tools.build_submission`."
    )


@needs_bundle
def test_the_contract_check_actually_detects_a_violation(tmp_path, monkeypatch) -> None:
    """Prove the gate can fail, not just that it happens to pass.

    A guard that has never been seen to reject anything is indistinguishable from
    one that always returns success, so append a trailing `def` to a copy of the
    real bundle and require the checker to catch it. Without this, a checker
    broken into vacuous truth would sail through every other test here.
    """
    import tools.verify_submission as verifier

    assert verifier.check_entry_point_contract() == [], "the real bundle should be clean"

    mutated = tmp_path / "main.py"
    mutated.write_bytes(
        MAIN.read_bytes() + b"\n\ndef sneaky_helper(x: int) -> int:\n    return x\n"
    )
    monkeypatch.setattr(verifier, "MAIN", mutated)
    problems = verifier.check_entry_point_contract()
    assert problems, "a trailing def was not detected"
    assert "sneaky_helper" in problems[0]


@needs_bundle
def test_shipped_policy_is_the_attributed_competitive_meta_agent() -> None:
    """The bundle must be the licensed policy measured by competitive screens."""

    text = MAIN.read_text(encoding="utf-8")
    tree = ast.parse(text)
    assert "Copyright (c) 2026 Jonathan Day" in text
    assert any(
        isinstance(node, ast.FunctionDef) and node.name == "_upstream_agent"
        for node in tree.body
    ), "competitive meta implementation is missing"
    assert any(
        isinstance(node, ast.FunctionDef) and node.name == "_linear_sum_assignment"
        for node in tree.body
    ), "standard-library global matcher is missing"
    assert not any(
        isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "ReactiveRouteAgent"
        for node in ast.walk(tree)
    ), "the retired reactive route is still in the submitted path"
    assert not any(
        isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "MCTSAgent"
        for node in ast.walk(tree)
    ), "research MCTS must not be present in the submitted path"
