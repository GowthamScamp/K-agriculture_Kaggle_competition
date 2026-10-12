"""Prove `submission/main.py` is actually submittable.

    python -m tools.verify_submission
    python -m tools.verify_submission --skip-env-doc   # skip the slow drift check

Checks, in the order that matters:

0. **`agent` is the LAST `def` in the file.** Kaggle binds a bare `.py` upload
   by position, not by name, so a helper defined below the entry point would
   silently take its place. Checked statically, before anything else runs.
1. It imports as a standalone file, with `src/` deliberately off `sys.path` - so
   a stray dependency on the development tree fails here rather than on Kaggle.
2. It exposes a module-level `agent`.
3. It imports nothing outside the standard library and `kaggle_environments`,
   which is what Kaggle's offline execution rules allow.
4. **It is reproducible from `src/`.** Without this, a hand-edit to `main.py`
   passes every other gate and then vanishes on the next rebuild - the bundle
   would be verified but unrepeatable.
5. **It still plays with `kaggle_environments` unimportable**, which is the
   claim that offline execution rests on (delegated to `tools/offline_check.py`).
6. **It survives a self-play validation episode** - the agent against a copy of
   itself, which is exactly what Kaggle runs before admitting a submission to
   the ladder, and the only configuration where both seats run the submitted policy.
7. It plays a full 720-turn episode against a real opponent and wins.
8. No single turn exceeds the 1-second `actTimeout`, in either configuration.

Any failure is reported as a blocker, because each one costs the whole episode.

Two things are reported as warnings instead, because neither makes the current
bundle unsubmittable: a turn that clears the `actTimeout` but eats more than
`SAFE_TURN_FRACTION` of it (a bet on Kaggle's hardware being no slower than
this box), and the state of `submission/kaggriculture.zip`, the archive
fallback that binds by filename rather than by position.

Separately and non-blocking, it reports whether `docs/ENVIRONMENT.md` still
matches the installed package. Upstream schema drift does not make the current
bundle unsubmittable, but it does mean the documented source of truth has gone
stale, so it is surfaced as a warning rather than swallowed.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "submission" / "main.py"
ARCHIVE = ROOT / "submission" / "kaggriculture.zip"
ENV_DOC = ROOT / "docs" / "ENVIRONMENT.md"
ACT_TIMEOUT_S = 1.0
#: Warn above this fraction of the actTimeout. A turn that fits in 436 ms on a
#: 22-core desktop is not guaranteed to fit on Kaggle's runner, so the margin is
#: reported rather than left implicit. Not a blocker - it is a judgement about
#: hardware we cannot measure, not a rule the submission breaks.
SAFE_TURN_FRACTION = 0.70
#: Fixed seed for the self-play validation episode, so the gate is repeatable.
SELF_PLAY_SEED = 7

#: Everything Kaggle's offline notebook image is guaranteed to provide.
ALLOWED_ROOTS = {
    "kaggle_environments",
    # stdlib used by the agent
    "json", "math", "random", "time", "dataclasses", "pathlib", "typing",
    "contextlib", "io", "os", "sys", "collections", "itertools", "functools",
    "__future__", "copy", "abc", "enum", "heapq", "bisect", "statistics",
    "base64", "zlib",
}
TARGET = "meta"

#: The one line of the bundle that legitimately changes on every rebuild.
_DATE_STAMP = "Built from src/ by"


def _margin_note(where: str, worst: float) -> str:
    """Phrase a near-the-limit turn as the hardware bet it actually is."""
    return (
        f"{where} slowest turn {worst * 1000:.0f} ms is over "
        f"{SAFE_TURN_FRACTION:.0%} of the {ACT_TIMEOUT_S * 1000:.0f} ms actTimeout. "
        "It passes here, but Kaggle's runner is not this machine - a slower box "
        "could push it over. Lower `budget_ms` in the agent if this creeps up."
    )


def check_entry_point_contract() -> list[str]:
    """Kaggle binds a bare `.py` upload to **the last `def` in the file**.

    From the submission dialog: *"Your submission should be a python file with
    the last 'def' accepting an observation and returning an action."* So the
    entry point is chosen by **position**, not by name - and `hasattr(module,
    "agent")` cannot detect a violation, because a bundle with `agent` defined
    halfway up and a helper below it passes that check and still binds the
    wrong function on Kaggle.

    This bundle inlines every module into one file, so the invariant holds only
    because `build_submission` appends its footer last. Nothing else enforces
    it. Adding a module to `MODULES` after the footer, or a helper below
    `agent`, would silently forfeit every episode. Hence a blocker.

    Static only - the file is parsed, never executed, so this stays valid even
    if importing were to fail for an unrelated reason.
    """
    problems: list[str] = []
    tree = ast.parse(MAIN.read_text(encoding="utf-8"))

    funcs = [n for n in tree.body
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not funcs:
        return ["bundle defines no top-level functions at all"]

    last_def = funcs[-1]
    if last_def.name != "agent":
        problems.append(
            f"the last top-level def is `{last_def.name}` (line {last_def.lineno}), "
            "not `agent` - Kaggle binds the LAST def, so it would run the wrong "
            "function; move the entry point to the end of the file"
        )

    # Nothing may trail the entry point either. A trailing `class` would not
    # change the binding, but it means the footer is no longer last, which is
    # the precondition the whole invariant rests on.
    if tree.body and tree.body[-1] is not last_def:
        trailing = type(tree.body[-1]).__name__
        problems.append(
            f"a top-level {trailing} at line {tree.body[-1].lineno} comes after "
            f"`{last_def.name}` - the entry point must be the final statement"
        )

    # Kaggle may call the entry point as `agent(obs)`; kaggle_environments calls
    # it as `agent(obs, config)`. Both must work, so every parameter past the
    # first needs a default.
    args = last_def.args
    positional = args.posonlyargs + args.args
    if not positional:
        problems.append(f"`{last_def.name}` takes no positional argument for the observation")
    elif len(positional) - len(args.defaults) > 1:
        required = len(positional) - len(args.defaults)
        problems.append(
            f"`{last_def.name}` requires {required} positional arguments; Kaggle may "
            "call it as agent(obs), so all but the first need defaults"
        )
    return problems


def check_kaggle_binding() -> list[str]:
    """Bind the entry point exactly as kaggle_environments does.

    `kaggle_environments.agent.get_last_callable` execs the file and takes the
    last *callable value inserted* into the namespace, not the last `def` by
    position: a redefined name keeps its original slot, and any callable
    assigned after `agent` wins. The static check above cannot see that.
    """
    from kaggle_environments.agent import get_last_callable

    raw = MAIN.read_text(encoding="utf-8")
    bound = get_last_callable(raw, path=str(MAIN))
    name = getattr(bound, "__name__", type(bound).__name__)
    line = getattr(getattr(bound, "__code__", None), "co_firstlineno", None)
    last_def = [n for n in ast.parse(raw).body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))][-1]
    if name != "agent" or line != last_def.lineno:
        return [f"Kaggle would bind `{name}` (line {line}), not the final `agent` "
                f"at line {last_def.lineno}"]
    return []


def check_archive() -> tuple[list[str], str]:
    """The zip is the fallback upload, and binds by filename instead of position.

    An archive is resolved by looking for `main.py` at the top level, which is
    immune to the last-def rule above. Kept in step with the `.py` byte for
    byte so the two artifacts can never disagree about what was verified.

    Non-blocking: a missing or stale zip does not make `main.py` unsubmittable.
    """
    if not ARCHIVE.exists():
        return ([f"{ARCHIVE.name} missing; run `python -m tools.build_submission`"],
                "not built")
    with zipfile.ZipFile(ARCHIVE) as zf:
        names = zf.namelist()
        if names != ["main.py"]:
            return ([f"{ARCHIVE.name} contains {names}, expected exactly ['main.py']"],
                    f"WRONG CONTENTS: {names}")
        packed = zf.read("main.py")
    if packed != MAIN.read_bytes():
        return ([f"{ARCHIVE.name} does not match submission/main.py - rebuild"],
                "STALE - does not match main.py")
    size_kb = ARCHIVE.stat().st_size / 1024
    return ([], f"main.py at top level, matches, {size_kb:.0f} KB")


def check_imports() -> list[str]:
    """Static scan: no third-party dependency may sneak into the bundle."""
    tree = ast.parse(MAIN.read_text(encoding="utf-8"))
    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                problems.append(f"line {node.lineno}: relative import survived the build")
                continue
            names = [node.module or ""]
        else:
            continue
        for name in names:
            root = name.split(".")[0]
            if root and root not in ALLOWED_ROOTS:
                problems.append(f"line {node.lineno}: imports {name!r}, not known-available")
    return problems


def _body(text: str) -> list[str]:
    """The bundle's lines, minus the build-date stamp that changes each rebuild."""
    return [ln for ln in text.splitlines() if _DATE_STAMP not in ln]


def check_reproducible() -> list[str]:
    """The committed bundle must be exactly what `build_submission` emits today.

    Rebuilt in memory rather than on disk, so this check cannot itself repair the
    drift it is looking for.
    """
    from tools.build_submission import render_submission, render_tape_submission

    if TARGET in ("tape", "primary"):
        if render_tape_submission() == MAIN.read_bytes():
            return []
        return ["tape artifact differs from the pinned chassis + wrapper - "
                "rebuild with `python -m tools.build_submission --target tape`"]

    expected = _body(render_submission())
    actual = _body(MAIN.read_text(encoding="utf-8"))
    if expected == actual:
        return []

    hint = "rebuild with `python -m tools.build_submission`"
    for i, (a, b) in enumerate(zip(actual, expected), 1):
        if a != b:
            return [f"main.py has drifted from src/ at line {i} - {hint}"]
    return [f"main.py has drifted from src/ ({len(actual)} lines on disk vs "
            f"{len(expected)} rebuilt) - {hint}"]


def check_env_doc() -> list[str]:
    """Non-blocking: has the installed environment moved out from under the doc?"""
    if not ENV_DOC.exists():
        return ["docs/ENVIRONMENT.md is missing"]
    with tempfile.TemporaryDirectory() as td:
        fresh = Path(td) / "ENVIRONMENT.md"
        proc = subprocess.run(
            [sys.executable, "-m", "tools.discover_env", "--out", str(fresh), "--quiet"],
            cwd=str(ROOT), capture_output=True, text=True,
        )
        if proc.returncode != 0 or not fresh.exists():
            tail = (proc.stderr or "").strip().splitlines()
            return ["could not regenerate ENVIRONMENT.md" + (f": {tail[-1]}" if tail else "")]
        new_lines = fresh.read_text(encoding="utf-8").splitlines()
    old_lines = ENV_DOC.read_text(encoding="utf-8").splitlines()
    if new_lines == old_lines:
        return []
    changed = sum(1 for a, b in zip(old_lines, new_lines) if a != b)
    changed += abs(len(old_lines) - len(new_lines))
    return [f"docs/ENVIRONMENT.md is stale ({changed} line(s) differ from the "
            f"installed package) - regenerate with `python -m tools.discover_env`"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify the submission bundle.")
    ap.add_argument("--skip-env-doc", action="store_true",
                    help="skip the schema-drift check, which replays a full episode")
    ap.add_argument("--target", choices=("meta", "primary", "tape"), default="meta",
                    help="meta: historical heuristic at submission/main.py; "
                         "primary: promoted tape at submission/main.py; "
                         "tape: separate research artifact")
    args = ap.parse_args(argv)

    global MAIN, ARCHIVE, TARGET
    TARGET = args.target
    if TARGET == "tape":
        from tools.build_submission import TAPE_ARCHIVE, TAPE_OUT
        MAIN, ARCHIVE = TAPE_OUT, TAPE_ARCHIVE

    problems: list[str] = []
    warnings: list[str] = []
    print(f"\n  verifying {MAIN.relative_to(ROOT)}")
    print("  " + "-" * 62)

    if not MAIN.exists():
        print("  BLOCKER: file does not exist; run `python -m tools.build_submission`")
        return 1
    size_kb = MAIN.stat().st_size / 1024
    print(f"  size                    {size_kb:.0f} KB")

    # Before anything that executes the bundle: Kaggle binds a bare `.py` to the
    # last def in the file, so getting this wrong runs the wrong function no
    # matter how well everything below passes.
    entry_problems = check_entry_point_contract()
    if entry_problems:
        problems.extend(entry_problems)
        print("  entry-point contract    BLOCKER")
        for p in entry_problems:
            print(f"    {p}")
    else:
        print("  entry-point contract    `agent` is the last def, callable as agent(obs)")

    binding_problems = check_kaggle_binding()
    if binding_problems:
        problems.extend(binding_problems)
        print("  kaggle binding          BLOCKER")
        for p in binding_problems:
            print(f"    {p}")
    else:
        print("  kaggle binding          get_last_callable -> the final `agent`")

    archive_problems, archive_line = check_archive()
    warnings.extend(archive_problems)
    print(f"  archive fallback        {archive_line}")

    import_problems = check_imports()
    if import_problems:
        problems.extend(import_problems)
        print(f"  imports                 {len(import_problems)} PROBLEM(S)")
        for p in import_problems[:5]:
            print(f"    {p}")
    else:
        print("  imports                 stdlib + kaggle_environments only")

    repro_problems = check_reproducible()
    if repro_problems:
        problems.extend(repro_problems)
        print("  reproducible from src/  NO - see blockers below")
    else:
        print("  reproducible from src/  yes, byte-identical apart from the date stamp")

    offline = subprocess.run(
        [sys.executable, "-m", "tools.offline_check", "--main", str(MAIN)],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    if offline.returncode != 0:
        problems.append("bundle does not run without kaggle_environments "
                        "(see `python -m tools.offline_check`)")
        print("  offline safety          BLOCKER: needs kaggle_environments")
    else:
        print("  offline safety          plays with kaggle_environments blocked")

    self_play_problems, self_play_warnings, self_play_line = check_self_play()
    problems.extend(self_play_problems)
    warnings.extend(self_play_warnings)
    print(f"  self-play validation    {self_play_line}")

    module = load_isolated()
    if not hasattr(module, "agent"):
        problems.append("no module-level `agent` function")
        print("  entry point             BLOCKER: no `agent`")
        return _report(problems, warnings)
    print("  entry point             agent() found")

    # -- full episode against a real opponent ------------------------------
    sys.path.insert(0, str(ROOT))
    from src.runner import _quiet, make_env

    env = make_env(seed=3)
    times: list[float] = []

    def timed(obs: Any, config: Any = None) -> dict[str, Any]:
        t0 = time.perf_counter()
        action = module.agent(obs, config)
        times.append(time.perf_counter() - t0)
        return action

    with _quiet():
        env.run([timed, "starter"])

    final = env.steps[-1]
    bank, opp = float(final[0]["reward"] or 0), float(final[1]["reward"] or 0)
    worst = max(times)

    print(f"  episode                 {len(env.steps)} steps, status {final[0]['status']}")
    print(f"  final bank              {bank:,.0f}   (opponent {opp:,.0f})")
    print(f"  turn time               mean {sum(times) / len(times) * 1000:.1f} ms, "
          f"max {worst * 1000:.0f} ms")

    if len(env.steps) != 720:
        problems.append(f"episode ran {len(env.steps)} steps, expected 720")
    if final[0]["status"] != "DONE":
        problems.append(f"agent finished with status {final[0]['status']}")
    if worst >= ACT_TIMEOUT_S:
        problems.append(f"slowest turn {worst * 1000:.0f} ms exceeds the 1000 ms actTimeout")
    elif worst >= ACT_TIMEOUT_S * SAFE_TURN_FRACTION:
        warnings.append(_margin_note("episode", worst))
    if bank <= opp:
        problems.append(f"lost to the built-in starter agent ({bank:,.0f} vs {opp:,.0f})")

    if args.skip_env_doc:
        print("  environment doc         skipped")
    else:
        doc_problems = check_env_doc()
        warnings.extend(doc_problems)
        print("  environment doc         "
              + ("STALE - see warnings" if doc_problems else "matches the installed package"))

    return _report(problems, warnings)


def load_isolated(name: str = "kaggriculture_submission") -> Any:
    """Import main.py with the project's own src/ hidden from sys.path.

    `name` exists so the bundle can be loaded more than once. Each load gets its
    own module object and therefore its own agent singleton, which is what makes
    the self-play check below a genuine two-player game rather than one agent
    playing both seats out of a shared cache.
    """
    saved_path = list(sys.path)
    saved_mods = {k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")}
    for k in saved_mods:
        del sys.modules[k]
    sys.path = [p for p in sys.path if Path(p).resolve() != ROOT] if sys.path else []
    try:
        spec = importlib.util.spec_from_file_location(name, MAIN)
        module = importlib.util.module_from_spec(spec)
        # `@dataclass(slots=True)` resolves annotations through
        # `sys.modules[cls.__module__]`, so the module must be registered before
        # it executes or every dataclass in the file raises.
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path = saved_path
        sys.modules.update(saved_mods)


def check_self_play() -> tuple[list[str], str]:
    """Kaggle's validation episode: the agent against a copy of itself.

    On upload, Kaggle runs exactly this before the agent is allowed onto the
    ladder, and a failure rejects the submission outright - so it is a blocker
    here rather than a warning.

    It is also the only configuration in which *both* seats run the submitted
    policy. Every other gate pairs the agent against a cheap built-in, so this is
    the more representative load test.
    """
    sys.path.insert(0, str(ROOT))
    from src.runner import _quiet, make_env

    seat_a = load_isolated("kaggriculture_selfplay_a")
    seat_b = load_isolated("kaggriculture_selfplay_b")
    times: list[list[float]] = [[], []]

    def timed(module: Any, sink: list[float]) -> Any:
        def play(obs: Any, config: Any = None) -> dict[str, Any]:
            t0 = time.perf_counter()
            try:
                return module.agent(obs, config)
            finally:
                sink.append(time.perf_counter() - t0)
        return play

    env = make_env(seed=SELF_PLAY_SEED)
    with _quiet():
        env.run([timed(seat_a, times[0]), timed(seat_b, times[1])])

    final = env.steps[-1]
    if not any(times):
        return (["self-play episode produced no turns at all"],
                "BLOCKER: no turns were played")
    # A seat that crashed on turn one has an empty list; default it rather than
    # letting max() raise and lose the diagnosis.
    worst = max(max(t, default=0.0) for t in times)
    banks = [float(final[i]["reward"] or 0) for i in range(2)]
    line = (f"720 steps OK, both seats played, "
            f"max turn {worst * 1000:.0f} ms")

    problems: list[str] = []
    if len(env.steps) != 720:
        problems.append(f"self-play episode ran {len(env.steps)} steps, expected 720")
        line = f"BLOCKER: ran {len(env.steps)} steps"
    for i in range(2):
        if final[i]["status"] != "DONE":
            problems.append(
                f"self-play seat {i} finished with status {final[i]['status']}"
            )
            line = f"BLOCKER: seat {i} status {final[i]['status']}"
    warnings: list[str] = []
    if worst >= ACT_TIMEOUT_S:
        problems.append(
            f"self-play slowest turn {worst * 1000:.0f} ms exceeds the "
            f"{ACT_TIMEOUT_S * 1000:.0f} ms actTimeout - Kaggle would reject this"
        )
        line = f"BLOCKER: max turn {worst * 1000:.0f} ms"
    elif worst >= ACT_TIMEOUT_S * SAFE_TURN_FRACTION:
        warnings.append(_margin_note("self-play", worst))
    if not problems:
        line += f", banks {banks[0]:,.0f} / {banks[1]:,.0f}"
    return problems, warnings, line


def _report(problems: list[str], warnings: list[str]) -> int:
    print()
    if warnings:
        print(f"  {len(warnings)} warning(s) - not submit blockers:")
        for w in warnings:
            print(f"    - {w}")
        print()
    if problems:
        print(f"  NOT SUBMITTABLE - {len(problems)} blocker(s):")
        for p in problems:
            print(f"    - {p}")
        print()
        return 1
    main_rel = MAIN.relative_to(ROOT).as_posix()
    print("  READY TO SUBMIT")
    print(f"  Browser:  drag {main_rel} onto the File Upload tab")
    print("            (docs/SUBMISSION.md walks through the dialog)")
    print(f"  Fallback: {ARCHIVE.relative_to(ROOT).as_posix()} - bound by filename rather")
    print("            than by last-def, if the .py entry point is ever missed")
    print("  CLI:      kaggle competitions submit kaggriculture "
          f"-f {main_rel} -m \"<message>\"")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
