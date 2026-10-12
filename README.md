<<<<<<< HEAD
# Kaggriculture agent research

An archived, reproducible research codebase for the Kaggle
[Kaggriculture](https://www.kaggle.com/competitions/kaggriculture) two-player
farm-economy environment. The project contains local agents, a simulator-backed
evaluation harness, submission packaging checks, and an evidence-driven
promotion gate.

> **Project status — archived.** The competition timeline recorded in this
> repository ended on **30 September 2026, 23:59 UTC**. The code is useful for
> study and local experimentation, but it does not imply an active Kaggle
> submission window or guarantee a leaderboard rating. Confirm the live
> competition page before attempting to submit anything.

## What is here

The source deliberately keeps three concerns separate:

- **Agent implementations** under `src/`, from simple controls through the
  observation-driven `competitive_meta` policy.
- **Local evaluation and diagnostics** under `src/runner.py` and `tools/`.
  These use the pinned Kaggle environment to test legality, ordered actions,
  timing, final liquidation, and both player seats.
- **Self-contained submission artifacts** under `submission/`. They use the
  standard library at runtime and expose `agent(obs, config=None)`.

The present canonical artifact, [`submission/main.py`](submission/main.py), is
the MIT-attributed `competitive_meta` heuristic. It is intentionally preserved
as the historical control that was uploaded to the ladder (reported around the
600 range). [`submission/tape/main.py`](submission/tape/main.py) is a separate
Apache-2.0-attributed research candidate. A promising local result or a higher
final bank is not enough to promote either policy: Kaggriculture rates match
outcomes, not coin margin, and the required held-out league gate was not yet
completed.

## Quick start

### Prerequisites

- Python 3.12 or newer
- A virtual environment
- `kaggle-environments==1.32.7` for local simulation

```powershell
git clone <your-github-url>
cd K_argiculture_kaggle_one

py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Run the complete local test suite.
.\.venv\Scripts\python.exe -m pytest -q tests

# Run a single, deterministic episode in both seats.
.\.venv\Scripts\python.exe -m src.evaluate `
  --agent competitive_meta --opponent starter --episodes 1 --seats both
```

The project is developed on Windows, but the runtime code is cross-platform.
Use your virtual environment's Python explicitly rather than a globally
installed interpreter.

## Repository layout

```text
src/
  agents/              policies and research controls
  diagnostics.py       exact, in-memory action telemetry
  runner.py            official-environment episode harness
  evaluate.py          paired, both-seat evaluation
  simulator.py         forward-model and fidelity support
tools/
  build_submission.py  reproducible artifact builder
  verify_submission.py Kaggle entry-point and offline safety checks
  score_gate.py        resumable paired promotion evaluation
  fetch_public_agents.py optional public-agent benchmark fetcher
tests/                 unit, environment-fidelity, packaging, and statistics tests
benchmarks/
  README.md            public benchmark provenance and policy
  public/tetsutani_dp/ reviewed Apache-2.0 tape chassis used by the optional T0 artifact
submission/
  main.py              canonical historical heuristic artifact
  kaggriculture.zip    same artifact, packaged with main.py at archive root
  tape/                separately attributed Apache-2.0 research artifact
docs/                  environment reference, experiment logs, and historical runbooks
```

`benchmarks/public/` is otherwise a local cache and is ignored by Git. It may
contain downloaded public agents or a nested Git repository; none of that cache
is required to run the core suite, and it is not redistributed by this project.

## Reproducible workflow

### 1. Make and test an agent change

Keep policy code in `src/` and test against the real interpreter first:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_agents.py tests\test_fidelity.py
.\.venv\Scripts\python.exe -m src.evaluate `
  --agent competitive_meta --opponent starter --episodes 16 --seats both
```

Use the same seeds and both player seats when comparing policies. A single
episode is a smoke test, not an economic or competitive conclusion.

### 2. Diagnose behaviour, not just final bank

```powershell
.\.venv\Scripts\python.exe -m tools.diagnose --agent competitive_meta --seed 1
```

`src.runner.EpisodeTelemetry` records successful and no-op unit operations,
movement, feed/water/care completion, market activity, farm composition, and
final unsold value in memory. It does not read or write Kaggle replay files.

### 3. Compare a candidate with a control

`tools.score_gate` is intentionally conservative. It loads new agent modules
for each game, pairs the same seeds in both seats, checkpoints only its JSON
report, and rejects a promotion if the held-out paired confidence interval does
not support it.

```powershell
.\.venv\Scripts\python.exe -m tools.score_gate `
  --candidate candidate=submission/tape/main.py `
  --control control=submission/main.py `
  --opponent tetsutani=benchmarks/public/tetsutani_dp/main.py `
  --seed-start 5000 --seeds 8 --workers 2 `
  --out logs\screen.json
```

The `logs/` directory is ignored because results are machine- and experiment-
specific. Record accepted or rejected decisions in `docs/` with the exact
agent hashes and seed range.

### 4. Build and verify a submission artifact

Do not hand-edit files in `submission/`. Build and verify instead:

```powershell
# Rebuild the checked-in historical heuristic and archive.
.\.venv\Scripts\python.exe -m tools.build_submission --target meta

# Validate the canonical historical artifact.
.\.venv\Scripts\python.exe -m tools.verify_submission --target meta

# Rebuild and validate the separately attributed tape artifact, if desired.
.\.venv\Scripts\python.exe -m tools.build_submission --target tape
.\.venv\Scripts\python.exe -m tools.verify_submission --target tape
```

The `primary` builder target is fail-closed: it will not replace
`submission/main.py` until it finds matching held-out evaluation evidence and a
verification certificate. After promotion, validate it with
`python -m tools.verify_submission --target primary`. The separate `tape`
target is a research artifact and does not replace `submission/main.py`.

## Evaluation principles

Kaggriculture uses the final bank to decide each match, but ladder and final
ranking reward wins, losses, and ties rather than the size of the coin margin.
For that reason this codebase uses:

1. identical seeds for candidate and control;
2. both player seats for every seed;
3. fresh agent instances per game;
4. outcome score (win = 1, tie = 0.5, loss = 0), alongside bank;
5. seed-clustered bootstrap intervals; and
6. no unexplained agent fallback/error counters before promotion.

See [`docs/COMPETITION.md`](docs/COMPETITION.md) for the historical rules
record and [`docs/REBUILD_TODO.md`](docs/REBUILD_TODO.md) for the uncompleted
research checklist.

## Licence and attribution

The original project code and documentation are available under the
[MIT License](LICENSE). The release includes derived MIT and Apache-2.0 source
where necessary for reproducibility; their notices remain in the corresponding
files. Read [`LICENSES/THIRD_PARTY.md`](LICENSES/THIRD_PARTY.md) before copying,
modifying, or redistributing any of those artifacts.

No Kaggle credentials, public replay files, or private data are included. The
ignore rules cover virtual environments, local logs, cache directories, and
common credential files.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md). In particular, preserve
attribution, keep submissions standard-library-only, and do not describe an
unvalidated local result as a live rating improvement.
=======
# K-agriculture_Kaggle_competition
>>>>>>> f8c9392d9f87fe6ec00d2a560b38e887a9d2f7fb
