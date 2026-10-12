# Contributing

This is an archived competition research project. Small, reproducible changes
are welcome if they improve correctness, observability, or local evaluation.

## Development setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q tests
```

Keep the submitted agent standard-library-only. `kaggle_environments` belongs
in the local test harness; it must never become a required import of
`submission/main.py`.

## Change rules

- Make changes under `src/` or `tools/`; do not hand-edit generated submission
  artifacts.
- Run the relevant focused tests, then the full suite before opening a pull
  request.
- Benchmark candidates in both seats and on fixed paired seeds. Do not claim a
  score improvement from a single game or from coin margin alone.
- Preserve upstream copyright and licence notices. Add a row to
  [`LICENSES/THIRD_PARTY.md`](LICENSES/THIRD_PARTY.md) before committing any
  redistributed third-party source.
- Never commit credentials, Kaggle replay data, environment directories, or
  files under `logs/`.

## Submission artifacts

Run `python -m tools.build_submission --target meta` to rebuild the historical
heuristic in `submission/main.py`, then `python -m tools.verify_submission --target meta`
to validate it. The primary promotion target intentionally fails closed until
its recorded evidence and verification gates exist; see `tools/build_submission.py`.
