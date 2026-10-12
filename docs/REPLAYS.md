# Using real public replay data

The build currently trains and evaluates against **self-play plus the environment's
built-in agents** (`starter`, `random`) because no Kaggle API credentials are present
on this machine. Everything is wired so that real replays drop in without a rewrite:
`src/replay_loader.py` is the single seam.

## Auth: the official routes

`kaggle_environments/envs/kaggriculture/AGENTS.md` documents three ways to
authenticate, and the token file it describes is *not* the `kaggle.json` route
below - either works, but prefer the official one:

```bash
mkdir -p ~/.kaggle
# paste the token from https://www.kaggle.com/settings/api into this file
nano ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

Alternatives: `kaggle auth login` (browser OAuth), or `export KAGGLE_API_TOKEN=...`.

Verify with `kaggle competitions list -s kaggriculture`, and note that you must
accept the rules by clicking **Join Competition** on the competition page before
any of the download commands below will work.

## Pulling replays directly from the competition

Once authenticated, replays can come straight from the competition rather than a
third-party dataset - this is the authoritative source and includes your own
ladder games:

```bash
kaggle competitions submissions kaggriculture        # find your SUBMISSION_ID
kaggle competitions episodes <SUBMISSION_ID>         # list its episodes
kaggle competitions replay <EPISODE_ID> -p ./data/replays
kaggle competitions logs <EPISODE_ID> 0 -p ./logs    # your agent's stderr
```

`kaggle competitions logs` is the only way to see why an agent misbehaved on the
ladder, and is worth pulling after the first real submission.

## To enable real replays

1. Create a Kaggle API token: kaggle.com → your profile → Settings → API →
   *Create New Token*. This downloads `kaggle.json`.
2. Place it in your user Kaggle configuration directory (for example,
   `<home>/.kaggle/kaggle.json`).
3. Install and download:

   ```bash
   .venv/Scripts/python.exe -m pip install kaggle
   .venv/Scripts/kaggle.exe datasets download -d georgymamarin/kaggriculture-episodes -p data/replays --unzip
   ```

   The competition's own episode API is the other source:

   ```bash
   .venv/Scripts/kaggle.exe competitions leaderboard kaggriculture --download
   ```

4. Verify the loader picks them up:

   ```bash
   .venv/Scripts/python.exe -m src.replay_loader --stat
   ```

## What changes once they are present

- `src/determinization.py` samples opponent behaviour from real submissions instead of
  from the built-in agent pool. No call-site changes: `build_opponent_pool()` returns
  replay-derived policies automatically when `data/replays/` is non-empty.
- `tests/test_fidelity.py` gains a second, stronger case: replaying a *real* recorded
  episode through `simulate_step` and asserting the observations match. Until then it
  validates against self-generated episodes, which still catches simulator drift but
  cannot catch a mismatch between the installed env version and the one the ladder runs.

## Caveat worth knowing

Replays are recorded under whatever environment version was live at the time. If the
competition updates `kaggriculture.py`, old replays may no longer reproduce exactly.
`docs/ENVIRONMENT.md` records the version in use; check it before trusting a fidelity
number derived from downloaded episodes.
