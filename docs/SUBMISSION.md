# SUBMISSION.md — uploading to Kaggle, step by step

> **Archived on 7 October 2026.** The instructions below record the project’s
> final competition workflow and are not an instruction to upload now. The
> repository does not upload or manage Kaggle submissions. Confirm the live
> competition state and rules before using any artifact.

Everything here is done in the browser. Nothing in this repo uploads anything;
there are no Kaggle credentials on this machine and no tool here will acquire
any. The rules this follows are recorded in
[COMPETITION.md §2](COMPETITION.md#2-submitting).

---

## Final-week plan (deadline 30 Sep 2026, 23:59 UTC)

Two artifacts, one per tracked slot. Only the **latest two** uploads count, so
order matters.

| Slot | File | What it is | Verify with |
|---|---|---|---|
| 1 (score) | `submission/tape/main.py` | Apache-2.0 `tetsutani_dp` tape chassis + fallback wrapper | `-m tools.verify_submission --target tape` |
| 2 (own work) | `submission/main.py` | our `competitive_meta` policy | `-m tools.verify_submission` |

Upload sequence:

1. **T0 now:** `submission/tape/main.py`. Description:
   `T0 tetsutani_dp Apache-2.0 chassis + fallback wrapper | 16-0 vs farm2945 locally`
2. **30 Sep, by 18:00 UTC:** `submission/main.py` (H1). Description:
   `H1 competitive_meta + sells-first | 24-0 vs prior, 24-0 vs lonespear v18`.
   The tracked pair becomes `[T0, H1]`. Upload nothing after it. (The only
   slot-1 edge tested, E3, measured 2-20-2 against T0, so there is no T1.)

Re-uploading an unchanged T0 would restart its rating from scratch, so it is
uploaded once.

---

## Before you upload — one command

```bash
cd Projects/K_argiculture_kaggle_one
./.venv/Scripts/python.exe -m tools.verify_submission
```

**Use `./.venv/Scripts/python.exe`, not bare `python`.** On this machine `python`
resolves to a hermes venv that has no `kaggle_environments`, and seven tests fail
for that reason alone — a misleading red that has nothing to do with the agent.

Wait for the last line to read **`READY TO SUBMIT`**. If it does not, stop: it
prints its blockers, and every one of them costs the whole episode on Kaggle.
The check that matters most is `entry-point contract`, explained below.

If the bundle is stale or missing, rebuild first:

```bash
./.venv/Scripts/python.exe -m tools.build_submission
```

---

## Which file to upload

| File | Size | When |
|---|---|---|
| **`submission/main.py`** | 53 KB | **Default. Upload this one.** |
| `submission/kaggriculture.zip` | 17 KB | Fallback — see below |

Both contain identical bytes. They differ only in how Kaggle locates the entry
point, and that difference is the reason both exist:

- A bare **`.py`** is bound to **the last `def` in the file**. Our bundle has ~45
  top-level `def`s and `agent` is last — verified as a blocker on every run, and
  guarded by a test that appends a trailing `def` and requires the check to fail.
- The **`.zip`** is bound to **`main.py` at the top level**, by name. Position
  cannot mislead it.

So: upload the `.py`. Reach for the `.zip` if Kaggle ever rejects the `.py` for
anything resembling a "could not find agent" reason — that is the one failure the
archive is immune to by construction.

---

## The upload dialog

On the competition page, **Submit Agent** opens a panel with three tabs.

**1. Stay on the `File Upload` tab.** The other two are not used here — `Notebook`
would move the agent into a Kaggle notebook where it can drift from `src/`, and
`MCP` is a programmatic route that needs credentials this machine does not have.

**2. Check the quota line** at the top: *"You have N submissions remaining today.
This resets in H hours."* It is a rolling timer, not midnight. Read it **before**
uploading, because of the tracking trap in step 5.

**3. Drag `submission/main.py` onto the drop zone**, or use `Browse Files`. On
Windows the full path is:

```
<repository-root>\submission\main.py
```

Accepted formats are `.py`, `.zip`, `.gz`, `.7z`. Ours is a plain `.py`.

**4. Write a description that identifies the build.** This field is the only
record of *which* agent a leaderboard row came from, and with only two
submissions tracked you will want to tell them apart later. Something like:

```
competitive meta v8 | attributed MIT heuristic + pure-Python global assignment | 16-0 vs prior route | verifier ready
```

If you change anything about the agent, change this text too. A description that
does not distinguish two uploads makes the leaderboard unreadable.

**5. Submit.** Then read this before you upload a second time.

---

## The tracking trap

> Only the **latest 2** submissions are tracked, and those same two are used for
> the final leaderboard evaluation.

**"Latest", not "best".** If you upload a strong agent and then two experiments,
the strong one is pushed out of the tracked pair even if it beats both. There is
no "keep my best" behaviour to fall back on.

Practical rule: with 5 uploads a day and 2 tracked slots, treat every upload after
a good result as spending something. Do not experiment close to the
**30 September 2026, 23:59 UTC** final submission deadline.

---

## What Kaggle does next

1. **Validation episode** — your agent plays *against a copy of itself*. A crash
   or a timeout here rejects the submission outright.

   This is gated locally, so it should not surprise you. `check_self_play()` runs
   exactly this configuration — the bundle loaded twice under different module
   names, so each seat gets its own agent instance, as two Kaggle processes
   would. Last run: **720 steps, both seats DONE, max turn 137 ms** against a
   1,000 ms limit, with final banks 41,966 and 41,966. Both seats run the exact
   submitted policy, so this is the honest load test.

2. **Ladder play** — your agent is matched against others continuously. Rating
   moves on **win/loss/tie only**; the coin margin is explicitly ignored. An
   agent that wins narrowly and often outranks one that wins hugely and rarely.

3. **Final standings** — a Bradley-Terry tournament over the collected episodes,
   not the running rating.

Expect your rating to be noisy early. Individual episodes prove very little:
locally, on seed 7 alone, this agent *loses* to its own baseline.

---

## If something goes wrong

| Symptom | Likely cause | Do this |
|---|---|---|
| Rejected at validation | crash or timeout in self-play | run `tools.verify_submission` — its self-play check is the same episode |
| "Agent not found" / wrong behaviour | last-`def` binding | upload `submission/kaggriculture.zip` instead; it binds by filename |
| Timeouts on Kaggle but not locally | their runner is slower than this box | profile and simplify the route hot path, rebuild, re-verify |
| Import error offline | a third-party import reached the bundle | `tools.verify_submission` blocks on this; rebuild from `src/` |

The local gate warns whenever a turn exceeds **70% of the 1 s limit**, precisely
because clearing the limit on a desktop is not the same as clearing it on
Kaggle's hardware. The latest self-play worst is 137 ms; the competitive-pool
worst observed for this policy is 185 ms, still more than 5x headroom here.

---

## What this agent is, in one line

An observation-driven livestock/strawberry economy derived from Jonathan Day's
MIT-licensed `lonespear` v18 agent, with a dependency-free Hungarian matcher,
overflow-safe shed deposits, same-turn liquidation, `step=None` handling and a
hard exception boundary. It uses live observations only; no replay action tape
is present in the submitted path.

Across seeds 0-7 in both seats it beat the previous `reactive_route` submission
**16-0**. Across seeds 0-11 in both seats it went **16-8** against the stronger
clean MIT reference, with combined mean bank about **64.9k versus 62.5k**. A
replay-router upper-bound still won its checked match, so these are improvement
measurements, not a promise of a particular Kaggle rating. The final verifier
reported `READY TO SUBMIT`, a starter bank of **123,149 versus 3,421**, and a
**3 ms** maximum turn in that episode. See
[competitive_meta_experiments.md](competitive_meta_experiments.md).
