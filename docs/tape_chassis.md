# Historical slot-1 tape-chassis record — 27 September 2026

> **Archived evidence.** The results and artifact hashes below refer to the
> September build. Treat them as experiment history, not a claim about the
> current working tree or an active Kaggle submission.

## Why

| Submission | Agent | Ladder |
|---|---|---:|
| 13-14 Sep | `reactive_route` | 605 / 607 |
| v7, v10 (20-21 Sep) | `competitive_meta` (lonespear v18 port) | 569 / 593 |

`competitive_meta` beat `reactive_route` 16-0 locally and moved the ladder by
zero. Public write-ups put hand-built heuristics at ~790-1,100. Everything above
~1,700 is a replay-tape agent from the public Apache-2.0 "V39" notebook lineage;
one team's byte-exact copy of `farm2945 v9/4` went 23-1 and passed ~2,000. The
owner chose a hybrid: slot 1 ships a licensed tape chassis, slot 2 keeps our own
policy.

We also verified that our 593 was not a wiring fault. `kaggle_environments`
binds the last *callable inserted* into the module namespace
(`agent.py:get_last_callable`), not the last `def`. Every current artifact binds
`agent` correctly under that rule.

## Candidates (A1)

Source: `github.com/warlord-2227/kaggle` @ `f8ddefb`, `refagents/public/`.
Provenance and hashes are in `benchmarks/README.md`.

| Candidate | Licence in file | Imports | Load | Max turn | vs starter, seed 0 |
|---|---|---|---:|---:|---|
| farm2945 | Apache-2.0 | stdlib | 0.18 s | 142 ms | 179,896 / 165,216 |
| tetsutani_dp | Apache-2.0 | stdlib | 0.39 s | 234 ms | 179,974 / 165,538 |
| shopwork | none (benchmark only) | stdlib | 0.02 s | 22 ms | 174,003 / 156,474 |
| public_state_router | none in file (notebook Apache-2.0) | stdlib | 0.04 s | 6 ms | 170,475 / 156,320 |

## Selection (A2)

Round-robin, seeds 0-7, both seats (`logs/tape_select_s0.json`,
`logs/tape_select_s4.json`); 160 games. Records are row vs column.

| | tetsutani_dp | farm2945 | router | shopwork | competitive_meta | Total |
|---|---|---|---|---|---|---|
| **tetsutani_dp** | - | 16-0 | 16-0 | 16-0 | 16-0 | **64-0** |
| **farm2945** | 0-16 | - | 16-0 | 16-0 | 16-0 | 48-16 |
| **router** | 0-16 | 0-16 | - | 16-0 | 16-0 | 32-32 |
| **shopwork** | 0-16 | 0-16 | 0-16 | - | 16-0 | 16-48 |
| **competitive_meta** | 0-16 | 0-16 | 0-16 | 0-16 | - | 0-64 |

Local Bradley-Terry: tetsutani_dp 2025.7, farm2945 1739.7, router 1500.0,
shopwork 1260.3, competitive_meta 974.3. Our shipped heuristic banked roughly
50-75k against 110-143k for every tape.

**Selected: `tetsutani_dp`** (`TAPE_SELECTED` in `tools/build_submission.py`).
The tape-vs-tape margins are narrow (farm2945 92,021 vs 94,916 on seeds 0-3)
but the result never varied across 16 games. This agrees with the public ladder
evidence (the tetsutani-based v26 reached ~2,140; farm2945 byte-exact ~2,000).

## Packaging (A3)

`python -m tools.build_submission --target tape` writes
`submission/tape/main.py` and `submission/tape/kaggriculture_tape.zip`:

1. the chassis bytes, verified against the pinned SHA-256;
2. an Apache §4(b) modification banner plus the fallback's MIT notice;
3. a wrapper that deletes the chassis `agent`, keeps it as
   `_KGW_CHASSIS_AGENT`, and defines the final `agent`. It calls the chassis and,
   only when an exception escapes, answers that turn with `competitive_meta`
   (zlib+base64, exec'd into its own namespace so no chassis global is shadowed).

Measured on the farm2945 build:

- Kaggle's own `get_last_callable` binds the wrapper `agent`.
- A forced chassis exception returns a full `competitive_meta` action; a double
  failure returns PASS.
- Seed 0 vs starter reproduces the raw chassis exactly: 179,896 / 165,216.
- `verify_submission --target tape`: READY TO SUBMIT; self-play 720 steps,
  max turn 392 ms under load, banks 92,970 / 92,970; starter episode 180,735
  vs 3,605.

Final build, `tetsutani_dp` (T0): `verify_submission --target tape` READY TO
SUBMIT; self-play 720 steps, max turn 106 ms, banks 91,246 / 91,246; starter
episode 181,132 vs 3,605, max turn 138 ms.

| Upload | File | SHA-256 |
|---|---|---|
| T0 (slot 1) | `submission/tape/main.py` | `61DB1C0A1A648A1E2E18697C46ED15F8146A44E90339A72FF6B2C459200BC3D7` |
| H1 (slot 2) | `submission/main.py` | `FB4CF00565FF76B363CC8590161534A2C1991BBA2721DEFAEFAAACED0A19551A` |

Tests: full suite 131 passed, plus the 18 artifact-contract tests re-run against these final builds.

## Edge layers (A4)

- **E1, sells-first market ordering:** already in the chassis
  (`_v224_sales_first`, active from step 144/288). Not re-implemented.
- **E2, observed front-run:** the chassis ships `front_run` disabled because no
  `opponent_plan` is passed. Feeding it our own route tape (clone assumption)
  would largely duplicate the existing `sell_lead` layer, which already sells the
  next step's planned lots one step early, and `_front_run` skips items already
  being sold that step. Not pursued.
- **E3, sells-first from step 0 (chassis applies it from step 144): REJECTED.**
  T0+E3 vs T0, seeds 0-11 both seats: **2W-20T-2L**, mean banks identical to the
  coin (90,268 / 90,268 on seeds 0-5; 84,822 / 84,822 on 6-11). The tapes barely
  sell before day 6, so the edge has nothing to act on. Logs:
  `logs/t1_e3_gate_s0.json`, `logs/t1_e3_gate_s6.json`.

**No edge passed, so T0 is final and there is no T1 upload.**

## Slot 2 (Track B)

- **B1, sells first: ACCEPTED.** `SELLS_FIRST` in `src/agents/competitive_meta.py`.
  Seeds 0-11, both seats:

  | Matchup | Record | Mean bank |
  |---|---|---|
  | B1 vs unmodified policy | **24-0** | 63,837 vs 59,764 |
  | B1 vs lonespear v18 | **24-0** | 61,986 vs 56,980 |
  | unmodified vs lonespear v18 (control) | 16-8 | 64,896 vs 62,547 |

  Both gates pass (>= 15/24, and no worse than 16-8 vs lonespear). H1 verifier:
  READY TO SUBMIT; self-play 720 steps, max turn 78 ms; starter 123,153 vs 3,421.
  Logs: `logs/h1_b1_gate.json`, `logs/h1_b1_vs_lonespear.json`,
  `logs/h1_meta_vs_lonespear.json`.
- **B2, front-run by opponent shed:** dropped. The opponent's shed and unit
  inventories are private (`docs/ENVIRONMENT.md` §3), so the premise was wrong.
