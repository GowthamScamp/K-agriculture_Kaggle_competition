# Offline benchmark pool

This directory contains public agents used as opponents and design evidence.
The one exception is the pinned Apache-2.0 chassis copied into the slot-1 tape
artifact (see "Policy change" below).

| Label | Source | License | Local status |
|---|---|---|---|
| `lonespear` | <https://github.com/lonespear/kaggriculture> at `774b26093ccf4246525517d48420349b841b6e50` | MIT | Callable on Windows; primary heuristic benchmark. |
| `adaptive_route_v2` | <https://www.kaggle.com/code/reyhanksatria/adaptive-route-agent-v2>, version 2 | Apache-2.0 notebook | Archive SHA-256 `b650a31d091323f2510aede0265937d3193a82a99109eadc8ab39c6e85db278d`; native Linux `.so`, so it cannot execute in the Windows arena. |
| `public_state_router` | <https://www.kaggle.com/code/thomastschinkel/kaggriculture-93-8-win-rate-public-state-router>, version 3 | Apache-2.0 notebook | Callable; compressed replay-tape portfolio. Benchmark opponent. |
| `farm2945` | `farm2945_v9_4.py` from <https://github.com/warlord-2227/kaggle> at `f8ddefbef8393011363746b158692ab2ca89d9aa`, `refagents/public/` | Apache-2.0 (V39 lineage; notices in file) | SHA-256 `BFEE70E9DAAEBEAE0737A880F1DF8F1C60D0783C59AF620136CC0D28EF482BC7`, 856 KB, stdlib-only, load 0.18 s, max turn 142 ms. **Eligible slot-1 chassis.** |
| `tetsutani_dp` | `tetsutani_demand_preserving.py`, same repository and commit | Apache-2.0 (V39 lineage; notices in file) | SHA-256 `9C0A8921AA49FBA080D53DEEE3C9FE8DB7A8E2B02881E9ADDA4090AF73E9C187`, 1.0 MB, stdlib-only, load 0.39 s, max turn 234 ms. **Eligible slot-1 chassis.** |
| `shopwork` | `shopwork_tetsutani.py`, same repository and commit | No notice in the file | SHA-256 `2B97E2C653018AC4AEFFB8463EC91C8F26B097B4CEF81289ACC25CCDBC68F916`, 45 KB. Benchmark opponent only: without a notice in the file its licence cannot be verified, so it is never shipped. |

## Policy change, 27 September 2026

The ladder above ~1,700 is held by replay-tape agents descended from public
Apache-2.0 Kaggle notebooks, and our own heuristics rated 569-607. The owner
chose a hybrid: slot 1 ships one Apache-2.0 chassis from the table above,
byte-for-byte with its notices retained, plus a wrapper documented as an
Apache §4(b) modification (`tools/build_submission.py --target tape`). Slot 2
remains our own policy with no tapes. The builder pins each eligible chassis by
SHA-256, so an edited benchmark file cannot reach an upload.

Use the seat-balanced arena with a fresh module instance per game:

```powershell
python -m tools.competitive_pool `
  --agent reactive_route `
  --opponent starter `
  --opponent lonespear=benchmarks/public/lonespear/main.py `
  --seeds 96 --round-robin
```

The harness reads Python agent files and live simulator observations only. It does
not inspect `.jsonl` replay files. No tape is built by this project; the only
tapes in any upload are those inside the licensed public chassis shipped in slot 1.
