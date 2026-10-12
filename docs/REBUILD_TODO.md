# Historical reactive-route rebuild checklist

> **Archived record.** This checklist describes the 13 September 2026
> `ReactiveRouteAgent` work. It is not the status of the later canonical
> `competitive_meta` submission or of the uncompleted October promotion gate.

Completed sequentially on 13 September 2026. The checklist distinguishes code
correctness from aspirational competitive targets; a checked implementation item
does not imply that every score target passed.

## Implementation

- [x] Audit the scored MCTS control and existing labor/liquidation experiments.
- [x] Add robust observation parsing, episode-reset detection, and in-memory
  operation/economy telemetry without reading replay `.jsonl` data.
- [x] Implement a standalone `ReactiveRouteAgent` while preserving baseline and
  MCTS as research controls.
- [x] Build compact quadrant/crop/pasture slots and persistent daily worker zones.
- [x] Add same-tile action chains and per-unit, per-quadrant inventory reservation.
- [x] Phase hiring, expansion, herd purchases, and the three-wheat feed reserve.
- [x] Make herd choice, sale batches, crop reuse, and fertilizer timing reactive.
- [x] Use post-action shadow inventory for market orders and implement exact final
  liquidation without attempting harvests that cannot be banked.
- [x] Add unit, integration, legality, completion, timing, reset, seat, market,
  hysteresis, route, terminal, and submission-contract tests.
- [x] Run matched parameter screens and revert measured regressions.
- [x] Remove rollout search and simulator modules from the generated competition
  artifact and promote `ReactiveRouteAgent` as the singleton entry point.
- [x] Rebuild `submission/main.py` and `submission/kaggriculture.zip`.
- [x] Verify source reproducibility, offline imports, archive identity, both seats,
  720-turn completion, and the one-second action limit.
- [x] Update submission instructions, experiment evidence, and limitations.

## Measured gates

| Gate | Result | Status |
|---|---:|---|
| Illegal/avoidable no-op operations | 0 | PASS |
| Water/feed/care completion | 99.99% / 99.60% / 99.58% | PASS |
| Final unsold inventory | at most 1 item in starter screen | PASS |
| Movement | 43.98% (target <=48%) | PASS |
| Maximum verified artifact turn | 203 ms (limit 1,000 ms) | PASS |
| Full 720-turn artifact episodes | completed in self-play and vs starter | PASS |
| Tests | 114 passed | PASS |
| Starter mean | 106.5k (target 130k) | FAIL |
| Starter 10th percentile | 71.3k (target 105k) | FAIL |
| Starter minimum | 70.2k (target 85k) | FAIL |
| PASS operations | 23.36% (target <=1.5%) | FAIL |
| Productive operations | 32.66% (target >45%) | FAIL |
| Admissible public benchmark | 1-7 (target >=60% aggregate wins) | FAIL |
| Submission packaging and entry point | `READY TO SUBMIT` | PASS |

## Release decision

`submission/main.py` is promoted because it replaces the approximately 51.8k
old submitted MCTS path with a deterministic, legal, offline-safe route averaging
106.5k against `starter`. It is the repository's best verified upload artifact,
but the failed competitive gates remain open improvement work and a Kaggle rating
increase cannot be guaranteed from local results.
