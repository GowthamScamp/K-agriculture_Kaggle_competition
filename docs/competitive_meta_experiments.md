# Competitive meta rebuild - 20 September 2026

## Decision

The 605.2 and 607.5 live submissions are a plateau. The old `reactive_route`
agent is safe but under-produces: its default flex crop count is zero, it locks
the herd from a single early shop signal, and it leaves roughly 23% of worker
actions as PASS in measured episodes.

The new submitted path is `src/agents/competitive_meta.py`, derived from the
MIT-licensed `lonespear/main_v18.py` policy. The original copyright and complete
MIT permission notice are embedded in both source and generated submission.
The existing route, baseline and MCTS code remain available as controls.

## Accepted changes

- Herd/feed capital is established before the strawberry ramp. The measured
  checkpoint shape is about 8 cows, 6 sheep, 10 wheat, 24 strawberry and 11
  hands over two quadrants.
- Standing-on-work selection and global assignment replace the static
  quadrant-slot economy for the submitted path.
- SciPy/NumPy assignment was replaced with an in-file pure-Python rectangular
  Hungarian solver so local and Kaggle behavior do not depend on third-party
  packages.
- Unsafe multi-worker DROP operations are capacity simulated. Overflowing drops
  become bounded PLACE operations instead of silently discarding inventory.
- The market sees successful same-turn deposits during liquidation, and step
  718 forces all bankable shed inventory into sell orders.
- `step=None`, one-argument and two-argument entry calls, exception fallback,
  standalone/offline execution and last-definition binding are tested.
- `tools/build_submission.py` now copies the measured source byte-for-byte into
  `submission/main.py`; the zip contains exactly the same bytes.

## Rejected changes

- Buying fertilizer: v18 collected 339 field units and picked up 74 but applied
  only 48; the larger v23 collected 330 and picked up 108 but applied only 78.
  Labor, not fertilizer supply, is binding.
- Large v23 shape: 3 quadrants, 44 strawberry, 14 wheat and 14 hands averaged
  only 85.1k against starter on seeds 0-3 and is documented 0-40 versus v18.
- Reactive-route cluster bonus 8: the first paired observations materially
  reduced bank, so it remains a disabled research parameter.
- Aggressive day-28 input ferry: it reduced seed-0 bank despite lowering final
  unsold inventory and was reverted. Production value exceeded the saved stock.
- Public 2,600-3,090 notebook schedules: inspected examples contain 719/720-turn
  replay tapes. They are benchmark evidence only and no tape was copied.

## Competitive evidence

All rows use fresh simulator episodes, both seats, and live observations.

| Matchup | Seeds | Record | Mean bank |
|---|---:|---:|---:|
| competitive_meta vs old reactive_route | 0-3 | 8-0 | 56,466 vs 51,291 |
| competitive_meta vs old reactive_route | 4-7 | 8-0 | 77,093 vs 67,970 |
| competitive_meta vs clean MIT v18 | 0-3 | 4-4 | 72,011 vs 69,700 |
| competitive_meta vs clean MIT v18 | 4-7 | 8-0 | 53,293 vs 50,002 |
| competitive_meta vs clean MIT v18 | 8-11 | 4-4 | 69,384 vs 67,937 |

Combined, the new policy is **16-0** against the prior submission and **16-8**
against v18. Its combined v18 mean is about **64,896 versus 62,546**.

The public-state replay router remains an upper bound. On seed 0 the old route
banked 55,008 against it; the new policy banked 79,524, but still lost to the
router's 136,158. This is an honest limitation and why no live rating is
guaranteed.

## Verification

- 104 core tests passed in the grouped run.
- The only grouped failure was Windows sandbox denial for a temporary directory;
  the exact offline test passed when rerun outside that sandbox.
- All 18 search/agent compatibility tests passed.
- Final verifier: standalone import, standard-library dependency scan,
  reproducible build, archive identity, offline execution, self-play and starter
  episode all passed; status `READY TO SUBMIT`.
- Final verifier self-play: 41,966 / 41,966, maximum turn 137 ms.
- Final verifier starter episode: 123,149 / 3,421, mean 0.5 ms, maximum 3 ms.
