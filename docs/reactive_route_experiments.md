# Reactive-route rebuild experiments

All comparisons below use the same listed seeds for every variant. Evaluation is
offline with `kaggle-environments==1.32.7`; no replay `.jsonl` file was accessed.
Bank is useful economic evidence, while the Kaggle ladder ultimately scores match
outcomes. Results from fewer than 32 seeds are screening experiments, not final
confidence claims.

## Incumbent diagnosis

| Variant | Seeds | Mean bank vs starter | Decision |
|---|---:|---:|---|
| Submitted MCTS artifact | 8 | 51.8k | Control only. The search chooses between `auto` and forced wheat with a two-day proxy. |
| Baseline + affordable labor floor + liquidation workforce | 8 | 74.6k | Accepted as evidence for the new labor rules. |
| Same, fertilizer disabled | 8 | 44.9k | Rejected. Fertilizer must be managed, not removed. |
| Same, realized shop demand | 8 | 71.3k | Kept as a market signal, not a standalone gain claim. |
| Fixed 10 hands / fixed 12 hands | 8 | 63.4k / 57.8k | Rejected. Static oversized labor loses money. |

## Route construction and crop allocation

Initial 24-crop route reached 102.4k on seed 7 but averaged 85.7k over eight
seeds. Increasing density without routing fixes made performance worse. After the
standing-on-work pass, bounded animal harvests, feed distribution, staggered
planting, and mandatory-work allocation, the matched crop screen was:

| Crop slots | Allocation (wheat/strawberry/flex) | Seeds | Mean | Minimum | Decision |
|---:|---:|---:|---:|---:|---|
| 32 | 16 / 6 / 10 | 0–7 | 90.0k | 46.4k | Rejected default. |
| 32 | 10 / 16 / 6 | 0–7 | 99.9k | 61.6k | Viable but dominated. |
| 32 | 10 / 22 / 0 | 0–7 | 102.5k | 78.1k | Accepted default. |
| 32 | 12 / 14 / 6 | 0–7 | 93.5k | 75.4k | Rejected. |
| 28 | 10 / 14 / 4 | 0–7 | 93.8k | 74.8k | Rejected. |

A later density screen on seeds 0–5 produced 106.2k / 103.3k / 99.5k / 55.7k
for 32 / 36 / 40 / 44 slots respectively. Thirty-two remains the stable choice.
The 22 strawberry positions are staggered at three new plantings per day; the
high count is a capacity, not a synchronized opening purchase.

## Execution and logistics

The first 32-crop route spent about 57% of unit operations moving. Claiming
zero-distance productive work before preserving a prior destination reduced the
12-seed movement mean to 46.5%. Reserving mandatory feed and late watering before
ordinary work then produced 99.88% watering, 99.05% feeding, and 99.92% care in a
12-seed screen, with a maximum observed decision time of 85.2 ms.

Returning only sellable loads of eight units, dropping loads of three already at
the shed, and collapsing both thresholds on day 28 reduced one representative
competitive episode from 168 DROP actions to 30. The operation precondition audit
reported zero guaranteed unit no-ops on the tested full episodes.

Deferring irreversible herd expansion until day 6 was screened on seeds
0, 1, 3, 7, 13, 23 and 26 in both seats. It produced a 90.6k mean, 62.4k
minimum, 40.4% movement, 99.2% feeding and 99.1% care. The early signal chose a
wool-heavy herd on a season where wool later collapsed, so the variant was
rejected and balanced 2/2 -> 4/3 -> 6/4 staging remains the default. The delayed
mode remains available through `RouteParams.herd_commit_day` for reproducible
research; it is not used by the submitted policy.

Three other structural variants were rejected on seeds 0-7 in both seats:
reducing the post-expansion crew to 9/10 hands gained only 176 mean coins from
one outlier while losing most paired games; a fixed 9/4 herd averaged 100.2k but
lost 11/16 paired comparisons; and planting the entire strawberry cohort at
once averaged 100.0k but lost 11/16. A two-quadrant packing experiment initially
failed to expand, and after its purchase window was relaxed it averaged only
94.3k with some feeding below 98%. These modes remain controllable parameters,
not defaults.

Isolating fertilizer timing found a real defect. The policy was fertilizing
strawberries from planting even though cover lasts only three days and the first
yield is on day 10. Starting applications two days before first yield reduced
expired applications and was retained for the full 32-seed economic validation.
In a four-seed, both-seat public-agent screen it increased contested-market bank
but remained 0-8 against `lonespear`; it is an economic correction, not evidence
that the competitive gate passed.

The final 32-seed, both-seat starter run produced 64/64 wins, **106,311 mean
bank**, 75,964 10th percentile and a 54,342-131,406 range. This passes the
100k first economic gate but fails the later 105k 10th-percentile and 85k
minimum requirements. Parallel evaluation reported one 741.1 ms outlier while
the sequential matched screens stayed below 223 ms; timing must be rechecked
sequentially before any artifact promotion.

Six-unit sell batches with a 45% normal price floor were retained. On the initial
three-seed screen they preserved about 97–98k solo mean and improved competitive
cash realization relative to 20-unit dumps, although they did not reverse the
public-agent matchup.

## Competitive results and failed gate

Against the MIT `lonespear` cow/sheep + strawberry benchmark, the first complete
screen was 0 wins in 16 games across eight seeds and both seats (mean banks about
62k vs 80k before later tuning). Crop diversification alone remained 0/8 in each
small matched screen, demonstrating that the issue was not merely strawberry
market collision.

With the completed source, the four-seed both-seat rematch was still 0-8, at
66.6k mean bank versus 82.4k. A same-seed production audit found
that `reactive_route` spent fewer actions moving (41.4% versus 51.1%) and had
perfect watering/feeding on that episode, but planted 121 short-cycle wheat
crops versus 70 and only 22 strawberries versus 28. Its PASS rate was 26.5%
versus 13.4%. The remaining gap is productive throughput and product mix, not
basic legality or liquidation.

A final sequential telemetry sample over seeds 0 and 1 in both seats measured
40.36% movement, 31.03% productive operations, 28.62% PASS, 100% watering,
98.99% feeding, 99.07% care and an 82.2 ms maximum turn. It clears the movement
and production-preservation thresholds but fails the planned >45% productive
and <=1.5% PASS gates by a wide margin. Those failures agree with the public
match audit and are blockers, not metrics hidden by the starter win rate.

The Apache public-state router was also tested on two seeds in both seats and won
all four games, scoring roughly 104–149k against 31–59k. Its notebook explicitly
states that it embeds top-game action tapes. It is therefore an adversarial
upper-bound benchmark and is not admissible implementation material for this
no-tape rebuild.

## 13 September rebuild completion

This section supersedes the earlier non-promotion decision above. The strict
aspirational economic and competitive gates still have not passed. The route was
nevertheless promoted because the actual incumbent was the approximately 51.8k
submitted MCTS artifact, not a 130k public-pool winner, and the completed route
is a substantially safer and stronger upload candidate.

### Accepted corrections

- Default layout reduced to two compact quadrants with 13 pasture positions and
  32 crop positions: 10 wheat and 22 staggered strawberry capacity.
- Daily worker-to-zone affinity and within-zone task matching now avoid routine
  quadrant switching. Feed pickups and reservations are also quadrant-specific.
- The day-6 observed Yarn-shop signal chooses and locks the balanced, milk-heavy,
  or wool-heavy herd plan; 2/2 -> 4/3 -> 6/4 staging protects opening cash.
- Same-tile `PLANT -> WATER` and `WATER -> HARVEST` chains use the environment's
  unit execution order, while per-unit reservations prevent duplicate carrying.
- Mandatory water/feed/care allocation, late-day care urgency, and telemetry
  credit for hour-23 completions close the production-preservation gaps.
- Market orders use shadow post-drop shed inventory, valid submitted sells to
  fund purchases, marginal sale batches, visible supply pressure, and exact
  step-718 liquidation. Late harvests that cannot reach the shed are skipped.
- The generated submission includes only the route, adapter, constants, and
  baseline fallback. MCTS, the simulator, and rollout policies are research-only.

### Rejected screens

The following were tested on matched seeds and reverted after lowering bank,
completion rates, or public-reference performance: 7/9/9/8 labor; three-quadrant
packing; 34-44 crop slots; 6/26 and 8/24 crop mixes; 14-animal and earlier full
herds; 11/2, 4/9 and 5/9 herd variants; wheat reserve of two; 11 hands after the
first expansion; four endgame spare slots; distance costs of two or six; strict
market slot reservation; three-unit scarce sale batches; global early care;
fertilizer liquidation at a floor of one; global Hungarian assignment;
sheep-first purchasing; and a six-slot flexible crop policy. Disabling
fertilizer entirely was also rejected earlier at 44.9k mean.

### Final fixed-seed evidence

Against `starter`, seeds 0-7 in both seats produced these 16 banks:

```
115730, 112354, 121636, 134412, 115169, 93046, 127423, 112933,
79817, 112870, 86246, 115808, 118079, 117667, 70237, 71290
```

Mean was **106,544.8**, 10th percentile **71,290**, and range
**70,237-134,412**. Aggregate telemetry was 43.98% movement, 23.36% PASS,
32.66% productive operations, 99.99% watering, 99.60% feeding, 99.58% care,
zero avoidable no-ops, and at most one final unsold item.

The final four-seed, both-seat admissible public screen was **1-7**, with route
mean 60,182.5 versus 63,127.8. This reduced the earlier measured mean deficit
from about 15.8k to 2.9k, but it does not clear the planned 60% win-rate gate.

The standalone artifact verifier completed a full 720-turn self-play episode at
88,640 and 89,672 bank with a 203 ms maximum turn, and a full starter episode at
127,423 versus 3,484 with an 89 ms maximum turn. The bundle is reproducible,
offline-safe, archive-identical, and marked `READY TO SUBMIT`.

Promotion therefore means "best verified artifact in this repository", not
"all aspirational gates passed" and not a guarantee of a live ladder rating.

## 13 September income-maximization follow-up

A second audit targeted the remaining gap rather than assuming that a larger
farm would earn more. All runs again used live simulator observations only; no
replay `.jsonl` file or fixed action tape was read or copied.

The following bounded variants were screened and reverted:

- Moving the shop-based herd decision from day 6 to day 9 improved the
  eight-seed starter mean to about 108.3k and the minimum to 89.6k, but lost
  0-8 to the admissible `lonespear` benchmark and worsened contested margin.
- Beginning short-crop refill on day 21 improved 11 of 16 starter games but
  added only 366 mean coins, lowered the minimum, and was neutral-to-negative
  against `lonespear`; day 24 remains the robust default.
- Four- and five-unit feed bundles, alternate pasture splits, earlier/later
  animal harvests, fertilizer timing of one or three days, and temporary wheat
  on newly purchased land all reduced matched bank.
- A survival-safe alternate-day watering schedule cut movement to roughly 40%
  but also broke useful route patrol/sale cadence. It scored 77.4k on the first
  seed pair; adding a third quadrant and 44 crop targets recovered only 97.5k.
- Against the 30+ strawberry public-state router, replacing six strawberry
  targets with tomato raised our bank from 55.0k to 70.5k but raised the rival
  from 102.4k to 126.9k. The opposite two-slot market attack lowered our bank to
  43.9k. Both lose on the actual match objective and were removed.
- Buying extra fertilizer, reserving 22 fertilizer units in the field, opening
  melon, forced crop interleaving, demand-timed selling, larger sale refresh
  batches, and a naive same-tile feed/care chain were also rejected after
  matched screens.

The retained work is diagnostic/correctness infrastructure, not an unsupported
score claim: daily crop and animal composition are now recorded in memory,
unsafe shed overflow is counted, and the post-unit shadow shed respects the
real 100-unit capacity. The incumbent discarded zero DROP units on the checked
episodes, so the competitive policy parameters remain unchanged. This is
intentional: a neutral verifier improvement is safer than promoting any of the
measured regressions above.
