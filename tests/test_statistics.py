"""The statistics that decide which agent ships.

These are worth testing directly because every published claim about the agent
rests on them, and because three of the four bugs they now guard against were
found by re-reading old results rather than by anything failing loudly:

* ties were counted as losses, which dragged a dead-even self-play sanity check
  down to 47% when the correct answer was 50.5%;
* `_student_t_critical` rounded degrees of freedom the wrong way, narrowing
  every confidence interval;
* `mean_turn_time` was accumulated with `max()`, so it reported the worst
  episode's mean under a label that said "mean".

The Wilson interval is here because the win rates that matter sit at 85-95%,
where the textbook Wald interval is skewed and can reach past 100%.
"""

from __future__ import annotations

import math

import pytest

from src.evaluate import (
    EvalReport,
    McNemarResult,
    WinRateResult,
    _binomial_two_sided_p,
    _student_t_critical,
)


# --------------------------------------------------------------------------
# Win rate and the Wilson interval
# --------------------------------------------------------------------------
def test_ties_are_half_a_win_not_a_loss() -> None:
    """The historical bug: 47W-7T-46L was reported as 47%, not 50.5%.

    These are the real `baseline` self-play numbers from the n=100 matrix.
    Counting the seven ties as losses dragged a dead-even self-play result down
    to 47%, which looks like an agent losing to itself. Scoring a tie as half a
    win puts it at 50.5% - one win above even out of a hundred - and, correctly,
    nowhere near significant.
    """
    margins = [1.0] * 47 + [0.0] * 7 + [-1.0] * 46
    w = WinRateResult.of(margins)
    assert (w.wins, w.ties, w.losses) == (47, 7, 46)
    assert w.rate == pytest.approx(0.505)
    assert abs(w.rate - 0.5) < 0.01
    assert not w.significant, "an even self-play record must not read as an edge"


def test_wilson_interval_brackets_the_estimate_and_stays_in_range() -> None:
    for wins in (0, 1, 50, 82, 99, 100):
        w = WinRateResult.of([1.0] * wins + [-1.0] * (100 - wins))
        assert 0.0 <= w.ci_low <= w.rate <= w.ci_high <= 1.0, wins


def test_wilson_interval_does_not_collapse_at_a_perfect_record() -> None:
    """Where Wald breaks: 100/100 gives it [1.0, 1.0], claiming zero uncertainty."""
    w = WinRateResult.of([1.0] * 100)
    assert w.rate == 1.0
    assert w.ci_high == 1.0
    assert 0.95 < w.ci_low < 0.97, w.ci_low
    assert w.significant


def test_wilson_matches_a_hand_computed_value() -> None:
    """82 wins in 100: the published `mcts` figure, checked against the formula."""
    w = WinRateResult.of([1.0] * 82 + [-1.0] * 18)
    z, n, p = 1.959963984540054, 100, 0.82
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    assert w.ci_low == pytest.approx(centre - half)
    assert w.ci_high == pytest.approx(centre + half)


def test_even_record_is_not_significant_but_a_lopsided_one_is() -> None:
    assert not WinRateResult.of([1.0] * 50 + [-1.0] * 50).significant
    assert WinRateResult.of([1.0] * 65 + [-1.0] * 35).significant
    losing = WinRateResult.of([1.0] * 35 + [-1.0] * 65)
    assert losing.significant and losing.rate < 0.5


def test_empty_record_is_inert() -> None:
    w = WinRateResult.of([])
    assert w.n == 0 and not w.significant


# --------------------------------------------------------------------------
# Exact binomial / McNemar
# --------------------------------------------------------------------------
@pytest.mark.parametrize("k, n, expected", [
    (0, 0, 1.0),
    (0, 1, 1.0),          # 2 * 1/2
    (0, 5, 2 / 32),
    (0, 10, 2 / 1024),
    (5, 10, 1.0),         # dead centre
    (10, 10, 2 / 1024),   # symmetric with k=0
])
def test_binomial_two_sided_p_is_exact(k: int, n: int, expected: float) -> None:
    assert _binomial_two_sided_p(k, n) == pytest.approx(expected)


def test_mcnemar_ignores_concordant_pairs() -> None:
    """Seeds where both agents won say nothing about which is better."""
    a = [True] * 90 + [True] * 5 + [False] * 5
    b = [True] * 90 + [False] * 5 + [True] * 5
    m = McNemarResult.of(a, b)
    assert m.both == 90 and m.a_only == 5 and m.b_only == 5
    assert m.p_value == pytest.approx(1.0)
    assert not m.significant


def test_mcnemar_detects_a_one_sided_disagreement() -> None:
    a = [True] * 10 + [True] * 10
    b = [True] * 10 + [False] * 10
    m = McNemarResult.of(a, b)
    assert (m.a_only, m.b_only) == (10, 0)
    assert m.significant and m.p_value < 0.01


def test_mcnemar_reproduces_the_stored_det_worst_result() -> None:
    """10 vs 4 discordant is suggestive but not significant - the real n=100 case.

    Recorded so the conclusion in docs/NOTES.md cannot drift away from the test
    that supports it.
    """
    a = [True] * 10 + [False] * 4
    b = [False] * 10 + [True] * 4
    m = McNemarResult.of(a, b)
    assert m.p_value == pytest.approx(0.17957, abs=1e-4)
    assert not m.significant


# --------------------------------------------------------------------------
# Regression guards
# --------------------------------------------------------------------------
@pytest.mark.parametrize("df, expected", [
    (1, 12.706), (10, 2.228), (30, 2.042), (60, 2.000),
    (11, 2.228),    # off-table: rounds DOWN to df=10, not up to df=12
    (99, 2.000),    # off-table: rounds DOWN to df=60
    (500, 1.980),
])
def test_t_critical_rounds_down_so_intervals_never_narrow(df: int, expected: float) -> None:
    assert _student_t_critical(df) == pytest.approx(expected)


def test_t_critical_is_monotonic_non_increasing() -> None:
    values = [_student_t_critical(df) for df in range(1, 400)]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_mean_turn_time_is_a_mean_not_a_maximum() -> None:
    report = EvalReport(agent="a", opponent="b", episodes=3)
    report.turn_time_means.extend([0.001, 0.002, 0.009])
    assert report.mean_turn_time == pytest.approx(0.004)


def test_report_summary_leads_with_the_ladder_metric() -> None:
    report = EvalReport(agent="a", opponent="b", episodes=2)
    report.banks.extend([10.0, 20.0])
    report.opponent_banks.extend([5.0, 25.0])
    report.margins.extend([5.0, -5.0])
    report.wins = 1
    text = report.summary()
    assert "LADDER METRIC" in text
    assert text.index("LADDER METRIC") < text.index("EFFECT SIZE")
    assert "1W-0T-1L" in text
