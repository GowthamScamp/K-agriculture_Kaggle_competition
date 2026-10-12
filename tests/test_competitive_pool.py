"""Statistical contracts for the offline competitive harness."""

from tools.competitive_pool import Competitor, PairResult, _competitor, bradley_terry


def test_competitor_spec_accepts_absolute_windows_paths(tmp_path):
    source = tmp_path / "opponent.py"
    source.write_text("def helper(obs, config=None):\n    return 'h'\n"
                      "def agent(obs, config=None):\n    return 'a'\n", encoding="utf-8")
    assert _competitor(f"x={source}").factory()(None) == "a"
    assert _competitor(f"x={source}:helper").factory()(None) == "h"


def _unused():
    raise AssertionError("factory is not used by the fit")


def test_bradley_terry_orders_a_consistent_three_agent_pool():
    competitors = [
        Competitor("strong", _unused),
        Competitor("middle", _unused),
        Competitor("weak", _unused),
    ]
    rows = [
        PairResult("strong", "middle", [2, 3], [1, 1], [0, 1], 1.0),
        PairResult("strong", "weak", [3, 4], [0, 1], [0, 1], 1.0),
        PairResult("middle", "weak", [2, 2], [1, 0], [0, 1], 1.0),
    ]
    ratings = bradley_terry(competitors, rows)
    assert ratings["strong"] > ratings["middle"] > ratings["weak"]


def test_pair_result_counts_ties_as_half_a_win():
    row = PairResult("a", "b", [5, 5, 1], [5, 2, 3], [0, 1, 0], 4.0)
    assert row.outcome.record == "1W-1T-1L"
    assert row.outcome.rate == 0.5
