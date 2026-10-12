from tools.score_gate import paired_comparison, summary


def row(agent, opponent, seed, seat, bank, other):
    return {"agent": agent, "opponent": opponent, "seed": seed, "seat": seat,
            "bank": bank, "opponent_bank": other, "steps": 720,
            "statuses": ["DONE", "DONE"], "diagnostic_errors": [{}, {}], "max_turn_ms": 1}


def test_both_seats_are_one_cluster_and_noop_cannot_pass():
    records = [row(agent, "rival", seed, seat, 2, 1)
               for agent in ("new", "old") for seed in range(96) for seat in (0, 1)]
    result = paired_comparison(records, "new", "old", ["rival"], list(range(96)), samples=50)
    assert result["seed_clusters"] == 96
    assert result["mean_outcome_improvement"] == 0
    assert not result["economic_gate_passed"]


def test_missing_seat_or_internal_error_blocks_promotion():
    records = [row(agent, "rival", seed, seat, 2 if agent == "new" else 0, 1)
               for agent in ("new", "old") for seed in range(96) for seat in (0, 1)]
    assert paired_comparison(records, "new", "old", ["rival"], list(range(96)), 50)["economic_gate_passed"]
    records[0]["diagnostic_errors"] = [{"errors": 1}, {}]
    assert not paired_comparison(records, "new", "old", ["rival"], list(range(96)), 50)["economic_gate_passed"]
    records.pop()
    assert not paired_comparison(records, "new", "old", ["rival"], list(range(96)), 50)["complete"]


def test_opponent_regression_cannot_hide_behind_aggregate_wins():
    records = []
    for opponent in ("good", "bad"):
        for seed in range(96):
            for seat in (0, 1):
                for agent in ("new", "old"):
                    wins = (agent == "new") == (opponent == "good")
                    records.append(row(agent, opponent, seed, seat, int(wins), 0.5))
    result = paired_comparison(records, "new", "old", ["good", "bad"], list(range(96)), 50)
    assert result["by_opponent_improvement"]["bad"] == -1
    assert not result["economic_gate_passed"]
    assert len(summary(records)) == 4
