"""Contracts for the attributed standard-library competitive policy."""

from __future__ import annotations

import ast
import itertools
from pathlib import Path

from src.agents import competitive_meta as meta
from src.runner import run_episode


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "agents" / "competitive_meta.py"


def test_vendored_policy_retains_mit_notice_and_has_no_scipy_dependency():
    text = SOURCE.read_text(encoding="utf-8")
    assert "Copyright (c) 2026 Jonathan Day" in text
    assert "Permission is hereby granted" in text

    roots = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    assert "numpy" not in roots
    assert "scipy" not in roots


def test_pure_python_hungarian_finds_rectangular_optimum():
    cost = [
        [9.0, 2.0, 7.0, 8.0],
        [6.0, 4.0, 3.0, 7.0],
        [5.0, 8.0, 1.0, 8.0],
    ]
    rows, cols = meta._linear_sum_assignment(cost)
    actual = sum(cost[row][col] for row, col in zip(rows, cols))
    expected = min(
        sum(cost[row][col] for row, col in enumerate(columns))
        for columns in itertools.permutations(range(4), 3)
    )
    assert actual == expected
    assert len(set(rows)) == len(rows) == 3
    assert len(set(cols)) == len(cols) == 3


def test_pure_python_hungarian_handles_more_rows_than_columns():
    cost = [[5.0, 1.0], [2.0, 4.0], [3.0, 0.0]]
    rows, cols = meta._linear_sum_assignment(cost)
    assert len(rows) == len(cols) == 2
    assert len(set(rows)) == 2
    assert len(set(cols)) == 2
    assert sum(cost[row][col] for row, col in zip(rows, cols)) == 2.0


def test_overflowing_drop_becomes_bounded_place_and_is_sold_on_final_turn():
    tiles = [[None for _ in range(10)] for _ in range(10)]
    obs = {
        "player": 0,
        "step": None,
        "day": 29,
        "hour": 22,
        "farms": [
            {
                "money": 1000,
                "tiles": tiles,
                "farmer": [4, 4],
                "hands": [],
                "unlocked_quadrants": ["NW"],
                "hires_today": 0,
            },
            {
                "money": 1000,
                "tiles": tiles,
                "farmer": [4, 4],
                "hands": [],
                "unlocked_quadrants": ["NW"],
                "hires_today": 0,
            },
        ],
        "private": {
            "shed": {"WHEAT": 99},
            "seeds": {},
            "inventories": [{"MILK": 3}],
        },
        "market": {
            "inventory": {item: 10000 for item in meta.BASE_PRICE},
            "prices": dict(meta.BASE_PRICE),
        },
        "town": {"unlocked_shops": []},
    }
    action = {"farmer": ["DROP"], "hands": [], "market": []}
    shed = meta._protect_drops_and_shadow_shed(obs, action)
    meta._liquidation_orders(obs, action, shed)

    assert action["farmer"] == ["PLACE", "MILK", 1]
    assert ["SELL", "MILK", 1] in action["market"]
    assert ["SELL", "WHEAT", 99] in action["market"]


def test_shared_shed_stock_is_reserved_across_worker_pickups():
    units = [(4, 4)] * 4
    actions = meta._assign(
        units=units,
        tasks=[],
        shed_adj=[(4, 4)],
        inventories=[{} for _ in units],
        shed={"FERTILIZER": 8, "WHEAT": 3, "COW": 1},
        needs={"fert": 11, "wheat": 10,
               "animals": {"COW": 2, "SHEEP": 0, "GOOSE": 0}},
        day=10,
    )

    assert actions == [
        ["PICKUP", "COW", 1],
        ["PICKUP", "FERTILIZER", 4],
        ["PICKUP", "FERTILIZER", 4],
        ["PICKUP", "WHEAT", 3],
    ]


def test_competitive_meta_completes_a_full_episode_without_invalid_unit_actions():
    result = run_episode(
        [meta.agent, "starter"],
        seed=0,
        time_seat=0,
        telemetry_seat=0,
    )
    assert result.statuses[0] == "DONE"
    assert result.rewards[0] > result.rewards[1]
    assert result.telemetry is not None
    assert result.telemetry.avoidable_noops == 0
    assert result.max_turn_time < 0.5
