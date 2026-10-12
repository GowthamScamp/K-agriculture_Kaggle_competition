"""Behavioral contracts for the competitive route agent."""

from __future__ import annotations

from src.agents.baseline_agent import Task
from src.agents.reactive_route import ReactiveRouteAgent
from src.env_adapter import View
from src.runner import EpisodeTelemetry, run_episode


def _tiles(unlocked=("NW",)):
    grid = [[None for _ in range(10)] for _ in range(10)]
    for y in range(10):
        for x in range(10):
            quadrant = ("N" if y < 5 else "S") + ("W" if x < 5 else "E")
            if quadrant not in unlocked:
                grid[y][x] = "LOCKED"
    return grid


def _obs(*, day=0, hour=0, unlocked=("NW",), hands=0, money=3000, private=None):
    positions = [[4, 4] for _ in range(hands)]
    return {
        "player": 0,
        "step": day * 24 + hour,
        "day": day,
        "hour": hour,
        "farms": [
            {
                "money": money,
                "tiles": _tiles(unlocked),
                "farmer": [4, 4],
                "hands": positions,
                "unlocked_quadrants": list(unlocked),
                "hires_today": 0,
            },
            {
                "money": money,
                "tiles": _tiles(unlocked),
                "farmer": [4, 4],
                "hands": [],
                "unlocked_quadrants": list(unlocked),
                "hires_today": 0,
            },
        ],
        "private": private or {
            "shed": {},
            "seeds": {},
            "inventories": [{} for _ in range(hands + 1)],
        },
        "market": {
            "inventory": {
                "WHEAT": 10000,
                "CARROT": 10000,
                "TOMATO": 10000,
                "STRAWBERRY": 10000,
                "MELON": 10000,
                "EGG": 10000,
                "MILK": 10000,
                "WOOL": 10000,
                "FERTILIZER": 10000,
            },
            "prices": {
                "WHEAT": 25,
                "CARROT": 35,
                "TOMATO": 60,
                "STRAWBERRY": 120,
                "MELON": 250,
                "EGG": 50,
                "MILK": 160,
                "WOOL": 200,
                "FERTILIZER": 100,
            },
        },
        "town": {"unlocked_shops": []},
    }


def test_route_has_stable_three_quadrant_capacity():
    agent = ReactiveRouteAgent(route_kwargs={"max_quadrants": 3})
    v = View.from_obs(_obs(unlocked=("NW", "NE", "SW")))
    pastures, crops = agent._route_slots(v)
    assert len(pastures) == 13
    assert len(crops) == 32
    assert not (set(pastures) & set(crops))
    assert all(not (x >= 5 and y >= 5) for x, y in (*pastures, *crops))


def test_two_quadrant_research_variant_preserves_route_capacity():
    agent = ReactiveRouteAgent(route_kwargs={"max_quadrants": 2})
    v = View.from_obs(_obs(unlocked=("NW", "NE")))
    pastures, crops = agent._route_slots(v)

    assert len(pastures) == 13
    assert len(crops) == 32
    assert all(y < 5 for _x, y in (*pastures, *crops))


def test_two_quadrant_variant_keeps_full_compact_capacity():
    agent = ReactiveRouteAgent(route_kwargs={"max_quadrants": 2})
    v = View.from_obs(_obs(unlocked=("NW", "NE")))
    pastures, crops = agent._route_slots(v)

    assert len(pastures) == 13
    assert len(crops) == 32
    assert not (set(pastures) & set(crops))
    assert all(y < 5 for _, y in (*pastures, *crops))
    assert not agent._make_plan(v).buy_land


def test_plan_exposes_only_slots_in_current_one_two_or_three_quadrants():
    agent = ReactiveRouteAgent(route_kwargs={"max_quadrants": 3})
    previous_pastures = 0
    previous_crops = 0
    for unlocked in (("NW",), ("NW", "NE"), ("NW", "NE", "SW")):
        plan = agent._make_plan(View.from_obs(_obs(unlocked=unlocked)))
        assert len(plan.pasture_slots) > previous_pastures
        assert len(plan.crop_targets) > previous_crops
        assert all(
            (("N" if y < 5 else "S") + ("W" if x < 5 else "E")) in unlocked
            for x, y in (*plan.pasture_slots, *plan.crop_targets)
        )
        previous_pastures = len(plan.pasture_slots)
        previous_crops = len(plan.crop_targets)


def test_plan_never_buys_fourth_quadrant_and_keeps_liquidation_labour():
    agent = ReactiveRouteAgent()
    v = View.from_obs(_obs(day=27, unlocked=("NW", "NE", "SW"), hands=0, money=100000))
    plan = agent._make_plan(v)
    assert not plan.buy_land
    assert plan.target_hands == 10
    assert plan.hires_wanted == 10


def test_default_route_stops_after_the_compact_second_quadrant():
    agent = ReactiveRouteAgent()
    v = View.from_obs(_obs(day=12, unlocked=("NW", "NE"), money=100000))

    assert agent.rp.max_quadrants == 2
    assert not agent._make_plan(v).buy_land


def test_episode_state_resets_on_player_change_and_clock_rollback():
    agent = ReactiveRouteAgent()
    first = View.from_obs(_obs(day=4, hour=5))
    agent._begin_turn(first)
    agent.memory.commitments[0] = ((1, 1), "WATER")

    switched_obs = _obs(day=4, hour=6)
    switched_obs["player"] = 1
    switched = View.from_obs(switched_obs)
    agent._begin_turn(switched)
    assert agent.memory.player == 1
    assert agent.memory.commitments == {}

    agent.memory.commitments[0] = ((2, 2), "CARE")
    rolled_back = View.from_obs(_obs(day=0, hour=0))
    agent._begin_turn(rolled_back)
    assert agent.memory.player == 0
    assert agent.memory.last_step == -1
    assert agent.memory.commitments == {}


def test_market_supply_estimate_subtracts_own_orders():
    agent = ReactiveRouteAgent()
    initial = View.from_obs(_obs(day=1, hour=1))
    agent._begin_turn(initial)
    agent._remember_market([["SELL", "MILK", 4]])

    later_obs = _obs(day=1, hour=2)
    later_obs["market"]["inventory"]["MILK"] += 10
    agent._begin_turn(View.from_obs(later_obs))
    # Market rose by ten, of which four units were ours.
    assert agent.memory.opponent_supply_ema["MILK"] == 0.18 * 6


def test_flex_and_herd_changes_require_persistent_signals():
    agent = ReactiveRouteAgent(route_kwargs={"shop_herd_decision_day": None})
    view = View.from_obs(_obs(day=3))

    agent._crop_score = lambda _v, crop: 10.0 if crop == "CARROT" else 1.0  # type: ignore[method-assign]
    agent._update_flex_crop(view)
    assert agent.memory.flex_crop == "TOMATO"
    agent._update_flex_crop(view)
    assert agent.memory.flex_crop == "CARROT"

    agent._animal_signal = lambda _v, animal: 2.0 if animal == "COW" else 1.0  # type: ignore[method-assign]
    agent._update_herd(view)
    assert agent.memory.herd_target == (8, 5)
    agent._update_herd(view)
    assert agent.memory.herd_target == agent.rp.milk_heavy


def test_irreversible_herd_expansion_waits_for_confirmed_market_signal():
    agent = ReactiveRouteAgent(route_kwargs={"herd_commit_day": 6})
    agent.memory.herd_target = agent.rp.wool_heavy

    assert agent._phased_herd_target(5) == (2, 2)
    assert agent._phased_herd_target(6) == agent.rp.wool_heavy


def test_default_herd_uses_robust_balanced_staging():
    agent = ReactiveRouteAgent()
    agent.memory.herd_target = agent.rp.milk_heavy

    assert agent._phased_herd_target(3) == (2, 2)
    assert agent._phased_herd_target(4) == (4, 3)
    assert agent._phased_herd_target(7) == (6, 3)
    assert agent._phased_herd_target(10) == agent.rp.milk_heavy


def test_shop_evidence_locks_irreversible_herd_route():
    agent = ReactiveRouteAgent()
    obs = _obs(day=6)
    obs["town"]["unlocked_shops"] = ["YARN_STORE", "FARMERS_MARKET"]
    view = View.from_obs(obs)

    agent._update_herd(view)
    assert agent.memory.herd_target == agent.rp.wool_heavy
    assert agent.memory.herd_locked

    agent._animal_signal = lambda _v, animal: 100.0 if animal == "COW" else 1.0  # type: ignore[method-assign]
    agent._update_herd(view)
    assert agent.memory.herd_target == agent.rp.wool_heavy


def test_daily_unit_zones_are_stable_and_new_hands_fill_unserved_zone():
    agent = ReactiveRouteAgent()
    view = View.from_obs(_obs(day=8, unlocked=("NW", "NE"), hands=1))
    tasks = [
        Task(80, (2, 2), ["DIG"], "DIG"),
        Task(80, (7, 2), ["DIG"], "DIG"),
    ]
    agent._ensure_unit_zones(view, tasks)
    original = dict(agent.memory.unit_zones)

    agent._ensure_unit_zones(view, [Task(100, (7, 2), ["DIG"], "DIG")])
    assert agent.memory.unit_zones == original

    grown = View.from_obs(_obs(day=8, hour=1, unlocked=("NW", "NE"), hands=2))
    agent._ensure_unit_zones(grown, tasks)
    assert len(agent.memory.unit_zones) == 3
    assert set(agent.memory.unit_zones.values()) == {"NW", "NE"}


def test_ordinary_matching_prefers_each_workers_persistent_zone():
    agent = ReactiveRouteAgent()
    view = View.from_obs(_obs(day=8, unlocked=("NW", "NE"), hands=1))
    agent.memory.zone_day = view.day
    agent.memory.unit_zones = {0: "NW", 1: "NE"}
    tasks = [
        Task(80, (3, 4), ["DIG"], "DIG"),
        Task(80, (5, 4), ["DIG"], "DIG"),
    ]

    ops, _ = agent._assign_route_units(view, tasks)

    assert ops == [["WEST"], ["EAST"]]


def test_same_tile_water_precedes_harvest():
    agent = ReactiveRouteAgent()
    obs = _obs(hands=1)
    v = View.from_obs(obs)
    pos = v.farmer_pos
    tasks = [
        Task(98, pos, ["WATER"], "WATER"),
        Task(92, pos, ["HARVEST"], "HARVEST"),
    ]
    ops, _ = agent._assign_route_units(v, tasks)
    assert ops == [["WATER"], ["HARVEST"]]


def test_same_tile_plant_precedes_dependent_water():
    agent = ReactiveRouteAgent()
    obs = _obs(
        hands=1,
        private={
            "shed": {},
            "seeds": {"STRAWBERRY": 1},
            "inventories": [{}, {}],
        },
    )
    v = View.from_obs(obs)
    pos = v.farmer_pos
    tasks = [Task(80, pos, ["PLANT", "STRAWBERRY"], "PLANT", "STRAWBERRY")]

    ops, _ = agent._assign_route_units(v, tasks)

    assert ops == [["PLANT", "STRAWBERRY"], ["WATER"]]

    telemetry = EpisodeTelemetry()
    telemetry.record(
        obs,
        {"farmer": ops[0], "hands": ops[1:], "market": []},
    )
    assert telemetry.avoidable_noops == 0


def test_telemetry_credits_successful_hour_23_preservation_action():
    obs = _obs(day=4, hour=23)
    obs["farms"][0]["tiles"][4][4] = {
        "kind": "PASTURE",
        "animal": "COW",
        "fed_today": True,
        "cared_today": False,
    }
    telemetry = EpisodeTelemetry()

    telemetry.record(
        obs,
        {"farmer": ["CARE"], "hands": [], "market": []},
    )

    assert telemetry.care_opportunities == 1
    assert telemetry.missed_care == 0


def test_inventory_reservation_is_per_carrier():
    agent = ReactiveRouteAgent()
    obs = _obs(
        hands=1,
        private={
            "shed": {},
            "seeds": {},
            "inventories": [{"WHEAT": 2}, {"FERTILIZER": 2}],
        },
    )
    v = View.from_obs(obs)
    tasks = [
        Task(100, (3, 4), ["FEED"], "FEED", requires_item="WHEAT"),
        Task(100, (2, 4), ["FEED"], "FEED", requires_item="WHEAT"),
    ]
    reserved = agent._per_unit_reservations(v, tasks)
    assert reserved == {0: {"WHEAT": 2}}


def test_shadow_shed_caps_a_drop_at_real_shed_capacity():
    agent = ReactiveRouteAgent()
    obs = _obs(
        private={
            "shed": {"WOOL": 98},
            "seeds": {},
            "inventories": [{"MILK": 5}],
        },
    )
    view = View.from_obs(obs)

    shadow = agent._shadow_shed(view, [["DROP"]])

    assert sum(shadow.values()) == 100
    assert shadow["MILK"] == 2


def test_telemetry_counts_units_discarded_by_an_over_capacity_drop():
    obs = _obs(
        private={
            "shed": {"WOOL": 98},
            "seeds": {},
            "inventories": [{"MILK": 5}],
        },
    )
    telemetry = EpisodeTelemetry()

    telemetry.record(obs, {"farmer": ["DROP"], "hands": [], "market": []})

    assert telemetry.discarded_drop_units == 3


def test_fertilizer_timing_variant_skips_applications_that_expire_before_yield():
    agent = ReactiveRouteAgent(route_kwargs={"fertilizer_start_slack": 2})
    obs = _obs(day=7)
    obs["farms"][0]["tiles"][0][0] = {
        "kind": "PLANT",
        "crop": "STRAWBERRY",
        "planted_day": 0,
        "yield_units": 0,
        "fertilized_until_day": -1,
    }
    early = View.from_obs(obs)
    assert not agent._fertilizer_is_worth_using(early, early.tile((0, 0)))

    obs["day"] = 8
    obs["step"] = 8 * 24
    useful = View.from_obs(obs)
    assert agent._fertilizer_is_worth_using(useful, useful.tile((0, 0)))


def test_market_sells_fund_buys_without_reordering_them_after_purchases():
    agent = ReactiveRouteAgent()
    obs = _obs(
        day=6,
        hands=8,
        money=1200,
        private={
            "shed": {"MILK": 20, "WHEAT": 40},
            "seeds": {},
            "inventories": [{} for _ in range(9)],
        },
    )
    v = View.from_obs(obs)
    plan = agent._make_plan(v)
    orders = agent._route_market_orders(v, plan, [["PASS"] for _ in range(v.n_units)])
    sell_indices = [i for i, order in enumerate(orders) if order[0] == "SELL"]
    buy_indices = [i for i, order in enumerate(orders) if order[0].startswith("BUY")]
    assert len(orders) <= 10
    assert sell_indices
    assert not buy_indices or max(sell_indices) < min(buy_indices)


def test_late_empty_long_crop_slots_are_reused_for_short_crops():
    agent = ReactiveRouteAgent()
    day_24 = View.from_obs(_obs(day=24, unlocked=("NW", "NE"), money=100000))
    plan_24 = agent._make_plan(day_24)
    assert set(plan_24.crop_targets.values()) == {"WHEAT"}

    day_26 = View.from_obs(_obs(day=26, unlocked=("NW", "NE"), money=100000))
    plan_26 = agent._make_plan(day_26)
    assert "CARROT" in plan_26.crop_targets.values()
    assert "STRAWBERRY" not in plan_26.crop_targets.values()


def test_visible_opponent_production_raises_market_supply_pressure():
    agent = ReactiveRouteAgent()
    empty = _obs(day=10, unlocked=("NW", "NE"))
    low = agent._market_supply_pressure(View.from_obs(empty), "MILK")

    crowded = _obs(day=10, unlocked=("NW", "NE"))
    for x in range(5):
        crowded["farms"][1]["tiles"][0][x] = {
            "kind": "PASTURE",
            "animal": "COW",
        }
    high = agent._market_supply_pressure(View.from_obs(crowded), "MILK")
    assert high > low


def test_route_cluster_bonus_counts_unique_pending_neighbours_in_same_zone():
    agent = ReactiveRouteAgent(
        route_kwargs={"cluster_bonus": 8.0, "cluster_radius": 2, "cluster_cap": 2}
    )
    view = View.from_obs(_obs(day=10, unlocked=("NW", "NE")))
    tasks = [
        Task(10, (1, 1), ["WATER"], "WATER"),
        Task(10, (2, 1), ["WATER"], "WATER"),
        Task(10, (2, 1), ["HARVEST"], "HARVEST"),  # same position counts once
        Task(10, (1, 3), ["WATER"], "WATER"),
        Task(10, (6, 1), ["WATER"], "WATER"),       # different quadrant
    ]

    assert agent._route_cluster_bonus(view, 0, tasks[0], tasks, set(), set()) == 16.0
    assert agent._route_cluster_bonus(view, 0, tasks[0], tasks, {1, 2}, set()) == 8.0
    assert agent._route_cluster_bonus(view, 0, tasks[0], tasks, set(), {(1, 3)}) == 8.0


def test_terminal_turn_drops_and_sells_every_available_product():
    agent = ReactiveRouteAgent()
    obs = _obs(
        day=29,
        hour=22,
        hands=0,
        money=1000,
        private={
            "shed": {"MILK": 50},
            "seeds": {"STRAWBERRY": 4},
            "inventories": [{"STRAWBERRY": 2}],
        },
    )
    action = agent.decide(View.from_obs(obs))
    sold = {order[1]: order[2] for order in action["market"] if order[0] == "SELL"}

    assert action["farmer"] == ["DROP"]
    assert sold["MILK"] == 50
    assert sold["STRAWBERRY"] == 2
    assert not any(order[0].startswith("BUY") for order in action["market"])


def test_terminal_harvest_is_skipped_when_it_cannot_reach_the_shed():
    agent = ReactiveRouteAgent()
    obs = _obs(day=29, hour=22, unlocked=("NW", "NE"), money=100000)
    initial = View.from_obs(obs)
    pos = next(iter(agent._make_plan(initial).crop_targets))
    x, y = pos
    obs["farms"][0]["tiles"][y][x] = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": 25,
        "yield_units": 4,
        "watered_today": True,
    }
    view = View.from_obs(obs)
    tasks = agent._route_tasks(view, agent._make_plan(view))

    assert not any(task.kind == "HARVEST" for task in tasks)


def test_episode_telemetry_is_populated_for_reactive_agent():
    result = run_episode(
        [ReactiveRouteAgent(), "starter"],
        seed=1,
        time_seat=0,
        telemetry_seat=0,
    )
    assert result.statuses[0] == "DONE"
    assert result.telemetry is not None
    assert result.telemetry.total_unit_ops > 5000
    assert result.telemetry.avoidable_noops == 0
    assert result.telemetry.max_quadrants <= 2
    assert result.telemetry.final_unsold_units <= 3
    assert result.telemetry.movement_rate <= 0.48
    assert result.telemetry.pass_rate <= 0.26
    assert result.telemetry.productive_rate >= 0.31
    assert result.telemetry.water_completion >= 0.99
    assert result.telemetry.feed_completion >= 0.98
    assert result.telemetry.care_completion >= 0.98
    assert result.telemetry.discarded_drop_units == 0
    assert len(result.telemetry.daily_crops) == 29
    assert len(result.telemetry.daily_animals) == 29
    assert result.max_turn_time < 0.5
