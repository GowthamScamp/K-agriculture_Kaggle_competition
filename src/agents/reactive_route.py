"""Competitive reactive route agent.

The original baseline chooses globally attractive tasks every turn.  That is a
useful correctness baseline, but it makes workers zig-zag between unrelated
jobs and it bases expansion, labour, and production on the amount of empty land
rather than on a coherent farm layout.  This agent keeps the same battle-tested
action primitives while supplying a stable three-quadrant route, an explicit
herd/crop economy, stateful market pressure, and a purchase queue funded by the
sells that precede it.

The implementation deliberately does not use replay tapes or hidden opponent
state.  Every adaptation is derived from the live public observation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Iterable

from .baseline_agent import BaselineAgent, FarmPlan, Params, Task, _cycle_actions, _cycle_days, _cycle_units
from ..env_adapter import Action, Pos, Tile, View, distance, safe_action, step_toward
from ..game_constants import (
    ANIMALS,
    CROPS,
    LAND_PRICES,
    MARKET_I0,
    MARKET_PARAMS,
    PRODUCTS,
    cumulative_hire_cost,
    daily_town_drain,
    hire_cost,
    marginal_sell_price,
    market_price,
    projected_sell_revenue,
    quadrant_of,
    sell_revenue,
)

SEASON_DAYS = 30
MAX_MARKET_ORDERS = 10
SHED_CAPACITY = 100
TARGET_QUADRANTS = ("NW", "NE", "SW")


@dataclass(frozen=True, slots=True)
class RouteParams:
    """Strategic controls for the production route."""

    max_quadrants: int = 2
    pasture_slots: int = 13
    pasture_distribution: tuple[int, ...] | None = None
    crop_slots: int = 32
    wheat_slots: int = 10
    strawberry_slots: int = 22
    flex_slots: int = 0

    first_expand_day: int = 5
    first_expand_last_day: int = 7
    second_expand_day: int = 8
    second_expand_last_day: int = 11
    first_land_reserve: int = 300
    second_land_reserve: int = 500

    opening_hands: int = 8
    first_expand_hands: int = 10
    second_expand_hands: int = 11
    liquidation_hands: int = 10
    minimum_hands: int = 7
    hire_cash_floor: int = 20
    hire_before_hour: int = 3

    default_cows: int = 8
    default_sheep: int = 5
    milk_heavy: tuple[int, int] = (10, 3)
    wool_heavy: tuple[int, int] = (5, 8)
    herd_switch_ratio_high: float = 1.18
    herd_switch_ratio_low: float = 0.92
    herd_commit_day: int | None = None
    shop_herd_decision_day: int | None = 6
    opening_cows: int = 2
    opening_sheep: int = 2
    herd_stage_two_day: int = 4
    herd_stage_three_day: int = 7
    herd_full_day: int = 10
    animal_last_buy_day: int = 17
    animal_cash_floor: int = 220
    wheat_per_animal: int = 3

    crop_switch_margin: float = 0.15
    switch_confirm_days: int = 2
    max_melon_slots: int = 0
    strawberry_per_day: int = 3
    strawberry_seed_min_cash: int = 1500
    flex_seed_min_cash: int = 500
    plant_before_hour: int = 18
    animal_harvest_slack: int = 2
    care_urgency_hour: int = 18
    endgame_fill_from_day: int | None = 24
    endgame_spare_slots: int = 0

    sell_batch: int = 6
    scarce_sell_batch: int = 6
    pressured_sell_batch: int = 8
    liquidation_sell_batch: int = 30
    normal_sell_floor: float = 0.45
    fertilizer_sell_floor: int = 42
    fertilizer_liquidation_collect_floor: int = 15
    fertilizer_field_reserve: int = 0
    fertilizer_start_slack: int | None = 2
    shed_pressure: float = 0.68
    supply_pressure_low: float = 0.55
    supply_pressure_high: float = 1.05

    route_switch_penalty: float = 9.0
    distance_cost: float = 4.0
    zone_affinity_bonus: float = 14.0
    zone_escape_penalty: float = 8.0
    # Prefer a destination surrounded by other pending work.  The bonus is
    # deliberately disabled in the incumbent configuration so it can be
    # screened on paired seeds before promotion.  A value of 8 matches the
    # scale of the independently validated public MIT route heuristic:
    # distance_cost (4) * two saved route steps per nearby task.
    cluster_bonus: float = 0.0
    cluster_radius: int = 2
    cluster_cap: int = 4
    max_feed_carriers: int = 6
    drop_at_shed: int = 3
    return_with_produce: int = 8
    endgame_dump_day: int = 28
    terminal_step: int = 718


@dataclass(slots=True)
class RoutePlan:
    liquidating: bool
    target_hands: int
    hires_wanted: int
    buy_land: bool
    pasture_slots: tuple[Pos, ...]
    crop_targets: dict[Pos, str]
    build_spots: tuple[Pos, ...]
    animal_orders: list[tuple[str, int]]
    seed_orders: list[tuple[str, int]]
    wheat_reserve: int
    feed_wheat_to_buy: int


@dataclass(slots=True)
class RouteMemory:
    player: int = -1
    last_step: int = -1
    last_day: int = -1
    flex_crop: str = "TOMATO"
    pending_flex: str | None = None
    pending_flex_days: int = 0
    herd_target: tuple[int, int] = (8, 5)
    pending_herd: tuple[int, int] | None = None
    pending_herd_days: int = 0
    herd_locked: bool = False
    previous_market: dict[str, float] = field(default_factory=dict)
    previous_step: int = -1
    own_market_supply: dict[str, float] = field(default_factory=dict)
    opponent_supply_ema: dict[str, float] = field(default_factory=dict)
    commitments: dict[int, tuple[Pos, str]] = field(default_factory=dict)
    unit_zones: dict[int, str] = field(default_factory=dict)
    zone_day: int = -1


def _competitive_baseline_params() -> Params:
    """Execution priorities used by the route agent, leaving DEFAULT_PARAMS intact."""

    return replace(
        Params(),
        MAX_HANDS=11,
        MIN_HANDS=7,
        HIRE_BEFORE_HOUR=3,
        HIRE_CASH_RESERVE=20.0,
        HIRE_FLOOR_IGNORES_RESERVE=True,
        LIQUIDATION_HIRES=True,
        LIQUIDATION_HAND_TARGET=10,
        LAND_LAST_USEFUL_DAY=11,
        LAND_CASH_RESERVE=300.0,
        REALIZED_SHOP_DEMAND=True,
        HOLD_EARLY_DAYS=0,
        P_FEED=100.0,
        P_WATER_CRITICAL=98.0,
        P_CARE=94.0,
        P_HARVEST_URGENT=92.0,
        P_WATER_BONUS=90.0,
        P_HARVEST=88.0,
        P_PLACE_ANIMAL=86.0,
        P_WATER_PLAIN=84.0,
        P_PLANT=80.0,
        P_DIG_WEED=78.0,
        P_DROP=76.0,
        P_BUILD=74.0,
        P_FERTILIZE=64.0,
        P_COLLECT_FERT=52.0,
    )


class ReactiveRouteAgent(BaselineAgent):
    """Stateful public-observation agent with a stable production route."""

    def __init__(
        self,
        route_params: RouteParams | None = None,
        route_kwargs: dict[str, Any] | None = None,
        params: Params | None = None,
    ) -> None:
        super().__init__(params or _competitive_baseline_params())
        rp = route_params or RouteParams()
        if route_kwargs:
            unknown = set(route_kwargs) - set(RouteParams.__dataclass_fields__)
            if unknown:
                raise ValueError(f"unknown RouteParams field(s): {sorted(unknown)}")
            rp = replace(rp, **route_kwargs)
        if rp.wheat_slots + rp.strawberry_slots + rp.flex_slots != rp.crop_slots:
            raise ValueError("route crop-slot partitions must sum to crop_slots")
        if not 1 <= rp.max_quadrants <= len(TARGET_QUADRANTS):
            raise ValueError("max_quadrants must be between one and three")
        if rp.pasture_distribution is not None:
            if len(rp.pasture_distribution) != rp.max_quadrants:
                raise ValueError("pasture_distribution must have one entry per route quadrant")
            if sum(rp.pasture_distribution) != rp.pasture_slots:
                raise ValueError("pasture_distribution must sum to pasture_slots")
            if any(value < 0 for value in rp.pasture_distribution):
                raise ValueError("pasture_distribution entries cannot be negative")
        self.rp = rp
        self.memory = RouteMemory(
            herd_target=(self.rp.default_cows, self.rp.default_sheep)
        )

    def __call__(self, obs: Any, config: Any = None) -> Action:
        n_hands = 0
        try:
            farms = obs.get("farms") or []
            player = int(obs.get("player") or 0)
            if player < len(farms):
                n_hands = len(farms[player].get("hands") or [])
            return self.decide(View.from_obs(obs))
        except Exception as exc:  # pragma: no cover - ladder safety boundary
            self._log_error(exc)
            return safe_action(n_hands)

    def decide(self, v: View) -> Action:
        self._begin_turn(v)
        plan = self._make_plan(v)
        tasks = self._route_tasks(v, plan)
        unit_ops, chosen = self._assign_route_units(v, tasks)
        market = self._route_market_orders(v, plan, unit_ops)
        self._remember_market(market)
        self.memory.last_step = v.step
        self.memory.last_day = v.day
        self._log_route(v, plan, market, chosen)
        return {
            "farmer": unit_ops[0],
            "hands": unit_ops[1:],
            "market": market[:MAX_MARKET_ORDERS],
        }

    # ------------------------------------------------------------------
    # Episode state and public-market inference
    # ------------------------------------------------------------------
    def _begin_turn(self, v: View) -> None:
        reset = (
            self.memory.player != v.player
            or v.step < self.memory.last_step
            or (self.memory.last_day >= 0 and v.day < self.memory.last_day)
        )
        if reset:
            self.memory = RouteMemory(
                player=v.player,
                herd_target=(self.rp.default_cows, self.rp.default_sheep),
            )
        elif v.day != self.memory.last_day:
            # Hands respawn at the shed every morning; yesterday's destinations
            # must not pull the new crew back across the board.
            self.memory.commitments.clear()
            self.memory.unit_zones.clear()
            self.memory.zone_day = v.day

        self._observe_market(v)
        if v.day != self.memory.last_day:
            self._update_flex_crop(v)
            self._update_herd(v)

    def _observe_market(self, v: View) -> None:
        if self.memory.previous_market and self.memory.previous_step >= 0:
            for item in PRODUCTS:
                previous = float(self.memory.previous_market.get(item, MARKET_I0))
                current = float(v.market_inventory.get(item, MARKET_I0))
                own = float(self.memory.own_market_supply.get(item, 0.0))
                # Positive means inferred opponent net supply.  Town consumption
                # is small at turn scale and is removed when the prior turn was a
                # shop/centre tick.
                town = 0.0
                if self.memory.previous_step % 4 == 0:
                    town += max(0.0, daily_town_drain(item, v.day, v.unlocked_shops) / 6.0)
                inferred = (current - previous) - own + town
                old = self.memory.opponent_supply_ema.get(item, 0.0)
                self.memory.opponent_supply_ema[item] = 0.82 * old + 0.18 * inferred
        self.memory.previous_market = dict(v.market_inventory)
        self.memory.previous_step = v.step

    def _remember_market(self, orders: list[list[Any]]) -> None:
        supply = {item: 0.0 for item in PRODUCTS}
        for order in orders:
            if len(order) < 3 or order[1] not in supply:
                continue
            qty = float(order[2])
            if order[0] == "SELL":
                supply[order[1]] += qty
            elif order[0] == "BUY_PRODUCT":
                supply[order[1]] -= qty
        self.memory.own_market_supply = supply

    # ------------------------------------------------------------------
    # Strategic choices
    # ------------------------------------------------------------------
    def _update_flex_crop(self, v: View) -> None:
        scores = {crop: self._crop_score(v, crop) for crop in CROPS}
        viable = {crop: score for crop, score in scores.items() if math.isfinite(score)}
        if not viable:
            return
        best = max(viable, key=viable.get)
        current = viable.get(self.memory.flex_crop, float("-inf"))
        threshold = current * (1.0 + self.rp.crop_switch_margin) if current > 0 else current
        if best == self.memory.flex_crop or viable[best] <= threshold:
            self.memory.pending_flex = None
            self.memory.pending_flex_days = 0
            return
        if self.memory.pending_flex == best:
            self.memory.pending_flex_days += 1
        else:
            self.memory.pending_flex = best
            self.memory.pending_flex_days = 1
        if self.memory.pending_flex_days >= self.rp.switch_confirm_days:
            self.memory.flex_crop = best
            self.memory.pending_flex = None
            self.memory.pending_flex_days = 0

    def _crop_score(self, v: View, crop: str) -> float:
        data = CROPS[crop]
        days_left = SEASON_DAYS - v.day
        first_return = data["first_yield_day"] + 1
        if first_return >= days_left:
            return float("-inf")
        if crop == "MELON" and v.day > 17:
            return float("-inf")
        units = _cycle_units(crop)
        cycle = min(_cycle_days(crop), days_left - 1)
        inventory = float(v.market_inventory.get(crop, MARKET_I0))
        opponent_daily = max(0.0, self.memory.opponent_supply_ema.get(crop, 0.0)) * 24.0
        own_future = self._public_crop_count(v.farm, crop) * units / max(1, cycle)
        adjusted = inventory + (opponent_daily + own_future) * cycle
        revenue = projected_sell_revenue(
            crop,
            units,
            adjusted,
            v.day,
            v.day + cycle,
            v.unlocked_shops,
        ) - data["seed"]
        per_action = revenue / max(1, _cycle_actions(crop))
        per_day = revenue / max(1, cycle)
        return 0.72 * per_action + 0.28 * per_day

    def _update_herd(self, v: View) -> None:
        if self.memory.herd_locked:
            return
        if self.rp.shop_herd_decision_day is not None:
            if v.day < self.rp.shop_herd_decision_day:
                return
            # Livestock cannot be converted after purchase, so route selection
            # is based on durable shop demand rather than a one-day spot price.
            # The first two public shop draws are known by day six; a yarn store
            # is the decisive wool signal because its single-product pull is 2x.
            self.memory.herd_target = (
                self.rp.wool_heavy
                if "YARN_STORE" in v.unlocked_shops
                else self.rp.milk_heavy
            )
            self.memory.pending_herd = None
            self.memory.pending_herd_days = 0
            self.memory.herd_locked = True
            return
        milk = self._animal_signal(v, "COW")
        wool = self._animal_signal(v, "SHEEP")
        ratio = milk / max(1e-9, wool)
        desired = (self.rp.default_cows, self.rp.default_sheep)
        if ratio >= self.rp.herd_switch_ratio_high:
            desired = self.rp.milk_heavy
        elif ratio <= self.rp.herd_switch_ratio_low:
            desired = self.rp.wool_heavy
        if desired == self.memory.herd_target:
            self.memory.pending_herd = None
            self.memory.pending_herd_days = 0
            return
        if self.memory.pending_herd == desired:
            self.memory.pending_herd_days += 1
        else:
            self.memory.pending_herd = desired
            self.memory.pending_herd_days = 1
        if self.memory.pending_herd_days >= self.rp.switch_confirm_days:
            self.memory.herd_target = desired
            self.memory.pending_herd = None
            self.memory.pending_herd_days = 0

    def _animal_signal(self, v: View, animal: str) -> float:
        product = ANIMALS[animal]["product"]
        # Targets already encode the stable cow/sheep production economics.
        # This bounded adjustment measures only the relative market signal so
        # a live price/shop/supply change does not double-count static yield.
        price = max(1, v.price(product)) / MARKET_PARAMS[product]["base"]
        town = daily_town_drain(product, v.day, v.unlocked_shops)
        rival = self._public_animal_count(v.opponent, animal)
        supply = max(0.0, self.memory.opponent_supply_ema.get(product, 0.0))
        return price * (1.0 + 0.035 * town) / (1.0 + 0.08 * rival + 0.12 * supply)

    @staticmethod
    def _public_crop_count(farm: dict[str, Any] | None, crop: str) -> int:
        if not farm:
            return 0
        return sum(
            1
            for row in farm.get("tiles") or []
            for tile in row
            if isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") == crop
        )

    @staticmethod
    def _public_animal_count(farm: dict[str, Any] | None, animal: str) -> int:
        if not farm:
            return 0
        return sum(
            1
            for row in farm.get("tiles") or []
            for tile in row
            if isinstance(tile, dict) and tile.get("animal") == animal
        )

    # ------------------------------------------------------------------
    # Layout and plan
    # ------------------------------------------------------------------
    def _route_slots(self, v: View) -> tuple[tuple[Pos, ...], tuple[Pos, ...]]:
        sheds = set(v.shed_tiles())
        route_quadrants = TARGET_QUADRANTS[: self.rp.max_quadrants]

        def ranked(quadrant: str) -> list[Pos]:
            positions = [
                (x, y)
                for y in range(v.board_size)
                for x in range(v.board_size)
                if quadrant_of(x, y, v.board_size) == quadrant and (x, y) not in sheds
            ]
            positions.sort(key=lambda p: (min(distance(p, s) for s in sheds), p[1], p[0]))
            return positions

        by_q = {q: ranked(q) for q in route_quadrants}
        if self.rp.pasture_distribution is None:
            base_pasture_quota, pasture_remainder = divmod(
                self.rp.pasture_slots, len(route_quadrants)
            )
            pasture_quota = {
                quadrant: base_pasture_quota + int(index < pasture_remainder)
                for index, quadrant in enumerate(route_quadrants)
            }
        else:
            pasture_quota = dict(zip(route_quadrants, self.rp.pasture_distribution))
        base_crop_quota, crop_remainder = divmod(
            self.rp.crop_slots, len(route_quadrants)
        )
        crop_quota = {
            quadrant: base_crop_quota + int(index < crop_remainder)
            for index, quadrant in enumerate(route_quadrants)
        }
        pastures: list[Pos] = []
        crops: list[Pos] = []
        for quadrant in route_quadrants:
            chosen = by_q[quadrant][: pasture_quota[quadrant]]
            pastures.extend(chosen)
            remainder = [p for p in by_q[quadrant] if p not in set(chosen)]
            crops.extend(remainder[: crop_quota[quadrant]])
        return tuple(pastures[: self.rp.pasture_slots]), tuple(crops[: self.rp.crop_slots])

    def _make_plan(self, v: View) -> RoutePlan:
        liquidating = v.day >= 27
        target_hands = self._target_hands(v, liquidating)
        hires = 0
        if v.hour <= self.rp.hire_before_hour and v.day < SEASON_DAYS - 1:
            hires = max(0, target_hands - v.n_hands)

        pastures, crop_slots = self._route_slots(v)
        unlocked = set(v.unlocked_quadrants)
        available_pastures = tuple(
            p for p in pastures if quadrant_of(*p, v.board_size) in unlocked
        )
        available_crops = tuple(
            p for p in crop_slots if quadrant_of(*p, v.board_size) in unlocked
        )

        crop_targets: dict[Pos, str] = {}
        melon_targets = 0
        for index, pos in enumerate(crop_slots):
            if pos not in available_crops:
                continue
            if index < self.rp.wheat_slots:
                crop_targets[pos] = "WHEAT"
            elif index < self.rp.wheat_slots + self.rp.strawberry_slots:
                crop_targets[pos] = "STRAWBERRY"
            else:
                flex = self.memory.flex_crop
                if flex == "MELON":
                    if melon_targets >= self.rp.max_melon_slots:
                        flex = "TOMATO"
                    else:
                        melon_targets += 1
                crop_targets[pos] = flex

        if (
            self.rp.endgame_fill_from_day is not None
            and v.day >= self.rp.endgame_fill_from_day
            and not liquidating
        ):
            # Ongoing strawberry/tomato slots become permanently idle once a
            # new plant cannot mature. Reuse only *empty* route slots for a
            # short crop that can still pay back before the terminal turn.
            fill_crop = "WHEAT" if v.day <= 25 else "CARROT"
            if self._crop_can_return(fill_crop, v.day):
                for pos in tuple(crop_targets):
                    if v.tile(pos).empty and not self._crop_can_return(crop_targets[pos], v.day):
                        crop_targets[pos] = fill_crop
                if self.rp.endgame_spare_slots > 0:
                    reserved = set(available_pastures) | set(crop_targets)
                    spare = sorted(
                        (
                            tile.pos
                            for tile in v.owned_tiles()
                            if tile.empty and tile.pos not in reserved
                        ),
                        key=lambda pos: min(
                            distance(pos, shed) for shed in v.owned_shed_tiles()
                        ),
                    )
                    for pos in spare[: self.rp.endgame_spare_slots]:
                        crop_targets[pos] = fill_crop

        desired_cows, desired_sheep = self._phased_herd_target(v.day)
        desired_total = min(len(available_pastures), desired_cows + desired_sheep)
        housing = sum(
            1
            for pos in available_pastures
            if self._is_pasture_or_animal(v.tile(pos))
        )
        missing_housing = max(0, desired_total - housing)
        build_spots = tuple(
            pos for pos in available_pastures if v.tile(pos).empty
        )[: min(2, missing_housing)]

        owned = {
            animal: self._owned_animal_count(v, animal)
            for animal in ("COW", "SHEEP")
        }
        free_housing = max(0, housing + len(build_spots) - sum(owned.values()))
        animal_orders: list[tuple[str, int]] = []
        if v.day <= self.rp.animal_last_buy_day and free_housing > 0:
            for animal, target in (("COW", desired_cows), ("SHEEP", desired_sheep)):
                missing = max(0, target - owned[animal])
                qty = min(2, missing, free_housing)
                if qty > 0:
                    animal_orders.append((animal, qty))
                    free_housing -= qty

        seed_orders: list[tuple[str, int]] = []
        if not liquidating:
            needs: dict[str, int] = {}
            for pos, crop in crop_targets.items():
                tile = v.tile(pos)
                if tile.empty:
                    needs[crop] = needs.get(crop, 0) + 1
            crop_order = dict.fromkeys(
                ("WHEAT", "STRAWBERRY", self.memory.flex_crop, *crop_targets.values())
            )
            for crop in crop_order:
                need = needs.get(crop, 0)
                held = int(v.seeds.get(crop, 0))
                cap = self.rp.strawberry_per_day + 2 if crop == "STRAWBERRY" else 20
                qty = max(0, min(cap, need + 2 - held))
                if qty > 0:
                    seed_orders.append((crop, qty))

        n_animals = sum(
            1 for pos in available_pastures if v.tile(pos).has_animal
        )
        wheat_reserve = n_animals * self.rp.wheat_per_animal
        wheat_total = self._total_private(v, "WHEAT")
        # Animals still produce at the end of days 27 and 28.  Selling all feed
        # merely because crop planting stopped silently discards that income.
        feed_buy = 0 if v.day >= 29 else max(0, wheat_reserve + 3 - wheat_total)

        return RoutePlan(
            liquidating=liquidating,
            target_hands=target_hands,
            hires_wanted=hires,
            buy_land=self._should_expand(v),
            pasture_slots=available_pastures,
            crop_targets=crop_targets,
            build_spots=build_spots,
            animal_orders=animal_orders,
            seed_orders=seed_orders,
            wheat_reserve=wheat_reserve,
            feed_wheat_to_buy=feed_buy,
        )

    def _target_hands(self, v: View, liquidating: bool) -> int:
        if liquidating:
            return self.rp.liquidation_hands
        quadrants = len(v.unlocked_quadrants)
        if quadrants >= 3:
            return self.rp.second_expand_hands
        if quadrants >= 2:
            return self.rp.first_expand_hands
        return self.rp.opening_hands

    def _should_expand(self, v: View) -> bool:
        n = len(v.unlocked_quadrants)
        if n >= self.rp.max_quadrants:
            return False
        if n == 1:
            return (
                self.rp.first_expand_day <= v.day <= self.rp.first_expand_last_day
                and v.money >= LAND_PRICES[0] + self.rp.first_land_reserve
            )
        return (
            self.rp.second_expand_day <= v.day <= self.rp.second_expand_last_day
            and v.money >= LAND_PRICES[1] + self.rp.second_land_reserve
        )

    def _phased_herd_target(self, day: int) -> tuple[int, int]:
        full_cows, full_sheep = self.memory.herd_target
        # Optional research variant: defer all irreversible expansion until the
        # configured day.  It is not the default because early shop/price data
        # proved too noisy in matched simulation screens.
        if self.rp.herd_commit_day is not None and day < self.rp.herd_commit_day:
            return (
                min(self.rp.opening_cows, full_cows),
                min(self.rp.opening_sheep, full_sheep),
            )
        if self.rp.herd_commit_day is not None:
            return full_cows, full_sheep
        if day < self.rp.herd_stage_two_day:
            return min(2, full_cows), min(2, full_sheep)
        if day < self.rp.herd_stage_three_day:
            return min(4, full_cows), min(3, full_sheep)
        if day < self.rp.herd_full_day:
            return min(6, full_cows), min(4, full_sheep)
        return full_cows, full_sheep

    @staticmethod
    def _is_pasture_or_animal(tile: Tile) -> bool:
        return tile.kind == "PASTURE" or tile.has_animal

    @staticmethod
    def _total_private(v: View, item: str) -> int:
        return int(v.shed.get(item, 0)) + sum(
            int(v.unit_inventory(i).get(item, 0)) for i in range(v.n_units)
        )

    def _owned_animal_count(self, v: View, animal: str) -> int:
        placed = sum(1 for t in v.owned_tiles() if t.animal == animal)
        return placed + self._total_private(v, animal)

    # ------------------------------------------------------------------
    # Work generation
    # ------------------------------------------------------------------
    def _route_tasks(self, v: View, plan: RoutePlan) -> list[Task]:
        tasks: list[Task] = []
        available_fertilizer = self._total_private(v, "FERTILIZER")
        strawberries_planted_today = sum(
            1
            for tile in v.owned_tiles()
            if tile.crop == "STRAWBERRY" and tile.planted_day == v.day
        )

        for pos in plan.pasture_slots:
            tile = v.tile(pos)
            if tile.has_animal:
                if v.day < 29 and not tile.fed_today:
                    tasks.append(Task(self.p.P_FEED, pos, ["FEED"], "FEED", requires_item="WHEAT"))
                if v.day < 29 and not tile.cared_today:
                    care_priority = (
                        self.p.P_WATER_CRITICAL
                        if v.hour >= self.rp.care_urgency_hour
                        else self.p.P_CARE
                    )
                    tasks.append(Task(care_priority, pos, ["CARE"], "CARE"))
                animal_info = ANIMALS.get(tile.animal or "", {})
                harvest_at = max(
                    2,
                    int(animal_info.get("max_held", 2)) - self.rp.animal_harvest_slack,
                )
                if (tile.yield_units >= harvest_at or (
                    plan.liquidating and tile.yield_units > 0
                )) and self._can_harvest_and_bank(v, pos):
                    tasks.append(Task(self.p.P_HARVEST_URGENT, pos, ["HARVEST"], "HARVEST"))
                if tile.fertilizer_available and self._fertilizer_is_worth_collecting(v, plan):
                    tasks.append(Task(self.p.P_COLLECT_FERT, pos, ["COLLECT_FERTILIZER"], "COLLECT_FERT"))
            elif tile.kind == "PASTURE":
                available = self._available_route_animal(v)
                if available:
                    tasks.append(Task(
                        self.p.P_PLACE_ANIMAL,
                        pos,
                        ["PLACE", available],
                        "PLACE_ANIMAL",
                        requires_item=available,
                    ))
            elif tile.is_weed:
                tasks.append(Task(self.p.P_DIG_WEED, pos, ["DIG"], "DIG"))
            elif tile.empty and pos in plan.build_spots:
                tasks.append(Task(self.p.P_BUILD, pos, ["BUILD_PASTURE"], "BUILD"))

        available_seeds = dict(v.seeds)
        for pos, target_crop in plan.crop_targets.items():
            tile = v.tile(pos)
            if tile.is_plant:
                if tile.needs_water(v.day):
                    priority = self.p.P_WATER_CRITICAL if v.hour >= 12 else (
                        self.p.P_WATER_BONUS if tile.in_water_bonus_window(v.day) else self.p.P_WATER_PLAIN
                    )
                    tasks.append(Task(priority, pos, ["WATER"], "WATER", tile.crop))
                if tile.harvestable(v.day) and self._can_harvest_and_bank(v, pos):
                    priority = self.p.P_HARVEST_URGENT if (
                        self._fully_ripe(tile, v.day) or plan.liquidating
                    ) else self.p.P_HARVEST
                    tasks.append(Task(priority, pos, ["HARVEST"], "HARVEST", tile.crop))
                if (
                    available_fertilizer > 0
                    and tile.fertilized_until_day < v.day
                    and tile.in_water_bonus_window(v.day)
                    and self._fertilizer_is_worth_using(v, tile)
                ):
                    tasks.append(Task(
                        self.p.P_FERTILIZE,
                        pos,
                        ["FERTILIZE"],
                        "FERTILIZE",
                        tile.crop,
                        requires_item="FERTILIZER",
                    ))
            elif tile.is_weed:
                tasks.append(Task(self.p.P_DIG_WEED, pos, ["DIG"], "DIG"))
            elif tile.empty and not plan.liquidating and v.hour <= self.rp.plant_before_hour:
                if self._crop_can_return(target_crop, v.day):
                    if (
                        target_crop == "STRAWBERRY"
                        and strawberries_planted_today >= self.rp.strawberry_per_day
                    ):
                        continue
                    held = int(available_seeds.get(target_crop, 0))
                    if held > 0:
                        tasks.append(Task(
                            self.p.P_PLANT,
                            pos,
                            ["PLANT", target_crop],
                            "PLANT",
                            target_crop,
                        ))
                        available_seeds[target_crop] = held - 1
                        if target_crop == "STRAWBERRY":
                            strawberries_planted_today += 1

        return tasks

    def _can_harvest_and_bank(self, v: View, pos: Pos) -> bool:
        if v.day < self.rp.endgame_dump_day:
            return True
        return_steps = min(distance(pos, shed) for shed in v.owned_shed_tiles())
        actions_left = self.rp.terminal_step - v.step + 1
        # HARVEST + movement to the shed + DROP must all execute by step 718.
        return return_steps + 2 <= actions_left

    def _crop_can_return(self, crop: str, day: int) -> bool:
        return day + int(CROPS[crop]["first_yield_day"]) + 1 < SEASON_DAYS

    def _fertilizer_is_worth_collecting(self, v: View, plan: RoutePlan) -> bool:
        if plan.liquidating:
            return v.price("FERTILIZER") >= self.rp.fertilizer_liquidation_collect_floor
        if v.price("FERTILIZER") >= self.rp.fertilizer_sell_floor:
            return True
        return any(
            t.is_plant and t.in_water_bonus_window(v.day) and t.fertilized_until_day < v.day
            for t in v.owned_tiles()
        )

    def _fertilizer_is_worth_using(self, v: View, tile: Tile) -> bool:
        crop = tile.crop
        # Fertilizer cover produces the clearest labour-adjusted gain on the
        # ongoing, high-value strawberry cycle.  Applying it to feed wheat made
        # wheat hit its cap early, which triggered extra harvest/replant work
        # and consumed fertilizer that could add far more crop revenue here.
        if crop != "STRAWBERRY":
            return False
        if self.rp.fertilizer_start_slack is not None:
            first_yield = int(CROPS[crop]["first_yield_day"])
            if tile.age_days(v.day) < first_yield - self.rp.fertilizer_start_slack:
                return False
        opportunity = max(self.rp.fertilizer_sell_floor, v.price("FERTILIZER"))
        extra_units = 2 if CROPS[crop]["ongoing"] else 1
        return extra_units * v.price(crop) >= opportunity * 1.20

    def _available_route_animal(self, v: View) -> str | None:
        cows, sheep = self.memory.herd_target
        counts = {
            "COW": self._public_animal_count(v.farm, "COW"),
            "SHEEP": self._public_animal_count(v.farm, "SHEEP"),
        }
        for animal, target in (("COW", cows), ("SHEEP", sheep)):
            if counts[animal] >= target:
                continue
            if self._total_private(v, animal) > 0:
                return animal
        return None

    # ------------------------------------------------------------------
    # Route-aware unit assignment
    # ------------------------------------------------------------------
    def _assign_route_units(
        self, v: View, tasks: list[Task]
    ) -> tuple[list[list[Any]], list[dict[str, Any]]]:
        ops: list[list[Any]] = [["PASS"] for _ in range(v.n_units)]
        chosen: list[dict[str, Any]] = []
        free = set(range(v.n_units))
        self._ensure_unit_zones(v, tasks)

        reservations = self._per_unit_reservations(v, tasks)
        for idx in list(free):
            drop = self._drop_task(v, idx, reservations.get(idx))
            if drop is None:
                continue
            ops[idx] = self._execute(v, idx, drop)
            chosen.append(self._record(idx, drop, ops[idx]))
            free.remove(idx)
            self._set_commitment(v, idx, drop, ops[idx])

        if v.step >= self.rp.terminal_step:
            for idx in free:
                ops[idx] = ["PASS"]
                self.memory.commitments.pop(idx, None)
            return ops, chosen

        self._assign_feed_pickups(v, tasks, free, ops, chosen)
        self._assign_one_pickup(v, tasks, free, ops, chosen, "ANIMAL")
        self._assign_one_pickup(v, tasks, free, ops, chosen, "FERTILIZER")

        used_tasks: set[int] = set()
        used_tiles: set[Pos] = set()

        # Plant and water an empty tile in one turn when two workers already
        # share it.  The lower unit index plants first because the environment
        # executes units in index order; the synthetic water task must never be
        # exposed to the general matcher while the observed tile is still empty.
        for ti, task in enumerate(tasks):
            if task.kind != "PLANT" or task.pos in used_tiles:
                continue
            standing = sorted(i for i in free if v.unit_pos(i) == task.pos)
            if len(standing) < 2:
                continue
            first, second = standing[:2]
            water = Task(
                self.p.P_WATER_CRITICAL,
                task.pos,
                ["WATER"],
                "WATER_AFTER_PLANT",
                task.crop,
            )
            for idx, chained_task in ((first, task), (second, water)):
                ops[idx] = list(chained_task.op)
                chosen.append(self._record(idx, chained_task, ops[idx]))
                free.remove(idx)
                self.memory.commitments.pop(idx, None)
            used_tasks.add(ti)
            used_tiles.add(task.pos)

        # Execute other valid two-action tile chains only when both workers
        # already stand on the tile; lower unit index acts first.
        by_pos: dict[Pos, dict[str, int]] = {}
        for ti, task in enumerate(tasks):
            by_pos.setdefault(task.pos, {})[task.kind] = ti
        for first_kind, second_kind in (("WATER", "HARVEST"),):
            for pos, kinds in by_pos.items():
                if pos in used_tiles:
                    continue
                if first_kind not in kinds or second_kind not in kinds:
                    continue
                first_ti, second_ti = kinds[first_kind], kinds[second_kind]
                if first_ti in used_tasks or second_ti in used_tasks:
                    continue
                standing = sorted(i for i in free if v.unit_pos(i) == pos)
                pair = next(
                    (
                        (first, second)
                        for first in standing
                        for second in standing
                        if first < second and self._unit_can_do(v, first, tasks[first_ti])
                    ),
                    None,
                )
                if pair is None:
                    continue
                for idx, chained_ti in zip(pair, (first_ti, second_ti)):
                    chained_task = tasks[chained_ti]
                    ops[idx] = list(chained_task.op)
                    chosen.append(self._record(idx, chained_task, ops[idx]))
                    free.remove(idx)
                    used_tasks.add(chained_ti)
                    self.memory.commitments.pop(idx, None)
                used_tiles.add(pos)

        # Feeding and late critical watering preserve future production.  Their
        # priority must not be diluted by distance in the general matcher: a
        # nearby low-value task is not a valid substitute for a hungry animal.
        mandatory_pairs: list[tuple[float, int, int]] = []
        for ti, task in enumerate(tasks):
            if ti in used_tasks or task.priority < self.p.P_WATER_CRITICAL:
                continue
            for idx in free:
                if self._unit_can_do(v, idx, task):
                    travel = distance(v.unit_pos(idx), task.pos)
                    same_zone = self._unit_zone(idx) == self._task_zone(v, task)
                    affinity = self.rp.zone_affinity_bonus if same_zone else 0.0
                    mandatory_pairs.append((task.priority + affinity - travel, idx, ti))
        mandatory_pairs.sort(key=lambda row: (-row[0], row[1], row[2]))
        for _score, idx, ti in mandatory_pairs:
            if idx not in free or ti in used_tasks:
                continue
            task = tasks[ti]
            if task.pos in used_tiles:
                continue
            ops[idx] = self._execute(v, idx, task)
            chosen.append(self._record(idx, task, ops[idx]))
            free.remove(idx)
            used_tasks.add(ti)
            used_tiles.add(task.pos)
            self._set_commitment(v, idx, task, ops[idx])
            if not free:
                break

        # A productive action on the current tile has zero travel cost and
        # should almost always beat preserving an old route commitment.  Doing
        # this before matching cuts the common "walk away, then come back"
        # pattern while priorities still decide between multiple local jobs.
        for idx in sorted(list(free)):
            position = v.unit_pos(idx)
            if position in used_tiles:
                continue
            local = [
                (ti, task)
                for ti, task in enumerate(tasks)
                if ti not in used_tasks
                and task.pos == position
                and self._unit_can_do(v, idx, task)
            ]
            if not local:
                continue
            ti, task = max(local, key=lambda row: (row[1].priority, -row[0]))
            ops[idx] = list(task.op)
            chosen.append(self._record(idx, task, ops[idx]))
            free.remove(idx)
            used_tasks.add(ti)
            used_tiles.add(task.pos)
            self.memory.commitments.pop(idx, None)

        # Honour unfinished destinations before assigning new work.
        for idx in sorted(list(free)):
            commitment = self.memory.commitments.get(idx)
            if commitment is None:
                continue
            pos, kind = commitment
            match = next(
                (
                    (ti, task)
                    for ti, task in enumerate(tasks)
                    if ti not in used_tasks
                    and task.pos == pos
                    and task.kind == kind
                    and pos not in used_tiles
                    and self._unit_can_do(v, idx, task)
                ),
                None,
            )
            if match is None:
                self.memory.commitments.pop(idx, None)
                continue
            ti, task = match
            urgent = max(
                (
                    other.priority
                    for other in tasks
                    if other.pos not in used_tiles
                    and self._unit_can_do(v, idx, other)
                    and self._task_zone(v, other) == self._unit_zone(idx)
                ),
                default=task.priority,
            )
            if urgent > task.priority + self.rp.route_switch_penalty:
                self.memory.commitments.pop(idx, None)
                continue
            ops[idx] = self._execute(v, idx, task)
            chosen.append(self._record(idx, task, ops[idx]))
            free.remove(idx)
            used_tasks.add(ti)
            used_tiles.add(task.pos)
            self._set_commitment(v, idx, task, ops[idx])

        # Stable zones are the main route boundary. Match ordinary work inside
        # each worker's quadrant first; only unclaimed work may borrow a worker
        # from an adjacent zone. This prevents a small priority difference from
        # pulling two crews through one another on every observation.
        for zone_only in (True, False):
            pairs: list[tuple[float, int, int]] = []
            for ti, task in enumerate(tasks):
                if ti in used_tasks or task.pos in used_tiles:
                    continue
                task_zone = self._task_zone(v, task)
                for idx in free:
                    if not self._unit_can_do(v, idx, task):
                        continue
                    same_zone = self._unit_zone(idx) == task_zone
                    if zone_only != same_zone:
                        continue
                    travel = distance(v.unit_pos(idx), task.pos)
                    old = self.memory.commitments.get(idx)
                    switch = (
                        self.rp.route_switch_penalty
                        if old and old[0] != task.pos
                        else 0.0
                    )
                    escape = 0.0 if same_zone else self.rp.zone_escape_penalty
                    cluster = self._route_cluster_bonus(
                        v, ti, task, tasks, used_tasks, used_tiles
                    )
                    score = (
                        task.priority
                        - self.rp.distance_cost * travel
                        - switch
                        - escape
                        + cluster
                    )
                    pairs.append((score, idx, ti))
            pairs.sort(key=lambda row: (-row[0], row[1], row[2]))

            for _score, idx, ti in pairs:
                if idx not in free or ti in used_tasks:
                    continue
                task = tasks[ti]
                if task.pos in used_tiles:
                    continue
                ops[idx] = self._execute(v, idx, task)
                chosen.append(self._record(idx, task, ops[idx]))
                free.remove(idx)
                used_tasks.add(ti)
                used_tiles.add(task.pos)
                self._set_commitment(v, idx, task, ops[idx])
                if not free:
                    break
            if not free:
                break

        # Intentional idling is cheaper than walking home and back tomorrow.
        for idx in free:
            ops[idx] = ["PASS"]
            self.memory.commitments.pop(idx, None)
        return ops, chosen

    def _route_cluster_bonus(
        self,
        v: View,
        task_index: int,
        task: Task,
        tasks: list[Task],
        used_tasks: set[int],
        used_tiles: set[Pos],
    ) -> float:
        """Discount travel to a compact bundle of still-actionable work.

        The ordinary matcher is otherwise one-turn greedy: it prefers an
        isolated task that is one square nearer even when a slightly farther
        tile has several useful neighbours.  Counting unique neighbouring
        positions gives the next assignment a cheap route-continuity signal
        without weakening mandatory feeding/watering or same-tile actions.
        """

        if self.rp.cluster_bonus <= 0 or self.rp.cluster_cap <= 0:
            return 0.0
        zone = self._task_zone(v, task)
        nearby: set[Pos] = set()
        for other_index, other in enumerate(tasks):
            if other_index == task_index or other_index in used_tasks:
                continue
            if other.pos == task.pos or other.pos in used_tiles:
                continue
            if self._task_zone(v, other) != zone:
                continue
            if distance(task.pos, other.pos) <= self.rp.cluster_radius:
                nearby.add(other.pos)
                if len(nearby) >= self.rp.cluster_cap:
                    break
        return self.rp.cluster_bonus * min(len(nearby), self.rp.cluster_cap)

    def _ensure_unit_zones(self, v: View, tasks: list[Task]) -> None:
        """Assign each live unit a stable productive quadrant for this day.

        Hands are recreated at the shed every morning and additional hands can
        appear during the first few turns. Existing assignments therefore stay
        fixed while newly created indices join the least-served active zone.
        Task counts are used as weights so a pasture-heavy quadrant receives a
        larger crew on animal-care mornings without causing mid-day reshuffles.
        """

        if self.memory.zone_day != v.day:
            self.memory.unit_zones.clear()
            self.memory.zone_day = v.day
        for idx in tuple(self.memory.unit_zones):
            if idx >= v.n_units:
                self.memory.unit_zones.pop(idx, None)

        allowed = set(TARGET_QUADRANTS[: self.rp.max_quadrants])
        active = [q for q in v.unlocked_quadrants if q in allowed]
        if not active:
            active = ["NW"]
        weights = {q: 1.0 for q in active}
        for task in tasks:
            zone = self._task_zone(v, task)
            if zone in weights:
                weights[zone] += 1.0 + max(0.0, task.priority - self.p.P_BUILD) / 100.0

        for idx in range(v.n_units):
            if self.memory.unit_zones.get(idx) in weights:
                continue
            counts = {
                q: sum(1 for assigned in self.memory.unit_zones.values() if assigned == q)
                for q in active
            }
            zone = min(
                active,
                key=lambda q: (counts[q] + 1.0) / weights[q],
            )
            self.memory.unit_zones[idx] = zone

    def _unit_zone(self, idx: int) -> str:
        return self.memory.unit_zones.get(idx, "NW")

    @staticmethod
    def _task_zone(v: View, task: Task) -> str:
        return quadrant_of(*task.pos, v.board_size)

    def _per_unit_reservations(self, v: View, tasks: list[Task]) -> dict[int, dict[str, int]]:
        reserved: dict[int, dict[str, int]] = {}
        remaining = {
            idx: dict(v.unit_inventory(idx)) for idx in range(v.n_units)
        }
        for task in sorted(tasks, key=lambda t: -t.priority):
            item = task.requires_item
            if not item:
                continue
            carriers = [
                idx for idx in range(v.n_units) if remaining[idx].get(item, 0) > 0
            ]
            if not carriers:
                continue
            task_zone = self._task_zone(v, task)
            idx = min(
                carriers,
                key=lambda i: (
                    self._unit_zone(i) != task_zone,
                    distance(v.unit_pos(i), task.pos),
                ),
            )
            reserved.setdefault(idx, {})[item] = reserved.setdefault(idx, {}).get(item, 0) + 1
            remaining[idx][item] -= 1
        return reserved

    def _drop_task(
        self, v: View, idx: int, reserved: dict[str, int] | None = None
    ) -> Task | None:
        """Return only worthwhile loads, retaining route inputs in the field.

        The control agent flushed every non-empty inventory late each day.  That
        paid for a shed round trip even when the unit carried only feed or one
        harvested item.  Inputs remain available to their assigned worker and
        sellable produce is returned in useful batches; the final two days use
        a threshold of one so nothing is stranded.
        """

        inventory = dict(v.unit_inventory(idx))
        for item, n in (reserved or {}).items():
            inventory[item] = max(0, inventory.get(item, 0) - n)
        if v.day >= self.rp.endgame_dump_day:
            load = sum(inventory.values())
            return_at = 1
            drop_at = 1
        else:
            input_items = {"WHEAT", "FERTILIZER", "COW", "SHEEP", "GOOSE"}
            load = sum(n for item, n in inventory.items() if item not in input_items)
            return_at = self.rp.return_with_produce
            drop_at = self.rp.drop_at_shed
        if load <= 0 or v.shed_room(SHED_CAPACITY) <= 0:
            return None
        position = v.unit_pos(idx)
        target = min(v.owned_shed_tiles(), key=lambda q: distance(position, q))
        threshold = drop_at if position == target else return_at
        if load < threshold:
            return None
        return Task(self.p.P_DROP, target, ["DROP"], "DROP")

    @staticmethod
    def _unit_can_do(v: View, idx: int, task: Task) -> bool:
        return task.requires_item is None or v.unit_inventory(idx).get(task.requires_item, 0) > 0

    def _set_commitment(self, v: View, idx: int, task: Task, op: list[Any]) -> None:
        if op and op[0] in ("NORTH", "SOUTH", "EAST", "WEST"):
            self.memory.commitments[idx] = (task.pos, task.kind)
        else:
            self.memory.commitments.pop(idx, None)

    def _assign_feed_pickups(
        self,
        v: View,
        tasks: list[Task],
        free: set[int],
        ops: list[list[Any]],
        chosen: list[dict[str, Any]],
    ) -> None:
        hungry_by_zone: dict[str, int] = {}
        for task in tasks:
            if task.kind == "FEED":
                zone = self._task_zone(v, task)
                hungry_by_zone[zone] = hungry_by_zone.get(zone, 0) + 1
        if not hungry_by_zone or not free:
            return

        # Feed is only useful in the zone that contains the hungry animal. An
        # aggregate inventory check let three units carry all wheat in NW while
        # NE/SW animals starved. Account for carried wheat by route and allocate
        # new three-feed bundles to each zone's uncovered demand.
        carried_by_zone = {zone: 0 for zone in hungry_by_zone}
        for idx in range(v.n_units):
            zone = self._unit_zone(idx)
            if zone in carried_by_zone:
                carried_by_zone[zone] += int(v.unit_inventory(idx).get("WHEAT", 0))
        deficits = {
            zone: max(0, hungry - carried_by_zone.get(zone, 0))
            for zone, hungry in hungry_by_zone.items()
        }
        stock = int(v.shed.get("WHEAT", 0))
        carriers_left = min(self.rp.max_feed_carriers, len(free))
        while stock > 0 and carriers_left > 0 and any(deficits.values()):
            destination = max(deficits, key=lambda zone: (deficits[zone], zone))
            if deficits[destination] <= 0:
                break
            candidates = sorted(
                free,
                key=lambda i: (
                    self._unit_zone(i) != destination,
                    min(distance(v.unit_pos(i), s) for s in v.owned_shed_tiles()),
                    i,
                ),
            )
            if not candidates:
                break
            idx = candidates[0]
            self.memory.unit_zones[idx] = destination
            qty = min(3, deficits[destination], stock)
            target = min(v.owned_shed_tiles(), key=lambda s: distance(v.unit_pos(idx), s))
            task = Task(self.p.P_FEED, target, ["PICKUP", "WHEAT", qty], "PICKUP_FEED")
            ops[idx] = self._execute(v, idx, task)
            chosen.append(self._record(idx, task, ops[idx]))
            free.remove(idx)
            self._set_commitment(v, idx, task, ops[idx])
            deficits[destination] -= qty
            stock -= qty
            carriers_left -= 1

    def _assign_one_pickup(
        self,
        v: View,
        tasks: list[Task],
        free: set[int],
        ops: list[list[Any]],
        chosen: list[dict[str, Any]],
        kind: str,
    ) -> None:
        if not free:
            return
        if kind == "ANIMAL":
            target_task = next(
                (
                    task
                    for task in tasks
                    if task.kind == "PLACE_ANIMAL"
                    and task.requires_item in ("COW", "SHEEP")
                    and v.shed.get(task.requires_item, 0) > 0
                ),
                None,
            )
            item = target_task.requires_item if target_task else None
            needed = target_task is not None
            priority = self.p.P_PLACE_ANIMAL
        else:
            item = "FERTILIZER" if v.shed.get("FERTILIZER", 0) > 0 else None
            needed = any(task.kind == "FERTILIZE" for task in tasks)
            priority = self.p.P_FERTILIZE
        if not item or not needed:
            return
        idx = min(
            free,
            key=lambda i: min(distance(v.unit_pos(i), s) for s in v.owned_shed_tiles()),
        )
        target = min(v.owned_shed_tiles(), key=lambda s: distance(v.unit_pos(idx), s))
        qty = 1 if kind == "ANIMAL" else min(3, int(v.shed.get(item, 0)))
        task = Task(priority, target, ["PICKUP", item, qty], f"PICKUP_{kind}")
        if kind == "ANIMAL" and target_task is not None:
            self.memory.unit_zones[idx] = self._task_zone(v, target_task)
        elif kind == "FERTILIZER":
            fertilizer_targets = [t for t in tasks if t.kind == "FERTILIZE"]
            if fertilizer_targets:
                nearest_target = min(
                    fertilizer_targets,
                    key=lambda t: distance(v.unit_pos(idx), t.pos),
                )
                self.memory.unit_zones[idx] = self._task_zone(v, nearest_target)
        ops[idx] = self._execute(v, idx, task)
        chosen.append(self._record(idx, task, ops[idx]))
        free.remove(idx)
        self._set_commitment(v, idx, task, ops[idx])

    # ------------------------------------------------------------------
    # Market queue
    # ------------------------------------------------------------------
    def _route_market_orders(
        self, v: View, plan: RoutePlan, unit_ops: list[list[Any]]
    ) -> list[list[Any]]:
        shadow = self._shadow_shed(v, unit_ops)
        sells = self._route_sell_orders(v, plan, shadow)
        # First discover which purchases are economically wanted using every
        # candidate sale, then reserve their order slots. Reprice a second time
        # using only the sales that really fit, so no purchase depends on revenue
        # from an order that the ten-order limit will omit.
        optimistic_budget, optimistic_inventory = self._funded_market_state(v, sells)
        wanted_buys, wanted_hires = self._route_purchase_orders(
            v,
            plan,
            optimistic_budget,
            optimistic_inventory,
            MAX_MARKET_ORDERS,
        )
        sell_capacity = max(
            0, MAX_MARKET_ORDERS - len(wanted_buys) - len(wanted_hires)
        )
        included_sells = sells[:sell_capacity]
        budget, shadow_inventory = self._funded_market_state(v, included_sells)
        buys, hires = self._route_purchase_orders(
            v,
            plan,
            budget,
            shadow_inventory,
            MAX_MARKET_ORDERS - len(included_sells),
        )
        return (included_sells + buys + hires)[:MAX_MARKET_ORDERS]

    @staticmethod
    def _funded_market_state(
        v: View, sells: list[list[Any]]
    ) -> tuple[float, dict[str, float]]:
        budget = float(v.money)
        shadow_inventory = dict(v.market_inventory)
        for _, item, qty in sells:
            budget += sell_revenue(item, int(qty), shadow_inventory.get(item, MARKET_I0))
            shadow_inventory[item] = shadow_inventory.get(item, MARKET_I0) + int(qty)
        return budget, shadow_inventory

    def _route_purchase_orders(
        self,
        v: View,
        plan: RoutePlan,
        budget: float,
        shadow_inventory: dict[str, float],
        slot_limit: int,
    ) -> tuple[list[list[Any]], list[list[Any]]]:
        buys: list[list[Any]] = []
        hires: list[list[Any]] = []
        if plan.buy_land:
            land_index = len(v.unlocked_quadrants) - 1
            cost = LAND_PRICES[land_index]
            if budget >= cost and len(buys) < slot_limit:
                buys.append(["BUY_LAND"])
                budget -= cost

        if plan.feed_wheat_to_buy > 0 and len(buys) < 4:
            inventory = float(shadow_inventory.get("WHEAT", MARKET_I0))
            price = market_price("WHEAT", inventory - 1)
            affordable = max(
                0,
                int((budget - self.rp.hire_cash_floor) // max(1, price)),
            )
            qty = min(plan.feed_wheat_to_buy, affordable)
            if qty > 0:
                buys.append(["BUY_PRODUCT", "WHEAT", qty])
                budget -= qty * price

        for animal, wanted in plan.animal_orders:
            if len(buys) >= 4:
                break
            cost = ANIMALS[animal]["cost"]
            feed_floor = max(
                self.rp.animal_cash_floor,
                plan.wheat_reserve * max(25, v.price("WHEAT")),
            )
            qty = min(wanted, max(0, int((budget - feed_floor) // cost)))
            if qty > 0:
                buys.append(["BUY_ANIMAL", animal, qty])
                budget -= qty * cost

        for crop, wanted in plan.seed_orders:
            if len(buys) >= 5:
                break
            cost = int(CROPS[crop]["seed"])
            seed_floor = self.rp.hire_cash_floor
            if crop == "STRAWBERRY":
                seed_floor = max(seed_floor, self.rp.strawberry_seed_min_cash)
            elif crop != "WHEAT":
                seed_floor = max(seed_floor, self.rp.flex_seed_min_cash)
            qty = min(wanted, max(0, int((budget - seed_floor) // cost)))
            if qty > 0:
                buys.append(["BUY_SEED", crop, qty])
                budget -= qty * cost

        for offset in range(plan.hires_wanted):
            if len(hires) >= 8 or len(buys) + len(hires) >= slot_limit:
                break
            cost = hire_cost(v.hires_today + offset)
            # Guarantee the cheap minimum crew, then honour the small cash floor.
            guaranteed = v.n_hands + offset < self.rp.minimum_hands
            floor = 0 if guaranteed else self.rp.hire_cash_floor
            if budget - cost < floor:
                break
            hires.append(["HIRE"])
            budget -= cost

        return buys, hires

    def _shadow_shed(self, v: View, unit_ops: list[list[Any]]) -> dict[str, int]:
        shadow = {k: int(n) for k, n in v.shed.items()}
        for idx, op in enumerate(unit_ops):
            if not op or not v.is_shed_tile(v.unit_pos(idx)):
                continue
            if op[0] == "DROP":
                for item, n in v.unit_inventory(idx).items():
                    room = max(0, SHED_CAPACITY - sum(shadow.values()))
                    accepted = min(room, max(0, int(n)))
                    if accepted > 0:
                        shadow[item] = shadow.get(item, 0) + accepted
            elif op[0] == "PICKUP" and len(op) >= 3:
                item, qty = str(op[1]), int(op[2])
                shadow[item] = max(0, shadow.get(item, 0) - min(qty, shadow.get(item, 0)))
            elif op[0] == "PLACE" and len(op) >= 3:
                item, qty = str(op[1]), int(op[2])
                room = max(0, SHED_CAPACITY - sum(shadow.values()))
                accepted = min(
                    room,
                    max(0, qty),
                    max(0, int(v.unit_inventory(idx).get(item, 0))),
                )
                if accepted > 0:
                    shadow[item] = shadow.get(item, 0) + accepted
        return shadow

    def _route_sell_orders(
        self, v: View, plan: RoutePlan, shadow: dict[str, int]
    ) -> list[list[Any]]:
        fill = sum(max(0, n) for n in shadow.values()) / SHED_CAPACITY
        terminal = v.step >= self.rp.terminal_step
        capital_gap = self._planned_capital_gap(v, plan)
        rows: list[tuple[float, str, int]] = []
        for item in PRODUCTS:
            held = int(shadow.get(item, 0))
            if item == "WHEAT" and v.day < 29 and not terminal:
                held -= plan.wheat_reserve
            if item == "FERTILIZER" and not plan.liquidating:
                strawberry_count = sum(
                    1 for tile in v.owned_tiles() if tile.crop == "STRAWBERRY"
                )
                held -= min(self.rp.fertilizer_field_reserve, strawberry_count)
            if held <= 0:
                continue
            price = v.price(item)
            base = MARKET_PARAMS[item]["base"]
            if item == "FERTILIZER" and not plan.liquidating:
                if price < self.rp.fertilizer_sell_floor and fill < self.rp.shed_pressure:
                    continue
            if not plan.liquidating and fill < self.rp.shed_pressure:
                if price < base * self.rp.normal_sell_floor:
                    continue
            pressure = self._market_supply_pressure(v, item)
            if terminal:
                cap = held
            elif plan.liquidating:
                cap = self.rp.liquidation_sell_batch
            elif capital_gap > 0 or pressure >= self.rp.supply_pressure_high:
                cap = self.rp.pressured_sell_batch
            elif pressure <= self.rp.supply_pressure_low:
                cap = self.rp.scarce_sell_batch
            else:
                cap = self.rp.sell_batch
            qty = min(held, cap)
            if plan.liquidating and v.day < 29:
                qty = min(qty, max(1, math.ceil(held / max(1, SEASON_DAYS - v.day))))
            inv = float(v.market_inventory.get(item, MARKET_I0))
            revenue = sell_revenue(item, qty, inv)
            rows.append((revenue, item, qty))
        rows.sort(reverse=True)
        return [["SELL", item, qty] for _revenue, item, qty in rows]

    def _planned_capital_gap(self, v: View, plan: RoutePlan) -> float:
        needed = 0.0
        if plan.buy_land:
            needed += LAND_PRICES[len(v.unlocked_quadrants) - 1]
        if plan.feed_wheat_to_buy > 0:
            buy_price = market_price(
                "WHEAT", v.market_inventory.get("WHEAT", MARKET_I0) - 1
            )
            needed += plan.feed_wheat_to_buy * buy_price
        needed += sum(ANIMALS[animal]["cost"] * qty for animal, qty in plan.animal_orders)
        needed += sum(CROPS[crop]["seed"] * qty for crop, qty in plan.seed_orders)
        needed += sum(
            hire_cost(v.hires_today + offset) for offset in range(plan.hires_wanted)
        )
        return max(0.0, needed + self.rp.hire_cash_floor - v.money)

    def _market_supply_pressure(self, v: View, item: str) -> float:
        town = max(1.0, daily_town_drain(item, v.day, v.unlocked_shops))
        own = self._farm_daily_supply(v.farm, item)
        opponent_visible = self._farm_daily_supply(v.opponent, item)
        opponent_observed = max(
            0.0, self.memory.opponent_supply_ema.get(item, 0.0) * 24.0
        )
        return (own + max(opponent_visible, opponent_observed)) / town

    @staticmethod
    def _farm_daily_supply(farm: dict[str, Any] | None, item: str) -> float:
        if not farm:
            return 0.0
        total = 0.0
        for row in farm.get("tiles") or []:
            for raw in row:
                if not isinstance(raw, dict):
                    continue
                crop = raw.get("crop") if raw.get("kind") == "PLANT" else None
                if crop == item and crop in CROPS:
                    total += _cycle_units(crop) / max(1, _cycle_days(crop))
                animal = raw.get("animal")
                if animal in ANIMALS:
                    if item == "FERTILIZER":
                        total += 1.0
                    elif ANIMALS[animal]["product"] == item:
                        # Consistent feed+care banks roughly three units per
                        # production interval in the mature herd.
                        total += 3.0 / max(1, ANIMALS[animal]["interval"])
        return total

    def _log_route(
        self,
        v: View,
        plan: RoutePlan,
        market: list[list[Any]],
        chosen: list[dict[str, Any]],
    ) -> None:
        if not self._log_handle:
            return
        # Reuse the baseline logger with a compatible lightweight FarmPlan.
        proxy = FarmPlan(
            liquidating=plan.liquidating,
            work_capacity=0.0,
            hires_wanted=plan.hires_wanted,
            buy_land=plan.buy_land,
            plant_crop=self.memory.flex_crop,
            plant_budget=len(plan.crop_targets),
            plant_spots=set(plan.crop_targets),
            seeds_to_buy=sum(q for _, q in plan.seed_orders),
            build_spots=set(plan.build_spots),
            build_kind="PASTURE",
            animal_to_buy=plan.animal_orders[0][0] if plan.animal_orders else None,
            wheat_reserve=plan.wheat_reserve,
            feed_wheat_to_buy=plan.feed_wheat_to_buy,
        )
        self._log(v, proxy, market, chosen)


_DEFAULT = ReactiveRouteAgent()


def reactive_route_agent(obs: Any, config: Any = None) -> Action:
    return _DEFAULT(obs, config)
