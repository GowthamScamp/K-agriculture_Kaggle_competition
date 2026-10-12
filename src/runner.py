"""Episode plumbing: build the environment, run agents, collect results.

Separated from `env_adapter` because this half *does* import
`kaggle_environments` and therefore must never be reachable from
`submission/main.py`.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .diagnostics import error_counters, preview_units

Agent = Callable[..., dict[str, Any]]

DEFAULT_EPISODE_STEPS = 720

MOVE_OPS = {"NORTH", "SOUTH", "EAST", "WEST"}
PRODUCTIVE_OPS = {
    "PICKUP", "PLACE", "DROP", "PLANT", "WATER", "HARVEST",
    "FERTILIZE", "BUILD_COOP", "BUILD_PASTURE", "FEED",
    "COLLECT_FERTILIZER", "CARE", "DIG",
}


@dataclass(slots=True)
class EpisodeTelemetry:
    """In-memory behavioral diagnostics for one callable agent.

    The competition reward only exposes the final bank. These counters make a
    regression actionable: they show whether money was lost to routing, idle
    labour, missed daily care, over-expansion, or failed liquidation. Nothing is
    written to disk and no replay data is required.
    """

    unit_ops: Counter[str] = field(default_factory=Counter)
    successful_unit_ops: Counter[str] = field(default_factory=Counter)
    noop_unit_ops: Counter[str] = field(default_factory=Counter)
    market_ops: Counter[str] = field(default_factory=Counter)
    market_units: Counter[str] = field(default_factory=Counter)
    crop_plants: Counter[str] = field(default_factory=Counter)
    daily_bank: list[tuple[int, float]] = field(default_factory=list)
    daily_hands: list[tuple[int, int]] = field(default_factory=list)
    daily_crops: list[tuple[int, dict[str, int]]] = field(default_factory=list)
    daily_animals: list[tuple[int, dict[str, int]]] = field(default_factory=list)
    avoidable_noops: int = 0
    water_opportunities: int = 0
    missed_water: int = 0
    feed_opportunities: int = 0
    missed_feed: int = 0
    care_opportunities: int = 0
    missed_care: int = 0
    max_hands: int = 0
    max_quadrants: int = 1
    final_unsold_units: int = 0
    final_unsold_value: float = 0.0
    final_plants: int = 0
    final_animals: int = 0
    final_weeds: int = 0
    final_crops: Counter[str] = field(default_factory=Counter)
    final_animal_types: Counter[str] = field(default_factory=Counter)
    final_shops: tuple[str, ...] = ()
    final_prices: dict[str, int] = field(default_factory=dict)
    discarded_drop_units: int = 0
    input_drop_units: int = 0
    discarded_nightly_units: int = 0
    executed_market_units: Counter[str] = field(default_factory=Counter)
    _turns_per_day: int = 24

    def record(self, obs: Any, action: dict[str, Any], config: Any = None) -> None:
        farms = obs.get("farms") or []
        player = int(obs.get("player") or 0)
        farm = farms[player] if player < len(farms) else {}
        hands = farm.get("hands") or []
        self.max_hands = max(self.max_hands, len(hands))
        self.max_quadrants = max(
            self.max_quadrants, len(farm.get("unlocked_quadrants") or ["NW"])
        )

        self._turns_per_day = int((config or {}).get("turnsPerDay", 24))
        post_farm, _, effects = preview_units(obs, action, config)
        unit_actions = [action.get("farmer") or ["PASS"], *(action.get("hands") or [])]
        for effect in effects:
            index, name = effect.actor, effect.operation
            self.unit_ops[name] += 1
            if effect.changed:
                self.successful_unit_ops[name] += 1
            else:
                self.noop_unit_ops[name] += 1
                self.avoidable_noops += int(name != "PASS")
            op = unit_actions[index] if index < len(unit_actions) else ["PASS"]
            if name == "PLANT" and effect.changed and len(op) > 1:
                self.crop_plants[str(op[1])] += 1
            self.discarded_drop_units += effect.discarded
            if name == "DROP" and effect.changed:
                inventories = (obs.get("private") or {}).get("inventories") or []
                inventory = inventories[index] if index < len(inventories) else {}
                self.input_drop_units += sum(max(0, int(inventory.get(item, 0)))
                                             for item in ("WHEAT", "FERTILIZER", "COW", "SHEEP", "GOOSE"))
        for order in action.get("market") or []:
            if order:
                operation = str(order[0])
                self.market_ops[operation] += 1
                if len(order) >= 3:
                    self.market_units[f"{operation}:{order[1]}"] += int(order[2])

        day = int(obs.get("day") or 0)
        hour = int(obs.get("hour") or 0)
        if hour == self._turns_per_day - 1:
            self.daily_bank.append((day, float(farm.get("money") or 0.0)))
            self.daily_hands.append((day, len(hands)))
            day_crops: Counter[str] = Counter()
            day_animals: Counter[str] = Counter()
            for y, row in enumerate(post_farm.get("tiles") or []):
                for x, tile in enumerate(row):
                    if not isinstance(tile, dict):
                        continue
                    if tile.get("kind") == "PLANT":
                        if tile.get("crop"):
                            day_crops[str(tile["crop"])] += 1
                        self.water_opportunities += 1
                        self.missed_water += int(
                            not tile.get("watered_today", False)
                        )
                    if "animal" in tile:
                        if tile.get("animal"):
                            day_animals[str(tile["animal"])] += 1
                        self.feed_opportunities += 1
                        self.care_opportunities += 1
                        self.missed_feed += int(
                            not tile.get("fed_today", False)
                        )
                        self.missed_care += int(
                            not tile.get("cared_today", False)
                        )
            self.daily_crops.append((day, dict(day_crops)))
            self.daily_animals.append((day, dict(day_animals)))


    def finalize(self, obs: Any) -> None:
        farms = obs.get("farms") or []
        player = int(obs.get("player") or 0)
        farm = farms[player] if player < len(farms) else {}
        day = int(obs.get("day") or 0)
        if not self.daily_bank or self.daily_bank[-1][0] != day:
            self.daily_bank.append((day, float(farm.get("money") or 0)))
        private = obs.get("private") or {}
        prices = ((obs.get("market") or {}).get("prices") or {})
        self.final_prices = {str(item): int(price) for item, price in prices.items()}
        self.final_shops = tuple((obs.get("town") or {}).get("unlocked_shops") or ())
        inventories = [private.get("shed") or {}, *(private.get("inventories") or [])]
        for inventory in inventories:
            for item, raw_n in inventory.items():
                n = max(0, int(raw_n or 0))
                self.final_unsold_units += n
                self.final_unsold_value += n * float(prices.get(item, 0) or 0)
        for row in farm.get("tiles") or []:
            for tile in row:
                if not isinstance(tile, dict):
                    continue
                kind = tile.get("kind")
                self.final_plants += int(kind == "PLANT")
                self.final_animals += int("animal" in tile)
                self.final_weeds += int(kind == "WEED")
                if kind == "PLANT" and tile.get("crop"):
                    self.final_crops[str(tile["crop"])] += 1
                if tile.get("animal"):
                    self.final_animal_types[str(tile["animal"])] += 1

    @property
    def total_unit_ops(self) -> int:
        return sum(self.unit_ops.values())

    @property
    def movement_rate(self) -> float:
        moves = sum(self.unit_ops[op] for op in MOVE_OPS)
        return moves / self.total_unit_ops if self.total_unit_ops else 0.0

    @property
    def pass_rate(self) -> float:
        return self.unit_ops["PASS"] / self.total_unit_ops if self.total_unit_ops else 0.0

    @property
    def productive_rate(self) -> float:
        productive = sum(self.successful_unit_ops[op] for op in PRODUCTIVE_OPS)
        return productive / self.total_unit_ops if self.total_unit_ops else 0.0

    @staticmethod
    def _completion(opportunities: int, missed: int) -> float:
        return 1.0 if opportunities <= 0 else 1.0 - missed / opportunities

    @property
    def water_completion(self) -> float:
        return self._completion(self.water_opportunities, self.missed_water)

    @property
    def feed_completion(self) -> float:
        return self._completion(self.feed_opportunities, self.missed_feed)

    @property
    def care_completion(self) -> float:
        return self._completion(self.care_opportunities, self.missed_care)


@contextlib.contextmanager
def _quiet():
    """Silence the OpenSpiel banner kaggle_environments emits when it loads.

    The chatter comes from a C++ extension writing straight to fd 1/2, so
    `redirect_stdout` alone does not catch it - the file descriptors themselves
    have to be swapped.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = (os.dup(1), os.dup(2))
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in (*saved, devnull):
            os.close(fd)


def preload_env() -> None:
    """Import kaggle_environments once, quietly, before anything prints."""
    with _quiet():
        import kaggle_environments  # noqa: F401


def make_env(seed: int | None = None, **config: Any) -> Any:
    """Create a kaggriculture environment with a deterministic seed.

    Returns `kaggle_environments`' own `Environment`, which has no importable
    type to annotate against - hence `Any` rather than a fabricated alias.
    """
    with _quiet():
        from kaggle_environments import make

    cfg: dict[str, Any] = {"episodeSteps": DEFAULT_EPISODE_STEPS}
    cfg.update(config)
    if seed is not None:
        cfg["seed"] = int(seed)
    with _quiet():
        return make("kaggriculture", configuration=cfg, debug=False)


@dataclass(slots=True)
class EpisodeResult:
    """Outcome of one head-to-head game."""

    seed: int
    rewards: list[float]
    statuses: list[str]
    steps: int
    wall_time: float
    turn_times: list[float] = field(default_factory=list)
    telemetry: EpisodeTelemetry | None = None
    diagnostic_errors: list[dict[str, int]] = field(default_factory=list)

    @property
    def bank(self) -> float:
        """Final bank of seat 0 - the agent under test."""
        return self.rewards[0] if self.rewards else 0.0

    @property
    def opponent_bank(self) -> float:
        return self.rewards[1] if len(self.rewards) > 1 else 0.0

    @property
    def won(self) -> bool:
        return self.bank > self.opponent_bank

    @property
    def margin(self) -> float:
        return self.bank - self.opponent_bank

    @property
    def max_turn_time(self) -> float:
        return max(self.turn_times) if self.turn_times else 0.0

    @property
    def mean_turn_time(self) -> float:
        return sum(self.turn_times) / len(self.turn_times) if self.turn_times else 0.0


def _timed(agent: Agent, sink: list[float]) -> Agent:
    """Wrap an agent so every decision's wall time is recorded."""

    def wrapped(obs: Any, config: Any = None) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            return agent(obs, config) if _takes_config(agent) else agent(obs)
        finally:
            sink.append(time.perf_counter() - t0)

    return wrapped


def _instrumented(agent: Agent, sink: EpisodeTelemetry) -> Agent:
    """Wrap a callable agent and record the observation/action stream."""

    def wrapped(obs: Any, config: Any = None) -> dict[str, Any]:
        action = agent(obs, config) if _takes_config(agent) else agent(obs)
        sink.record(obs, action, config)
        return action

    return wrapped


def _takes_config(agent: Agent) -> bool:
    try:
        import inspect

        return len(inspect.signature(agent).parameters) >= 2
    except (TypeError, ValueError):
        return False


@contextlib.contextmanager
def _execution_hooks(env: Any, sink: EpisodeTelemetry | None, seat: int | None):
    """Observe actual market commits and nightly losses, restoring every hook.

This process-local instrumentation is only enabled for telemetry episodes.
Parallel evaluations use processes, never threads sharing engine functions.
"""
    if sink is None or seat is None:
        yield
        return
    from .simulator import _load_env_module

    kg = _load_env_module()
    original_market, original_commit, original_night = kg._process_market, kg._commit_unit, kg._end_of_day
    own_private = [None]

    def committed(op, item, price, farm, private, market, shed_capacity=100):
        success = original_commit(op, item, price, farm, private, market, shed_capacity)
        if success and private is own_private[0]:
            sink.executed_market_units[f"{op}:{item}"] += 1
        return success

    def market(state, active_env):
        own_private[0] = state[seat].observation.private
        original_market(state, active_env)

    def night(state, active_env, day):
        private = state[seat].observation.private
        before = sum(private["shed"].values())
        carried = sum(sum(inv.values()) for inv in private["inventories"])
        money = float(state[0].observation.farms[seat]["money"])
        if sink.daily_bank and sink.daily_bank[-1][0] == day:
            sink.daily_bank[-1] = (day, money)
        original_night(state, active_env, day)
        accepted = sum(private["shed"].values()) - before
        sink.discarded_nightly_units += max(0, carried - accepted)

    kg._process_market, kg._commit_unit, kg._end_of_day = market, committed, night
    try:
        yield
    finally:
        kg._process_market, kg._commit_unit, kg._end_of_day = original_market, original_commit, original_night


def run_episode(
    agents: Sequence[Agent | str],
    seed: int = 0,
    time_agent_0: bool = False,
    time_seat: int | None = None,
    telemetry_seat: int | None = None,
    **config: Any,
) -> EpisodeResult:
    """Run one full episode and return the result.

    `agents` may mix callables and built-in names ("pass", "random", "starter").

    Timing follows the agent under test, which is not always seat 0. Pass
    `time_seat` when the agent under test has been moved to seat 1 for a
    seat-swapped evaluation - otherwise the turn times would describe the
    *opponent*, which is both useless and quietly misleading. `time_agent_0`
    remains as the seat-0 shorthand.
    """
    env = make_env(seed=seed, **config)
    turn_times: list[float] = []
    telemetry = EpisodeTelemetry() if telemetry_seat is not None else None
    runnable = list(agents)
    if telemetry_seat is not None and callable(runnable[telemetry_seat]):
        runnable[telemetry_seat] = _instrumented(runnable[telemetry_seat], telemetry)
    seat = time_seat if time_seat is not None else (0 if time_agent_0 else None)
    if seat is not None and callable(runnable[seat]):
        runnable[seat] = _timed(runnable[seat], turn_times)

    t0 = time.perf_counter()
    with _quiet(), _execution_hooks(env, telemetry, telemetry_seat):
        env.run(runnable)
    wall = time.perf_counter() - t0

    final = env.steps[-1]
    if telemetry is not None:
        telemetry.finalize(final[telemetry_seat]["observation"])
    return EpisodeResult(
        seed=seed,
        rewards=[float(s["reward"] or 0.0) for s in final],
        statuses=[str(s["status"]) for s in final],
        steps=len(env.steps),
        wall_time=wall,
        turn_times=turn_times,
        telemetry=telemetry,
        diagnostic_errors=[error_counters(agent) if callable(agent) else {} for agent in agents],
    )


def _cli() -> int:
    import argparse

    from .registry import build_agent, AGENT_NAMES

    parser = argparse.ArgumentParser(description="Run a single kaggriculture episode.")
    parser.add_argument("--agent", default="do_nothing", help=f"one of {AGENT_NAMES}")
    parser.add_argument("--opponent", default="pass")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true", help="print the final board")
    args = parser.parse_args()

    result = run_episode(
        [build_agent(args.agent), build_agent(args.opponent)],
        seed=args.seed,
        time_agent_0=True,
    )
    print(f"seed={result.seed}  steps={result.steps}  wall={result.wall_time:.1f}s")
    print(f"  seat 0 ({args.agent:>12s}): bank = {result.bank:>12,.0f}  [{result.statuses[0]}]")
    print(f"  seat 1 ({args.opponent:>12s}): bank = {result.opponent_bank:>12,.0f}  [{result.statuses[1]}]")
    if result.turn_times:
        print(f"  turn time: mean {result.mean_turn_time * 1000:.1f} ms, "
              f"max {result.max_turn_time * 1000:.1f} ms  (limit 1000 ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
