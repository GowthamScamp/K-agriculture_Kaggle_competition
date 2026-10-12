"""Phase 3: forward model. `simulate_step(state, my_action, opp_action) -> state`.

The plan called for building an approximate model and calibrating it against
replays. That turned out to be the wrong job: the competition ships its own
`interpreter()` as a plain function over state dicts, so this module **reuses the
real turn logic** instead of imitating it. Fidelity is therefore exact by
construction, and `tests/test_fidelity.py` asserts that rather than measuring a
tolerable error.

Two things that are genuinely ours to solve:

1. **Cloning cost.** MCTS clones state thousands of times per decision, and
   `copy.deepcopy` on a structified two-seat state is far too slow. `clone_state`
   below is a hand-written shallow-deep hybrid that copies exactly the mutable
   parts and shares everything immutable.
2. **Hidden opponent state.** `interpreter` needs *both* seats' actions and
   private state, but an agent only ever sees its own. `state_from_obs` fills the
   opponent's private state with a stated assumption, which is precisely the seam
   `determinization.py` plugs into.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Iterable, KeysView

from .env_adapter import Action, View
from .game_constants import ANIMALS, CROPS, PRODUCTS

# The environment module is imported lazily so this file stays importable in
# contexts where kaggle_environments is absent (the search simply disables).
_KG: Any = None
_STRUCTIFY: Any = None


def _load_env_module() -> Any:
    """Import the live kaggriculture module, or return None if unavailable."""
    global _KG, _STRUCTIFY
    if _KG is not None:
        return _KG
    try:
        import contextlib
        import io
        import os
        import sys

        # The package prints an OpenSpiel banner from C++ on import.
        devnull = os.open(os.devnull, os.O_WRONLY)
        saved = (os.dup(1), os.dup(2))
        try:
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                from kaggle_environments.envs.kaggriculture import kaggriculture as kg
                from kaggle_environments.utils import structify
        finally:
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            for fd in (*saved, devnull):
                os.close(fd)
        _KG = kg
        _STRUCTIFY = structify
    except Exception:  # pragma: no cover - environment simply not present
        _KG = False
        _STRUCTIFY = False
    return _KG


class Node:
    """Attribute-and-item accessible bag, used for state entries and observations.

    `kaggle_environments.structify` would also work, but it walks the object
    graph recursively and so re-wraps all 200 board tiles on every construction -
    which measured at 691 us per clone, an order of magnitude too slow to search
    with. This wrapper is deliberately flat: nested dicts stay plain dicts,
    exactly as `interpreter` expects them.
    """

    __slots__ = ("__dict__",)

    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)

    # `get(d, key, default)` in the environment falls back to getattr for
    # non-dicts, but View and the tests build plain dicts from observations.
    def get(self, key: str, default: Any = None) -> Any:
        return self.__dict__.get(key, default)

    def keys(self) -> KeysView[str]:
        return self.__dict__.keys()

    def __getitem__(self, key: str) -> Any:
        return self.__dict__[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.__dict__[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__


def simulator_available() -> bool:
    """True when the real turn logic can be reused for search."""
    return bool(_load_env_module())


DEFAULT_CONFIG: dict[str, Any] = {
    "episodeSteps": 720,
    "actTimeout": 1,
    "boardSize": 10,
    "startingMoney": 3000,
    "maxMarketOrdersPerTurn": 10,
    "turnsPerDay": 24,
    "shedCapacity": 100,
    "weedSpawnChance": 0.005,
    "townShopUnlockInterval": 3,
    "townShopSellInterval": 4,
    "townCenterSellInterval": 24,
    "farmHandCostMult": 1,
    "marketParams": {},
}


# ==========================================================================
# Minimal env shim
# ==========================================================================
class _Cfg(dict):
    """Configuration that supports both `cfg["k"]` and `cfg.k`.

    `interpreter` reaches for `cfg.episodeSteps` directly in the terminal check
    while using `get(cfg, ...)` everywhere else, so both must work.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class EnvShim:
    """The three attributes `interpreter` actually touches on `env`."""

    __slots__ = ("configuration", "info", "done")

    def __init__(self, config: dict[str, Any] | None = None, seed: int = 0) -> None:
        cfg = dict(DEFAULT_CONFIG)
        if config:
            cfg.update(config)
        self.configuration = _Cfg(cfg)
        self.info: dict[str, Any] = {"seed": int(seed)}
        self.done = False


# ==========================================================================
# State construction and cloning
# ==========================================================================
def _blank_private() -> dict[str, Any]:
    return {
        "shed": {item: 0 for item in PRODUCTS + list(ANIMALS)},
        "seeds": {crop: 0 for crop in CROPS},
        "inventories": [{}],
    }


def state_from_obs(
    obs: Any,
    opponent_private: dict[str, Any] | None = None,
    step: int | None = None,
) -> list[Any]:
    """Rebuild a two-seat interpreter state from one agent's observation.

    The opponent's shed, seeds and unit inventories are never observable. They
    are filled from `opponent_private` when a model supplies one, and otherwise
    assumed empty - an explicit, wrong-but-bounded assumption. It affects only
    what the opponent could sell, which shows up as market inventory drift.
    """
    kg = _load_env_module()
    if not kg:
        raise RuntimeError("kaggriculture environment unavailable; cannot simulate")

    o = dict(obs)
    player = int(o.get("player", 0))
    farms = _clone_farms(o.get("farms") or [])
    market = _clone_market(o.get("market") or {})
    town = {"unlocked_shops": list((o.get("town") or {}).get("unlocked_shops") or [])}
    cur_step = int(o.get("step", 0)) if step is None else step
    day = int(o.get("day", 0))
    hour = int(o.get("hour", 0))

    mine = _clone_private(o.get("private") or {})
    theirs = _clone_private(opponent_private) if opponent_private else _blank_private()
    privates = [mine, theirs] if player == 0 else [theirs, mine]

    state = []
    for i in range(len(farms)):
        state.append(Node(
            action={"farmer": ["PASS"], "hands": [], "market": []},
            status="ACTIVE",
            reward=0.0,
            info={},
            observation=Node(
                player=i,
                step=cur_step,
                day=day,
                hour=hour,
                farms=farms,
                private=privates[i],
                market=market,
                town=town,
                remainingOverageTime=60,
            ),
        ))
    # Seats share the same objects for shared state, exactly as the real env does.
    for s in state[1:]:
        s.observation.farms = state[0].observation.farms
        s.observation.market = state[0].observation.market
        s.observation.town = state[0].observation.town
    return state


def _clone_tiles(tiles: Iterable[Any]) -> list[list[Any]]:
    """Tiles are the only deeply nested mutable structure; everything else is flat."""
    return [[dict(t) if isinstance(t, dict) else t for t in row] for row in tiles]


def _clone_farms(farms: Iterable[Any]) -> list[dict[str, Any]]:
    out = []
    for f in farms:
        f = dict(f)
        out.append({
            "money": float(f.get("money", 0.0)),
            "tiles": _clone_tiles(f.get("tiles") or []),
            "farmer": list(f.get("farmer") or [0, 0]),
            "hands": [list(h) for h in (f.get("hands") or [])],
            "unlocked_quadrants": list(f.get("unlocked_quadrants") or ["NW"]),
            "hires_today": int(f.get("hires_today", 0)),
        })
    return out


def _clone_private(private: Any) -> dict[str, Any]:
    if private is None:
        return _blank_private()
    shed = private["shed"] if isinstance(private, dict) and "shed" in private else None
    if shed is None:
        p = dict(private)
        base = _blank_private()
        base["shed"].update({k: int(v) for k, v in (p.get("shed") or {}).items()})
        base["seeds"].update({k: int(v) for k, v in (p.get("seeds") or {}).items()})
        base["inventories"] = [dict(i) for i in (p.get("inventories") or [{}])]
        return base
    return {
        "shed": dict(shed),
        "seeds": dict(private["seeds"]),
        "inventories": [dict(i) for i in private["inventories"]],
    }


def _clone_market(market: Any) -> dict[str, Any]:
    m = dict(market or {})
    out: dict[str, Any] = {
        "inventory": dict(m.get("inventory") or {}),
        "prices": dict(m.get("prices") or {}),
    }
    if "params" in m:
        out["params"] = m["params"]  # read-only, safe to share
    return out


def clone_state(state: list[Any]) -> list[Any]:
    """Fast structural copy of a two-seat state.

    Roughly 20x cheaper than `copy.deepcopy` because it copies only what
    `interpreter` mutates and shares the rest. This is the single hottest
    function in the search, so it is written flat rather than generically.
    """
    obs0 = state[0].observation
    farms = _clone_farms(obs0.farms)
    market = _clone_market(obs0.market)
    town = {"unlocked_shops": list(obs0.town["unlocked_shops"])}

    out = []
    for i, s in enumerate(state):
        out.append(Node(
            action={"farmer": ["PASS"], "hands": [], "market": []},
            status=s.status,
            reward=s.reward,
            info={},
            observation=Node(
                player=i,
                step=s.observation.step,
                day=obs0.day,
                hour=obs0.hour,
                farms=farms,
                private=_clone_private(s.observation.private),
                market=market,
                town=town,
                remainingOverageTime=60,
            ),
        ))
    for s in out[1:]:
        s.observation.farms = out[0].observation.farms
        s.observation.market = out[0].observation.market
        s.observation.town = out[0].observation.town
    return out


# ==========================================================================
# Stepping
# ==========================================================================
def simulate_step(
    state: list[Any],
    my_action: Action,
    opp_action: Action,
    env: EnvShim,
    seat: int = 0,
    in_place: bool = True,
) -> list[Any]:
    """Advance one turn using the environment's own interpreter.

    `in_place=True` mutates and returns `state`, which is what the search wants
    once it has already cloned. Pass `False` for a functional call.
    """
    kg = _load_env_module()
    if not kg:
        raise RuntimeError("kaggriculture environment unavailable; cannot simulate")

    if not in_place:
        state = clone_state(state)

    state[seat].action = my_action
    state[1 - seat].action = opp_action

    obs0 = state[0].observation
    step = obs0.step
    kg.interpreter(state, env)

    # `interpreter` advances day/hour but leaves `step` to the framework.
    next_step = step + 1
    for s in state:
        s.observation.step = next_step
    return state


def bank_of(state: list[Any], seat: int = 0) -> float:
    return float(state[0].observation.farms[seat]["money"])


def view_of(state: list[Any], seat: int = 0) -> View:
    """Agent-facing view of a simulated state, so policies are reusable."""
    return View.from_obs(state[seat].observation.__dict__)


# ==========================================================================
# Rollout
# ==========================================================================
def rollout(
    state: list[Any],
    env: EnvShim,
    policy: Callable[[View], Action],
    opponent_policy: Callable[[View], Action],
    depth: int,
    seat: int = 0,
    evaluate: Callable[[View], float] | None = None,
) -> float:
    """Play `depth` turns forward and score the result.

    `evaluate` defaults to shaped net worth rather than raw cash: over a horizon
    of a few dozen turns, cash alone would rate every investment as a pure loss.
    """
    steps_left = int(env.configuration["episodeSteps"]) - state[0].observation.step - 1
    for _ in range(max(0, min(depth, steps_left))):
        try:
            mine = policy(view_of(state, seat))
            theirs = opponent_policy(view_of(state, 1 - seat))
        except Exception:
            break
        simulate_step(state, mine, theirs, env, seat=seat, in_place=True)
    v = view_of(state, seat)
    return evaluate(v) if evaluate else v.net_worth()
