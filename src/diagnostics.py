"""Development-only, exact ordered unit previews using the official engine.

This module is deliberately excluded from submissions. No replay is read and
no caller-owned observation is modified. Atomic seed rejection mirrors the
interpreter, before calling the engine's actual unit primitive in actor order.
"""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .simulator import _load_env_module


@dataclass(slots=True)
class UnitEffect:
    actor: int
    operation: str
    changed: bool
    discarded: int = 0
    deposited: int = 0
    blocked_seed: bool = False


def preview_units(obs: Any, action: dict, config: Any = None) -> tuple[dict, dict, list[UnitEffect]]:
    """Return exact post-unit state and actual effects, before market/night."""
    kg = _load_env_module()
    if not kg:
        raise RuntimeError("Exact diagnostics require the pinned Kaggriculture engine")
    seat = int(obs.get("player") or 0)
    farm = copy.deepcopy(obs["farms"][seat])
    private = copy.deepcopy(obs.get("private") or {})
    for key, default in (("shed", {}), ("seeds", {}), ("inventories", [])):
        private.setdefault(key, default)
    config = config or {}
    board = int(config.get("boardSize", len(farm["tiles"])))
    turns = int(config.get("turnsPerDay", 24))
    capacity = int(config.get("shedCapacity", 100))
    step = obs.get("step")
    step = int(step) if step is not None else int(obs.get("day") or 0) * turns + int(obs.get("hour") or 0)
    day = step // turns
    hands = action.get("hands") or []
    if not isinstance(hands, list):
        hands = []
    commands = [action.get("farmer", ["PASS"]), *hands]
    demand = Counter(a[1] for a in commands
                     if isinstance(a, list) and len(a) >= 2 and a[0] == "PLANT")
    blocked = {crop for crop, n in demand.items() if n > private["seeds"].get(crop, 0)}
    # Count implicit PASS for actual hands, as well as any surplus commands.
    while len(commands) < 1 + len(farm.get("hands") or []):
        commands.append(["PASS"])
    effects = []
    for actor, command in enumerate(commands):
        operation = command[0] if isinstance(command, list) and command else "PASS"
        denied = operation == "PLANT" and len(command) > 1 and command[1] in blocked
        before = copy.deepcopy((farm, private))
        load_before = sum(private["inventories"][actor].values()) if actor < len(private["inventories"]) else 0
        shed_before = sum(private["shed"].values())
        kg._apply_unit_action(farm, private, actor, ["PASS"] if denied else command,
                              board, day, turns, capacity)
        changed = operation != "PASS" and (farm, private) != before
        deposited = max(0, sum(private["shed"].values()) - shed_before)
        load_after = sum(private["inventories"][actor].values()) if actor < len(private["inventories"]) else 0
        discarded = max(0, load_before - load_after - deposited) if operation == "DROP" else 0
        effects.append(UnitEffect(actor, str(operation), changed, discarded, deposited, denied))
    return farm, private, effects


def error_counters(agent: Any) -> dict[str, int]:
    """Read public diagnostics without serializing route tapes or private caches."""
    namespace = getattr(agent, "__globals__", {})
    result: dict[str, int] = {}
    candidates = [(name, value) for name, value in namespace.items()
                  if isinstance(value, dict) and any(token in name.upper() for token in ("STATE", "REPORT", "DIAG"))]
    native = namespace.get("_IMPL")
    diagnostics = getattr(getattr(native, "chassis", None), "diagnostics", None)
    if isinstance(diagnostics, dict):
        candidates.append(("chassis", diagnostics))
    telemetry = getattr(agent, "telemetry", None)
    if isinstance(telemetry, dict):
        candidates.append(("telemetry", telemetry))
    fallback = (namespace.get("_KGW_STATE") or {}).get("fallback")
    fallback_stats = getattr(fallback, "__globals__", {}).get("_POLICY_DIAGNOSTICS")
    if isinstance(fallback_stats, dict):
        candidates.append(("fallback", fallback_stats))
    for name, mapping in candidates:
        for key, value in mapping.items():
            if isinstance(value, int) and value >= 0 and ("error" in str(key).lower() or "exception" in str(key).lower()):
                result[f"{name}.{key}"] = value
    return result
