# SPDX-License-Identifier: Apache-2.0
"""Small independently testable overlays for the pinned public route chassis.

Disabled by default. They use only observations and the chassis's existing
exact unit primitive. No hidden opponent inventory, episode seed, deadline
search, third-party dependency, or external replay is consulted.
"""

import copy as _overlay_copy


class RouteOverlay:
    def __init__(self, parent, namespace, mode="none"):
        if mode not in ("none", "shadow_terminal", "dead_buys", "safe_repair", "demand_order"):
            raise ValueError("Unknown overlay: " + mode)
        self.parent = parent
        self.namespace = namespace
        self.mode = mode
        self.states = {}
        self.stats = {"errors": 0, "changed_turns": 0, "repair_started": 0,
                      "repair_completed": 0, "repair_cancelled": 0}

    @staticmethod
    def step(obs):
        raw = obs.get("step")
        return int(raw) if raw is not None else int(obs.get("day") or 0) * 24 + int(obs.get("hour") or 0)

    def state(self, seat, step):
        state = self.states.get(seat)
        if state is None or step < state["step"]:
            state = self.states[seat] = {"step": -1, "pending": {}}
            self.stats.update(errors=0, changed_turns=0, repair_started=0,
                              repair_completed=0, repair_cancelled=0)
        state["step"] = step
        return state

    def primitive(self):
        for name in ("_PLANNER_NS", "_TP_NS"):
            namespace = self.namespace.get(name, {})
            if callable(namespace.get("_apply_unit_action")):
                return namespace["_apply_unit_action"]
        raise RuntimeError("Chassis lacks the pinned exact unit primitive")

    def preview(self, obs, action):
        farm = _overlay_copy.deepcopy(obs["farms"][int(obs["player"])])
        private = _overlay_copy.deepcopy(obs["private"])
        commands = [action.get("farmer") or ["PASS"], *(action.get("hands") or [])]
        demand = {}
        for command in commands:
            if len(command) > 1 and command[0] == "PLANT":
                demand[command[1]] = demand.get(command[1], 0) + 1
        blocked = {item for item, n in demand.items() if n > private["seeds"].get(item, 0)}
        apply = self.primitive()
        for actor, command in enumerate(commands):
            op = ["PASS"] if len(command) > 1 and command[0] == "PLANT" and command[1] in blocked else command
            apply(farm, private, actor, op, 10, self.step(obs) // 24, 24, 100)
        return farm, private

    def planned(self, seat, step, actor):
        native = self.namespace["_IMPL"].chassis.players[seat]
        route = 2 if step >= 648 else native["route"]
        action = self.namespace["_IMPL"].chassis.routes[route][step]
        if actor == 0:
            return action.get("farmer") or ["PASS"]
        hands = action.get("hands") or []
        return hands[actor - 1] if actor <= len(hands) else ["PASS"]

    def repair(self, obs, action, state):
        step, seat = self.step(obs), int(obs["player"])
        farm = obs["farms"][seat]
        positions = [farm["farmer"], *(farm.get("hands") or [])]
        commands = [action.get("farmer") or ["PASS"], *(action.get("hands") or [])]
        for actor, command in enumerate(commands):
            if actor >= len(positions):
                continue
            position = tuple(positions[actor])
            pending = state["pending"].get(actor)
            if pending:
                if step != pending["step"] or position != pending["position"] or command != ["PASS"]:
                    self.stats["repair_cancelled"] += 1
                    del state["pending"][actor]
                    continue
                tile = farm["tiles"][position[1]][position[0]]
                if pending["phase"] == "plant":
                    if tile is not None or obs["private"]["seeds"].get(pending["crop"], 0) < 1:
                        del state["pending"][actor]
                        self.stats["repair_cancelled"] += 1
                        continue
                    commands[actor] = ["PLANT", pending["crop"]]
                    pending.update(step=step + 1, phase="water")
                else:
                    if isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") == pending["crop"]:
                        commands[actor] = ["WATER"]
                        self.stats["repair_completed"] += 1
                    else:
                        self.stats["repair_cancelled"] += 1
                    del state["pending"][actor]
                continue
            if command[0] != "PLANT" or len(command) < 2 or step % 24 > 20 or step >= 648:
                continue
            tile = farm["tiles"][position[1]][position[0]]
            if not (isinstance(tile, dict) and tile.get("kind") == "WEED"):
                continue
            if not all(self.planned(seat, step + offset, actor) == ["PASS"] for offset in (1, 2)):
                continue
            if obs["private"]["seeds"].get(command[1], 0) < 1:
                continue
            commands[actor] = ["DIG"]
            state["pending"][actor] = {"position": position, "step": step + 1,
                                      "crop": command[1], "phase": "plant"}
            self.stats["repair_started"] += 1
        return dict(action, farmer=commands[0], hands=commands[1:])

    def transform(self, obs, action, state):
        step = self.step(obs)
        if self.mode == "shadow_terminal" and step == 718:
            _, private = self.preview(obs, action)
            prices = obs["market"]["prices"]
            sells = [["SELL", item, n] for item, n in private["shed"].items()
                     if n > 0 and item in prices]
            sells.sort(key=lambda order: (-prices[order[1]], order[1]))
            return dict(action, market=sells[:10])
        if self.mode == "dead_buys":
            first = {"WHEAT": 2, "CARROT": 2, "TOMATO": 8, "STRAWBERRY": 10, "MELON": 10,
                     "COW": 8, "SHEEP": 6, "GOOSE": 4}
            earliest = (step + 1) // 24
            orders = [order for order in action.get("market", [])
                      if not (len(order) >= 3 and order[0] in ("BUY_SEED", "BUY_ANIMAL")
                              and earliest + first.get(order[1], 0) > 29)]
            return dict(action, market=orders)
        if self.mode == "safe_repair":
            return self.repair(obs, action, state)
        if self.mode == "demand_order":
            orders = list(action.get("market") or [])
            end = 0
            while end < len(orders) and orders[end] and orders[end][0] == "SELL":
                end += 1
            prices = obs["market"]["prices"]
            # Reorder only an existing sale prefix. Never cross a feed purchase
            # or add unsupported stock; the route's quantities/timing stay fixed.
            front = sorted(orders[:end], key=lambda o: -prices.get(o[1], 0))
            return dict(action, market=front + orders[end:])
        return action

    def __call__(self, obs, config=None):
        step = self.step(obs)
        if obs.get("step") is None:
            obs = dict(obs, step=step)
        state = self.state(int(obs["player"]), step)
        original = self.parent(obs, config)
        if config and any(config.get(key, value) != value for key, value in
                          (("boardSize", 10), ("turnsPerDay", 24), ("shedCapacity", 100), ("episodeSteps", 720))):
            return original
        try:
            changed = self.transform(obs, original, state)
            self.stats["changed_turns"] += int(changed != original)
            return changed
        except Exception:
            self.stats["errors"] += 1
            return original
