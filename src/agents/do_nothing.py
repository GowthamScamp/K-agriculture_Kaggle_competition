"""Phase 1 deliverable: an agent that does nothing, correctly.

Its only job is to prove the plumbing - that a full 720-turn episode runs end to
end without crashing and reports a final bank. It should always finish on
exactly the starting money.
"""

from __future__ import annotations

from typing import Any

from ..env_adapter import Action, safe_action


def do_nothing_agent(obs: Any, config: Any = None) -> Action:
    """Pass every turn. Never touches the market, never moves."""
    farms = obs.get("farms") or []
    player = int(obs.get("player", 0))
    n_hands = len(farms[player].get("hands", [])) if player < len(farms) else 0
    return safe_action(n_hands)
