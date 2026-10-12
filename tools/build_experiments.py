"""Build isolated standard-library overlays without touching accepted artifacts."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODES = ("none", "shadow_terminal", "dead_buys", "safe_repair", "demand_order")


def render(mode: str) -> bytes:
    if mode not in MODES:
        raise ValueError(mode)
    parent = (ROOT / "submission/controls/20261002/tape_t0.py").read_bytes()
    overlay = (ROOT / "src/agents/route_overlay.py").read_text(encoding="utf-8")
    footer = f'''
# Submitter experimental modifications, Apache-2.0 section 4(b).
_OVERLAY_PARENT = agent
del agent
_OVERLAY = RouteOverlay(_OVERLAY_PARENT, globals(), mode={mode!r})
_OVERLAY_DIAGNOSTICS = _OVERLAY.stats


def agent(obs, config=None):
    return _OVERLAY(obs, config)
'''
    return parent + b"\n" + overlay.encode("utf-8") + footer.encode("utf-8")


def main() -> int:
    out = ROOT / "submission/candidates"
    out.mkdir(parents=True, exist_ok=True)
    for mode in MODES:
        path = out / (mode + ".py")
        path.write_bytes(render(mode))
        print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
