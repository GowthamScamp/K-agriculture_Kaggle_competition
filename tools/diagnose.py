"""Run one episode with logging and summarise what the agent actually did.

Answers the questions that decide what to tune next: is the land being used, are
hands being hired, what is the tile mix, where did the money come from, and how
much was thrown away as end-of-day shed overflow.

    python -m tools.diagnose --agent baseline --seed 1
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.env_adapter import View
from src.game_constants import CROPS, MARKET_PARAMS, PRODUCTS
from src.registry import build_agent
from src.runner import make_env, _quiet

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="baseline")
    ap.add_argument("--opponent", default="starter")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--log", default=str(ROOT / "logs" / "diagnose.jsonl"))
    args = ap.parse_args()

    kwargs = {"log_path": args.log} if args.agent in ("baseline", "baseline_noisy") else {}
    agent = build_agent(args.agent, **kwargs)
    opponent = build_agent(args.opponent)

    env = make_env(seed=args.seed)

    snapshots: list[dict] = []

    def probe(obs: Any, config: Any = None) -> dict[str, Any]:
        act = agent(obs, config) if callable(agent) else {}
        v = View.from_obs(obs)
        if v.hour == 0:
            tile_mix = Counter()
            for t in v.owned_tiles():
                if t.is_plant:
                    tile_mix[t.crop] += 1
                elif t.has_animal:
                    tile_mix[t.animal] += 1
                elif t.is_weed:
                    tile_mix["WEED"] += 1
                elif t.empty:
                    tile_mix["EMPTY"] += 1
                elif t.is_structure:
                    tile_mix[t.kind] += 1
            snapshots.append({
                "day": v.day,
                "money": round(v.money),
                "net_worth": round(v.net_worth()),
                "hands": v.n_hands,
                "owned": sum(1 for _ in v.owned_tiles()),
                "quads": len(v.unlocked_quadrants),
                "shed": v.shed_used,
                "mix": dict(tile_mix),
                "prices": {k: v.price(k) for k in ("WHEAT", "CARROT", "MELON", "EGG", "STRAWBERRY")},
                "mkt_delta": {
                    k: int(v.market_inventory.get(k, 10000) - 10000)
                    for k in ("WHEAT", "CARROT", "MELON", "EGG", "STRAWBERRY")
                },
                "shops": len(v.unlocked_shops),
            })
        return act

    with _quiet():
        env.run([probe, opponent])
    if hasattr(agent, "close"):
        agent.close()

    final = env.steps[-1]
    print(f"\nseed {args.seed}:  {args.agent} = {final[0]['reward']:,.0f}   "
          f"{args.opponent} = {final[1]['reward']:,.0f}\n")

    hdr = (f"{'day':>4} {'money':>9} {'worth':>9} {'hnd':>4} {'own':>4} {'q':>2} "
           f"{'shed':>5} {'shops':>6}  tile mix")
    print(hdr)
    print("-" * len(hdr) + "-" * 30)
    for s in snapshots:
        mix = " ".join(f"{k[:4]}:{v}" for k, v in sorted(s["mix"].items(),
                                                         key=lambda kv: -kv[1]))
        print(f"{s['day']:>4} {s['money']:>9,} {s['net_worth']:>9,} {s['hands']:>4} "
              f"{s['owned']:>4} {s['quads']:>2} {s['shed']:>5} {s['shops']:>6}  {mix}")

    print("\nmarket inventory delta from I0 (positive = we glutted it):")
    print(f"{'day':>4} " + " ".join(f"{k[:5]:>7}" for k in
                                    ("WHEAT", "CARROT", "MELON", "EGG", "STRAWBERRY")))
    for s in snapshots[::5]:
        print(f"{s['day']:>4} " + " ".join(
            f"{s['mkt_delta'][k]:>7}" for k in
            ("WHEAT", "CARROT", "MELON", "EGG", "STRAWBERRY")))

    print("\nprices:")
    print(f"{'day':>4} " + " ".join(f"{k[:5]:>7}" for k in
                                    ("WHEAT", "CARROT", "MELON", "EGG", "STRAWBERRY")))
    for s in snapshots[::5]:
        print(f"{s['day']:>4} " + " ".join(
            f"{s['prices'][k]:>7}" for k in
            ("WHEAT", "CARROT", "MELON", "EGG", "STRAWBERRY")))

    # -- action census from the decision log --------------------------------
    log = Path(args.log)
    if log.exists():
        kinds = Counter()
        ops = Counter()
        market_ops = Counter()
        idle = 0
        total_unit_slots = 0
        for line in log.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "tasks" not in rec:
                continue
            n_units = rec.get("hands", 0) + 1
            total_unit_slots += n_units
            for t in rec["tasks"]:
                kinds[t["kind"]] += 1
                ops[t["op"][0]] += 1
            idle += n_units - len(rec["tasks"])
            for m in rec.get("market", []):
                market_ops[m[0]] += 1
        print(f"\nunit-turn census over the episode ({total_unit_slots:,} unit-turns):")
        moves = sum(ops[d] for d in ("NORTH", "SOUTH", "EAST", "WEST"))
        print(f"  movement       {moves:>7,}  ({moves / max(1, total_unit_slots):.1%})")
        for op, n in ops.most_common():
            if op in ("NORTH", "SOUTH", "EAST", "WEST"):
                continue
            print(f"  {op:<14} {n:>7,}  ({n / max(1, total_unit_slots):.1%})")
        print(f"  (unassigned)   {idle:>7,}  ({idle / max(1, total_unit_slots):.1%})")
        print("\nmarket orders issued:")
        for op, n in market_ops.most_common():
            print(f"  {op:<14} {n:>7,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
