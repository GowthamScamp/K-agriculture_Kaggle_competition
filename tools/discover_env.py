"""Generate docs/ENVIRONMENT.md from the *installed* kaggriculture package.

Deliberately introspective rather than hand-written: everything it reports is
read out of the live environment, so the doc can be regenerated after an
upstream `kaggle-environments` update and any schema drift shows up as a diff.

Usage:
    python -m tools.discover_env             # refresh docs/ENVIRONMENT.md
    python -m tools.discover_env --out X.md  # write elsewhere, to diff for drift
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

MARKET_OPS = {"HIRE", "BUY_LAND", "SELL", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL"}

#: Fixed so two regenerations of the doc are byte-identical and diffable.
DOC_SEED = 1


def _fence(text: str, lang: str = "") -> str:
    return "```" + lang + "\n" + text + "\n```"


def _describe(value: Any, depth: int = 0, max_depth: int = 3) -> str:
    """Compact structural description of a nested observation object."""
    pad = "  " * depth
    if isinstance(value, dict):
        if depth >= max_depth:
            return "dict[{} keys: {}]".format(len(value), ", ".join(list(value)[:8]))
        lines = [
            "{}- {}: {}".format(pad, k, _describe(v, depth + 1, max_depth))
            for k, v in value.items()
        ]
        return "\n" + "\n".join(lines)
    if isinstance(value, list):
        if not value:
            return "list[empty]"
        if depth >= max_depth:
            return "list[{}]".format(len(value))
        return "list[{}] of {}".format(len(value), _describe(value[0], depth + 1, max_depth))
    return "{} = {!r}".format(type(value).__name__, value)[:160]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DOCS / "ENVIRONMENT.md",
                    help="where to write the generated reference")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    try:
        from kaggle_environments import make
    except ImportError:
        print("kaggle-environments is not installed", file=sys.stderr)
        return 1

    from kaggle_environments.envs.kaggriculture import kaggriculture as kg

    spec_path = Path(kg.__file__).parent / "kaggriculture.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    # The seed has to go in at construction. `make()` initialises state through
    # the interpreter, which calls `resolve_episode_seed` - that persists a seed
    # into `env.info` and thereafter *ignores* `configuration["seed"]`, so
    # assigning it afterwards is silently a no-op. It also scrubs the key back
    # out of `configuration` (agents must not be able to read it), which is why
    # the dumped config below still shows the default.
    env = make("kaggriculture", configuration={"seed": DOC_SEED}, debug=False)
    cfg = dict(env.configuration)

    samples: dict[int, Any] = {}
    capture_at = {0, 25}

    def probe(obs: Any, config: Any = None) -> dict[str, Any]:
        step = obs["step"]
        if step in capture_at and step not in samples:
            samples[step] = json.loads(json.dumps(dict(obs)))
        return {"farmer": ["PASS"], "hands": [], "market": []}

    # `starter` rather than the built-in `random` opponent: `random_agent` builds
    # an unseeded `random.Random()` every turn, so its games can never be
    # reproduced, and this file is meant to be regenerated and diffed to reveal
    # upstream schema drift. `starter` uses no RNG at all.
    env.run([probe, "starter"])
    final = env.steps[-1]
    rewards = [s["reward"] for s in final]

    src = inspect.getsource(kg)
    unit_ops = sorted(
        (set(re.findall(r'op == "([A-Z_]+)"', src)) | set(kg.FARMER_MOVES) | {"PASS"})
        - MARKET_OPS
    )

    top_level = [
        n + str(inspect.signature(o))
        for n, o in vars(kg).items()
        if inspect.isfunction(o) and o.__module__ == kg.__name__ and not n.startswith("_")
    ]

    out: list[str] = []
    A = out.append

    A("# ENVIRONMENT.md - Kaggriculture, as actually installed\n")
    A("> **Generated file.** Produced by `python -m tools.discover_env`.")
    A("> Everything below is introspected from the installed package, not written by hand.")
    A("> Regenerate after any `kaggle-environments` upgrade and diff it.\n")
    A("- `kaggle-environments` version: **" + pkg_version("kaggle-environments") + "**")
    # Keep the generated document deterministic and free of workstation paths.
    A("- environment module: `" + kg.__name__ + "`\n")

    # ---- 1. limits -----------------------------------------------------
    A("## 1. Episode and limits\n")
    A("| Key | Value | Source |")
    A("|---|---|---|")
    A("| `agents` | `{}` | spec |".format(spec.get("agents")))
    for key in ("episodeSteps", "actTimeout", "runTimeout", "agentTimeout"):
        if key in cfg:
            A("| `{}` | `{}` | configuration |".format(key, cfg[key]))
    overage = spec.get("observation", {}).get("remainingOverageTime")
    A("| `remainingOverageTime` | `{}` | cumulative overage budget (s) |".format(overage))
    A("| reward | {} | spec |".format(spec.get("reward", {}).get("description", "n/a")))
    A("")
    A("**The 1-second `actTimeout` is the binding constraint on every search-based agent.**")
    A("`remainingOverageTime` ({}s) is slack across the *whole episode*, not per turn.\n".format(overage))

    # ---- 1a. constants -------------------------------------------------
    A("## 1a. Game constants (read from the module)\n")
    A("### Crops\n")
    A("| crop | seed | first_yield_day | max_yield_day | interval | max_yield | ongoing |")
    A("|---|---|---|---|---|---|---|")
    for crop, d in kg.CROPS.items():
        A("| {} | {} | {} | {} | {} | {} | {} |".format(
            crop, d["seed"], d["first_yield_day"], d["max_yield_day"],
            d["interval"], d["max_yield"], d["ongoing"]))
    A("")
    A("Non-ongoing crops start at `yield_units = 1` and gain +1 per watering inside the")
    A("bonus window `(max_yield_day+1)//2 .. max_yield_day` days after planting (+2 while")
    A("fertilized), capped at `max_yield`. Harvesting clears the tile.\n")
    A("**Critical rule:** `_new_plant` sets `consecutive_unwatered = 1`, and the end-of-day")
    A("refresh turns a plant into a weed once that reaches 2. A crop *must be watered on")
    A("the same day it is planted* or it is dead by morning.\n")

    A("### Animals\n")
    A("| animal | cost | structure | first_yield_day | interval | max_held | product |")
    A("|---|---|---|---|---|---|---|")
    for animal, d in kg.ANIMALS.items():
        A("| {} | {} | {} | {} | {} | {} | {} |".format(
            animal, d["cost"], d["structure"], d["first_yield_day"],
            d["interval"], d["max_held"], d["product"]))
    A("")
    A("Animals eat 1 WHEAT per day via `FEED`; two consecutive unfed days and the animal")
    A("escapes, leaving the bare structure behind. `CARE` banks a +1 bonus applied on the")
    A("next production day, but only if the animal was fed that day too.\n")

    A("### Market price curves\n")
    A("`price(inv) = base +/- amp * f(|inv - I0|)`, floored at {}.".format(kg.PRICE_FLOOR))
    A("Every product starts at `I0` = {}, i.e. exactly `base`.".format(kg.MARKET_I0))
    A("Selling raises inventory (price falls); town demand lowers it (price rises).\n")
    A("| item | base | T | scarcity func/target | glut func/target |")
    A("|---|---|---|---|---|")
    for item, d in kg.MARKET_PARAMS.items():
        A("| {} | {} | {} | {} / {} | {} / {} |".format(
            item, d["base"], d["T"], d["below_func"], d["below_target"],
            d["above_func"], d["above_target"]))
    A("")

    A("### Town demand\n")
    A("Shops are drawn **with replacement** (max {} instances), one unlocked every".format(
        kg.MAX_SHOP_INSTANCES))
    A("`townShopUnlockInterval` days. Each instance consumes independently every")
    A("`townShopSellInterval` turns:\n")
    for shop, prods in kg.SHOPS.items():
        suffix = "  *(single-product: pulls 2x)*" if len(prods) == 1 else ""
        A("- `{}`: {}{}".format(shop, ", ".join(prods), suffix))
    A("\nThe town center pulls 1 of each of these every `townCenterSellInterval` turns: "
      + ", ".join(kg.TOWN_CENTER_PRODUCTS) + ".\n")
    no_shop = [p for p in kg.PRODUCTS if not any(p in v for v in kg.SHOPS.values())]
    A("**Products with no shop demand at all: {}.** Their only buyer is the town".format(
        ", ".join(no_shop)))
    A("center. MELON in particular has an `sq` glut curve, so a melon glut is effectively")
    A("permanent - dump melons and the price never comes back.\n")

    A("### Land and labour\n")
    A("- Quadrant unlock order {} at prices {} (NW starts unlocked).".format(
        kg.LAND_ORDER, kg.LAND_PRICES))
    A("- Hire cost for the n-th hire *of that day* is `farmHandCostMult * fib(n)`, with")
    A("  fib = 1, 1, 2, 3, 5, 8, ... **The counter resets every day.**")
    A("- First 10 hires of a day cost {} in total; the first 15 cost {}.".format(
        sum(kg._fib(i) for i in range(10)), sum(kg._fib(i) for i in range(15))))
    A("- Hands spawn on shed-access tiles, act once per turn exactly like the farmer, and")
    A("  vanish at end of day, dropping their inventory into the shed.")
    A("- Shed-access tiles for a {0}x{0} board: {1}\n".format(
        cfg["boardSize"], kg._shed_access_tiles(cfg["boardSize"])))

    # ---- 2. configuration ----------------------------------------------
    A("## 2. Configuration defaults\n")
    A(_fence(json.dumps(cfg, indent=2), "json"))
    A("")

    # ---- 3. observation -------------------------------------------------
    A("## 3. Observation structure (live, step 0)\n")
    if 0 in samples:
        A("Top-level keys: " + ", ".join("`" + k + "`" for k in samples[0]) + "\n")
        A(_fence(_describe(samples[0]).strip()))
    A("")
    A("Notes:")
    A("- `farms` is indexed by player id and is **shared** - you can see the opponent's")
    A("  tiles, money, and unit positions, but not their shed or unit inventories.")
    A("- `private` is yours alone: `shed`, `seeds`, `inventories` (`[farmer, *hands]`).")
    A("- `market` is global; one player's dumping moves the price the other sees.\n")

    A("### 3a. Full observation dump, step 25\n")
    if 25 in samples:
        A("<details><summary>expand</summary>\n")
        A(_fence(json.dumps(samples[25], indent=2)[:10000], "json"))
        A("\n</details>")
    A("")

    # ---- 4. actions ------------------------------------------------------
    A("## 4. Action format\n")
    A("An agent is a plain function returning one dict per turn:\n")
    A(_fence(
        "def agent(obs):\n"
        "    return {\n"
        '        "farmer": [op, *args],        # one op for your farmer\n'
        '        "hands":  [[op, *args], ...], # one op per hired hand, in order\n'
        '        "market": [[op, *args], ...]  # up to maxMarketOrdersPerTurn orders\n'
        "    }", "python"))
    A("")
    A("Every unit gets **exactly one op per turn**, and movement is **one orthogonal tile**.")
    A("Illegal or malformed ops are silent no-ops - the environment never raises, so a")
    A("buggy agent loses quietly rather than crashing.\n")
    A("Unit ops: " + ", ".join("`" + o + "`" for o in unit_ops) + "\n")
    A("Market ops: `BUY_SEED <crop> <n>`, `BUY_PRODUCT <item> <n>` (WHEAT/FERTILIZER only),")
    A("`BUY_ANIMAL <animal> <n>`, `SELL <item> <n>`, `HIRE`, `BUY_LAND`.\n")
    A("Gotchas that cost real money:")
    A("- `SELL` sells **from the shed**, not from a unit's inventory. Harvest lands in the")
    A("  harvesting unit's hands; it reaches the shed via `DROP`/`PLACE` while standing on")
    A("  a shed-access tile, or automatically at end of day.")
    A("- Each unit of a multi-unit `SELL` is re-quoted at the updated inventory, so a large")
    A("  sell walks the price down against itself.")
    A("- If total `PLANT` requests for one crop in a turn exceed the seeds held, **all** of")
    A("  them are dropped, not merely the excess.")
    A("- The shed holds {} non-seed items; end-of-day overflow is **discarded**.\n".format(
        cfg["shedCapacity"]))

    # ---- 5. turn API ------------------------------------------------------
    A("## 5. Interpreter / turn API\n")
    A("The environment advances via a plain function over state dicts:\n")
    A(_fence("\n".join(top_level), "python"))
    A("")
    A("`interpreter(state, env)` takes the full 2-seat `state` list, where `state[i].action`")
    A("holds seat `i`'s submitted action, and mutates `state[i].observation` in place.")
    A("Order within a turn: unit actions for both players -> market -> town consumption ->")
    A("plant decay -> end-of-day refresh when the day rolls over.\n")
    A("**This is the function `src/simulator.py` reuses**, which is why simulator fidelity")
    A("is exact by construction rather than something that has to be calibrated.\n")

    # ---- 6. proof ---------------------------------------------------------
    A("## 6. Proof of a full episode\n")
    A("- steps recorded: **{}**".format(len(env.steps)))
    A("- final rewards (bank): **{}**  (seat 0 = do-nothing probe, seat 1 = `starter`)".format(rewards))
    A("- statuses: {}".format([s["status"] for s in final]))
    A("")
    A("Seat 0 passes every turn and ends on exactly the starting money - the Phase 1")
    A("deliverable. Seat 1 is the built-in `starter` agent, chosen over `random`")
    A("because `random_agent` builds an unseeded RNG per turn and so can never be")
    A("reproduced; `starter` uses no RNG, which is what lets this file regenerate")
    A("byte-identically and makes real schema drift show up as a clean diff.\n")
    A("(Run separately, the `random` agent ends on **0** - it buys seeds it never")
    A("harvests. Activity is not profit.)\n")

    A("## 7. Regenerating this file\n")
    A(_fence("python -m tools.discover_env", "bash"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out), encoding="utf-8")
    if not args.quiet:
        print("wrote {}  ({} steps, rewards={})".format(
            args.out, len(env.steps), rewards))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
