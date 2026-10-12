# ENVIRONMENT.md - Kaggriculture, as actually installed

> **Generated file.** Produced by `python -m tools.discover_env`.
> Everything below is introspected from the installed package, not written by hand.
> Regenerate after any `kaggle-environments` upgrade and diff it.

- `kaggle-environments` version: **1.32.7**
- environment module: `kaggle_environments.envs.kaggriculture.kaggriculture`

## 1. Episode and limits

| Key | Value | Source |
|---|---|---|
| `agents` | `[2]` | spec |
| `episodeSteps` | `720` | configuration |
| `actTimeout` | `1` | configuration |
| `runTimeout` | `1200` | configuration |
| `remainingOverageTime` | `60` | cumulative overage budget (s) |
| reward | Player money at end of game (final score). | spec |

**The 1-second `actTimeout` is the binding constraint on every search-based agent.**
`remainingOverageTime` (60s) is slack across the *whole episode*, not per turn.

## 1a. Game constants (read from the module)

### Crops

| crop | seed | first_yield_day | max_yield_day | interval | max_yield | ongoing |
|---|---|---|---|---|---|---|
| WHEAT | 10 | 2 | 4 | 0 | 6 | False |
| CARROT | 20 | 2 | 3 | 0 | 4 | False |
| TOMATO | 50 | 8 | 8 | 1 | 4 | True |
| STRAWBERRY | 100 | 10 | 10 | 2 | 4 | True |
| MELON | 80 | 10 | 12 | 0 | 6 | False |

Non-ongoing crops start at `yield_units = 1` and gain +1 per watering inside the
bonus window `(max_yield_day+1)//2 .. max_yield_day` days after planting (+2 while
fertilized), capped at `max_yield`. Harvesting clears the tile.

**Critical rule:** `_new_plant` sets `consecutive_unwatered = 1`, and the end-of-day
refresh turns a plant into a weed once that reaches 2. A crop *must be watered on
the same day it is planted* or it is dead by morning.

### Animals

| animal | cost | structure | first_yield_day | interval | max_held | product |
|---|---|---|---|---|---|---|
| GOOSE | 300 | COOP | 4 | 1 | 4 | EGG |
| COW | 400 | PASTURE | 8 | 2 | 6 | MILK |
| SHEEP | 500 | PASTURE | 6 | 3 | 6 | WOOL |

Animals eat 1 WHEAT per day via `FEED`; two consecutive unfed days and the animal
escapes, leaving the bare structure behind. `CARE` banks a +1 bonus applied on the
next production day, but only if the animal was fed that day too.

### Market price curves

`price(inv) = base +/- amp * f(|inv - I0|)`, floored at 1.
Every product starts at `I0` = 10000, i.e. exactly `base`.
Selling raises inventory (price falls); town demand lowers it (price rises).

| item | base | T | scarcity func/target | glut func/target |
|---|---|---|---|---|
| WHEAT | 25 | 400 | sqrt / 0.8 | log / 0.2 |
| CARROT | 35 | 450 | hinge / 1.0 | sqrt / 0.7 |
| TOMATO | 60 | 200 | hinge / 0.4 | sqrt / 0.6 |
| STRAWBERRY | 120 | 100 | sqrt / 0.7 | linear / 1.6 |
| MELON | 250 | 300 | log / 0.2 | sq / 3.6 |
| EGG | 50 | 332 | hinge / 0.4 | log / 0.2 |
| MILK | 160 | 122 | sqrt / 0.6 | linear / 1.6 |
| WOOL | 200 | 105 | log / 0.2 | sq / 3.2 |
| FERTILIZER | 100 | 200 | linear / 0.4 | linear / 0.4 |

### Town demand

Shops are drawn **with replacement** (max 8 instances), one unlocked every
`townShopUnlockInterval` days. Each instance consumes independently every
`townShopSellInterval` turns:

- `BAKERY`: EGG, WHEAT
- `PIZZA_SHOP`: MILK, TOMATO, WHEAT
- `BRUNCH_SPOT`: EGG, WHEAT, STRAWBERRY
- `YARN_STORE`: WOOL  *(single-product: pulls 2x)*
- `ICE_CREAM_SHOP`: STRAWBERRY, MILK, WHEAT
- `PET_CAFE`: CARROT  *(single-product: pulls 2x)*
- `SMOOTHIE_SHOP`: STRAWBERRY, MILK
- `FARMERS_MARKET`: WHEAT, CARROT, TOMATO, STRAWBERRY

The town center pulls 1 of each of these every `townCenterSellInterval` turns: WHEAT, CARROT, TOMATO, STRAWBERRY, MELON, EGG, MILK, WOOL.

**Products with no shop demand at all: MELON, FERTILIZER.** Their only buyer is the town
center. MELON in particular has an `sq` glut curve, so a melon glut is effectively
permanent - dump melons and the price never comes back.

### Land and labour

- Quadrant unlock order ['NE', 'SW', 'SE'] at prices [1000, 2000, 4000] (NW starts unlocked).
- Hire cost for the n-th hire *of that day* is `farmHandCostMult * fib(n)`, with
  fib = 1, 1, 2, 3, 5, 8, ... **The counter resets every day.**
- First 10 hires of a day cost 143 in total; the first 15 cost 1596.
- Hands spawn on shed-access tiles, act once per turn exactly like the farmer, and
  vanish at end of day, dropping their inventory into the shed.
- Shed-access tiles for a 10x10 board: [(4, 4), (5, 4), (4, 5), (5, 5)]

## 2. Configuration defaults

```json
{
  "seed": null,
  "episodeSteps": 720,
  "actTimeout": 1,
  "runTimeout": 1200,
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
  "marketParams": {}
}
```

## 3. Observation structure (live, step 0)

Top-level keys: `remainingOverageTime`, `step`, `player`, `farms`, `private`, `market`, `town`, `day`, `hour`

```
- remainingOverageTime: int = 60
- step: int = 0
- player: int = 0
- farms: list[2] of 
    - money: float = 3000.0
    - tiles: list[10]
    - farmer: list[2]
    - hands: list[empty]
    - unlocked_quadrants: list[1]
    - hires_today: int = 0
- private: 
  - shed: 
    - WHEAT: int = 0
    - CARROT: int = 0
    - TOMATO: int = 0
    - STRAWBERRY: int = 0
    - MELON: int = 0
    - EGG: int = 0
    - MILK: int = 0
    - WOOL: int = 0
    - FERTILIZER: int = 0
    - GOOSE: int = 0
    - COW: int = 0
    - SHEEP: int = 0
  - seeds: 
    - WHEAT: int = 0
    - CARROT: int = 0
    - TOMATO: int = 0
    - STRAWBERRY: int = 0
    - MELON: int = 0
  - inventories: list[1] of dict[0 keys: ]
- market: 
  - inventory: 
    - WHEAT: int = 10000
    - CARROT: int = 10000
    - TOMATO: int = 10000
    - STRAWBERRY: int = 10000
    - MELON: int = 10000
    - EGG: int = 10000
    - MILK: int = 10000
    - WOOL: int = 10000
    - FERTILIZER: int = 10000
  - prices: 
    - WHEAT: int = 25
    - CARROT: int = 35
    - TOMATO: int = 60
    - STRAWBERRY: int = 120
    - MELON: int = 250
    - EGG: int = 50
    - MILK: int = 160
    - WOOL: int = 200
    - FERTILIZER: int = 100
- town: 
  - unlocked_shops: list[empty]
- day: int = 0
- hour: int = 0
```

Notes:
- `farms` is indexed by player id and is **shared** - you can see the opponent's
  tiles, money, and unit positions, but not their shed or unit inventories.
- `private` is yours alone: `shed`, `seeds`, `inventories` (`[farmer, *hands]`).
- `market` is global; one player's dumping moves the price the other sees.

### 3a. Full observation dump, step 25

<details><summary>expand</summary>

```json
{
  "remainingOverageTime": 60,
  "step": 25,
  "player": 0,
  "farms": [
    {
      "money": 3000.0,
      "tiles": [
        [
          null,
          null,
          null,
          null,
          null,
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED"
        ],
        [
          null,
          null,
          null,
          null,
          null,
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED"
        ],
        [
          null,
          null,
          null,
          null,
          null,
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED"
        ],
        [
          null,
          null,
          null,
          null,
          null,
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED"
        ],
        [
          null,
          null,
          null,
          null,
          null,
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED"
        ],
        [
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED"
        ],
        [
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED"
        ],
        [
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED"
        ],
        [
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED"
        ],
        [
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED"
        ]
      ],
      "farmer": [
        4,
        4
      ],
      "hands": [],
      "unlocked_quadrants": [
        "NW"
      ],
      "hires_today": 0
    },
    {
      "money": 2960.0,
      "tiles": [
        [
          null,
          null,
          null,
          null,
          null,
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED"
        ],
        [
          null,
          null,
          null,
          null,
          null,
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED"
        ],
        [
          null,
          null,
          null,
          null,
          null,
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED"
        ],
        [
          null,
          null,
          null,
          null,
          null,
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED"
        ],
        [
          null,
          null,
          null,
          null,
          {
            "kind": "PLANT",
            "crop": "CARROT",
            "planted_day": 0,
            "watered_today": true,
            "consecutive_unwatered": 0,
            "yield_units": 1,
            "max_lifespan_step": 96,
            "fertilized_until_day": -1
          },
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED"
        ],
        [
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED"
        ],
        [
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED"
        ],
        [
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED"
        ],
        [
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED"
        ],
        [
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED",
          "LOCKED"
        ]
      ],
      "farmer": [
        4,
        4
      ],
      "hands": [],
      "unlocked_quadrants": [
        "NW"
      ],
      "hires_today": 0
    }
  ],
  "private": {
    "shed": {
      "WHEAT": 0,
      "CARROT": 0,
      "TOMATO": 0,
      "STRAWBERRY": 0,
      "MELON": 0,
      "EGG": 0,
      "MILK": 0,
      "WOOL": 0,
      "FERTILIZER": 0,
      "GOOSE": 0,
      "COW": 0,
      "SHEEP": 0
    },
    "seeds": {
      "WHEAT": 0,
      "CARROT": 0,
      "TOMATO": 0,
      "STRAWBERRY": 0,
      "MELON": 0
    },
    "inventories": [
      {}
    ]
  },
  "market": {
    "inventory": {
      "WHEAT": 9998,
      "CARROT": 9998,
      "TOMATO": 9998,
      "STRAWBERRY": 9998,
      "MELON": 9998,
      "EGG": 9998,
      "MILK": 9998,
      "WOOL": 9998,
      "FERTILIZER": 10000
    },
    "prices": {
      "WHEAT": 26,
      "CARROT": 35,
      "TOMATO": 60,
      "STRAWBERRY": 132,
      "MELON": 260,
      "EGG": 50,
      "MILK": 172,
      "WOOL": 209,
      "FERTILIZER": 100
    }
  },
  "town": {
    "unlocked_shops": []
  },
  "day": 1,
  "hour": 1
}
```

</details>

## 4. Action format

An agent is a plain function returning one dict per turn:

```python
def agent(obs):
    return {
        "farmer": [op, *args],        # one op for your farmer
        "hands":  [[op, *args], ...], # one op per hired hand, in order
        "market": [[op, *args], ...]  # up to maxMarketOrdersPerTurn orders
    }
```

Every unit gets **exactly one op per turn**, and movement is **one orthogonal tile**.
Illegal or malformed ops are silent no-ops - the environment never raises, so a
buggy agent loses quietly rather than crashing.

Unit ops: `BUILD_COOP`, `BUILD_PASTURE`, `CARE`, `COLLECT_FERTILIZER`, `DIG`, `DROP`, `EAST`, `FEED`, `FERTILIZE`, `HARVEST`, `NORTH`, `PASS`, `PICKUP`, `PLACE`, `PLANT`, `SOUTH`, `WATER`, `WEST`

Market ops: `BUY_SEED <crop> <n>`, `BUY_PRODUCT <item> <n>` (WHEAT/FERTILIZER only),
`BUY_ANIMAL <animal> <n>`, `SELL <item> <n>`, `HIRE`, `BUY_LAND`.

Gotchas that cost real money:
- `SELL` sells **from the shed**, not from a unit's inventory. Harvest lands in the
  harvesting unit's hands; it reaches the shed via `DROP`/`PLACE` while standing on
  a shed-access tile, or automatically at end of day.
- Each unit of a multi-unit `SELL` is re-quoted at the updated inventory, so a large
  sell walks the price down against itself.
- If total `PLANT` requests for one crop in a turn exceed the seeds held, **all** of
  them are dropped, not merely the excess.
- The shed holds 100 non-seed items; end-of-day overflow is **discarded**.

## 5. Interpreter / turn API

The environment advances via a plain function over state dicts:

```python
get(d, key, default)
market_price(item, inventory, params=None)
interpreter(state, env)
renderer(state, env)
html_renderer(env, mode)
pass_agent(obs)
random_agent(obs)
starter_agent(obs)
```

`interpreter(state, env)` takes the full 2-seat `state` list, where `state[i].action`
holds seat `i`'s submitted action, and mutates `state[i].observation` in place.
Order within a turn: unit actions for both players -> market -> town consumption ->
plant decay -> end-of-day refresh when the day rolls over.

**This is the function `src/simulator.py` reuses**, which is why simulator fidelity
is exact by construction rather than something that has to be calibrated.

## 6. Proof of a full episode

- steps recorded: **720**
- final rewards (bank): **[3000.0, 3506.0]**  (seat 0 = do-nothing probe, seat 1 = `starter`)
- statuses: ['DONE', 'DONE']

Seat 0 passes every turn and ends on exactly the starting money - the Phase 1
deliverable. Seat 1 is the built-in `starter` agent, chosen over `random`
because `random_agent` builds an unseeded RNG per turn and so can never be
reproduced; `starter` uses no RNG, which is what lets this file regenerate
byte-identically and makes real schema drift show up as a clean diff.

(Run separately, the `random` agent ends on **0** - it buys seeds it never
harvests. Activity is not profit.)

## 7. Regenerating this file

```bash
python -m tools.discover_env
```