# Derived from lonespear/kaggriculture main_v18.py.
#
# MIT License
#
# Copyright (c) 2026 Jonathan Day
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Kaggriculture agent — v11: livestock economy.

WHY THIS IS A REWRITE, NOT A TUNE
=================================
Reading the #1-vs-#2 replay (episode 90303687, 135k and 151k against our 57k)
and tracing market inventory exposed the actual structure of the game.

Market inventory starts at I0 = 10,000 per product. Town shops CONSUME product
every few turns, draining inventory; players selling ADD to it. Price rises as
inventory falls. So the question for any product is whether town demand outruns
player supply. Over a full leaders' game:

    WHEAT       inventory -860   price  25 -> 54
    STRAWBERRY  inventory  -35   price 120 -> 170  (peaked 248)
    EGG         inventory -318   price  50 -> 69
    MILK        inventory    ~0  price 160 -> 158  (peaked 217)
    WOOL        inventory    0   price 200 -> 200
    MELON       inventory +144   price 250 -> 43
    FERTILIZER  inventory +353   price 100 -> 29

Only melon and fertilizer collapse — and they are exactly the two products no
shop demands. Checking the shop table: wheat is wanted by 5 shops, strawberry
by 4, milk by 3, eggs by 2, wool by 1 at double rate. Melon appears in none of
them, so its only sink is the town centre; fertilizer has no sink at all.

Our v9 economy was 17 melon tiles plus goose fertilizer — i.e. built entirely
on the two dead products. Their prices crashed *because we produced them*.

The livestock economy instead:
  * COW   $400, milk every 2 days, base 160, 3 shops demand it
  * SHEEP $500, wool every 3 days, base 200, yarn store demands 2x
  * GOOSE $300, egg every day,     base  50, 2 shops demand it

CARE matters here in a way it never did for a melon farm. It banks +1 yield per
fed-and-cared day and pays out on the next scheduled production, so a cow fed
and cared daily yields 3 milk per 2 days rather than 1. That is ~$270/day on a
$400 animal. v9 measured CARE as a loss only because its actions competed with
melon watering; with livestock as the economy, CARE *is* the economy.
"""

CROP_INFO = {
    "WHEAT": {"seed": 10, "first": 2, "maxday": 4, "wcap": 4, "ongoing": False},
    "CARROT": {"seed": 20, "first": 2, "maxday": 3, "wcap": 3, "ongoing": False},
    "MELON": {"seed": 80, "first": 10, "maxday": 10, "wcap": 6, "ongoing": False},
    "TOMATO": {"seed": 50, "first": 8, "maxday": 11, "wcap": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first": 10, "maxday": 16, "wcap": 4, "ongoing": True},
}

ANIMAL_INFO = {
    "COW": {"cost": 400, "struct": "PASTURE", "build": "BUILD_PASTURE",
            "first": 8, "interval": 2, "max_held": 6, "product": "MILK"},
    "SHEEP": {"cost": 500, "struct": "PASTURE", "build": "BUILD_PASTURE",
              "first": 6, "interval": 3, "max_held": 6, "product": "WOOL"},
    "GOOSE": {"cost": 300, "struct": "COOP", "build": "BUILD_COOP",
              "first": 4, "interval": 1, "max_held": 4, "product": "EGG"},
}

BASE_PRICE = {
    "WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120,
    "MELON": 250, "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100,
}

# Products town shops demand, so their price holds up or climbs. Safe to sell
# in volume. Melon and fertilizer are deliberately absent.
TOWN_DEMANDED = {"WHEAT", "STRAWBERRY", "EGG", "MILK", "WOOL", "CARROT", "TOMATO"}
DRIP_LIMIT = {"MILK": 8, "WOOL": 8, "EGG": 10, "STRAWBERRY": 6,
              "MELON": 2, "FERTILIZER": 6, "TOMATO": 4, "CARROT": 4}

# --- Tunables -------------------------------------------------------------
COW_MAX = 8             # leaders run 8 cows + 6 sheep and stop. Buying past
SHEEP_MAX = 6           # that reinvests every coin into livestock instead of
GOOSE_MAX = 0           # Measured 0 = 86,878, 4 = 73,579, 8 = 59,386. Geese
                        # are a net drag here: each eats a wheat a day and
                        # costs FEED + CARE + HARVEST + fertilizer actions,
                        # while eggs fetch only $50-69. Leaders run none.
LAST_ANIMAL_DAY = 16    # stop buying in time to actually accumulate cash
MELON_MAX = 0           # OFF, and not marginally: enabling melon costs ~41k.
                        # The meta plants 11 melon seeds on day 0 for early
                        # capital, so this was worth a real try — but with melon
                        # actually reachable we score 52,720 against 93,553 with
                        # it off (margin +47,059, 20/20, significant).
                        # Our opening bankroll is already fully committed to
                        # livestock; $880 of melon seed starves the herd, and
                        # the tiles displace feed wheat for 10 days. Copying a
                        # stronger agent's opening without its cash flow just
                        # breaks your own.
MELON_LAST_PLANT_DAY = 4  # plant melon only in the opening window
MELON_SEED_MIN_CASH = 250  # cash floor for buying melon seed
STRAW_SEED_MIN_CASH = 1500  # Herd BEFORE crops. Spending opening cash on
                            # seed starves the livestock: 200 = -58,751 margin
                            # (0/20, herd collapses), 900 = -84, 1500 = +4,344
                            # (19/20), 2500 = +1,660, 6000 = +577.
STRAWBERRY_MAX = 24     # 4 shops demand strawberry so its price CLIMBS
                        # (120 -> 248) however much is produced. Scaling it up
                        # only pays once FERTILIZE actually fires: before that
                        # fix 10 beat 22, after it 16 = 100,780 beats
                        # 10 = 98,395. A tuned quota can hide a broken feature.
WHEAT_TILES = 10        # Home-grown feed plus surplus to sell; wheat is
                        # demanded by 5 shops and climbs 25 -> 54.
                        #
                        # Chosen on MARGIN, not absolute score. Against a
                        # 14-tile opponent over 16 mirror seeds, 10 scores
                        # LESS (70,417 vs 75,947) but wins 12/16 because it
                        # suppresses the opponent harder (68,384 vs 76,491).
                        # The competition ranks by match outcome, so margin is
                        # the objective and mean is a decoy.
MAX_QUADRANTS = 2       # Land stopped mattering once the farm became
                        # labour-bound: 1/2/3 quadrants land within ~2% of each
                        # other. A 2-D grid put 2 quadrants + 13 hands on top.
MAX_HANDS = 11          # Hire cost is fib-SUMMED per day and climbs steeply:
                        # 15 hands loses 18.7k of margin, 18 bankrupts the farm.
                        # 11 over 13 is a margin call — 13 scores more in
                        # absolute terms but 11 wins 12/16 head to head.
MIN_HANDS = 4
HAND_DIVISOR = 4  # hands = workload // this. Raised from 2 once the
                  # standing-on-work pass made the crew finish faster:
                  # 2 = -808 margin, 3 = +1,255, 4 = +3,093 (20/20), 5 = +3,895.
CASH_BUFFER = 500
ANIMAL_BUY_FEED_DAYS = 0  # Days of herd feed to hold before buying another
                          # animal, at the live wheat price. Raising this to
                          # protect against the starvation spiral in replay
                          # 90323915 MEASURED WORSE across 16 mirror seeds:
                          # 0 = 80,680, 2 = 79,954, 4 = 72,697, 6 = 70,472.
                          # An occasional starved opening costs less than
                          # permanently under-building the herd.
ENABLE_FERT = 1         # collect fertilizer (crashes to ~$29, still >0)

# Produce only becomes sellable once it is in the SHED, and anything still
# carried at the end-of-day drop above shedCapacity is DISCARDED. These
# thresholds are PER UNIT: with ~15 units each holding 6-10 items, a threshold
# of 14 never fired, the team carried 80-150 units of product all day, and
# **963 units a game were thrown away** at the nightly dump (~$190k). Keep
# them low enough that produce flows shed-ward continuously.
DROP_AT = 3             # drop when this much PRODUCE is carried and
                        # shed-adjacent. 3 = 97,876, 6 = 96,775, 9 = 91,993.
RETURN_AT = 8           # walk back to the shed at this many. Measured
                        # 8 = 71,951, 12 = 63,150, 16 = 56,240. Produce is
                        # worthless until it reaches the shed, so ferrying it
                        # promptly beats squeezing more work out of each trip.
WHEAT_CARRY = 3         # Feeds carried per trip. Carrying more cuts shed trips
                        # but the nightly drop dumps carried feed into the shed
                        # too, so big loads overflow it. 3 = 96,775,
                        # 4 = 88,667, 5 = 82,526, 6 = 76,342, 8 = 71,622.
FERT_CARRY = 4          # fertilizer carried per trip
ANIMAL_HARVEST_SLACK = 2  # harvest an animal at max_held minus this
STRAW_PER_DAY = 99      # OFF. Staggering the cohorts to avoid a synchronised
                        # death measured WORSE, monotonically: 6 = -255,
                        # 4 = -3,044, 3 = -5,491, 2 = -11,235 margin.
                        # Strawberry needs 17 days for its 4 productions and the
                        # season is 30, so there is room for ~1.5 cohorts.
                        # Delaying a planting does not move income later, it
                        # deletes productions. Cohort-staggering logic assumes a
                        # long horizon; this one is too short for it.
ENDGAME_FILL_FROM = 6   # days_left at which spare tiles take any crop that can
                        # still mature, ignoring the normal wheat quota
LAST_DAY = 29           # final scoring day (0-indexed, 30-day season)
ENDGAME_DUMP_DAY = 28   # from this day, ferry and sell everything — unsold
                        # produce scores zero
TERMINAL_STOP = 1       # stop FEED/CARE once an animal has no production left
                        # inside the horizon
PRIORITY_WEIGHT = 3     # tiles of travel one priority level is worth in the
                        # Hungarian cost matrix
CAND_MULT = 4           # candidate tasks per free unit in the matrix
CLUSTER_BONUS = 2.0     # Cheap route bundling: discount a task by how many
                        # other pending tasks sit nearby, so the matching
                        # prefers work it can chain. Raced over 40 seeds:
                        # 2.0 sole survivor at +1,478 margin, with 0.0 and 0.5
                        # eliminated (1.0 and 4.0 dropped in round 1).
CLUSTER_RADIUS = 2      # Manhattan radius counted as "neighbouring"
CLUSTER_CAP = 4         # ignore neighbours past this, so dense blocks do not
                        # dominate the cost entirely
TIER_CAP_SCALE = 8.0    # Effectively uncapped. The per-tier caps were a
                        # GREEDY-era workaround to stop one crowded priority
                        # monopolising the crew; optimal matching with a
                        # priority handicap already handles that, and the caps
                        # were throttling the solver (widening CAND_MULT did
                        # nothing because the caps bounded the candidate set).
                        # 1.0 = -190, 4 = +1,230, 8 = +2,271 (19/20); 8 and 20
                        # are identical, so 8 is already uncapped.
DIG_CAP = 0.10          # share of the workforce allowed to clear weeds. Weeds
                        # are not cosmetic: a weed tile cannot be planted until
                        # dug, so they silently shrink the farm. Ours reach 16
                        # by day 18 while units sit at 16-20% PASS.
FEED_BUFFER = 1.5       # days of feed held in the shed, per animal. Every slot
                        # held is a slot unavailable to produce, and a full
                        # shed also blocks BUY_PRODUCT.
# Carried inputs, excluded from the drop/return thresholds so that restocking
# feed does not immediately trigger a drop of the feed just collected.
INPUT_ITEMS = {"WHEAT", "FERTILIZER", "COW", "SHEEP", "GOOSE"}
FERTILIZE_CROPS = 1     # spend animal fertilizer on strawberry instead of
                        # selling it (see the FERTILIZE task for the maths)
SHED_CAP = 100
LAND_COSTS = [1000, 2000, 4000]
SELL_THRESHOLD = 0.45
# Both players' orders settle index by index against one shared price, so a
# sale placed after our hires and purchases trades behind the rival's earlier
# indices. Measured 27 Sep: 24-0 vs this policy with it off, 24-0 vs lonespear
# (16-8 with it off), seeds 0-11 both seats.
SELLS_FIRST = True
_POLICY_DIAGNOSTICS = {"errors": 0, "last_step": -1, "player": None}

# Task priorities — lower is more urgent.
P_RESCUE = 0     # dies tonight without action
P_FEED = 1       # daily feed keeps the CARE bank alive, which is the economy
P_HARVEST = 2
P_CARE = 3
P_WATER = 4
P_PLACE = 5
# Using fertilizer beats selling it by ~5.7x: a unit spent on strawberry adds
# ~1.5 units at ~$249 (~$374) against a ~$66 sale price. It therefore outranks
# collecting more of it.
P_FERTILIZE = 6
P_FERT = 7
P_BUILD = 8
P_PLANT = 9
P_DIG = 10

# The farm generates ~100 tasks/day against ~11 units x 24 turns, so it runs
# right at capacity and whatever sits at the bottom of the order never happens.
# Promoting P_PLANT above upkeep to fix that was tried and MEASURED WORSE
# (47,628 vs 73,579): strawberry sales doubled 20 -> 40 but milk fell 210 -> 140
# and wool 125 -> 62, and the animals are worth more than the crop.
#
# The fix is not reordering the queue but reducing what competes for it — wheat
# is the labour hog (plant + 5 waters + harvest per 4-day cycle) for the
# smallest return, so WHEAT_TILES is the real control.
TIER_CAP = {
    P_RESCUE: 1.00, P_FEED: 0.55, P_HARVEST: 0.40, P_CARE: 0.50,
    P_WATER: 0.45, P_PLACE: 0.25, P_FERTILIZE: 0.35, P_FERT: 0.30,
    P_BUILD: 0.25, P_PLANT: 0.30, P_DIG: DIG_CAP,
}


def _unlocked_cells(tiles):
    return [(x, y) for y, row in enumerate(tiles)
            for x, t in enumerate(row) if t != "LOCKED"]


# Optimal assignment without third-party dependencies.  The upstream MIT agent
# used SciPy here and silently fell back to a substantially weaker greedy route
# when SciPy was unavailable.  Kaggle only guarantees the Python standard
# library, so this rectangular Hungarian implementation makes the measured
# policy identical in local and competition runtimes.  Farms create at most a
# few dozen tasks, keeping O(n^2*m) comfortably below the turn budget.
def _linear_sum_assignment(cost):
    if not cost or not cost[0]:
        return [], []
    n_rows = len(cost)
    n_cols = len(cost[0])
    transposed = n_rows > n_cols
    matrix = (
        [[cost[row][col] for row in range(n_rows)] for col in range(n_cols)]
        if transposed
        else cost
    )
    n = len(matrix)
    m = len(matrix[0])
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)

    for row in range(1, n + 1):
        p[0] = row
        col0 = 0
        minimum = [float("inf")] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[col0] = True
            row0 = p[col0]
            delta = float("inf")
            col1 = 0
            for col in range(1, m + 1):
                if used[col]:
                    continue
                current = matrix[row0 - 1][col - 1] - u[row0] - v[col]
                if current < minimum[col]:
                    minimum[col] = current
                    way[col] = col0
                if minimum[col] < delta:
                    delta = minimum[col]
                    col1 = col
            for col in range(m + 1):
                if used[col]:
                    u[p[col]] += delta
                    v[col] -= delta
                else:
                    minimum[col] -= delta
            col0 = col1
            if p[col0] == 0:
                break
        while True:
            col1 = way[col0]
            p[col0] = p[col1]
            col0 = col1
            if col0 == 0:
                break

    row_indices = []
    col_indices = []
    for col in range(1, m + 1):
        if p[col] == 0:
            continue
        row_indices.append(p[col] - 1)
        col_indices.append(col - 1)
    if transposed:
        return col_indices, row_indices
    return row_indices, col_indices


_HUNGARIAN = True


def _step_toward_action(ux, uy, cell):
    return [_step_toward(ux, uy, cell[0], cell[1])]


def _step_toward(ux, uy, tx, ty):
    if ux < tx:
        return "EAST"
    if ux > tx:
        return "WEST"
    if uy < ty:
        return "SOUTH"
    return "NORTH"


def _survey(tiles, day=0):
    s = {"empty": [], "animals": {"COW": 0, "SHEEP": 0, "GOOSE": 0},
         "free_pasture": [], "free_coop": [], "pastures": 0, "coops": 0,
         "plants": 0, "wheat": 0, "straw": 0, "melon": 0, "n_animals": 0,
         "straw_today": 0}
    for (x, y) in _unlocked_cells(tiles):
        t = tiles[y][x]
        if t is None:
            s["empty"].append((x, y))
        elif isinstance(t, dict):
            k = t.get("kind")
            if k == "PASTURE":
                s["pastures"] += 1
                a = t.get("animal")
                if a:
                    s["animals"][a] = s["animals"].get(a, 0) + 1
                    s["n_animals"] += 1
                else:
                    s["free_pasture"].append((x, y))
            elif k == "COOP":
                s["coops"] += 1
                a = t.get("animal")
                if a:
                    s["animals"][a] = s["animals"].get(a, 0) + 1
                    s["n_animals"] += 1
                else:
                    s["free_coop"].append((x, y))
            elif k == "PLANT":
                s["plants"] += 1
                if t.get("crop") == "WHEAT":
                    s["wheat"] += 1
                elif t.get("crop") == "STRAWBERRY":
                    s["straw"] += 1
                    # Cohort age matters: strawberry fires 4 scheduled
                    # productions then decays into a weed. Tracking today's
                    # plantings is what lets us stagger instead of planting the
                    # whole block in one wave (see STRAW_PER_DAY).
                    if t.get("planted_day") == day:
                        s["straw_today"] += 1
                elif t.get("crop") == "MELON":
                    s["melon"] += 1
    return s


def _next_production_day(tile, animal, day):
    """Day of this animal's next scheduled production, or None if never again.

    Terminal-condition audit: FEED and CARE only pay through a future
    production. FEED supplies the CARE bonus and keeps the animal alive; CARE
    banks a bonus redeemed at the next production. Once no production remains
    inside the horizon, both are pure cost — and so is the walking to reach
    them. Harvesting and fertilizer collection still matter, so the animal is
    not abandoned, just no longer fed.
    """
    info = ANIMAL_INFO[animal]
    placed = tile.get("placed_day", 0)
    first, interval = info["first"], info["interval"]
    age = day - placed
    if age < first:
        next_age = first
    else:
        steps = ((age + 1) - first + interval - 1) // interval
        next_age = first + interval * max(steps, 0)
    nd = placed + next_age
    return nd if nd <= LAST_DAY else None


def _wanted(surv, shed, carried):
    """Per-type animal deficit, counting stock already bought but not placed."""
    out = {}
    for a in ("COW", "SHEEP", "GOOSE"):
        cap = {"COW": COW_MAX, "SHEEP": SHEEP_MAX, "GOOSE": GOOSE_MAX}[a]
        have = surv["animals"].get(a, 0) + shed.get(a, 0) + carried.get(a, 0)
        out[a] = max(0, cap - have)
    return out


def _build_tasks(obs, me, priv, surv, wanted, available):
    tiles = me["tiles"]
    day = obs["day"]
    days_left = 30 - day
    tasks = []

    for (x, y) in _unlocked_cells(tiles):
        t = tiles[y][x]
        if not isinstance(t, dict):
            continue
        kind = t.get("kind")

        if kind == "WEED":
            tasks.append((P_DIG, (x, y), ["DIG"], None))

        elif kind == "PLANT":
            crop = t.get("crop")
            info = CROP_INFO.get(crop, CROP_INFO["WHEAT"])
            age = day - t.get("planted_day", day)
            units = t.get("yield_units", 0)
            if info["ongoing"]:
                ready = units > 0
            else:
                ready = units >= info["wcap"] or (units > 0 and age > info["maxday"])
            if ready:
                tasks.append((P_HARVEST, (x, y), ["HARVEST"], None))
            if not t.get("watered_today", False) and (
                    info["ongoing"] or age <= info["maxday"]):
                pri = P_RESCUE if t.get("consecutive_unwatered", 0) >= 1 else P_WATER
                tasks.append((pri, (x, y), ["WATER"], None))
            # Fertilize ongoing crops: a fertilized AND watered scheduled
            # production yields 2 instead of 1, and cover lasts 3 days. On
            # strawberry that is ~+1.5 units (~$300) per fertilizer, against
            # ~$60 selling the fertilizer itself. The animals supply it free.
            if (FERTILIZE_CROPS and info["ongoing"] and crop == "STRAWBERRY"
                    and t.get("fertilized_until_day", -1) < day
                    and age >= info["first"] - 2):
                tasks.append((P_FERTILIZE, (x, y), ["FERTILIZE"], "FERTILIZER"))

        elif kind in ("PASTURE", "COOP"):
            animal = t.get("animal")
            if not animal:
                # Place from STOCK ON HAND, not from the deficit. Gating this
                # on `wanted` was self-cancelling: buying an animal counts it
                # as "have", which removed the very task that would place it,
                # so 13 animals sat in the shed all game eating capacity.
                for a in ("COW", "SHEEP", "GOOSE"):
                    if ANIMAL_INFO[a]["struct"] == kind and available.get(a, 0) > 0:
                        tasks.append((P_PLACE, (x, y), ["PLACE", a], a))
                        available[a] -= 1
                        break
                continue
            info = ANIMAL_INFO[animal]
            # Feed EVERY day: survival needs it every other day, but the CARE
            # bank is wiped on any unfed production day, and the bank is where
            # most of the yield comes from.
            #
            # ...until the animal has no production left inside the horizon, at
            # which point feeding and caring buy nothing and the walk to reach
            # them is wasted too.
            still_produces = (not TERMINAL_STOP) or (
                _next_production_day(t, animal, day) is not None)
            if still_produces:
                if not t.get("fed_today"):
                    pri = P_RESCUE if t.get("consecutive_unfed", 0) >= 1 else P_FEED
                    tasks.append((pri, (x, y), ["FEED"], "WHEAT"))
                elif not t.get("cared_today"):
                    tasks.append((P_CARE, (x, y), ["CARE"], None))
            if ENABLE_FERT and t.get("fertilizer_available"):
                tasks.append((P_FERT, (x, y), ["COLLECT_FERTILIZER"], None))
            units = t.get("yield_units", 0)
            # Do not let yield_units sit at max_held; production above the cap
            # is silently lost.
            # A cared-for cow adds 3 milk every 2 days into a 6 cap, so a late
            # harvest silently loses the overflow. Smaller slack harvests
            # sooner (more actions, less waste).
            if (units >= max(2, info["max_held"] - ANIMAL_HARVEST_SLACK)
                    or (units > 0 and days_left <= 2)):
                tasks.append((P_HARVEST, (x, y), ["HARVEST"], None))

    # --- Empty tiles: structures first, then feed wheat, then strawberry.
    seeds = dict(priv.get("seeds", {}) or {})
    # Build structures just ahead of the herd, never to the full quota. An
    # empty pasture is dead land, and building all 29 on day 1 both wasted the
    # opening actions and encouraged buying animals we could not feed.
    grazers = surv["animals"]["COW"] + surv["animals"]["SHEEP"]
    want_pasture = max(0, min(COW_MAX + SHEEP_MAX, grazers + 3) - surv["pastures"])
    want_coop = max(0, min(GOOSE_MAX, surv["animals"]["GOOSE"] + 2) - surv["coops"])
    wheat_planned = surv["wheat"]
    straw_planned = surv["straw"]
    melon_planned = surv["melon"]
    straw_today = surv["straw_today"]

    # Build CLOSE TO THE SHED. Livestock needs FEED + CARE + HARVEST every
    # single day, so a pasture's distance from the shed is paid over and over.
    # Row-major order put the first pastures in the far corner and buried the
    # workforce in walking.
    half = len(tiles) // 2
    shed_cells = [(half - 1, half - 1), (half, half - 1),
                  (half - 1, half), (half, half)]

    def _shed_dist(c):
        return min(abs(c[0] - sx) + abs(c[1] - sy) for sx, sy in shed_cells)

    for (x, y) in sorted(surv["empty"], key=_shed_dist):
        if want_pasture > 0 and days_left > 6:
            tasks.append((P_BUILD, (x, y), ["BUILD_PASTURE"], None))
            want_pasture -= 1
            continue
        if want_coop > 0 and days_left > 5:
            tasks.append((P_BUILD, (x, y), ["BUILD_COOP"], None))
            want_coop -= 1
            continue
        if (wheat_planned < WHEAT_TILES and seeds.get("WHEAT", 0) > 0
                and days_left > 4):
            tasks.append((P_PLANT, (x, y), ["PLANT", "WHEAT"], None))
            seeds["WHEAT"] = seeds.get("WHEAT", 0) - 1
            wheat_planned += 1
            continue
        # Melon BEFORE strawberry. It is gated to the opening window anyway, and
        # placing it last made the quota dead code: wheat and strawberry
        # consumed every empty tile first, so melon was never once planted
        # (MELON_MAX 0/6/11/16 all scored byte-identically).
        if (melon_planned < MELON_MAX and seeds.get("MELON", 0) > 0
                and day <= MELON_LAST_PLANT_DAY and days_left > 11):
            tasks.append((P_PLANT, (x, y), ["PLANT", "MELON"], None))
            seeds["MELON"] = seeds.get("MELON", 0) - 1
            melon_planned += 1
            continue
        # Stagger the cohorts. Strawberry fires exactly 4 scheduled productions
        # and then decays, so planting the whole block in one wave gives one
        # wave of income and one synchronised death: v14's 24 tiles all expire
        # around days 26-28 and leave 24-31 tiles idle for the rest of the game,
        # which is where the meta triples from day 20 and we only double.
        # Capping plantings per day spreads the ages, so some tiles are always
        # mid-production and replanting is continuous. It also smooths the daily
        # watering load instead of spiking it.
        if (straw_today < STRAW_PER_DAY
                and straw_planned < STRAWBERRY_MAX and seeds.get("STRAWBERRY", 0) > 0
                and days_left > 12):
            tasks.append((P_PLANT, (x, y), ["PLANT", "STRAWBERRY"], None))
            seeds["STRAWBERRY"] = seeds.get("STRAWBERRY", 0) - 1
            straw_today += 1
            straw_planned += 1
            continue
        # Endgame fill. Strawberry decays after its fourth production, so from
        # ~day 26 a large block of tiles falls vacant with nothing eligible to
        # take it: strawberry can no longer mature, and the normal wheat quota
        # is already met. Those tiles then earn nothing for the rest of the game
        # — 24 to 31 of them in a v14 trace, which is exactly the stretch where
        # the meta triples from day 20 and we only double.
        #
        # Anything that still fits the remaining horizon beats leaving them
        # empty. Wheat needs 4 days to harvest, carrot only 3, so carrot covers
        # the last window wheat cannot.
        if days_left <= ENDGAME_FILL_FROM:
            for crop in ("WHEAT", "CARROT"):
                if (seeds.get(crop, 0) > 0
                        and days_left > CROP_INFO[crop]["maxday"]):
                    tasks.append((P_PLANT, (x, y), ["PLANT", crop], None))
                    seeds[crop] = seeds.get(crop, 0) - 1
                    break
        # Melon last. No shop demands it, so its price only falls — but the
        # first ~150 units of the season still clear near the $250 base, and
        # the winner banked 144 of them. Strictly a leftover-tile crop.
        # (melon is handled above, before strawberry)

    tasks.sort(key=lambda r: r[0])
    return tasks


def _assign(units, tasks, shed_adj, inventories, shed, needs, day=0):
    n = len(units)
    actions = [None] * n
    free = set(range(n))

    def inv_of(i):
        return (inventories[i] if i < len(inventories) else {}) or {}

    def produce_load(inv):
        """Carried SELLABLE produce only.

        The drop/return thresholds must ignore carried inputs. Counting them
        made a unit that had just picked up feed immediately trip the drop
        threshold and dump it straight back, which forced WHEAT_CARRY below
        DROP_AT and cost ~474 shed round-trips a game.
        """
        return sum(v for k, v in inv.items() if k not in INPUT_ITEMS)

    # --- Shed logistics: restock wheat (feed) and pick up animals to place.
    for i in list(free):
        ux, uy = units[i]
        if (ux, uy) not in shed_adj:
            continue
        inv = inv_of(i)
        # Endgame liquidation. Only coins score, so anything still carried when
        # the season ends is worth exactly zero — an audit found 57 units
        # ($3,287) stranded in inventories at the buzzer while units sat at
        # 25.5% PASS in the last three days. From ENDGAME_DUMP_DAY the
        # thresholds collapse to 1: ferry everything to the shed and let the
        # market orders bank it.
        drop_at = 1 if day >= ENDGAME_DUMP_DAY else DROP_AT
        if produce_load(inv) >= drop_at:
            actions[i] = ["DROP"]
            free.discard(i)
            continue
        picked = False
        for a in ("COW", "SHEEP", "GOOSE"):
            if needs["animals"].get(a, 0) > 0 and shed.get(a, 0) > 0 and not inv.get(a):
                take = min(2, shed.get(a, 0), needs["animals"][a])
                actions[i] = ["PICKUP", a, take]
                shed[a] -= take
                needs["animals"][a] -= take
                free.discard(i)
                picked = True
                break
        if picked:
            continue
        if (needs["fert"] > 0 and shed.get("FERTILIZER", 0) > 0
                and inv.get("FERTILIZER", 0) < 2):
            take = min(FERT_CARRY, shed.get("FERTILIZER", 0), needs["fert"])
            actions[i] = ["PICKUP", "FERTILIZER", take]
            shed["FERTILIZER"] -= take
            needs["fert"] -= take
            free.discard(i)
            continue
        if needs["wheat"] > 0 and shed.get("WHEAT", 0) > 0 and inv.get("WHEAT", 0) < 2:
            # Carry only a few feeds' worth. Hauling 10 each meant ~15 units
            # held ~150 wheat that overflowed the shed every night — that, not
            # produce, was the bulk of the 963 units lost per game.
            take = min(WHEAT_CARRY, shed.get("WHEAT", 0), needs["wheat"])
            actions[i] = ["PICKUP", "WHEAT", take]
            shed["WHEAT"] -= take
            needs["wheat"] -= take
            free.discard(i)

    for i in list(free):
        if produce_load(inv_of(i)) >= (1 if day >= ENDGAME_DUMP_DAY else RETURN_AT):
            ux, uy = units[i]
            tx, ty = min(shed_adj, key=lambda c: abs(c[0] - ux) + abs(c[1] - uy))
            actions[i] = [_step_toward(ux, uy, tx, ty)]
            free.discard(i)

    used = {}
    claimed_cells = set()

    # --- Standing-on-work pass.
    #
    # A unit already standing on a tile that needs something can act for FREE;
    # any other assignment costs it a walk. The priority loop below does not
    # know that: it walks tasks in priority order and hands each one its
    # nearest free unit, so a unit standing on ready work is routinely dragged
    # away to something marginally more urgent several tiles off, and BOTH
    # jobs then cost travel.
    #
    # The top agents spend 53.9% of actions moving and PASS 5.4%; we spend
    # 59.7% moving and PASS 13.4%. Claiming zero-distance work first is the
    # cheapest way to close that: it never adds a step, and it frees the
    # priority loop to send someone who has to walk anyway.
    unit_at = {}
    for i in free:
        unit_at.setdefault(units[i], []).append(i)
    for t_idx, (pri, cell, op, req) in enumerate(tasks):
        if cell in claimed_cells or cell not in unit_at:
            continue
        for i in list(unit_at[cell]):
            if i not in free:
                continue
            if req is not None and inv_of(i).get(req, 0) <= 0:
                continue
            actions[i] = list(op)
            free.discard(i)
            used[pri] = used.get(pri, 0) + 1
            claimed_cells.add(cell)
            break
    # Tier caps existed to stop one crowded priority monopolising a GREEDY
    # loop. With optimal matching and a priority handicap in the cost, that job
    # is already done by the objective — and the caps now throttle the solver's
    # options instead: widening CAND_MULT changed nothing because the caps, not
    # the multiplier, were bounding the candidate set.
    caps = {p: max(1, int(round(n * f * TIER_CAP_SCALE)))
            for p, f in TIER_CAP.items()}

    # --- Candidate set: tasks the crew may take this turn, tier caps applied.
    cand = []
    for pri, cell, op, req in tasks:
        if cell in claimed_cells:
            continue
        if used.get(pri, 0) + sum(1 for c in cand if c[0] == pri) >= \
                caps.get(pri, max(1, n // 3)):
            continue
        cand.append((pri, cell, op, req))
        if len(cand) >= len(free) * CAND_MULT + 4:
            break

    # --- Global assignment.
    #
    # Greedy sequential assignment — walk tasks in priority order, give each its
    # nearest free unit — is a classic instantaneous-assignment heuristic and is
    # known to be up to 2x worse than an optimal matching on total distance,
    # because an early task can claim the one unit a later task needed. Solving
    # the whole units x tasks matrix at once with the Hungarian algorithm
    # (linear_sum_assignment) removes that.
    #
    # Cost is Manhattan distance plus a priority handicap, so urgency is worth a
    # fixed number of tiles of travel rather than an absolute ordering.
    # A 12x40 solve costs ~0.002 ms, against a ~0.14 ms/turn budget.
    free_list = sorted(free)
    if _HUNGARIAN and free_list and cand:
        big = 10_000.0

        # --- Cluster discount (cheap route bundling).
        #
        # The matching minimises THIS turn's travel, which is myopic: sending a
        # unit to an isolated near task looks better than sending it to a
        # slightly farther task that has three neighbours, even though the
        # second lets it chain work for free on following turns. A full VRP is
        # overkill at this size; discounting a task by how many other pending
        # tasks sit within CLUSTER_RADIUS captures most of the savings idea and
        # costs one pass over the candidate list.
        if CLUSTER_BONUS:
            neigh = [0] * len(cand)
            for a in range(len(cand)):
                ax, ay = cand[a][1]
                for b in range(a + 1, len(cand)):
                    bx, by = cand[b][1]
                    if abs(ax - bx) + abs(ay - by) <= CLUSTER_RADIUS:
                        neigh[a] += 1
                        neigh[b] += 1
        else:
            neigh = [0] * len(cand)

        cost = [[0.0 for _ in cand] for _ in free_list]
        for r, i in enumerate(free_list):
            ux, uy = units[i]
            inv = inv_of(i)
            for c, (pri, cell, op, req) in enumerate(cand):
                if req is not None and inv.get(req, 0) <= 0:
                    cost[r][c] = big
                else:
                    cost[r][c] = (abs(ux - cell[0]) + abs(uy - cell[1])
                                  + PRIORITY_WEIGHT * pri
                                  - CLUSTER_BONUS * min(neigh[c], CLUSTER_CAP))
        rows, cols = _linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            if cost[r][c] >= big:
                continue          # unit cannot service this task; leave it free
            i = free_list[r]
            pri, cell, op, _req = cand[c]
            ux, uy = units[i]
            actions[i] = list(op) if (ux, uy) == cell else \
                _step_toward_action(ux, uy, cell)
            free.discard(i)
            claimed_cells.add(cell)
    else:
        # Fallback: original greedy pass, used if scipy is unavailable.
        for pri, cell, op, req in cand:
            if not free:
                break
            if cell in claimed_cells:
                continue
            cands = [u for u in free if req is None or inv_of(u).get(req, 0) > 0]
            if not cands:
                continue
            tx, ty = cell
            i = min(cands, key=lambda u: abs(units[u][0] - tx) + abs(units[u][1] - ty))
            ux, uy = units[i]
            actions[i] = list(op) if (ux, uy) == cell else [_step_toward(ux, uy, tx, ty)]
            free.discard(i)
            claimed_cells.add(cell)

    # Leftover units take the nearest unclaimed task regardless of tier cap.
    if free:
        for pri, cell, op, req in tasks:
            if not free:
                break
            if cell in claimed_cells:
                continue
            cands = [u for u in free if req is None or inv_of(u).get(req, 0) > 0]
            if not cands:
                continue
            tx, ty = cell
            i = min(cands, key=lambda u: abs(units[u][0] - tx) + abs(units[u][1] - ty))
            ux, uy = units[i]
            actions[i] = list(op) if (ux, uy) == cell else [_step_toward(ux, uy, tx, ty)]
            free.discard(i)
            claimed_cells.add(cell)

    for i in free:
        actions[i] = ["PASS"]
    return actions


def _market_orders(obs, me, priv, surv, wanted):
    orders = []
    day, hour = obs["day"], obs["hour"]
    money = me["money"]
    prices = dict(obs.get("market", {}).get("prices", {}) or {})
    shed = dict(priv.get("shed", {}) or {})
    shed_total = sum(v for v in shed.values() if v > 0)
    n_unlocked = len(_unlocked_cells(me["tiles"]))

    workload = surv["n_animals"] * 3 + surv["plants"] + len(surv["empty"]) // 3
    if hour <= 2:
        target = min(MAX_HANDS, max(MIN_HANDS, workload // HAND_DIVISOR))
        have = len(me.get("hands", []) or [])
        if me.get("hires_today", 0) < target and money > 40:
            for _ in range(min(target - have, 6)):
                orders.append(["HIRE"])

    # --- FEED FIRST. An animal is a $400-500 asset producing ~$270/day, and it
    # escapes permanently after two unfed days. Buying feed therefore outranks
    # buying more animals and buying land. v11's first cut ordered these the
    # other way round, spent the bankroll on turn one, and watched the herd
    # starve and escape for the rest of the game (final score: 2,448).
    # Feed buffer. Every slot of wheat held is a shed slot NOT available for
    # produce, and end-of-day overflow is discarded outright — a 3-day buffer
    # choked the shed and threw away the milk it was meant to protect.
    need_feed = int(surv["n_animals"] * FEED_BUFFER) + 5
    seeds = priv.get("seeds", {}) or {}
    if shed.get("WHEAT", 0) < need_feed and money > 60:
        short = need_feed - shed.get("WHEAT", 0)
        orders.append(["BUY_PRODUCT", "WHEAT", min(short, 15)])
        money -= min(short, 15) * prices.get("WHEAT", 25)
    if surv["wheat"] < WHEAT_TILES and seeds.get("WHEAT", 0) < 6 and money > 300:
        orders.append(["BUY_SEED", "WHEAT", 8])
        money -= 80

    # --- Endgame fill seed. The normal wheat order is capped at WHEAT_TILES, so
    # once strawberry decays and frees a large block there is nothing in stock
    # to plant on it. Buy cheap short-horizon seed to cover the spare tiles:
    # wheat while it can still reach harvest (4 days), then carrot (3 days).
    days_left_m = 30 - day
    if days_left_m <= ENDGAME_FILL_FROM and money > 400:
        spare = len(surv["empty"])
        if spare > 0:
            if days_left_m > CROP_INFO["WHEAT"]["maxday"]:
                want = spare - seeds.get("WHEAT", 0)
                if want > 0:
                    orders.append(["BUY_SEED", "WHEAT", min(want, 12)])
            elif days_left_m > CROP_INFO["CARROT"]["maxday"]:
                want = spare - seeds.get("CARROT", 0)
                if want > 0:
                    orders.append(["BUY_SEED", "CARROT", min(want, 12)])

    # --- Animals, best payback first. Cow returns ~$270/day on $400. Keep a
    # real reserve so the next few days of feed are always affordable.
    if day <= LAST_ANIMAL_DAY and shed_total < SHED_CAP - 5:
        # Reserve enough cash to keep the WHOLE herd fed for several days at
        # the CURRENT wheat price, not a flat sum. An animal we cannot feed is
        # not an asset: it escapes after two unfed days and the $400 is gone.
        #
        # Replay 90323915 is what the flat reserve bought — every coin went
        # into livestock, the herd starved from 2 cows to 0 by day 6, money sat
        # near $30 for sixteen days (too poor even to hire), and we lost 47,005
        # to 140,747. Wheat is contested, so its price is exactly what should
        # gate the buying.
        wheat_px = max(prices.get("WHEAT", 25), 25)
        reserve = 300 + wheat_px * (surv["n_animals"] + 1) * ANIMAL_BUY_FEED_DAYS
        for a in ("COW", "SHEEP", "GOOSE"):
            need = wanted.get(a, 0)
            if need <= 0:
                continue
            cost = ANIMAL_INFO[a]["cost"]
            n = min(need, 2)
            if money >= cost * n + reserve:
                orders.append(["BUY_ANIMAL", a, n])
                money -= cost * n

    # --- Land last: it competes directly with livestock for cash, and an
    # animal pays back in under two days.
    owned = len(me.get("unlocked_quadrants", []) or [])
    if owned < MAX_QUADRANTS and 5 <= day <= 20:
        idx = owned - 1
        if 0 <= idx < len(LAND_COSTS) and money >= LAND_COSTS[idx] + 2500:
            orders.append(["BUY_LAND"])
            money -= LAND_COSTS[idx]

    if STRAWBERRY_MAX and day <= 13:
        want = STRAWBERRY_MAX - surv["straw"] - seeds.get("STRAWBERRY", 0)
        # The meta starts buying strawberry seed on day 4; a $900 floor delays
        # us to ~day 8 because the opening bankroll is in livestock. Unlike
        # melon, strawberry is town-demanded and its price climbs, so buying
        # earlier may pay even though the cash competes with the herd.
        if want > 0 and money > STRAW_SEED_MIN_CASH:
            orders.append(["BUY_SEED", "STRAWBERRY", min(want, 5)])
    if MELON_MAX and day <= MELON_LAST_PLANT_DAY:
        want = MELON_MAX - surv["melon"] - seeds.get("MELON", 0)
        # A $1,200 gate made this unreachable: the opening bankroll goes into
        # livestock, so cash sits at $7-547 through day 8 and melon seed was
        # never once bought.
        if want > 0 and money > MELON_SEED_MIN_CASH:
            orders.append(["BUY_SEED", "MELON", min(want, 4)])

    # --- Sales. Town-demanded goods hold their price, so move them in volume.
    pressure = shed_total > 0.7 * SHED_CAP
    # Feed reserve protects the herd — but on the closing days there is no herd
    # left to protect, and held wheat scores nothing. An audit found 23 wheat
    # ($1,242) sitting unsold in the shed at the buzzer purely because of this.
    wheat_reserve = 0 if day >= ENDGAME_DUMP_DAY else need_feed
    sells = []
    for item, qty in sorted(shed.items(), key=lambda kv: -BASE_PRICE.get(kv[0], 0)):
        if qty <= 0 or item not in BASE_PRICE:
            continue
        if item == "WHEAT":
            sellable = qty - wheat_reserve
            if sellable > 0:
                sells.append(["SELL", "WHEAT", sellable])
            continue
        if item == "FERTILIZER" and FERTILIZE_CROPS and surv["straw"] > 0:
            # Worth ~$300 of strawberry each; only sell the surplus.
            sellable = qty - min(12, surv["straw"])
            if sellable > 0:
                sells.append(["SELL", "FERTILIZER", sellable])
            continue
        price = prices.get(item, BASE_PRICE[item])
        if pressure or day >= 29:
            sells.append(["SELL", item, qty])
        elif item in TOWN_DEMANDED and price >= SELL_THRESHOLD * BASE_PRICE[item]:
            sells.append(["SELL", item, min(qty, DRIP_LIMIT.get(item, 6))])
        elif price >= SELL_THRESHOLD * BASE_PRICE[item]:
            sells.append(["SELL", item, min(qty, DRIP_LIMIT.get(item, 2))])

    if SELLS_FIRST:
        # Purchases keep their slots; sales take the front indices.
        return (sells[:max(0, 10 - len(orders))] + orders)[:10]
    return (orders + sells)[:10]


def _upstream_agent(obs):
    player = obs["player"]
    me = obs["farms"][player]
    priv = obs["private"]
    tiles = me["tiles"]

    half = len(tiles) // 2
    shed_adj = [(half - 1, half - 1), (half, half - 1),
                (half - 1, half), (half, half)]
    usable = [c for c in shed_adj if tiles[c[1]][c[0]] != "LOCKED"] or shed_adj

    surv = _survey(tiles, obs["day"])
    inventories = list(priv.get("inventories", []) or [])
    shed = dict(priv.get("shed", {}) or {})
    carried = {}
    for iv in inventories:
        for k, v in (iv or {}).items():
            carried[k] = carried.get(k, 0) + v

    wanted = _wanted(surv, shed, carried)
    available = {a: shed.get(a, 0) + carried.get(a, 0)
                 for a in ("COW", "SHEEP", "GOOSE")}
    units = [tuple(me["farmer"])] + [tuple(h) for h in (me.get("hands", []) or [])]
    tasks = _build_tasks(obs, me, priv, surv, wanted, available)
    needs = {
        "wheat": sum(1 for t in tasks if t[3] == "WHEAT"),
        "fert": sum(1 for t in tasks if t[3] == "FERTILIZER"),
        "animals": {a: sum(1 for t in tasks if t[3] == a)
                    for a in ("COW", "SHEEP", "GOOSE")},
    }
    acts = _assign(units, tasks, usable, inventories, shed, needs, obs["day"])
    market = _market_orders(obs, me, priv, surv, wanted)

    return {"farmer": acts[0], "hands": acts[1:], "market": market}


def _safe_step(obs):
    """Return a concrete step even for Kaggle's occasional ``step=None``."""

    raw = obs.get("step")
    if raw is not None:
        return int(raw)
    return int(obs.get("day") or 0) * 24 + int(obs.get("hour") or 0)


def _protect_drops_and_shadow_shed(obs, action):
    """Avoid shed-overflow loss and expose same-turn deposits to the seller.

    Unit actions execute before market orders.  The upstream policy did not
    model that ordering: several workers could all issue DROP against the same
    last shed slots, silently discarding the overflow, while goods successfully
    deposited on the final turn were invisible to its market planner.  We walk
    unit actions in execution order, replace unsafe all-item drops with a
    capacity-bounded PLACE of the most valuable carried product, and return the
    exact post-action shed used for liquidation orders.
    """

    player = int(obs.get("player") or 0)
    farms = obs.get("farms") or []
    farm = farms[player]
    private = obs.get("private") or {}
    inventories = list(private.get("inventories") or [])
    shed = {item: max(0, int(qty or 0))
            for item, qty in (private.get("shed") or {}).items()}
    prices = ((obs.get("market") or {}).get("prices") or {})
    positions = [farm.get("farmer") or [0, 0], *(farm.get("hands") or [])]
    tiles = farm.get("tiles") or []
    half = max(1, len(tiles) // 2)
    shed_cells = {(half - 1, half - 1), (half, half - 1),
                  (half - 1, half), (half, half)}
    unit_ops = [action.get("farmer") or ["PASS"], *(action.get("hands") or [])]

    for index, op in enumerate(unit_ops):
        if not op or op[0] not in ("PICKUP", "DROP", "PLACE"):
            continue
        if index >= len(positions) or tuple(positions[index]) not in shed_cells:
            continue
        inventory = inventories[index] if index < len(inventories) else {}
        inventory = {item: max(0, int(qty or 0))
                     for item, qty in (inventory or {}).items()}
        if op[0] == "PICKUP":
            if len(op) >= 2:
                item = op[1]
                requested = max(0, int(op[2] if len(op) >= 3 else 1))
                shed[item] = max(0, shed.get(item, 0) - min(requested, shed.get(item, 0)))
            continue
        # The engine places an animal before checking shed adjacency. A pasture
        # on a shed-access tile must not turn an animal placement into a sale.
        x, y = positions[index]
        tile = tiles[y][x]
        if (op[0] == "PLACE" and len(op) >= 2 and op[1] in ANIMAL_INFO
                and isinstance(tile, dict) and "animal" not in tile
                and tile.get("kind") == ANIMAL_INFO[op[1]]["struct"]):
            continue
        room = max(0, SHED_CAP - sum(shed.values()))

        if op[0] == "DROP":
            load = sum(inventory.values())
            if load <= room:
                for item, qty in inventory.items():
                    shed[item] = shed.get(item, 0) + qty
                continue
            candidates = [(item, qty) for item, qty in inventory.items()
                          if qty > 0 and item in BASE_PRICE]
            if room > 0 and candidates:
                item, qty = max(
                    candidates,
                    key=lambda pair: (prices.get(pair[0], BASE_PRICE[pair[0]]), pair[1]),
                )
                accepted = min(room, qty)
                unit_ops[index] = ["PLACE", item, accepted]
                shed[item] = shed.get(item, 0) + accepted
            else:
                unit_ops[index] = ["PASS"]
            continue

        if len(op) >= 2:
            item = op[1]
            requested = int(op[2] if len(op) >= 3 else 1)
            accepted = min(room, max(0, requested), inventory.get(item, 0))
            if accepted > 0:
                unit_ops[index] = ["PLACE", item, accepted]
                shed[item] = shed.get(item, 0) + accepted
            else:
                unit_ops[index] = ["PASS"]

    action["farmer"] = unit_ops[0]
    action["hands"] = unit_ops[1:]
    return shed


def _liquidation_orders(obs, action, shadow_shed):
    """Sell newly dropped produce on the same turn and guarantee final cash."""

    step = _safe_step(obs)
    day = int(obs.get("day") or step // 24)
    original = list(action.get("market") or [])[:10]
    force_all = step >= 718

    requested = {}
    for order in original:
        if order and order[0] == "SELL" and len(order) >= 3:
            requested[order[1]] = requested.get(order[1], 0) + int(order[2])

    additions = []
    for item, qty in shadow_shed.items():
        if item not in BASE_PRICE or qty <= 0:
            continue
        if force_all:
            extra = qty - requested.get(item, 0)
        elif day >= ENDGAME_DUMP_DAY:
            # On liquidation days the base planner already sells observed shed
            # stock; this captures only goods deposited by this turn's actions.
            observed = int((obs.get("private") or {}).get("shed", {}).get(item, 0) or 0)
            extra = max(0, qty - max(observed, requested.get(item, 0)))
        else:
            extra = 0
        if extra > 0:
            additions.append(["SELL", item, extra])

    if force_all:
        merged = []
        for item, qty in sorted(
            ((item, int(qty)) for item, qty in shadow_shed.items()
             if item in BASE_PRICE and int(qty) > 0),
            key=lambda pair: -BASE_PRICE[pair[0]],
        ):
            merged.append(["SELL", item, qty])
        action["market"] = merged[:10]
        return

    # Existing strategic orders keep their timing; newly bankable terminal
    # produce uses any remaining order slots.
    action["market"] = (original + additions)[:10]


def agent(obs, config=None):
    """Kaggle entry point for the attributed competitive meta policy."""

    try:
        step = _safe_step(obs)
        player = int(obs.get("player") or 0)
        if step < _POLICY_DIAGNOSTICS["last_step"] or player != _POLICY_DIAGNOSTICS["player"]:
            _POLICY_DIAGNOSTICS["errors"] = 0
        _POLICY_DIAGNOSTICS.update(last_step=step, player=player)
        if obs.get("step") is None:
            obs = dict(obs)
            obs["step"] = _safe_step(obs)
        action = _upstream_agent(obs)
        shadow_shed = _protect_drops_and_shadow_shed(obs, action)
        _liquidation_orders(obs, action, shadow_shed)
        return action
    except Exception:
        _POLICY_DIAGNOSTICS["errors"] += 1
        try:
            player = int(obs.get("player") or 0)
            farms = obs.get("farms") or []
            n_hands = len(farms[player].get("hands") or [])
        except Exception:
            n_hands = 0
        return {"farmer": ["PASS"], "hands": [["PASS"]] * n_hands, "market": []}
