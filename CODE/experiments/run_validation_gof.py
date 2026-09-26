"""Goodness-of-fit between the live model and the UCI invoices, in-sample
and on a held-out period.

The Validation tab runs these tests interactively on whatever simulation
the user has just watched. This runner does the same thing headlessly and
reproducibly: it builds the store the other live diagnostics use
(``experiments._live_store``), runs the agent model under the shared live
protocol, and then puts the window's output distributions against the
data with the tests in ``dataset_validation`` at alpha = 0.05:

  * basket size    -- distinct items per visit (KS, 2-sample)
  * per-visit revenue                            (KS, 2-sample)
  * category purchase shares -- purchases per category of the products
    placed in the store, whole simulated visits against whole invoices
    (chi-square statistic; p from reassigning the visit / invoice labels
    at random, which keeps each basket's purchases together, with the
    item-level chi-square p beside it, labelled naive). Its effect size is
    the total-variation distance between the two sides' category shares
    (``share_tv_distance``), which does not depend on how many purchases
    each side contributes; Cramer's V is recorded beside it with its
    balanced form, but V = sqrt(chi2 / N) of a 2 x K table scales with
    sqrt(p (1 - p)), p the simulated share of the purchases, so it shrinks
    whenever one side is much larger -- as the reference is -- and is not
    to be read against conventional benchmarks. The replicas below carry
    both at the simulator's own sample sizes.
  * inter-arrival times -- informational, a test of the DATA and not of the
    simulator: the dataset's own gaps, each rescaled by the rate the
    calibrated hour-of-day profile integrates over it, against Exp(1) (the
    null: a non-homogeneous Poisson process with that profile and the same
    volume every trading day, the live simulator's arrival law), both
    sides stamped at the source's 60 s resolution

The category row is decomposed along the chain that produces it (review
R29): reference invoices -> the list each visit DRAWS at spawn (a stocked
invoice) -> whether the visit pays -> which listed items it buys. Every
cohort visit's drawn list and what it bought are written, one record per
visit, to ``visits.jsonl`` in the run directory, so the chain can be
audited and recomputed. Two basket-level permutation tests of the category
row's own kind run on DISJOINT samples: drawn lists against the reference
invoices (the list law), and the drawn lists of paying visits against
those of visits that did not pay (the selection of which drawn invoices
end as paid visits). The last link is within each visit, so it is counted
rather than tested: how many paying visits bought their whole list, the
share of listed items bought, and the total-variation distance between
the paying visits' drawn and bought category shares. They are reported in
``list_chain``, apart from the four rows above.

The analysed sample is the window's COHORT (review R46), the rule the other
live diagnostics follow: the visits that arrive after the warm-up boundary
and by the end of the window, followed until they leave. After the window
the run carries on, arrivals included at the rate in force when the window
closed, until the last cohort member has left; later arrivals are not
analysed. The earlier version of this runner analysed the visits that
EXITED during the window, whenever they arrived; in steady state that
sample is not length-biased either (up to an edge term the warm-up rule
bounds), so the change aligns the runners' cohort rule rather than
correcting a bias, and a change in a decision between the two versions is
not evidence of one. Each design also records its realised door rate
(arrivals per second over the window), the rate the protocol offered it
(spawn rate times the store's own hour-of-day multiplier) and the share of
arrivals that balked at the cap.

Two designs run in the same invocation, under the same protocol and the
same number of replications:

  in_sample  The store is calibrated on the CURRENT period (the workbook's
      last sheet, Year 2010-2011, every row) and tested against that same
      period. The basket, revenue and category parameters were estimated
      FROM the distributions under test and nothing is held out, so
      agreement shows that the calibrated inputs survive the pipeline --
      adapter, calibration, layout, agent model -- and come back out in the
      agents' behaviour. It cannot show the model predicts data it has not
      seen: these rows are a transfer check on the pipeline. Every
      top-level field of ``summary.json`` keeps this design's meaning.

  held_out   The store is calibrated on the PRIOR period (the first sheet,
      Year 2009-2010, cut to the invoices dated before the current period
      starts, so no invoice or calendar day is shared) -- its layout,
      assortment, prices, basket law and hour-of-day profile all come from
      that year -- and tested against the CURRENT period:
      ``validate_against_simulation(params_current, sim)``. The references
      are therefore the current period's invoices cut to the products the
      held-out store stocks, which the model never saw. What the store
      charges is the prior period's price for each product while the
      revenue reference prices the current invoices at the current
      period's, so the revenue row also carries a year of price drift. The
      inter-arrival row describes the current period's own arrivals and is
      the same in both designs.

``summary.json`` holds the in-sample results at top level, the held-out
results in a ``held_out`` block of the same structure, and a ``design``
block with both periods' date ranges and invoice counts, and for each
design the products its store placed, how many reference invoices hold
at least one of them, ``year_shift``: the same basket, revenue and
category tests between the two periods' own invoices on those products,
``year_shift_fixed_assortment``: the same tests on the products BOTH
designs' stores stock (review R16) -- the drift a store that kept its
assortment would face -- with the remainder labelled popularity turnover
(a store stocking one year's best sellers meets the next year's
regression to the mean), and ``replica``: those tests on replicas that
draw as many of the store period's invoices as the simulator produced
visits -- what a model that transmitted its calibration period perfectly
would score at the simulator's own sample size -- with the direction of
the simulator's statistic relative to the replicas' median. The year shift
compares two whole periods, and a chi-square grows with the sample, so
only the replicas put the simulator's statistics against a yardstick on the
same footing. ``results.csv`` has one row per design, replication and test
(the list-chain tests included); ``visits.jsonl`` one record per cohort
visit (``visits_file`` in the summary and the sidecar names it).

Every replication is a fixed-step headless run
(``CustomerFlowSimulation.run_headless``: the GUI loop's ``step`` with no
sleeping) for ``--warmup`` + ``--seconds`` simulated seconds under its own
seed, then drained: in-sample replication r under ``--seed`` + r, held-out
replication r under ``--seed`` + ``--reps`` + r, so every run has its own
seed. The store starts empty, so the warm-up is deleted. ``--workers`` only
decides how many replications run at once; results are put back in design
order before anything is pooled, so the summary is the same for any worker
count.

The protocol defaults are the nominal load, warm-up and window the other
live diagnostics use; ``run_abm_diagnostics`` records how each value was
chosen and ``run_live_protocol_study`` derives them. Both designs run at
them; the held-out store's own hour-of-day profile sets the multiplier its
arrivals run at, so its door rate differs, and the load block says by how
much.

    python -m experiments.run_validation_gof
    python -m experiments.run_validation_gof --reps 10 --workers 10
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.stats import ks_2samp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from dataset_calibration import (  # noqa: E402
    anonymous_placed_shares, placed_invoice_sample)
from dataset_validation import (  # noqa: E402
    CATEGORY_TEST_KIND, cluster_permutation_chi2, invoice_category_clusters,
    item_product_map, merge_unreferenced_categories,
    placed_product_categories, placed_product_ids,
    validate_against_simulation, visit_category_clusters)
from experiments import _live_store as LS  # noqa: E402
from experiments._common import (make_run_dir, provenance_snapshot,  # noqa: E402
                                 write_sidecar)

# Default protocol, shared with run_abm_diagnostics / measure_queueing.
NOMINAL_SPAWN = 0.27
NOMINAL_CAP = 45
WARMUP_S = 1020.0
COLLECT_S = 1860.0

ALPHA = 0.05

# The analysed sample: the window's arrivals, followed until they leave.
COHORT = 'window_arrivals_drained'
# After the window the run carries on until the cohort has left; the chunk
# is how often that is checked and the cap how long it may take before the
# run is treated as stuck (as in run_abm_diagnostics).
DRAIN_CHUNK_S = 60.0
MAX_DRAIN_S = 3600.0

# Size-matched replicas of each store's calibration period (``_replicas``):
# how many, and the seed their invoice draws come from.
N_REPLICAS = 200
REPLICA_SEED = 20260925

# The two tests of the list chain, each on disjoint samples. Their names
# carry none of the words the paper-macro generator files the four GoF rows
# under.
LIST_CHAIN_DRAWN = 'Drawn lists vs reference invoices'
LIST_CHAIN_SELECTION = 'Drawn lists, paying vs unpaid visits'
LIST_CHAIN_TESTS = (LIST_CHAIN_DRAWN, LIST_CHAIN_SELECTION)

# One record per cohort visit, in design, replication and exit order.
VISITS_FILE = 'visits.jsonl'
VISIT_FIELDS = ('design', 'rep', 'seed', 'spawn_t', 'paid', 'drawn',
                'bought', 'impulse', 'basket', 'revenue')

# The anonymous invoices' share of the lists a store's agents draw from,
# as ``CalibratedParams.seed_into(sim, shop=...)`` writes them into the
# store's calibration (review R27): of the stored lists, and of the items
# on them. Absent (None) when the source has no customer column.
ANONYMOUS_LIST_KEYS = ('anonymous_list_share', 'anonymous_list_item_share')

# Designs, in the order their replications are seeded. Each builds its store
# from its own period and is tested against the reference period.
DESIGNS = ('in_sample', 'held_out')
STORE_PERIOD = {'in_sample': 'current', 'held_out': 'prior'}
REFERENCE_PERIOD = 'current'
DESIGN_LABELS = {'in_sample': 'in_sample_pipeline_transfer',
                 'held_out': 'held_out_prior_period_store'}


class ShopItems:
    """The items of a store, as the category test reads them.

    Only each item's category, product id and zone are kept, as plain data,
    so a worker process can hand the store's assortment back to the parent
    without the simulation attached to it."""

    def __init__(self, floors):
        self.floors = floors


class WindowSim:
    """One collection window's analytics in the shape the tests read.

    ``validate_against_simulation`` takes a simulation and touches only its
    ``analytics`` and the items of its ``shop`` -- the latter to put the
    purchased items in their categories and to narrow the dataset side to
    the products the store carries. A window can therefore be handed to it
    as if it were a run of its own, which is what lets the warm-up be
    deleted and replications be pooled before testing."""

    def __init__(self, analytics, shop):
        self.analytics = analytics
        self.shop = shop


_ITEM_FIELDS = ('category', 'product_id', 'zone')


def _shop_items(shop):
    """Per floor, the metadata of every item that the category test reads."""
    return {fid: {'items': {str(name): {k: (data or {}).get(k)
                                        for k in _ITEM_FIELDS
                                        if (data or {}).get(k) is not None}
                            for name, data in
                            ((floor or {}).get('items') or {}).items()}}
            for fid, floor in shop.floors.items()}


def _shop_items_for(shop_items, product_ids):
    """``shop_items`` cut to the items carrying one of ``product_ids``."""
    keep = {str(p) for p in product_ids}
    return {fid: {'items': {name: data for name, data
                            in (floor.get('items') or {}).items()
                            if str(data.get('product_id')) in keep}}
            for fid, floor in shop_items.items()}


def _arrival_counts(sim):
    """Arrivals admitted and arrivals that balked at the customer cap."""
    A = sim.analytics
    return int(A.get('total_customers', 0)), int(A.get('balked_arrivals', 0))


def _mean_profile_multiplier(sim, t0, t1):
    """Time-average of the hour-of-day multiplier over simulated seconds
    ``[t0, t1]``, as ``_process_arrivals`` applies it (piecewise constant by
    clock hour from ``sim_clock_start_hour``)."""
    prof = np.asarray(sim.hourly_profile, dtype=float)
    h0 = float(sim.sim_clock_start_hour)
    if t1 <= t0:
        return float(prof[int(h0 + t0 / 3600.0) % 24])
    total, t = 0.0, float(t0)
    while t < t1 - 1e-9:
        hour = int(math.floor(h0 + t / 3600.0))
        nxt = min(float(t1), (hour + 1 - h0) * 3600.0)
        if nxt <= t:            # round-off at an hour boundary
            nxt = min(float(t1), t + 1e-6)
        total += float(prof[hour % 24]) * (nxt - t)
        t = nxt
    return total / (t1 - t0)


def _record_visits(sim):
    """Wrap the simulation's spawn and exit so every visit's drawn list and
    outcome are recorded; returns the list the exit records go to.

    A spawning agent's ``shopping_list`` is, at that moment, the stocked
    invoice it drew (the list is later cut when an item proves unreachable,
    by reassignment, so the copy taken here is the draw). At exit the
    simulation's own roll-up runs first; a paying visit's purchases, basket
    size and revenue are then read back from the analytics it appended, so
    the definitions are the Validation tab's. Only reads: the wrappers draw
    no random number and change no state the model reads."""
    exits = []
    spawn_orig = sim._spawn_customer
    exit_orig = sim._process_customer_exit

    def _spawn():
        n = len(sim.customers)
        spawn_orig()
        if len(sim.customers) > n:
            cust = sim.customers[-1]
            cust._gof_drawn = [str(i) for i in cust.shopping_list]

    def _exit(cust):
        A = sim.analytics
        zones = sorted(str(z) for z in cust.visited_zones)
        paid = bool(cust.has_checked_out)
        impulse = [str(i) for i in cust.impulse_items] if paid else []
        n_before = len(A.get('visit_purchases', []))
        exit_orig(cust)
        rec = {'spawn_t': float(getattr(cust, 'spawn_sim_time', 0.0)),
               'paid': paid,
               'drawn': list(getattr(cust, '_gof_drawn', [])),
               'impulse': impulse, 'zones': zones}
        if paid and len(A.get('visit_purchases', [])) > n_before:
            rec['bought'] = [str(i) for i in A['visit_purchases'][-1]]
            rec['basket'] = float(A['basket_sizes'][-1])
            rec['revenue'] = float(A['customer_revenues'][-1])
        exits.append(rec)

    sim._spawn_customer = _spawn
    sim._process_customer_exit = _exit
    return exits


def _visit_record(r):
    """One cohort visit as written to ``visits.jsonl`` (without the design,
    replication and seed, which the writer adds): when it arrived, whether
    it paid, the list it drew, what it bought (None when it did not pay;
    till pick-ups also listed apart in ``impulse``), its basket size and
    revenue."""
    return {'spawn_t': float(r['spawn_t']), 'paid': bool(r['paid']),
            'drawn': list(r['drawn']),
            'bought': list(r['bought']) if 'bought' in r else None,
            'impulse': list(r['impulse']),
            'basket': r.get('basket'), 'revenue': r.get('revenue')}


def _cohort_analytics(cohort):
    """The analytics the tests read, from the cohort's exit records, in
    exit order: basket sizes, per-visit revenues and purchases of the
    paying visits, per-item purchase counts (shopped-for items and till
    pick-ups apart), zone entries, every visit's drawn list, and every
    visit's record for ``visits.jsonl``."""
    paying = [r for r in cohort if 'bought' in r]
    bought, impulse, zones = {}, {}, {}
    for r in paying:
        imp = set(r['impulse'])
        for item in r['bought']:
            if item in imp:
                continue
            bought[item] = bought.get(item, 0) + 1
        for item in r['impulse']:
            impulse[item] = impulse.get(item, 0) + 1
    for r in cohort:
        for z in r['zones']:
            zones[z] = zones.get(z, 0) + 1
    return {
        'basket_sizes': [r['basket'] for r in paying],
        'customer_revenues': [r['revenue'] for r in paying],
        'visit_purchases': [list(r['bought']) for r in paying],
        'item_conversion_rates': {item: {'purchases': n}
                                  for item, n in sorted(bought.items())},
        'impulse_item_sales': dict(sorted(impulse.items())),
        'area_visits': dict(sorted(zones.items())),
        # Every cohort visit draws a list, paying or not.
        'drawn_lists': [list(r['drawn']) for r in cohort],
        'chain': _chain_counts(cohort),
        'visits': [_visit_record(r) for r in cohort],
    }


def _chain_counts(cohort):
    """How the drawn lists became purchases, over the cohort: visits, paying
    visits, paying visits that bought every item of their list, and the
    listed items the paying visits drew and bought."""
    paying = [r for r in cohort if 'bought' in r]
    drawn_items = sum(len(set(r['drawn'])) for r in paying)
    bought_items = sum(len(set(r['drawn']) & set(r['bought'])) for r in paying)
    return {'n_visits': len(cohort),
            'n_paying': len(paying),
            'n_abandoned': sum(1 for r in cohort if not r['paid']),
            'n_paying_whole_list': sum(
                1 for r in paying if set(r['drawn']) <= set(r['bought'])),
            'items_drawn_by_paying': int(drawn_items),
            'items_bought_from_list': int(bought_items)}


def run_once(params, spawn, cap, seconds, warmup=WARMUP_S, dt=0.04, seed=None,
             drain_chunk_s=DRAIN_CHUNK_S, max_drain_s=MAX_DRAIN_S):
    """One replication, fixed-step headless for ``warmup + seconds``
    simulated seconds and then drained, returning the window cohort's
    output samples, its arrival load and the store's items.

    The cohort is the agents spawned after the warm-up boundary and by the
    end of the window. The run then carries on, arrivals included at the
    rate in force when the window closed, until the last of them has left
    (see the module docstring); agents arriving during the drain are not
    analysed. Arrivals and balks are counted over the window itself."""
    shop = LS.build_live_store(params)
    sim = shop.customer_simulation
    sim.max_customers = cap
    sim.spawn_rate = spawn
    exits = _record_visits(sim)

    start = {}

    def _mark_warmup(s):
        # The first call lands on the first tick at or past the warm-up;
        # later periodic calls fall inside the collection window.
        if start:
            return
        start['t'] = float(s.sim_time)
        start['arrivals'] = _arrival_counts(s)

    sim.run_headless(warmup + seconds, dt=dt, seed=seed,
                     callback=_mark_warmup, callback_every_s=warmup)
    t_warm = start['t']
    t_end = float(sim.sim_time)
    admitted, balked = _arrival_counts(sim)
    arrivals = (admitted + balked) - sum(start['arrivals'])
    balks = balked - start['arrivals'][1]
    window = t_end - t_warm
    # The simulated clock, not the window, sets the hour: arrivals read
    # sim_clock_start_hour + sim_time / 3600.
    multiplier = _mean_profile_multiplier(sim, t_warm, t_end)

    def _in_cohort(t):
        return t_warm < t <= t_end

    def _still_in_store():
        return sum(1 for c in sim.customers
                   if _in_cohort(float(getattr(c, 'spawn_sim_time', 0.0))))

    # The drain keeps the arrival rate the window closed at, so the cohort's
    # last visits finish in a store as busy as the one they arrived in.
    hour = int(sim.sim_clock_start_hour + t_end / 3600.0) % 24
    sim.hourly_profile = np.full(
        24, float(np.asarray(sim.hourly_profile, dtype=float)[hour]))
    drained = 0.0
    while _still_in_store() and drained < max_drain_s:
        sim.run_headless(drain_chunk_s, dt=dt)
        drained += drain_chunk_s
    n_censored = _still_in_store()
    if n_censored:
        raise RuntimeError(
            f"{n_censored} agents of the collection window were still in "
            f"the store after draining for {drained:.0f} simulated s; the "
            "cohort would be length-biased")

    # step() counts and survives per-agent exceptions so a GUI session
    # keeps running; a measurement run must not have had any.
    errs = dict(getattr(sim, 'suppressed_errors', {}))
    if errs:
        raise RuntimeError(
            f"live loop suppressed exceptions during measurement: {errs}")

    cohort = [r for r in exits if _in_cohort(r['spawn_t'])]
    win = _cohort_analytics(cohort)
    win['load'] = {'arrivals': int(arrivals), 'balked': int(balks),
                   'window_s': float(window),
                   'profile_multiplier': float(multiplier),
                   'offered_rate_per_s': float(spawn * multiplier),
                   'door_rate_per_s': float(arrivals / window) if window > 0
                   else None,
                   'cohort_visits': len(cohort),
                   'drain_s': float(drained)}
    win['shop_items'] = _shop_items(shop)
    # What share of the lists the agents drew from are anonymous
    # invoices, read from the store's own seeded calibration.
    cal = sim.analytics.get('calibration') or {}
    win['anonymous_lists'] = {k: _json_value(cal.get(k))
                              for k in ANONYMOUS_LIST_KEYS}
    return win


def _design_run(j, plan, stores, args):
    """Run ``j`` of the plan: one replication of one design, on the store
    calibrated for that design, under the seed the plan gives it. Top-level
    so a worker process can run it; it returns the window's samples and the
    parent does every test, so pooling and per-replication testing see the
    same data."""
    design, r, seed = plan[j]
    win = run_once(stores[design], args.spawn, args.cap, args.seconds,
                   warmup=args.warmup, dt=args.dt, seed=seed,
                   max_drain_s=args.max_drain)
    win['design'], win['rep'], win['seed'] = design, r, seed
    return win


def _map_runs(fn, n_runs, workers, *fn_args):
    """``[fn(r, *fn_args) for r in range(n_runs)]``, in replication order.

    With ``workers`` > 1 the replications run in separate processes. Each
    builds its own store and seeds every RNG it draws from, and results are
    put back in replication order before anything is aggregated, so the
    summary is identical to the serial run."""
    if workers <= 1:
        return [fn(r, *fn_args) for r in range(n_runs)]
    done = {}
    pool = ProcessPoolExecutor(max_workers=min(workers, n_runs))
    try:
        futures = {pool.submit(fn, r, *fn_args): r for r in range(n_runs)}
        for fut in as_completed(futures):
            done[futures[fut]] = fut.result()
    except BaseException:
        # Leaving the pool's context manager would wait for every queued
        # replication first, so a failure would only report once they had
        # all run. Drop what has not started and let the error out now.
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return [done[r] for r in sorted(done)]


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--seconds', type=float, default=COLLECT_S,
                   help="Collection window after the warm-up, simulated s")
    p.add_argument('--warmup', type=float, default=WARMUP_S,
                   help="Deleted warm-up, simulated s")
    p.add_argument('--reps', type=int, default=5)
    p.add_argument('--spawn', type=float, default=NOMINAL_SPAWN,
                   help="Arrivals per simulated s, before the hour-of-day "
                        "profile")
    p.add_argument('--cap', type=int, default=NOMINAL_CAP,
                   help="Most customers in the store at once")
    # 0.04 s is the threaded GUI loop's fixed tick, so the headless runs
    # advance the same discrete-time model the GUI does.
    p.add_argument('--dt', type=float, default=0.04,
                   help="Fixed tick, simulated s")
    p.add_argument('--seed', type=int, default=4000,
                   help="Base seed; in-sample replication r runs under "
                        "seed + r, held-out replication r under "
                        "seed + reps + r")
    p.add_argument('--alpha', type=float, default=ALPHA,
                   help="Significance level of every test")
    p.add_argument('--max-drain', type=float, default=MAX_DRAIN_S,
                   help="Longest the run may continue past the window, "
                        "simulated s, while the window's cohort leaves")
    p.add_argument('--workers', type=int, default=1,
                   help="Processes running replications in parallel")
    p.add_argument('--retail-path', type=str, default=None,
                   help="Path to the UCI Online Retail II workbook. Default: "
                        "found by dataset_paths.uci_workbook() "
                        "($UCI_RETAIL_XLSX, then DATASETS/)")
    p.add_argument('--out-root', type=str,
                   default=os.path.join(_HERE, 'results'))
    args = p.parse_args(argv)
    if args.workers < 1:
        p.error("--workers must be >= 1")
    if args.reps < 1:
        p.error("--reps must be >= 1")
    if args.warmup <= 0 or args.seconds <= 0 or args.dt <= 0:
        p.error("--warmup, --seconds and --dt must be positive")
    if args.max_drain <= 0:
        p.error("--max-drain must be positive")
    if not 0.0 < args.alpha < 1.0:
        p.error("--alpha must lie strictly between 0 and 1")
    return args


def _json_value(v):
    """One value of a test's extra record as JSON: a non-finite float
    becomes None, as the statistic and p-value do."""
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, np.integer):
        return int(v)
    return v


def _decision(p, alpha):
    if p is None or not np.isfinite(p):
        return 'N/A'
    return 'PASS' if p > alpha else 'FAIL'


def _test_rows(params, analytics, shop_items, alpha):
    """One dict per goodness-of-fit test of a window's analytics, and the
    report's notes, which say why a test that could not run is missing.

    A test's extra record -- for the category test the naive item-level p,
    the permutation count and seed, the categories compared, the effect
    sizes -- goes into its row beside the common fields."""
    res = validate_against_simulation(
        params, WindowSim(analytics, ShopItems(shop_items)), alpha=alpha)
    rows = []
    for t in res.tests:
        row = {'test': t.name,
               'kind': t.test,
               'statistic': (float(t.statistic)
                             if np.isfinite(t.statistic) else None),
               'p_value': float(t.p_value) if t.applicable() else None,
               'n_observed': int(t.n_observed),
               'n_simulated': int(t.n_simulated),
               'decision': t.verdict(alpha),
               'note': t.note}
        clash = set(row) & set(t.extra)
        if clash:
            raise RuntimeError(f"{t.name}: extra fields {sorted(clash)} would "
                               "overwrite the common ones")
        row.update({k: _json_value(v) for k, v in t.extra.items()})
        rows.append(row)
    return rows, list(res.summary_lines)


def _chain_row(name, r, alpha, note):
    """One list-chain test as a row of the same shape as the GoF rows."""
    p = r['p_value']
    return {'test': name, 'kind': CATEGORY_TEST_KIND,
            'statistic': _json_value(r['statistic']),
            'p_value': _json_value(p),
            'n_observed': int(r['n_ref_clusters']),
            'n_simulated': int(r['n_sim_clusters']),
            'decision': _decision(p, alpha),
            'naive_p_value': _json_value(r['naive_p_value']),
            'share_tv_distance': _json_value(r['share_tv_distance']),
            'effect_size_primary': 'share_tv_distance',
            'cramers_v': _json_value(r['cramers_v']),
            'cramers_v_balanced': _json_value(r['cramers_v_balanced']),
            'sim_share_of_items': _json_value(r['sim_share_of_items']),
            'n_items_observed': int(r['n_items_ref']),
            'n_items_simulated': int(r['n_items_sim']),
            'n_categories': int(r['n_categories']),
            'n_permutations': int(r['n_permutations']),
            'permutation_seed': int(r['permutation_seed']),
            'note': (('Not run: ' + r['reason'] + ' ') if r['reason'] else '')
                    + note}


def _category_totals(clusters):
    """Category shares of a set of per-basket count rows, or None when the
    rows hold no purchase."""
    a = np.asarray(clusters, dtype=float)
    if a.ndim != 2 or not a.size:
        return None
    tot = a.sum(axis=0)
    return tot / tot.sum() if tot.sum() > 0 else None


def _list_chain_tests(params_ref, shop_items, visits, alpha):
    """The category row decomposed along the chain that produces it, over
    ``visits`` (cohort records, ``_visit_record``): per category of the
    products the store placed, in the categories the reference gives them
    (review R29).

    Two links are tested with the category row's own basket-level
    permutation test, each on disjoint samples:

      * the list law -- every visit's drawn list (the stocked invoice it
        took at spawn, paying or not) against the reference invoices. In
        sample each drawn list is a draw from the reference itself.
      * selection -- the drawn lists of the visits that paid against those
        of the visits that did not. Whether a visit pays is decided before
        it walks (the cart decision) or by what it could not reach, so a
        difference here says which drawn invoices end as paid visits.

    The last link -- which listed items a paying visit buys -- is within
    each visit, so the two samples would be the same baskets; it is
    COUNTED instead (``_chain_counts``), and described by the total-variation
    distance between the paying visits' drawn and bought category shares
    (``completion``). Returns the two test rows and the completion
    record."""
    shop = ShopItems(shop_items)
    product_category = placed_product_categories(params_ref, shop)
    categories = sorted(set(product_category.values()))
    paying = [v for v in visits if v['bought'] is not None]
    unpaid = [v for v in visits if v['bought'] is None]
    if not categories:
        empty = {'statistic': float('nan'), 'p_value': float('nan'),
                 'naive_p_value': float('nan'), 'cramers_v': float('nan'),
                 'cramers_v_balanced': float('nan'),
                 'sim_share_of_items': float('nan'),
                 'share_tv_distance': float('nan'), 'n_ref_clusters': 0,
                 'n_sim_clusters': 0, 'n_items_ref': 0, 'n_items_sim': 0,
                 'n_categories': 0, 'n_permutations': 0,
                 'permutation_seed': 0,
                 'reason': "the store's items carry no product ids."}
        return ([_chain_row(name, empty, alpha, '')
                 for name in LIST_CHAIN_TESTS],
                {'share_tv_distance_drawn_vs_bought': None,
                 'n_paying': len(paying)})
    item_product = item_product_map(shop)
    ref = invoice_category_clusters(params_ref, product_category, categories)

    def _clusters(lists):
        rows, _, n_empty = visit_category_clusters(
            lists, item_product, product_category, categories)
        return rows, n_empty

    drawn, drawn_empty = _clusters([v['drawn'] for v in visits])
    r_ref, r_drawn, _, _ = merge_unreferenced_categories(ref, drawn,
                                                         categories)
    first = cluster_permutation_chi2(r_drawn, r_ref)

    paid_drawn, _ = _clusters([v['drawn'] for v in paying])
    unpaid_drawn, _ = _clusters([v['drawn'] for v in unpaid])
    p_rows, u_rows, _, _ = merge_unreferenced_categories(
        paid_drawn, unpaid_drawn, categories)
    # The smaller side, the unpaid visits, sits in the "simulated" row.
    second = cluster_permutation_chi2(u_rows, p_rows)

    # Completion: the paying visits' purchases of their own listed items
    # (till pick-ups are not on a list) against their drawn lists.
    bought, _ = _clusters([[i for i in v['bought'] if i not in
                            set(v['impulse'])] for v in paying])
    d_share, b_share = _category_totals(paid_drawn), _category_totals(bought)
    completion = {
        'share_tv_distance_drawn_vs_bought': (
            float(0.5 * np.abs(d_share - b_share).sum())
            if d_share is not None and b_share is not None else None),
        'n_paying': len(paying),
        'note': ('within-visit, so described rather than tested: the '
                 'paying visits\' drawn and bought category shares')}
    rows = [
        _chain_row(LIST_CHAIN_DRAWN, first, alpha,
                   "Every cohort visit's drawn list (the stocked invoice it "
                   "took at spawn, paying or not) against the reference "
                   "invoices holding a placed product, per category, whole "
                   "baskets relabelled."
                   + (f" {drawn_empty} drawn lists held no placed product."
                      if drawn_empty else '')),
        _chain_row(LIST_CHAIN_SELECTION, second, alpha,
                   f"The drawn lists of the {len(unpaid)} cohort visits that "
                   f"did not pay against those of the {len(paying)} that "
                   "did, per category, whole baskets relabelled: which "
                   "drawn invoices end as paid visits. Disjoint samples; "
                   "the unpaid side is small, so the test has little power "
                   "and a pass is not evidence of no selection."),
    ]
    return rows, completion


def _decision_counts(per_rep, name):
    """PASS / FAIL / N/A counts of one test over the replications."""
    counts = {d: int(sum(1 for rep in per_rep for r in rep['tests']
                         if r['test'] == name and r['decision'] == d))
              for d in ('PASS', 'FAIL')}
    counts['N/A'] = len(per_rep) - counts['PASS'] - counts['FAIL']
    return counts


def _summed(counts):
    out = {}
    for c in counts:
        for k, n in c.items():
            out[k] = out.get(k, 0) + int(n)
    return out


def _chain_summary(chains):
    """The chain counts of several windows, added, with the shares."""
    tot = _summed(chains)
    paying = max(tot.get('n_paying', 0), 1)
    drawn = max(tot.get('items_drawn_by_paying', 0), 1)
    tot['share_paying_whole_list'] = tot.get('n_paying_whole_list', 0) / paying
    tot['share_listed_items_bought'] = (tot.get('items_bought_from_list', 0)
                                        / drawn)
    return tot


def _load_summary(wins):
    """The arrival load of a design's windows: arrivals and balks over all
    of them, the realised door rate, the rate the protocol offered and the
    cohort sizes (review R46)."""
    arrivals = int(sum(w['load']['arrivals'] for w in wins))
    balked = int(sum(w['load']['balked'] for w in wins))
    window = float(sum(w['load']['window_s'] for w in wins))
    return {'arrivals': arrivals, 'balked': balked,
            'balked_frac': round(balked / max(arrivals, 1), 4),
            'window_s_total': window,
            'door_rate_per_s': (arrivals / window) if window > 0 else None,
            'offered_rate_per_s': float(np.mean(
                [w['load']['offered_rate_per_s'] for w in wins])),
            'profile_multiplier': float(np.mean(
                [w['load']['profile_multiplier'] for w in wins])),
            'cohort_visits': int(sum(w['load']['cohort_visits']
                                     for w in wins)),
            'mean_drain_s': float(np.mean([w['load']['drain_s']
                                           for w in wins]))}


def _design_block(design, wins, params_ref, args):
    """Everything one design reports: per-replication and pooled tests of
    its windows against ``params_ref``, the decision split, the list chain,
    the arrival load and the protocol with this design's seeds, in the
    shape the summary's top level has always had."""
    # Per replication first: independent windows, so the spread of the
    # decisions says how much one window's sample size is doing.
    per_rep = []
    for win in wins:
        rows, _ = _test_rows(params_ref, win, win['shop_items'], args.alpha)
        chain, completion = _list_chain_tests(params_ref, win['shop_items'],
                                              win['visits'], args.alpha)
        per_rep.append({'rep': win['rep'], 'seed': win['seed'],
                        'load': win['load'], 'tests': rows,
                        'list_chain': chain, 'chain': win['chain'],
                        'completion': completion})
        print(f"  [{design}] rep {win['rep'] + 1}/{args.reps} "
              f"(seed {win['seed']}): "
              + ', '.join(f"{r['test']}={r['decision']}"
                          for r in rows + chain), flush=True)

    # Pooled: the replications share one calibrated store and differ only in
    # their random streams, so their samples are draws from the same output
    # distribution and can be tested together. The store is built from the
    # calibration alone, so every replication must have laid out the same
    # items; per-item counts are only summed under that condition.
    shop_items = wins[0]['shop_items']
    if any(w['shop_items'] != shop_items for w in wins[1:]):
        raise RuntimeError(f"{design} replications laid out different "
                           "stores; their item purchases cannot be pooled")

    purchases = _summed({item: d['purchases'] for item, d
                         in w['item_conversion_rates'].items()} for w in wins)
    pooled_analytics = {
        'basket_sizes': [v for w in wins for v in w['basket_sizes']],
        'customer_revenues': [v for w in wins for v in w['customer_revenues']],
        'visit_purchases': [v for w in wins for v in w['visit_purchases']],
        'item_conversion_rates': {item: {'purchases': n}
                                  for item, n in purchases.items()},
        'impulse_item_sales': _summed(w['impulse_item_sales'] for w in wins),
        'area_visits': _summed(w['area_visits'] for w in wins),
    }
    pooled, pooled_notes = _test_rows(params_ref, pooled_analytics, shop_items,
                                      args.alpha)
    chain_tests, completion = _list_chain_tests(
        params_ref, shop_items, [v for w in wins for v in w['visits']],
        args.alpha)
    names = [r['test'] for r in pooled]
    return {
        'alpha': args.alpha,
        'pooled': pooled,
        # The report's notes on the pooled window: why a test that could not
        # run is absent above, and the zone traffic shown beside the tests.
        'pooled_notes': pooled_notes,
        # How the replications split, so a pooled decision that rests on one
        # window's sample size is visible as such. 'N/A' is a test that could
        # not run on that window (an empty side, or too few samples) --
        # including one the report left out altogether, as it does for the
        # category test when a window recorded no purchases -- so the three
        # counts always add up to the number of replications.
        'per_test_decisions': {name: _decision_counts(per_rep, name)
                               for name in names},
        # The category row along its chain (review R29), pooled over the
        # replications: drawn lists vs the reference (list law), paying vs
        # unpaid visits' drawn lists (selection), and the within-visit
        # completion counted and described.
        'list_chain': {
            'tests': chain_tests,
            'counts': _chain_summary([w['chain'] for w in wins]),
            'completion': completion,
            'per_test_decisions': {
                name: {d: int(sum(1 for rep in per_rep
                                  for r in rep['list_chain']
                                  if r['test'] == name
                                  and r['decision'] == d))
                       for d in ('PASS', 'FAIL', 'N/A')}
                for name in LIST_CHAIN_TESTS}},
        'n_reps': int(len(wins)),
        'load': _load_summary(wins),
        'protocol': {'mode': 'fixed_step', 'dt': args.dt,
                     'seed_base': args.seed,
                     'seeds': [w['seed'] for w in wins],
                     'reps': args.reps, 'warmup_s': args.warmup,
                     'collect_s': args.seconds, 'spawn': args.spawn,
                     'cap': args.cap, 'workers': args.workers,
                     'cohort': COHORT, 'drain_chunk_s': DRAIN_CHUNK_S,
                     'max_drain_s': args.max_drain,
                     'framing': 'steady_state_replication_deletion'},
        'per_rep': per_rep,
    }


def _sample_stats(sizes):
    """n, median, mean and 90th percentile of a per-invoice size sample."""
    a = np.asarray(sizes, dtype=float)
    if not a.size:
        return {'n': 0, 'median': None, 'mean': None, 'p90': None}
    return {'n': int(a.size), 'median': float(np.median(a)),
            'mean': float(a.mean()), 'p90': float(np.percentile(a, 90))}


def _year_shift(store_sample, ref_sample, store_clusters, ref_clusters):
    """The basket, revenue and category tests between the data the store
    was calibrated on and the reference data it is tested against, on the
    same stocked products: two-sample KS on the stocked part of each
    invoice, and the category row's own test (``cluster_permutation_chi2``)
    with the store period's invoices in the simulated visits' place.

    In the in-sample design the two samples are one, so every distance is
    zero and every p-value 1. In the held-out design it is how far the two
    periods' own invoices differ -- the yardstick for the simulator's
    held-out distance: a simulator that transmitted its calibration period
    perfectly would sit about this far from the reference."""
    out = {}
    for name, key in (('basket', 'sizes'), ('revenue', 'revenues')):
        a = np.asarray(store_sample[key], dtype=float)
        b = np.asarray(ref_sample[key], dtype=float)
        if a.size and b.size:
            r = ks_2samp(a, b)
            out[name] = {'statistic': float(r.statistic),
                         'p_value': float(r.pvalue),
                         'n_store': int(a.size), 'n_reference': int(b.size),
                         'mean_store': float(a.mean()),
                         'mean_reference': float(b.mean())}
        else:
            out[name] = None
    r = cluster_permutation_chi2(store_clusters, ref_clusters)
    out['category'] = ({
        'statistic': float(r['statistic']),
        'p_value': float(r['p_value']),
        'naive_p_value': _json_value(r['naive_p_value']),
        'share_tv_distance': _json_value(r['share_tv_distance']),
        'cramers_v': _json_value(r['cramers_v']),
        'cramers_v_balanced': _json_value(r['cramers_v_balanced']),
        'n_store': int(r['n_sim_clusters']),
        'n_reference': int(r['n_ref_clusters']),
        'n_categories': int(r['n_categories']),
        'n_permutations': int(r['n_permutations']),
        'permutation_seed': int(r['permutation_seed'])}
        if np.isfinite(r['p_value']) else None)
    return out


def _category_clusters(params_ref, params_store, shop_items):
    """Reference and store-period invoices as per-category basket rows over
    the products in ``shop_items``, categories as the reference gives them,
    unreferenced categories pooled as the category row pools them."""
    shop = ShopItems(shop_items)
    product_category = placed_product_categories(params_ref, shop)
    categories = sorted(set(product_category.values()))
    if not categories:
        return np.zeros((0, 0)), np.zeros((0, 0))
    ref_clusters, store_clusters, _, _ = merge_unreferenced_categories(
        invoice_category_clusters(params_ref, product_category, categories),
        invoice_category_clusters(params_store, product_category, categories),
        categories)
    return ref_clusters, store_clusters


def _turnover(shift, fixed):
    """The part of the year shift the fixed assortment does not show,
    labelled popularity turnover (review R16): on the store's own
    assortment minus on the products both stores stock -- the KS distance
    for basket and revenue, the total-variation distance between category
    shares for the category row (a chi-square is not comparable across
    assortments of different sizes)."""
    out = {}
    for name, measure in (('basket', 'statistic'), ('revenue', 'statistic'),
                          ('category', 'share_tv_distance')):
        a, b = (shift or {}).get(name), (fixed or {}).get(name)
        if a is None or b is None or a.get(measure) is None \
                or b.get(measure) is None:
            out[name] = None
            continue
        out[name] = {'measure': ('ks_distance' if measure == 'statistic'
                                 else 'share_tv_distance'),
                     'store_assortment': float(a[measure]),
                     'fixed_assortment': float(b[measure]),
                     'popularity_turnover': float(a[measure] - b[measure])}
    return out


# The category row's effect sizes; the first is the one to read.
EFFECT_SIZES = ('share_tv_distance', 'cramers_v', 'cramers_v_balanced')


def _effect_summary(values, sim_value):
    """Median and central 95% of one effect size over the replicas, with the
    simulator's own value and the share of replicas at or below it."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    out = {'median': float(np.median(v)) if v.size else None,
           'lo': float(np.percentile(v, 2.5)) if v.size else None,
           'hi': float(np.percentile(v, 97.5)) if v.size else None,
           'simulator': None, 'simulator_percentile': None}
    if sim_value is not None and np.isfinite(sim_value) and v.size:
        out['simulator'] = float(sim_value)
        out['simulator_percentile'] = float(np.mean(v <= sim_value))
    return out


def _replica_summary(stats, pvals, sim_stat, alpha, n, seed):
    stats = np.asarray(stats, dtype=float)
    pvals = np.asarray(pvals, dtype=float)
    med = float(np.median(stats))
    out = {'n_replicas': int(stats.size), 'n_per_replica': int(n),
           'seed': int(seed),
           'median': med,
           'lo': float(np.percentile(stats, 2.5)),
           'hi': float(np.percentile(stats, 97.5)),
           'reject_rate': float(np.mean(pvals < alpha)),
           'simulator_statistic': None, 'simulator_percentile': None,
           'simulator_minus_median': None, 'simulator_vs_median': None}
    if sim_stat is not None and np.isfinite(sim_stat):
        out['simulator_statistic'] = float(sim_stat)
        out['simulator_percentile'] = float(np.mean(stats <= sim_stat))
        # Which side of a perfect replay's typical distance the simulator
        # falls (review R16): 'above' is further from the reference.
        out['simulator_minus_median'] = float(sim_stat - med)
        out['simulator_vs_median'] = ('above' if sim_stat > med else
                                      'below' if sim_stat < med else 'at')
    return out


def _location(summary, sim_values, ref_mean, diffs):
    """The direction of the simulator's error in the row's own units: its
    mean minus the reference mean, beside the same difference for the
    replicas (median and central 95%) and where it falls among them."""
    diffs = np.asarray(diffs, dtype=float)
    summary['replica_mean_minus_reference'] = {
        'median': float(np.median(diffs)),
        'lo': float(np.percentile(diffs, 2.5)),
        'hi': float(np.percentile(diffs, 97.5))}
    sim_values = np.asarray(sim_values if sim_values is not None else (),
                            dtype=float)
    if sim_values.size:
        d = float(sim_values.mean() - ref_mean)
        summary['simulator_mean_minus_reference'] = d
        summary['simulator_mean_percentile'] = float(np.mean(diffs <= d))
    else:
        summary['simulator_mean_minus_reference'] = None
        summary['simulator_mean_percentile'] = None


def _replicas(store_sample, ref_sample, store_clusters, ref_clusters,
              pooled, alpha, n_rep=N_REPLICAS, seed=REPLICA_SEED,
              sim_samples=None):
    """What a model that transmitted its calibration period perfectly would
    score, at the sample size the simulator actually produced.

    The simulator's statistics come from its pooled visits; the year shift
    compares two whole periods, so its statistics are not on the same
    footing -- a chi-square grows with the sample, and a KS distance
    carries less sampling noise at full size. Each replica draws as many
    of the store period's invoices, with replacement, as the pooled test
    had simulated visits -- the stocked part of each, which is exactly
    what a shopper of this store draws its list from -- and scores them
    against the reference with the row's own test. Recorded per row: the
    replicas' median statistic and central 95% range, the share of them
    the test rejects at ``alpha`` (the test's power against a perfect
    transfer of the calibration period), where the simulator's own
    statistic falls among them and on which side of their median.

    Given ``sim_samples`` (``{'basket': ..., 'revenue': ...}``, the pooled
    simulated values), the basket and revenue rows also record the
    direction of the error in the row's units: the simulator's mean minus
    the reference mean, beside the replicas' own.

    In the in-sample design the store period is the reference, so the
    replicas are the test's null and their rejection rate sits near
    ``alpha``. In the held-out design they carry the year-to-year drift
    and nothing else."""
    sim, sim_effect = {}, {}
    for row in pooled:
        name = str(row['test']).lower()
        key = next((k for k in ('basket', 'revenue', 'categor') if k in name),
                   None)
        if key and row.get('statistic') is not None:
            sim[key] = (float(row['statistic']), int(row['n_simulated']))
            if key == 'categor':
                sim_effect = {k: row.get(k) for k in EFFECT_SIZES}
    out = {}
    for i, (name, key) in enumerate((('basket', 'sizes'),
                                     ('revenue', 'revenues'))):
        a = np.asarray(store_sample[key], dtype=float)
        b = np.asarray(ref_sample[key], dtype=float)
        if name not in sim or not (a.size and b.size):
            out[name] = None
            continue
        stat, n = sim[name]
        rng = np.random.default_rng([seed, i])
        stats, pvals, diffs = [], [], []
        ref_mean = float(b.mean())
        for _ in range(n_rep):
            draw = a[rng.integers(0, a.size, n)]
            r = ks_2samp(draw, b)
            stats.append(r.statistic)
            pvals.append(r.pvalue)
            diffs.append(float(draw.mean()) - ref_mean)
        out[name] = _replica_summary(stats, pvals, stat, alpha, n, seed)
        if sim_samples is not None:
            _location(out[name], sim_samples.get(name), ref_mean, diffs)
    rows = np.asarray(store_clusters, dtype=float)
    rows = rows[rows.sum(axis=1) > 0] if rows.size else rows
    if 'categor' in sim and len(rows):
        stat, n = sim['categor']
        rng = np.random.default_rng([seed, 2])
        stats, pvals = [], []
        effects = {k: [] for k in EFFECT_SIZES}
        for _ in range(n_rep):
            r = cluster_permutation_chi2(rows[rng.integers(0, len(rows), n)],
                                         ref_clusters)
            stats.append(r['statistic'])
            pvals.append(r['p_value'])
            for k in EFFECT_SIZES:
                effects[k].append(r[k])
        out['category'] = _replica_summary(stats, pvals, stat, alpha, n, seed)
        # The effect sizes of a perfect replay at the simulator's own sample
        # sizes: Cramer's V depends on how the purchases split between the
        # two sides, so only this range makes the simulator's V readable.
        out['category']['effect_sizes'] = {
            k: _effect_summary(effects[k], sim_effect.get(k))
            for k in EFFECT_SIZES}
    else:
        out['category'] = None
    return out


def _stocked_per_invoice(params, pids):
    """Mean stocked products per invoice holding one of ``pids``."""
    if not pids:
        return None
    sizes = np.asarray(placed_invoice_sample(params, pids)['sizes'],
                       dtype=float)
    return float(sizes.mean()) if sizes.size else None


def _store_facts(design, shop_items, params_ref, params_store, pooled=None,
                 alpha=ALPHA, fixed_pids=None, other_pids=None,
                 sim_samples=None, anonymous_lists=None):
    """How a design's store meets the reference period: the products it
    placed, how many of them the reference period sells at all, and how
    many reference invoices hold at least one of them -- the invoices the
    basket and revenue references are cut from.

    Also the two list-length samples involved: the one the store's agents
    draw their shopping-list length from (the stocked part of each invoice
    of the period the store was calibrated on, as seed_into writes it) and
    the reference the basket test compares them against. In the in-sample
    design the two are the same sample.

    The category yardstick puts each product in the category the reference
    calibration gives it and pools the categories no reference invoice
    touches, exactly as the category row does for the simulated visits.

    ``fixed_pids`` -- the products both designs' stores stock -- gives the
    year shift on a fixed assortment and the remainder, popularity
    turnover; ``other_pids`` -- the other design's products -- the mean
    stocked products per invoice of both periods on each assortment, which
    shows the turnover's direction (review R16).

    Given the design's pooled test rows, also ``replica``: the same tests
    on size-matched replicas of the store period (``_replicas``).

    ``anonymous_lists`` -- the store's seeded ``anonymous_list_share``
    and ``anonymous_list_item_share`` (``ANONYMOUS_LIST_KEYS``) -- is
    recorded as it stands: the share of the store's lists, and of the
    items on them, that come from invoices with no customer id
    (review R27). Beside it, ``anonymous_lists_reference`` gives the
    same two shares among the reference period's invoices on the
    store's products (``anonymous_placed_shares``), the invoices the
    basket and revenue rows compare against."""
    shop = ShopItems(shop_items)
    pids = placed_product_ids(shop)
    sold = {str(p) for p in (params_ref.item_visit_counts or {})}
    sample = placed_invoice_sample(params_ref, pids)
    agent_sample = placed_invoice_sample(params_store, pids)
    ref_clusters, store_clusters = _category_clusters(params_ref, params_store,
                                                      shop_items)
    shift = _year_shift(agent_sample, sample, store_clusters, ref_clusters)
    facts = {'label': DESIGN_LABELS[design],
             'store_period': STORE_PERIOD[design],
             'reference_period': REFERENCE_PERIOD,
             'n_placed_products': len(pids),
             'n_placed_products_in_reference': len(pids & sold),
             'n_reference_invoices': int(sample['n_invoices']),
             'n_reference_invoices_with_placed': int(sample['n_with_placed']),
             'list_length_agents': _sample_stats(agent_sample['sizes']),
             'list_length_reference': _sample_stats(sample['sizes']),
             'year_shift': shift}
    anon = dict(anonymous_lists or {})
    for k in ANONYMOUS_LIST_KEYS:
        facts[k] = _json_value(anon.get(k))
    ref_anon = anonymous_placed_shares(params_ref, pids)
    facts['anonymous_lists_reference'] = {
        'anonymous_list_share': _json_value(ref_anon['invoice_share']),
        'anonymous_list_item_share': _json_value(
            ref_anon['purchase_share']),
        'n_invoices_with_placed': int(ref_anon['n_with_placed']),
        'n_anonymous_with_placed': int(
            ref_anon['n_anonymous_with_placed'])}
    if fixed_pids is not None:
        fixed = {str(p) for p in fixed_pids}
        f_items = _shop_items_for(shop_items, fixed)
        f_ref, f_store = _category_clusters(params_ref, params_store, f_items)
        f_shift = _year_shift(placed_invoice_sample(params_store, fixed),
                              placed_invoice_sample(params_ref, fixed),
                              f_store, f_ref)
        facts['fixed_assortment'] = {'n_products': len(fixed),
                                     'n_products_store': len(pids)}
        facts['year_shift_fixed_assortment'] = f_shift
        facts['popularity_turnover'] = _turnover(shift, f_shift)
        assortments = {'store': pids, 'fixed': fixed}
        if other_pids is not None:
            assortments['other_design_store'] = {str(p) for p in other_pids}
        facts['stocked_per_invoice'] = {
            name: {'n_products': len(ps),
                   'store_period': _stocked_per_invoice(params_store, ps),
                   'reference_period': _stocked_per_invoice(params_ref, ps)}
            for name, ps in assortments.items()}
    if pooled is not None:
        facts['replica'] = _replicas(agent_sample, sample, store_clusters,
                                     ref_clusters, pooled, alpha,
                                     sim_samples=sim_samples)
    return facts


def _print_pooled(title, pooled, alpha, n_reps):
    print(f"\n== {title}, {n_reps} replications pooled (alpha = {alpha}) ==")
    for r in pooled:
        stat = 'n/a' if r['statistic'] is None else f"{r['statistic']:.4f}"
        pval = 'n/a' if r['p_value'] is None else f"{r['p_value']:.4g}"
        print(f"  {r['test']:40s} {r['kind'][:12]:12s} stat={stat:>10s} "
              f"p={pval:>10s}  n_obs={r['n_observed']:>7d} "
              f"n_sim={r['n_simulated']:>6d}  {r['decision']}")


def main(argv=None):
    args = parse_args(argv)
    out_dir = make_run_dir(args.out_root, 'validation_gof')
    # The git state at the start of the run, so the sidecar can say which
    # committed code produced the summary and whether it changed mid-run.
    prov = provenance_snapshot()
    print('[gof] calibrating the live store on the current and prior '
          'periods once...', flush=True)
    params = {period: LS.calibrate_live_store(args.retail_path, period)
              for period in ('current', 'prior')}
    periods = {period: LS.period_record(args.retail_path, period)
               for period in ('current', 'prior')}
    params_ref = params[REFERENCE_PERIOD]
    stores = {d: params[STORE_PERIOD[d]] for d in DESIGNS}

    # The held-out replications continue the seed sequence after the
    # in-sample ones, so every run of the invocation has a seed of its own.
    plan = [(d, r, args.seed + k * args.reps + r)
            for k, d in enumerate(DESIGNS) for r in range(args.reps)]
    print(f"[gof] {args.reps} reps per design ({', '.join(DESIGNS)}) of "
          f"warm-up {args.warmup:.0f}s + collect {args.seconds:.0f}s "
          f"simulated, cohort drained (dt {args.dt}s, seeds "
          f"{plan[0][2]}..{plan[-1][2]}) on {args.workers} worker(s)...",
          flush=True)
    t0 = time.perf_counter()
    runs = _map_runs(_design_run, len(plan), args.workers, plan, stores, args)
    wall = time.perf_counter() - t0
    by_design = {d: runs[k * args.reps:(k + 1) * args.reps]
                 for k, d in enumerate(DESIGNS)}

    blocks = {d: _design_block(d, by_design[d], params_ref, args)
              for d in DESIGNS}
    placed = {d: placed_product_ids(ShopItems(by_design[d][0]['shop_items']))
              for d in DESIGNS}
    fixed = set.intersection(*placed.values())
    design = {}
    for d in DESIGNS:
        other = [o for o in DESIGNS if o != d][0]
        wins = by_design[d]
        # Every replication of a design runs the same store.
        anon_lists = wins[0]['anonymous_lists']
        if any(w['anonymous_lists'] != anon_lists for w in wins):
            raise RuntimeError(
                f"{d}: the replications' stores disagree on the "
                "anonymous list shares")
        design[d] = _store_facts(
            d, wins[0]['shop_items'], params_ref, stores[d],
            blocks[d]['pooled'], args.alpha, fixed_pids=fixed,
            other_pids=placed[other],
            sim_samples={'basket': [v for w in wins for v in w['basket_sizes']],
                         'revenue': [v for w in wins
                                     for v in w['customer_revenues']]},
            anonymous_lists=anon_lists)
    design['periods'] = periods
    design['fixed_assortment_products'] = sorted(fixed)

    # Every cohort visit's drawn list and what it bought (review R29), so
    # the list chain can be recomputed from the artifact.
    visits_path = os.path.join(out_dir, VISITS_FILE)
    n_visits = 0
    with open(visits_path, 'w', encoding='utf-8', newline='\n') as f:
        for d in DESIGNS:
            for win in by_design[d]:
                for v in win['visits']:
                    rec = {'design': d, 'rep': win['rep'],
                           'seed': win['seed'], **v}
                    f.write(json.dumps({k: rec[k] for k in VISIT_FIELDS})
                            + '\n')
                    n_visits += 1
    visits_file = {'path': VISITS_FILE, 'n_records': n_visits,
                   'fields': list(VISIT_FIELDS),
                   'order': 'design, replication, exit',
                   'cohort': COHORT}

    csv_path = os.path.join(out_dir, 'results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['design', 'rep', 'seed', 'test', 'kind', 'statistic',
                    'p_value', 'n_observed', 'n_simulated', 'decision'])
        for d in DESIGNS:
            for rep in blocks[d]['per_rep']:
                for r in rep['tests'] + rep['list_chain']:
                    w.writerow([d, rep['rep'], rep['seed'], r['test'],
                                r['kind'],
                                '' if r['statistic'] is None
                                else f"{r['statistic']:.6f}",
                                '' if r['p_value'] is None
                                else f"{r['p_value']:.6g}",
                                r['n_observed'], r['n_simulated'],
                                r['decision']])

    # Top level: the in-sample design, field for field as before. The
    # held-out design sits in a block of the same shape, and 'design' says
    # what each store was built from and tested against.
    ins = blocks['in_sample']
    summary = {'alpha': ins['alpha'], 'design': design,
               'visits_file': visits_file}
    summary.update({k: v for k, v in ins.items() if k != 'alpha'})
    per_rep = summary.pop('per_rep')
    summary['held_out'] = blocks['held_out']
    summary['per_rep'] = per_rep
    # The sidecar goes first: summary.json is the file that marks a run as
    # finished, so it is written last.
    write_sidecar(out_dir, {
        'experiment': 'validation_gof',
        'args': vars(args), 'wall_seconds': wall,
        'csv_path': os.path.relpath(csv_path, out_dir),
        'visits_path': os.path.relpath(visits_path, out_dir),
        'summary': {k: ({kk: vv for kk, vv in v.items() if kk != 'per_rep'}
                        if k == 'held_out' else v)
                    for k, v in summary.items() if k != 'per_rep'}},
        provenance=prov)
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    _print_pooled('in-sample goodness of fit', ins['pooled']
                  + ins['list_chain']['tests'], args.alpha, args.reps)
    print("These are in-sample transfer checks: the parameters under test "
          "were calibrated from the same distributions.")
    ho = design['held_out']
    _print_pooled('held-out goodness of fit', blocks['held_out']['pooled']
                  + blocks['held_out']['list_chain']['tests'], args.alpha,
                  args.reps)
    print(f"Store calibrated on {periods['prior']['first_date']} .. "
          f"{periods['prior']['last_date']} "
          f"({periods['prior']['n_invoices']:,} invoices), tested against "
          f"{periods['current']['first_date']} .. "
          f"{periods['current']['last_date']}: "
          f"{ho['n_reference_invoices_with_placed']:,} of "
          f"{ho['n_reference_invoices']:,} invoices hold one of its "
          f"{ho['n_placed_products']} products "
          f"({ho['n_placed_products_in_reference']} of them sold in that "
          f"period).")
    shift = ', '.join(
        f"{k} stat={v['statistic']:.4f} p={v['p_value']:.4g}"
        for k, v in ho['year_shift'].items() if v)
    print(f"The two periods' own invoices on those products, against each "
          f"other: {shift or 'n/a'}.")
    fshift = ', '.join(
        f"{k} stat={v['statistic']:.4f} p={v['p_value']:.4g}"
        for k, v in ho['year_shift_fixed_assortment'].items() if v)
    print(f"... on the {len(fixed)} products both stores stock: "
          f"{fshift or 'n/a'}.")
    for d in DESIGNS:
        ld = blocks[d]['load']
        print(f"[{d}] door rate {ld['door_rate_per_s'] or 0:.3f}/s "
              f"(offered {ld['offered_rate_per_s']:.3f}/s), balked "
              f"{100 * ld['balked_frac']:.1f}%, cohort "
              f"{ld['cohort_visits']} visits")
        for k, v in design[d]['replica'].items():
            if v:
                print(f"[{d}] {k}: simulator {v['simulator_statistic']:.4f} "
                      f"at replica percentile {v['simulator_percentile']:.2f} "
                      f"({v['simulator_vs_median']} their median); "
                      f"{v['n_replicas']} replicas of {v['n_per_replica']} "
                      f"store-period invoices: median {v['median']:.4f} "
                      f"(95% {v['lo']:.4f}-{v['hi']:.4f}), rejected "
                      f"{100 * v['reject_rate']:.0f}% at alpha {args.alpha}")
    print(f"wall {wall:.0f}s; artifacts in: {out_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
