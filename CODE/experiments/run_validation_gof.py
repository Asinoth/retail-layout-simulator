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
    item-level chi-square p beside it, labelled naive)
  * inter-arrival times                          (KS, informational: it
    asks whether the dataset's own arrivals are Poisson-like, which is the
    family the simulator assumes, and says nothing about the simulator's
    output)

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
and ``replica``: those tests on replicas that draw as many of the store
period's invoices as the simulator produced visits -- what a model that
transmitted its calibration period perfectly would score at the
simulator's own sample size. The year shift compares two whole periods,
and a chi-square grows with the sample, so only the replicas put the
simulator's statistics against a yardstick on the same footing.
``results.csv`` has one row per design, replication and test.

Every replication is a fixed-step headless run
(``CustomerFlowSimulation.run_headless``: the GUI loop's ``step`` with no
sleeping) for ``--warmup`` + ``--seconds`` simulated seconds under its own
seed: in-sample replication r under ``--seed`` + r, held-out replication r
under ``--seed`` + ``--reps`` + r, so every run has its own seed. The store
starts empty, so the warm-up is deleted: only samples recorded after the
warm-up boundary enter a test. ``--workers`` only decides how many
replications run at once; results are put back in design order before
anything is pooled, so the summary is the same for any worker count.

The protocol defaults are the nominal load, warm-up and window the other
live diagnostics use; ``run_abm_diagnostics`` records how each value was
chosen. Both designs run at them; the held-out store's own hour-of-day
profile sets the multiplier its arrivals run at.

    python -m experiments.run_validation_gof
    python -m experiments.run_validation_gof --reps 10 --workers 10
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.stats import ks_2samp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from dataset_calibration import placed_invoice_sample  # noqa: E402
from dataset_validation import (  # noqa: E402
    cluster_permutation_chi2, invoice_category_clusters,
    merge_unreferenced_categories, placed_product_categories,
    placed_product_ids, validate_against_simulation)
from experiments import _live_store as LS  # noqa: E402
from experiments._common import make_run_dir, write_sidecar  # noqa: E402

# Default protocol, shared with run_abm_diagnostics / measure_queueing.
NOMINAL_SPAWN = 0.27
NOMINAL_CAP = 45
WARMUP_S = 1020.0
COLLECT_S = 1860.0

ALPHA = 0.05

# Size-matched replicas of each store's calibration period (``_replicas``):
# how many, and the seed their invoice draws come from.
N_REPLICAS = 200
REPLICA_SEED = 20260925

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


def _arrival_counts(sim):
    """Arrivals admitted and arrivals that balked at the customer cap."""
    A = sim.analytics
    return int(A.get('total_customers', 0)), int(A.get('balked_arrivals', 0))


def _item_purchases(A):
    """Cumulative purchases per item: items shopped for and taken to the
    till, and the separate count of items picked up at the till."""
    bought = {str(item): int((data or {}).get('purchases', 0) or 0)
              for item, data in dict(A.get('item_conversion_rates', {})).items()}
    impulse = {str(item): int(n or 0)
               for item, n in dict(A.get('impulse_item_sales', {})).items()}
    return bought, impulse


def _increments(now, before):
    """Positive per-key increase of a cumulative counter since ``before``."""
    out = {}
    for key, n in now.items():
        delta = int(n) - int(before.get(key, 0))
        if delta > 0:
            out[key] = delta
    return out


def _window_analytics(sim, start):
    """The post-warm-up part of the samples the tests read.

    Basket sizes, per-visit revenues and per-visit purchases are appended
    as agents leave, so the window's samples are the tail past the boundary
    snapshot; zone visits and item purchases are cumulative counts, so the
    window's are their increments. The per-visit purchases are always
    written, empty or not: the simulator records them for every paying
    visit, and a missing list would tell the category test that the record
    predates them."""
    A = sim.analytics
    bought, impulse = _item_purchases(A)
    visits = {str(z): int(n) for z, n in dict(A.get('area_visits', {})).items()}
    return {
        'basket_sizes': [float(v) for v in
                         list(A.get('basket_sizes', []))[start['n_baskets']:]],
        'customer_revenues': [
            float(v) for v in
            list(A.get('customer_revenues', []))[start['n_revenues']:]],
        'visit_purchases': [
            [str(item) for item in basket] for basket in
            list(A.get('visit_purchases', []))[start['n_visit_purchases']:]],
        'item_conversion_rates': {
            item: {'purchases': n}
            for item, n in _increments(bought, start['purchases']).items()},
        'impulse_item_sales': _increments(impulse, start['impulse_sales']),
        'area_visits': _increments(visits, start['area_visits']),
    }


def run_once(params, spawn, cap, seconds, warmup=WARMUP_S, dt=0.04, seed=None):
    """One replication, fixed-step headless for ``warmup + seconds``
    simulated seconds, returning the window's output samples, its arrival
    load and the store's items."""
    shop = LS.build_live_store(params)
    sim = shop.customer_simulation
    sim.max_customers = cap
    sim.spawn_rate = spawn

    start = {}

    def _mark_warmup(s):
        # The first call lands on the first tick at or past the warm-up;
        # later periodic calls fall inside the collection window.
        if start:
            return
        A = s.analytics
        start['t'] = float(s.sim_time)
        start['n_baskets'] = len(A.get('basket_sizes', []))
        start['n_revenues'] = len(A.get('customer_revenues', []))
        start['n_visit_purchases'] = len(A.get('visit_purchases', []))
        start['area_visits'] = {str(z): int(n) for z, n
                                in dict(A.get('area_visits', {})).items()}
        start['purchases'], start['impulse_sales'] = _item_purchases(A)
        start['arrivals'] = _arrival_counts(s)

    sim.run_headless(warmup + seconds, dt=dt, seed=seed,
                     callback=_mark_warmup, callback_every_s=warmup)
    # step() counts and survives per-agent exceptions so a GUI session
    # keeps running; a measurement run must not have had any.
    errs = dict(getattr(sim, 'suppressed_errors', {}))
    if errs:
        raise RuntimeError(
            f"live loop suppressed exceptions during measurement: {errs}")

    win = _window_analytics(sim, start)
    admitted, balked = _arrival_counts(sim)
    win['load'] = {'arrivals': (admitted + balked) - sum(start['arrivals']),
                   'balked': balked - start['arrivals'][1]}
    win['shop_items'] = _shop_items(shop)
    return win


def _design_run(j, plan, stores, args):
    """Run ``j`` of the plan: one replication of one design, on the store
    calibrated for that design, under the seed the plan gives it. Top-level
    so a worker process can run it; it returns the window's samples and the
    parent does every test, so pooling and per-replication testing see the
    same data."""
    design, r, seed = plan[j]
    win = run_once(stores[design], args.spawn, args.cap, args.seconds,
                   warmup=args.warmup, dt=args.dt, seed=seed)
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
    # 0.04 s is the threaded GUI loop's 25 fps ceiling, so agents move
    # with the per-tick resolution they have in the GUI.
    p.add_argument('--dt', type=float, default=0.04,
                   help="Fixed tick, simulated s")
    p.add_argument('--seed', type=int, default=4000,
                   help="Base seed; in-sample replication r runs under "
                        "seed + r, held-out replication r under "
                        "seed + reps + r")
    p.add_argument('--alpha', type=float, default=ALPHA,
                   help="Significance level of every test")
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


def _test_rows(params, analytics, shop_items, alpha):
    """One dict per goodness-of-fit test of a window's analytics, and the
    report's notes, which say why a test that could not run is missing.

    A test's extra record -- for the category test the naive item-level p,
    the permutation count and seed, the categories compared -- goes into
    its row beside the common fields."""
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


def _design_block(design, wins, params_ref, args):
    """Everything one design reports: per-replication and pooled tests of
    its windows against ``params_ref``, the decision split, the arrival
    load and the protocol with this design's seeds, in the shape the
    summary's top level has always had."""
    # Per replication first: independent windows, so the spread of the
    # decisions says how much one window's sample size is doing.
    per_rep = []
    for win in wins:
        rows, _ = _test_rows(params_ref, win, win['shop_items'], args.alpha)
        per_rep.append({'rep': win['rep'], 'seed': win['seed'],
                        'load': win['load'], 'tests': rows})
        print(f"  [{design}] rep {win['rep'] + 1}/{args.reps} "
              f"(seed {win['seed']}): "
              + ', '.join(f"{r['test']}={r['decision']}" for r in rows),
              flush=True)

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
        'n_reps': int(len(wins)),
        'load': {'arrivals': int(sum(w['load']['arrivals'] for w in wins)),
                 'balked': int(sum(w['load']['balked'] for w in wins))},
        'protocol': {'mode': 'fixed_step', 'dt': args.dt,
                     'seed_base': args.seed,
                     'seeds': [w['seed'] for w in wins],
                     'reps': args.reps, 'warmup_s': args.warmup,
                     'collect_s': args.seconds, 'spawn': args.spawn,
                     'cap': args.cap, 'workers': args.workers,
                     'framing': 'terminating'},
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
                         'n_store': int(a.size), 'n_reference': int(b.size)}
        else:
            out[name] = None
    r = cluster_permutation_chi2(store_clusters, ref_clusters)
    out['category'] = ({
        'statistic': float(r['statistic']),
        'p_value': float(r['p_value']),
        'naive_p_value': _json_value(r['naive_p_value']),
        'n_store': int(r['n_sim_clusters']),
        'n_reference': int(r['n_ref_clusters']),
        'n_categories': int(r['n_categories']),
        'n_permutations': int(r['n_permutations']),
        'permutation_seed': int(r['permutation_seed'])}
        if np.isfinite(r['p_value']) else None)
    return out


def _replica_summary(stats, pvals, sim_stat, alpha, n, seed):
    stats = np.asarray(stats, dtype=float)
    pvals = np.asarray(pvals, dtype=float)
    out = {'n_replicas': int(stats.size), 'n_per_replica': int(n),
           'seed': int(seed),
           'median': float(np.median(stats)),
           'lo': float(np.percentile(stats, 2.5)),
           'hi': float(np.percentile(stats, 97.5)),
           'reject_rate': float(np.mean(pvals < alpha)),
           'simulator_statistic': None, 'simulator_percentile': None}
    if sim_stat is not None and np.isfinite(sim_stat):
        out['simulator_statistic'] = float(sim_stat)
        out['simulator_percentile'] = float(np.mean(stats <= sim_stat))
    return out


def _replicas(store_sample, ref_sample, store_clusters, ref_clusters,
              pooled, alpha, n_rep=N_REPLICAS, seed=REPLICA_SEED):
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
    transfer of the calibration period), and where the simulator's own
    statistic falls among them.

    In the in-sample design the store period is the reference, so the
    replicas are the test's null and their rejection rate sits near
    ``alpha``. In the held-out design they carry the year-to-year drift
    and nothing else."""
    sim = {}
    for row in pooled:
        name = str(row['test']).lower()
        key = next((k for k in ('basket', 'revenue', 'categor') if k in name),
                   None)
        if key and row.get('statistic') is not None:
            sim[key] = (float(row['statistic']), int(row['n_simulated']))
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
        stats, pvals = [], []
        for _ in range(n_rep):
            r = ks_2samp(a[rng.integers(0, a.size, n)], b)
            stats.append(r.statistic)
            pvals.append(r.pvalue)
        out[name] = _replica_summary(stats, pvals, stat, alpha, n, seed)
    rows = np.asarray(store_clusters, dtype=float)
    rows = rows[rows.sum(axis=1) > 0] if rows.size else rows
    if 'categor' in sim and len(rows):
        stat, n = sim['categor']
        rng = np.random.default_rng([seed, 2])
        stats, pvals = [], []
        for _ in range(n_rep):
            r = cluster_permutation_chi2(rows[rng.integers(0, len(rows), n)],
                                         ref_clusters)
            stats.append(r['statistic'])
            pvals.append(r['p_value'])
        out['category'] = _replica_summary(stats, pvals, stat, alpha, n, seed)
    else:
        out['category'] = None
    return out


def _store_facts(design, shop_items, params_ref, params_store, pooled=None,
                 alpha=ALPHA):
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

    Given the design's pooled test rows, also ``replica``: the same tests
    on size-matched replicas of the store period (``_replicas``)."""
    shop = ShopItems(shop_items)
    pids = placed_product_ids(shop)
    sold = {str(p) for p in (params_ref.item_visit_counts or {})}
    sample = placed_invoice_sample(params_ref, pids)
    agent_sample = placed_invoice_sample(params_store, pids)
    product_category = placed_product_categories(params_ref, shop)
    categories = sorted(set(product_category.values()))
    ref_clusters, store_clusters, _, _ = merge_unreferenced_categories(
        invoice_category_clusters(params_ref, product_category, categories),
        invoice_category_clusters(params_store, product_category, categories),
        categories)
    facts = {'label': DESIGN_LABELS[design],
             'store_period': STORE_PERIOD[design],
             'reference_period': REFERENCE_PERIOD,
             'n_placed_products': len(pids),
             'n_placed_products_in_reference': len(pids & sold),
             'n_reference_invoices': int(sample['n_invoices']),
             'n_reference_invoices_with_placed': int(sample['n_with_placed']),
             'list_length_agents': _sample_stats(agent_sample['sizes']),
             'list_length_reference': _sample_stats(sample['sizes']),
             'year_shift': _year_shift(agent_sample, sample, store_clusters,
                                       ref_clusters)}
    if pooled is not None:
        facts['replica'] = _replicas(agent_sample, sample, store_clusters,
                                     ref_clusters, pooled, alpha)
    return facts


def _print_pooled(title, pooled, alpha, n_reps):
    print(f"\n== {title}, {n_reps} replications pooled (alpha = {alpha}) ==")
    for r in pooled:
        stat = 'n/a' if r['statistic'] is None else f"{r['statistic']:.4f}"
        pval = 'n/a' if r['p_value'] is None else f"{r['p_value']:.4g}"
        print(f"  {r['test']:34s} {r['kind']:12s} stat={stat:>10s} "
              f"p={pval:>10s}  n_obs={r['n_observed']:>7d} "
              f"n_sim={r['n_simulated']:>6d}  {r['decision']}")


def main(argv=None):
    args = parse_args(argv)
    out_dir = make_run_dir(args.out_root, 'validation_gof')
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
          f"simulated (dt {args.dt}s, seeds {plan[0][2]}..{plan[-1][2]}) on "
          f"{args.workers} worker(s)...", flush=True)
    t0 = time.perf_counter()
    runs = _map_runs(_design_run, len(plan), args.workers, plan, stores, args)
    wall = time.perf_counter() - t0
    by_design = {d: runs[k * args.reps:(k + 1) * args.reps]
                 for k, d in enumerate(DESIGNS)}

    blocks = {d: _design_block(d, by_design[d], params_ref, args)
              for d in DESIGNS}
    design = {d: _store_facts(d, by_design[d][0]['shop_items'], params_ref,
                              stores[d], blocks[d]['pooled'], args.alpha)
              for d in DESIGNS}
    design['periods'] = periods

    csv_path = os.path.join(out_dir, 'results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['design', 'rep', 'seed', 'test', 'kind', 'statistic',
                    'p_value', 'n_observed', 'n_simulated', 'decision'])
        for d in DESIGNS:
            for rep in blocks[d]['per_rep']:
                for r in rep['tests']:
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
    summary = {'alpha': ins['alpha'], 'design': design}
    summary.update({k: v for k, v in ins.items() if k != 'alpha'})
    per_rep = summary.pop('per_rep')
    summary['held_out'] = blocks['held_out']
    summary['per_rep'] = per_rep
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    write_sidecar(out_dir, {
        'experiment': 'validation_gof',
        'args': vars(args), 'wall_seconds': wall,
        'csv_path': os.path.relpath(csv_path, out_dir),
        'summary': {k: ({kk: vv for kk, vv in v.items() if kk != 'per_rep'}
                        if k == 'held_out' else v)
                    for k, v in summary.items() if k != 'per_rep'}})

    _print_pooled('in-sample goodness of fit', ins['pooled'], args.alpha,
                  args.reps)
    print("These are in-sample transfer checks: the parameters under test "
          "were calibrated from the same distributions.")
    ho = design['held_out']
    _print_pooled('held-out goodness of fit', blocks['held_out']['pooled'],
                  args.alpha, args.reps)
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
    for d in DESIGNS:
        for k, v in design[d]['replica'].items():
            if v:
                print(f"[{d}] {k}: simulator {v['simulator_statistic']:.4f} "
                      f"at replica percentile {v['simulator_percentile']:.2f}; "
                      f"{v['n_replicas']} replicas of {v['n_per_replica']} "
                      f"store-period invoices: median {v['median']:.4f} "
                      f"(95% {v['lo']:.4f}-{v['hi']:.4f}), rejected "
                      f"{100 * v['reject_rate']:.0f}% at alpha {args.alpha}")
    print(f"wall {wall:.0f}s; artifacts in: {out_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
