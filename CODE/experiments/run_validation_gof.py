"""Goodness-of-fit between the live model and the data it was calibrated on.

The Validation tab runs these tests interactively on whatever simulation
the user has just watched. This runner does the same thing headlessly and
reproducibly: it calibrates the UCI store the other live diagnostics use,
runs the agent model under the shared live protocol, and then puts the
window's output distributions against the calibration with the tests in
``dataset_validation`` at alpha = 0.05:

  * basket size    -- distinct items per visit (KS, 2-sample)
  * per-visit revenue                            (KS, 2-sample)
  * category purchase shares -- item purchases per category against the
    dataset's product-invoice touches, both over the products placed in
    the store                                    (chi-square)
  * inter-arrival times                          (KS, informational: it
    asks whether the dataset's own arrivals are Poisson-like, which is the
    family the simulator assumes, and says nothing about the simulator's
    output)

What this is NOT: predictive validation. The simulator's basket, revenue
and category parameters were estimated FROM these same distributions, and
nothing is held out, so agreement shows that the calibrated inputs survive
the pipeline -- adapter, calibration, layout, agent model -- and come back
out in the agents' behaviour. It cannot show the model predicts data it
has not seen. These rows are a transfer check on the pipeline, not
evidence of predictive power.

Every replication is a fixed-step headless run
(``CustomerFlowSimulation.run_headless``: the GUI loop's ``step`` with no
sleeping) for ``--warmup`` + ``--seconds`` simulated seconds under its own
seed, replication r using ``--seed`` + r. The store starts empty, so the
warm-up is deleted: only samples recorded after the warm-up boundary enter
a test. ``--workers`` only decides how many replications run at once;
results are put back in replication order before anything is pooled, so
the summary is the same for any worker count.

The protocol defaults are the nominal load, warm-up and window the other
live diagnostics use; ``run_abm_diagnostics`` records how each value was
chosen.

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

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import dataset_adapters as DA          # noqa: E402
import dataset_calibration as DC       # noqa: E402
from dataset_validation import validate_against_simulation  # noqa: E402
from experiments._common import (build_headless_shop_from_calibration,  # noqa: E402
                                 make_run_dir, write_sidecar)

# Default protocol, shared with run_abm_diagnostics / measure_queueing.
NOMINAL_SPAWN = 0.22
NOMINAL_CAP = 45
WARMUP_S = 600.0
COLLECT_S = 2280.0

ALPHA = 0.05


def _uci_path():
    root = os.path.dirname(_HERE)
    for c in (os.path.join(root, 'DATASETS', 'UCI Online Retail II .xlsx.xlsx'),
              os.path.join(os.path.dirname(root), 'DATASETS',
                           'UCI Online Retail II .xlsx.xlsx')):
        if os.path.exists(c):
            return c
    raise FileNotFoundError('UCI dataset not found under DATASETS/')


def _calibrate():
    df, _ = DA.read_excel_sheets(_uci_path(),
                                 [DA.list_excel_sheets(_uci_path())[-1][0]])
    df = df.sample(n=60000, random_state=0).reset_index(drop=True)
    norm, _ = DA.OnlineRetailIIAdapter().adapt(df)
    return DC.calibrate_transactional(norm, currency='GBP')


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

    Basket sizes and per-visit revenues are appended as agents leave, so the
    window's samples are the tail past the boundary snapshot; zone visits
    and item purchases are cumulative counts, so the window's are their
    increments."""
    A = sim.analytics
    bought, impulse = _item_purchases(A)
    visits = {str(z): int(n) for z, n in dict(A.get('area_visits', {})).items()}
    return {
        'basket_sizes': [float(v) for v in
                         list(A.get('basket_sizes', []))[start['n_baskets']:]],
        'customer_revenues': [
            float(v) for v in
            list(A.get('customer_revenues', []))[start['n_revenues']:]],
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
    shop = build_headless_shop_from_calibration(params,
                                                max_items_per_category=8,
                                                naive=True)
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


def _replication(r, params, args):
    """Replication ``r`` under seed ``args.seed + r``. Top-level so a worker
    process can run it; it returns the window's samples and the parent does
    every test, so pooling and per-replication testing see the same data."""
    seed = args.seed + r
    win = run_once(params, args.spawn, args.cap, args.seconds,
                   warmup=args.warmup, dt=args.dt, seed=seed)
    win['rep'], win['seed'] = r, seed
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
                   help="Base seed; replication r runs under seed + r")
    p.add_argument('--alpha', type=float, default=ALPHA,
                   help="Significance level of every test")
    p.add_argument('--workers', type=int, default=1,
                   help="Processes running replications in parallel")
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


def _test_rows(params, analytics, shop_items, alpha):
    """One dict per goodness-of-fit test of a window's analytics, and the
    report's notes, which say why a test that could not run is missing."""
    res = validate_against_simulation(
        params, WindowSim(analytics, ShopItems(shop_items)), alpha=alpha)
    return [{'test': t.name,
             'kind': t.test,
             'statistic': (float(t.statistic)
                           if np.isfinite(t.statistic) else None),
             'p_value': float(t.p_value) if t.applicable() else None,
             'n_observed': int(t.n_observed),
             'n_simulated': int(t.n_simulated),
             'decision': t.verdict(alpha),
             'note': t.note}
            for t in res.tests], list(res.summary_lines)


def _decision_counts(per_rep, name):
    """PASS / FAIL / N/A counts of one test over the replications."""
    counts = {d: int(sum(1 for rep in per_rep for r in rep['tests']
                         if r['test'] == name and r['decision'] == d))
              for d in ('PASS', 'FAIL')}
    counts['N/A'] = len(per_rep) - counts['PASS'] - counts['FAIL']
    return counts


def main(argv=None):
    args = parse_args(argv)
    out_dir = make_run_dir(args.out_root, 'validation_gof')
    print('[gof] calibrating UCI shop once...', flush=True)
    params = _calibrate()

    print(f"[gof] {args.reps} reps of warm-up {args.warmup:.0f}s + collect "
          f"{args.seconds:.0f}s simulated (dt {args.dt}s, seeds "
          f"{args.seed}..{args.seed + args.reps - 1}) on {args.workers} "
          f"worker(s)...", flush=True)
    t0 = time.perf_counter()
    reps = _map_runs(_replication, args.reps, args.workers, params, args)
    wall = time.perf_counter() - t0

    # Per replication first: independent windows, so the spread of the
    # decisions says how much one window's sample size is doing.
    per_rep = []
    for win in reps:
        rows, _ = _test_rows(params, win, win['shop_items'], args.alpha)
        per_rep.append({'rep': win['rep'], 'seed': win['seed'],
                        'load': win['load'], 'tests': rows})
        print(f"  rep {win['rep'] + 1}/{args.reps} (seed {win['seed']}): "
              + ', '.join(f"{r['test']}={r['decision']}" for r in rows),
              flush=True)

    # Pooled: the replications share one calibrated store and differ only in
    # their random streams, so their samples are draws from the same output
    # distribution and can be tested together. The store is built from the
    # calibration alone, so every replication must have laid out the same
    # items; per-item counts are only summed under that condition.
    shop_items = reps[0]['shop_items']
    if any(w['shop_items'] != shop_items for w in reps[1:]):
        raise RuntimeError("replications laid out different stores; their "
                           "item purchases cannot be pooled")

    def _summed(counts):
        out = {}
        for c in counts:
            for k, n in c.items():
                out[k] = out.get(k, 0) + int(n)
        return out

    purchases = _summed({item: d['purchases'] for item, d
                         in w['item_conversion_rates'].items()} for w in reps)
    pooled_analytics = {
        'basket_sizes': [v for w in reps for v in w['basket_sizes']],
        'customer_revenues': [v for w in reps for v in w['customer_revenues']],
        'item_conversion_rates': {item: {'purchases': n}
                                  for item, n in purchases.items()},
        'impulse_item_sales': _summed(w['impulse_item_sales'] for w in reps),
        'area_visits': _summed(w['area_visits'] for w in reps),
    }
    pooled, pooled_notes = _test_rows(params, pooled_analytics, shop_items,
                                      args.alpha)

    csv_path = os.path.join(out_dir, 'results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['rep', 'seed', 'test', 'kind', 'statistic', 'p_value',
                    'n_observed', 'n_simulated', 'decision'])
        for rep in per_rep:
            for r in rep['tests']:
                w.writerow([rep['rep'], rep['seed'], r['test'], r['kind'],
                            '' if r['statistic'] is None
                            else f"{r['statistic']:.6f}",
                            '' if r['p_value'] is None
                            else f"{r['p_value']:.6g}",
                            r['n_observed'], r['n_simulated'], r['decision']])

    names = [r['test'] for r in pooled]
    summary = {
        'alpha': args.alpha,
        # These tests compare the simulator against the distributions its
        # own parameters were estimated from; nothing is held out.
        'design': 'in_sample_pipeline_transfer',
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
        'n_reps': int(args.reps),
        'load': {'arrivals': int(sum(w['load']['arrivals'] for w in reps)),
                 'balked': int(sum(w['load']['balked'] for w in reps))},
        'protocol': {'mode': 'fixed_step', 'dt': args.dt,
                     'seed_base': args.seed,
                     'seeds': [w['seed'] for w in reps],
                     'reps': args.reps, 'warmup_s': args.warmup,
                     'collect_s': args.seconds, 'spawn': args.spawn,
                     'cap': args.cap, 'workers': args.workers,
                     'framing': 'terminating'},
        'per_rep': per_rep,
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    write_sidecar(out_dir, {'experiment': 'validation_gof',
                            'args': vars(args), 'wall_seconds': wall,
                            'csv_path': os.path.relpath(csv_path, out_dir),
                            'summary': {k: v for k, v in summary.items()
                                        if k != 'per_rep'}})

    print(f"\n== goodness of fit, {args.reps} replications pooled "
          f"(alpha = {args.alpha}) ==")
    for r in pooled:
        stat = 'n/a' if r['statistic'] is None else f"{r['statistic']:.4f}"
        pval = 'n/a' if r['p_value'] is None else f"{r['p_value']:.4g}"
        print(f"  {r['test']:34s} {r['kind']:12s} stat={stat:>10s} "
              f"p={pval:>10s}  n_obs={r['n_observed']:>7d} "
              f"n_sim={r['n_simulated']:>6d}  {r['decision']}")
    print("These are in-sample transfer checks: the parameters under test "
          "were calibrated from the same distributions.")
    print(f"wall {wall:.0f}s; artifacts in: {out_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
