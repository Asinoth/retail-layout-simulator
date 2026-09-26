"""Held-out transfer of the optimized layouts (review R12).

The simulator's distributions are tested on a held-out year
(``run_validation_gof``), but Figure C searches and scores on one pooled
calibration, so nothing tests whether a layout optimized on one year keeps
its lift under the next year's data. This runner does: it searches on the
PRIOR period only, then scores what it found under the CURRENT period.

Design:
  * Periods: ``experiments._live_store``'s -- 'prior' is the 2009-2010
    sheet cut to whole invoices dated before the current period's first
    day, 'current' the whole 2010-2011 sheet; no invoice or calendar day is
    in both. Each is calibrated by ``calibrate_transactional`` with Figure
    C's options (assumed conversion, anonymous invoices in or out).
  * Fixtures: Figure C's store (``run_real_data_example.build_store``, its
    cap on items per category, naive as-built layout) built from the PRIOR
    calibration -- the store a retailer would have laid out from last
    year's data. The current calibration is re-keyed onto the same
    fixtures (``experiments._rekey``): its per-item statistics and base
    parameters, anchored at the same as-built layout re-scored under them.
    A fixture whose product did not sell in the current period stays on
    the floor with no demand; the summary lists them, and counts the
    current period's own top products the fixtures do not carry.
  * Searches: for each search seed s (``--ga-seed`` + k, k <
    ``--n-search-seeds``), Figure C's three equal-budget searches (GA,
    random search, simulated annealing; ``run_real_data_example.
    run_searches``, all from the as-built layout) run twice: on the prior
    store (the layout that would be deployed) and on the current store
    with the same fixtures (the hindsight optimum, what the search finds
    knowing the current data).
  * Scoring, per method and seed, every lift against the as-built layout:
      promised     -- prior layout on the prior calibration;
      transferred  -- prior layout on the current calibration;
      hindsight    -- current layout on the current calibration;
    in closed form (``experiments.closed_form.expected_revenue``) and, on
    the current calibration, by Monte Carlo under Figure C's held-out
    evaluation seeds (``EVAL_SEED_BASE + r``), paired across layouts.
    ``transfer_gap`` is hindsight minus transferred: how much of the lift
    achievable on the current data a layout found a year early misses.
    ``transfer_ratio`` is transferred / hindsight, the share it still
    earns, defined only where the hindsight lift is positive (null
    otherwise: a ratio to a zero or negative lift says nothing).
  * Across seeds: the mean of each per-seed value with a Student-t interval
    over the seeds, its range and the number of seeds with a positive
    transferred lift; the transfer gap as a PAIRED comparison -- a
    t-interval over the seeds of each seed's hindsight minus transferred
    lift (GBP, percentage points, and Monte Carlo GBP), with the number of
    seeds whose transferred lift falls short of the hindsight one -- which
    is the headline test of whether transfer loses lift; the ratio of the
    mean transferred to the mean hindsight lift; and the GA's transferred
    lift against each comparator's, per seed and across seeds.

``--workers`` spreads the (period, seed) searches over processes; each
rebuilds its store, checks it against the parent's fingerprint and seeds
every generator it draws from, and the scoring runs in the parent in a
fixed order, so the output does not depend on the worker count.

Outputs (under ``--out-root``, one ``heldout_transfer_<time>`` directory;
``noanon_heldout_transfer_<time>`` for an ``--exclude-anonymous``
sensitivity run, so it cannot stand in for the headline):
results.csv (one row per method and seed), mc_replicates.csv (every
Monte Carlo evaluation), sidecar.json (provenance: both periods' records
with the workbook's SHA-256, base parameters of both stores), summary.json
(written last).

Smoke (tiny budget; the periods' calibration dominates):
    python -m experiments.run_heldout_transfer --n-search-seeds 2 ^
        --mc-iters 100 --mc-days 7 --n-gens 2 --pop-size 6 ^
        --n-mc-replicates 3 --out-root <scratch folder>

Paper-grade (Figure C's budget, five search seeds; about ten Figure C runs
of compute):
    python -m experiments.run_heldout_transfer --workers 4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from dataset_adapters import OnlineRetailIIAdapter, READER_VERSION  # noqa: E402
from dataset_calibration import CalibratedParams, calibrate_transactional  # noqa: E402
from dataset_layout import _items_per_category  # noqa: E402
from dataset_provenance import stamp as stamp_provenance  # noqa: E402
from experiments import _live_store as LS  # noqa: E402
from experiments import _rekey as RK  # noqa: E402
from experiments._common import (GA_N_FINAL_SEEDS,  # noqa: E402
                                 aisle_rule_summary, base_params_record,
                                 bootstrap_ci, make_run_dir,
                                 provenance_snapshot, repair_stats,
                                 write_sidecar)
from experiments.run_real_data_example import (  # noqa: E402
    EVAL_SEED_BASE, GA_ELITE_FRAC, GA_MUT_RATE, SA_STEP_FRAC, add_arguments,
    build_store, check_reported, check_search_arguments, experiment_name,
    reported_layouts, resolve_workbook, run_dir_prefix, run_searches,
    score_layouts, search_operators)
from experiments.run_real_data_seeds import (seed_plan, seed_plan_error,  # noqa: E402
                                             store_fingerprint, t_interval)

#: Search seeds unless told otherwise.
DEFAULT_N_SEARCH_SEEDS = 5
#: Most worker processes the runner accepts (the workstation's thermal
#: limit; the output does not depend on the count).
MAX_WORKERS = 4
#: Two-sided level of the across-seed intervals.
CI_LEVEL = 0.95

PERIODS = ('prior', 'current')
METHODS = ('GA', 'random_search', 'simulated_annealing')
#: ``run_searches``' key for each method's layout.
LAYOUT_KEY = {'GA': 'optimized', 'random_search': 'rs',
              'simulated_annealing': 'sa'}
LIFTS = ('promised', 'transferred', 'hindsight')


# --- Arguments ---------------------------------------------------------------

def _figure_c_defaults() -> Dict[str, Any]:
    p = argparse.ArgumentParser(add_help=False)
    add_arguments(p)
    return vars(p.parse_args([]))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Figure C's store and search options, defaults taken from its parser
    (``--sheets`` excepted: the periods are ``_live_store``'s), plus the
    search seeds and the worker count."""
    d = _figure_c_defaults()
    p = argparse.ArgumentParser(
        description="Search on the prior period, score under the current "
                    "one: does the optimized layout keep its lift?")
    p.add_argument('--retail-path', type=str, default=None,
                   help="The UCI Online Retail II workbook; default: found "
                        "by dataset_paths.uci_workbook()")
    p.add_argument('--max-items-per-category', type=int,
                   default=d['max_items_per_category'])
    p.add_argument('--assumed-conversion', type=float,
                   default=d['assumed_conversion'])
    p.add_argument('--exclude-anonymous', action='store_true',
                   help="Calibrate both periods without the anonymous "
                        "invoices (sensitivity run)")
    p.add_argument('--mc-iters', type=int, default=d['mc_iters'])
    p.add_argument('--mc-days', type=int, default=d['mc_days'])
    p.add_argument('--n-gens', type=int, default=d['n_gens'])
    p.add_argument('--pop-size', type=int, default=d['pop_size'])
    p.add_argument('--n-mc-replicates', type=int,
                   default=d['n_mc_replicates'],
                   help="Held-out Monte Carlo evaluation seeds per layout")
    p.add_argument('--ga-seed', type=int, default=d['ga_seed'],
                   help="First search seed; seed s seeds the three searches "
                        "as Figure C seeds them from --ga-seed")
    p.add_argument('--sa-initial-accept', type=float,
                   default=d['sa_initial_accept'])
    p.add_argument('--n-search-seeds', type=int,
                   default=DEFAULT_N_SEARCH_SEEDS)
    p.add_argument('--workers', type=int, default=1,
                   help=f"Processes the searches are spread over (at most "
                        f"{MAX_WORKERS}); the output is the same for any value")
    p.add_argument('--out-root', type=str, default=d['out_root'])
    args = p.parse_args(argv)
    if args.n_search_seeds < 2:
        p.error("--n-search-seeds must be >= 2 (an across-seed interval)")
    if not 1 <= args.workers <= MAX_WORKERS:
        p.error(f"--workers must be in [1, {MAX_WORKERS}]")
    if args.n_mc_replicates < 2:
        p.error("--n-mc-replicates must be >= 2")
    check_search_arguments(p, args)
    bad = seed_plan_error(args.ga_seed, args.n_search_seeds, args.n_gens,
                          args.pop_size)
    if bad:
        p.error(bad)
    resolve_workbook(p, args)
    return args


# --- Stores and searches -------------------------------------------------------

@dataclass(frozen=True)
class TransferDesign:
    """What every search task needs besides the two calibrations."""
    max_items_per_category: int
    n_gens: int
    pop_size: int
    mc_iters: int
    mc_days: int
    sa_initial_accept: float
    fingerprints: Tuple[Tuple[str, str], ...]     # (period, fingerprint)


def period_store(params: Mapping[str, CalibratedParams], period: str,
                 max_items_per_category: int):
    """The prior store (fixtures and statistics from the prior period), or
    the current calibration on the prior store's fixtures."""
    if period == 'prior':
        return build_store(params['prior'], max_items_per_category,
                           verbose=False)
    return RK.store_on_fixtures(params['prior'], params['current'],
                                max_items_per_category)


def _layout_json(layout: Mapping[str, Tuple[float, float]]) -> Dict[str, List[float]]:
    return {k: [float(v[0]), float(v[1])] for k, v in layout.items()}


def run_search_task(params: Mapping[str, CalibratedParams], period: str,
                    seed: int, design: TransferDesign) -> Dict[str, Any]:
    """Figure C's three searches on ``period``'s store at ``seed``: the
    layouts and the search records. A function of (period, seed) alone."""
    t0 = time.perf_counter()
    store = period_store(params, period, design.max_items_per_category)
    fp = store_fingerprint(store)
    want = dict(design.fingerprints)[period]
    if fp != want:
        raise RuntimeError(f"{period} store rebuilt for seed {seed} differs "
                           f"from the parent's ({fp[:12]} != {want[:12]})")
    s = run_searches(store, seed=seed, n_gens=design.n_gens,
                     pop_size=design.pop_size, mc_iters=design.mc_iters,
                     mc_days=design.mc_days,
                     sa_initial_accept=design.sa_initial_accept,
                     verbose=False)
    # Every layout the run reports keeps the floor-plan invariants; a
    # layout that breaks one stops the run (LayoutInvariantError). Both
    # periods' stores share the prior store's fixtures and floor plan.
    invariants = check_reported(store, reported_layouts(store, s),
                                f"{period} seed {seed} ")
    print(f"[transfer] {period} searches, seed {seed}: "
          f"{time.perf_counter() - t0:.1f}s", flush=True)
    return {'period': period, 'seed': int(seed),
            'layouts': {m: _layout_json(s[LAYOUT_KEY[m]]) for m in METHODS},
            'evaluation_counts': dict(s['eval_counts']),
            'final_evaluation_counts': dict(s['final_eval_counts']),
            'search_start': dict(s['starts']),
            'budget': int(s['budget']),
            'sa_T0': s['sa_stats'].get('sa_T0'),
            'invariants': invariants,
            'repair_stats': repair_stats(store.shop),
            'wall_seconds': time.perf_counter() - t0}


_WORKER: Dict[str, Any] = {}


def _init_worker(params: Mapping[str, CalibratedParams]) -> None:
    _WORKER['params'] = params


def _task(period: str, seed: int, design: TransferDesign) -> Dict[str, Any]:
    return run_search_task(_WORKER['params'], period, seed, design)


def map_searches(params: Mapping[str, CalibratedParams], seeds: Sequence[int],
                 design: TransferDesign, workers: int) -> List[Dict[str, Any]]:
    """Every (period, seed) search task, in (period, seed) order."""
    tasks = [(p, s) for p in PERIODS for s in seeds]
    if workers <= 1:
        return [run_search_task(params, p, s, design) for p, s in tasks]
    done: Dict[Tuple[str, int], Dict[str, Any]] = {}
    pool = ProcessPoolExecutor(max_workers=min(workers, len(tasks)),
                               initializer=_init_worker, initargs=(params,))
    try:
        futures = {pool.submit(_task, p, s, design): (p, s) for p, s in tasks}
        for fut in as_completed(futures):
            done[futures[fut]] = fut.result()
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return [done[t] for t in tasks]


def check_search_results(results: Sequence[Mapping[str, Any]],
                         budget: int) -> None:
    """Refuse searches that are not the comparison reported: unequal search
    budgets, or a start other than the as-built layout."""
    for r in results:
        if any(int(r['evaluation_counts'][m]) != budget for m in METHODS):
            raise AssertionError(f"{r['period']} seed {r['seed']}: budgets "
                                 f"{r['evaluation_counts']} != {budget}")
        if any(r['search_start'][m] != 'asbuilt' for m in METHODS):
            raise AssertionError(f"{r['period']} seed {r['seed']}: starts "
                                 f"{r['search_start']}")


# --- Scoring -------------------------------------------------------------------

def _as_layout(obj: Mapping[str, Sequence[float]]) -> Dict[str, Tuple[float, float]]:
    return {k: (float(v[0]), float(v[1])) for k, v in obj.items()}


def score_transfer(stores: Mapping[str, Any],
                   results: Sequence[Mapping[str, Any]],
                   seeds: Sequence[int], mc_days: int, mc_iters: int,
                   n_replicates: int) -> Dict[str, Any]:
    """Closed-form and held-out Monte Carlo scores of every layout.

    Returns ``{'baseline': {...}, 'per_seed': [...], 'mc_rows': [...]}``;
    each per-seed record holds, per method, the promised, transferred and
    hindsight lifts (GBP and % of the as-built revenue of the period they
    are scored in), the Monte Carlo lifts on the current period with
    bootstrap intervals, and the transfer ratio."""
    prior, current = stores['prior'], stores['current']
    base = prior.baseline_layout
    base_cf = {'prior': RK.layout_value(prior, base, mc_days),
               'current': RK.layout_value(current, base, mc_days)}
    found = {(r['period'], r['seed']): r['layouts'] for r in results}

    # Monte Carlo on the current period: the as-built layout and every
    # transferred and hindsight layout, all under the same held-out seeds.
    mc_layouts = [('baseline', base)]
    for s in seeds:
        for m in METHODS:
            mc_layouts.append((f'transferred|{m}|{s}',
                               _as_layout(found[('prior', s)][m])))
            mc_layouts.append((f'hindsight|{m}|{s}',
                               _as_layout(found[('current', s)][m])))
    revs = score_layouts(current, mc_layouts, n_replicates=n_replicates,
                         mc_iters=mc_iters, mc_days=mc_days)
    mc_base = revs['baseline']

    per_seed = []
    for s in seeds:
        rec: Dict[str, Any] = {'seed': int(s), 'methods': {}}
        for m in METHODS:
            lay_t = _as_layout(found[('prior', s)][m])
            lay_h = _as_layout(found[('current', s)][m])
            lifts = {
                'promised': RK.layout_value(prior, lay_t, mc_days)
                - base_cf['prior'],
                'transferred': RK.layout_value(current, lay_t, mc_days)
                - base_cf['current'],
                'hindsight': RK.layout_value(current, lay_h, mc_days)
                - base_cf['current'],
            }
            out = {k: {'lift': v,
                       'pct': v / base_cf['prior' if k == 'promised'
                                          else 'current'] * 100.0}
                   for k, v in lifts.items()}
            for k in ('transferred', 'hindsight'):
                d = revs[f'{k}|{m}|{s}'] - mc_base
                mean, lo, hi = bootstrap_ci(d, alpha=1.0 - CI_LEVEL,
                                            n_boot=2000)
                out[k]['mc'] = {'lift': mean, 'ci': [lo, hi],
                                'pct': mean / float(mc_base.mean()) * 100.0}
            h = lifts['hindsight']
            out['transfer_ratio'] = transfer_ratio(lifts['transferred'], h)
            out['transfer_gap'] = h - lifts['transferred']
            rec['methods'][m] = out
        per_seed.append(rec)
    mc_rows = [(name, r, EVAL_SEED_BASE + r, float(v[r]))
               for name, v in revs.items() for r in range(v.size)]
    return {'baseline': {'closed_form': base_cf,
                         'mc_current_mean': float(mc_base.mean())},
            'per_seed': per_seed, 'mc_rows': mc_rows}


def transfer_ratio(transferred: float, hindsight: float) -> Optional[float]:
    """Transferred / hindsight lift, or None when the hindsight lift is not
    positive: with no achievable lift to share, the ratio is undefined (a
    zero denominator) or meaningless (a negative one flips its sign)."""
    return transferred / hindsight if hindsight > 0.0 else None


def _t(values: Sequence[float]) -> Dict[str, Any]:
    mean, lo, hi, sd = t_interval(values, CI_LEVEL)
    return {'mean': mean, 'ci': [lo, hi], 'sd': sd,
            'min': float(np.min(values)), 'max': float(np.max(values))}


def summarize_transfer(per_seed: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Across seeds, per method: each lift's mean with a t-interval over
    the seeds, range, and percentage; the transfer gap (hindsight minus
    transferred) paired within each seed, with its t-interval over the
    seeds and the seeds where transfer falls short; the ratio of the mean
    transferred to the mean hindsight lift (null unless the mean hindsight
    lift is positive); seeds with a positive transferred lift. And the
    GA's transferred lift against each comparator's, per seed."""
    across: Dict[str, Any] = {}
    for m in METHODS:
        block: Dict[str, Any] = {}
        for k in LIFTS:
            block[k] = {
                'gbp': _t([r['methods'][m][k]['lift'] for r in per_seed]),
                'pct': _t([r['methods'][m][k]['pct'] for r in per_seed])}
        for k in ('transferred', 'hindsight'):
            block[k]['mc_gbp'] = _t([r['methods'][m][k]['mc']['lift']
                                     for r in per_seed])
        # The paired question: does a layout found on the prior period
        # earn less on the current one than a search that knew the current
        # data? Hindsight minus transferred, within each seed.
        gap = [r['methods'][m]['hindsight']['lift']
               - r['methods'][m]['transferred']['lift'] for r in per_seed]
        block['transfer_gap'] = {
            'gbp': {**_t(gap), 'per_seed': gap},
            'pct': _t([r['methods'][m]['hindsight']['pct']
                       - r['methods'][m]['transferred']['pct']
                       for r in per_seed]),
            'mc_gbp': _t([r['methods'][m]['hindsight']['mc']['lift']
                          - r['methods'][m]['transferred']['mc']['lift']
                          for r in per_seed]),
            'n_seeds_transferred_below_hindsight': int(sum(
                g > 0.0 for g in gap)),
            'meaning': ('hindsight minus transferred lift, paired within '
                        'each search seed; positive = transfer loses '
                        'lift'),
        }
        block['transfer_ratio_of_means'] = transfer_ratio(
            block['transferred']['gbp']['mean'],
            block['hindsight']['gbp']['mean'])
        block['n_seeds_transfer_ratio_defined'] = int(sum(
            r['methods'][m]['transfer_ratio'] is not None
            for r in per_seed))
        block['n_seeds_transferred_positive'] = int(sum(
            r['methods'][m]['transferred']['lift'] > 0 for r in per_seed))
        across[m] = block
    ga_vs = {}
    for m in METHODS[1:]:
        diffs = [r['methods']['GA']['transferred']['lift']
                 - r['methods'][m]['transferred']['lift'] for r in per_seed]
        ga_vs[f'GA_minus_{m}'] = {**_t(diffs), 'per_seed': diffs,
                                  'n_ga_leads': int(sum(d > 0 for d in diffs))}
    return {'level': CI_LEVEL, 'n_seeds': len(per_seed), 'methods': across,
            'transferred_ga_vs': ga_vs}


def fixture_facts(stores: Mapping[str, Any],
                  params: Mapping[str, CalibratedParams],
                  max_items_per_category: int) -> Dict[str, Any]:
    """The fixtures against the current period: those whose product did
    not sell in it (kept on the floor with no demand), and how much of the
    current period's own top-N assortment they carry."""
    current = stores['current']
    absent = RK.absent_fixtures(current, params['current'])
    prior_visits = params['prior'].item_visit_counts
    for a in absent:
        a['prior_invoices'] = int(prior_visits.get(a['product_id'], 0))
    stocked = set(RK.fixture_products(current))
    own = {str(p) for pids in _items_per_category(
        params['current'], max_items_per_category).values() for p in pids}
    return {'n_fixtures': len(stocked),
            'n_absent_in_current': len(absent),
            'absent_in_current': absent,
            'handling': ('kept on the floor with no current demand: out of '
                         'the co-purchase pairs, the popularity waypoints '
                         'and the revenue and accessibility weights'),
            'current_top_n_size': len(own),
            'current_top_n_stocked': len(own & stocked),
            'current_top_n_not_stocked': len(own - stocked)}


# --- Main ----------------------------------------------------------------------

def write_results_csv(out_dir: str, per_seed) -> str:
    path = os.path.join(out_dir, 'results.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['seed', 'method', 'promised_lift', 'promised_pct',
                    'transferred_lift', 'transferred_pct',
                    'hindsight_lift', 'hindsight_pct', 'transfer_ratio',
                    'transferred_mc_lift', 'hindsight_mc_lift'])
        for r in per_seed:
            for m in METHODS:
                x = r['methods'][m]
                w.writerow([r['seed'], m,
                            *(f"{x[k][v]:.6f}" for k in LIFTS
                              for v in ('lift', 'pct')),
                            ('' if x['transfer_ratio'] is None
                             else f"{x['transfer_ratio']:.6f}"),
                            f"{x['transferred']['mc']['lift']:.6f}",
                            f"{x['hindsight']['mc']['lift']:.6f}"])
    return path


def write_mc_csv(out_dir: str, rows) -> str:
    path = os.path.join(out_dir, 'mc_replicates.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['layout', 'replicate', 'mc_seed', 'revenue'])
        for name, r, seed, v in rows:
            w.writerow([name, r, seed, f"{v:.4f}"])
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_dir = make_run_dir(args.out_root,
                           run_dir_prefix('heldout_transfer', args))
    experiment = experiment_name('heldout_transfer_uci', args)
    prov_run = provenance_snapshot()
    print(f"output dir: {out_dir}", flush=True)
    wall_t0 = time.perf_counter()

    # -- 1. Both periods, calibrated with Figure C's options ---------------
    records = {p: LS.period_record(args.retail_path, p) for p in PERIODS}
    params = {p: calibrate_transactional(
        LS.period_frame(args.retail_path, p),
        assumed_conversion_rate=args.assumed_conversion,
        currency=LS.CURRENCY, exclude_anonymous=args.exclude_anonymous)
        for p in PERIODS}
    for p in PERIODS:
        print(f"[transfer] {p}: {records[p]['first_date']} .. "
              f"{records[p]['last_date']}, {params[p].n_invoices:,} "
              f"invoices", flush=True)

    # -- 2. The prior store, and the current calibration on its fixtures ---
    stores = {p: period_store(params, p, args.max_items_per_category)
              for p in PERIODS}
    if stores['prior'].baseline_layout != stores['current'].baseline_layout:
        raise AssertionError("the two stores do not share the as-built "
                             "layout")
    fingerprints = {p: store_fingerprint(stores[p]) for p in PERIODS}
    seeds = [args.ga_seed + k for k in range(args.n_search_seeds)]
    budget = args.pop_size * args.n_gens
    design = TransferDesign(
        max_items_per_category=args.max_items_per_category,
        n_gens=args.n_gens, pop_size=args.pop_size, mc_iters=args.mc_iters,
        mc_days=args.mc_days, sa_initial_accept=args.sa_initial_accept,
        fingerprints=tuple(sorted(fingerprints.items())))

    # -- 3. The searches, on the prior store and in hindsight ---------------
    print(f"[transfer] {len(seeds)} search seeds x 2 periods x 3 searches "
          f"(budget {budget} each) on {args.workers} worker(s)...",
          flush=True)
    t0 = time.perf_counter()
    results = map_searches(params, seeds, design, args.workers)
    search_wall = time.perf_counter() - t0
    check_search_results(results, budget)

    # -- 4. Scoring ------------------------------------------------------------
    scored = score_transfer(stores, results, seeds, args.mc_days,
                            args.mc_iters, args.n_mc_replicates)
    across = summarize_transfer(scored['per_seed'])
    facts = fixture_facts(stores, params, args.max_items_per_category)

    plan = seed_plan(args.ga_seed, args.n_search_seeds, args.n_gens,
                     args.pop_size, args.n_mc_replicates)
    design_block = {
        # ``cut_applied_to`` / ``reversals_across_cut`` mark a prior period
        # cut before cleaning, so no current-period cancellation reached
        # into it (a validator can require them).
        'periods': {p: {k: records[p].get(k) for k in (
            'sheet', 'first_date', 'last_date', 'n_invoices', 'cut_before',
            'cut_applied_to', 'reversals_across_cut',
            'n_invoices_shared_with_current', 'source_sha256',
            'adapter_version') if k in records[p]} for p in PERIODS},
        'n_invoices_calibrated': {p: int(params[p].n_invoices)
                                  for p in PERIODS},
        'fixtures': ('Figure C\'s store built from the prior calibration '
                     '(build_store, naive as-built layout)'),
        'rekey': ('the current calibration re-seeded onto the same '
                  'fixtures, base parameters re-anchored at the same '
                  'as-built layout'),
        'lifts': {'promised': 'prior layout, prior calibration',
                  'transferred': 'prior layout, current calibration',
                  'hindsight': 'current-searched layout, current '
                               'calibration'},
        'scoring': ('experiments.closed_form.expected_revenue; Monte Carlo '
                    'on the current calibration under the held-out '
                    'evaluation seeds'),
        **plan,
        'n_search_seeds': args.n_search_seeds,
        'budget_search_evals': budget,
        'block': args.pop_size,
        'n_gens': args.n_gens,
        'pop_size': args.pop_size,
        'n_final_seeds': GA_N_FINAL_SEEDS,
        'ga_mut_rate': GA_MUT_RATE,
        'ga_elite_frac': GA_ELITE_FRAC,
        'mc_iters': args.mc_iters,
        'mc_days': args.mc_days,
        'n_mc_replicates': args.n_mc_replicates,
        'max_items_per_category': args.max_items_per_category,
        'assumed_conversion': args.assumed_conversion,
        'exclude_anonymous': bool(args.exclude_anonymous),
        'sa_initial_accept': args.sa_initial_accept,
        'rs_sampler': 'zone_sampler',
        'sa_neighbor': f'zone_neighbor(step_frac={SA_STEP_FRAC})',
        # What each search's operators draw from: the same space for all
        # three (each item's zone less the aisles the repair keeps).
        'operators': search_operators(),
        'search_start': 'asbuilt',
        'ci': (f'{CI_LEVEL:.0%} Student-t over search seeds; per-seed Monte '
               f'Carlo lifts with percentile bootstrap over replicates'),
    }
    summary = {
        'experiment': experiment,
        'design': design_block,
        'baseline': scored['baseline'],
        'per_seed': scored['per_seed'],
        'across_seeds': across,
        'fixtures': facts,
        # The floor-plan invariants of every reported layout, per (period,
        # seed), what the shared repair did in each search task, and what
        # the aisle rule takes from the search space (as Figure C records
        # them).
        'feasibility': {
            'invariants': {f"{r['period']}|{r['seed']}": r['invariants']
                           for r in results},
            'repair_stats': {f"{r['period']}|{r['seed']}": r['repair_stats']
                             for r in results},
            'aisle_rule': aisle_rule_summary(stores['prior'].shop,
                                             stores['prior'].item_names)},
        'evaluation_counts': {f"{r['period']}|{r['seed']}":
                              r['evaluation_counts'] for r in results},
        'final_evaluation_counts': {f"{r['period']}|{r['seed']}":
                                    r['final_evaluation_counts']
                                    for r in results},
        'store': {'fingerprints': fingerprints,
                  'width_m': stores['prior'].shop.width,
                  'height_m': stores['prior'].shop.height,
                  'sections': stores['prior'].n_sections,
                  'items_placed': len(stores['prior'].item_names)},
    }

    prov = stamp_provenance(
        source_path=args.retail_path,
        adapter_name=OnlineRetailIIAdapter.name,
        adapter_version=OnlineRetailIIAdapter.version,
        rows_in=int(sum(records[p]['rows_in'] for p in PERIODS)),
        rows_kept=int(sum(records[p]['rows_calibrated'] for p in PERIODS)),
        currency=LS.CURRENCY, seed=args.ga_seed,
        extra={'periods': records, 'reader_version': READER_VERSION,
               'exclude_anonymous': bool(args.exclude_anonymous)})
    csv_path = write_results_csv(out_dir, scored['per_seed'])
    mc_path = write_mc_csv(out_dir, scored['mc_rows'])
    write_sidecar(out_dir, {
        'experiment':          experiment,
        'args':                vars(args),
        'wall_seconds':        time.perf_counter() - wall_t0,
        'search_wall_seconds': search_wall,
        'task_wall_seconds':   {f"{r['period']}|{r['seed']}":
                                r['wall_seconds'] for r in results},
        'csv_path':            os.path.relpath(csv_path, out_dir),
        'mc_csv_path':         os.path.relpath(mc_path, out_dir),
        'provenance':          prov.to_dict(),
        'base_params':         {p: base_params_record(stores[p].base_params)
                                for p in PERIODS},
        'sa_T0':               {f"{r['period']}|{r['seed']}": r['sa_T0']
                                for r in results},
        'summary':             summary,
    }, provenance=prov_run)
    with open(os.path.join(out_dir, 'summary.json'), 'w',
              encoding='utf-8') as f:
        # Standard JSON: every undefined quantity is null by construction
        # (transfer_ratio), so a NaN here is a bug to surface, not write.
        json.dump(summary, f, indent=2, allow_nan=False)

    print()
    for m in METHODS:
        a = across['methods'][m]
        ratio = a['transfer_ratio_of_means']
        gap = a['transfer_gap']['gbp']
        print(f"[transfer] {m:>19s}: promised "
              f"{a['promised']['pct']['mean']:+.3f}%  transferred "
              f"{a['transferred']['pct']['mean']:+.3f}%  hindsight "
              f"{a['hindsight']['pct']['mean']:+.3f}%  (gap "
              f"{gap['mean']:+,.2f}, CI {gap['ci'][0]:+,.2f} .. "
              f"{gap['ci'][1]:+,.2f}; ratio of means "
              f"{'n/a' if ratio is None else f'{ratio:.2f}'}; transferred "
              f"> 0 in {a['n_seeds_transferred_positive']}/{len(seeds)})",
              flush=True)
    print(f"[transfer] fixtures absent from the current period: "
          f"{facts['n_absent_in_current']} of {facts['n_fixtures']}; current "
          f"top-{args.max_items_per_category} products not stocked: "
          f"{facts['current_top_n_not_stocked']} of "
          f"{facts['current_top_n_size']}")
    print(f"\nArtifacts in: {out_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
