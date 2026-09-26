"""Figure C at 1x, 3x and 10x its search budget: has any search converged?

Figure C compares the GA, random search and simulated annealing on the
108-fixture calibrated store at one budget -- 750 search evaluations, about
seven per fixture, where the synthetic templates get about 75 per fixture.
A lead at that budget can mean one search is better, or only that it is
faster to a point none of them has reached. This runner measures how far
each search still is from the best layout any of them finds:

  * the same store as Figure C (``run_real_data_example.build_store``: the
    same calibration, the naive as-built layout, the same item cap and
    assumed conversion rate), and Figure C's population and block;
  * budgets ``--budget-multipliers`` (1, 3, 10) times Figure C's
    ``--n-gens``, with the population held at ``--pop-size``: ``m`` x 750
    search evaluations plus the final selection's ``pop_size`` x 5;
  * ``--n-search-seeds`` (5) search seeds from ``--ga-seed``. The same
    search seed is used at every budget, so the GA's and random search's
    shorter runs are exact prefixes of their longer ones (the same
    generations, the same draws) and the budget comparison is paired; the
    annealer's schedule is stretched to each budget, so its runs share only
    their first block;
  * four searches per (budget, seed), all from the as-built layout and at
    one budget: the GA, random search, the one-item annealer of Figure C,
    and an annealer whose moves displace ``k`` items at once (``--sa-k-move``;
    default ``_common.k_move_items`` of the item count -- ten on this store),
    since one-item moves give the annealer about seven proposals per item at
    1x;
  * every final layout scored under Figure C's held-out evaluation seeds
    (``EVAL_SEED_BASE + r``) and in closed form (``experiments.closed_form``:
    its exact expected revenue), after a check of the floor-plan invariants;
  * the BEST-KNOWN reference: the final layout of any method, budget and
    seed with the highest closed-form revenue -- an exact score, so picking
    it carries no winner's curse -- re-scored on the held-out seeds. Each
    method's gap to it is reported per budget, in currency and as a share of
    the GA's Figure C lift (the 1x GA at ``--ga-seed`` over the as-built
    layout), both in closed form and in held-out Monte Carlo;
  * every run's convergence trace and final layout are saved
    (``traces.json``, ``layouts.json``);
  * whether each conclusion -- the sign of every method's gap, the ranking
    of the methods, the sign and significance over search seeds of the GA's
    lift and of the GA minus each other search -- is the same in closed
    form as in held-out Monte Carlo, per budget
    (``results.closed_form_conclusions_unchanged``, ``all_unchanged``).

``--workers N`` spreads the (budget, seed) runs over N processes; each
rebuilds the store and checks it against the parent's fingerprint, and the
results are put back in design order, so the output does not depend on N.

Outputs (under ``--out-root``, one ``real_data_budget_<time>`` directory):
results.csv (one row per budget, seed, method and replicate), layouts.json,
traces.json, sidecar.json and, last, summary.json.

Smoke (one sheet, tiny budget):
    python -m experiments.run_real_data_budget --sheets "Year 2010-2011" ^
        --max-items-per-category 8 --mc-iters 100 --mc-days 5 --n-gens 2 ^
        --pop-size 6 --n-mc-replicates 3 --n-search-seeds 2 ^
        --budget-multipliers 1,2 --out-root <a folder outside the repository>

Paper-grade (Figure C's design; about 2 hours per worker-seed at 10x):
    python -m experiments.run_real_data_budget --workers 2
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from dataset_adapters import OnlineRetailIIAdapter
from dataset_calibration import CalibratedParams
from dataset_provenance import stamp as stamp_provenance

from experiments._common import (
    GA_N_FINAL_SEEDS,
    aisle_rule_summary,
    base_params_record,
    closed_form_revenue,
    k_move_items,
    layout_json,
    make_run_dir,
    provenance_snapshot,
    repair_stats,
    write_json,
    write_sidecar,
)
from experiments.run_real_data_example import (
    EVAL_SEED_BASE,
    SA_STEP_FRAC,
    add_arguments,
    build_store,
    calibrate_from_file,
    check_reported,
    check_search_arguments,
    data_provenance_extra,
    experiment_name,
    reported_layouts,
    resolve_workbook,
    run_dir_prefix,
    run_searches,
    score_layouts,
    search_operators,
    search_seeds_fit,
)
from experiments.run_real_data_seeds import store_fingerprint, t_interval

#: Budgets, as multiples of Figure C's search budget.
DEFAULT_BUDGET_MULTIPLIERS = (1, 3, 10)

#: Search seeds per budget unless told otherwise.
DEFAULT_N_SEARCH_SEEDS = 5

#: The searches every (budget, seed) runs, by the key ``run_searches``
#: returns each final layout under.
METHODS = {'GA': 'optimized', 'random_search': 'rs',
           'simulated_annealing': 'sa', 'simulated_annealing_k': 'sa_k'}


# --- Arguments ---------------------------------------------------------------

def _multipliers(text: str) -> Tuple[int, ...]:
    vals = tuple(int(v) for v in str(text).split(',') if v.strip())
    if not vals or any(v < 1 for v in vals) or len(set(vals)) != len(vals):
        raise argparse.ArgumentTypeError(
            "budget multipliers must be distinct positive integers")
    return tuple(sorted(vals))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Figure C's searches at several multiples of its budget, "
                    "against a best-known reference.")
    add_arguments(p, ga_seed_help="First search seed; seed s = --ga-seed + k "
                                  "seeds every search at every budget")
    p.add_argument('--budget-multipliers', type=_multipliers,
                   default=DEFAULT_BUDGET_MULTIPLIERS,
                   help="Comma-separated multiples of --n-gens (population "
                        "held at --pop-size); the smallest should be 1 so "
                        "Figure C's own budget is among them")
    p.add_argument('--n-search-seeds', type=int,
                   default=DEFAULT_N_SEARCH_SEEDS,
                   help="Search seeds per budget (at least 2)")
    p.add_argument('--sa-k-move', type=int, default=0,
                   help="Items per move of the k-item annealer; 0 (default) "
                        "takes _common.k_move_items of the item count")
    p.add_argument('--workers', type=int, default=1,
                   help="Processes the (budget, seed) runs are spread over; "
                        "the output is the same for any value")
    args = p.parse_args(argv)
    if args.n_search_seeds < 2:
        p.error("--n-search-seeds must be >= 2")
    if args.workers < 1:
        p.error("--workers must be >= 1")
    if args.sa_k_move < 0:
        p.error("--sa-k-move must be >= 0")
    check_search_arguments(p, args)
    top = max(args.budget_multipliers)
    bad = [s for s in range(args.ga_seed, args.ga_seed + args.n_search_seeds)
           if not search_seeds_fit(s, args.n_gens * top, args.pop_size)]
    if bad:
        p.error(f"search seeds {bad[:3]} at {top}x the budget leave their "
                f"block of 1000 Monte Carlo seeds or reach the evaluation "
                f"seeds; lower --ga-seed, --n-search-seeds or the largest "
                f"multiplier")
    resolve_workbook(p, args)
    return args


# --- One (budget, seed) run --------------------------------------------------

@dataclass(frozen=True)
class RunDesign:
    """What every (budget, seed) run needs besides the calibration."""
    max_items_per_category: int
    n_gens: int
    pop_size: int
    mc_iters: int
    mc_days: int
    sa_initial_accept: float
    n_mc_replicates: int
    sa_k_move: int
    store_fingerprint: str


def run_one(params: CalibratedParams, multiplier: int, seed: int,
            design: RunDesign) -> Dict[str, Any]:
    """The four searches at ``multiplier`` x the budget from search seed
    ``seed``, each final layout checked, scored on the held-out seeds and in
    closed form. The store is rebuilt here and checked against the parent's
    fingerprint, so the result is a function of (multiplier, seed) alone."""
    t0 = time.perf_counter()
    store = build_store(params, design.max_items_per_category, verbose=False)
    fp = store_fingerprint(store)
    if fp != design.store_fingerprint:
        raise RuntimeError(f"budget {multiplier}x seed {seed}: the rebuilt "
                           f"store differs from the one the run was set up on")
    n_gens = design.n_gens * multiplier
    searches = run_searches(
        store, seed=seed, n_gens=n_gens, pop_size=design.pop_size,
        mc_iters=design.mc_iters, mc_days=design.mc_days,
        sa_initial_accept=design.sa_initial_accept, verbose=False,
        sa_k_move=design.sa_k_move)
    reported = reported_layouts(store, searches)
    invariants = check_reported(store, reported,
                                f"budget {multiplier}x seed {seed} ")
    revs = score_layouts(store, tuple(reported.items()),
                         n_replicates=design.n_mc_replicates,
                         mc_iters=design.mc_iters, mc_days=design.mc_days)
    cf = {k: closed_form_revenue(store.shop, store.item_names, lay,
                                 store.base_params, design.mc_days)
          for k, lay in reported.items()}
    out = {
        'multiplier': int(multiplier), 'search_seed': int(seed),
        'n_gens': int(n_gens), 'budget': int(searches['budget']),
        'revenues': {k: v.tolist() for k, v in revs.items()},
        'closed_form': cf,
        'layouts': {k: layout_json(lay) for k, lay in reported.items()},
        'invariants': invariants,
        'traces': searches['traces'],
        'evaluation_counts': dict(searches['eval_counts']),
        'final_evaluation_counts': dict(searches['final_eval_counts']),
        'search_start': dict(searches['starts']),
        'sa_T0': {'simulated_annealing': searches['sa_stats'].get('sa_T0'),
                  'simulated_annealing_k':
                      searches['sak_stats'].get('sa_T0')},
        'repair_stats': repair_stats(store.shop),
        'wall_seconds': {**searches['walls'],
                         'total': time.perf_counter() - t0},
    }
    print(f"[budget] {multiplier}x seed {seed}: "
          + '  '.join(f"{m}={cf[k] - cf['baseline']:+.0f}"
                      for m, k in METHODS.items())
          + f"  ({out['wall_seconds']['total']:.0f}s)", flush=True)
    return out


_WORKER_PARAMS: Optional[CalibratedParams] = None


def _init_worker(params: CalibratedParams) -> None:
    global _WORKER_PARAMS
    _WORKER_PARAMS = params


def _task(multiplier: int, seed: int, design: RunDesign) -> Dict[str, Any]:
    return run_one(_WORKER_PARAMS, multiplier, seed, design)


def map_runs(params: CalibratedParams, plan: Sequence[Tuple[int, int]],
             design: RunDesign, workers: int) -> List[Dict[str, Any]]:
    """``[run_one(params, m, s, design) for m, s in plan]`` in plan order,
    over ``workers`` processes."""
    plan = list(plan)
    if workers <= 1 or len(plan) <= 1:
        return [run_one(params, m, s, design) for m, s in plan]
    done: Dict[Tuple[int, int], Dict[str, Any]] = {}
    pool = ProcessPoolExecutor(max_workers=min(workers, len(plan)),
                               initializer=_init_worker, initargs=(params,))
    try:
        futures = {pool.submit(_task, m, s, design): (m, s) for m, s in plan}
        for fut in as_completed(futures):
            done[futures[fut]] = fut.result()
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return [done[k] for k in plan]


# --- Aggregation -------------------------------------------------------------

def check_runs(runs: Sequence[Mapping[str, Any]]) -> None:
    """Refuse results that are not the comparison reported: a search that
    did not start as-built or spent another budget than its peers, or an
    as-built baseline that scored differently in two runs (every run scores
    the same layout under the same seeds)."""
    for r in runs:
        for key in ('evaluation_counts', 'final_evaluation_counts'):
            if len(set(r[key].values())) != 1:
                raise AssertionError(f"{r['multiplier']}x seed "
                                     f"{r['search_seed']}: unequal {key} "
                                     f"{r[key]}")
        if set(r['search_start'].values()) != {'asbuilt'}:
            raise AssertionError(f"{r['multiplier']}x seed "
                                 f"{r['search_seed']}: searches did not all "
                                 f"start as-built {r['search_start']}")
    first = runs[0]['revenues']['baseline']
    for r in runs[1:]:
        if r['revenues']['baseline'] != first:
            raise AssertionError("the as-built layout scored differently in "
                                 "two runs")


def summarize(runs: Sequence[Mapping[str, Any]], ga_seed: int,
              level: float = 0.95) -> Dict[str, Any]:
    """The best-known reference, each method's gap to it per budget, and
    the lift over the as-built layout per budget.

    The reference is the final layout with the highest closed-form revenue
    over every method, budget and seed. Its gap to a layout is taken in
    closed form (exact) and in held-out Monte Carlo (the mean over the
    evaluation replicates of the paired difference). The unit of the
    shares is the GA's Figure C lift: the 1x GA at ``ga_seed`` over the
    as-built layout, in the same currency (closed form for closed-form
    gaps, held-out Monte Carlo for Monte Carlo ones)."""
    ref = max(((r, k) for r in runs for k in METHODS.values()
               if k in r['closed_form']),
              key=lambda rk: (rk[0]['closed_form'][rk[1]],
                              -rk[0]['multiplier'], -rk[0]['search_seed']))
    ref_run, ref_key = ref
    ref_cf = ref_run['closed_form'][ref_key]
    ref_mc = np.asarray(ref_run['revenues'][ref_key])
    base_cf = runs[0]['closed_form']['baseline']
    base_mc = np.asarray(runs[0]['revenues']['baseline'])
    m_min = min(r['multiplier'] for r in runs)
    figc = next((r for r in runs if r['multiplier'] == m_min
                 and r['search_seed'] == ga_seed), runs[0])
    figc_lift_cf = figc['closed_form']['optimized'] - base_cf
    figc_lift_mc = float((np.asarray(figc['revenues']['optimized'])
                          - base_mc).mean())

    per: Dict[str, Dict[str, Any]] = {}
    for method, key in METHODS.items():
        rows: Dict[str, Any] = {}
        for m in sorted({r['multiplier'] for r in runs}):
            sel = [r for r in runs if r['multiplier'] == m
                   and key in r['closed_form']]
            if not sel:
                continue
            gap_cf = np.array([ref_cf - r['closed_form'][key] for r in sel])
            gap_mc = np.array([float((ref_mc - np.asarray(r['revenues'][key]))
                                     .mean()) for r in sel])
            lift_cf = np.array([r['closed_form'][key] - base_cf for r in sel])
            mean, lo, hi, sd = t_interval(gap_cf, level)
            rows[str(m)] = {
                'n_seeds': len(sel),
                'gap_cf': {'mean': mean, 'ci_lo': lo, 'ci_hi': hi, 'sd': sd,
                           'min': float(gap_cf.min()),
                           'max': float(gap_cf.max()),
                           'per_seed': gap_cf.tolist()},
                'gap_cf_share_of_figc_lift': {
                    'mean': mean / figc_lift_cf if figc_lift_cf else None,
                    'min': (float(gap_cf.min()) / figc_lift_cf
                            if figc_lift_cf else None),
                    'max': (float(gap_cf.max()) / figc_lift_cf
                            if figc_lift_cf else None)},
                'gap_mc': {'mean': float(gap_mc.mean()),
                           'min': float(gap_mc.min()),
                           'max': float(gap_mc.max()),
                           'per_seed': gap_mc.tolist()},
                'gap_mc_share_of_figc_lift': (float(gap_mc.mean())
                                              / figc_lift_mc
                                              if figc_lift_mc else None),
                'lift_cf': {'mean': float(lift_cf.mean()),
                            'min': float(lift_cf.min()),
                            'max': float(lift_cf.max()),
                            'pct_mean': float(lift_cf.mean()) / base_cf
                            * 100.0},
            }
        per[method] = rows
    return {
        'reference': {'method': next(m for m, k in METHODS.items()
                                     if k == ref_key),
                      'multiplier': int(ref_run['multiplier']),
                      'search_seed': int(ref_run['search_seed']),
                      'closed_form': ref_cf,
                      'held_out_mc_mean': float(ref_mc.mean()),
                      'lift_over_asbuilt_cf': ref_cf - base_cf,
                      'lift_over_asbuilt_pct': (ref_cf - base_cf) / base_cf
                      * 100.0,
                      'selection': 'highest closed-form revenue over every '
                                   'method, budget and seed'},
        'figc_lift': {'closed_form': figc_lift_cf,
                      'held_out_mc': figc_lift_mc,
                      'pct_cf': figc_lift_cf / base_cf * 100.0,
                      'from': {'method': 'GA', 'multiplier': m_min,
                               'search_seed': int(figc['search_seed'])}},
        'baseline': {'closed_form': base_cf,
                     'held_out_mc_mean': float(base_mc.mean())},
        'per_method': per,
        'ci': f't-interval over search seeds, {level * 100:g}% two-sided',
    }


def _sign(x: float) -> int:
    return int(np.sign(x))


def _excludes_zero(lo: float, hi: float) -> Optional[bool]:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return None
    return bool(lo > 0 or hi < 0)


def closed_form_conclusions(runs: Sequence[Mapping[str, Any]],
                            level: float = 0.95) -> Dict[str, Any]:
    """Whether every conclusion this runner reports holds with each layout
    at its exact mean (closed form) as it does in held-out Monte Carlo.

    Per budget multiplier:

      * per method, the sign of its mean gap to the best-known reference
        (``gap_sign``) -- the reference is the closed-form best, so a
        method's closed-form gap is never negative, and a negative Monte
        Carlo gap would say the evaluation noise reorders them;
      * the ranking of the methods by mean gap (``ranking``);
      * the GA's lift over the as-built layout, and the GA minus each other
        search (``ga_minus``): per seed, the closed-form difference and the
        Monte Carlo one (mean over the evaluation replicates), their means,
        the t-interval over the search seeds at ``level``, the seeds on
        which the GA is ahead, and whether the sign and the significance
        (the interval excluding zero) agree.

    ``all_unchanged`` is True when every sign, every ranking and every
    significance decision agrees."""
    out: Dict[str, Any] = {'level': level, 'per_multiplier': {}}
    checks: List[bool] = []
    ref = max(((r, k) for r in runs for k in METHODS.values()
               if k in r['closed_form']),
              key=lambda rk: (rk[0]['closed_form'][rk[1]],
                              -rk[0]['multiplier'], -rk[0]['search_seed']))
    ref_cf = ref[0]['closed_form'][ref[1]]
    ref_mc = np.asarray(ref[0]['revenues'][ref[1]])
    for m in sorted({r['multiplier'] for r in runs}):
        sel = [r for r in runs if r['multiplier'] == m]
        blk: Dict[str, Any] = {'gap_sign': {}, 'ga_minus': {}}
        mean_gap = {'cf': {}, 'mc': {}}
        for method, key in METHODS.items():
            rows = [r for r in sel if key in r['closed_form']]
            if not rows:
                continue
            g_cf = float(np.mean([ref_cf - r['closed_form'][key]
                                  for r in rows]))
            g_mc = float(np.mean([(ref_mc
                                   - np.asarray(r['revenues'][key])).mean()
                                  for r in rows]))
            mean_gap['cf'][method], mean_gap['mc'][method] = g_cf, g_mc
            same = _sign(g_cf) == _sign(g_mc)
            blk['gap_sign'][method] = {'cf': _sign(g_cf), 'mc': _sign(g_mc),
                                       'unchanged': bool(same)}
            checks.append(same)
        rank_cf = sorted(mean_gap['cf'], key=lambda k: mean_gap['cf'][k])
        rank_mc = sorted(mean_gap['mc'], key=lambda k: mean_gap['mc'][k])
        blk['ranking'] = {'cf': rank_cf, 'mc': rank_mc,
                          'unchanged': rank_cf == rank_mc}
        checks.append(rank_cf == rank_mc)
        pairs = [('lift_over_asbuilt', 'optimized', 'baseline')]
        pairs += [(method, 'optimized', key) for method, key in METHODS.items()
                  if key != 'optimized']
        for label, a, b in pairs:
            rows = [r for r in sel if a in r['closed_form']
                    and b in r['closed_form']]
            if not rows:
                continue
            d_cf = np.array([r['closed_form'][a] - r['closed_form'][b]
                             for r in rows])
            d_mc = np.array([float((np.asarray(r['revenues'][a])
                                    - np.asarray(r['revenues'][b])).mean())
                             for r in rows])
            mc_cf, lo_cf, hi_cf, _ = t_interval(d_cf, level)
            mc_mc, lo_mc, hi_mc, _ = t_interval(d_mc, level)
            sig_cf, sig_mc = _excludes_zero(lo_cf, hi_cf),                 _excludes_zero(lo_mc, hi_mc)
            sign_same = _sign(mc_cf) == _sign(mc_mc)
            sig_same = sig_cf == sig_mc
            blk['ga_minus'][label] = {
                'cf': {'mean': mc_cf, 'ci_lo': lo_cf, 'ci_hi': hi_cf,
                       'excludes_zero': sig_cf,
                       'seeds_ga_ahead': int((d_cf > 0).sum()),
                       'per_seed': d_cf.tolist()},
                'mc': {'mean': mc_mc, 'ci_lo': lo_mc, 'ci_hi': hi_mc,
                       'excludes_zero': sig_mc,
                       'seeds_ga_ahead': int((d_mc > 0).sum()),
                       'per_seed': d_mc.tolist()},
                'n_seeds': int(len(rows)),
                'sign_unchanged': bool(sign_same),
                'significance_unchanged': bool(sig_same),
                'wins_unchanged': bool((d_cf > 0).sum() == (d_mc > 0).sum())}
            checks.extend([sign_same, sig_same])
        blk['all_unchanged'] = bool(
            all(v['unchanged'] for v in blk['gap_sign'].values())
            and blk['ranking']['unchanged']
            and all(v['sign_unchanged'] and v['significance_unchanged']
                    for v in blk['ga_minus'].values()))
        out['per_multiplier'][str(m)] = blk
    out['all_unchanged'] = bool(all(checks))
    return out


def write_results_csv(out_dir: str, runs: Sequence[Mapping[str, Any]]) -> str:
    path = os.path.join(out_dir, 'results.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['multiplier', 'search_seed', 'layout', 'replicate',
                    'mc_seed', 'revenue', 'cf_revenue'])
        for r in runs:
            for k, vals in r['revenues'].items():
                for i, v in enumerate(vals):
                    w.writerow([r['multiplier'], r['search_seed'], k, i,
                                EVAL_SEED_BASE + i, f"{v:.4f}",
                                f"{r['closed_form'][k]:.4f}"])
    return path


# --- Main --------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    # A --exclude-anonymous run is a sensitivity analysis: it writes to a
    # family of its own (noanon_real_data_budget_*), so the macro
    # generator's newest-valid-run rule can never take it for the headline.
    out_dir = make_run_dir(args.out_root,
                           run_dir_prefix('real_data_budget', args))
    experiment = experiment_name('real_data_uci_budget', args)
    prov_run = provenance_snapshot()
    print(f"output dir: {out_dir}", flush=True)
    wall_t0 = time.perf_counter()

    params, report, reader = calibrate_from_file(args)
    store = build_store(params, args.max_items_per_category)
    k = args.sa_k_move or k_move_items(len(store.item_names))
    design = RunDesign(
        max_items_per_category=args.max_items_per_category,
        n_gens=args.n_gens, pop_size=args.pop_size, mc_iters=args.mc_iters,
        mc_days=args.mc_days, sa_initial_accept=args.sa_initial_accept,
        n_mc_replicates=args.n_mc_replicates, sa_k_move=int(k),
        store_fingerprint=store_fingerprint(store))
    seeds = [args.ga_seed + i for i in range(args.n_search_seeds)]
    plan = [(m, s) for m in args.budget_multipliers for s in seeds]
    print(f"[budget] {len(plan)} runs: multipliers {args.budget_multipliers} "
          f"x seeds {seeds}; {k}-item annealer; on {args.workers} worker(s)",
          flush=True)
    runs = map_runs(params, plan, design, args.workers)
    check_runs(runs)
    summary_block = summarize(runs, args.ga_seed)
    # The same conclusions with every layout at its exact mean.
    summary_block['closed_form_conclusions_unchanged'] = \
        closed_form_conclusions(runs)

    csv_path = write_results_csv(out_dir, runs)
    write_json(out_dir, 'layouts.json', {
        f"{r['multiplier']}x/{r['search_seed']}": r['layouts'] for r in runs})
    write_json(out_dir, 'traces.json', {
        f"{r['multiplier']}x/{r['search_seed']}": r['traces'] for r in runs})
    budget_design = {
        'budget_multipliers': list(args.budget_multipliers),
        'n_gens_per_multiplier': {str(m): args.n_gens * m
                                  for m in args.budget_multipliers},
        'search_evals_per_multiplier': {
            str(m): args.n_gens * m * args.pop_size
            for m in args.budget_multipliers},
        'final_evals': args.pop_size * GA_N_FINAL_SEEDS,
        'pop_size': args.pop_size, 'block': args.pop_size,
        'n_final_seeds': GA_N_FINAL_SEEDS,
        'search_seeds': seeds, 'n_search_seeds': args.n_search_seeds,
        'first_search_seed': args.ga_seed,
        'paired_eval_seeds': [EVAL_SEED_BASE,
                              EVAL_SEED_BASE + args.n_mc_replicates],
        'n_mc_replicates': args.n_mc_replicates,
        'mc_iters': args.mc_iters, 'mc_days': args.mc_days,
        'sheets': args.sheets,
        'max_items_per_category': args.max_items_per_category,
        'assumed_conversion': args.assumed_conversion,
        'exclude_anonymous': bool(args.exclude_anonymous),
        'methods': list(METHODS),
        'sa_neighbor': f'zone_neighbor(step_frac={SA_STEP_FRAC})',
        'sa_k_neighbor': f'zone_neighbor(step_frac={SA_STEP_FRAC}, '
                         f'n_move={k})',
        'operators': search_operators(sa_k_move=k),
        'sa_k_move': int(k),
        'sa_initial_accept': args.sa_initial_accept,
        'nesting': 'the same search seed at every budget: GA and random '
                   'search runs are prefixes of their longer runs',
    }
    per_run = [{'multiplier': r['multiplier'],
                'search_seed': r['search_seed'],
                'closed_form': r['closed_form'],
                'held_out_mc_mean': {k2: float(np.mean(v))
                                     for k2, v in r['revenues'].items()},
                'evaluation_counts': r['evaluation_counts'],
                'final_evaluation_counts': r['final_evaluation_counts'],
                'search_start': r['search_start'],
                'sa_T0': r['sa_T0'],
                'invariants': r['invariants'],
                'repair_stats': r['repair_stats']} for r in runs]
    search_start = {m: '/'.join(sorted({r['search_start'][m] for r in runs}))
                    for m in runs[0]['search_start']}
    prov = stamp_provenance(
        source_path=args.retail_path,
        adapter_name=OnlineRetailIIAdapter.name,
        adapter_version=OnlineRetailIIAdapter.version,
        rows_in=report.rows_in, rows_kept=report.rows_kept,
        currency=report.extra.get('currency', 'GBP'), seed=args.ga_seed,
        extra={'sheets': args.sheets, 'reader': reader,
               'items_placed': len(store.item_names),
               # Reader version, cleaning counts, anonymous invoices.
               **data_provenance_extra(report, params, args)})
    # What the store was calibrated from, as Figure C's summary records
    # it: a headline validator must refuse a no-anonymous sensitivity run
    # and a run under older cleaning.
    data_design = {
        'sheets':                 args.sheets,
        'max_items_per_category': args.max_items_per_category,
        'assumed_conversion':     args.assumed_conversion,
        'exclude_anonymous':      bool(args.exclude_anonymous),
        'adapter_version':        OnlineRetailIIAdapter.version,
    }
    summary = {
        'experiment': experiment,
        'data': data_design,
        'results': summary_block,
        'design': budget_design,
        'search_start': search_start,
        'per_run': per_run,
        'store': {'fingerprint': design.store_fingerprint,
                  'width_m': store.shop.width, 'height_m': store.shop.height,
                  'sections': store.n_sections,
                  'items_placed': len(store.item_names)},
        'feasibility': {'aisle_rule': aisle_rule_summary(store.shop,
                                                         store.item_names)},
        'layouts_path': 'layouts.json', 'traces_path': 'traces.json',
    }
    write_sidecar(out_dir, {
        'experiment': experiment,
        'args': vars(args),
        'wall_seconds': time.perf_counter() - wall_t0,
        'run_wall_seconds': {f"{r['multiplier']}x/{r['search_seed']}":
                             r['wall_seconds'] for r in runs},
        'csv_path': os.path.relpath(csv_path, out_dir),
        'provenance': prov.to_dict(),
        'reader': reader,
        'base_params': base_params_record(store.base_params),
        'run_design': asdict(design),
        'summary': summary,
    }, provenance=prov_run)
    # Written last: its presence marks a finished run.
    write_json(out_dir, 'summary.json', summary)

    ref = summary_block['reference']
    print(f"\n[budget] conclusions unchanged in closed form: "
          f"{summary_block['closed_form_conclusions_unchanged']['all_unchanged']}")
    print(f"[budget] best-known: {ref['method']} at {ref['multiplier']}x, "
          f"seed {ref['search_seed']}: lift {ref['lift_over_asbuilt_pct']:+.2f}%"
          f" (Figure C GA lift {summary_block['figc_lift']['pct_cf']:+.2f}%)")
    for method, rows in summary_block['per_method'].items():
        for m, row in rows.items():
            sh = row['gap_cf_share_of_figc_lift']['mean']
            print(f"   {method:<24s} {m:>3s}x  gap {row['gap_cf']['mean']:>+10.1f}"
                  f"  ({'' if sh is None else f'{sh * 100:.0f}% of the Fig C lift'})")
    print(f"\nArtifacts in: {out_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
