"""Figure C over several search seeds: does the GA's lead replicate?

``run_real_data_example`` (Figure C) runs the GA, random search and
simulated annealing once each, from one search seed (``--ga-seed``), and
scores the three layouts under held-out paired Monte Carlo replicates.
Those replicates re-score three FIXED layouts, so their intervals cover
evaluation noise only. Another search seed can return other layouts, and
the GA's lead over the two comparators is a finding about the searches only
if it survives that change. This runner measures the search-to-search
spread directly.

Design:
  * The same store as Figure C: the same calibration (both sheets by
    default), the naive as-built layout, the same cap on items per
    category and the same assumed conversion rate. The options and their
    defaults come from ``run_real_data_example.add_arguments``, and the
    store from its ``build_store``.
  * K search seeds s = ``--ga-seed`` + k, k in [0, ``--n-search-seeds``).
    For each, the three searches run at Figure C's equal budget from the
    same naive start (the zone hooks, ``start='asbuilt'``) through
    ``run_real_data_example.run_searches``, each seeded from s exactly as
    Figure C seeds them from ``--ga-seed``. Seed s = ``--ga-seed`` is
    therefore Figure C's own run at the same options.
  * Every layout of every seed is scored under the SAME held-out evaluation
    seeds as Figure C (``EVAL_SEED_BASE + r``, r in [0,
    ``--n-mc-replicates``)), so every comparison is paired across replicates
    and across search seeds. The as-built baseline is scored with each seed
    too and must come out identical for all of them, which checks that each
    seed searched the same store.
  * Per seed, the paired mean over the replicates of GA minus random
    search, GA minus annealing and GA minus the as-built layout (the lift).
    Across seeds, the mean of those per-seed means with a Student-t
    interval over the K seeds: the search-to-search uncertainty. The seeds
    share one evaluation panel, so the interval is conditional on it; the
    evaluation-noise part is what Figure C's bootstrap covers, and
    ``replicate_se_mean`` records its size beside the across-seed spread.
  * Seeds: search seed s scores and selects under ``s*1000 + [0, n_gens +
    1 + GA_N_FINAL_SEEDS)``, inside its own block of 1000, so no two search
    seeds share a Monte Carlo draw and none reaches the evaluation seeds at
    ``EVAL_SEED_BASE``; ``parse_args`` refuses a range that would.
  * ``--workers N`` maps the search seeds over N processes. Each seed
    rebuilds the store from the calibration, checks it against the
    parent's fingerprint, and draws only from generators seeded for that
    seed; results are put back in seed order before anything is
    aggregated, so the output does not depend on N.

Outputs (under ``--out-root``, one ``real_data_seeds_<time>`` directory):
  * results.csv  -- one row per (search seed, replicate): baseline, GA,
                    random-search and annealing revenue and the paired
                    differences, in Figure C's column names;
  * sidecar.json -- provenance (checkout at start, dataset SHA-256), the
                    calibration and store, wall times; written before
  * summary.json -- per-seed and across-seed results, the equal-budget
                    evaluation counts per method and seed, the searches'
                    starts and the design. Written last: it marks a
                    finished run.

Smoke (one sheet, tiny budget; a few minutes, mostly the workbook read):
    python -m experiments.run_real_data_seeds ^
        --sheets "Year 2010-2011" --n-search-seeds 3 ^
        --mc-iters 100 --mc-days 5 --n-gens 2 --pop-size 6 ^
        --n-mc-replicates 5 --out-root <a folder outside the repository>

Paper-grade (Figure C's design, ten search seeds; about ten Figure C runs
of compute, spread over ``--workers``):
    python -m experiments.run_real_data_seeds --workers 2
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sps

# Make sibling project modules importable when run as ``python -m ...``.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from dataset_adapters import OnlineRetailIIAdapter
from dataset_calibration import CalibratedParams
from dataset_provenance import stamp as stamp_provenance

from experiments._common import (
    GA_N_FINAL_SEEDS,
    make_run_dir,
    provenance_snapshot,
    write_sidecar,
)
from experiments.run_real_data_example import (
    EVAL_SEED_BASE,
    SA_STEP_FRAC,
    CalibratedStore,
    add_arguments,
    build_store,
    calibrate_from_file,
    check_search_arguments,
    resolve_workbook,
    run_searches,
    score_layouts,
    search_seed_ranges,
    search_seeds_fit,
)


#: Search seeds per run unless told otherwise. Ten gives the across-seed
#: t-interval nine degrees of freedom, so its half-width is within about
#: 12% of the normal-theory one (t = 2.26 against 1.96).
DEFAULT_N_SEARCH_SEEDS = 10

#: Two-sided level of the across-seed t-interval, the level Figure C's
#: bootstrap intervals are quoted at.
CI_LEVEL = 0.95

#: The searches every seed runs; all three must start from the as-built
#: layout and spend the same search budget.
SEARCHES = ('GA', 'random_search', 'simulated_annealing')

#: The layouts ``score_layouts`` scores per seed, by the names Figure C
#: gives them, and the comparisons made from them: summary key, the other
#: layout's name, and the key of that layout's mean in the per-seed record.
LAYOUTS = ('baseline', 'optimized', 'rs', 'sa')
COMPARISONS = (('random_search', 'rs'),
               ('simulated_annealing', 'sa'),
               ('lift_over_baseline', 'baseline'))


# --- Seeds -------------------------------------------------------------------

def search_seeds(first_seed: int, n_search_seeds: int) -> List[int]:
    """The run's search seeds, in the order they are aggregated."""
    return [first_seed + k for k in range(n_search_seeds)]


def seed_plan_error(first_seed: int, n_search_seeds: int, n_gens: int,
                    pop_size: int) -> Optional[str]:
    """Why the search seeds cannot be used, or None when they can.

    Every search seed must keep its own search and selection seeds inside
    its block of 1000 and below the evaluation seeds (``search_seeds_fit``,
    Figure C's rule), for every seed of the range, not only the first."""
    bad = [s for s in search_seeds(first_seed, n_search_seeds)
           if not search_seeds_fit(s, n_gens, pop_size)]
    if not bad:
        return None
    return (f"--ga-seed + --n-search-seeds - 1 must be <= 999 (and >= 0) and "
            f"--n-gens + {1 + GA_N_FINAL_SEEDS} <= 1000, so the search seeds "
            f"of every search seed stay inside its own block of 1000, below "
            f"the evaluation seeds at {EVAL_SEED_BASE}; search seed(s) "
            f"{bad[:3]}{'...' if len(bad) > 3 else ''} do not")


def seed_plan(first_seed: int, n_search_seeds: int, n_gens: int,
              pop_size: int, n_replicates: int) -> Dict[str, Any]:
    """The Monte Carlo seeds the run uses, as recorded in the summary:
    each search seed's half-open search ranges per method and the
    evaluation range every layout is scored under."""
    seeds = search_seeds(first_seed, n_search_seeds)
    return {
        'search_seeds': seeds,
        'search_seed_ranges': {str(s): search_seed_ranges(s, n_gens,
                                                          pop_size)
                               for s in seeds},
        'paired_eval_seeds': [EVAL_SEED_BASE, EVAL_SEED_BASE + n_replicates],
    }


# --- Arguments ---------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Figure C's three equal-budget searches over several "
                    "search seeds, scored under Figure C's evaluation seeds.")
    add_arguments(p, ga_seed_help="First search seed; seed s = --ga-seed + k "
                                  "seeds the GA, random search and simulated "
                                  "annealing as run_real_data_example seeds "
                                  "them from --ga-seed")
    p.add_argument('--n-search-seeds', type=int,
                   default=DEFAULT_N_SEARCH_SEEDS,
                   help="Number of search seeds K (at least 2, for an "
                        "across-seed interval)")
    p.add_argument('--workers', type=int, default=1,
                   help="Processes the search seeds are spread over; the "
                        "output is the same for any value")
    args = p.parse_args(argv)
    if args.n_search_seeds < 2:
        p.error("--n-search-seeds must be >= 2: one seed has no "
                "search-to-search spread to measure")
    if args.workers < 1:
        p.error("--workers must be >= 1")
    check_search_arguments(p, args)
    bad = seed_plan_error(args.ga_seed, args.n_search_seeds, args.n_gens,
                          args.pop_size)
    if bad:
        p.error(bad)
    resolve_workbook(p, args)
    return args


# --- One search seed ---------------------------------------------------------

@dataclass(frozen=True)
class SeedDesign:
    """What every search seed needs besides the calibration: the store
    options, Figure C's search budget and scoring, and the fingerprint of
    the store the parent built, which each seed's rebuild must match."""
    max_items_per_category: int
    n_gens: int
    pop_size: int
    mc_iters: int
    mc_days: int
    sa_initial_accept: float
    n_mc_replicates: int
    store_fingerprint: str


def store_fingerprint(store: CalibratedStore) -> str:
    """SHA-256 over what the searches and the scoring read from the store:
    the floor's dimensions, every placed item (position, size, category,
    zone) and every wall in layout order, the MC base parameters and the
    heat-map prior the GA score reads. Floats go through JSON, which writes
    their shortest exact form, so equal stores hash equal and any drift in a
    rebuild shows."""
    shop = store.shop
    f1 = shop.floors[1]
    items = [[n, [float(v) for v in d['position']],
              [float(v) for v in d.get('size', (0.0, 0.0))],
              str(d.get('category')), str(d.get('zone'))]
             for n, d in f1['items'].items()]
    walls = [[n, [float(v) for v in w['position']],
              [float(v) for v in w['size']]]
             for n, w in f1['walls'].items()]
    h = hashlib.sha256(json.dumps(
        {'dims': [float(shop.width), float(shop.height)],
         'items': items, 'walls': walls,
         'base_params': store.base_params}, sort_keys=True).encode())
    heat = np.ascontiguousarray(shop.customer_simulation.heat_raw)
    h.update(str(heat.dtype).encode() + str(heat.shape).encode())
    h.update(heat.tobytes())
    return h.hexdigest()


def run_search_seed(params: CalibratedParams, seed: int,
                    design: SeedDesign) -> Dict[str, Any]:
    """Figure C's searches and scoring for one search seed.

    The store is rebuilt here rather than shared: a seed then starts from
    the state Figure C's own run starts from, whichever process it runs in
    and whatever seed ran there before it. Every draw comes from a
    generator seeded for this seed (the searches' own streams and the
    evaluation seeds), so the result is a function of ``seed`` alone."""
    t0 = time.perf_counter()
    store = build_store(params, design.max_items_per_category, verbose=False)
    fp = store_fingerprint(store)
    if fp != design.store_fingerprint:
        raise RuntimeError(
            f"search seed {seed}: the rebuilt store differs from the one the "
            f"run was set up on (fingerprint {fp[:12]} != "
            f"{design.store_fingerprint[:12]}); its layouts could not be "
            f"compared with the other seeds'")
    searches = run_searches(
        store, seed=seed, n_gens=design.n_gens, pop_size=design.pop_size,
        mc_iters=design.mc_iters, mc_days=design.mc_days,
        sa_initial_accept=design.sa_initial_accept, verbose=False)
    revs = score_layouts(
        store, (('baseline', store.baseline_layout),
                ('optimized', searches['optimized']),
                ('rs', searches['rs']),
                ('sa', searches['sa'])),
        n_replicates=design.n_mc_replicates, mc_iters=design.mc_iters,
        mc_days=design.mc_days)
    walls = {**searches['walls'], 'total': time.perf_counter() - t0}
    out = {
        'search_seed':             int(seed),
        'revenues':                {k: v.tolist() for k, v in revs.items()},
        'budget':                  int(searches['budget']),
        'evaluation_counts':       dict(searches['eval_counts']),
        'final_evaluation_counts': dict(searches['final_eval_counts']),
        'search_start':            dict(searches['starts']),
        'ga_best_fit':             float(searches['ga_out']['best_fit']),
        'sa_T0':                   searches['sa_stats'].get('sa_T0'),
        'wall_seconds':            walls,
    }
    o, b = revs['optimized'], revs['baseline']
    print(f"[seeds] search seed {seed}: lift={float((o - b).mean()):+.2f}  "
          f"GA-RS={float((o - revs['rs']).mean()):+.2f}  "
          f"GA-SA={float((o - revs['sa']).mean()):+.2f}  "
          f"({walls['total']:.1f}s)", flush=True)
    return out


# Set in each worker process by ``_init_worker``: the calibration is handed
# over once per process rather than once per seed.
_WORKER_PARAMS: Optional[CalibratedParams] = None


def _init_worker(params: CalibratedParams) -> None:
    global _WORKER_PARAMS
    _WORKER_PARAMS = params


def _seed_task(seed: int, design: SeedDesign) -> Dict[str, Any]:
    return run_search_seed(_WORKER_PARAMS, seed, design)


def map_search_seeds(params: CalibratedParams, seeds: Sequence[int],
                     design: SeedDesign, workers: int) -> List[Dict[str, Any]]:
    """``[run_search_seed(params, s, design) for s in seeds]``, in seed
    order.

    With ``workers`` > 1 the seeds run in separate processes. Each builds
    its own store and seeds every generator it draws from, and results are
    put back in seed order before anything is aggregated, so the output is
    identical to the serial run."""
    seeds = list(seeds)
    if workers <= 1 or len(seeds) <= 1:
        return [run_search_seed(params, s, design) for s in seeds]
    done: Dict[int, Dict[str, Any]] = {}
    pool = ProcessPoolExecutor(max_workers=min(workers, len(seeds)),
                               initializer=_init_worker, initargs=(params,))
    try:
        futures = {pool.submit(_seed_task, s, design): s for s in seeds}
        for fut in as_completed(futures):
            done[futures[fut]] = fut.result()
    except BaseException:
        # Leaving the pool's context manager would wait for every queued
        # seed first, so a seed that fails early would only report once the
        # rest had run. Drop what has not started and let the error out now.
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return [done[s] for s in seeds]


def check_seed_results(per_seed: Sequence[Mapping[str, Any]],
                       seeds: Sequence[int], budget: int) -> None:
    """Refuse results that are not the comparison the summary reports:
    seeds out of order, a search that did not start as-built or spent a
    different search budget, or a baseline that scored differently under
    one seed than under another -- the baseline layout and the evaluation
    seeds are the same for every seed, so any difference means the seeds
    did not search the same store."""
    got = [int(r['search_seed']) for r in per_seed]
    if got != list(seeds):
        raise AssertionError(f"seed results out of order: {got} != "
                             f"{list(seeds)}")
    for r in per_seed:
        counts = r['evaluation_counts']
        if any(int(counts[m]) != budget for m in SEARCHES):
            raise AssertionError(f"search seed {r['search_seed']}: search "
                                 f"budgets {counts} differ from {budget}")
        if any(r['search_start'][m] != 'asbuilt' for m in SEARCHES):
            raise AssertionError(f"search seed {r['search_seed']}: searches "
                                 f"did not all start as-built: "
                                 f"{r['search_start']}")
    first = per_seed[0]['revenues']['baseline']
    for r in per_seed[1:]:
        if r['revenues']['baseline'] != first:
            raise AssertionError(
                f"search seed {r['search_seed']} scored the as-built layout "
                f"differently from search seed {per_seed[0]['search_seed']}")


# --- Aggregation -------------------------------------------------------------

def t_interval(x: Sequence[float],
               level: float = CI_LEVEL) -> Tuple[float, float, float, float]:
    """``(mean, lo, hi, sd)``: the Student-t interval for the mean of ``x``
    at two-sided ``level``, df = len(x) - 1. NaN bounds for fewer than two
    values, where no spread can be estimated."""
    x = np.asarray(x, dtype=np.float64)
    mean = float(x.mean())
    if x.size < 2:
        return mean, float('nan'), float('nan'), float('nan')
    sd = float(x.std(ddof=1))
    half = float(sps.t.ppf(0.5 + level / 2.0, x.size - 1)) * sd \
        / math.sqrt(x.size)
    return mean, mean - half, mean + half, sd


def summarize_seeds(per_seed: Sequence[Mapping[str, Any]],
                    level: float = CI_LEVEL) -> Dict[str, Any]:
    """Per-seed and across-seed results from each seed's replicate revenues.

    ``per_seed`` holds, in seed order, ``{'search_seed': s, 'revenues':
    {'baseline' | 'optimized' | 'rs' | 'sa': [revenue per replicate]}}``.
    Per seed, each comparison is the paired mean over the replicates of the
    GA's revenue minus the other layout's, and its percentage is taken of
    that layout's mean revenue, as in Figure C. Across seeds, the per-seed
    means are summarised by their mean, a t-interval over the seeds, their
    spread and range, and the number of seeds in which the GA's layout
    earned more; the percentages are taken of the other layout's mean
    revenue over every seed and replicate. ``replicate_se_mean`` is the mean
    over seeds of the standard error of a per-seed mean from its replicates:
    the evaluation noise inside each per-seed value, to set against the
    across-seed ``sd``."""
    rows: List[Dict[str, Any]] = []
    diffs = {key: [] for key, _ in COMPARISONS}
    rep_se = {key: [] for key, _ in COMPARISONS}
    other_revs = {key: [] for key, _ in COMPARISONS}
    for rec in per_seed:
        rev = {k: np.asarray(rec['revenues'][k], dtype=np.float64)
               for k in LAYOUTS}
        n_rep = {v.size for v in rev.values()}
        if len(n_rep) != 1:
            raise ValueError(f"search seed {rec['search_seed']}: layouts "
                             f"scored over unequal replicate counts {n_rep}")
        row: Dict[str, Any] = {'search_seed': int(rec['search_seed']),
                               'n_replicates': int(n_rep.pop())}
        for k in LAYOUTS:
            row[f'{k}_mean'] = float(rev[k].mean())
        for key, other in COMPARISONS:
            d = rev['optimized'] - rev[other]
            m = float(d.mean())
            denom = max(float(rev[other].mean()), 1e-6)
            se = (float(d.std(ddof=1)) / math.sqrt(d.size)
                  if d.size > 1 else float('nan'))
            row[key] = {'paired_mean_diff_GA_minus_X': m,
                        'pct_diff': m / denom * 100.0,
                        'replicate_se': se,
                        'ga_leads': bool(m > 0)}
            diffs[key].append(m)
            rep_se[key].append(se)
            other_revs[key].append(rev[other])
        rows.append(row)

    across: Dict[str, Any] = {}
    for key, _ in COMPARISONS:
        x = np.asarray(diffs[key], dtype=np.float64)
        mean, lo, hi, sd = t_interval(x, level)
        denom = max(float(np.concatenate(other_revs[key]).mean()), 1e-6)
        across[key] = {
            'n_seeds':          int(x.size),
            'mean':             mean,
            'ci_lo':            lo,
            'ci_hi':            hi,
            'ci_level':         level,
            'sd':               sd,
            'min':              float(x.min()),
            'max':              float(x.max()),
            'n_ga_leads':       int((x > 0).sum()),
            'excludes_zero':    bool(lo > 0 or hi < 0),
            'pct':              mean / denom * 100.0,
            'pct_ci':           [lo / denom * 100.0, hi / denom * 100.0],
            'pct_min':          float(x.min()) / denom * 100.0,
            'pct_max':          float(x.max()) / denom * 100.0,
            'pct_of':           'mean revenue of the other layout over '
                                'every seed and replicate',
            'replicate_se_mean': float(np.mean(rep_se[key])),
        }
    return {'per_seed': rows, 'across_seeds': across,
            'ci': f't-interval over the per-seed paired means, df = K - 1, '
                  f'{level * 100:g}% two-sided; every seed is scored under '
                  f'the same evaluation seeds'}


def search_start(per_seed: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    """Each search's start over the seeds; a search that started from
    different layouts in different seeds is recorded as their '/'-joined
    names, which the macro check refuses."""
    return {m: '/'.join(sorted({str(r['search_start'][m]) for r in per_seed}))
            for m in SEARCHES}


def write_results_csv(out_dir: str,
                      per_seed: Sequence[Mapping[str, Any]]) -> str:
    """One row per (search seed, replicate), in seed then replicate order,
    with Figure C's column names and precision."""
    path = os.path.join(out_dir, 'results.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['search_seed', 'replicate', 'mc_seed',
                    'baseline_revenue', 'optimized_revenue', 'diff',
                    'rs_revenue', 'sa_revenue', 'diff_rs', 'diff_sa'])
        for rec in per_seed:
            rev = rec['revenues']
            for r, (b, o, rs, sa) in enumerate(zip(
                    rev['baseline'], rev['optimized'], rev['rs'],
                    rev['sa'])):
                w.writerow([rec['search_seed'], r, EVAL_SEED_BASE + r,
                            f"{b:.4f}", f"{o:.4f}", f"{o - b:.4f}",
                            f"{rs:.4f}", f"{sa:.4f}",
                            f"{o - rs:.4f}", f"{o - sa:.4f}"])
    return path


# --- Main --------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_dir = make_run_dir(args.out_root, 'real_data_seeds')
    # The checkout as the run starts, so the sidecar names the code that
    # produced the numbers and says whether it moved while they were made.
    prov_run = provenance_snapshot()
    print(f"output dir: {out_dir}", flush=True)
    wall_t0 = time.perf_counter()

    # -- 1. Calibrate and build Figure C's store once, in this process --
    params, report, reader = calibrate_from_file(args)
    store = build_store(params, args.max_items_per_category)
    fingerprint = store_fingerprint(store)
    seeds = search_seeds(args.ga_seed, args.n_search_seeds)
    budget = args.pop_size * args.n_gens
    design = SeedDesign(
        max_items_per_category=args.max_items_per_category,
        n_gens=args.n_gens, pop_size=args.pop_size,
        mc_iters=args.mc_iters, mc_days=args.mc_days,
        sa_initial_accept=args.sa_initial_accept,
        n_mc_replicates=args.n_mc_replicates,
        store_fingerprint=fingerprint)

    # -- 2. The three searches per seed, scored on the held-out seeds ---
    print(f"[seeds] {len(seeds)} search seeds {seeds[0]}..{seeds[-1]}, "
          f"budget {budget} search evaluations each (block {args.pop_size}, "
          f"{GA_N_FINAL_SEEDS} final-selection seeds), "
          f"{args.n_mc_replicates} evaluation seeds from {EVAL_SEED_BASE}, "
          f"on {args.workers} worker(s)...", flush=True)
    t0 = time.perf_counter()
    per_seed = map_search_seeds(params, seeds, design, args.workers)
    search_wall = time.perf_counter() - t0
    check_seed_results(per_seed, seeds, budget)

    # -- 3. Aggregate in seed order --------------------------------------
    results = summarize_seeds(per_seed)
    across = results['across_seeds']
    starts = search_start(per_seed)
    eval_counts = {str(r['search_seed']): r['evaluation_counts']
                   for r in per_seed}
    final_eval_counts = {str(r['search_seed']): r['final_evaluation_counts']
                         for r in per_seed}
    sa_schedule = {'sa_initial_accept': args.sa_initial_accept,
                   'sa_T0': {str(r['search_seed']): r['sa_T0']
                             for r in per_seed},
                   'sa_start': starts['simulated_annealing']}
    plan = seed_plan(args.ga_seed, args.n_search_seeds, args.n_gens,
                     args.pop_size, args.n_mc_replicates)
    design_block = {
        'n_search_seeds':         args.n_search_seeds,
        'first_search_seed':      args.ga_seed,
        **plan,
        'n_mc_replicates':        args.n_mc_replicates,
        'budget_search_evals':    budget,
        'block':                  args.pop_size,
        'n_gens':                 args.n_gens,
        'pop_size':               args.pop_size,
        'n_final_seeds':          GA_N_FINAL_SEEDS,
        'mc_iters':               args.mc_iters,
        'mc_days':                args.mc_days,
        'sheets':                 args.sheets,
        'max_items_per_category': args.max_items_per_category,
        'assumed_conversion':     args.assumed_conversion,
        'layout':                 'naive',
        'rs_sampler':             'zone_sampler',
        'sa_neighbor':            f'zone_neighbor(step_frac={SA_STEP_FRAC})',
        'seeding':                'search seed s seeds the GA, random search '
                                  'and annealing as run_real_data_example '
                                  'seeds them from --ga-seed',
        'ci':                     results['ci'],
    }
    store_block = {
        'fingerprint':  fingerprint,
        'width_m':      store.shop.width,
        'height_m':     store.shop.height,
        'sections':     store.n_sections,
        'items_placed': len(store.item_names),
    }

    print()
    print(f"[seeds] ACROSS {len(seeds)} SEARCH SEEDS "
          f"(mean of per-seed paired means, {CI_LEVEL * 100:g}% t-interval):")
    for label, key in (('lift over as-built', 'lift_over_baseline'),
                       ('GA - random search', 'random_search'),
                       ('GA - sim. annealing', 'simulated_annealing')):
        a = across[key]
        print(f"       {label + ':':<22s}{a['mean']:>+12.2f}  "
              f"({a['pct']:+.2f}%)  CI [{a['ci_lo']:+.2f}, "
              f"{a['ci_hi']:+.2f}]  range [{a['min']:+.2f}, "
              f"{a['max']:+.2f}]  GA ahead in {a['n_ga_leads']}/{len(seeds)}")

    # -- 4. Provenance, CSV, sidecar, summary ------------------------------
    prov = stamp_provenance(
        source_path=args.retail_path,
        adapter_name=OnlineRetailIIAdapter.name,
        adapter_version=OnlineRetailIIAdapter.version,
        rows_in=report.rows_in,
        rows_kept=report.rows_kept,
        currency=report.extra.get('currency', 'GBP'),
        seed=args.ga_seed,
        extra={
            'sheets':                 args.sheets,
            'reader':                 reader,
            'max_items_per_category': args.max_items_per_category,
            'mc_iters':               args.mc_iters,
            'mc_days':                args.mc_days,
            'n_gens':                 args.n_gens,
            'pop_size':               args.pop_size,
            'n_mc_replicates':        args.n_mc_replicates,
            'search_seeds':           seeds,
            'shop_dims_m':            [store.shop.width, store.shop.height],
            'sections':               store.n_sections,
            'items_placed':           len(store.item_names),
        },
    )
    csv_path = write_results_csv(out_dir, per_seed)

    # The sidecar goes first: summary.json is the file that marks a run as
    # finished, so it is written last.
    write_sidecar(out_dir, {
        'experiment':          'real_data_uci_search_seeds',
        'args':                vars(args),
        'wall_seconds':        time.perf_counter() - wall_t0,
        'search_wall_seconds': search_wall,
        'seed_wall_seconds':   {str(r['search_seed']): r['wall_seconds']
                                for r in per_seed},
        'csv_path':            os.path.relpath(csv_path, out_dir),
        'provenance':          prov.to_dict(),
        'reader':              reader,
        'calibration_summary': {
            'n_invoices':              params.n_invoices,
            'n_unique_products':       params.n_unique_products,
            'n_unique_customers':      params.n_unique_customers,
            'n_unique_categories':     params.n_unique_categories,
            'span_seconds':            params.span_seconds,
            'arrivals_per_hour':       params.arrivals_per_hour,
            'visitors_per_hour':       params.visitors_per_hour,
            'assumed_conversion_rate': params.assumed_conversion_rate,
            'conversion_rate_source':  'assumption',
            'currency':                params.currency,
        },
        'store':                   store_block,
        'seed_design':             asdict(design),
        'results':                 results,
        'evaluation_counts':       eval_counts,
        'final_evaluation_counts': final_eval_counts,
        'search_start':            starts,
        'sa_schedule':             sa_schedule,
        'design':                  design_block,
    }, provenance=prov_run)

    with open(os.path.join(out_dir, 'summary.json'), 'w',
              encoding='utf-8') as f:
        json.dump({
            'experiment':              'real_data_uci_search_seeds',
            'results':                 results,
            'evaluation_counts':       eval_counts,
            'final_evaluation_counts': final_eval_counts,
            'search_start':            starts,
            'sa_schedule':             sa_schedule,
            'design':                  design_block,
            'store':                   store_block,
        }, f, indent=2)

    print(f"\nArtifacts in: {out_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
