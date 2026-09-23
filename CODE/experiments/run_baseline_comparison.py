"""Figure B: GA vs informed baselines (paired-MC, bootstrap CI).

For each of N synthetic shops we evaluate, under FIXED paired-MC seeds:
  * random_valid       (Larson rejection baseline)
  * perimeter_only     (Larson 2005 racetrack heuristic)
  * popularity_rank    (dataset_layout's default heuristic)
  * greedy_swap        (cheap local search from popularity_rank)
  * random_search      (best of the GA's search budget of random layouts)
  * simulated_annealing (Metropolis search at the GA's search budget)
  * GA (the simulator-backed genetic optimizer)
  * Oracle (analytical-optimum layout from oracle.solve_oracle)

The three searches -- GA, random_search and simulated_annealing -- start
from the same layout, the as-built one, and spend the same search budget,
so a difference between them measures the search rather than the seed it
was handed. One sensitivity row is scored beside them:
  * simulated_annealing_popstart (the same annealer, warm-started from
                                  popularity_rank as it was before the
                                  searches shared a start)
It gets the same budget, block and seeds, is scored under the same
held-out paired seeds and is written to results.csv like any method, but
it is neither one of the comparisons the headline family corrects for nor
in the equal-budget assertion set. ``--no-sa-popstart`` skips it. The
sidecar and summary.json name both sets (``comparison_family``,
``sensitivity_methods``).

The headline number is the GA's MC revenue minus the best non-GA baseline
(popularity_rank or greedy_swap, whichever is higher per scenario),
with a bootstrap 95% CI on the paired difference across scenarios x
seeds. Artefacts:
  * results.csv       -- one row per (scenario, method, seed)
  * methods_bar.png   -- mean MC revenue per method with 95% CI whiskers
  * sidecar.json      -- run metadata
  * summary.json      -- comparison family, sensitivity rows and their
                         paired differences, written last

Important constraint: every method is evaluated with the SAME mc seed
on the SAME scenario, so the comparison is a paired difference (not an
independent-samples test). This removes MC noise from the comparison.
The evaluation seed is held out: it lies outside every seed the GA,
random search and SA use while searching. Every non-GA layout is mapped
through ``feasible_layout`` (the GA's repair chain) before it is scored,
so all methods are compared on the same feasible set.

Smoke:
    python -m experiments.run_baseline_comparison --n-scenarios 3 \
        --n-seeds 2 --mc-iters 200 --n-gens 8 --pop-size 16

Paper-grade:
    python -m experiments.run_baseline_comparison --n-scenarios 30 \
        --n-seeds 10 --mc-iters 2000 --n-gens 25

``--workers N`` runs scenarios in N processes; the CSV is unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from synthetic_shops import generate_synthetic_shop, analytical_revenue
from oracle import solve_oracle
from baselines import (
    random_valid, perimeter_only, popularity_rank, greedy_swap,
    assert_layout_valid, assert_no_strict_overlap,
)
from experiments._common import (
    build_headless_shop, base_params_for,
    paired_mc_revenue, run_ga_headless, chromosome_to_layout,
    feasible_layout, bootstrap_ci, make_run_dir, write_sidecar,
)
from experiments.metaheuristics import (random_search, simulated_annealing,
                                        DEFAULT_INITIAL_ACCEPT)


METHODS = [
    "random_valid",
    "perimeter_only",
    "popularity_rank",
    "greedy_swap",
    "random_search",
    "simulated_annealing",
    "simulated_annealing_popstart",
    "GA",
    "oracle",
]

#: Rows scored beside the comparison but outside it: not in the family of
#: GA-minus-X comparisons the headline corrects for, and not in the
#: equal-budget assertion set. ``simulated_annealing_popstart`` is the
#: annealer warm-started from ``popularity_rank``; set beside
#: ``simulated_annealing`` it shows how much of the annealer's result that
#: warm start carried.
SENSITIVITY_METHODS = ["simulated_annealing_popstart"]

#: The comparators whose GA-minus-X paired differences form the family.
COMPARISON_FAMILY = [m for m in METHODS
                     if m != "GA" and m not in SENSITIVITY_METHODS]

#: The searches held to one search-evaluation budget and one start.
EQUAL_BUDGET_METHODS = ("GA", "random_search", "simulated_annealing")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--n-scenarios', type=int, default=30)
    p.add_argument('--n-seeds', type=int, default=10)
    p.add_argument('--n-items', type=int, default=10)
    p.add_argument('--mc-iters', type=int, default=2000)
    p.add_argument('--mc-days', type=int, default=30)
    p.add_argument('--n-gens', type=int, default=25)
    p.add_argument('--pop-size', type=int, default=30)
    p.add_argument('--sa-initial-accept', type=float,
                   default=DEFAULT_INITIAL_ACCEPT,
                   help="Acceptance probability of a median worsening move "
                        "at the annealer's starting temperature")
    p.add_argument('--no-sa-popstart', dest='sa_popstart',
                   action='store_false',
                   help="Skip the simulated_annealing_popstart sensitivity "
                        "row (the annealer warm-started from "
                        "popularity_rank)")
    p.add_argument('--workers', type=int, default=1,
                   help="Processes running scenarios in parallel")
    p.add_argument('--out-root', type=str,
                   default=os.path.join(_HERE, 'results'))
    args = p.parse_args()
    if args.workers < 1:
        p.error("--workers must be >= 1")
    if not 0.0 < args.sa_initial_accept < 1.0:
        p.error("--sa-initial-accept must lie strictly between 0 and 1")
    # Search seeds of run seed s are s*1000 + [0, n_gens + 6) (generation or
    # block seeds, a gap, then 5 final-selection seeds). These bounds keep
    # each run's range clear of the next and all of them below the held-out
    # evaluation seeds of ``eval_seed``.
    if args.n_seeds > 1000 or args.n_gens + 6 > 1000:
        p.error("--n-seeds and --n-gens + 6 must both be <= 1000 so search "
                "and evaluation seeds stay disjoint")
    return args


def eval_seed(scenario_idx: int, seed: int) -> int:
    """Held-out MC seed for the paired evaluation of one (scenario, seed).

    Starts at 1_000_000, above every search seed (see ``parse_args``), so no
    method has scored or selected a candidate under the noise draw it is
    finally compared on."""
    return 1_000_000 + 1000 * scenario_idx + seed


def evaluate_methods_one_scenario(scenario_idx: int,
                                  seeds: List[int],
                                  args: argparse.Namespace
                                  ) -> Tuple[List[Dict], List[Dict]]:
    """Returns the flat list of per-(method, seed) records for this
    scenario, and per seed the fitness evaluations each search method spent.

    All MC evaluations share the same mc_seed -- the comparison is the
    PAIRED difference between method outputs."""
    shop_synth = generate_synthetic_shop(
        name=f"synth_{scenario_idx:03d}",
        seed=10_000 + scenario_idx,
        n_items=args.n_items,
        width=12.0, height=10.0,
    )
    shop = build_headless_shop(shop_synth)
    item_names = [it.name for it in shop_synth.items]
    base_params = base_params_for(shop_synth)
    # Every GA run starts from the as-built layout, whatever positions
    # earlier calls on this shop have left behind.
    init_layout = {n: tuple(shop.floors[1]['items'][n]['position'])
                   for n in item_names}

    print(f"\n[scenario {scenario_idx:>3d}] shop dims "
          f"{shop_synth.width:.1f}x{shop_synth.height:.1f}, "
          f"{len(shop_synth.items)} items", flush=True)

    # Pre-compute deterministic layouts (independent of seed).
    pre_layouts = {
        'random_valid':    random_valid(shop_synth, seed=scenario_idx),
        'perimeter_only':  perimeter_only(shop_synth, seed=scenario_idx),
        'popularity_rank': popularity_rank(shop_synth, seed=scenario_idx),
        'greedy_swap':     greedy_swap(shop_synth, seed=scenario_idx),
    }
    # Oracle layout
    oracle_t0 = time.perf_counter()
    oracle_result = solve_oracle(shop_synth, n_restarts=24)
    pre_layouts['oracle'] = oracle_result.layout
    oracle_wall = time.perf_counter() - oracle_t0
    # Score every method on the GA's feasible set: GA candidates always go
    # through its repair (zone clipping, impulse projection, overlap
    # resolution), so an unrepaired layout would be judged on different
    # rules.
    pre_layouts = {name: feasible_layout(shop, item_names, lay)
                   for name, lay in pre_layouts.items()}

    # Validate the layouts that are scored (raise if any are bad). The
    # simulator fitness penalizes any positive overlap, so the reference
    # also gets the zero-tolerance check: solver contact residue would
    # otherwise pass the tolerant check and still collapse its revenue.
    for name, lay in pre_layouts.items():
        assert_layout_valid(shop_synth, lay)
        if name == 'oracle':
            assert_no_strict_overlap(shop_synth, lay)

    # For each seed, run the GA fresh; baselines re-use their layout
    # but get re-evaluated against the same mc_seed for paired comparison.
    records: List[Dict] = []
    eval_counts: List[Dict] = []
    for seed in seeds:
        t_seed = time.perf_counter()
        # GA run (fresh per seed)
        ga_t0 = time.perf_counter()
        ga_out = run_ga_headless(
            shop, item_names, base_params,
            pop_size=args.pop_size,
            n_gens=args.n_gens,
            mut_rate=0.18,
            elite_frac=0.20,
            mc_iters=args.mc_iters,
            mc_days=args.mc_days,
            rng_seed=seed,
            init_layout=init_layout,
        )
        ga_layout = chromosome_to_layout(ga_out['best_chrom'], item_names)
        ga_wall = time.perf_counter() - ga_t0

        # Equal-budget metaheuristic comparators: each searches the SAME
        # MC objective under the GA's evaluation budget (pop_size * n_gens
        # search evaluations with one shared seed per pop_size block, then
        # pop_size candidates under the GA's final selection seeds), and
        # each starts from the as-built layout the GA's population is
        # seeded from, so no search is handed a better seed than the
        # others. Run fresh per seed, like the GA.
        budget = args.pop_size * args.n_gens
        rs_stats: Dict = {}
        rs_t0 = time.perf_counter()
        rs_layout = random_search(
            shop, shop_synth, item_names, base_params,
            seed=seed, budget=budget, block=args.pop_size,
            mc_iters=args.mc_iters, mc_days=args.mc_days, stats=rs_stats,
            init_layout=init_layout)
        rs_wall = time.perf_counter() - rs_t0
        sa_stats: Dict = {}
        sa_t0 = time.perf_counter()
        sa_layout = simulated_annealing(
            shop, shop_synth, item_names, base_params,
            seed=seed, budget=budget, block=args.pop_size,
            mc_iters=args.mc_iters, mc_days=args.mc_days,
            initial_accept=args.sa_initial_accept, stats=sa_stats,
            init_layout=init_layout, start='asbuilt')
        sa_wall = time.perf_counter() - sa_t0
        # Sensitivity row: the same annealer, budget, block and seeds, but
        # warm-started from popularity_rank. It is scored with the others
        # and kept out of the equal-budget set and the comparison family.
        sp_stats: Dict = {}
        sp_wall = 0.0
        if args.sa_popstart:
            sp_t0 = time.perf_counter()
            sp_layout = simulated_annealing(
                shop, shop_synth, item_names, base_params,
                seed=seed, budget=budget, block=args.pop_size,
                mc_iters=args.mc_iters, mc_days=args.mc_days,
                initial_accept=args.sa_initial_accept, stats=sp_stats,
                start='popularity')
            sp_wall = time.perf_counter() - sp_t0

        # The equal-budget check is on the SEARCH evaluations, which is what
        # the budget argument buys. The final selection stage is not held to
        # equality: SA ranks distinct archived states, so a search that
        # revisits a state hands fewer than pop_size candidates to the final
        # seeds, while the GA always re-evaluates a full population. Both
        # counts are recorded so the difference stays visible.
        counts = {'GA': ga_out['n_search_evals'],
                  'random_search': rs_stats['n_search_evals'],
                  'simulated_annealing': sa_stats['n_search_evals']}
        if len(set(counts.values())) != 1:
            raise AssertionError(
                f"scenario {scenario_idx} seed {seed}: search methods spent "
                f"unequal search-evaluation budgets {counts}")
        # The start each search reported. The GA is handed init_layout
        # above; the other two say which start they actually used.
        starts = {'GA': 'asbuilt',
                  'random_search': rs_stats['rs_start'],
                  'simulated_annealing': sa_stats['sa_start']}
        if set(starts.values()) != {'asbuilt'}:
            raise AssertionError(
                f"scenario {scenario_idx} seed {seed}: the searches did not "
                f"all start from the as-built layout {starts}")
        count_record = {
            'scenario': scenario_idx, 'seed': seed, **counts,
            'final_evals': {'GA': ga_out['n_final_evals'],
                            'random_search': rs_stats['n_final_evals'],
                            'simulated_annealing': sa_stats['n_final_evals']},
            'sa_T0': sa_stats.get('sa_T0'),
            'sa_initial_accept': sa_stats.get('sa_initial_accept'),
            'search_start': starts,
            'sensitivity': {},
        }
        if args.sa_popstart:
            # Held to the budget on its own; it is not part of the
            # equal-budget set above.
            if sp_stats['n_search_evals'] != budget:
                raise AssertionError(
                    f"scenario {scenario_idx} seed {seed}: "
                    f"simulated_annealing_popstart spent "
                    f"{sp_stats['n_search_evals']} search evaluations, "
                    f"budget {budget}")
            count_record['sensitivity']['simulated_annealing_popstart'] = {
                'n_search_evals': sp_stats['n_search_evals'],
                'n_final_evals': sp_stats['n_final_evals'],
                'sa_T0': sp_stats.get('sa_T0'),
                'sa_initial_accept': sp_stats.get('sa_initial_accept'),
                'start': sp_stats['sa_start'],
            }
        eval_counts.append(count_record)

        # Paired-MC evaluation: same held-out mc_seed for all methods on
        # this scenario+seed.
        mc_seed = eval_seed(scenario_idx, seed)
        all_layouts = dict(pre_layouts)
        # Both searchers only evaluate repaired candidates, so these are
        # fixed points of the repair; mapping them keeps the rule uniform.
        all_layouts['random_search'] = feasible_layout(shop, item_names,
                                                       rs_layout)
        all_layouts['simulated_annealing'] = feasible_layout(shop, item_names,
                                                             sa_layout)
        if args.sa_popstart:
            all_layouts['simulated_annealing_popstart'] = feasible_layout(
                shop, item_names, sp_layout)
        all_layouts['GA'] = ga_layout

        for method, lay in all_layouts.items():
            mc_rev = paired_mc_revenue(
                shop, item_names, lay, base_params,
                seed=mc_seed,
                mc_iters=args.mc_iters,
                mc_days=args.mc_days,
            )
            true_R = analytical_revenue(shop_synth, lay)
            records.append({
                'scenario': scenario_idx,
                'seed':     seed,
                'method':   method,
                'mc_revenue': mc_rev,
                'analytical_R': true_R,
            })

        seed_wall = time.perf_counter() - t_seed
        sp_note = f", SA-popstart {sp_wall:.1f}s" if args.sa_popstart else ""
        # Print compact per-seed summary
        ga_mc = next(r['mc_revenue']
                     for r in records[-len(all_layouts):]
                     if r['method'] == 'GA')
        best_baseline = max(
            (r for r in records[-len(all_layouts):]
             if r['method'] in {'popularity_rank', 'greedy_swap',
                                'perimeter_only', 'random_valid',
                                'random_search', 'simulated_annealing'}),
            key=lambda r: r['mc_revenue'],
        )
        print(f"  seed {seed:>3d}: GA mc={ga_mc:>10.2f}  "
              f"vs best non-GA ({best_baseline['method']}) "
              f"= {best_baseline['mc_revenue']:>10.2f}  "
              f"diff={ga_mc - best_baseline['mc_revenue']:+10.2f}  "
              f"({seed_wall:.1f}s: GA {ga_wall:.1f}s, "
              f"RS {rs_wall:.1f}s, SA {sa_wall:.1f}s{sp_note})",
              flush=True)
    return records, eval_counts


def _map_scenarios(fn, n_scenarios: int, workers: int, *fn_args) -> list:
    """``[fn(s, *fn_args) for s in range(n_scenarios)]``, in scenario order.

    With ``workers`` > 1 the scenarios run in separate processes. They are
    independent (each builds its own shop and reseeds every RNG it draws
    from), and results are put back in scenario order, so the output is
    identical to the serial run."""
    if workers <= 1:
        return [fn(s, *fn_args) for s in range(n_scenarios)]
    done = {}
    pool = ProcessPoolExecutor(max_workers=workers)
    try:
        futures = {pool.submit(fn, s, *fn_args): s for s in range(n_scenarios)}
        for fut in as_completed(futures):
            done[futures[fut]] = fut.result()
    except BaseException:
        # Leaving the pool's context manager would wait for every queued
        # scenario first, so a scenario that fails minutes into a run of
        # hours would only report at the end. Drop what has not started and
        # let the error out now.
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return [done[s] for s in sorted(done)]


def write_results_csv(out_dir: str, all_records: List[Dict]) -> str:
    path = os.path.join(out_dir, 'results.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['scenario', 'seed', 'method', 'mc_revenue', 'analytical_R'])
        for r in all_records:
            w.writerow([r['scenario'], r['seed'], r['method'],
                        f"{r['mc_revenue']:.4f}",
                        f"{r['analytical_R']:.4f}"])
    return path


def aggregate_per_method(all_records: List[Dict]
                         ) -> Dict[str, Dict[str, float]]:
    """Mean MC revenue per method + bootstrap CI (over scenarios x seeds)."""
    out = {}
    for method in METHODS:
        samples = np.array([r['mc_revenue'] for r in all_records
                            if r['method'] == method])
        if samples.size == 0:
            continue
        mean, lo, hi = bootstrap_ci(samples, alpha=0.05, n_boot=2000)
        out[method] = {'mean': mean, 'ci_lo': lo, 'ci_hi': hi,
                       'n': int(samples.size)}
    return out


def aggregate_paired_diffs(all_records: List[Dict]
                           ) -> Dict[str, Dict[str, float]]:
    """Per-method paired difference of (GA - method) across all
    (scenario, seed) pairs. Bootstrap CI on the mean difference."""
    # Index by (scenario, seed, method) -> mc_revenue
    idx: Dict[tuple, float] = {}
    for r in all_records:
        idx[(r['scenario'], r['seed'], r['method'])] = r['mc_revenue']
    keys_ga = [(s, sd) for (s, sd, m) in idx if m == 'GA']
    out = {}
    for method in METHODS:
        if method == 'GA':
            continue
        diffs = []
        for s, sd in keys_ga:
            if (s, sd, method) in idx:
                diffs.append(idx[(s, sd, 'GA')] - idx[(s, sd, method)])
        if not diffs:
            continue
        arr = np.array(diffs)
        mean, lo, hi = bootstrap_ci(arr, alpha=0.05, n_boot=2000)
        out[method] = {'mean_diff_GA_minus_X': mean,
                       'ci_lo': lo, 'ci_hi': hi,
                       'n': int(arr.size)}
    return out


def sa_start_effect(all_records: List[Dict]) -> Dict[str, float]:
    """Paired difference (popularity-started SA - as-built SA) over every
    (scenario, seed) pair that has both, with a bootstrap CI: what the
    popularity warm start was worth to the annealer at equal budget and
    seeds. Empty when the sensitivity row was not run."""
    idx = {(r['scenario'], r['seed'], r['method']): r['mc_revenue']
           for r in all_records}
    diffs = [idx[(s, sd, 'simulated_annealing_popstart')]
             - idx[(s, sd, 'simulated_annealing')]
             for (s, sd, m) in idx
             if m == 'simulated_annealing_popstart'
             and (s, sd, 'simulated_annealing') in idx]
    if not diffs:
        return {}
    mean, lo, hi = bootstrap_ci(np.array(diffs), alpha=0.05, n_boot=2000)
    return {'mean_diff_popstart_minus_asbuilt': mean,
            'ci_lo': lo, 'ci_hi': hi, 'n': len(diffs)}


def make_bar_chart(out_dir: str,
                   per_method: Dict[str, Dict[str, float]]) -> str:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    methods = [m for m in METHODS if m in per_method]
    means = [per_method[m]['mean'] for m in methods]
    lo = [per_method[m]['ci_lo'] for m in methods]
    hi = [per_method[m]['ci_hi'] for m in methods]
    err_lo = [means[i] - lo[i] for i in range(len(methods))]
    err_hi = [hi[i] - means[i] for i in range(len(methods))]

    colors = []
    for m in methods:
        if m == 'GA':       colors.append('#FF6B6B')
        elif m == 'oracle': colors.append('#FFD166')
        elif m in SENSITIVITY_METHODS: colors.append('#BBBBBB')
        else:               colors.append('#4ECDC4')

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(methods))
    ax.bar(x, means, yerr=[err_lo, err_hi], color=colors,
           alpha=0.85, capsize=6, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha='right')
    ax.set_ylabel('MC mean revenue (paired-MC, 95% CI)')
    ax.set_title('Figure B. Layout method comparison '
                 f'(N = {per_method[methods[0]]["n"]} paired evaluations)')
    fig.tight_layout()
    path = os.path.join(out_dir, 'methods_bar.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> int:
    args = parse_args()
    out_dir = make_run_dir(args.out_root, 'baseline_comparison')
    print(f"output dir: {out_dir}", flush=True)

    seeds = list(range(args.n_seeds))
    wall_t0 = time.perf_counter()

    all_records: List[Dict] = []
    all_counts: List[Dict] = []
    for records, counts in _map_scenarios(evaluate_methods_one_scenario,
                                          args.n_scenarios, args.workers,
                                          seeds, args):
        all_records.extend(records)
        all_counts.extend(counts)

    wall = time.perf_counter() - wall_t0
    print(f"\nTotal wall: {wall:.1f}s  ({len(all_records)} records)",
          flush=True)

    csv_path = write_results_csv(out_dir, all_records)
    per_method = aggregate_per_method(all_records)
    diffs = aggregate_paired_diffs(all_records)
    png_path = make_bar_chart(out_dir, per_method)

    print("\nPer-method MC revenue (95% bootstrap CI):")
    for method in METHODS:
        if method not in per_method:
            continue
        d = per_method[method]
        print(f"  {method:18s}  mean={d['mean']:>10.2f}  "
              f"CI=[{d['ci_lo']:>10.2f}, {d['ci_hi']:>10.2f}]  "
              f"(n={d['n']})")

    print("\nPaired difference (GA - X), 95% bootstrap CI:")
    for method in METHODS:
        if method not in diffs:
            continue
        d = diffs[method]
        sig = '  <-- significant' if d['ci_lo'] > 0 or d['ci_hi'] < 0 else ''
        tag = '  [sensitivity]' if method in SENSITIVITY_METHODS else ''
        print(f"  GA - {method:15s}  mean={d['mean_diff_GA_minus_X']:>+10.2f}  "
              f"CI=[{d['ci_lo']:>+10.2f}, {d['ci_hi']:>+10.2f}]"
              f"{sig}{tag}")
    start_effect = sa_start_effect(all_records)
    if start_effect:
        se = start_effect
        print(f"\nSA start effect (popstart - as-built): "
              f"mean={se['mean_diff_popstart_minus_asbuilt']:>+10.2f}  "
              f"CI=[{se['ci_lo']:>+10.2f}, {se['ci_hi']:>+10.2f}]  "
              f"(n={se['n']})")

    # The rows this run actually contains in each role. The sensitivity
    # rows are absent under --no-sa-popstart and are never part of the
    # family, whatever the run contains.
    sensitivity = [m for m in SENSITIVITY_METHODS if m in per_method]
    # The start every search reported, over all runs; each run already
    # held the equal-budget searches to the as-built start.
    starts: Dict[str, set] = {}
    for c in all_counts:
        for m, s in c['search_start'].items():
            starts.setdefault(m, set()).add(s)
        for m, rec in c['sensitivity'].items():
            starts.setdefault(m, set()).add(rec['start'])
    search_start = {m: '/'.join(sorted(v)) for m, v in starts.items()}
    summary = {
        'comparison_family': list(COMPARISON_FAMILY),
        'sensitivity_methods': sensitivity,
        'equal_budget_methods': list(EQUAL_BUDGET_METHODS),
        'search_start': search_start,
        'paired_diff_GA_minus_X': {m: diffs[m] for m in COMPARISON_FAMILY
                                   if m in diffs},
        'sensitivity_paired_diff_GA_minus_X': {m: diffs[m]
                                               for m in sensitivity
                                               if m in diffs},
        'sa_start_effect': start_effect,
        'n_scenarios': args.n_scenarios,
        'n_seeds_per_scenario': args.n_seeds,
    }

    write_sidecar(out_dir, {
        'experiment': 'baseline_comparison',
        'args': vars(args),
        'wall_seconds': wall,
        'csv_path': os.path.relpath(csv_path, out_dir),
        'figure_path': os.path.relpath(png_path, out_dir),
        'per_method': per_method,
        'paired_diff_GA_minus_X': diffs,
        'n_scenarios': args.n_scenarios,
        'n_seeds_per_scenario': args.n_seeds,
        'workers': args.workers,
        'paired_eval_seed': '1_000_000 + 1000*scenario + seed',
        # Distinct per-run search-evaluation totals of each search method
        # (checked equal within every run), and the final-selection
        # evaluations each spent on top of them.
        'evaluation_counts': {
            m: sorted({c[m] for c in all_counts})
            for m in EQUAL_BUDGET_METHODS
        },
        'final_evaluation_counts': {
            m: sorted({c['final_evals'][m] for c in all_counts})
            for m in EQUAL_BUDGET_METHODS
        },
        # Which rows form the comparison family and which are sensitivity
        # rows scored beside it, and the start every search used.
        'comparison_family': summary['comparison_family'],
        'sensitivity_methods': summary['sensitivity_methods'],
        'equal_budget_methods': summary['equal_budget_methods'],
        'search_start': summary['search_start'],
        # The sensitivity rows' own counts, kept apart from the equal-budget
        # record above so they cannot be read as part of it.
        'sensitivity_evaluation_counts': {
            m: sorted({c['sensitivity'][m]['n_search_evals']
                       for c in all_counts})
            for m in sensitivity
        },
        'sensitivity_final_evaluation_counts': {
            m: sorted({c['sensitivity'][m]['n_final_evals']
                       for c in all_counts})
            for m in sensitivity
        },
        # The annealing schedule each run actually used: T0 is calibrated
        # per run from its own first block, so it is recorded per run
        # rather than assumed constant.
        'sa_schedule': {
            'sa_initial_accept': args.sa_initial_accept,
            'sa_start': 'asbuilt',
            'sa_T0': [{'scenario': c['scenario'], 'seed': c['seed'],
                       'sa_T0': c['sa_T0']} for c in all_counts],
        },
        # The popularity-started annealer calibrates its own T0 from its
        # own first block, so its schedule is recorded separately.
        'sa_popstart_schedule': ({
            'sa_initial_accept': args.sa_initial_accept,
            'sa_start': 'popularity',
            'sa_T0': [{'scenario': c['scenario'], 'seed': c['seed'],
                       'sa_T0': c['sensitivity'][
                           'simulated_annealing_popstart']['sa_T0']}
                      for c in all_counts],
        } if 'simulated_annealing_popstart' in sensitivity else None),
        'summary': summary,
    })
    # Written last, so its presence marks a finished run.
    with open(os.path.join(out_dir, 'summary.json'), 'w',
              encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"\nArtifacts in: {out_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
