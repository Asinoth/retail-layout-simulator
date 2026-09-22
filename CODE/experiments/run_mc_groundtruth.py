"""MC-objective ground-truth regret (audit R4.3).

Reviewer 4 objected that the recovery/regret story (Figure A) is
measured against the *analytical* reference, which optimizes a
closed-form objective only weakly correlated with the Monte Carlo
objective the GA actually maximizes -- so "recovers 98.6%" quietly
measures the wrong objective, and there is no ground-truth optimum for
the MC objective itself.

This experiment supplies one. On a set of small scenarios we
approximate the MC-objective optimum by a **best-known reference**: we
run three independent searches -- random search, simulated annealing,
and the GA -- each at an order of magnitude MORE evaluation budget than
the paper's operating point, then take the best layout any search found
(the normal-budget GA's own layout included) and confirm it under many
fresh seeds. The GA at its *normal* budget is then scored against this
best-known optimum, giving a regret on the MC objective directly (not
the analytical proxy).

If the normal-budget GA lands within a few percent of a 10x-budget,
three-method best-known optimum, the GA is near the MC-optimum at its
operating point -- which is what Figure A could only assert on a
different objective.

The reference is selected on one seed block and both layouts are then
re-estimated on a disjoint one, which keeps the comparison free of the
selection's own winner's-curse bias but leaves regret non-negative only up
to confirmation noise: the summary reports the minimum regret and how many
scenarios came out negative.

    python -m experiments.run_mc_groundtruth --n-scenarios 6 \
        --normal-budget 750 --big-budget 7500 --mc-iters 1000

``--workers N`` runs scenarios in N processes; the CSV is unchanged.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from synthetic_shops import generate_synthetic_shop
from experiments._common import (build_headless_shop, base_params_for,
                                 paired_mc_revenue, run_ga_headless,
                                 chromosome_to_layout, feasible_layout,
                                 make_run_dir, write_sidecar)
from experiments.metaheuristics import (random_search, simulated_annealing,
                                        _final_select, DEFAULT_INITIAL_ACCEPT)

# Candidate labels for the best-known selection, in the order passed.
CANDIDATES = ('random_search_big', 'simulated_annealing_big', 'GA_big',
              'GA_normal')
# GA population size, also the seed block of RS and SA.
POP_SIZE = 30


def _confirm(shop, names, layout, base_params, base_seed, mc_iters,
             mc_days, n_seeds):
    """Mean MC revenue of a layout under ``n_seeds`` fresh shared seeds."""
    vals = [paired_mc_revenue(shop, names, layout, base_params,
                              seed=base_seed + k, mc_iters=mc_iters,
                              mc_days=mc_days) for k in range(n_seeds)]
    return float(np.mean(vals)), layout


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n-scenarios', type=int, default=6)
    p.add_argument('--n-items', type=int, default=8)
    p.add_argument('--normal-budget', type=int, default=750)  # 30 x 25
    p.add_argument('--big-budget', type=int, default=7500)    # 10x
    p.add_argument('--mc-iters', type=int, default=1000)
    p.add_argument('--mc-days', type=int, default=30)
    p.add_argument('--confirm-seeds', type=int, default=15)
    p.add_argument('--sa-initial-accept', type=float,
                   default=DEFAULT_INITIAL_ACCEPT,
                   help="Acceptance probability of a median worsening move "
                        "at the annealer's starting temperature")
    p.add_argument('--workers', type=int, default=1,
                   help="Processes running scenarios in parallel")
    p.add_argument('--out-root', type=str,
                   default=os.path.join(_HERE, 'results'))
    args = p.parse_args()
    if args.workers < 1:
        p.error("--workers must be >= 1")
    if not 0.0 < args.sa_initial_accept < 1.0:
        p.error("--sa-initial-accept must lie strictly between 0 and 1")
    # Seed layout per scenario idx: normal GA idx*1000 + [0, gens + 6),
    # RS/SA/big GA (1000|2000|3000 + idx)*1000 + [0, big_gens + 6), best-known
    # selection 900_000 + idx*1000 + [0, confirm_seeds) and confirmation
    # 500 above that. These bounds keep the ranges of one scenario disjoint.
    if (args.big_budget // POP_SIZE + 6 > 1000
            or args.normal_budget // POP_SIZE + 6 > 1000
            or args.confirm_seeds > 500):
        p.error(f"--big-budget/{POP_SIZE} + 6 and --normal-budget/{POP_SIZE} "
                "+ 6 must be <= 1000 and --confirm-seeds <= 500 so the "
                "search, selection and confirmation seeds stay disjoint")
    return args


def one_scenario(idx: int, args) -> Dict:
    shop_synth = generate_synthetic_shop(
        name=f"mcgt_{idx:03d}", seed=20_000 + idx,
        n_items=args.n_items, width=12.0, height=10.0)
    shop = build_headless_shop(shop_synth)
    names = [it.name for it in shop_synth.items]
    bp = base_params_for(shop_synth)
    # Both GA runs start from the as-built layout, so the big GA does not
    # inherit whatever layout the other searches scored last.
    init_layout = {n: tuple(shop.floors[1]['items'][n]['position'])
                   for n in names}

    # GA at NORMAL budget (the paper's operating point). pop 30 x gens 25.
    pop = POP_SIZE
    gens = max(1, args.normal_budget // pop)
    ga_out = run_ga_headless(shop, names, bp, pop_size=pop, n_gens=gens,
                             mut_rate=0.18, elite_frac=0.20,
                             mc_iters=args.mc_iters, mc_days=args.mc_days,
                             rng_seed=idx, init_layout=init_layout)
    ga_layout = chromosome_to_layout(ga_out['best_chrom'], names)

    # BIG-budget searches (10x): random, SA, and a big GA. RS and SA get
    # exactly the big GA's search budget (big_gens * pop) and its seed
    # block, so all three spend the same number of evaluations.
    big_gens = max(1, args.big_budget // pop)
    rs_stats: Dict = {}
    rs_big = random_search(shop, shop_synth, names, bp, seed=1000 + idx,
                           budget=big_gens * pop, block=pop,
                           mc_iters=args.mc_iters, mc_days=args.mc_days,
                           stats=rs_stats)
    sa_stats: Dict = {}
    sa_big = simulated_annealing(shop, shop_synth, names, bp, seed=2000 + idx,
                                 budget=big_gens * pop, block=pop,
                                 mc_iters=args.mc_iters, mc_days=args.mc_days,
                                 initial_accept=args.sa_initial_accept,
                                 stats=sa_stats)
    ga_big_out = run_ga_headless(shop, names, bp, pop_size=pop,
                                 n_gens=big_gens, mut_rate=0.18,
                                 elite_frac=0.20, mc_iters=args.mc_iters,
                                 mc_days=args.mc_days, rng_seed=3000 + idx,
                                 init_layout=init_layout)
    ga_big = chromosome_to_layout(ga_big_out['best_chrom'], names)

    # Equal budget is checked on the SEARCH evaluations: SA's final stage
    # ranks distinct archived states, so a search that revisits a state
    # hands fewer candidates to its final seeds than the GA's full
    # population, without having searched any less. The final-selection
    # evaluations are recorded beside the search counts.
    counts = {'random_search_big': rs_stats['n_search_evals'],
              'simulated_annealing_big': sa_stats['n_search_evals'],
              'GA_big': ga_big_out['n_search_evals']}
    if len(set(counts.values())) != 1:
        raise AssertionError(f"scenario {idx}: big searches spent unequal "
                             f"search-evaluation budgets {counts}")
    counts['GA_normal'] = ga_out['n_search_evals']
    final_counts = {'random_search_big': rs_stats['n_final_evals'],
                    'simulated_annealing_big': sa_stats['n_final_evals'],
                    'GA_big': ga_big_out['n_final_evals'],
                    'GA_normal': ga_out['n_final_evals']}

    # Best-known = best of every layout found, the normal-budget GA's
    # included, so the reference is never worse on the selection seeds than
    # a layout already in hand. Confirmed under many seeds. RS/SA outputs
    # are fixed points of the repair; mapping them keeps the rule uniform.
    candidates = [feasible_layout(shop, names, rs_big),
                  feasible_layout(shop, names, sa_big),
                  ga_big, ga_layout]
    cseed = 900_000 + idx * 1000
    best_known = _final_select(shop, names, candidates, bp,
                               cseed, args.mc_iters, args.mc_days,
                               n_final_seeds=args.confirm_seeds)
    best_source = CANDIDATES[next(i for i, c in enumerate(candidates)
                                  if c is best_known)]
    bk_mean, _ = _confirm(shop, names, best_known, bp, cseed + 500,
                          args.mc_iters, args.mc_days, args.confirm_seeds)
    ga_mean, _ = _confirm(shop, names, ga_layout, bp, cseed + 500,
                          args.mc_iters, args.mc_days, args.confirm_seeds)

    regret_pct = (bk_mean - ga_mean) / max(abs(bk_mean), 1e-9) * 100.0
    print(f"[mcgt {idx:>2d}] GA(normal)={ga_mean:>10.1f}  "
          f"best-known(10x,3-method)={bk_mean:>10.1f} [{best_source}]  "
          f"MC-regret={regret_pct:+.2f}%", flush=True)
    return {'scenario': idx, 'ga_normal_mc': ga_mean,
            'best_known_mc': bk_mean, 'mc_regret_pct': regret_pct,
            'best_known_source': best_source, 'evaluation_counts': counts,
            'final_evaluation_counts': final_counts,
            'sa_T0': sa_stats.get('sa_T0')}


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


def main():
    args = parse_args()
    out_dir = make_run_dir(args.out_root, 'mc_groundtruth')
    print(f"output dir: {out_dir}", flush=True)
    t0 = time.perf_counter()
    rows: List[Dict] = _map_scenarios(one_scenario, args.n_scenarios,
                                      args.workers, args)
    wall = time.perf_counter() - t0

    reg = np.array([r['mc_regret_pct'] for r in rows])
    path = os.path.join(out_dir, 'results.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['scenario', 'ga_normal_mc', 'best_known_mc',
                    'mc_regret_pct', 'best_known_source'])
        for r in rows:
            w.writerow([r['scenario'], f"{r['ga_normal_mc']:.4f}",
                        f"{r['best_known_mc']:.4f}",
                        f"{r['mc_regret_pct']:.4f}",
                        r['best_known_source']])

    summary = {
        'mc_regret_median_pct': float(np.median(reg)),
        'mc_regret_mean_pct': float(np.mean(reg)),
        'mc_regret_max_pct': float(np.max(reg)),
        # The best-known reference is chosen on the selection seeds and both
        # layouts are then re-estimated on disjoint confirmation seeds, so a
        # scenario whose reference came from a big-budget search can confirm
        # below the normal-budget GA. Regret is therefore non-negative only
        # up to confirmation noise; the minimum and the count of negative
        # scenarios say how often that happened.
        'mc_regret_min_pct': float(np.min(reg)),
        'n_negative_regret': int(np.sum(reg < 0.0)),
        # Scenarios where no 10x search beat the normal-budget GA on the
        # selection seeds: their regret is 0 because the GA layout itself
        # is the best-known reference.
        'n_best_known_is_ga_normal': int(sum(
            r['best_known_source'] == 'GA_normal' for r in rows)),
        'n_scenarios': args.n_scenarios,
        'normal_budget': args.normal_budget,
        'big_budget': args.big_budget,
        'mc_iters': args.mc_iters,
    }
    write_sidecar(out_dir, {'experiment': 'mc_groundtruth',
                            'args': vars(args), 'wall_seconds': wall,
                            'workers': args.workers,
                            'evaluation_counts': {
                                m: sorted({r['evaluation_counts'][m]
                                           for r in rows})
                                for m in CANDIDATES},
                            'final_evaluation_counts': {
                                m: sorted({r['final_evaluation_counts'][m]
                                           for r in rows})
                                for m in CANDIDATES},
                            # The big annealer's starting temperature is
                            # calibrated per scenario from its own first
                            # block, so each one is recorded.
                            'sa_schedule': {
                                'sa_initial_accept': args.sa_initial_accept,
                                'sa_T0': [{'scenario': r['scenario'],
                                           'sa_T0': r['sa_T0']}
                                          for r in rows]},
                            'summary': summary})
    import json
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nMC-objective regret vs 10x best-known: "
          f"median={np.median(reg):+.2f}%  mean={np.mean(reg):+.2f}%  "
          f"min={np.min(reg):+.2f}%  max={np.max(reg):+.2f}%  "
          f"({summary['n_negative_regret']} of {len(reg)} scenarios "
          f"negative)   (wall {wall:.0f}s)")
    print(f"Artifacts in: {out_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
