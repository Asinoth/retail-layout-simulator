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
the paper's operating point, then take the best layout any of them
finds and confirm it under many fresh seeds. The GA at its *normal*
budget is then scored against this best-known optimum, giving a regret
on the MC objective directly (not the analytical proxy).

If the normal-budget GA lands within a few percent of a 10x-budget,
three-method best-known optimum, the GA is near the MC-optimum at its
operating point -- which is what Figure A could only assert on a
different objective.

    python -m experiments.run_mc_groundtruth --n-scenarios 6 \
        --normal-budget 750 --big-budget 7500 --mc-iters 1000
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Dict, List

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from synthetic_shops import generate_synthetic_shop
from experiments._common import (build_headless_shop, base_params_for,
                                 paired_mc_revenue, run_ga_headless,
                                 chromosome_to_layout, make_run_dir,
                                 write_sidecar)
from experiments.metaheuristics import (random_search, simulated_annealing,
                                        _final_select)


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
    p.add_argument('--out-root', type=str,
                   default=os.path.join(_HERE, 'results'))
    return p.parse_args()


def one_scenario(idx: int, args) -> Dict:
    shop_synth = generate_synthetic_shop(
        name=f"mcgt_{idx:03d}", seed=20_000 + idx,
        n_items=args.n_items, width=12.0, height=10.0)
    shop = build_headless_shop(shop_synth)
    names = [it.name for it in shop_synth.items]
    bp = base_params_for(shop_synth)

    # GA at NORMAL budget (the paper's operating point). pop 30 x gens 25.
    pop = 30
    gens = max(1, args.normal_budget // pop)
    ga_out = run_ga_headless(shop, names, bp, pop_size=pop, n_gens=gens,
                             mut_rate=0.18, elite_frac=0.20,
                             mc_iters=args.mc_iters, mc_days=args.mc_days,
                             rng_seed=idx)
    ga_layout = chromosome_to_layout(ga_out['best_chrom'], names)

    # BIG-budget searches (10x): random, SA, and a big GA.
    rs_big = random_search(shop, shop_synth, names, bp, seed=1000 + idx,
                           budget=args.big_budget, mc_iters=args.mc_iters,
                           mc_days=args.mc_days)
    sa_big = simulated_annealing(shop, shop_synth, names, bp, seed=2000 + idx,
                                 budget=args.big_budget, mc_iters=args.mc_iters,
                                 mc_days=args.mc_days)
    big_gens = max(1, args.big_budget // pop)
    ga_big_out = run_ga_headless(shop, names, bp, pop_size=pop,
                                 n_gens=big_gens, mut_rate=0.18,
                                 elite_frac=0.20, mc_iters=args.mc_iters,
                                 mc_days=args.mc_days, rng_seed=3000 + idx)
    ga_big = chromosome_to_layout(ga_big_out['best_chrom'], names)

    # Best-known = best of the three big searches, confirmed under many seeds.
    cseed = 900_000 + idx * 1000
    best_known = _final_select(shop, names, [rs_big, sa_big, ga_big], bp,
                               cseed, args.mc_iters, args.mc_days,
                               n_final_seeds=args.confirm_seeds)
    bk_mean, _ = _confirm(shop, names, best_known, bp, cseed + 500,
                          args.mc_iters, args.mc_days, args.confirm_seeds)
    ga_mean, _ = _confirm(shop, names, ga_layout, bp, cseed + 500,
                          args.mc_iters, args.mc_days, args.confirm_seeds)

    regret_pct = (bk_mean - ga_mean) / max(abs(bk_mean), 1e-9) * 100.0
    print(f"[mcgt {idx:>2d}] GA(normal)={ga_mean:>10.1f}  "
          f"best-known(10x,3-method)={bk_mean:>10.1f}  "
          f"MC-regret={regret_pct:+.2f}%", flush=True)
    return {'scenario': idx, 'ga_normal_mc': ga_mean,
            'best_known_mc': bk_mean, 'mc_regret_pct': regret_pct}


def main():
    args = parse_args()
    out_dir = make_run_dir(args.out_root, 'mc_groundtruth')
    print(f"output dir: {out_dir}", flush=True)
    t0 = time.perf_counter()
    rows: List[Dict] = [one_scenario(i, args) for i in range(args.n_scenarios)]
    wall = time.perf_counter() - t0

    reg = np.array([r['mc_regret_pct'] for r in rows])
    path = os.path.join(out_dir, 'results.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['scenario', 'ga_normal_mc', 'best_known_mc',
                    'mc_regret_pct'])
        for r in rows:
            w.writerow([r['scenario'], f"{r['ga_normal_mc']:.4f}",
                        f"{r['best_known_mc']:.4f}",
                        f"{r['mc_regret_pct']:.4f}"])

    summary = {
        'mc_regret_median_pct': float(np.median(reg)),
        'mc_regret_mean_pct': float(np.mean(reg)),
        'mc_regret_max_pct': float(np.max(reg)),
        'n_scenarios': args.n_scenarios,
        'normal_budget': args.normal_budget,
        'big_budget': args.big_budget,
        'mc_iters': args.mc_iters,
    }
    write_sidecar(out_dir, {'experiment': 'mc_groundtruth',
                            'args': vars(args), 'wall_seconds': wall,
                            'summary': summary})
    import json
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nMC-objective regret vs 10x best-known: "
          f"median={np.median(reg):+.2f}%  mean={np.mean(reg):+.2f}%  "
          f"max={np.max(reg):+.2f}%   (wall {wall:.0f}s)")
    print(f"Artifacts in: {out_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
