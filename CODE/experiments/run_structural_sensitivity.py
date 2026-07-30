"""Structural (micro-rule) sensitivity sweep (audit R6.5).

Reviewer 6 noted that the paper's sensitivity work is purely parametric
(elasticity bands, GA hyperparameters) and never varies a behavioral
RULE. Micro-rule choices can dominate macro outcomes in an ABM. This
sweeps one structural knob -- the anti-congestion separation strength
that governs how strongly agents avoid stacking -- and reports how much
the macro outcome (live revenue / conversion over a fixed window) moves.

A small movement demonstrates the macro result is not an artifact of the
particular separation strength; a large one would be a warning. Either
way it converts an unexamined assumption into a measured one.

    python -m experiments.run_structural_sensitivity --seconds 45
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import dataset_adapters as DA          # noqa: E402
import dataset_calibration as DC       # noqa: E402
from experiments._common import (build_headless_shop_from_calibration,  # noqa: E402
                                 make_run_dir)

# Anti-congestion separation strengths to sweep; 0.08 is the shipped default.
STRENGTHS = [0.0, 0.04, 0.08, 0.16, 0.32]
DEFAULT = 0.08


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


def run_one(params, strength, seconds, spawn, cap, warmup=30.0):
    """One terminating run at a given separation strength. The empty-store
    fill-up transient is deleted: counters are snapshotted at the warm-up
    boundary and the reported conversion uses post-warm-up increments only."""
    shop = build_headless_shop_from_calibration(params,
                                                max_items_per_category=8,
                                                naive=True)
    sim = shop.customer_simulation
    sim.max_customers = cap
    sim.spawn_rate = spawn
    sim.separation_strength = strength
    sim.start_simulation()
    t0 = time.time()
    while time.time() - t0 < warmup:
        time.sleep(0.3)
    A = sim.analytics
    comp0 = int(A.get('completed_purchases', 0))
    tot0 = int(A.get('total_customers', 0))
    rev0 = float(A.get('total_revenue', 0.0))
    t1 = time.time()
    while time.time() - t1 < seconds:
        time.sleep(0.3)
    rev = float(A.get('total_revenue', 0.0)) - rev0
    comp = int(A.get('completed_purchases', 0)) - comp0
    tot = int(A.get('total_customers', 0)) - tot0
    sim.hard_stop()
    # The live loop suppresses per-agent exceptions to survive a GUI
    # session; a measurement run must not have had any (audit R38).
    errs = dict(getattr(sim, 'suppressed_errors', {}))
    if errs:
        raise RuntimeError(
            f"live loop suppressed exceptions during measurement: {errs}")
    conv = comp / max(tot, 1)
    return {'strength': strength, 'revenue': rev, 'completed': comp,
            'customers': tot, 'conversion': conv}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seconds', type=float, default=45.0)
    p.add_argument('--warmup', type=float, default=30.0)
    p.add_argument('--spawn', type=float, default=3.0)
    p.add_argument('--cap', type=int, default=55)
    # A single run per setting is noise-dominated: two unreplicated
    # sweeps produced per-customer revenue series that disagreed in
    # both magnitude and ordering, and a monotone-looking decline in
    # one did not survive in the other (audit R73). Replicate.
    p.add_argument('--reps', type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = make_run_dir(os.path.join(_HERE, 'results'),
                           'structural_sensitivity')
    print('[struct] calibrating UCI shop once...', flush=True)
    params = _calibrate()

    rows, per_setting_reps = [], {}
    for s in STRENGTHS:
        reps = [run_one(params, s, args.seconds, args.spawn, args.cap,
                        warmup=args.warmup)
                for _ in range(args.reps)]
        per_setting_reps[str(s)] = reps
        # Pool replications: counts add, so the chi-square below sees
        # reps x exposure rather than a single noisy window.
        r = {
            'strength': s,
            'revenue': float(np.sum([x['revenue'] for x in reps])),
            'completed': int(np.sum([x['completed'] for x in reps])),
            'customers': int(np.sum([x['customers'] for x in reps])),
        }
        r['conversion'] = r['completed'] / max(r['customers'], 1)
        rpc = [x['revenue'] / max(x['completed'], 1) for x in reps]
        r['rev_per_cust_mean'] = float(np.mean(rpc))
        r['rev_per_cust_sd'] = float(np.std(rpc, ddof=1)) if len(rpc) > 1 else 0.0
        rows.append(r)
        print(f"  separation strength {s:>4}: "
              f"rev/cust={r['rev_per_cust_mean']:.3f}"
              f" +/-{r['rev_per_cust_sd']:.3f} (sd over {args.reps} reps)  "
              f"completed={r['completed']}  customers={r['customers']}",
              flush=True)

    # Macro outcome = post-warm-up THROUGHPUT (completed purchases per
    # window). Conversion saturates near 1.0 once the fill-up transient
    # is deleted (in equilibrium almost every exiting agent purchases at
    # nominal load), so it cannot discriminate settings; throughput is a
    # proper count that congestion would move. (The live total_revenue
    # accumulator is a GUI display, unreliable headless -- the paper's
    # revenue comes from the MC engine, so it is not used here.)
    comp = np.array([r['completed'] for r in rows], dtype=float)
    conv = np.array([r['conversion'] for r in rows], dtype=float)
    thr_range_pct = (comp.max() - comp.min()) / max(comp.mean(), 1e-9) * 100
    # Chi-square homogeneity on the completion counts: do the per-setting
    # throughputs differ beyond chance for equal expected rates? A
    # non-significant p means no detectable structural effect.
    from scipy.stats import chisquare
    try:
        chi2, pval = chisquare(comp)
        dof = comp.size - 1
    except Exception:
        chi2, pval, dof = float('nan'), float('nan'), 0

    # Between- vs within-setting variation in per-customer revenue. If the
    # spread across settings is not large relative to the replication
    # spread within a setting, the sweep cannot support any claim about a
    # revenue effect -- which is exactly what an unreplicated sweep hid.
    rpc_means = np.array([r['rev_per_cust_mean'] for r in rows], dtype=float)
    rpc_sds = np.array([r['rev_per_cust_sd'] for r in rows], dtype=float)
    try:
        from scipy.stats import f_oneway
        groups = [[x['revenue'] / max(x['completed'], 1)
                   for x in per_setting_reps[str(s)]] for s in STRENGTHS]
        f_stat, rpc_p = f_oneway(*groups)
    except Exception:
        f_stat, rpc_p = float('nan'), float('nan')

    summary = {
        'strengths': STRENGTHS,
        'n_reps': int(args.reps),
        'rev_per_cust_mean': [round(float(v), 4) for v in rpc_means],
        'rev_per_cust_sd': [round(float(v), 4) for v in rpc_sds],
        'rev_per_cust_anova_F': (round(float(f_stat), 3)
                                 if np.isfinite(f_stat) else None),
        'rev_per_cust_anova_p': (round(float(rpc_p), 4)
                                 if np.isfinite(rpc_p) else None),
        'completed_per_setting': [int(c) for c in comp],
        'conversion': [round(float(v), 4) for v in conv],
        'default_strength': DEFAULT,
        'throughput_range_pct': round(float(thr_range_pct), 2),
        'chi2': round(float(chi2), 3),
        'chi2_p': round(float(pval), 3),
        'chi2_dof': int(dof),
        'homogeneous': bool(pval > 0.05) if np.isfinite(pval) else None,
        'metric': 'post_warmup_throughput',
        'rows': rows,
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    verdict = ('no detectable' if (np.isfinite(pval) and pval > 0.05)
               else 'a detectable' if np.isfinite(pval) else 'an untestable')
    print(f"\nPost-warm-up throughput across separation strengths spans "
          f"{thr_range_pct:.1f}%; chi-square homogeneity "
          f"p={pval:.3f} (dof {dof}) -> {verdict} structural effect.")
    print(f"Artifacts in: {out_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
