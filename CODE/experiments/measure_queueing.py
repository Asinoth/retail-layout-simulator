"""Measure checkout-lane occupancy over a live simulation (audit R1.7).

Reviewer 1 asked whether the claimed *emergent* checkout congestion
actually materializes at the concurrent-agent cap. This script runs the
real simulation worker loop on a calibrated shop and samples, at fixed
intervals: total active agents, and per-lane the number of agents
currently checking out (occupancy) plus the number en route to that
lane. It writes ``../figs/queue_lengths.png`` and prints summary
statistics so the congestion claim can be kept or honestly downgraded
based on evidence.

    python -m experiments.measure_queueing
    python -m experiments.measure_queueing --spawn 2.5 --cap 40 --seconds 45
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import figstyle  # noqa: E402  (shared style, audit R9)
figstyle.apply()

import dataset_adapters as DA          # noqa: E402
import dataset_calibration as DC       # noqa: E402
from experiments._common import build_headless_shop_from_calibration  # noqa: E402


def _uci_path():
    root = os.path.dirname(_HERE)
    for c in (os.path.join(root, 'DATASETS', 'UCI Online Retail II .xlsx.xlsx'),
              os.path.join(os.path.dirname(root), 'DATASETS',
                           'UCI Online Retail II .xlsx.xlsx')):
        if os.path.exists(c):
            return c
    raise FileNotFoundError('UCI dataset not found under DATASETS/')


def _figs_dir():
    # _HERE is CODE/experiments; the repo-root figs/ is two levels up.
    root = os.path.dirname(os.path.dirname(_HERE))
    d = os.path.join(root, 'figs')
    os.makedirs(d, exist_ok=True)
    return d


def _lane_names(shop):
    return [nm for nm in shop.floors[1]['walls']
            if nm == 'Checkout' or nm.startswith('Checkout_Lane')]


def sample_state(sim, lanes):
    occ = {L: 0 for L in lanes}       # currently checking out at L
    approaching = {L: 0 for L in lanes}
    for c in list(sim.customers):
        L = getattr(c, 'checkout_lane', None)
        if L not in occ:
            continue
        if c.state == 'checking_out':
            occ[L] += 1
        elif getattr(c, 'current_target_type', None) == 'checkout':
            approaching[L] += 1
    return occ, approaching


def run_once(spawn, cap, seconds, warmup=20.0, dt=0.4):
    df, _ = DA.read_excel_sheets(_uci_path(),
                                 [DA.list_excel_sheets(_uci_path())[-1][0]])
    df = df.sample(n=60000, random_state=0).reset_index(drop=True)
    norm, _ = DA.OnlineRetailIIAdapter().adapt(df)
    params = DC.calibrate_transactional(norm, currency='GBP')
    shop = build_headless_shop_from_calibration(params,
                                                max_items_per_category=8,
                                                naive=True)
    sim = shop.customer_simulation
    sim.max_customers = cap
    sim.spawn_rate = spawn
    lanes = _lane_names(shop)

    ts, totals, occ_series, appr_series = [], [], [], []
    sim.start_simulation()
    # Warm-up deletion: the store starts empty, so occupancy samples from
    # the fill-up transient would understate congestion. Discard them.
    t_w = time.time()
    while time.time() - t_w < warmup:
        time.sleep(dt)
    t0 = time.time()
    while time.time() - t0 < seconds:
        time.sleep(dt)
        occ, appr = sample_state(sim, lanes)
        ts.append(time.time() - t0)
        totals.append(len(sim.customers))
        occ_series.append([occ[L] for L in lanes])
        appr_series.append([appr[L] for L in lanes])
    sim.hard_stop()

    return {
        'lanes': lanes, 'ts': np.array(ts), 'totals': np.array(totals),
        'occ': np.array(occ_series), 'appr': np.array(appr_series),
        'spawn': spawn, 'cap': cap,
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spawn', type=float, default=1.5)
    ap.add_argument('--cap', type=int, default=20)
    ap.add_argument('--seconds', type=float, default=40.0)
    ap.add_argument('--stress-spawn', type=float, default=3.0)
    ap.add_argument('--stress-cap', type=int, default=50)
    # A single window per load level is not enough: across unreplicated
    # runs the nominal busy fraction came out 23%, 37% and 20% while the
    # stress figure barely moved. Report a mean over replications with a
    # spread so the reader can see which of the two is actually stable
    # (audit R74).
    ap.add_argument('--reps', type=int, default=5)
    return ap.parse_args()


def summarize(tag, r, quiet=False):
    per_lane_occ = r['occ']  # (T, L)
    max_lane = int(per_lane_occ.max()) if per_lane_occ.size else 0
    busy_frac = float((per_lane_occ >= 2).any(axis=1).mean()) if per_lane_occ.size else 0.0
    if not quiet:
        print(f'  [{tag}] cap={r["cap"]} spawn={r["spawn"]}: '
              f'peak concurrent agents={int(r["totals"].max())}, '
              f'max per-lane checkout occupancy={max_lane}, '
              f'fraction of time some lane has >=2 queued={busy_frac*100:.1f}%')
    return max_lane, busy_frac


def main():
    args = parse_args()
    figs = _figs_dir()

    print(f'[queue] nominal load: {args.reps} replications...', flush=True)
    nom_reps = [run_once(args.spawn, args.cap, args.seconds)
                for _ in range(args.reps)]
    print(f'[queue] stress load: {args.reps} replications...', flush=True)
    str_reps = [run_once(args.stress_spawn, args.stress_cap, args.seconds)
                for _ in range(args.reps)]

    # Keep one representative run for the time-series figure; report the
    # scalar statistics as means over replications.
    r_nom, r_str = nom_reps[0], str_reps[0]

    def _pool(tag, reps):
        stats = [summarize(tag, r, quiet=True) for r in reps]
        lanes = [s[0] for s in stats]
        fracs = np.array([s[1] for s in stats], dtype=float)
        sd = float(fracs.std(ddof=1)) if fracs.size > 1 else 0.0
        print(f'  [{tag}] over {len(reps)} reps: '
              f'busy-fraction mean={fracs.mean()*100:.1f}% '
              f'(sd {sd*100:.1f} pp, min {fracs.min()*100:.1f}%, '
              f'max {fracs.max()*100:.1f}%), max lane occupancy={max(lanes)}')
        return max(lanes), float(fracs.mean()), sd

    m_nom, b_nom, sd_nom = _pool('nominal', nom_reps)
    m_str, b_str, sd_str = _pool('stress', str_reps)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    for ax, r, title in ((axes[0], r_nom, f'Nominal (cap {r_nom["cap"]}, spawn {r_nom["spawn"]}/s)'),
                         (axes[1], r_str, f'Stress (cap {r_str["cap"]}, spawn {r_str["spawn"]}/s)')):
        total_q = r['occ'].sum(axis=1) + r['appr'].sum(axis=1)
        ax.plot(r['ts'], r['occ'].sum(axis=1), color='#E15759', lw=1.6,
                label='checking out (all lanes)')
        ax.plot(r['ts'], total_q, color='#4E79A7', lw=1.4, ls='--',
                label='checking out + en route')
        ax.plot(r['ts'], r['totals'], color='#59A14F', lw=1.0, alpha=0.6,
                label='total active agents')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('time (s)'); ax.grid(alpha=0.25)
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel('count')
    fig.suptitle('Checkout-lane occupancy over a live run', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    figstyle.save(fig, 'queue_lengths')
    print('wrote figs/queue_lengths.png')

    verdict = ('emergent congestion is observable'
               if (m_str >= 2 or b_str > 0.05)
               else 'congestion negligible even under stress '
                    '-> downgrade claim to layout-aware lane assignment')
    print('[queue] VERDICT:', verdict)

    import json
    summary = {
        'nominal': {'cap': r_nom['cap'], 'spawn': r_nom['spawn'],
                    'peak_agents': int(r_nom['totals'].max()),
                    'max_lane_occupancy': m_nom, 'busy_frac': b_nom,
                    'busy_frac_sd': sd_nom, 'n_reps': int(args.reps)},
        'stress': {'cap': r_str['cap'], 'spawn': r_str['spawn'],
                   'peak_agents': int(r_str['totals'].max()),
                   'max_lane_occupancy': m_str, 'busy_frac': b_str,
                   'busy_frac_sd': sd_str, 'n_reps': int(args.reps)},
        'verdict': verdict,
    }
    with open(os.path.join(figs, 'queue_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print('wrote figs/queue_summary.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
