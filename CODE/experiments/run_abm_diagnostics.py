"""ABM methodology diagnostics from a live run (audit R6.2 + R6.4).

Two ABM-reviewer questions, answered from one calibrated live simulation:

  R6.2  First-order Markov adequacy. The paper computes an absorbing
        first-order Markov summary of the agents' embedded jump chain.
        Hui (2009) argues shopper paths carry longer memory, so we test
        whether a SECOND-order chain would change the next-state
        distribution appreciably. We log ordered per-agent state
        sequences, then compare, for each current state, the next-state
        distribution conditioned on one vs. two prior states, via the
        information gain H(next|cur) - H(next|cur,prev) and the mean
        total-variation distance between the first- and second-order
        conditionals. Small values => first order is an adequate summary
        for the absorption statistics we actually use.

  R6.4  Emergent traffic pattern. Larson (2005) documents perimeter-
        dominant supermarket traffic. We measure the emergent
        perimeter-to-interior foot-traffic ratio from the accumulated
        heat map -- a quantitative face-validation number, not a picture.

    python -m experiments.run_abm_diagnostics --seconds 90 --spawn 2.5 --cap 45
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import dataset_adapters as DA          # noqa: E402
import dataset_calibration as DC       # noqa: E402
from experiments._common import build_headless_shop_from_calibration  # noqa: E402

TRANSIENT = ('entering', 'moving', 'shopping', 'checking_out', 'exiting')
ABSORB = ('purchased', 'abandoned')


def _uci_path():
    root = os.path.dirname(_HERE)
    for c in (os.path.join(root, 'DATASETS', 'UCI Online Retail II .xlsx.xlsx'),
              os.path.join(os.path.dirname(root), 'DATASETS',
                           'UCI Online Retail II .xlsx.xlsx')):
        if os.path.exists(c):
            return c
    raise FileNotFoundError('UCI dataset not found under DATASETS/')


def _results_dir():
    from experiments._common import make_run_dir
    return make_run_dir(os.path.join(_HERE, 'results'), 'abm_diagnostics')


def _entropy(counts):
    tot = sum(counts.values())
    if tot <= 0:
        return 0.0
    h = 0.0
    for v in counts.values():
        if v > 0:
            p = v / tot
            h -= p * math.log2(p)
    return h


def _tv(p, q, support):
    return 0.5 * sum(abs(p.get(s, 0.0) - q.get(s, 0.0)) for s in support)


def _calibrate_once():
    df, _ = DA.read_excel_sheets(_uci_path(),
                                 [DA.list_excel_sheets(_uci_path())[-1][0]])
    df = df.sample(n=60000, random_state=0).reset_index(drop=True)
    norm, _ = DA.OnlineRetailIIAdapter().adapt(df)
    return DC.calibrate_transactional(norm, currency='GBP')


def collect_sequences(params, seconds, spawn, cap, warmup=30.0, dt=0.3):
    """One replicated live run. The store starts empty, so the first
    ``warmup`` seconds (several door-to-back traversal times at nominal
    speed) are DELETED before any statistic is collected: sequence
    logging begins at the warm-up boundary and the heat map is measured
    as the post-warm-up increment (baseline snapshot subtracted). The
    run is a terminating simulation over a trading window; the deletion
    removes the empty-store fill-up transient from the within-window
    equilibrium statistics."""
    shop = build_headless_shop_from_calibration(params,
                                                max_items_per_category=8,
                                                naive=True)
    sim = shop.customer_simulation
    sim.max_customers = cap
    sim.spawn_rate = spawn

    seq_by_id = {}          # id -> [states...] (jump chain, distinct)
    last_state = {}
    finished = []           # completed jump sequences

    sim.start_simulation()
    t0 = time.time()
    while time.time() - t0 < warmup:      # warm-up: no collection
        time.sleep(dt)
    heat_baseline = np.array(sim.heat_raw, dtype=float).copy()

    t1 = time.time()
    while time.time() - t1 < seconds:
        time.sleep(dt)
        cur_ids = set()
        for c in list(sim.customers):
            cid = c.id
            cur_ids.add(cid)
            st = c.state
            if cid not in seq_by_id:
                seq_by_id[cid] = [st]
                last_state[cid] = st
            elif st != last_state[cid]:
                seq_by_id[cid].append(st)
                last_state[cid] = st
        # Flush sequences whose agent despawned since last sample.
        for cid in list(seq_by_id.keys()):
            if cid not in cur_ids:
                finished.append(seq_by_id.pop(cid))
                last_state.pop(cid, None)
    heat = np.array(sim.heat_raw, dtype=float) - heat_baseline
    W, H = shop.width, shop.height
    sim.hard_stop()
    # The live loop suppresses per-agent exceptions to survive a GUI
    # session; a measurement run must not have had any (audit R38).
    errs = dict(getattr(sim, 'suppressed_errors', {}))
    if errs:
        raise RuntimeError(
            f"live loop suppressed exceptions during measurement: {errs}")
    # Include still-active agents' partial sequences.
    finished.extend(seq_by_id.values())
    return finished, heat, W, H


def markov_order_analysis(sequences):
    """Return first-order-adequacy diagnostics from jump sequences."""
    cnt1 = defaultdict(lambda: defaultdict(int))          # cur -> next
    cnt2 = defaultdict(lambda: defaultdict(int))          # (prev,cur) -> next
    n_trans = 0
    for seq in sequences:
        for i in range(len(seq) - 1):
            cur, nxt = seq[i], seq[i + 1]
            cnt1[cur][nxt] += 1
            n_trans += 1
            if i >= 1:
                prev = seq[i - 1]
                cnt2[(prev, cur)][nxt] += 1

    support = list(TRANSIENT + ABSORB)
    MIN_CELL = 8            # minimum counts for a reliable second-order cell
    # Overall first-order conditional entropy (context only).
    tot1 = sum(sum(d.values()) for d in cnt1.values())
    H1 = 0.0
    for cur, d in cnt1.items():
        w = sum(d.values()) / max(tot1, 1)
        H1 += w * _entropy(d)

    # For each DENSE second-order context (prev,cur), pair it against its
    # OWN first-order baseline P(next|cur). Report the frequency-weighted
    # mean entropy reduction from knowing `prev` and the mean TV distance
    # between the first- and second-order conditionals. Small values =>
    # `prev` adds little, i.e. first order is an adequate summary.
    info_gain = 0.0
    tv_acc = 0.0
    wsum = 0
    for (prev, cur), d in cnt2.items():
        n = sum(d.values())
        if n < MIN_CELL:
            continue
        d1 = cnt1[cur]
        t1 = sum(d1.values())
        h1_cur = _entropy(d1)
        h2_cell = _entropy(d)
        info_gain += n * (h1_cur - h2_cell)
        p1 = {s: d1.get(s, 0) / max(t1, 1) for s in support}
        p2 = {s: d.get(s, 0) / n for s in support}
        tv_acc += n * _tv(p2, p1, support)
        wsum += n
    if wsum > 0:
        info_gain /= wsum
        tv_acc /= wsum
    return {
        'n_sequences': len(sequences),
        'n_transitions': int(n_trans),
        'mean_seq_len': float(np.mean([len(s) for s in sequences])) if sequences else 0.0,
        'dense_second_order_mass': int(wsum),
        'H_next_given_cur_bits': round(H1, 4),
        'info_gain_second_order_bits': round(info_gain, 4),
        'mean_tv_first_vs_second': round(tv_acc, 4),
    }


def perimeter_ratio(heat, W, H, band_m=2.5):
    """Emergent perimeter-to-interior mean foot-traffic ratio (R6.4)."""
    wcells, hcells = heat.shape
    xs = np.linspace(0, W, wcells)
    ys = np.linspace(0, H, hcells)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    perim = (X < band_m) | (X > W - band_m) | (Y < band_m) | (Y > H - band_m)
    interior = ~perim
    mp = float(heat[perim].mean()) if perim.any() else 0.0
    mi = float(heat[interior].mean()) if interior.any() else 0.0
    return {
        'perimeter_mean_intensity': round(mp, 4),
        'interior_mean_intensity': round(mi, 4),
        'perimeter_interior_ratio': round(mp / mi, 3) if mi > 1e-9 else None,
        'band_m': band_m,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seconds', type=float, default=90.0)
    p.add_argument('--warmup', type=float, default=30.0)
    p.add_argument('--reps', type=int, default=3)
    p.add_argument('--spawn', type=float, default=2.5)
    p.add_argument('--cap', type=int, default=45)
    return p.parse_args()


def _mean_ci(xs):
    """Mean and 95% t-interval half-width for a small sample."""
    from scipy.stats import t as tdist
    xs = np.asarray(xs, dtype=float)
    n = xs.size
    m = float(xs.mean())
    if n < 2:
        return m, float('nan')
    hw = float(tdist.ppf(0.975, n - 1) * xs.std(ddof=1) / np.sqrt(n))
    return m, hw


def main():
    args = parse_args()
    out_dir = _results_dir()
    print('[abm] calibrating UCI shop once...', flush=True)
    params = _calibrate_once()

    reps = []
    for r in range(args.reps):
        print(f"[abm] rep {r + 1}/{args.reps}: warm-up {args.warmup:.0f}s + "
              f"collect {args.seconds:.0f}s...", flush=True)
        seqs, heat, W, H = collect_sequences(
            params, args.seconds, args.spawn, args.cap, warmup=args.warmup)
        mk = markov_order_analysis(seqs)
        pr = perimeter_ratio(heat, W, H)
        # The perimeter ratio depends on how wide a band counts as
        # "perimeter", so archive the sweep alongside the headline value
        # rather than leaving that analysis choice unauditable (R49/R72).
        pr['band_sweep'] = {
            f"{b:.1f}": perimeter_ratio(heat, W, H, band_m=b)
                        ['perimeter_interior_ratio']
            for b in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
        }
        reps.append({'markov_order': mk, 'emergence': pr})
        print(f"    info_gain={mk['info_gain_second_order_bits']:.3f}  "
              f"TV={mk['mean_tv_first_vs_second']:.3f}  "
              f"perim_ratio={pr['perimeter_interior_ratio']}", flush=True)

    ig_m, ig_hw = _mean_ci([r['markov_order']['info_gain_second_order_bits']
                            for r in reps])
    tv_m, tv_hw = _mean_ci([r['markov_order']['mean_tv_first_vs_second']
                            for r in reps])
    h1_m, _ = _mean_ci([r['markov_order']['H_next_given_cur_bits']
                        for r in reps])
    pr_m, pr_hw = _mean_ci([r['emergence']['perimeter_interior_ratio']
                            for r in reps if
                            r['emergence']['perimeter_interior_ratio']])
    n_seq = int(sum(r['markov_order']['n_sequences'] for r in reps))
    n_tr = int(sum(r['markov_order']['n_transitions'] for r in reps))

    # Aggregate keys keep the single-run names (means), so downstream
    # macro generation stays stable; *_ci95 carries the t half-width.
    summary = {
        'markov_order': {
            'n_sequences': n_seq,
            'n_transitions': n_tr,
            'H_next_given_cur_bits': round(h1_m, 4),
            'info_gain_second_order_bits': round(ig_m, 4),
            'info_gain_ci95': round(ig_hw, 4),
            'mean_tv_first_vs_second': round(tv_m, 4),
            'tv_ci95': round(tv_hw, 4),
        },
        'emergence': {
            'perimeter_interior_ratio': round(pr_m, 3),
            'perimeter_ratio_ci95': round(pr_hw, 3),
            'band_m': reps[0]['emergence']['band_m'],
        },
        'protocol': {'reps': args.reps, 'warmup_s': args.warmup,
                     'collect_s': args.seconds, 'spawn': args.spawn,
                     'cap': args.cap, 'framing': 'terminating'},
        'per_rep': reps,
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n== R6.2 first-order Markov adequacy "
          f"({args.reps} reps, warm-up deleted) ==")
    print(f"  sequences={n_seq}  transitions={n_tr}")
    print(f"  H(next|cur)={h1_m:.3f} bits")
    print(f"  info gain from 2nd order = {ig_m:.3f} +/- {ig_hw:.3f} bits")
    print(f"  mean TV(first,second) = {tv_m:.3f} +/- {tv_hw:.3f}")
    print("\n== R6.4 emergent perimeter dominance ==")
    print(f"  perimeter/interior ratio = {pr_m:.2f} +/- {pr_hw:.2f}")
    print(f"\nArtifacts in: {out_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
