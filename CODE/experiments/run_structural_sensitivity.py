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

Every run is a fixed-step headless run of the agent model
(``CustomerFlowSimulation.run_headless``: the GUI loop's ``step`` with no
sleeping) for ``--warmup`` + ``--seconds`` simulated seconds under its own
seed. Runs are independent, so ``--workers`` runs them in parallel; results
are put back in design order before aggregation, so the summary is the
same for any worker count.

The store is ``experiments._live_store``'s: the UCI workbook's current
period (its last sheet, every row), laid out naively and seeded with the
shop so the agents draw their list lengths from the invoices cut to the
stocked products. ``--retail-path`` names the workbook; without it
``dataset_paths.uci_workbook()`` finds it. The summary records the period,
its date range and the workbook.

The sweep runs at the nominal protocol shared with run_abm_diagnostics.
The defaults were set by a transient study on this store (3,600 s runs
from 09:00 on seeds disjoint from the runners' own), each value by a fixed
rule:

  --spawn 0.26 --cap 45  The highest arrival rate on a 0.01/s grid at which
      occupancy stays below the cap at least 99% of the time after the
      warm-up. The cap binds 0.66% of the time at 0.26/s and 1.49% at
      0.27/s (16 replications each). The calibrated hour-of-day profile
      scales the rate by 0.744 in the 09:00-10:00 hour a run falls in.
  --warmup 900  Welch's procedure on the replication-mean occupancy at the
      nominal rate. The moving average is taken only where its window is
      complete, with the smallest half-window (from 30, 60, 120, 240 and
      480 s) whose plateau noise is under a third of the 5% band: 240 s.
      The smoothed curve then stays within 5% of its plateau (32.4
      customers in store) from 435 s on; doubled and rounded up to the
      next 60 s that is 900 s. MSER-5 on the same curve truncates at
      600 s, so it asks for no more.
  --seconds 1860  The shortest whole-minute window in which every study
      replication completes at least 300 visits (agents that arrive after
      the warm-up and leave inside the window; the fewest was 301 and the
      mean 329) and that spans at least ten median visits (median 120 s).
      Warm-up plus window ends at 2,760 s, inside the first trading hour,
      so the arrival rate is constant over a run.

    python -m experiments.run_structural_sensitivity
    python -m experiments.run_structural_sensitivity --reps 5 --workers 14
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from experiments import _live_store as LS  # noqa: E402
from experiments._common import (make_run_dir, provenance_snapshot,  # noqa: E402
                                 write_sidecar)

# Anti-congestion separation strengths to sweep; 0.08 is the shipped default.
STRENGTHS = [0.0, 0.04, 0.08, 0.16, 0.32]
DEFAULT = 0.08

# Default protocol; the module docstring records how each value was chosen.
NOMINAL_SPAWN = 0.26
NOMINAL_CAP = 45
WARMUP_S = 900.0
COLLECT_S = 1860.0


def _counters(sim):
    A = sim.analytics
    return {'completed': int(A.get('completed_purchases', 0)),
            'abandoned': int(A.get('abandoned_carts', 0)),
            'customers': int(A.get('total_customers', 0)),
            'revenue': float(A.get('total_revenue', 0.0)),
            'balked': int(A.get('balked_arrivals', 0))}


def run_one(params, strength, seconds, spawn, cap, warmup=WARMUP_S, dt=0.04,
            seed=None):
    """One terminating run at a given separation strength, fixed-step
    headless for ``warmup + seconds`` simulated seconds. The empty-store
    fill-up transient is deleted: counters are snapshotted at the warm-up
    boundary and every reported count is a post-warm-up increment."""
    shop = LS.build_live_store(params)
    sim = shop.customer_simulation
    sim.max_customers = cap
    sim.spawn_rate = spawn
    sim.separation_strength = strength

    start = {}

    def _mark_warmup(s):
        # The first call lands on the first tick at or past the warm-up;
        # later periodic calls fall inside the collection window.
        if not start:
            start.update(_counters(s))

    sim.run_headless(warmup + seconds, dt=dt, seed=seed,
                     callback=_mark_warmup, callback_every_s=warmup)
    # step() counts and survives per-agent exceptions so a GUI session
    # keeps running; a measurement run must not have had any.
    errs = dict(getattr(sim, 'suppressed_errors', {}))
    if errs:
        raise RuntimeError(
            f"live loop suppressed exceptions during measurement: {errs}")
    end = _counters(sim)
    comp = end['completed'] - start['completed']
    aband = end['abandoned'] - start['abandoned']
    tot = end['customers'] - start['customers']
    # Conversion is taken over the EXITS of the window, not over its
    # arrivals: purchases are counted when an agent leaves and arrivals when
    # it spawns, so dividing one by the other mixes two populations and can
    # exceed 1 whenever the store empties a little over the window. Every
    # exit is either a purchase or an abandonment.
    return {'strength': strength, 'seed': seed,
            'revenue': end['revenue'] - start['revenue'],
            'completed': comp, 'abandoned': aband, 'customers': tot,
            'balked': end['balked'] - start['balked'],
            'conversion': comp / max(comp + aband, 1)}


def _design_run(j, plan, params, args):
    """Run ``j`` of the sweep design. Top-level so a worker process can run
    it; the setting comes from ``plan`` rather than the module constant so
    the worker runs exactly the design the parent built."""
    strength, rep, seed = plan[j]
    out = run_one(params, strength, args.seconds, args.spawn, args.cap,
                  warmup=args.warmup, dt=args.dt, seed=seed)
    out['rep'] = rep
    return out


def _map_runs(fn, n_runs, workers, *fn_args):
    """``[fn(j, *fn_args) for j in range(n_runs)]``, in design order.

    With ``workers`` > 1 the runs execute in separate processes. Each
    builds its own store and seeds every RNG it draws from, and results are
    put back in design order before anything is aggregated, so the summary
    is identical to the serial run."""
    if workers <= 1:
        return [fn(j, *fn_args) for j in range(n_runs)]
    done = {}
    pool = ProcessPoolExecutor(max_workers=min(workers, n_runs))
    try:
        futures = {pool.submit(fn, j, *fn_args): j for j in range(n_runs)}
        for fut in as_completed(futures):
            done[futures[fut]] = fut.result()
    except BaseException:
        # Leaving the pool's context manager would wait for every queued run
        # first, so a run that fails early would only report once all the
        # others had finished. Drop what has not started and let the error
        # out now.
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return [done[j] for j in sorted(done)]


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--seconds', type=float, default=COLLECT_S,
                   help="Collection window after the warm-up, simulated s")
    p.add_argument('--warmup', type=float, default=WARMUP_S,
                   help="Deleted warm-up, simulated s")
    p.add_argument('--spawn', type=float, default=NOMINAL_SPAWN,
                   help="Arrivals per simulated s, before the hour-of-day "
                        "profile")
    p.add_argument('--cap', type=int, default=NOMINAL_CAP,
                   help="Most customers in the store at once")
    # A single run per setting is noise-dominated: two unreplicated
    # sweeps produced per-customer revenue series that disagreed in
    # both magnitude and ordering, and a monotone-looking decline in
    # one did not survive in the other (audit R73). Replicate.
    p.add_argument('--reps', type=int, default=5)
    # 0.04 s is the threaded GUI loop's 25 fps ceiling, so agents move
    # with the per-tick resolution they have in the GUI.
    p.add_argument('--dt', type=float, default=0.04,
                   help="Fixed tick, simulated s")
    p.add_argument('--seed', type=int, default=2000,
                   help="Base seed; run j of the design runs under seed + j")
    p.add_argument('--workers', type=int, default=1,
                   help="Processes running the sweep's runs in parallel")
    p.add_argument('--retail-path', type=str, default=None,
                   help="Path to the UCI Online Retail II workbook. Default: "
                        "found by dataset_paths.uci_workbook() "
                        "($UCI_RETAIL_XLSX, then DATASETS/)")
    p.add_argument('--out-root', type=str,
                   default=os.path.join(_HERE, 'results'))
    args = p.parse_args(argv)
    if args.workers < 1:
        p.error("--workers must be >= 1")
    if args.reps < 1:
        p.error("--reps must be >= 1")
    if args.warmup <= 0 or args.seconds <= 0 or args.dt <= 0:
        p.error("--warmup, --seconds and --dt must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    out_dir = make_run_dir(args.out_root, 'structural_sensitivity')
    # The git state at the start of the run, so the sidecar can say which
    # committed code produced the summary and whether it changed mid-run.
    prov = provenance_snapshot()
    print('[struct] calibrating the live store (current period) once...',
          flush=True)
    params = LS.calibrate_live_store(args.retail_path, 'current')
    data = LS.period_record(args.retail_path, 'current')

    # Run j = k * reps + r (setting k, replication r) uses seed base + j.
    # Every setting thus draws its own arrival and agent streams, so the
    # chi-square and the ANOVA below compare independent samples. Common
    # random numbers would hand every setting the same arrivals, making
    # the completion counts move together and both tests conservative.
    plan = [(s, r, args.seed + k * args.reps + r)
            for k, s in enumerate(STRENGTHS) for r in range(args.reps)]
    print(f"[struct] {len(plan)} runs ({len(STRENGTHS)} settings x "
          f"{args.reps} reps) of warm-up {args.warmup:.0f}s + collect "
          f"{args.seconds:.0f}s simulated (dt {args.dt}s) on "
          f"{args.workers} worker(s)...", flush=True)
    t0 = time.perf_counter()
    runs = _map_runs(_design_run, len(plan), args.workers, plan, params, args)
    wall = time.perf_counter() - t0

    rows, per_setting_reps = [], {}
    for k, s in enumerate(STRENGTHS):
        reps = runs[k * args.reps:(k + 1) * args.reps]
        per_setting_reps[str(s)] = reps
        # Pool replications: counts add, so the chi-square below sees
        # reps x exposure rather than a single noisy window.
        r = {
            'strength': s,
            'revenue': float(np.sum([x['revenue'] for x in reps])),
            'completed': int(np.sum([x['completed'] for x in reps])),
            'abandoned': int(np.sum([x['abandoned'] for x in reps])),
            'customers': int(np.sum([x['customers'] for x in reps])),
            'balked': int(np.sum([x['balked'] for x in reps])),
        }
        r['conversion'] = r['completed'] / max(r['completed']
                                               + r['abandoned'], 1)
        rpc = [x['revenue'] / max(x['completed'], 1) for x in reps]
        r['rev_per_cust_mean'] = float(np.mean(rpc))
        r['rev_per_cust_sd'] = float(np.std(rpc, ddof=1)) if len(rpc) > 1 else 0.0
        r['runs'] = reps
        rows.append(r)
        print(f"  separation strength {s:>4}: "
              f"rev/cust={r['rev_per_cust_mean']:.3f}"
              f" +/-{r['rev_per_cust_sd']:.3f} (sd over {args.reps} reps)  "
              f"completed={r['completed']}  customers={r['customers']}  "
              f"balked={r['balked']}",
              flush=True)

    # Macro outcome = post-warm-up THROUGHPUT (completed purchases per
    # window). Conversion saturates near 1.0 once the fill-up transient
    # is deleted (in equilibrium almost every exiting agent purchases at
    # nominal load), so it cannot discriminate settings; throughput is a
    # proper count that congestion would move. Revenue per completing
    # customer is reported alongside it, since basket size is where a
    # separation effect would land once throughput is flat.
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

    # The pooled chi-square above assumes the counts are Poisson, which the
    # replications can check rather than assume: a run-to-run spread wider
    # than Poisson makes that p-value optimistic. The primary test is
    # therefore a one-way ANOVA on the per-replication completion counts,
    # which needs no variance assumption, backed by a quasi-Poisson
    # correction that divides the chi-square by the estimated dispersion.
    comp_reps = [[x['completed'] for x in per_setting_reps[str(s)]]
                 for s in STRENGTHS]
    try:
        from scipy.stats import f_oneway
        comp_F, comp_p = f_oneway(*comp_reps)
    except Exception:
        comp_F, comp_p = float('nan'), float('nan')
    phi = 1.0
    if args.reps > 1:
        # Pearson dispersion: mean squared deviation from the setting mean
        # in units of that mean, over the within-setting degrees of freedom.
        # It is clamped at 1: counts tighter than Poisson would sharpen the
        # test, and with this few replications an apparent underdispersion
        # is as likely to be noise as a real one.
        dev = sum(float(np.sum((np.array(c, dtype=float) - np.mean(c)) ** 2
                               / max(np.mean(c), 1e-9)))
                  for c in comp_reps)
        phi = max(dev / (len(STRENGTHS) * (args.reps - 1)), 1.0)
    try:
        from scipy.stats import chi2 as chi2_dist
        quasi_p = (float(chi2_dist.sf(chi2 / phi, dof))
                   if np.isfinite(chi2) and dof > 0 else float('nan'))
    except Exception:
        quasi_p = float('nan')

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

    n_arr = int(sum(r['customers'] + r['balked'] for r in rows))
    n_balk = int(sum(r['balked'] for r in rows))
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
        'completed_per_rep': [[int(x) for x in c] for c in comp_reps],
        'conversion': [round(float(v), 4) for v in conv],
        'default_strength': DEFAULT,
        'throughput_range_pct': round(float(thr_range_pct), 2),
        'chi2': round(float(chi2), 3),
        'chi2_p': round(float(pval), 3),
        'chi2_dof': int(dof),
        # Assumption-free primary test on the per-replication counts, plus
        # the dispersion the replications show and the chi-square corrected
        # by it.
        'completions_anova_F': (round(float(comp_F), 3)
                                if np.isfinite(comp_F) else None),
        'completions_anova_p': (round(float(comp_p), 4)
                                if np.isfinite(comp_p) else None),
        'dispersion_phi': round(float(phi), 3),
        'quasi_poisson_p': (round(float(quasi_p), 4)
                            if np.isfinite(quasi_p) else None),
        # Read off the primary test, not the pooled chi-square it replaces.
        'homogeneous': bool(comp_p > 0.05) if np.isfinite(comp_p) else None,
        'metric': 'post_warmup_throughput',
        # Post-warm-up arrivals over the whole sweep; balks are arrivals
        # refused at the customer cap.
        'load': {'arrivals': n_arr, 'balked': n_balk,
                 'balked_frac': round(n_balk / max(n_arr, 1), 4)},
        'protocol': {'mode': 'fixed_step', 'dt': args.dt,
                     'seed_base': args.seed,
                     'seeds': {str(s): [x['seed'] for x in
                                        per_setting_reps[str(s)]]
                               for s in STRENGTHS},
                     'reps': args.reps, 'warmup_s': args.warmup,
                     'collect_s': args.seconds, 'spawn': args.spawn,
                     'cap': args.cap, 'workers': args.workers,
                     'framing': 'terminating'},
        # The data the store was calibrated from.
        'data': data,
        'rows': rows,
    }
    # The sidecar goes first: summary.json is the file that marks a run as
    # finished, so it is written last.
    write_sidecar(out_dir, {'experiment': 'structural_sensitivity',
                            'args': vars(args), 'wall_seconds': wall},
                  provenance=prov)
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    verdict = ('no detectable' if (np.isfinite(comp_p) and comp_p > 0.05)
               else 'a detectable' if np.isfinite(comp_p) else 'an untestable')
    print(f"\nPost-warm-up throughput across separation strengths spans "
          f"{thr_range_pct:.1f}%; one-way ANOVA on the per-replication "
          f"counts p={comp_p:.3f} -> {verdict} structural effect "
          f"(pooled chi-square p={pval:.3f} at dof {dof}, "
          f"p={quasi_p:.3f} after the quasi-Poisson correction at "
          f"dispersion {phi:.2f}).")
    print(f"post-warm-up arrivals={n_arr}, balked at the cap={n_balk}")
    print(f"wall {wall:.0f}s; artifacts in: {out_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
