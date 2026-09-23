"""Measure checkout-lane queues over a live simulation (audit R1.7).

Reviewer 1 asked whether the claimed *emergent* checkout congestion
actually materializes at the concurrent-agent cap. This script runs the
agent model on a calibrated shop. Each lane is a single-server FIFO
queue (head in service, the rest waiting), and after a warm-up the script
samples, at fixed simulated intervals: total active agents, and per lane
the queue length plus the number of agents en route to that lane. From
the samples and the recorded service-start waits it reports per-lane mean
waiting count, fraction of time with at least one customer waiting,
server occupancy and mean wait. It writes ``queue_lengths.{pdf,png}`` and
``queue_summary.json`` to ``--figs-dir`` (the repo figs/ by default), a
copy of the summary to a run directory under ``--out-root``, and prints
summary statistics so the congestion claim can be kept or honestly
downgraded based on evidence.

Every replication is a fixed-step headless run
(``CustomerFlowSimulation.run_headless``: the GUI loop's ``step`` with no
sleeping) for ``--warmup`` + ``--seconds`` simulated seconds under its own
seed, so ``--workers`` can run replications in parallel; results are put
back in replication order before aggregation, so the summary and the
figure are the same for any worker count.

The store is ``experiments._live_store``'s: the UCI workbook's current
period (its last sheet, every row), laid out naively and seeded with the
shop so each agent's shopping list is the stocked part of one invoice
of that period. ``--retail-path`` names the workbook; without it
``dataset_paths.uci_workbook()`` finds it. The summary records the period,
its date range and the workbook.

The defaults were set by a transient study on this store (3,600 s runs
from 09:00 on seeds disjoint from the runners' own), each value by a fixed
rule. Lane utilisation below is the fraction of time a lane is serving a
customer, averaged over the lanes.

  --spawn 0.27 --cap 45  The nominal load shared with the other live
      diagnostics: the highest arrival rate on a 0.01/s grid at which
      occupancy stays below the cap at least 99% of the time after the
      warm-up. The cap binds 0.88% of the time at 0.27/s and 1.27% at
      0.28/s (16 replications each); lane utilisation there is 0.29.
  --stress-spawn 0.70 --stress-cap 110  The lowest rate on the grid at
      which lane utilisation reaches 0.7 while the cap binds less than 5%
      of the time, with the cap raised only as far as that needs. Cap 45
      cannot get there: the cap binds long before the lanes saturate. With
      the cap out of reach (200) utilisation climbs 0.32 / 0.36 / 0.42 /
      0.48 / 0.51 / 0.55 / 0.63 / 0.71 at 0.30 / 0.35 / 0.40 / 0.45 /
      0.50 / 0.55 / 0.60 / 0.70 arrivals per second (six replications
      each), so 0.70/s is the first rate on the grid that reaches 0.7. At
      0.70/s a cap of 90 reaches 0.700 but binds 5.3% of the time, and a
      cap of 100 binds 1.3% but holds utilisation to 0.697, because
      refused arrivals never reach a lane; a cap of 110 binds 0.34% with
      utilisation 0.720 (eight replications each), so the stress level
      uses 110.
  --warmup 1020  The nominal level's warm-up (see run_abm_diagnostics:
      the first whole minute after which an initially empty store's
      expected occupancy is within 1% of its steady state, from the
      visit-length distribution; 965 s, rounded up). The same criterion
      at the stress level, on the study's eight 1,800 s replications
      there, gives 845 s, and MSER-5 truncates at 450 s, so the nominal
      1,020 s serves both levels.
  --seconds 1860  The nominal collection window of the other live
      diagnostics: every study replication completes at least 300 visits
      in it (the fewest 306) and it spans at least ten median visits
      (median 118 s). Runs end at 2,880 s, inside the first trading hour,
      so both rates are constant over a run; the calibrated hour-of-day
      profile scales them by 0.744 in that hour.

    python -m experiments.measure_queueing
    python -m experiments.measure_queueing --reps 5 --workers 10
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import figstyle  # noqa: E402  (shared style, audit R9)
figstyle.apply()

from experiments import _live_store as LS  # noqa: E402
from experiments._common import (make_run_dir, package_versions,  # noqa: E402
                                 provenance_snapshot, write_sidecar)

SAMPLE_INTERVAL_S = 0.4     # simulated seconds between queue samples

# Default protocol; the module docstring records how each value was chosen.
NOMINAL_SPAWN = 0.27
NOMINAL_CAP = 45
STRESS_SPAWN = 0.70
STRESS_CAP = 110
WARMUP_S = 1020.0
COLLECT_S = 1860.0


def _figs_dir():
    # _HERE is CODE/experiments; the repo-root figs/ is two levels up.
    root = os.path.dirname(os.path.dirname(_HERE))
    d = os.path.join(root, 'figs')
    os.makedirs(d, exist_ok=True)
    return d


def _lane_names(shop):
    return [nm for nm in shop.floors[1]['walls']
            if nm == 'Checkout' or nm.startswith('Checkout_Lane')]


def _arrival_counts(sim):
    """Arrivals admitted and arrivals that balked at the customer cap."""
    A = sim.analytics
    return int(A.get('total_customers', 0)), int(A.get('balked_arrivals', 0))


def sample_state(sim, lanes):
    """Queue length per lane (in service + waiting) and agents en route.

    Lengths come from the simulation's lane queues. Counting agents in the
    'checking_out' state instead would also include customers who have
    paid and are about to leave, and says nothing about who is waiting."""
    queues = sim.lane_queues
    at_lane = {L: len(queues.get(L) or ()) for L in lanes}
    approaching = {L: 0 for L in lanes}
    for c in list(sim.customers):
        L = getattr(c, 'checkout_lane', None)
        if L not in approaching:
            continue
        # Paid agents walking back past the lane will not queue again.
        if (c.state != 'checking_out' and not c.has_checked_out
                and getattr(c, 'current_target_type', None) == 'checkout'):
            approaching[L] += 1
    return at_lane, approaching


def run_once(params, spawn, cap, seconds, warmup=WARMUP_S, dt=0.04, seed=None,
             sample_every=SAMPLE_INTERVAL_S):
    """One replication, fixed-step headless for ``warmup + seconds``
    simulated seconds, sampling the lanes every ``sample_every`` simulated
    seconds after the warm-up."""
    shop = LS.build_live_store(params)
    sim = shop.customer_simulation
    sim.max_customers = cap
    sim.spawn_rate = spawn
    lanes = _lane_names(shop)

    ts, totals, q_series, appr_series = [], [], [], []
    start = {}
    t_begin = float(sim.sim_time)

    def _sample(s):
        # Warm-up deletion: the store starts empty, so queue samples from
        # the fill-up transient would understate congestion. The first
        # sampling call at or past the warm-up marks the boundary, the
        # same one the other live diagnostics delete; half a tick of slack
        # absorbs round-off in the summed dt.
        if not start:
            if s.sim_time - t_begin < warmup - dt / 2:
                return
            start['t'] = float(s.sim_time)
            # Waits are logged when service starts; keep only services
            # that start inside the measurement window.
            start['n_wait'] = len(s.analytics.get('queue_wait_times', []))
            start['arrivals'] = _arrival_counts(s)
            return
        t = float(s.sim_time) - start['t']
        # run_headless also calls back after the last tick. When the run
        # does not end on the sampling grid that call comes early, and the
        # time averages below would give its sample a full interval's weight.
        if ts and t - ts[-1] < sample_every - dt / 2:
            return
        at_lane, appr = sample_state(s, lanes)
        ts.append(t)
        totals.append(len(s.customers))
        q_series.append([at_lane[L] for L in lanes])
        appr_series.append([appr[L] for L in lanes])

    sim.run_headless(warmup + seconds, dt=dt, seed=seed,
                     callback=_sample, callback_every_s=sample_every)
    # step() counts and survives per-agent exceptions so a GUI session
    # keeps running; a skipped agent update would silently bias occupancy,
    # so a measurement run must not have had any.
    errs = dict(getattr(sim, 'suppressed_errors', {}))
    if errs:
        raise RuntimeError(
            f"live loop suppressed exceptions during measurement: {errs}")
    waits = [float(w) for w in
             list(sim.analytics.get('queue_wait_times', []))[start['n_wait']:]]
    admitted, balked = _arrival_counts(sim)

    return {
        'lanes': lanes, 'ts': np.array(ts), 'totals': np.array(totals),
        'q': np.array(q_series, dtype=float).reshape(len(ts), len(lanes)),
        'appr': np.array(appr_series, dtype=float).reshape(len(ts), len(lanes)),
        'waits': waits, 'spawn': spawn, 'cap': cap, 'warmup': warmup,
        'seed': seed,
        'load': {'arrivals': (admitted + balked) - sum(start['arrivals']),
                 'balked': balked - start['arrivals'][1]},
    }


def _level_run(j, plan, params, args):
    """Run ``j`` of the design (one load level, one replication). Top-level
    so a worker process can run it."""
    level, spawn, cap, rep, seed = plan[j]
    out = run_once(params, spawn, cap, args.seconds, warmup=args.warmup,
                   dt=args.dt, seed=seed)
    out['level'], out['rep'] = level, rep
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
    ap = argparse.ArgumentParser()
    ap.add_argument('--spawn', type=float, default=NOMINAL_SPAWN,
                    help="Nominal arrivals per simulated s, before the "
                         "hour-of-day profile")
    ap.add_argument('--cap', type=int, default=NOMINAL_CAP,
                    help="Nominal limit on customers in the store at once")
    ap.add_argument('--seconds', type=float, default=COLLECT_S,
                    help="Collection window after the warm-up, simulated s")
    ap.add_argument('--warmup', type=float, default=WARMUP_S,
                    help="Deleted warm-up, simulated s")
    ap.add_argument('--stress-spawn', type=float, default=STRESS_SPAWN,
                    help="Stress arrivals per simulated s, before the "
                         "hour-of-day profile")
    ap.add_argument('--stress-cap', type=int, default=STRESS_CAP,
                    help="Stress limit on customers in the store at once")
    # A single window per load level is not enough: across unreplicated
    # runs the nominal busy fraction came out 23%, 37% and 20% while the
    # stress figure barely moved. Report a mean over replications with a
    # spread so the reader can see which of the two is actually stable
    # (audit R74).
    ap.add_argument('--reps', type=int, default=5)
    # 0.04 s is the threaded GUI loop's 25 fps ceiling, so agents move
    # with the per-tick resolution they have in the GUI.
    ap.add_argument('--dt', type=float, default=0.04,
                    help="Fixed tick, simulated s")
    ap.add_argument('--seed', type=int, default=3000,
                    help="Base seed; nominal replication r runs under "
                         "seed + r, stress replication r under "
                         "seed + reps + r")
    ap.add_argument('--workers', type=int, default=1,
                    help="Processes running replications in parallel")
    ap.add_argument('--retail-path', type=str, default=None,
                    help="Path to the UCI Online Retail II workbook. Default: "
                         "found by dataset_paths.uci_workbook() "
                         "($UCI_RETAIL_XLSX, then DATASETS/)")
    ap.add_argument('--out-root', type=str,
                    default=os.path.join(_HERE, 'results'))
    ap.add_argument('--figs-dir', type=str, default=None,
                    help="Directory for queue_lengths.{pdf,png} and "
                         "queue_summary.json (default: repo figs/)")
    args = ap.parse_args(argv)
    if args.workers < 1:
        ap.error("--workers must be >= 1")
    if args.reps < 1:
        ap.error("--reps must be >= 1")
    if args.warmup <= 0 or args.seconds <= 0 or args.dt <= 0:
        ap.error("--warmup, --seconds and --dt must be positive")
    return args


def summarize(tag, r, quiet=False):
    """Queue statistics of one run. ``q`` holds agents at each lane; the
    head is in service, so the waiting count is ``max(q - 1, 0)``."""
    q = r['q']  # (T, L)
    waiting = np.maximum(q - 1, 0)
    has_samples = q.size > 0
    stats = {
        'max_lane_occupancy': int(q.max()) if has_samples else 0,
        # Some lane has a customer waiting behind the one being served.
        'busy_frac': float((waiting >= 1).any(axis=1).mean()) if has_samples else 0.0,
        'mean_waiting': float(waiting.mean()) if has_samples else 0.0,
        'occupancy': float((q > 0).mean()) if has_samples else 0.0,
        'mean_waiting_per_lane': (waiting.mean(axis=0) if has_samples
                                  else np.zeros(len(r['lanes']))),
        'wait_frac_per_lane': ((waiting >= 1).mean(axis=0) if has_samples
                               else np.zeros(len(r['lanes']))),
        'occupancy_per_lane': ((q > 0).mean(axis=0) if has_samples
                               else np.zeros(len(r['lanes']))),
        'mean_wait_s': float(np.mean(r['waits'])) if r['waits'] else None,
        'n_waits': len(r['waits']),
    }
    if not quiet:
        wait_txt = ('n/a' if stats['mean_wait_s'] is None
                    else f"{stats['mean_wait_s']:.1f}s")
        peak = int(r['totals'].max()) if r['totals'].size else 0
        print(f'  [{tag}] cap={r["cap"]} spawn={r["spawn"]}: '
              f'peak concurrent agents={peak}, '
              f'max per-lane queue length={stats["max_lane_occupancy"]}, '
              f'fraction of time some lane has >=1 waiting='
              f'{stats["busy_frac"]*100:.1f}%, '
              f'mean waiting={stats["mean_waiting"]:.2f}, '
              f'lane occupancy={stats["occupancy"]*100:.1f}%, '
              f'mean wait={wait_txt}')
    return stats


def main(argv=None):
    args = parse_args(argv)
    figs = args.figs_dir or _figs_dir()
    os.makedirs(figs, exist_ok=True)
    out_dir = make_run_dir(args.out_root, 'measure_queueing')
    prov = provenance_snapshot()
    print('[queue] calibrating the live store (current period) once...',
          flush=True)
    params = LS.calibrate_live_store(args.retail_path, 'current')
    data = LS.period_record(args.retail_path, 'current')

    # The stress replications continue the seed sequence after the nominal
    # ones, so the two load levels are independent samples.
    plan = ([('nominal', args.spawn, args.cap, r, args.seed + r)
             for r in range(args.reps)]
            + [('stress', args.stress_spawn, args.stress_cap, r,
                args.seed + args.reps + r) for r in range(args.reps)])
    print(f'[queue] {args.reps} replications per load level of warm-up '
          f'{args.warmup:.0f}s + collect {args.seconds:.0f}s simulated '
          f'(dt {args.dt}s) on {args.workers} worker(s)...', flush=True)
    t0 = time.perf_counter()
    runs = _map_runs(_level_run, len(plan), args.workers, plan, params, args)
    wall = time.perf_counter() - t0
    nom_reps, str_reps = runs[:args.reps], runs[args.reps:]

    # Keep one representative run for the time-series figure; report the
    # scalar statistics as means over replications.
    r_nom, r_str = nom_reps[0], str_reps[0]

    def _sd(xs):
        xs = np.asarray(xs, dtype=float)
        return float(xs.std(ddof=1)) if xs.size > 1 else 0.0

    def _pool(tag, reps):
        stats = [summarize(tag, r, quiet=True) for r in reps]
        lanes = reps[0]['lanes']
        fracs = np.array([s['busy_frac'] for s in stats], dtype=float)
        waiting = [s['mean_waiting'] for s in stats]
        occ = [s['occupancy'] for s in stats]
        # Mean wait is a per-customer average, so pool the customers of all
        # replications; the SD is over the replication means.
        all_waits = [w for r in reps for w in r['waits']]
        rep_waits = [s['mean_wait_s'] for s in stats
                     if s['mean_wait_s'] is not None]
        n_arr = int(sum(r['load']['arrivals'] for r in reps))
        n_balk = int(sum(r['load']['balked'] for r in reps))
        # A replication whose collection window is shorter than one sampling
        # interval yields no samples, so the peak is read off the guarded
        # per-replication values and pooled like every other statistic --
        # taking it from the representative run alone would report a peak
        # the load level did reach as one it never did.
        peaks = [int(r['totals'].max()) if r['totals'].size else 0
                 for r in reps]
        pooled = {
            'peak_agents': max(peaks),
            'max_lane_occupancy': max(s['max_lane_occupancy'] for s in stats),
            'busy_frac': float(fracs.mean()),
            'busy_frac_sd': _sd(fracs),
            'mean_waiting': float(np.mean(waiting)),
            'mean_waiting_sd': _sd(waiting),
            'occupancy': float(np.mean(occ)),
            'occupancy_sd': _sd(occ),
            'mean_wait_s': float(np.mean(all_waits)) if all_waits else None,
            'mean_wait_s_sd': _sd(rep_waits),
            'n_waits': len(all_waits),
            'mean_waiting_per_lane': {
                L: float(np.mean([s['mean_waiting_per_lane'][i] for s in stats]))
                for i, L in enumerate(lanes)},
            'wait_frac_per_lane': {
                L: float(np.mean([s['wait_frac_per_lane'][i] for s in stats]))
                for i, L in enumerate(lanes)},
            'occupancy_per_lane': {
                L: float(np.mean([s['occupancy_per_lane'][i] for s in stats]))
                for i, L in enumerate(lanes)},
            # Post-warm-up arrivals; balks are arrivals refused at the cap.
            'load': {'arrivals': n_arr, 'balked': n_balk,
                     'balked_frac': round(n_balk / max(n_arr, 1), 4)},
            'per_rep': [{'rep': r['rep'], 'seed': r['seed'],
                         'busy_frac': s['busy_frac'],
                         'mean_waiting': s['mean_waiting'],
                         'occupancy': s['occupancy'],
                         'mean_wait_s': s['mean_wait_s'],
                         'n_waits': s['n_waits'],
                         'max_lane_occupancy': s['max_lane_occupancy'],
                         'peak_agents': peak,
                         **r['load']}
                        for r, s, peak in zip(reps, stats, peaks)],
        }
        wait_txt = ('n/a' if pooled['mean_wait_s'] is None
                    else f"{pooled['mean_wait_s']:.1f}s")
        print(f'  [{tag}] over {len(reps)} reps: '
              f'busy-fraction mean={fracs.mean()*100:.1f}% '
              f'(sd {pooled["busy_frac_sd"]*100:.1f} pp, '
              f'min {fracs.min()*100:.1f}%, max {fracs.max()*100:.1f}%), '
              f'mean waiting={pooled["mean_waiting"]:.2f}, '
              f'lane occupancy={pooled["occupancy"]*100:.1f}%, '
              f'mean wait={wait_txt} over {pooled["n_waits"]} services, '
              f'max lane queue length={pooled["max_lane_occupancy"]}, '
              f'arrivals={n_arr}, balked={n_balk}')
        return pooled

    p_nom = _pool('nominal', nom_reps)
    p_str = _pool('stress', str_reps)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    for ax, r, title in ((axes[0], r_nom, f'Nominal (cap {r_nom["cap"]}, spawn {r_nom["spawn"]}/s)'),
                         (axes[1], r_str, f'Stress (cap {r_str["cap"]}, spawn {r_str["spawn"]}/s)')):
        at_lanes = r['q'].sum(axis=1)
        waiting = np.maximum(r['q'] - 1, 0).sum(axis=1)
        ax.plot(r['ts'], at_lanes, color='#E15759', lw=1.6,
                label='at lanes: in service + waiting')
        ax.plot(r['ts'], waiting, color='#B07AA1', lw=1.2,
                label='waiting (all lanes)')
        ax.plot(r['ts'], at_lanes + r['appr'].sum(axis=1), color='#4E79A7',
                lw=1.4, ls='--', label='at lanes + en route')
        ax.plot(r['ts'], r['totals'], color='#59A14F', lw=1.0, alpha=0.6,
                label='total active agents')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('simulated time after warm-up (s)'); ax.grid(alpha=0.25)
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel('count')
    fig.suptitle('Checkout-lane occupancy over a live run', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    figstyle.save(fig, 'queue_lengths', out_dir=figs)
    plt.close(fig)
    print(f"wrote {os.path.join(figs, 'queue_lengths.png')}")

    m_str, b_str = p_str['max_lane_occupancy'], p_str['busy_frac']
    verdict = ('emergent congestion is observable'
               if (m_str >= 2 or b_str > 0.05)
               else 'congestion negligible even under stress '
                    '-> downgrade claim to layout-aware lane assignment')
    print('[queue] VERDICT:', verdict)

    summary = {
        'nominal': {'cap': r_nom['cap'], 'spawn': r_nom['spawn'],
                    'n_reps': int(args.reps), **p_nom},
        'stress': {'cap': r_str['cap'], 'spawn': r_str['spawn'],
                   'n_reps': int(args.reps), **p_str},
        'protocol': {'mode': 'fixed_step', 'dt': float(args.dt),
                     'seed_base': int(args.seed),
                     'seeds': {'nominal': [r['seed'] for r in nom_reps],
                               'stress': [r['seed'] for r in str_reps]},
                     'warmup_s': float(args.warmup),
                     'collect_s': float(args.seconds),
                     'spawn': {'nominal': args.spawn,
                               'stress': args.stress_spawn},
                     'cap': {'nominal': args.cap, 'stress': args.stress_cap},
                     'workers': int(args.workers),
                     'sample_interval_s': SAMPLE_INTERVAL_S,
                     'n_reps': int(args.reps),
                     'framing': 'terminating',
                     'discipline': 'single-server FIFO per lane',
                     'busy_frac': 'fraction of samples in which some lane '
                                  'has >= 1 customer waiting',
                     'max_lane_occupancy': 'largest queue length at one '
                                           'lane (in service + waiting)',
                     'wait': 'simulated seconds from joining the lane '
                             'queue to service start'},
        'verdict': verdict,
        # The data the store was calibrated from.
        'data': data,
        # The paper's queue numbers are read straight out of the copy in
        # figs/, so the file carries the checkout that produced it and the
        # resolved numerics stack next to the protocol above; the protocol
        # block is this run's configuration.
        'provenance': {**prov, 'python': sys.version.split()[0],
                       'packages': package_versions()},
    }
    for path in (os.path.join(figs, 'queue_summary.json'),
                 os.path.join(out_dir, 'summary.json')):
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f'wrote {path}')
    write_sidecar(out_dir, {'experiment': 'measure_queueing',
                            'args': vars(args), 'wall_seconds': wall,
                            'figs_dir': figs, 'summary': summary},
                  provenance=prov)
    print(f'wall {wall:.0f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
