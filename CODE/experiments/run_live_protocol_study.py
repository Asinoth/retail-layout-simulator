"""Derive the live-measurement protocol from seeded runs (review R19).

The four live diagnostics (``run_abm_diagnostics``,
``run_structural_sensitivity``, ``measure_queueing``,
``run_validation_gof``) share one protocol -- arrival rate and customer
cap, deleted warm-up, collection window, and a stress level for the
queueing measurement -- and each value was set by a fixed rule applied to a
transient study of the live store. That study existed only as scratch
scripts and prose in the runners' docstrings. This runner IS the study:
seeded fixed-step replications of the store the diagnostics measure
(``experiments._live_store``: the UCI workbook's current period, naive
layout, lists drawn from the stocked invoices), from the store's first
trading hour (09:00), with every rule applied to their output and the
derived constants written as an artifact beside the constants the runners
actually use.

The analysis the protocol serves is steady-state replication-deletion
(Welch 1983; Law 2015, ch. 9): each replication deletes the empty-store
fill-up and averages over a window in which the arrival rate is constant.

Phases, in order (each value by a fixed rule; the thresholds are the
module constants below):

  1. Nominal grid. ``--reps`` (16) replications of ``--seconds`` (3,600 s)
     at every rate of ``--grid`` (0.25-0.30/s) and the nominal cap
     (NOMINAL_CAP, 45). For each rate:

       warm-up   the ANALYTIC rule: the first whole minute after which an
                 initially empty store's expected occupancy is within
                 WARMUP_LEVEL (1%) of its steady state. With arrivals at
                 a constant rate and visits that do not depend on one
                 another (an infinite-server queue; Eick, Massey & Whitt
                 1993) E[N(t)] / L = 1 - E[(S - t)+] / E[S] for visit
                 length S, so the rule needs only S's distribution,
                 estimated by Kaplan-Meier from the visits starting after
                 KM_DROP_BEFORE_S (still-open visits censored at the run's
                 end);
       cap share the share of the 1 s occupancy samples after that
                 warm-up at which the store is at the cap.

     NOMINAL rate: the highest grid rate whose cap share, and every lower
     grid rate's, is at most CAP_SHARE_MAX (1%). The protocol's warm-up is
     that rate's.

  2. Window, at the nominal rate: the shortest whole minute W such that
     every replication completes at least MIN_COHORT (300) visits that
     arrive after the warm-up and leave inside the window, and W spans at
     least MIN_MEDIAN_VISITS (10) median visits (Kaplan-Meier median).

  3. Drift after the warm-up, on the same runs: each replication's OLS
     slope of occupancy on time over the measured part (warm-up to
     warm-up + window), as customers per 1,000 s with a t-interval over
     replications, and as the relative change across the window, tested
     for EQUIVALENCE against the warm-up rule's own 1% (two one-sided
     t-tests, margin WARMUP_LEVEL); plus the first 300 s after the warm-up
     against the rest of the window (paired t).

  4. Double warm-up sensitivity: ``--reps`` replications at the nominal
     rate and cap, long enough for a second window after twice the
     warm-up, with the hour-of-day multiplier held at the first hour's so
     the rate does not change at 10:00, and then drained -- the run goes
     on until every agent that arrived by the end of the second window
     has left, the cohort rule of the live runners. The window's
     statistics after 2 x warm-up against after 1 x, paired per
     replication: the load (mean occupancy, lane utilisation, cap share,
     completed cohort visits, their mean length, throughput) and the
     outcomes the live runners report, over the window's drained cohort
     (revenue per visit, conversion, mean basket and revenue of the paying
     visits, mean visit length), the mean queue wait of the services
     starting in the window, and the perimeter ratio of the window's heat
     map. When the drift test of phase 3 fails, this comparison is what
     says whether the warm-up is long enough for the reported numbers.

  5. Stress level: ``--stress-reps`` (6) replications of
     ``--stress-seconds`` (1,800 s) at every rate of ``--stress-grid`` with
     the cap out of reach (STRESS_SCAN_CAP, 200); STRESS rate: the lowest
     grid rate whose mean lane utilisation after the nominal warm-up
     reaches STRESS_UTILISATION (0.7). Then ``--cap-reps`` (8)
     replications at that rate for every cap of ``--cap-grid``; STRESS cap:
     the smallest cap keeping utilisation at 0.7 while binding less than
     STRESS_CAP_SHARE_MAX (5%) of the time -- the cap raised only as far as
     needed. The analytic warm-up at the stress level is reported too.

``summary.json`` holds ``derived`` (the six constants), ``runner_constants``
(what each live runner uses) with ``matches`` / ``differences``,
``drift_equivalent`` and ``protocol_status`` -- ``matches`` says only that
the rules reproduce the runners' constants; whether the warm-up then meets
its own 1% criterion is ``drift_equivalent`` -- every
phase's per-rate or per-cap table, the rules and thresholds, the design
(seeds, replications, run lengths, grids, the start hour and its
multiplier) and the nominal replication-mean occupancy curve. The study
never changes a runner: when a rule gives a value other than the runner's,
``differences`` says so and the choice is left to the author.

Seeds are disjoint from every live runner's (their bases 1000-5000 and the
next few dozen): phase p, grid point k, replication r runs under
``--seed`` + PHASE_OFFSET[p] + 1000 k + r. Every replication is a seeded
fixed-step ``run_headless`` run, and results are put back in design order
before anything is aggregated, so the summary is the same for any
``--workers``.

    python -m experiments.run_live_protocol_study --workers 2
    python -m experiments.run_live_protocol_study --reps 2 --seconds 900 \\
        --grid 0.27 --stress-grid 0.7 --cap-grid 110 --stress-reps 1 \\
        --cap-reps 1 --stress-seconds 600          # smoke only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from experiments import _live_store as LS  # noqa: E402
from experiments._common import (make_run_dir, provenance_snapshot,  # noqa: E402
                                 write_sidecar)
from experiments.run_abm_diagnostics import perimeter_ratio  # noqa: E402

# --- the rules' thresholds ----------------------------------------------------
NOMINAL_CAP = 45
CAP_SHARE_MAX = 0.01          # nominal: the cap binds at most this often
WARMUP_LEVEL = 0.01           # warm-up: expected occupancy within 1% of L
KM_DROP_BEFORE_S = 300.0      # visits of the near-empty store left out
MIN_COHORT = 300              # window: completed cohort visits per rep
MIN_MEDIAN_VISITS = 10        # window: spans this many median visits
WINDOW_STEP_S = 60.0          # warm-up and window are whole minutes
WINDOW_MIN_S = 600.0
EARLY_S = 300.0               # drift: first part of the window vs the rest
STRESS_SCAN_CAP = 200         # stress scan: a cap out of reach
STRESS_UTILISATION = 0.7      # stress: mean lane utilisation to reach
STRESS_CAP_SHARE_MAX = 0.05   # stress: the cap binds less than this often
SAMPLE_S = 1.0                # occupancy / lane samples, simulated s
TOST_ALPHA = 0.05
DRAIN_CHUNK_S = 60.0          # double warm-up: drain checked this often
MAX_DRAIN_S = 3600.0          # ... and given up on after this long

# --- the default design ---------------------------------------------------------
NOMINAL_GRID = (0.25, 0.26, 0.27, 0.28, 0.29, 0.30)
STRESS_GRID = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80)
CAP_GRID = (60, 70, 80, 90, 100, 110, 120, 140, 160)
N_REPS = 16
STRESS_REPS = 6
CAP_REPS = 8
RUN_S = 3600.0
STRESS_RUN_S = 1800.0

SEED_BASE = 700_000
PHASE_OFFSET = {'nominal': 0, 'double_warmup': 100_000, 'stress_rate': 200_000,
                'stress_cap': 300_000}
# The live runners' seed bases, which the study's seeds must avoid; each
# runner uses at most a few dozen seeds above its base (and seed + 1).
RUNNER_SEED_BASES = (1000, 2000, 3000, 4000, 5000)
RUNNER_SEED_SPAN = 200

LIVE_RUNNERS = ('run_abm_diagnostics', 'run_structural_sensitivity',
                'measure_queueing', 'run_validation_gof')


# --- one replication ------------------------------------------------------------

def probe(params, rate, cap, seconds, seed, dt=0.04, freeze_profile=False,
          heat_windows=(), drain_after=None):
    """One seeded fixed-step run of the live store at ``rate`` and ``cap``.

    Samples, every SAMPLE_S simulated seconds, the number of agents in the
    store, the share of checkout lanes serving someone and how many
    services have started; records every visit's arrival and departure
    from the simulation's own state histories with its outcome (paid,
    basket size, revenue, read back from the exit roll-up), every queue
    wait, and the arrival time of the agents still inside at the end
    (right-censored visits). With ``freeze_profile`` the hour-of-day
    multiplier is held at the start hour's for the whole run.

    ``heat_windows`` -- (t0, t1) pairs -- adds the perimeter ratio
    (``run_abm_diagnostics.perimeter_ratio``, headline band) of each
    window's heat-map increment. With ``drain_after`` the run carries on
    past ``seconds``, arrivals included, until every agent that arrived by
    ``drain_after`` has left (checked every DRAIN_CHUNK_S, at most
    MAX_DRAIN_S), so a cohort arriving by then is followed to its exit."""
    shop = LS.build_live_store(params)
    sim = shop.customer_simulation
    sim.max_customers = int(cap)
    sim.spawn_rate = float(rate)
    sim.record_state_histories = True
    sim.completed_state_histories = []
    start_hour = int(sim.sim_clock_start_hour) % 24
    multiplier = float(np.asarray(sim.hourly_profile, dtype=float)[start_hour])
    if freeze_profile:
        sim.hourly_profile = np.full(24, multiplier)
    times, occ, serving, services = [], [], [], []
    lanes = [nm for nm in shop.floors[1]['walls']
             if nm == 'Checkout' or nm.startswith('Checkout_Lane')]
    marks = sorted({float(t) for w in heat_windows for t in w})
    heat_at = {}

    # Every exit appends one state-history record; the outcome read back
    # here after the simulation's own roll-up is appended in the same
    # call, so the two lists stay aligned. Reads only: no random number is
    # drawn and no state the model reads is changed.
    outcomes = []
    exit_orig = sim._process_customer_exit

    def _exit(cust):
        A = sim.analytics
        n_before = len(A.get('customer_revenues', []))
        exit_orig(cust)
        if cust.has_checked_out and len(A.get('customer_revenues',
                                              [])) > n_before:
            outcomes.append((1.0, float(A['basket_sizes'][-1]),
                             float(A['customer_revenues'][-1])))
        else:
            outcomes.append((0.0, 0.0, 0.0))

    sim._process_customer_exit = _exit

    def _sample(s):
        # A lane serves while its FIFO holds anyone: the head is in service.
        t = float(s.sim_time)
        times.append(t)
        occ.append(len(s.customers))
        busy = sum(1 for nm in lanes if (s.lane_queues or {}).get(nm))
        serving.append(busy / len(lanes) if lanes else 0.0)
        services.append(len(s.analytics.get('queue_wait_times', [])))
        for m in marks:
            if m not in heat_at and t >= m - dt / 2:
                heat_at[m] = np.array(s.heat_raw, dtype=float)

    sim.run_headless(seconds, dt=dt, seed=seed, callback=_sample,
                     callback_every_s=SAMPLE_S)
    drained = 0.0
    if drain_after is not None:
        def _cohort_left():
            return sum(1 for c in sim.customers
                       if float(c.spawn_sim_time) <= drain_after)
        while _cohort_left() and drained < MAX_DRAIN_S:
            sim.run_headless(DRAIN_CHUNK_S, dt=dt)
            drained += DRAIN_CHUNK_S
    errs = dict(getattr(sim, 'suppressed_errors', {}))
    if errs:
        raise RuntimeError(f"suppressed exceptions at rate {rate}, cap {cap}, "
                           f"seed {seed}: {errs}")
    visits = [(float(r['spawn_t']), float(r['exit_t']))
              for r in sim.completed_state_histories]
    if len(outcomes) != len(visits):
        raise RuntimeError("visit outcomes and state histories out of step")
    open_spawns = [float(c.spawn_sim_time) for c in sim.customers]
    ratios = {}
    for t0, t1 in heat_windows:
        if float(t0) in heat_at and float(t1) in heat_at:
            ratios[f'{float(t0):.0f}'] = perimeter_ratio(
                heat_at[float(t1)] - heat_at[float(t0)], shop.width,
                shop.height)['perimeter_interior_ratio']
    return {'rate': float(rate), 'cap': int(cap), 'seconds': float(seconds),
            'seed': int(seed), 'end': float(sim.sim_time),
            'times': np.asarray(times), 'occ': np.asarray(occ, dtype=float),
            'serving': np.asarray(serving, dtype=float),
            'services': np.asarray(services, dtype=np.int64),
            'waits': np.asarray(sim.analytics.get('queue_wait_times', []),
                                dtype=float),
            'visits': np.asarray(visits, dtype=float).reshape(-1, 2),
            'outcomes': np.asarray(outcomes, dtype=float).reshape(-1, 3),
            'open_spawns': np.asarray(open_spawns, dtype=float),
            'perimeter_ratio': ratios, 'drained_s': drained,
            'arrivals': int(sim.analytics.get('total_customers', 0))
            + int(sim.analytics.get('balked_arrivals', 0)),
            'balked': int(sim.analytics.get('balked_arrivals', 0)),
            'start_hour': start_hour, 'multiplier': multiplier,
            'frozen_profile': bool(freeze_profile)}


def _job(j, jobs, params, dt):
    job = jobs[j]
    return probe(params, job['rate'], job['cap'], job['seconds'], job['seed'],
                 dt=dt, freeze_profile=job.get('freeze', False),
                 heat_windows=job.get('heat_windows', ()),
                 drain_after=job.get('drain_after'))


def _map_runs(fn, n_runs, workers, *fn_args):
    """``[fn(j, *fn_args) for j in range(n_runs)]``, in design order; with
    ``workers`` > 1 in separate processes, results put back in order."""
    if workers <= 1:
        return [fn(j, *fn_args) for j in range(n_runs)]
    done = {}
    pool = ProcessPoolExecutor(max_workers=min(workers, n_runs))
    try:
        futures = {pool.submit(fn, j, *fn_args): j for j in range(n_runs)}
        for fut in as_completed(futures):
            done[futures[fut]] = fut.result()
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return [done[j] for j in sorted(done)]


# --- the rules ------------------------------------------------------------------

def km_survival(runs, drop_before=KM_DROP_BEFORE_S):
    """Kaplan-Meier survival of visit length over the visits starting after
    ``drop_before``: a finished visit is an event at its length, a visit
    still open at the run's end is censored at its length so far. Returns
    (event times, S just after each, n visits, n censored)."""
    dur, event = [], []
    for r in runs:
        v = r['visits']
        keep = v[:, 0] > drop_before
        dur.extend((v[keep, 1] - v[keep, 0]).tolist())
        event.extend([True] * int(keep.sum()))
        o = r['open_spawns']
        o = o[o > drop_before]
        dur.extend((r['end'] - o).tolist())
        event.extend([False] * int(o.size))
    dur = np.asarray(dur, dtype=float)
    event = np.asarray(event, dtype=bool)
    if not dur.size or not event.any():
        return np.zeros(0), np.zeros(0), int(dur.size), int((~event).sum())
    order = np.argsort(dur, kind='stable')
    dur, event = dur[order], event[order]
    uniq = np.unique(dur[event])
    at_risk = dur.size - np.searchsorted(dur, uniq, side='left')
    deaths = np.array([np.count_nonzero((dur == u) & event) for u in uniq])
    surv = np.cumprod(1.0 - deaths / at_risk)
    return uniq, surv, int(dur.size), int((~event).sum())


def _survival_grid(uniq, surv, step=1.0):
    """S on a ``step`` grid from 0 to just past the last event time; the
    integrals below end there, i.e. S is taken as 0 beyond the longest
    finished visit."""
    grid = np.arange(0.0, float(uniq[-1]) + 5.0, step)
    idx = np.searchsorted(uniq, grid, side='right') - 1
    s = np.where(idx >= 0, surv[np.clip(idx, 0, None)], 1.0)
    return grid, s


def fill_time(uniq, surv, level=WARMUP_LEVEL):
    """First t (1 s grid) at which an initially empty infinite-server
    store's expected occupancy is within ``level`` of steady state:
    int_t^inf S / int_0^inf S <= level. Also returns E[S]. Rounded up to a
    whole minute (``whole_minutes``) the 1 s grid gives the same warm-up as
    the 5 s grid of the scratch study it replaces, 60 being a multiple of
    5."""
    if not uniq.size:
        return None, None
    grid, s = _survival_grid(uniq, surv)
    tail = s[::-1].cumsum()[::-1]
    mean = float(tail[0])
    hit = np.flatnonzero(tail / tail[0] <= level)
    return (float(grid[hit[0]]) if hit.size else None), mean


def km_median(uniq, surv):
    hit = np.flatnonzero(surv <= 0.5)
    return float(uniq[hit[0]]) if hit.size else None


def whole_minutes(t):
    return None if t is None else math.ceil(t / WINDOW_STEP_S) * WINDOW_STEP_S


def cap_share(runs, cap, warm):
    at = tot = 0
    for r in runs:
        m = r['times'] >= warm
        tot += int(m.sum())
        at += int((r['occ'][m] >= cap).sum())
    return at / tot if tot else None


def utilisation(runs, warm, end=None):
    vals = [r['serving'][(r['times'] >= warm)
                         & ((r['times'] <= end) if end else True)]
            for r in runs]
    vals = np.concatenate(vals) if vals else np.zeros(0)
    return float(vals.mean()) if vals.size else None


def cohort_counts(runs, warm, window):
    """Per replication: visits arriving in (warm, warm + window] that left
    by warm + window."""
    return [int(np.count_nonzero((r['visits'][:, 0] > warm)
                                 & (r['visits'][:, 0] <= warm + window)
                                 & (r['visits'][:, 1] <= warm + window)))
            for r in runs]


def naive_median_visit(runs, warm):
    d = [v[1] - v[0] for r in runs for v in r['visits'] if v[0] > warm]
    return float(np.median(d)) if d else None


def rate_row(runs, cap):
    """The rules' inputs at one grid rate."""
    uniq, surv, n, cens = km_survival(runs)
    fills = {lvl: fill_time(uniq, surv, lvl)[0] for lvl in (0.05, 0.02, 0.01)}
    fill1, mean_s = fill_time(uniq, surv, WARMUP_LEVEL)
    warm = whole_minutes(fill1)
    return {'reps': len(runs), 'visits_km': n, 'censored': cens,
            'km_mean_visit_s': mean_s,
            'km_median_visit_s': km_median(uniq, surv) if uniq.size else None,
            'fill_5pct_s': fills[0.05], 'fill_2pct_s': fills[0.02],
            'fill_1pct_s': fills[0.01], 'warmup_s': warm,
            'cap_share_after_warmup': (cap_share(runs, cap, warm)
                                       if warm is not None else None),
            'utilisation_after_warmup': (utilisation(runs, warm)
                                         if warm is not None else None),
            'mean_occupancy_after_warmup': (
                float(np.mean(np.concatenate(
                    [r['occ'][r['times'] >= warm] for r in runs])))
                if warm is not None else None),
            'balked_frac': (sum(r['balked'] for r in runs)
                            / max(sum(r['arrivals'] for r in runs), 1))}


def nominal_rule(table):
    """Highest grid rate whose cap share, and every lower rate's, is at most
    CAP_SHARE_MAX."""
    chosen = None
    for rate in sorted(table):
        share = table[rate]['cap_share_after_warmup']
        if share is None or share > CAP_SHARE_MAX:
            break
        chosen = rate
    return chosen


def window_rule(runs, warm, median, horizon):
    """Shortest whole minute with at least MIN_COHORT completed cohort
    visits in every replication and at least MIN_MEDIAN_VISITS medians."""
    counts_at = {}
    chosen = None
    W = WINDOW_MIN_S
    while warm + W <= horizon + 1e-9:
        c = cohort_counts(runs, warm, W)
        counts_at[f'{W:.0f}'] = {'min': int(min(c)),
                                 'mean': float(np.mean(c))}
        if (min(c) >= MIN_COHORT and median is not None
                and W >= MIN_MEDIAN_VISITS * median):
            chosen = float(W)
            break
        W += WINDOW_STEP_S
    return chosen, counts_at


def _t_interval(x, level=0.95):
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2:
        return float(x.mean()) if n else None, None
    se = float(x.std(ddof=1)) / math.sqrt(n)
    return float(x.mean()), float(stats.t.ppf(0.5 + level / 2, n - 1) * se)


def drift(runs, warm, window, margin=WARMUP_LEVEL, alpha=TOST_ALPHA):
    """Occupancy drift over the measured part of each replication: the OLS
    slope per 1,000 s with a t-interval, the relative change across the
    window (slope x window / the window's mean occupancy) with 90% and 95%
    intervals and two one-sided tests of |change| < ``margin``, and the
    first EARLY_S after the warm-up against the rest (paired t)."""
    slopes, rel, early, late = [], [], [], []
    for r in runs:
        m = (r['times'] >= warm) & (r['times'] <= warm + window)
        t, o = r['times'][m], r['occ'][m]
        if t.size < 3:
            continue
        b = float(np.polyfit(t, o, 1)[0])
        slopes.append(b * 1000.0)
        rel.append(b * window / max(float(o.mean()), 1e-9))
        early.append(float(o[t < warm + EARLY_S].mean()))
        late.append(float(o[t >= warm + EARLY_S].mean()))
    if len(slopes) < 2:
        return {'n_reps': len(slopes)}
    slopes, rel = np.asarray(slopes), np.asarray(rel)
    n = rel.size
    se = float(rel.std(ddof=1)) / math.sqrt(n)
    mean = float(rel.mean())
    sm, shw = _t_interval(slopes)
    _, hw95 = _t_interval(rel, 0.95)
    _, hw90 = _t_interval(rel, 1 - 2 * alpha)
    if se > 0:
        p_lower = float(stats.t.sf((mean + margin) / se, n - 1))
        p_upper = float(stats.t.cdf((mean - margin) / se, n - 1))
        p_zero = float(2 * stats.t.sf(abs(mean) / se, n - 1))
    else:
        p_lower = p_upper = 0.0 if abs(mean) < margin else 1.0
        p_zero = 1.0 if mean == 0 else 0.0
    p_tost = max(p_lower, p_upper)
    pe = stats.ttest_rel(early, late)
    return {'n_reps': int(n),
            'slope_per_1000s_mean': sm, 'slope_per_1000s_ci95': shw,
            'slope_per_1000s_sd': float(slopes.std(ddof=1)),
            'slope_p': p_zero,
            'relative_change_over_window_mean': mean,
            'relative_change_ci95': hw95, 'relative_change_ci90': hw90,
            'tost_margin': margin, 'tost_alpha': alpha,
            'tost_p': p_tost,
            'equivalent_within_margin': bool(p_tost < alpha),
            'early_s': EARLY_S,
            'early_mean_occupancy': float(np.mean(early)),
            'late_mean_occupancy': float(np.mean(late)),
            'early_minus_late_p': float(pe.pvalue)}


def _mean(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean()) if x.size else None


def window_stats(r, t0, window):
    """One replication's statistics over (t0, t0 + window].

    The load: mean occupancy, lane utilisation, cap share, the cohort
    visits that also LEFT inside the window and their mean length, exits
    per 1,000 s. The outcomes the live runners report, over the window's
    whole cohort -- every visit arriving in the window, followed to its
    exit (a run that was drained holds them all; ``cohort_censored``
    counts any still open) -- revenue per visit (unpaid visits count 0),
    conversion, the paying visits' mean basket and revenue, and the mean
    visit length; the mean wait of the services that started in the
    window; and, when the run recorded it, the window's perimeter
    ratio."""
    m = (r['times'] > t0) & (r['times'] <= t0 + window)
    v = r['visits']
    arrived = (v[:, 0] > t0) & (v[:, 0] <= t0 + window)
    cohort = v[arrived & (v[:, 1] <= t0 + window)]
    exits = int(np.count_nonzero((v[:, 1] > t0) & (v[:, 1] <= t0 + window)))
    out = {'mean_occupancy': float(r['occ'][m].mean()) if m.any() else None,
           'utilisation': float(r['serving'][m].mean()) if m.any() else None,
           'cap_share': (float((r['occ'][m] >= r['cap']).mean())
                         if m.any() else None),
           'cohort_completed': int(cohort.shape[0]),
           'cohort_mean_visit_s': (float((cohort[:, 1] - cohort[:, 0]).mean())
                                   if cohort.size else None),
           'throughput_per_1000s': exits * 1000.0 / window}
    oc = r.get('outcomes')
    if oc is not None and len(oc) == len(v):
        o = oc[arrived]
        paid = o[:, 0] > 0
        o_spawn = r['open_spawns']
        out.update({
            'cohort_censored': int(np.count_nonzero(
                (o_spawn > t0) & (o_spawn <= t0 + window))),
            'revenue_per_visit': _mean(o[:, 2]),
            'conversion': _mean(paid.astype(float)),
            'mean_basket_paying': _mean(o[paid, 1]),
            'revenue_per_paying_visit': _mean(o[paid, 2]),
            'cohort_mean_visit_all_s': _mean(v[arrived, 1] - v[arrived, 0]),
        })
    if r.get('services') is not None and len(r['services']) == len(r['times']):
        before = r['times'] <= t0
        upto = r['times'] <= t0 + window
        c0 = int(r['services'][before][-1]) if before.any() else 0
        c1 = int(r['services'][upto][-1]) if upto.any() else c0
        out['mean_queue_wait_s'] = _mean(r['waits'][c0:c1])
    if f'{float(t0):.0f}' in (r.get('perimeter_ratio') or {}):
        out['perimeter_ratio'] = r['perimeter_ratio'][f'{float(t0):.0f}']
    return out


def double_warmup(runs, warm, window):
    """The window's statistics after 2 x warm-up against after 1 x, paired
    per replication: mean of each, the paired difference with a 95%
    t-interval, and the difference as a percentage of the 1 x value."""
    once = [window_stats(r, warm, window) for r in runs]
    twice = [window_stats(r, 2 * warm, window) for r in runs]
    out = {}
    for key in once[0] if once else ():
        if key == 'cohort_censored':
            out[key] = int(sum(o[key] + t[key] for o, t in zip(once, twice)))
            continue
        a = [o[key] for o, t in zip(once, twice)
             if o[key] is not None and t[key] is not None]
        b = [t[key] for o, t in zip(once, twice)
             if o[key] is not None and t[key] is not None]
        if not a:
            out[key] = None
            continue
        d = np.asarray(b) - np.asarray(a)
        dm, dhw = _t_interval(d)
        base = float(np.mean(a))
        out[key] = {'after_1x_warmup': base, 'after_2x_warmup': float(np.mean(b)),
                    'diff_mean': dm, 'diff_ci95': dhw,
                    'diff_pct': (100.0 * dm / base) if base else None,
                    'p': (float(stats.ttest_rel(b, a).pvalue)
                          if len(a) > 1 and np.any(d != d[0]) else None)}
    return out


# What the live runners report, compared after 1 x and 2 x the warm-up.
HEADLINE_OUTCOMES = ('revenue_per_visit', 'conversion', 'mean_basket_paying',
                     'revenue_per_paying_visit', 'cohort_mean_visit_all_s',
                     'mean_queue_wait_s', 'perimeter_ratio')


# --- the design -----------------------------------------------------------------

def _seed(phase, k, r, base):
    return base + PHASE_OFFSET[phase] + 1000 * k + r


def check_seed_ranges(base, n_grid, n_reps):
    """Refuse a design whose seeds (and seed + 1, the arrival stream) would
    meet a live runner's."""
    lo = base
    hi = base + max(PHASE_OFFSET.values()) + 1000 * n_grid + n_reps + 1
    for rb in RUNNER_SEED_BASES:
        if lo <= rb + RUNNER_SEED_SPAN and rb <= hi:
            raise ValueError(f"study seeds {lo}..{hi} overlap a live runner's "
                             f"seeds from {rb}")


def runner_constants():
    """The protocol constants each live runner actually uses."""
    import importlib
    out = {}
    for name in LIVE_RUNNERS:
        mod = importlib.import_module(f'experiments.{name}')
        out[name] = {k: getattr(mod, k) for k in
                     ('NOMINAL_SPAWN', 'NOMINAL_CAP', 'WARMUP_S', 'COLLECT_S',
                      'STRESS_SPAWN', 'STRESS_CAP') if hasattr(mod, k)}
    return out


def compare(derived, used):
    """Where a runner's constant differs from the study's value."""
    diffs = []
    for runner, consts in used.items():
        for k, v in consts.items():
            d = derived.get(k)
            if d is None or not np.isclose(float(v), float(d)):
                diffs.append({'runner': runner, 'constant': k,
                              'runner_value': v, 'study_value': d})
    return diffs


def protocol_status(matches, drift_ok):
    """One label for what the study established: whether its rules give the
    runners' constants, and whether the derived warm-up meets its own 1%
    criterion afterwards."""
    head = 'reproduced' if matches else 'differs_from_runners'
    tail = {True: 'drift_equivalent', False: 'drift_not_equivalent',
            None: 'drift_not_tested'}[drift_ok]
    return f'{head}_{tail}'


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--grid', type=float, nargs='+', default=list(NOMINAL_GRID))
    p.add_argument('--cap', type=int, default=NOMINAL_CAP)
    p.add_argument('--reps', type=int, default=N_REPS)
    p.add_argument('--seconds', type=float, default=RUN_S)
    p.add_argument('--stress-grid', type=float, nargs='+',
                   default=list(STRESS_GRID))
    p.add_argument('--cap-grid', type=int, nargs='+', default=list(CAP_GRID))
    p.add_argument('--stress-reps', type=int, default=STRESS_REPS)
    p.add_argument('--cap-reps', type=int, default=CAP_REPS)
    p.add_argument('--stress-seconds', type=float, default=STRESS_RUN_S)
    p.add_argument('--dt', type=float, default=0.04)
    p.add_argument('--seed', type=int, default=SEED_BASE)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--retail-path', type=str, default=None)
    p.add_argument('--out-root', type=str,
                   default=os.path.join(_HERE, 'results'))
    args = p.parse_args(argv)
    if args.workers < 1:
        p.error("--workers must be >= 1")
    if min(args.reps, args.stress_reps, args.cap_reps) < 1:
        p.error("replication counts must be >= 1")
    if min(args.seconds, args.stress_seconds, args.dt) <= 0:
        p.error("--seconds, --stress-seconds and --dt must be positive")
    args.grid = sorted(set(args.grid))
    args.stress_grid = sorted(set(args.stress_grid))
    args.cap_grid = sorted(set(args.cap_grid))
    check_seed_ranges(args.seed,
                      max(len(args.grid), len(args.stress_grid),
                          len(args.cap_grid)),
                      max(args.reps, args.stress_reps, args.cap_reps))
    return args


def _curve(runs, every=10):
    """Replication-mean occupancy every ``every`` samples, for plotting."""
    n = min(len(r['times']) for r in runs)
    arr = np.array([r['occ'][:n] for r in runs])
    t = runs[0]['times'][:n]
    return {'t': [round(float(x), 2) for x in t[::every]],
            'mean_occupancy': [round(float(x), 3)
                               for x in arr.mean(axis=0)[::every]]}


def _phase(name, jobs, params, args):
    print(f"[protocol] {name}: {len(jobs)} runs on {args.workers} worker(s)",
          flush=True)
    t0 = time.perf_counter()
    runs = _map_runs(_job, len(jobs), args.workers, jobs, params, args.dt)
    print(f"[protocol] {name}: {time.perf_counter() - t0:.0f}s", flush=True)
    return runs


def main(argv=None):
    args = parse_args(argv)
    out_dir = make_run_dir(args.out_root, 'live_protocol_study')
    prov = provenance_snapshot()
    params = LS.calibrate_live_store(args.retail_path, 'current')
    data = LS.period_record(args.retail_path, 'current')
    t_start = time.perf_counter()

    # 1. Nominal grid.
    jobs = [{'rate': rate, 'cap': args.cap, 'seconds': args.seconds,
             'seed': _seed('nominal', k, r, args.seed)}
            for k, rate in enumerate(args.grid) for r in range(args.reps)]
    runs = _phase('nominal grid', jobs, params, args)
    by_rate = {rate: runs[k * args.reps:(k + 1) * args.reps]
               for k, rate in enumerate(args.grid)}
    table = {rate: rate_row(by_rate[rate], args.cap) for rate in args.grid}
    nominal = nominal_rule(table)
    derived = {'NOMINAL_SPAWN': nominal, 'NOMINAL_CAP': args.cap,
               'WARMUP_S': None, 'COLLECT_S': None,
               'STRESS_SPAWN': None, 'STRESS_CAP': None}
    window_info, drift_info, dbl_info = {}, {}, {}
    if nominal is not None:
        warm = table[nominal]['warmup_s']
        derived['WARMUP_S'] = warm
        median = table[nominal]['km_median_visit_s']
        window, counts_at = window_rule(by_rate[nominal], warm, median,
                                        args.seconds)
        derived['COLLECT_S'] = window
        window_info = {'rate': nominal, 'warmup_s': warm,
                       'km_median_visit_s': median,
                       'naive_median_visit_s': naive_median_visit(
                           by_rate[nominal], warm),
                       'cohort_counts': counts_at, 'window_s': window,
                       'cohort_min_at_window': (counts_at.get(f'{window:.0f}')
                                                if window else None)}
        # Visit length once the store has filled against the pool the rule
        # used: a longer visit in the full store is the infinite-server
        # assumption behind the analytic rule failing, and a reason the
        # occupancy can still drift after the warm-up.
        u_after, s_after, _, _ = km_survival(by_rate[nominal], warm)
        window_info['km_mean_visit_s_rule_pool'] = \
            table[nominal]['km_mean_visit_s']
        window_info['km_mean_visit_s_after_warmup'] = (
            fill_time(u_after, s_after)[1] if u_after.size else None)
        if window:
            drift_info = drift(by_rate[nominal], warm, window)
            # 4. Double warm-up, rate held at the first hour's, drained.
            end2 = 2 * warm + window
            jobs = [{'rate': nominal, 'cap': args.cap,
                     'seconds': end2, 'freeze': True, 'drain_after': end2,
                     'heat_windows': [(warm, warm + window),
                                      (2 * warm, end2)],
                     'seed': _seed('double_warmup', 0, r, args.seed)}
                    for r in range(args.reps)]
            dbl_runs = _phase('double warm-up', jobs, params, args)
            dbl_info = {'rate': nominal, 'cap': args.cap, 'warmup_s': warm,
                        'window_s': window, 'reps': args.reps,
                        'run_s': end2,
                        'profile': 'held at the start hour',
                        'cohort': 'window_arrivals_drained',
                        'mean_drain_s': float(np.mean(
                            [r['drained_s'] for r in dbl_runs])),
                        'headline_outcomes': list(HEADLINE_OUTCOMES),
                        'stats': double_warmup(dbl_runs, warm, window)}

    # 5. Stress level, after the nominal warm-up -- or, when no grid rate
    # passed the nominal rule, after the longest warm-up of the grid.
    stress_info = {}
    if derived['WARMUP_S'] is not None:
        warm_for_util, warm_source = derived['WARMUP_S'], 'nominal warm-up'
    else:
        warms = [row['warmup_s'] for row in table.values()
                 if row['warmup_s'] is not None]
        warm_for_util = max(warms) if warms else 0.0
        warm_source = 'longest grid warm-up (no nominal rate)'
    jobs = [{'rate': rate, 'cap': STRESS_SCAN_CAP,
             'seconds': args.stress_seconds,
             'seed': _seed('stress_rate', k, r, args.seed)}
            for k, rate in enumerate(args.stress_grid)
            for r in range(args.stress_reps)]
    s_runs = _phase('stress rate scan', jobs, params, args)
    s_by = {rate: s_runs[k * args.stress_reps:(k + 1) * args.stress_reps]
            for k, rate in enumerate(args.stress_grid)}
    rate_scan = {f'{rate:.2f}': {'utilisation': utilisation(s_by[rate],
                                                            warm_for_util),
                                 'cap_share': cap_share(s_by[rate],
                                                        STRESS_SCAN_CAP,
                                                        warm_for_util)}
                 for rate in args.stress_grid}
    reaching = [rate for rate in args.stress_grid
                if (rate_scan[f'{rate:.2f}']['utilisation'] or 0.0)
                >= STRESS_UTILISATION]
    stress_rate = min(reaching) if reaching else None
    derived['STRESS_SPAWN'] = stress_rate
    cap_scan = {}
    if stress_rate is not None:
        jobs = [{'rate': stress_rate, 'cap': cap,
                 'seconds': args.stress_seconds,
                 'seed': _seed('stress_cap', k, r, args.seed)}
                for k, cap in enumerate(args.cap_grid)
                for r in range(args.cap_reps)]
        c_runs = _phase('stress cap scan', jobs, params, args)
        c_by = {cap: c_runs[k * args.cap_reps:(k + 1) * args.cap_reps]
                for k, cap in enumerate(args.cap_grid)}
        for cap in args.cap_grid:
            cap_scan[str(cap)] = {'utilisation': utilisation(c_by[cap],
                                                             warm_for_util),
                                  'cap_share': cap_share(c_by[cap], cap,
                                                         warm_for_util)}
        ok = [cap for cap in args.cap_grid
              if (cap_scan[str(cap)]['utilisation'] or 0.0)
              >= STRESS_UTILISATION
              and (cap_scan[str(cap)]['cap_share'] if
                   cap_scan[str(cap)]['cap_share'] is not None else 1.0)
              < STRESS_CAP_SHARE_MAX]
        derived['STRESS_CAP'] = min(ok) if ok else None
        if derived['STRESS_CAP'] is not None:
            srow = rate_row(c_by[derived['STRESS_CAP']], derived['STRESS_CAP'])
            stress_info['analytic_warmup_at_stress'] = {
                k: srow[k] for k in ('fill_1pct_s', 'warmup_s',
                                     'km_mean_visit_s', 'censored')}
    stress_info.update({'rate_scan': rate_scan, 'cap_scan': cap_scan,
                        'utilisation_after_s': warm_for_util,
                        'utilisation_after': warm_source})
    wall = time.perf_counter() - t_start

    used = runner_constants()
    diffs = compare(derived, used)
    first = runs[0]
    drift_ok = (drift_info.get('equivalent_within_margin')
                if drift_info.get('n_reps', 0) > 1 else None)
    summary = {
        'derived': derived,
        'runner_constants': used,
        # The rules reproduce the runners' constants. That is all it says.
        'matches': not diffs,
        'differences': diffs,
        # Whether the occupancy after the derived warm-up is within the
        # warm-up rule's own 1% across the window (TOST); None when the
        # drift could not be computed. When it is False the protocol is
        # reproduced but not validated by its own criterion, and the
        # double_warmup block is the evidence on the reported outcomes.
        'drift_equivalent': drift_ok,
        'protocol_status': protocol_status(not diffs, drift_ok),
        'rules': {
            'nominal': (f'highest grid rate whose cap share after its own '
                        f'warm-up, and every lower rate\'s, is <= '
                        f'{CAP_SHARE_MAX}'),
            'warmup': (f'first whole minute with 1 - E[(S-t)+]/E[S] within '
                       f'{WARMUP_LEVEL} of 1, S by Kaplan-Meier over visits '
                       f'starting after {KM_DROP_BEFORE_S:.0f} s, open visits '
                       f'censored at the run end'),
            'window': (f'shortest whole minute >= {WINDOW_MIN_S:.0f} s with '
                       f'>= {MIN_COHORT} completed cohort visits in every '
                       f'replication and >= {MIN_MEDIAN_VISITS} KM-median '
                       f'visits'),
            'stress': (f'lowest grid rate with mean lane utilisation >= '
                       f'{STRESS_UTILISATION} at cap {STRESS_SCAN_CAP}; then '
                       f'the smallest grid cap keeping utilisation >= '
                       f'{STRESS_UTILISATION} with the cap binding < '
                       f'{STRESS_CAP_SHARE_MAX}'),
            'drift': (f'per-replication OLS slope over the window; TOST of '
                      f'the relative change across the window against '
                      f'+-{WARMUP_LEVEL} at alpha {TOST_ALPHA}'),
            'thresholds': {'CAP_SHARE_MAX': CAP_SHARE_MAX,
                           'WARMUP_LEVEL': WARMUP_LEVEL,
                           'KM_DROP_BEFORE_S': KM_DROP_BEFORE_S,
                           'MIN_COHORT': MIN_COHORT,
                           'MIN_MEDIAN_VISITS': MIN_MEDIAN_VISITS,
                           'WINDOW_STEP_S': WINDOW_STEP_S,
                           'STRESS_SCAN_CAP': STRESS_SCAN_CAP,
                           'STRESS_UTILISATION': STRESS_UTILISATION,
                           'STRESS_CAP_SHARE_MAX': STRESS_CAP_SHARE_MAX,
                           'TOST_ALPHA': TOST_ALPHA}},
        'nominal_grid': {f'{rate:.2f}': table[rate] for rate in args.grid},
        'window': window_info,
        'drift': drift_info,
        'double_warmup': dbl_info,
        'stress': stress_info,
        'nominal_occupancy_curve': (_curve(by_rate[nominal])
                                    if nominal is not None else None),
        'design': {'mode': 'fixed_step', 'dt': args.dt,
                   'sample_s': SAMPLE_S, 'seed_base': args.seed,
                   'phase_offsets': PHASE_OFFSET,
                   'seed_rule': 'seed_base + phase_offset + 1000 k + r',
                   'nominal': {'grid': args.grid, 'cap': args.cap,
                               'reps': args.reps, 'run_s': args.seconds},
                   'stress': {'grid': args.stress_grid,
                              'scan_cap': STRESS_SCAN_CAP,
                              'cap_grid': args.cap_grid,
                              'rate_reps': args.stress_reps,
                              'cap_reps': args.cap_reps,
                              'run_s': args.stress_seconds},
                   'start_hour': first['start_hour'],
                   'profile_multiplier': first['multiplier'],
                   'n_runs': (len(runs) + len(s_runs)
                              + (args.reps if dbl_info else 0)
                              + (len(args.cap_grid) * args.cap_reps
                                 if cap_scan else 0)),
                   'workers': args.workers,
                   'framing': 'steady_state_replication_deletion'},
        'data': data,
    }
    write_sidecar(out_dir, {'experiment': 'live_protocol_study',
                            'args': vars(args), 'wall_seconds': wall},
                  provenance=prov)
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({'derived': derived, 'matches': not diffs,
                      'differences': diffs,
                      'protocol_status': summary['protocol_status']},
                     indent=1))
    if drift_info.get('n_reps', 0) > 1:
        print(f"drift: {drift_info['slope_per_1000s_mean']:.2f} +- "
              f"{drift_info['slope_per_1000s_ci95']:.2f} customers per 1000 s; "
              f"change across the window "
              f"{100 * drift_info['relative_change_over_window_mean']:.1f}% "
              f"(90% CI +-{100 * drift_info['relative_change_ci90']:.1f}%), "
              f"TOST vs +-{100 * WARMUP_LEVEL:.0f}% p="
              f"{drift_info['tost_p']:.3f}")
    for key in HEADLINE_OUTCOMES:
        row = (dbl_info.get('stats') or {}).get(key)
        if row and row.get('diff_ci95') is not None:
            print(f"double warm-up {key}: {row['after_1x_warmup']:.4g} -> "
                  f"{row['after_2x_warmup']:.4g} (diff {row['diff_mean']:.3g}"
                  f" +- {row['diff_ci95']:.3g})")
    print(f"wall {wall:.0f}s; artifacts in: {out_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
