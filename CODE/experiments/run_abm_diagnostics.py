"""ABM methodology diagnostics from a live run (audit R6.2 + R6.4).

Two ABM-reviewer questions, answered from one calibrated live simulation:

  R6.2  First-order Markov adequacy. The paper computes an absorbing
        first-order Markov summary of the agents' embedded jump chain.
        Hui (2009) argues shopper paths carry longer memory, so we test
        whether a SECOND-order chain would change the next-state
        distribution appreciably. The simulation records every agent's
        complete state sequence when it leaves, closed by its absorbing
        outcome ('purchased' / 'abandoned'). For each current state we
        compare the next-state distribution conditioned on one vs. two
        prior states, via the information gain H(next|cur) -
        H(next|cur,prev) and the mean total-variation distance between
        the first- and second-order conditionals. Both conditionals are
        estimated from the same transitions (those with a predecessor),
        entropies carry the Miller-Madow correction, and each statistic
        is set against a permutation null that shuffles the predecessor
        labels within each current state, so what is reported is the
        memory beyond what the estimator shows for a first-order chain.

  R6.4  Emergent traffic pattern. Larson (2005) documents perimeter-
        dominant supermarket traffic. We measure the emergent
        perimeter-to-interior foot-traffic ratio from the accumulated
        heat map -- a quantitative face-validation number, not a picture.
        The ratio averages each band over ALL of its cells, including the
        cells under fixtures, where no one walks, so on its own it mixes
        where people walk with how much of each band is walkable. Every
        replication therefore also reports a geometric null: the value
        the same function returns for a heat map uniform over the walkable
        cells (floor 1's pathfinding grid after the run, obstacles
        inflated by the agent radius), i.e. the ratio the floor plan
        produces with no routing at all. Observed / null is the part of
        the ratio the agents' routes add; both are reported per
        perimeter band width too.

The store is ``experiments._live_store``'s: the UCI workbook's current
period (its last sheet, every row), laid out naively and seeded with the
shop so each agent's shopping list is the stocked part of one invoice
of that period. ``--retail-path`` names the workbook; without it
``dataset_paths.uci_workbook()`` finds it. The summary records the period,
its date range and the workbook.

Every replication is a fixed-step headless run: the agent model advances
through ``CustomerFlowSimulation.run_headless`` (the same ``step`` the GUI
loop calls, with no sleeping) for ``--warmup`` + ``--seconds`` simulated
seconds under its own seed, replication r using ``--seed`` + r. A
replication is therefore reproducible, and ``--workers`` only decides how
many run at once: results are put back in replication order before
anything is aggregated, so the summary is the same for any worker count.

The defaults were set by a transient study on this store (3,600 s runs
from 09:00 on seeds disjoint from the runners' own), each value by a fixed
rule:

  --spawn 0.27 --cap 45  The highest arrival rate on a 0.01/s grid at which
      occupancy stays below the cap at least 99% of the time after the
      warm-up. The cap binds 0.38% of the time at 0.26/s, 0.88% at
      0.27/s and 1.27% at 0.28/s (16 replications each). The calibrated
      hour-of-day profile scales the rate by 0.744 in the 09:00-10:00
      hour a run falls in.
  --warmup 1020  The first whole minute after which the expected
      occupancy of the store, which starts empty, is within 1% of its
      steady state. With arrivals at a constant rate and visits that do
      not depend on one another (an infinite-server queue; Eick, Massey
      and Whitt 1993) that expectation is E[N(t)] / L = 1 - E[(S - t)+] /
      E[S] for visit length S, so the rule needs only the visit-length
      distribution, estimated by Kaplan-Meier from the study's visits at
      the nominal rate (still-open visits censored at the run's end;
      mean 163 s). The 1% point is 965 s, rounded up to 1,020 s. Visits
      do interact through the checkout lanes and the cap, but at this
      load a lane is busy 29% of the time and the cap binds under 1%,
      and the occupancy after 1,020 s shows no drift: the per-replication
      slope over the measured part of a run is 0.9 +- 1.9 customers per
      1,000 s (p = 0.09), and the first 300 s after the warm-up average
      32.1 customers against 32.8 over the rest of the window (paired
      p = 0.42). The same criterion gives 895-1,030 s at every rate from
      0.26 to 0.32/s. Welch's procedure and MSER-5, which set the
      previous protocol's warm-up, did not give a stable answer on this
      store's 16 replications: across those neighbouring rates Welch's
      doubled settle time ranged 840-3,360 s and MSER-5's truncation
      435-1,695 s, with no trend in the rate -- the replication-mean
      curve's slow wander decided them, not the transient.
  --seconds 1860  The shortest whole-minute window in which every study
      replication completes at least 300 visits (agents that arrive after
      the warm-up and leave inside the window; the fewest was 306 and the
      mean 339) and that spans at least ten median visits (median 118 s).
      Warm-up plus window ends at 2,880 s, inside the first trading hour,
      so the arrival rate is constant over the measured part of a run. The run then continues,
      arrivals included, only until the window's own arrivals have left.
      That adds the cohort's visits still under way at the window's end to
      the ones the rule counted and removes none, so the analysed cohort is
      never smaller than the count the window was chosen on.

    python -m experiments.run_abm_diagnostics
    python -m experiments.run_abm_diagnostics --reps 10 --workers 10
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

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from experiments import _live_store as LS  # noqa: E402
from experiments._common import (make_run_dir, provenance_snapshot,  # noqa: E402
                                 write_sidecar)

TRANSIENT = ('entering', 'moving', 'shopping', 'checking_out', 'exiting')
ABSORB = ('purchased', 'abandoned')

MIN_CELL = 8            # minimum counts for a reliable second-order cell
N_PERM = 200            # permutations in the first-order null
PERM_SEED = 0
BAND_SWEEP_M = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)

# After the collection window the run carries on, arrivals included, until
# the last agent that arrived inside the window has left (see
# ``collect_sequences``). The chunk is how often that is checked and the
# cap how long it may take before the run is treated as stuck.
DRAIN_CHUNK_S = 60.0
MAX_DRAIN_S = 3600.0

# Default protocol; the module docstring records how each value was chosen.
NOMINAL_SPAWN = 0.27
NOMINAL_CAP = 45
WARMUP_S = 1020.0
COLLECT_S = 1860.0


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


def _entropy_mm(row):
    """Entropy in bits of a count vector with the Miller-Madow correction
    (K_obs - 1) / (2 n ln 2). The plug-in estimate is biased low by about
    that much, far more in a small second-order cell than in the large
    first-order cell it is compared with, which on its own makes a
    first-order chain look as if the predecessor carried information."""
    n = float(row.sum())
    if n <= 0:
        return 0.0
    k_obs = int(np.count_nonzero(row))
    h = _entropy({i: float(v) for i, v in enumerate(row)})
    return h + (k_obs - 1) / (2.0 * n * math.log(2))


def window_sequences(records, t_warm, t_end):
    """Jump sequences of the agents that spawned inside ``(t_warm, t_end]``,
    from completed state-history records. Agents that spawned during the
    warm-up entered an emptier store, so their paths are part of the
    fill-up transient. An agent stamped exactly ``t_warm`` arrived in the
    boundary tick itself, which the arrival counters snapshotted at the
    boundary already count as warm-up, so it is left out as well. The upper
    bound keeps out the agents that arrive while the window's own cohort is
    being drained, which would otherwise re-introduce the short-visit bias
    the drain removes."""
    out = []
    for rec in records:
        spawn_t = float(rec['spawn_t'])
        if spawn_t <= t_warm or spawn_t > t_end:
            continue
        seq = []
        for st in rec['states']:
            if not seq or seq[-1] != st:
                seq.append(st)
        out.append(seq)
    return out


def _arrival_counts(sim):
    """Arrivals admitted and arrivals that balked at the customer cap."""
    A = sim.analytics
    return int(A.get('total_customers', 0)), int(A.get('balked_arrivals', 0))


def collect_sequences(params, seconds, spawn, cap, warmup=WARMUP_S, dt=0.04,
                      seed=None, drain_chunk_s=DRAIN_CHUNK_S,
                      max_drain_s=MAX_DRAIN_S):
    """One replication, run fixed-step headless for ``warmup + seconds``
    simulated seconds. The store starts empty, so the first ``warmup``
    seconds, by which occupancy has settled on its plateau, are
    DELETED before any statistic is collected: the analysed cohort is the
    agents that spawn after the warm-up boundary and before the end of the
    window, and the heat map is measured as the increment over the window
    (the snapshot taken at the boundary is subtracted, and the map is read
    at the window's end). The run is a terminating simulation over a
    trading window; the deletion removes the empty-store fill-up transient
    from the within-window equilibrium statistics.

    Sequences come from the simulation's own state histories rather than
    from polling agent states: a poll misses any state shorter than its
    interval and records a jump between the states either side of it.

    Stopping at the end of the window would keep only the cohort members
    that had already left, which is a length-biased sample: among the
    agents arriving in the last few minutes, the short visits finish inside
    the window and the long ones do not. So after the window the run
    CARRIES ON, arrivals included at the rate in force when the window
    closed so the store stays as busy as it was, until every agent of the
    cohort has left; agents arriving during that drain are not analysed.
    Nothing measured over the window changes, because its counters and
    heat map are read before the drain begins.

    Also returns the post-warm-up arrival load (admitted and balked), so
    a summary shows whether the customer cap bound during collection, the
    simulated seconds the drain took, and which heat-map cells an agent
    can stand in (``walkable_heat_cells`` on floor 1's pathfinding grid),
    which the perimeter ratio's geometric null is computed from."""
    shop = LS.build_live_store(params)
    sim = shop.customer_simulation
    sim.max_customers = cap
    sim.spawn_rate = spawn
    sim.record_state_histories = True
    sim.completed_state_histories = []

    boundary = {}

    def _mark_warmup(s):
        # The first call lands on the first tick at or past the warm-up;
        # later periodic calls fall inside the collection window.
        if boundary:
            return
        boundary['t'] = float(s.sim_time)
        boundary['heat'] = np.array(s.heat_raw, dtype=float)
        boundary['arrivals'] = _arrival_counts(s)

    sim.run_headless(warmup + seconds, dt=dt, seed=seed,
                     callback=_mark_warmup, callback_every_s=warmup)

    t_warm = boundary['t']
    t_end = float(sim.sim_time)
    # Read over the window only: everything below happens after it closed.
    heat = np.array(sim.heat_raw, dtype=float) - boundary['heat']
    walkable = _floor1_walkable(sim, heat.shape)
    admitted, balked = _arrival_counts(sim)
    load = {'arrivals': (admitted + balked) - sum(boundary['arrivals']),
            'balked': balked - boundary['arrivals'][1]}

    def _still_in_store():
        return sum(1 for c in sim.customers
                   if t_warm < float(getattr(c, 'spawn_sim_time', 0.0))
                   <= t_end)

    # The drain keeps the arrival rate the window closed at. A drain that
    # ran past the top of the hour would otherwise pick up the next hour's
    # profile multiplier, and the cohort's last visits would finish in a
    # busier or quieter store than the one they arrived in. Within the
    # hour this leaves the arrival stream exactly as it was.
    hour = int(sim.sim_clock_start_hour + t_end / 3600.0) % 24
    sim.hourly_profile = np.full(
        24, float(np.asarray(sim.hourly_profile, dtype=float)[hour]))

    drained = 0.0
    while _still_in_store() and drained < max_drain_s:
        sim.run_headless(drain_chunk_s, dt=dt)
        drained += drain_chunk_s
    n_censored = _still_in_store()
    if n_censored:
        raise RuntimeError(
            f"{n_censored} agents of the collection window were still in "
            f"the store after draining for {drained:.0f} simulated s; "
            "the cohort's sequences would be length-biased")

    # step() counts and survives per-agent exceptions so a GUI session
    # keeps running; a measurement run must not have had any.
    errs = dict(getattr(sim, 'suppressed_errors', {}))
    if errs:
        raise RuntimeError(
            f"live loop suppressed exceptions during measurement: {errs}")

    sequences = window_sequences(sim.completed_state_histories, t_warm, t_end)
    return (sequences, n_censored, drained, heat, walkable,
            shop.width, shop.height, load)


def _order_stats(prev, cur, nxt, n_states):
    """Miller-Madow first-order conditional entropy, information gain from
    the predecessor, and n-weighted mean TV between each (prev, cur)
    conditional and its P(next|cur) -- all over the same transitions."""
    S = n_states
    N = int(cur.size)
    if N == 0:
        return 0.0, 0.0, 0.0
    c1 = np.bincount(cur * S + nxt, minlength=S * S).reshape(S, S)
    c2 = np.bincount((prev * S + cur) * S + nxt,
                     minlength=S * S * S).reshape(S * S, S)
    h1 = sum(float(c1[c].sum()) * _entropy_mm(c1[c]) for c in range(S)) / N
    h2 = 0.0
    tv = 0.0
    for cell in range(S * S):
        n = float(c2[cell].sum())
        if n <= 0:
            continue
        c = cell % S
        h2 += n * _entropy_mm(c2[cell])
        p1 = c1[c] / float(c1[c].sum())
        tv += n * 0.5 * float(np.abs(c2[cell] / n - p1).sum())
    return h1, h1 - h2 / N, tv / N


def markov_order_analysis(sequences, n_perm=N_PERM, seed=PERM_SEED):
    """Return first-order-adequacy diagnostics from jump sequences."""
    states = list(TRANSIENT + ABSORB)
    states += sorted({s for seq in sequences for s in seq} - set(states))
    index = {s: i for i, s in enumerate(states)}
    S = len(states)

    n_trans = sum(max(len(seq) - 1, 0) for seq in sequences)
    # Only transitions with a predecessor can enter a second-order cell, so
    # the first-order baseline is estimated from exactly those; a sequence's
    # first jump would otherwise sit in one conditional and not the other.
    triples = [(index[seq[i - 1]], index[seq[i]], index[seq[i + 1]])
               for seq in sequences for i in range(1, len(seq) - 1)]
    arr = np.array(triples, dtype=np.int64).reshape(-1, 3)
    prev, cur, nxt = arr[:, 0], arr[:, 1], arr[:, 2]
    cell_n = np.bincount(prev * S + cur, minlength=S * S)
    dense = cell_n[prev * S + cur] >= MIN_CELL
    prev, cur, nxt = prev[dense], cur[dense], nxt[dense]

    h1, gain, tv = _order_stats(prev, cur, nxt, S)

    # Null: shuffling predecessor labels within each current state keeps
    # every (prev, cur) and (cur, next) count but breaks any dependence of
    # next on prev given cur, i.e. it realises a first-order chain on the
    # same data. The null mean is the estimator's residual bias.
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(cur == c) for c in range(S)]
    null_gain = np.empty(n_perm)
    null_tv = np.empty(n_perm)
    for b in range(n_perm):
        perm_prev = prev.copy()
        for idx in groups:
            if idx.size > 1:
                perm_prev[idx] = prev[idx][rng.permutation(idx.size)]
        _, null_gain[b], null_tv[b] = _order_stats(perm_prev, cur, nxt, S)
    tol = 1e-12
    gain_p = (1 + int(np.sum(null_gain >= gain - tol))) / (n_perm + 1)
    tv_p = (1 + int(np.sum(null_tv >= tv - tol))) / (n_perm + 1)
    gain_null = float(null_gain.mean()) if n_perm else 0.0
    tv_null = float(null_tv.mean()) if n_perm else 0.0

    return {
        'n_sequences': len(sequences),
        'n_transitions': int(n_trans),
        'mean_seq_len': float(np.mean([len(s) for s in sequences])) if sequences else 0.0,
        'dense_second_order_mass': int(cur.size),
        'H_next_given_cur_bits': round(h1, 4),
        'info_gain_second_order_bits': round(gain, 4),
        'info_gain_null_mean_bits': round(gain_null, 4),
        'info_gain_excess_bits': round(gain - gain_null, 4),
        'info_gain_perm_p': round(gain_p, 4),
        'mean_tv_first_vs_second': round(tv, 4),
        'tv_null_mean': round(tv_null, 4),
        'tv_excess': round(tv - tv_null, 4),
        'tv_perm_p': round(tv_p, 4),
        'n_permutations': int(n_perm),
    }


def _band_means(heat, W, H, band_m):
    """Mean of ``heat`` over the cells within ``band_m`` of a wall and over
    the rest, every cell counted whether or not anyone can stand in it."""
    wcells, hcells = heat.shape
    xs = np.linspace(0, W, wcells)
    ys = np.linspace(0, H, hcells)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    perim = (X < band_m) | (X > W - band_m) | (Y < band_m) | (Y > H - band_m)
    interior = ~perim
    mp = float(heat[perim].mean()) if perim.any() else 0.0
    mi = float(heat[interior].mean()) if interior.any() else 0.0
    return mp, mi


def _ratio(mp, mi):
    return mp / mi if mi > 1e-9 else None


def _rounded(x, nd):
    return None if x is None else round(float(x), nd)


def perimeter_ratio(heat, W, H, band_m=2.5):
    """Emergent perimeter-to-interior mean foot-traffic ratio (R6.4)."""
    mp, mi = _band_means(heat, W, H, band_m)
    return {
        'perimeter_mean_intensity': round(mp, 4),
        'interior_mean_intensity': round(mi, 4),
        'perimeter_interior_ratio': _rounded(_ratio(mp, mi), 3),
        'band_m': band_m,
    }


def walkable_heat_cells(blocked, grid_res, heat_shape, heat_res):
    """Heat-map cells an agent can stand in, from a floor's blocked grid.

    ``blocked[gx, gy]`` is ``sim_geometry``'s pathfinding grid: True when an
    agent standing at ``(gx * grid_res, gy * grid_res)`` metres would collide
    with a wall or fixture (every obstacle inflated by the agent radius).
    Heat-map cell ``(i, j)`` collects the agents with
    ``int(x * heat_res) == i`` and ``int(y * heat_res) == j``, so it spans
    ``1 / heat_res`` metres from ``i / heat_res``; each cell takes the state
    of the grid point nearest its centre. The heat map is the finer grid
    (20 cells per metre against a 0.25 m pathfinding step), so this reads
    the floor plan at the resolution the planner itself sees it."""
    blocked = np.asarray(blocked, dtype=bool)
    nx, ny = heat_shape

    def _nearest(n_cells, n_points):
        centre = (np.arange(n_cells) + 0.5) / float(heat_res)
        idx = np.floor(centre / float(grid_res) + 0.5).astype(np.int64)
        return np.clip(idx, 0, n_points - 1)

    gx = _nearest(nx, blocked.shape[0])
    gy = _nearest(ny, blocked.shape[1])
    return ~blocked[np.ix_(gx, gy)]


def perimeter_ratio_geometric_null(walkable, W, H, band_m=2.5):
    """The ratio ``perimeter_ratio`` returns for a heat map uniform over the
    ``walkable`` cells, unrounded, or None when the interior holds none.

    ``perimeter_ratio`` averages each band over all of its cells, and cells
    under a fixture carry no traffic, so a floor whose interior is mostly
    shelving scores above 1 even if shoppers spread evenly over every spot
    they can stand on. This is that geometric baseline: the walkable share
    of the perimeter band over the walkable share of the interior. With
    nothing blocked it is exactly 1."""
    mp, mi = _band_means(np.asarray(walkable, dtype=float), W, H, band_m)
    return _ratio(mp, mi)


def _floor1_walkable(sim, heat_shape):
    """``walkable_heat_cells`` for floor 1 from the simulation's geometry
    caches, the grid the agents planned on; built first if no tick has."""
    grids = getattr(sim, 'path_blocked_grid_by_floor', None) or {}
    if 1 not in grids:
        sim._rebuild_geometry_caches()
        grids = sim.path_blocked_grid_by_floor
    return walkable_heat_cells(grids[1], sim.path_grid_resolution, heat_shape,
                               sim.heat_map_resolution)


def emergence_stats(heat, walkable, W, H, band_m=2.5):
    """The perimeter ratio of one window's heat map next to its geometric
    null and their quotient, for the headline band and every band of the
    sweep."""
    def _triple(b):
        obs = _ratio(*_band_means(heat, W, H, b))
        null = perimeter_ratio_geometric_null(walkable, W, H, b)
        to_null = obs / null if obs is not None and null else None
        return obs, null, to_null

    pr = perimeter_ratio(heat, W, H, band_m)
    _, null, to_null = _triple(band_m)
    pr['perimeter_interior_ratio_geometric_null'] = _rounded(null, 3)
    pr['ratio_to_geometric_null'] = _rounded(to_null, 3)
    # The perimeter ratio depends on how wide a band counts as
    # "perimeter", so archive the sweep alongside the headline value
    # rather than leaving that analysis choice unrecorded; the null moves
    # with the band too, so it is swept with it.
    sweep = {f"{b:.1f}": _triple(b) for b in BAND_SWEEP_M}
    pr['band_sweep'] = {k: _rounded(v[0], 3) for k, v in sweep.items()}
    pr['band_sweep_geometric_null'] = {k: _rounded(v[1], 3)
                                       for k, v in sweep.items()}
    pr['band_sweep_ratio_to_geometric_null'] = {k: _rounded(v[2], 3)
                                                for k, v in sweep.items()}
    return pr


def _replication(r, params, args):
    """Replication ``r`` under seed ``args.seed + r``, analysed where it ran.

    Top-level so a worker process can run it. Replication r + 1's agent
    seed equals replication r's arrival seed, but run_headless feeds them
    to different generators (the global Mersenne Twister and a PCG64
    arrival stream), so no stream is shared."""
    seed = args.seed + r
    seqs, n_cens, drained, heat, walkable, W, H, load = collect_sequences(
        params, args.seconds, args.spawn, args.cap, warmup=args.warmup,
        dt=args.dt, seed=seed, max_drain_s=args.max_drain)
    mk = markov_order_analysis(seqs)
    mk['n_censored_final'] = int(n_cens)
    mk['drain_seconds'] = float(drained)
    pr = emergence_stats(heat, walkable, W, H)
    return {'rep': r, 'seed': seed, 'load': load, 'markov_order': mk,
            'emergence': pr, 'sequences': seqs}


def _map_runs(fn, n_runs, workers, *fn_args):
    """``[fn(r, *fn_args) for r in range(n_runs)]``, in replication order.

    With ``workers`` > 1 the replications run in separate processes. Each
    builds its own store and seeds every RNG it draws from, and results are
    put back in replication order before anything is aggregated, so the
    summary is identical to the serial run."""
    if workers <= 1:
        return [fn(r, *fn_args) for r in range(n_runs)]
    done = {}
    pool = ProcessPoolExecutor(max_workers=min(workers, n_runs))
    try:
        futures = {pool.submit(fn, r, *fn_args): r for r in range(n_runs)}
        for fut in as_completed(futures):
            done[futures[fut]] = fut.result()
    except BaseException:
        # Leaving the pool's context manager would wait for every queued
        # replication first, so one that fails early -- an agent of its
        # cohort still in the store, or a suppressed exception -- would only
        # report once all the others had finished. Drop what has not started
        # and let the error out now.
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return [done[r] for r in sorted(done)]


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--seconds', type=float, default=COLLECT_S,
                   help="Collection window after the warm-up, simulated s")
    p.add_argument('--warmup', type=float, default=WARMUP_S,
                   help="Deleted warm-up, simulated s")
    p.add_argument('--reps', type=int, default=10)
    p.add_argument('--spawn', type=float, default=NOMINAL_SPAWN,
                   help="Arrivals per simulated s, before the hour-of-day "
                        "profile")
    p.add_argument('--cap', type=int, default=NOMINAL_CAP,
                   help="Most customers in the store at once")
    # 0.04 s is the threaded GUI loop's 25 fps ceiling, so agents move
    # with the per-tick resolution they have in the GUI.
    p.add_argument('--dt', type=float, default=0.04,
                   help="Fixed tick, simulated s")
    p.add_argument('--seed', type=int, default=1000,
                   help="Base seed; replication r runs under seed + r")
    p.add_argument('--max-drain', type=float, default=MAX_DRAIN_S,
                   help="Longest the run may continue past the window, "
                        "simulated s, while the window's cohort leaves")
    p.add_argument('--workers', type=int, default=1,
                   help="Processes running replications in parallel")
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
    if args.max_drain <= 0:
        p.error("--max-drain must be positive")
    return args


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


def main(argv=None):
    args = parse_args(argv)
    out_dir = make_run_dir(args.out_root, 'abm_diagnostics')
    # The git state at the start of the run, so the sidecar can say which
    # committed code produced the summary and whether it changed mid-run.
    prov = provenance_snapshot()
    print('[abm] calibrating the live store (current period) once...',
          flush=True)
    params = LS.calibrate_live_store(args.retail_path, 'current')
    data = LS.period_record(args.retail_path, 'current')

    print(f"[abm] {args.reps} reps of warm-up {args.warmup:.0f}s + collect "
          f"{args.seconds:.0f}s simulated (dt {args.dt}s, seeds "
          f"{args.seed}..{args.seed + args.reps - 1}) on {args.workers} "
          f"worker(s)...", flush=True)
    t0 = time.perf_counter()
    reps = _map_runs(_replication, args.reps, args.workers, params, args)
    wall = time.perf_counter() - t0

    # Pooled in replication order, so the permutation test below sees the
    # same transition list for any worker count.
    all_seqs = []
    for rep in reps:
        all_seqs.extend(rep.pop('sequences'))
        mk, pr = rep['markov_order'], rep['emergence']
        print(f"  rep {rep['rep'] + 1}/{args.reps} (seed {rep['seed']}): "
              f"info_gain={mk['info_gain_second_order_bits']:.3f} "
              f"(null {mk['info_gain_null_mean_bits']:.3f}, "
              f"p={mk['info_gain_perm_p']:.3f})  "
              f"TV={mk['mean_tv_first_vs_second']:.3f} "
              f"(null {mk['tv_null_mean']:.3f})  "
              f"perim_ratio={pr['perimeter_interior_ratio']} "
              f"(geometric null "
              f"{pr['perimeter_interior_ratio_geometric_null']}, "
              f"ratio to null {pr['ratio_to_geometric_null']})  "
              f"arrivals={rep['load']['arrivals']} "
              f"balked={rep['load']['balked']}", flush=True)

    def _rep_values(key):
        return [r['markov_order'][key] for r in reps]

    ig_m, ig_hw = _mean_ci(_rep_values('info_gain_second_order_bits'))
    tv_m, tv_hw = _mean_ci(_rep_values('mean_tv_first_vs_second'))
    h1_m, _ = _mean_ci(_rep_values('H_next_given_cur_bits'))
    ign_m, _ = _mean_ci(_rep_values('info_gain_null_mean_bits'))
    igx_m, igx_hw = _mean_ci(_rep_values('info_gain_excess_bits'))
    tvn_m, _ = _mean_ci(_rep_values('tv_null_mean'))
    tvx_m, tvx_hw = _mean_ci(_rep_values('tv_excess'))
    pr_m, pr_hw = _mean_ci([r['emergence']['perimeter_interior_ratio']
                            for r in reps if
                            r['emergence']['perimeter_interior_ratio']])

    def _emergence_values(key):
        return [r['emergence'][key] for r in reps
                if r['emergence'][key] is not None]

    nl_m, nl_hw = _mean_ci(_emergence_values(
        'perimeter_interior_ratio_geometric_null'))
    rn_m, rn_hw = _mean_ci(_emergence_values('ratio_to_geometric_null'))
    n_seq = int(sum(r['markov_order']['n_sequences'] for r in reps))
    n_tr = int(sum(r['markov_order']['n_transitions'] for r in reps))
    n_cens = int(sum(r['markov_order']['n_censored_final'] for r in reps))
    drain_m, _ = _mean_ci([r['markov_order']['drain_seconds'] for r in reps])
    n_arr = int(sum(r['load']['arrivals'] for r in reps))
    n_balk = int(sum(r['load']['balked'] for r in reps))

    # One permutation test over the transitions of all replications gives
    # the single p-value; replications share the calibrated store, so the
    # predecessor labels are exchangeable across them under the null.
    pooled = markov_order_analysis(all_seqs)

    # Aggregate keys keep the single-run names (means), so downstream
    # macro generation stays stable; *_ci95 carries the t half-width.
    summary = {
        'markov_order': {
            'n_sequences': n_seq,
            'n_transitions': n_tr,
            # Every agent of every replication's cohort left before its run
            # stopped, so no visit is missing for being long.
            'n_censored_final': n_cens,
            'mean_drain_seconds': round(float(drain_m), 1),
            'H_next_given_cur_bits': round(h1_m, 4),
            'info_gain_second_order_bits': round(ig_m, 4),
            'info_gain_ci95': round(ig_hw, 4),
            'info_gain_null_mean_bits': round(ign_m, 4),
            'info_gain_excess_bits': round(igx_m, 4),
            'info_gain_excess_ci95': round(igx_hw, 4),
            'info_gain_perm_p': pooled['info_gain_perm_p'],
            'mean_tv_first_vs_second': round(tv_m, 4),
            'tv_ci95': round(tv_hw, 4),
            'tv_null_mean': round(tvn_m, 4),
            'tv_excess': round(tvx_m, 4),
            'tv_excess_ci95': round(tvx_hw, 4),
            'tv_perm_p': pooled['tv_perm_p'],
            'n_permutations': N_PERM,
            'perm_seed': PERM_SEED,
            'min_cell': MIN_CELL,
            'entropy_estimator': 'miller_madow',
            'sequence_source': 'completed_state_histories',
            'pooled': pooled,
        },
        'emergence': {
            'perimeter_interior_ratio': round(pr_m, 3),
            'perimeter_ratio_ci95': round(pr_hw, 3),
            # What the floor plan alone gives: the same ratio for traffic
            # spread evenly over the walkable cells. The quotient is the
            # part the agents' routes add, averaged over replications.
            'perimeter_interior_ratio_geometric_null': round(nl_m, 3),
            'geometric_null_ci95': round(nl_hw, 3),
            'ratio_to_geometric_null': round(rn_m, 3),
            'ratio_to_geometric_null_ci95': round(rn_hw, 3),
            'band_m': reps[0]['emergence']['band_m'],
        },
        # Post-warm-up arrivals over all replications; balks are arrivals
        # refused at the customer cap.
        'load': {'arrivals': n_arr, 'balked': n_balk,
                 'balked_frac': round(n_balk / max(n_arr, 1), 4)},
        'protocol': {'mode': 'fixed_step', 'dt': args.dt,
                     'seed_base': args.seed,
                     'seeds': [r['seed'] for r in reps],
                     'reps': args.reps, 'warmup_s': args.warmup,
                     'collect_s': args.seconds, 'spawn': args.spawn,
                     'cap': args.cap, 'workers': args.workers,
                     'framing': 'terminating'},
        # The data the store was calibrated from.
        'data': data,
        'per_rep': reps,
    }
    # The sidecar goes first: summary.json is the file that marks a run as
    # finished, so it is written last.
    write_sidecar(out_dir, {'experiment': 'abm_diagnostics',
                            'args': vars(args), 'wall_seconds': wall},
                  provenance=prov)
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n== R6.2 first-order Markov adequacy "
          f"({args.reps} reps, warm-up deleted) ==")
    print(f"  sequences={n_seq}  transitions={n_tr}  "
          f"agents still in store at the end={n_cens} "
          f"(mean drain {drain_m:.0f}s past the window)")
    print(f"  H(next|cur)={h1_m:.3f} bits")
    print(f"  info gain from 2nd order = {ig_m:.3f} +/- {ig_hw:.3f} bits "
          f"(null {ign_m:.3f}, excess {igx_m:.3f} +/- {igx_hw:.3f}, "
          f"pooled permutation p={pooled['info_gain_perm_p']:.3f})")
    print(f"  mean TV(first,second) = {tv_m:.3f} +/- {tv_hw:.3f} "
          f"(null {tvn_m:.3f}, excess {tvx_m:.3f} +/- {tvx_hw:.3f}, "
          f"pooled permutation p={pooled['tv_perm_p']:.3f})")
    print("\n== R6.4 emergent perimeter dominance ==")
    print(f"  perimeter/interior ratio = {pr_m:.2f} +/- {pr_hw:.2f}  "
          f"(geometric null {nl_m:.2f} +/- {nl_hw:.2f}, "
          f"observed/null {rn_m:.2f} +/- {rn_hw:.2f})")
    print(f"\npost-warm-up arrivals={n_arr}, balked at the cap={n_balk}")
    print(f"wall {wall:.0f}s; artifacts in: {out_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
