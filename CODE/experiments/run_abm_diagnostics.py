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

        Two more baselines answer the objection that the ratio is
        produced by construction (review R13). The door, the checkout
        lanes and the washroom all sit in the perimeter band, and the heat
        map counts agents standing still -- shopping, queuing, being
        served -- as well as walking. So:

          * a ROUTING null that knows the endpoints: the metres the agents'
            own A* planner walks on WHOLE VISITS, one tour per visit. Each
            of ROUTING_NULL_TOURS seeded tours draws a stocked invoice
            uniformly (as ``Customer._draw_invoice`` does), visits its
            fixtures' access points in a uniformly random order (the
            agents pick each next item uniformly from those left), takes
            the washroom after half the list when the drawn customer type
            needs it, and ends at the nearest lane (the tie-break of an
            unloaded store) and the door -- or, for an abandoned cart,
            goes from its last stop straight to the door. So every visit
            walks its entrance, lane and exit legs once, whatever its
            list length. Observed / routing null is what the agents add
            beyond shortest paths between where they must go; it is
            computed once per run from the store, with its own Monte
            Carlo standard error (the ratio estimator over iid tours);
          * the ratio from MOVING agents only: a second heat map sampled
            on the same one-second grid, counting only agents that are
            walking ('moving' or 'exiting') and have moved at least
            MOVING_MIN_STEP_M since the previous sample -- no shopper at
            a shelf, no queue, no one being served -- against both nulls.
            The routing null counts metres walked and has no dwell, so
            moving-only / routing null is the like-for-like comparison and
            the headline one (``emergence.headline``).

  Bookkeeping null for the Markov memory. The second-order information
        gain can come from the state machine's bookkeeping rather than
        from how shoppers move: the outcome appended to every sequence
        ('purchased' after 'checking_out', 'abandoned' after a shopping
        visit), the 'moving' leg between paying and leaving, the washroom
        reusing the 'shopping' state. ``bookkeeping_sequences`` generates
        the jump sequences the agents' automaton produces with NO
        geometry -- every list item reached, no stuck agent -- from the
        same draws the agents make at spawn (list length from the store's
        stocked invoices, customer type and washroom need, cart
        abandonment), and the same estimator is run on as many of them as
        each replication recorded. The information gain the bookkeeping
        alone produces is reported beside the observed one, so the paper
        can attribute the memory to the state machine (``tests/
        test_abm_nulls.py`` pins the generator to the real automaton on
        an open floor).

  All three are plausibility checks of the agent model, not validation
  against shopper data.

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
rule; ``experiments/run_live_protocol_study.py`` is that study as a seeded
runner, which derives every value below and writes it as an artifact. The
analysis is steady-state replication-deletion: each replication deletes
the empty-store fill-up and averages over a window in which the arrival
rate is constant.

  --spawn 0.27 --cap 45  The highest arrival rate on a 0.01/s grid at which
      occupancy stays below the cap at least 99% of the time after the
      warm-up. The cap binds 0.45% of the time at 0.26/s, 0.97% at
      0.27/s and 1.28% at 0.28/s (16 replications each). The calibrated
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
      mean 168 s). The 1% point is 969 s, rounded up to 1,020 s. Visits
      do interact through the checkout lanes and the cap, but at this
      load a lane is busy 29% of the time and the cap binds under 1%,
      and the occupancy after 1,020 s shows no detectable drift: the
      per-replication slope over the measured part of a run is
      0.5 +- 1.6 customers per 1,000 s (p = 0.48), and the first 300 s
      after the warm-up average 33.6 customers against 33.4 over the
      rest of the window (paired p = 0.87). Sixteen replications cannot
      bound the change across the window within 1% (TOST p = 0.68), so
      the study also doubles the warm-up: every outcome it checks
      (conversion, spend, basket, visit length, waits, perimeter ratio)
      moves by less than its own noise. The same criterion gives
      899-994 s at every rate from 0.25 to 0.30/s. Welch's
      procedure and MSER-5, which set an earlier protocol's warm-up, did
      not give a stable answer in an earlier study on this store's 16
      replications: across neighbouring rates Welch's
      doubled settle time ranged 840-3,360 s and MSER-5's truncation
      435-1,695 s, with no trend in the rate -- the replication-mean
      curve's slow wander decided them, not the transient.
  --seconds 1860  The shortest whole-minute window in which every study
      replication completes at least 300 visits (agents that arrive after
      the warm-up and leave inside the window; the fewest was 303 and the
      mean 336) and that spans at least ten median visits (Kaplan-Meier
      median 123 s).
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

from customer_pathfinding import AGENT_RADIUS_M, PathfindingMixin  # noqa: E402
from experiments import _live_store as LS  # noqa: E402
from experiments._common import (make_run_dir, provenance_snapshot,  # noqa: E402
                                 write_sidecar)
from retail_literature import (  # noqa: E402
    AGENT_ABANDON_PROB_RANGE, AGENT_MAX_IMPULSE_ITEMS_RANGE, CUSTOMER_TYPES,
    CUSTOMER_TYPE_PROBS, IMPULSE_BASE_PROPENSITY, IMPULSE_BROWSER_BONUS,
    IMPULSE_CHECKOUT_RADIUS_M, IMPULSE_DECAY_PER_ITEM, LIST_LENGTH_BY_TYPE,
    WC_PROBABILITY_BY_TYPE)

TRANSIENT = ('entering', 'moving', 'shopping', 'checking_out', 'exiting')
ABSORB = ('purchased', 'abandoned')

MIN_CELL = 8            # minimum counts for a reliable second-order cell
N_PERM = 200            # permutations in the first-order null
PERM_SEED = 0
BAND_SWEEP_M = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)

# The moving-only heat map (review R13): sampled every HEAT_SAMPLE_S
# simulated seconds -- the simulation's own heat-map cadence, so both maps
# see the same instants -- counting an agent only in a walking state and
# only when it has moved at least MOVING_MIN_STEP_M since its previous
# sample. The slowest walker covers 0.8 m in a second, so the threshold
# drops agents that stand still in a walking state (held at a corner, or
# about to change state) and nobody who is walking. Analysis parameters of
# this diagnostic, not model constants.
HEAT_SAMPLE_S = 1.0
MOVING_STATES = ('moving', 'exiting')
MOVING_MIN_STEP_M = 0.05

# The routing null (``routing_null_heat``): how many whole-visit tours it
# walks and the seed of their draws. Analysis parameters of this
# diagnostic; the null's Monte Carlo standard error is reported with it.
ROUTING_NULL_TOURS = 20000
ROUTING_NULL_SEED = 20260926

# Seed of the bookkeeping-only sequences (``bookkeeping_sequences``);
# replication r's are drawn from [BOOKKEEPING_SEED, r].
BOOKKEEPING_SEED = 20260925
# Sequences drawn to map the bookkeeping generator's support, against which
# the observed sequences are checked, and the stream they come from
# (replications use streams 0, 1, ...).
BOOKKEEPING_SUPPORT_N = 20000
SUPPORT_STREAM = 1_000_000

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
    at the window's end). The analysis is steady-state replication-
    deletion: the deletion removes the empty-store fill-up transient, and
    the window, inside one trading hour, runs at a constant arrival rate.

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
    simulated seconds the drain took, which heat-map cells an agent can
    stand in (``walkable_heat_cells`` on floor 1's pathfinding grid),
    which the perimeter ratio's geometric null is computed from, and the
    moving-only heat map of the window (walking agents that moved since
    their previous sample; see HEAT_SAMPLE_S), with the load's sample
    counts saying what share of the window's agent samples it kept."""
    shop = LS.build_live_store(params)
    sim = shop.customer_simulation
    sim.max_customers = cap
    sim.spawn_rate = spawn
    sim.record_state_histories = True
    sim.completed_state_histories = []

    boundary = {}
    res = sim.heat_map_resolution
    moving_heat = np.zeros(np.asarray(sim.heat_raw).shape, dtype=float)
    last_pos = {}
    n_samples = {'all': 0, 'moving': 0}
    t_begin = float(sim.sim_time)

    def _sample(s):
        # Called every HEAT_SAMPLE_S, the simulation's heat-map cadence.
        # The first call at or past the warm-up (half a tick of slack for
        # round-off in the summed dt) marks the boundary, on the same tick a
        # call every `warmup` seconds would; every later call falls inside
        # the collection window and adds the walking agents to the
        # moving-only map.
        if not boundary:
            if float(s.sim_time) - t_begin < warmup - dt / 2:
                return
            boundary['t'] = float(s.sim_time)
            boundary['heat'] = np.array(s.heat_raw, dtype=float)
            boundary['arrivals'] = _arrival_counts(s)
            for c in s.customers:
                last_pos[c.id] = (c.position[0], c.position[1])
            return
        for c in s.customers:
            x, y = c.position[0], c.position[1]
            prev = last_pos.get(c.id)
            last_pos[c.id] = (x, y)
            if getattr(c, 'floor', 1) != 1:
                continue
            n_samples['all'] += 1
            if (c.state in MOVING_STATES and prev is not None
                    and math.hypot(x - prev[0], y - prev[1])
                    >= MOVING_MIN_STEP_M):
                xi, yi = int(x * res), int(y * res)
                if 0 <= xi < moving_heat.shape[0] and 0 <= yi < moving_heat.shape[1]:
                    moving_heat[xi, yi] += 1.0
                    n_samples['moving'] += 1

    sim.run_headless(warmup + seconds, dt=dt, seed=seed,
                     callback=_sample, callback_every_s=HEAT_SAMPLE_S)

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
    load['samples'] = int(n_samples['all'])
    load['moving_samples'] = int(n_samples['moving'])
    return (sequences, n_censored, drained, heat, walkable,
            shop.width, shop.height, load, moving_heat)


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


def _band_masks(shape, W, H, band_m):
    """The heat-map cells within ``band_m`` of a wall (perimeter) and the
    rest (interior), as boolean masks of ``shape``. The one definition of
    the bands: ``_band_means`` and the routing null both read it."""
    wcells, hcells = shape
    xs = np.linspace(0, W, wcells)
    ys = np.linspace(0, H, hcells)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    perim = (X < band_m) | (X > W - band_m) | (Y < band_m) | (Y > H - band_m)
    return perim, ~perim


def _band_means(heat, W, H, band_m):
    """Mean of ``heat`` over the cells within ``band_m`` of a wall and over
    the rest, every cell counted whether or not anyone can stand in it."""
    perim, interior = _band_masks(heat.shape, W, H, band_m)
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


def _quotient(a, b):
    return a / b if a is not None and b else None


def emergence_stats(heat, walkable, W, H, band_m=2.5, routing=None,
                    moving=None):
    """The perimeter ratio of one window's heat map next to its geometric
    null and their quotient, for the headline band and every band of the
    sweep.

    ``routing`` (``routing_null_heat``'s whole-visit map) adds the routing
    null and the observed ratio over it; ``moving`` (the window's
    moving-only heat map)
    adds a ``moving`` block with the same ratio from walking agents alone,
    over both nulls (review R13)."""
    def _triple(b, h=heat):
        obs = _ratio(*_band_means(h, W, H, b))
        null = perimeter_ratio_geometric_null(walkable, W, H, b)
        return obs, null, _quotient(obs, null)

    def _routing(b):
        if routing is None:
            return None
        return _ratio(*_band_means(routing, W, H, b))

    pr = perimeter_ratio(heat, W, H, band_m)
    obs, null, to_null = _triple(band_m)
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
    if routing is not None:
        rnull = _routing(band_m)
        pr['perimeter_interior_ratio_routing_null'] = _rounded(rnull, 3)
        pr['ratio_to_routing_null'] = _rounded(_quotient(obs, rnull), 3)
        pr['band_sweep_routing_null'] = {
            f"{b:.1f}": _rounded(_routing(b), 3) for b in BAND_SWEEP_M}
        pr['band_sweep_ratio_to_routing_null'] = {
            f"{b:.1f}": _rounded(_quotient(sweep[f"{b:.1f}"][0],
                                           _routing(b)), 3)
            for b in BAND_SWEEP_M}
    if moving is not None:
        m_obs, _, m_to_geo = _triple(band_m, moving)
        mv = {'perimeter_interior_ratio': _rounded(m_obs, 3),
              'ratio_to_geometric_null': _rounded(m_to_geo, 3)}
        m_sweep = {f"{b:.1f}": _triple(b, moving) for b in BAND_SWEEP_M}
        mv['band_sweep'] = {k: _rounded(v[0], 3) for k, v in m_sweep.items()}
        mv['band_sweep_ratio_to_geometric_null'] = {
            k: _rounded(v[2], 3) for k, v in m_sweep.items()}
        if routing is not None:
            mv['ratio_to_routing_null'] = _rounded(
                _quotient(m_obs, _routing(band_m)), 3)
            mv['band_sweep_ratio_to_routing_null'] = {
                f"{b:.1f}": _rounded(_quotient(m_sweep[f"{b:.1f}"][0],
                                               _routing(b)), 3)
                for b in BAND_SWEEP_M}
        pr['moving'] = mv
    return pr


# --- the routing null ---------------------------------------------------------

class _RoutePlanner(PathfindingMixin):
    """The agents' own A* planner without an agent: floor 1 of ``sim``'s
    grid, an agent's radius, and the target type that makes an exit path
    end on the door, as ``Customer._compute_path`` does."""

    def __init__(self, sim):
        self._simulation_ref = sim
        self.floor = 1
        self.size = AGENT_RADIUS_M
        self.shop_dimensions = (sim.shop.width, sim.shop.height)
        self.current_target_type = None

    def route(self, start, goal, target_type, walls):
        """The points an agent at ``start`` walks through to ``goal``:
        itself, then the planner's waypoints."""
        self.current_target_type = target_type
        pts = self._compute_path(list(start), list(goal), walls)
        return [tuple(map(float, start))] + [tuple(map(float, p))
                                              for p in pts]


def _raster_cells(points, res, shape):
    """The heat-map cells the polyline ``points`` crosses, with the metres
    walked in each: sampled every half cell, each sample carrying its share
    of the segment's length (a walker at a constant speed spends time in a
    cell in proportion to the length of path inside it). Returns
    ``(xi, yi, metres)`` arrays, samples outside ``shape`` dropped."""
    step = 0.5 / float(res)
    xs, ys, ws = [], [], []
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        length = math.hypot(x1 - x0, y1 - y0)
        if length <= 0:
            continue
        n = max(1, int(math.ceil(length / step)))
        t = (np.arange(n) + 0.5) / n
        xi = ((x0 + t * (x1 - x0)) * res).astype(np.int64)
        yi = ((y0 + t * (y1 - y0)) * res).astype(np.int64)
        ok = (xi >= 0) & (xi < shape[0]) & (yi >= 0) & (yi < shape[1])
        xs.append(xi[ok])
        ys.append(yi[ok])
        ws.append(np.full(int(ok.sum()), length / n))
    if not xs:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros(0)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)


def _rasterize(heat, points, res, weight):
    """Add ``weight`` per metre walked along the polyline ``points`` to the
    heat-map cells it crosses (``_raster_cells``)."""
    xi, yi, w = _raster_cells(points, res, heat.shape)
    np.add.at(heat, (xi, yi), weight * w)


def _path_length(points):
    return float(sum(math.hypot(b[0] - a[0], b[1] - a[1])
                     for a, b in zip(points[:-1], points[1:])))


def routing_context(shop):
    """What the routing null draws a visit from: the store's stocked
    invoices as ``seed_into`` wrote them (the lists the agents draw), the
    regular items for a store without them, and whether it has a
    washroom."""
    from customer import is_regular_item
    sim = shop.customer_simulation
    cal = sim.analytics.get('calibration') or {}
    items = shop.floors[1].get('items') or {}
    regular = sorted(str(k) for k, d in items.items() if is_regular_item(k, d))
    keys, ptr = cal.get('list_invoice_keys'), cal.get('list_invoice_ptr')
    entries = cal.get('list_invoice_items')
    has_invoices = bool(keys is not None and ptr is not None
                        and entries is not None and len(keys)
                        and len(ptr) > 1)
    return {'invoice_keys': [str(k) for k in keys] if has_invoices else [],
            'invoice_ptr': (np.asarray(ptr, dtype=np.int64) if has_invoices
                            else np.zeros(1, dtype=np.int64)),
            'invoice_items': (np.asarray(entries, dtype=np.int64)
                              if has_invoices
                              else np.zeros(0, dtype=np.int64)),
            'regular_items': regular,
            'has_wc': 'WC' in (shop.floors[1].get('walls') or {}),
            'list_source': ('stocked_invoices' if has_invoices
                            else 'type_law_uniform_items')}


def routing_tour_draw(ctx, rng):
    """One visit's endpoints, drawn as an agent draws them at spawn: its
    customer type (which sets the washroom need), its abandonment
    probability and cart decision, and its list -- a stored invoice chosen
    uniformly, cut to the store's regular items (``Customer._draw_invoice``),
    or, with no invoice stored or none of the drawn one's items on the
    floor, the type law's length and a uniform sample of regular items
    (``_generate_shopping_list``) -- then the order the agent visits the
    list in: uniformly random, since ``_choose_next_target`` picks each next
    item uniformly from those left. Returns ``(items in visiting order,
    needs_wc, abandon)``. The draws come from ``rng`` alone, in that
    order."""
    types = list(CUSTOMER_TYPES)
    ctype = types[int(rng.choice(len(types), p=list(CUSTOMER_TYPE_PROBS)))]
    needs_wc = bool(rng.random() < WC_PROBABILITY_BY_TYPE[ctype])
    abandon = bool(rng.random() < rng.uniform(*AGENT_ABANDON_PROB_RANGE))
    regular = set(ctx['regular_items'])
    lst = []
    ptr = ctx['invoice_ptr']
    if len(ptr) > 1:
        i = int(rng.integers(0, len(ptr) - 1))
        keys = ctx['invoice_keys']
        lst = [keys[j] for j in ctx['invoice_items'][ptr[i]:ptr[i + 1]]
               if keys[j] in regular]
    if not lst and regular:
        lo, hi = LIST_LENGTH_BY_TYPE[ctype]
        n = min(int(rng.integers(lo, hi + 1)), len(regular))
        pool = sorted(regular)
        lst = [pool[j] for j in rng.choice(len(pool), size=n, replace=False)]
    order = rng.permutation(len(lst))
    return [lst[j] for j in order], needs_wc, abandon


def routing_tour_stops(items_in_order, needs_wc, abandon, has_wc=True):
    """The stops of one visit in the order an agent with no geometry in the
    way makes them: ``('item', key)`` for each list item, ``('wc', None)``
    after the first shelf visit at which half the list (rounded down) is
    done -- the rule ``bookkeeping_sequence`` follows, pinned to the
    automaton -- then ``('checkout', None)`` unless the cart is abandoned,
    and ``('exit', None)``."""
    stops = []
    n = len(items_in_order)
    wc_due = bool(needs_wc and has_wc)
    for k, key in enumerate(items_in_order, start=1):
        stops.append(('item', key))
        if wc_due and k >= n // 2:
            stops.append(('wc', None))
            wc_due = False
    if not abandon:
        stops.append(('checkout', None))
        if wc_due:              # only reachable with an empty list
            stops.append(('wc', None))
    stops.append(('exit', None))
    return stops


def routing_null_heat(shop, n_tours=ROUTING_NULL_TOURS,
                      seed=ROUTING_NULL_SEED):
    """Heat map of shortest paths over whole visits (review R13).

    ``n_tours`` visits drawn by ``routing_tour_draw`` from a generator
    seeded with ``seed``, each walked by the agents' own A* planner from the
    spawn point through its stops (``routing_tour_stops``): the fixtures'
    access points in visiting order, the washroom when it is due, the
    checkout lane an agent picks when no lane is loaded (the nearest lane
    centre -- ``Customer._pick_checkout_lane`` breaks the tie of empty
    queues by distance) and the door; an abandoned cart goes from its last
    stop to the door. Each leg starts where the last one ended, and each is
    rasterised per metre walked (``_raster_cells``). One tour per visit
    means the entrance, lane and exit legs -- which lie in the perimeter
    band -- are walked once per visit, not once per listed item. The map
    is what the perimeter ratio would be if agents did nothing but walk
    shortest paths between the places a visit must go, with no dwell, no
    queue and no crowding.

    Legs are planned once and counted; the map is the sum of each distinct
    leg's cells times the times it was walked, so it is exactly the sum
    over tours. The ratio at every band of the sweep is also computed per
    tour, which gives its Monte Carlo standard error as a ratio estimator
    over iid tours: se = sqrt(sum_t (P_t - R I_t)^2 / (n (n - 1))) / mean I,
    with P_t, I_t tour t's contributions to the perimeter and interior band
    means. Returns the map and a record of how it was built."""
    if int(n_tours) < 2:
        raise ValueError("the routing null needs at least two tours")
    sim = shop.customer_simulation
    sim._prepare_run()
    sim._rebuild_geometry_caches()
    sim.geometry_dirty = False
    floor = shop.floors[1]
    items, walls = floor.get('items') or {}, floor.get('walls') or {}
    ctx = routing_context(shop)
    lanes = [(w['position'][0] + w['size'][0] / 2.0,
              w['position'][1] + w['size'][1] / 2.0)
             for nm, w in sorted(walls.items())
             if nm == 'Checkout' or nm.startswith('Checkout_Lane')]
    wc = walls.get('WC')
    wc_centre = ((wc['position'][0] + wc['size'][0] / 2.0,
                  wc['position'][1] + wc['size'][1] / 2.0)
                 if wc is not None else None)
    entrance = tuple(map(float, sim._find_entrance_position()))
    door = tuple(map(float, sim.door_position))
    res = sim.heat_map_resolution
    shape = np.asarray(sim.heat_raw).shape
    W, H = float(shop.width), float(shop.height)
    bands = tuple(BAND_SWEEP_M)
    nb = len(bands)
    masks = [_band_masks(shape, W, H, b) for b in bands]
    sizes = [(float(p.sum()), float(q.sum())) for p, q in masks]
    planner = _RoutePlanner(sim)
    near = 2.0 * sim.path_grid_resolution
    access = {}
    legs, cells, band_rows, lengths, ends, counts = {}, [], [], [], [], []
    unroutable = set()

    def _goal(kind, key, here):
        if kind == 'item':
            if key not in access:
                access[key] = tuple(map(float, sim.item_access_point(
                    key, items[key], 1)))
            return access[key], 'item'
        if kind == 'wc':
            return wc_centre, 'wc'
        if kind == 'checkout' and lanes:
            lane = min(lanes, key=lambda c: math.hypot(c[0] - here[0],
                                                       c[1] - here[1]))
            return lane, 'checkout'
        return door, 'exit'

    def _leg(start, goal, ttype):
        """Index of the leg start -> goal, planned on first use. The planner
        returns no waypoint when start and goal share a grid cell (nothing
        to walk) or when no path exists; the latter is recorded."""
        key = (start, goal, ttype)
        j = legs.get(key)
        if j is None:
            pts = planner.route(start, goal, ttype, walls)
            if len(pts) > 1:
                end = pts[-1]
            else:
                end = start
                if math.hypot(goal[0] - start[0], goal[1] - start[1]) > near:
                    unroutable.add(key)
            xi, yi, w = _raster_cells(pts, res, shape)
            row = np.empty(2 * nb)
            for b, ((perim, _), (n_p, n_i)) in enumerate(zip(masks, sizes)):
                on = perim[xi, yi]
                row[b] = float(w[on].sum()) / n_p if n_p else 0.0
                row[nb + b] = float(w[~on].sum()) / n_i if n_i else 0.0
            j = len(cells)
            legs[key] = j
            cells.append((xi, yi, w))
            band_rows.append(row)
            lengths.append(_path_length(pts))
            ends.append(end)
            counts.append(0)
        counts[j] += 1
        return j

    rng = np.random.default_rng(seed)
    per_tour = np.zeros((int(n_tours), 2 * nb))
    tour_m = np.zeros(int(n_tours))
    n_wc = n_abandoned = n_items = n_lane = n_exit = 0
    for t in range(int(n_tours)):
        lst, needs_wc, abandon = routing_tour_draw(ctx, rng)
        here = entrance
        for kind, key in routing_tour_stops(lst, needs_wc, abandon,
                                            ctx['has_wc']):
            goal, ttype = _goal(kind, key, here)
            j = _leg(here, goal, ttype)
            per_tour[t] += band_rows[j]
            tour_m[t] += lengths[j]
            here = ends[j]
            n_wc += kind == 'wc'
            n_items += kind == 'item'
            n_lane += kind == 'checkout'
            n_exit += kind == 'exit'
        n_abandoned += abandon

    heat = np.zeros(shape, dtype=float)
    for (xi, yi, w), c in zip(cells, counts):
        np.add.at(heat, (xi, yi), c * w)

    ratio, se = {}, {}
    for b, band in enumerate(bands):
        P, I = per_tour[:, b], per_tour[:, nb + b]
        k = f"{band:.1f}"
        if I.sum() <= 0:
            ratio[k] = se[k] = None
            continue
        r = float(P.sum() / I.sum())
        resid = P - r * I
        n = P.size
        ratio[k] = r
        se[k] = float(math.sqrt(float((resid ** 2).sum()) / (n * (n - 1)))
                      / float(I.mean()))
    p_wc = float(sum(p * WC_PROBABILITY_BY_TYPE[c]
                     for c, p in zip(CUSTOMER_TYPES, CUSTOMER_TYPE_PROBS)))
    return heat, {
        'construction': 'whole_visit_tours',
        'n_tours': int(n_tours),
        'seed': int(seed),
        'list_source': ctx['list_source'],
        'invoice_draw': ('uniform over the stored invoices, cut to the '
                         'regular items (Customer._draw_invoice)'),
        'visiting_order': ('uniformly random permutation '
                           '(Customer._choose_next_target)'),
        'legs': ('spawn -> list items in visiting order (-> washroom after '
                 'half the list when needed) -> nearest lane -> door; an '
                 'abandoned cart goes from its last stop to the door'),
        'lane_rule': 'nearest lane centre (every queue empty)',
        'wc_leg': {'included': bool(ctx['has_wc']),
                   'rule': ('after the first shelf visit at which half the '
                            'list (rounded down) is done, as the agent '
                            'automaton does'),
                   'type_mixed_probability': round(p_wc, 4),
                   'n_tours_with_wc': int(n_wc)},
        'abandonment': {'included': True,
                        'probability_range': list(AGENT_ABANDON_PROB_RANGE),
                        'rule': ('a per-visit probability drawn from the '
                                 'range, then the cart decision; an '
                                 'abandoned visit walks no lane leg'),
                        'n_abandoned': int(n_abandoned)},
        'impulse_legs': ('omitted: paid agents walking to impulse displays '
                         'are not routed'),
        # One entrance, one exit and at most one lane leg per visit.
        'n_list_items_walked': int(n_items),
        'n_lane_legs': int(n_lane),
        'n_exit_legs': int(n_exit),
        'n_distinct_legs': len(cells),
        'walked_m': float(heat.sum()),
        'mean_tour_m': float(tour_m.mean()),
        'n_lanes': len(lanes),
        'unroutable_legs': len(unroutable),
        'ratio_by_band': {k: _rounded(v, 4) for k, v in ratio.items()},
        'ratio_mc_se_by_band': {k: _rounded(v, 4) for k, v in se.items()},
    }


# --- the bookkeeping null ------------------------------------------------------

def bookkeeping_sequence(n_items, needs_wc, abandon, n_impulse=0,
                         has_wc=True):
    """The jump sequence ``Customer`` produces for one visit with no
    geometry in the way: a list of ``n_items`` items, all reached, a
    washroom trip when ``needs_wc`` (and the store has one), the cart
    abandoned or paid for, and ``n_impulse`` impulse pick-ups after paying.

    It follows ``Customer.update``'s control flow state by state: the
    washroom check runs after each shelf visit once half the list is done
    (before the list-complete check, so a one-item list goes to the
    washroom first); after the washroom the agent moves on and, with
    nothing left, goes to the till; an abandoned cart turns for the door
    from wherever the list ends; a paid visit walks from the lane to each
    impulse display and back, then leaves. Repeated states collapse, as
    ``state_history`` records them; the outcome closes the sequence."""
    seq = ['entering']

    def go(state):
        if seq[-1] != state:
            seq.append(state)

    visited = 0
    visited_wc = not has_wc if needs_wc else False
    paid = False

    def wc_trip():
        nonlocal visited_wc
        go('moving')
        go('shopping')
        visited_wc = True
        # After the washroom dwell: move on to the next item, or to the till.
        go('moving')

    def finish():
        nonlocal paid
        if abandon:
            go('exiting')
            return
        go('moving')
        go('checking_out')
        paid = True
        if n_impulse > 0:
            for _ in range(int(n_impulse)):
                go('moving')
                go('shopping')
            go('moving')        # back to the lane, already paid
        elif needs_wc and not visited_wc:
            wc_trip()           # only reachable with an empty list
        go('exiting')

    if n_items <= 0:
        finish()
    else:
        go('moving')
        while True:
            go('shopping')
            visited += 1
            if needs_wc and not visited_wc and visited >= n_items // 2:
                wc_trip()
                if visited >= n_items:
                    finish()
                    break
                continue
            if visited >= n_items:
                finish()
                break
            go('moving')
    return seq + ['purchased' if paid else 'abandoned']


def bookkeeping_context(shop):
    """What the bookkeeping generator needs from a store: the list lengths
    its agents draw (the stocked invoices' sizes), whether it has a
    washroom, and how many impulse displays stand within the checkout
    catchment of its lanes."""
    sim = shop.customer_simulation
    cal = sim.analytics.get('calibration') or {}
    lengths = [int(x) for x in (cal.get('list_length_sample') or [])]
    floor = shop.floors[1]
    walls, items = floor.get('walls') or {}, floor.get('items') or {}
    lanes = [((w['position'][0] + w['size'][0] / 2.0),
              (w['position'][1] + w['size'][1] / 2.0))
             for nm, w in walls.items()
             if nm == 'Checkout' or nm.startswith('Checkout_Lane')]
    n_impulse = 0
    for d in items.values():
        if d.get('category') != 'Impulse':
            continue
        cx = d['position'][0] + d['size'][0] / 2.0
        cy = d['position'][1] + d['size'][1] / 2.0
        if any(math.hypot(cx - lx, cy - ly) <= IMPULSE_CHECKOUT_RADIUS_M
               for lx, ly in lanes):
            n_impulse += 1
    return {'list_lengths': lengths, 'has_wc': 'WC' in walls,
            'n_impulse_candidates': int(n_impulse),
            'list_length_source': ('stocked_invoices' if lengths
                                   else 'none')}


def bookkeeping_sequences(n, context, rng):
    """``n`` bookkeeping-only jump sequences (``bookkeeping_sequence``),
    each from the draws an agent makes at spawn: a list length from the
    store's stocked invoices (uniform over them, as ``_draw_invoice``
    picks one), a customer type and, from it, the washroom need, an
    abandonment probability and the cart decision, and the impulse
    pick-ups a paying agent makes at a lane with the store's impulse
    displays (the long-dwell bonus, which needs a clock, left out)."""
    lengths = np.asarray(context['list_lengths'], dtype=np.int64)
    if not lengths.size:
        raise ValueError("the store records no list lengths to draw from")
    types = list(CUSTOMER_TYPES)
    out = []
    for _ in range(int(n)):
        n_items = int(lengths[rng.integers(0, lengths.size)])
        ctype = types[int(rng.choice(len(types), p=list(CUSTOMER_TYPE_PROBS)))]
        needs_wc = bool(rng.random() < WC_PROBABILITY_BY_TYPE[ctype])
        abandon = bool(rng.random() < rng.uniform(*AGENT_ABANDON_PROB_RANGE))
        n_imp = 0
        cand = int(context.get('n_impulse_candidates', 0))
        if cand and not abandon:
            chance = IMPULSE_BASE_PROPENSITY[ctype] + (
                IMPULSE_BROWSER_BONUS if ctype == 'browser' else 0.0)
            cap = int(rng.integers(*AGENT_MAX_IMPULSE_ITEMS_RANGE))
            while n_imp < cap and n_imp < cand and rng.random() < chance:
                n_imp += 1
                chance *= IMPULSE_DECAY_PER_ITEM
        out.append(bookkeeping_sequence(n_items, needs_wc, abandon, n_imp,
                                        context.get('has_wc', True)))
    return out


def _replication(r, params, args, routing=None):
    """Replication ``r`` under seed ``args.seed + r``, analysed where it ran.

    Top-level so a worker process can run it. Replication r + 1's agent
    seed equals replication r's arrival seed, but run_headless feeds them
    to different generators (the global Mersenne Twister and a PCG64
    arrival stream), so no stream is shared. ``routing`` is the store's
    routing-null heat map, computed once by the parent."""
    seed = args.seed + r
    (seqs, n_cens, drained, heat, walkable, W, H, load,
     moving) = collect_sequences(
        params, args.seconds, args.spawn, args.cap, warmup=args.warmup,
        dt=args.dt, seed=seed, max_drain_s=args.max_drain)
    mk = markov_order_analysis(seqs)
    mk['n_censored_final'] = int(n_cens)
    mk['drain_seconds'] = float(drained)
    pr = emergence_stats(heat, walkable, W, H, routing=routing, moving=moving)
    pr['moving']['share_of_samples'] = (
        round(load['moving_samples'] / load['samples'], 4)
        if load['samples'] else None)
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
    # 0.04 s is the threaded GUI loop's fixed tick, so the headless runs
    # advance the same discrete-time model the GUI does.
    p.add_argument('--dt', type=float, default=0.04,
                   help="Fixed tick, simulated s")
    p.add_argument('--seed', type=int, default=1000,
                   help="Base seed; replication r runs under seed + r")
    p.add_argument('--max-drain', type=float, default=MAX_DRAIN_S,
                   help="Longest the run may continue past the window, "
                        "simulated s, while the window's cohort leaves")
    p.add_argument('--routing-tours', type=int, default=ROUTING_NULL_TOURS,
                   help="Whole-visit tours the routing null walks")
    p.add_argument('--routing-seed', type=int, default=ROUTING_NULL_SEED,
                   help="Seed of the routing null's visit draws")
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
    if args.routing_tours < 2:
        p.error("--routing-tours must be >= 2")
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

    # The store's own endpoints and draws, for the routing null and the
    # bookkeeping null; neither runs an agent, so neither draws from any
    # replication's stream.
    print(f'[abm] routing null: {args.routing_tours} whole-visit A* tours '
          f'(seed {args.routing_seed}) ...', flush=True)
    null_store = LS.build_live_store(params)
    routing, routing_info = routing_null_heat(
        null_store, n_tours=args.routing_tours, seed=args.routing_seed)
    context = bookkeeping_context(null_store)
    del null_store

    print(f"[abm] {args.reps} reps of warm-up {args.warmup:.0f}s + collect "
          f"{args.seconds:.0f}s simulated (dt {args.dt}s, seeds "
          f"{args.seed}..{args.seed + args.reps - 1}) on {args.workers} "
          f"worker(s)...", flush=True)
    t0 = time.perf_counter()
    reps = _map_runs(_replication, args.reps, args.workers, params, args,
                     routing)
    wall = time.perf_counter() - t0

    # The bookkeeping-only null: as many generated sequences as each
    # replication recorded, each replication from its own seed, analysed
    # by the same estimator.
    bk_reps = []
    for rep in reps:
        n = len(rep['sequences'])
        bk = bookkeeping_sequences(
            n, context, np.random.default_rng([BOOKKEEPING_SEED, rep['rep']]))
        bk_reps.append((bk, markov_order_analysis(bk)))

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

    # The same for the bookkeeping-only sequences, and the observed gain
    # over theirs per replication: what the agents' movement adds to the
    # memory the state machine's bookkeeping already produces.
    bk_all = [s for bk, _ in bk_reps for s in bk]
    bk_pooled = markov_order_analysis(bk_all)
    bk_gain_m, bk_gain_hw = _mean_ci([m['info_gain_second_order_bits']
                                      for _, m in bk_reps])
    bk_exc_m, bk_exc_hw = _mean_ci([m['info_gain_excess_bits']
                                    for _, m in bk_reps])
    bk_tv_m, bk_tv_hw = _mean_ci([m['mean_tv_first_vs_second']
                                  for _, m in bk_reps])
    diff_m, diff_hw = _mean_ci([r['markov_order']['info_gain_second_order_bits']
                                - m['info_gain_second_order_bits']
                                for r, (_, m) in zip(reps, bk_reps)])
    # Which jump sequences the bookkeeping can produce at all, and the
    # share of the observed ones among them: an observed sequence outside
    # it took a route only geometry makes (a stuck agent, a dropped item).
    support = {tuple(s) for s in bookkeeping_sequences(
        BOOKKEEPING_SUPPORT_N, context,
        np.random.default_rng([BOOKKEEPING_SEED, SUPPORT_STREAM]))}
    in_support = (sum(1 for s in all_seqs if tuple(s) in support)
                  / max(len(all_seqs), 1))
    for rep, (_, m) in zip(reps, bk_reps):
        rep['markov_order']['bookkeeping_null'] = {
            k: m[k] for k in ('n_sequences', 'info_gain_second_order_bits',
                              'info_gain_null_mean_bits',
                              'info_gain_excess_bits', 'info_gain_perm_p',
                              'mean_tv_first_vs_second')}

    # Routing null and moving-only ratios, replication means with t
    # half-widths like the ratios above.
    rr_m, rr_hw = _mean_ci(_emergence_values('ratio_to_routing_null'))
    rnull = reps[0]['emergence'].get('perimeter_interior_ratio_routing_null')

    def _moving_values(key):
        return [r['emergence']['moving'][key] for r in reps
                if r['emergence']['moving'].get(key) is not None]

    mv_m, mv_hw = _mean_ci(_moving_values('perimeter_interior_ratio'))
    mvg_m, mvg_hw = _mean_ci(_moving_values('ratio_to_geometric_null'))
    mvr_m, mvr_hw = _mean_ci(_moving_values('ratio_to_routing_null'))
    mvs_m, _ = _mean_ci(_moving_values('share_of_samples'))
    band_key = f"{reps[0]['emergence']['band_m']:.1f}"
    rnull_se = routing_info['ratio_mc_se_by_band'].get(band_key)

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
            # The memory the state machine's bookkeeping produces with no
            # geometry (review R13): the same estimator on as many
            # generated sequences per replication, the observed gain over
            # it, and the share of observed sequences the bookkeeping can
            # produce at all.
            'bookkeeping_null': {
                'n_sequences': int(len(bk_all)),
                'info_gain_second_order_bits': round(bk_gain_m, 4),
                'info_gain_ci95': round(bk_gain_hw, 4),
                'info_gain_excess_bits': round(bk_exc_m, 4),
                'info_gain_excess_ci95': round(bk_exc_hw, 4),
                'mean_tv_first_vs_second': round(bk_tv_m, 4),
                'tv_ci95': round(bk_tv_hw, 4),
                'info_gain_perm_p': bk_pooled['info_gain_perm_p'],
                'tv_perm_p': bk_pooled['tv_perm_p'],
                'observed_minus_bookkeeping_bits': round(diff_m, 4),
                'observed_minus_bookkeeping_ci95': round(diff_hw, 4),
                'share_observed_in_bookkeeping_support': round(in_support, 4),
                'support_sample': BOOKKEEPING_SUPPORT_N,
                'seed': BOOKKEEPING_SEED,
                'context': {k: v for k, v in context.items()
                            if k != 'list_lengths'},
                'n_list_lengths': len(context['list_lengths']),
                'pooled': bk_pooled,
            },
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
            # What shortest paths over whole visits give (review R13): one
            # map per store, so the null has no replication spread, only
            # the Monte Carlo error of its tours, recorded beside it (the
            # *_ci95 of the quotients is the replication spread alone).
            'perimeter_interior_ratio_routing_null': rnull,
            'routing_null_mc_se': rnull_se,
            'routing_null_mc_rse': (round(rnull_se / rnull, 4)
                                    if rnull and rnull_se is not None
                                    else None),
            'ratio_to_routing_null': round(rr_m, 3),
            'ratio_to_routing_null_ci95': round(rr_hw, 3),
            'routing_null': routing_info,
            # The like-for-like comparison: the routing null counts metres
            # walked and has no dwell, so it is set against the map of
            # walking agents. The all-agent ratios above include time
            # spent standing at shelves, in queues and at the tills.
            'headline': {
                'comparison': 'moving_only / routing_null',
                'ratio_to_routing_null': round(mvr_m, 3),
                'ratio_to_routing_null_ci95': round(mvr_hw, 3),
                'routing_null': rnull,
                'moving_only_ratio': round(mv_m, 3),
            },
            # The same ratio from walking agents only, against both nulls.
            'moving_only': {
                'perimeter_interior_ratio': round(mv_m, 3),
                'perimeter_ratio_ci95': round(mv_hw, 3),
                'ratio_to_geometric_null': round(mvg_m, 3),
                'ratio_to_geometric_null_ci95': round(mvg_hw, 3),
                'ratio_to_routing_null': round(mvr_m, 3),
                'ratio_to_routing_null_ci95': round(mvr_hw, 3),
                'share_of_samples': round(mvs_m, 4),
                'states': list(MOVING_STATES),
                'min_step_m': MOVING_MIN_STEP_M,
                'sample_interval_s': HEAT_SAMPLE_S,
            },
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
                     'framing': 'steady_state_replication_deletion'},
        # The data the store was calibrated from.
        'data': data,
        'per_rep': reps,
    }
    # The sidecar goes first: summary.json is the file that marks a run as
    # finished, so it is written last.
    write_sidecar(out_dir, {'experiment': 'abm_diagnostics',
                            'args': vars(args), 'wall_seconds': wall,
                            # The workbook's SHA-256 and the adapter
                            # version, with the rest of the period record.
                            'data': data},
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
    print(f"  bookkeeping-only sequences: info gain {bk_gain_m:.3f} +/- "
          f"{bk_gain_hw:.3f} bits (pooled p={bk_pooled['info_gain_perm_p']:.3f}); "
          f"observed minus bookkeeping {diff_m:.3f} +/- {diff_hw:.3f}; "
          f"{100 * in_support:.1f}% of observed sequences are ones the "
          f"bookkeeping produces")
    print("\n== R6.4 emergent perimeter dominance ==")
    print(f"  perimeter/interior ratio = {pr_m:.2f} +/- {pr_hw:.2f}  "
          f"(geometric null {nl_m:.2f} +/- {nl_hw:.2f}, "
          f"observed/null {rn_m:.2f} +/- {rn_hw:.2f})")
    print(f"  routing null {rnull} (MC se {rnull_se}; observed/routing null "
          f"{rr_m:.2f} +/- {rr_hw:.2f}); moving agents only: ratio "
          f"{mv_m:.2f} +/- "
          f"{mv_hw:.2f}, /geometric null {mvg_m:.2f} +/- {mvg_hw:.2f}, "
          f"/routing null {mvr_m:.2f} +/- {mvr_hw:.2f} "
          f"({100 * mvs_m:.0f}% of agent samples)")
    print(f"\npost-warm-up arrivals={n_arr}, balked at the cap={n_balk}")
    print(f"wall {wall:.0f}s; artifacts in: {out_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
