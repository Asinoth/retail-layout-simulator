"""
Calibration and statistically consistent downstream models for the retail ABM.

Used by the live simulation (transition logging, sim-time rates) and by
Markov / Monte Carlo projection layers (empirical estimation, Laplace smoothing).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

# Transient states observed during the ABM; terminal outcomes are absorbing.
MARKOV_TRANSIENT = ('entering', 'moving', 'shopping', 'checking_out', 'exiting')
MARKOV_ABSORBING = ('purchased', 'abandoned')
MARKOV_STATES = MARKOV_TRANSIENT + MARKOV_ABSORBING

MIN_TRANSITIONS_FOR_EMPIRICAL = 25


def new_markov_transition_counts():
    """Sparse (from, to) -> count for empirical P-hat."""
    return defaultdict(int)


def record_transition(counts, from_state, to_state):
    if from_state is None or to_state is None:
        return
    if from_state not in MARKOV_STATES or to_state not in MARKOV_STATES:
        return
    if from_state in MARKOV_ABSORBING:
        return
    if from_state == to_state:
        return
    counts[(from_state, to_state)] += 1


def laplace_row_probs(counts, states, alpha=1.0):
    """Row-stochastic matrix with additive smoothing alpha per cell."""
    n = len(states)
    si = {s: i for i, s in enumerate(states)}
    raw = np.full((n, n), alpha, dtype=np.float64)
    for (a, b), c in counts.items():
        if a in si and b in si:
            raw[si[a], si[b]] += float(c)
    T = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        rs = raw[i].sum()
        T[i] = raw[i] / rs if rs > 0 else 0.0
        if T[i].sum() <= 0:
            T[i, i] = 1.0
    return T


def build_empirical_transition_matrix(analytics):
    """
    Estimate P from logged ABM transitions. Absorbing rows are identity.
    Falls back to a weakly informative prior if data are sparse.
    """
    counts = analytics.get('markov_transition_counts') or {}
    total_trans = sum(counts.values())
    meta = {
        'total_transitions': int(total_trans),
        'empirical': total_trans >= MIN_TRANSITIONS_FOR_EMPIRICAL,
        'laplace_alpha': 1.0,
    }

    states = list(MARKOV_STATES)
    si = {s: i for i, s in enumerate(states)}

    if meta['empirical']:
        T = laplace_row_probs(counts, states, alpha=meta['laplace_alpha'])
    else:
        T = _prior_transition_matrix(analytics, states, si)

    for s in MARKOV_ABSORBING:
        i = si[s]
        T[i, :] = 0.0
        T[i, i] = 1.0

    for s in MARKOV_TRANSIENT:
        i = si[s]
        row = T[i].copy()
        for j, sj in enumerate(states):
            if sj in MARKOV_ABSORBING and sj != 'purchased' and sj != 'abandoned':
                pass
        rs = row.sum()
        if rs > 0:
            T[i] = row / rs

    return states, T, meta


def _prior_transition_matrix(analytics, states, si):
    """Data-informed prior when empirical transitions are insufficient."""
    counts = defaultdict(int)
    total = max(1, analytics.get('total_customers', 0))
    completed = analytics.get('completed_purchases', 0)
    abandoned = analytics.get('abandoned_carts', 0)
    browsed = analytics.get('total_items_browsed', 0) or max(total * 2, 1)
    avg_loops = max(browsed / total, 1.0)

    counts[('entering', 'moving')] = total
    p_shop = min(avg_loops / (avg_loops + 1.0), 0.85)
    p_chk = min((completed + abandoned) / total, 1.0 - p_shop - 0.02)
    p_exit = max(1.0 - p_shop - p_chk, 0.02)
    counts[('moving', 'shopping')] = int(total * p_shop)
    counts[('moving', 'checking_out')] = int(total * p_chk)
    counts[('moving', 'exiting')] = int(total * p_exit)

    p_cont = min(0.7, (avg_loops - 1.0) / max(avg_loops, 1.0))
    p_done = 1.0 - p_cont
    conv = completed / max(completed + abandoned, 1)
    counts[('shopping', 'moving')] = int(total * p_cont)
    counts[('shopping', 'checking_out')] = int(total * p_done * conv)
    counts[('shopping', 'exiting')] = int(total * p_done * (1.0 - conv) * 0.5)
    counts[('shopping', 'abandoned')] = int(total * abandoned / total)

    p_purch = completed / max(completed + abandoned, 1)
    counts[('checking_out', 'purchased')] = int(total * p_purch)
    counts[('checking_out', 'abandoned')] = int(total * (1.0 - p_purch) * 0.5)
    counts[('checking_out', 'moving')] = int(total * 0.1)
    counts[('exiting', 'purchased')] = int(completed * 0.3)
    counts[('exiting', 'abandoned')] = int(abandoned)

    return laplace_row_probs(counts, states, alpha=1.0)


def compute_absorbing_analysis(states, T):
    """
    Standard absorbing-chain analysis on transient vs absorbing partition.
    Only purchased / abandoned are absorbing; exiting is transient until exit.
    """
    si = {s: i for i, s in enumerate(states)}
    transient = [s for s in MARKOV_TRANSIENT if s in si]
    absorbing = [s for s in MARKOV_ABSORBING if s in si]
    t_idx = [si[s] for s in transient]
    a_idx = [si[s] for s in absorbing]

    Q = T[np.ix_(t_idx, t_idx)]
    R = T[np.ix_(t_idx, a_idx)]
    I = np.eye(Q.shape[0])
    try:
        N = np.linalg.inv(I - Q)
    except np.linalg.LinAlgError:
        N = np.linalg.pinv(I - Q)

    B = N @ R
    expected_steps = N.sum(axis=1)
    time_in_state = N[0, :]

    return {
        'transient_states': transient,
        'absorbing_states': absorbing,
        'fundamental_N': N,
        'absorption_B': B,
        'expected_steps_from_entering': float(expected_steps[0]),
        'time_in_each_state': dict(zip(transient, time_in_state)),
        'p_purchase_from_entering': float(B[0, 0]),
        'p_abandon_from_entering': float(B[0, 1]),
    }


def transient_occupancy_distribution(analytics):
    """Empirical fraction of time spent in each transient state (not eig(T))."""
    occ = analytics.get('markov_state_occupancy') or {}
    trans = {s: float(occ.get(s, 0)) for s in MARKOV_TRANSIENT}
    total = sum(trans.values())
    if total <= 0:
        n = len(MARKOV_TRANSIENT)
        return {s: 1.0 / n for s in MARKOV_TRANSIENT}
    return {s: trans[s] / total for s in MARKOV_TRANSIENT}


def sim_hours(sim):
    """Hours of simulated time. Falls back to wall-clock run_time if the
    simulation engine doesn't track sim_time. Floors at 0.01 hours (= 36
    seconds) so per-hour rates can't explode when the user runs Optimize
    against a near-zero-duration simulation."""
    sim_time = getattr(sim, 'sim_time', None)
    if sim_time is None or sim_time <= 0.0:
        sim_time = getattr(sim, 'run_time', 0.0) or 0.0
    return max(sim_time / 3600.0, 0.01)


def extract_simulation_parameters(sim):
    """Calibrated parameters for MC / reporting from analytics + sim clock."""
    A = sim.analytics
    total = max(1, A.get('total_customers', 0))
    completed = max(1, A.get('completed_purchases', 0))
    run_sim_hrs = sim_hours(sim)

    conversion_rate = A['completed_purchases'] / total
    abandonment_rate = A.get('abandoned_carts', 0) / total

    baskets = A.get('basket_sizes', [])
    avg_basket = float(np.mean(baskets)) if baskets else 3.0
    std_basket = float(np.std(baskets, ddof=1)) if len(baskets) > 1 else max(avg_basket * 0.3, 0.5)

    rev_obs = A.get('customer_revenues', [])
    if len(rev_obs) >= 2:
        rev_per_cust = float(np.mean(rev_obs))
        rev_std = float(np.std(rev_obs, ddof=1))
    else:
        rev_per_cust = A['total_revenue'] / completed if completed else 0.0
        rev_std = max(rev_per_cust * 0.35, 0.1)

    rev_std = max(rev_std, 0.05)

    impulse_rate = A.get('impulse_purchases', 0) / completed if completed else 0.0
    impulse_sales = A.get('impulse_item_sales', {})
    avg_impulse_val = 0.0
    if impulse_sales and A.get('impulse_purchases', 0) > 0:
        total_impulse_items = sum(impulse_sales.values())
        if total_impulse_items > 0:
            shop = sim.shop
            total_impulse_rev = sum(
                _price(shop, name) * cnt for name, cnt in impulse_sales.items()
            )
            avg_impulse_val = total_impulse_rev / total_impulse_items
    if avg_impulse_val <= 0.0:
        avg_impulse_val = max(rev_per_cust * 0.15, 0.01)

    imp_rev_obs = A.get('impulse_revenues', [])
    if len(imp_rev_obs) >= 2:
        imp_std = float(np.std(imp_rev_obs, ddof=1))
    else:
        imp_std = max(avg_impulse_val * 0.3, 0.01)

    customers_per_hour = total / run_sim_hrs

    rev_by_area = dict(A.get('revenue_by_area', {}))
    total_area_rev = sum(rev_by_area.values()) or 1.0
    area_shares = {k: v / total_area_rev for k, v in rev_by_area.items()}

    queue_times = A.get('queue_wait_times', [])
    avg_queue = float(np.mean(queue_times)) if queue_times else 5.0

    dwell = A.get('dwell_times_by_zone', {})
    avg_dwell = {}
    for zone, times in dwell.items():
        if times:
            avg_dwell[zone] = float(np.mean(times))

    return {
        'customers_per_hour': customers_per_hour,
        'conversion_rate': max(0.001, min(conversion_rate, 0.999)),
        'abandonment_rate': abandonment_rate,
        'avg_basket_size': avg_basket,
        'std_basket_size': std_basket,
        'basket_sizes_observed': list(baskets) if baskets else [],
        'rev_per_converting_customer': rev_per_cust,
        'rev_std': rev_std,
        'impulse_rate': min(max(impulse_rate, 0.0), 1.0),
        'avg_impulse_value': avg_impulse_val,
        'impulse_value_std': imp_std,
        'area_revenue_shares': area_shares,
        'avg_queue_time': avg_queue,
        'avg_dwell_by_zone': avg_dwell,
        'total_observed_customers': total,
        'total_observed_revenue': A['total_revenue'],
        'run_hours': run_sim_hrs,
        'run_wall_hours': max(sim.run_time / 3600.0, 1e-6),
        'sim_time_seconds': getattr(sim, 'sim_time', 0.0),
        'simulation_speed': getattr(sim, 'simulation_speed', 1.0),
    }


def _price(shop, item_name):
    p = shop.prices.get(item_name)
    if p is not None:
        return float(p)
    try:
        for fdata in shop.floors.values():
            p2 = fdata.get('prices', {}).get(item_name)
            if p2 is not None:
                return float(p2)
    except Exception:
        pass
    return 0.0


def mc_engine(
    cph,
    conv,
    rev_mean,
    rev_std,
    imp_rate,
    imp_val,
    avg_bsk,
    std_bsk,
    observed_baskets,
    n_days,
    n_iter,
    op_hours,
    wknd_mult,
    monthly_growth,
    impulse_value_std=None,
    day_noise_std=0.08,
    rng=None,
):
    """
    Monte Carlo daily revenue with:
    - Poisson arrivals (per simulated day)
    - Binomial conversion
    - Per-converting-customer revenue draws (preserves variance)
    - Optional lognormal day-level traffic multiplier (serial correlation)

    Note: within-day non-homogeneity (the empirical hour-of-day arrival
    shape) is applied to the LIVE simulator's spawn loop via
    ``simulation.hourly_profile``, but NOT to this MC daily aggregator.
    The empirical hourly profile is mean-preserving (normalized to mean=1
    over open hours), so daily expected arrivals are identical to the
    homogeneous case. Modelling within-day Jensen-effects on conversion
    or basket-size would require restructuring this engine to be hourly
    rather than daily and is out of scope for the current paper.
    """
    if rng is None:
        rng = np.random

    if impulse_value_std is None:
        impulse_value_std = max(imp_val * 0.3, 0.01)

    day_multipliers = np.ones(7)
    day_multipliers[5] = wknd_mult
    day_multipliers[6] = wknd_mult

    daily_rev = np.zeros((n_iter, n_days))
    daily_cust = np.zeros((n_iter, n_days))
    daily_converting = np.zeros((n_iter, n_days))
    daily_impulse_rev = np.zeros((n_iter, n_days))
    daily_baskets = np.zeros((n_iter, n_days))
    log_day_factor = np.zeros(n_iter)
    obs_baskets = (
        list(observed_baskets) if observed_baskets and len(observed_baskets) >= 5
        else None
    )
    # Pre-compute empirical mean/std of observed baskets ONCE. We use these
    # to draw the per-iteration basket total via a normal approximation
    # (sum of n i.i.d. ~ N(n*mu, sqrt(n)*sigma)) instead of materialising a
    # giant (n_iter, max_nc) sample matrix per day — which made the
    # sensitivity loop allocate gigabytes and hang the GUI for realistic
    # customers_per_hour values.
    if obs_baskets is not None:
        _obs_arr = np.asarray(obs_baskets, dtype=np.float64)
        obs_mean = float(_obs_arr.mean())
        obs_std  = float(_obs_arr.std(ddof=1)) if _obs_arr.size > 1 else max(obs_mean * 0.3, 0.5)
        obs_std  = max(obs_std, 0.05)
    else:
        obs_mean = obs_std = None

    # Hard cap to keep Poisson sampling numerically sane even when the
    # caller passes a pathological customers_per_hour (e.g. sim_time floor
    # was zero on first Optimize click before enough data accumulated).
    _LAM_CAP = 1.0e6

    # Sanitize the caller-supplied rate up-front so we never end up with
    # NaN/inf inside the loop.
    if not np.isfinite(cph) or cph < 0:
        cph = 0.0

    for d in range(n_days):
        dow = d % 7
        gf = 1.0 + monthly_growth * (d / 30.0)
        lam = max(cph * op_hours * day_multipliers[dow] * gf, 0.1)
        if not np.isfinite(lam):
            lam = _LAM_CAP
        lam = min(lam, _LAM_CAP)

        if day_noise_std > 0:
            log_day_factor = np.clip(
                log_day_factor
                + rng.normal(0.0, day_noise_std, n_iter)
                - 0.5 * day_noise_std ** 2,
                -0.5,
                0.5,
            )
            lam_vec = lam * np.exp(log_day_factor)
            lam_vec = np.clip(lam_vec, 0.0, _LAM_CAP)
        else:
            lam_vec = lam

        n_cust = rng.poisson(lam_vec, n_iter)
        daily_cust[:, d] = n_cust
        n_conv = rng.binomial(n_cust, min(max(conv, 0.001), 0.999))
        daily_converting[:, d] = n_conv

        if obs_baskets is not None:
            # Sum of n_conv i.i.d. basket draws from the empirical distribution
            # is approximately N(n_conv*mu_obs, sqrt(n_conv)*sigma_obs). This
            # is O(n_iter) per day instead of O(n_iter * max(n_conv)) and uses
            # tens of bytes instead of hundreds of MB.
            nc_f = n_conv.astype(np.float64)
            total_items = np.where(
                n_conv > 0,
                rng.normal(nc_f * obs_mean, np.sqrt(np.maximum(nc_f, 0)) * obs_std, n_iter),
                0.0,
            )
            total_items = np.maximum(total_items, 0.0)
        else:
            total_items = np.maximum(
                rng.normal(avg_bsk, max(std_bsk, 0.5), n_iter) * n_conv, 0.0
            )
        daily_baskets[:, d] = total_items

        # Sum of n i.i.d. N(mu, sigma) ~ N(n*mu, sqrt(n)*sigma)
        rs = max(rev_std, 0.01)
        base_rev = np.where(
            n_conv > 0,
            rng.normal(
                n_conv.astype(np.float64) * rev_mean,
                np.sqrt(np.maximum(n_conv, 0).astype(np.float64)) * rs,
                n_iter,
            ),
            0.0,
        )
        base_rev = np.maximum(base_rev, 0.0)

        n_imp = rng.binomial(np.maximum(n_conv, 0), min(max(imp_rate, 0.0), 1.0))
        is_ = max(impulse_value_std, 0.01)
        imp_rev = np.where(
            n_imp > 0,
            rng.normal(
                n_imp.astype(np.float64) * imp_val,
                np.sqrt(np.maximum(n_imp, 0).astype(np.float64)) * is_,
                n_iter,
            ),
            0.0,
        )
        imp_rev = np.maximum(imp_rev, 0.0)
        daily_impulse_rev[:, d] = imp_rev

        daily_rev[:, d] = base_rev + imp_rev

    totals = daily_rev.sum(axis=1)
    return {
        'mean': float(totals.mean()),
        'std': float(totals.std()),
        'p5': float(np.percentile(totals, 5)),
        'p95': float(np.percentile(totals, 95)),
        'daily_means': daily_rev.mean(axis=0),
        'daily_revenue': daily_rev,
        'daily_customers': daily_cust,
        'daily_converting': daily_converting,
        'daily_impulse_rev': daily_impulse_rev,
        'daily_baskets': daily_baskets,
    }
