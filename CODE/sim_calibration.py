"""
Calibration and statistically consistent downstream models for the retail ABM.

Used by the live simulation (transition logging, sim-time rates) and by
Markov / Monte Carlo projection layers (empirical estimation, Laplace smoothing).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from retail_literature import (
    ABANDON_FRAC_OF_NONCONVERTERS, ASSUMED_CONVERSION_RATE,
    DEFAULT_DAY_NOISE_STD, DEFAULT_QUEUE_TIME_S, IMPULSE_RATE_STANDIN,
    IMPULSE_VALUE_CV, IMPULSE_VALUE_FRAC_OF_GROSS, MC_CONV_CLAMP_HI,
    MC_CONV_CLAMP_LO, MC_LAMBDA_CAP, MC_LAMBDA_FLOOR, MC_SD_FLOOR,
    MC_SPEND_LAW, NET_BASE_REVENUE_MIN_FRAC, REV_STD_FRAC_OF_MEAN,
    STANDIN_AVG_BASKET,
)

# Transient states observed during the ABM; terminal outcomes are absorbing.
MARKOV_TRANSIENT = ('entering', 'moving', 'shopping', 'checking_out', 'exiting')
MARKOV_ABSORBING = ('purchased', 'abandoned')
MARKOV_STATES = MARKOV_TRANSIENT + MARKOV_ABSORBING

MIN_TRANSITIONS_FOR_EMPIRICAL = 25

# Stand-in impulse share of purchases for a calibrated session. Transaction
# data does not label impulse buys, so there is nothing to estimate it from;
# it is retail_literature.IMPULSE_RATE_STANDIN, the value the headless
# calibrated path uses too, so both paths read the same number. Omnichannel
# measures a per-family impulse rate and overrides it.
DEFAULT_CALIBRATED_IMPULSE_RATE = IMPULSE_RATE_STANDIN


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
    # The estimate is only worth using once outcomes have been observed.
    # Counting every jump would declare the chain empirical from a handful
    # of moving/shopping loops, before anyone has bought or left -- and
    # then the absorption probabilities are pure Laplace smoothing (0.5
    # each), not data. Downstream consumers read P(purchase | entering) as
    # a conversion level, so that is not a harmless placeholder.
    n_absorbed = sum(int(c) for (_, to), c in counts.items()
                     if to in MARKOV_ABSORBING)
    meta = {
        'total_transitions': int(total_trans),
        'absorptions_observed': int(n_absorbed),
        'empirical': n_absorbed >= MIN_TRANSITIONS_FOR_EMPIRICAL,
        'laplace_alpha': 1.0,
    }

    states = list(MARKOV_STATES)
    si = {s: i for i, s in enumerate(states)}

    if meta['empirical']:
        T = laplace_row_probs(counts, states, alpha=meta['laplace_alpha'])
    else:
        T = _prior_transition_matrix(analytics, states, si)

    # Absorbing rows are identity (purchased / abandoned never leave).
    for s in MARKOV_ABSORBING:
        i = si[s]
        T[i, :] = 0.0
        T[i, i] = 1.0

    # Re-normalize transient rows so each is a proper probability
    # distribution after the absorbing-row overwrite above. (laplace_row_probs
    # already returns row-stochastic rows, so this is a guard against any
    # in-place edits leaving a row un-normalized.)
    for s in MARKOV_TRANSIENT:
        i = si[s]
        rs = T[i].sum()
        if rs > 0:
            T[i] = T[i] / rs

    return states, T, meta


def _prior_transition_matrix(analytics, states, si):
    """Data-informed prior when empirical transitions are insufficient.

    Calibration-preferring: when a dataset has been loaded, the counts come
    from ``analytics['calibration']`` (visitors, not just buyers) so the
    prior reflects the dataset's implied conversion + abandonment rather
    than whatever a (possibly empty) live run has accumulated.

    The heuristic edge weights set the shape of the chain; the abandonment
    edges are then rescaled so P(purchase | entering) equals the same
    conversion rate ``extract_simulation_parameters`` reports."""
    counts = defaultdict(int)
    total = max(1, int(_cval(analytics, 'total_customers', 0)))
    completed = int(_cval(analytics, 'completed_purchases', 0))
    abandoned = int(_cval(analytics, 'abandoned_carts', 0))
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

    T = laplace_row_probs(counts, states, alpha=1.0)

    # The edge weights above mix raw counts with probability-scaled counts,
    # so on their own they absorb into 'purchased' at a rate unrelated to
    # the observed conversion (0.22 against 0.30 on a calibrated dataset).
    # Consumers scale conversion by this probability, which would turn that
    # mismatch into a layout-independent revenue bias. Multiply every
    # transient->abandoned edge by one factor k (rows renormalized) and
    # bisect log k: P(purchase | entering) is monotone decreasing in k, and
    # Laplace smoothing keeps both absorbing columns positive, so any
    # target in (0, 1) is reachable.
    target = _observed_conversion_rate(analytics)
    t_idx = [si[s] for s in MARKOV_TRANSIENT]
    i_pur, i_ab = si['purchased'], si['abandoned']
    eye = np.eye(len(t_idx))

    def _scaled(log_k):
        Ts = T.copy()
        for i in t_idx:
            Ts[i, i_ab] *= np.exp(log_k)
            Ts[i] /= Ts[i].sum()
        return Ts

    def _p_purchase(Ts):
        Q = Ts[np.ix_(t_idx, t_idx)]
        return float(np.linalg.solve(eye - Q, Ts[t_idx, i_pur])[0])

    lo, hi = -30.0, 30.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _p_purchase(_scaled(mid)) > target:
            lo = mid
        else:
            hi = mid
    return _scaled(0.5 * (lo + hi))


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


def _calib(analytics):
    """Return ``analytics['calibration']`` (or {} if absent) -- the single
    home for dataset-derived inputs (rates + distributions + per-item dicts)
    after ``CalibratedParams.seed_into``. Live counters and arrays stay at
    the top of ``analytics``."""
    if not isinstance(analytics, dict):
        return {}
    cal = analytics.get('calibration')
    return cal if isinstance(cal, dict) else {}


def _calibrated(analytics):
    """True iff a dataset has been loaded (i.e. a non-empty calibration
    block exists). Used by analytical-tab gates so a calibrated session can
    run Optimize / Sensitivity / Markov / Projections / What-If / GA
    without first running a live simulation."""
    return bool(_calib(analytics))


def _cval(analytics, key, default=None):
    """Calibration-preferring read: ``analytics['calibration'][key]`` if
    present, else ``analytics[key]`` if present, else ``default``."""
    cal = _calib(analytics)
    if key in cal:
        return cal[key]
    if isinstance(analytics, dict):
        return analytics.get(key, default)
    return default


def _observed_conversion_rate(analytics):
    """Conversion rate the MC / projection layers use, clamped to
    [MC_CONV_CLAMP_LO, MC_CONV_CLAMP_HI] (the engine's own guard, 0.001
    to 0.999 in retail_literature): the calibrated assumption when a
    dataset is loaded, else purchases over customers who have already left. Customers still in
    the store have not had the chance to buy yet, so counting them in the
    denominator biases the rate low on short live runs. Before anyone has
    exited there is no evidence either way, so use the same 0.30
    assumption the dataset path defaults to rather than 1/total."""
    cal = _calib(analytics)
    if 'conversion_rate' in cal:
        rate = float(cal['conversion_rate'])
    else:
        completed = int(_cval(analytics, 'completed_purchases', 0))
        abandoned = int(_cval(analytics, 'abandoned_carts', 0))
        exited = completed + abandoned
        rate = completed / exited if exited > 0 else ASSUMED_CONVERSION_RATE
    return max(MC_CONV_CLAMP_LO, min(rate, MC_CONV_CLAMP_HI))


def net_base_revenue(rev_mean, imp_rate, imp_val):
    """Base (non-impulse) revenue per converting customer.

    Calibrated ``rev_per_converting_customer`` comes from TOTAL observed
    spend, which already includes impulse purchases; ``mc_engine`` then
    adds an impulse term on top. Passing the GROSS mean therefore
    double-counted the baseline impulse component (audit issue #3).
    Netting the expected impulse spend out here makes the decomposition
    exact: baseline projections reproduce the observed gross mean
    (net + imp_rate*imp_val = gross), and optimized projections add only
    the impulse DELTA the layout actually induces."""
    net = float(rev_mean) - float(imp_rate) * float(imp_val)
    return max(net, NET_BASE_REVENUE_MIN_FRAC * float(rev_mean), 0.01)


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
    """Calibrated parameters for MC / reporting from analytics + sim clock.

    Calibration-preferring: when a dataset has been loaded (a non-empty
    ``analytics['calibration']`` block exists), pull the empirical rates and
    distributions from there. Otherwise fall back to live counters and the
    ``sim_hours`` clock -- exactly the old behavior.
    """
    A = sim.analytics
    cal = _calib(A)
    has_cal = bool(cal)

    total = max(1, int(_cval(A, 'total_customers', 0)))
    n_completed = int(_cval(A, 'completed_purchases', 0))
    abandoned = int(_cval(A, 'abandoned_carts', 0))
    # Guarded denominator for per-purchase means only; the conversion and
    # abandonment rates below divide by customers who have exited.
    completed = max(1, n_completed)

    # Customers/hour: prefer the calibrated arrival rate, else derive from
    # live counters and the (floored) simulated-time clock. The calibration
    # path bypasses the ``sim_hours`` floor that otherwise inflates the rate
    # for short live runs.
    #
    # ``mc_engine`` draws visitors and applies conversion itself, so it needs
    # the VISITOR rate. The calibrated ``arrivals_per_hour`` counts buyers
    # (invoices); feeding it in directly would apply conversion twice.
    # A calibration block without ``visitors_per_hour`` is converted here.
    if has_cal and cal.get('visitors_per_hour'):
        customers_per_hour = float(cal['visitors_per_hour'])
    elif has_cal and cal.get('arrivals_per_hour'):
        customers_per_hour = (float(cal['arrivals_per_hour'])
                              / _observed_conversion_rate(A))
    else:
        run_sim_hrs = sim_hours(sim)
        customers_per_hour = total / run_sim_hrs

    conversion_rate = _observed_conversion_rate(A)
    if 'conversion_rate' in cal:
        # The calibrated abandoned count is implied non-buyers (visitors -
        # invoices), i.e. 1 - conversion, not cart abandonment. At 0.70 it
        # would saturate the 0.5 abandonment clamp in the fitness functions
        # and inflate conversion even for layouts with no abandonment change.
        # Use a dataset-provided rate if there is one, else the same
        # assumption the headless calibrated path makes (10% of
        # non-converters). A trajectory-only calibration carries no
        # purchase counts, so it keeps the live rate below.
        if 'abandonment_rate' in cal:
            abandonment_rate = float(cal['abandonment_rate'])
        else:
            abandonment_rate = (max(0.0, 1.0 - conversion_rate)
                                * ABANDON_FRAC_OF_NONCONVERTERS)
    else:
        exited = n_completed + abandoned
        abandonment_rate = abandoned / exited if exited > 0 else 0.0

    baskets = list(_cval(A, 'basket_sizes', []) or [])
    avg_basket = float(np.mean(baskets)) if baskets else STANDIN_AVG_BASKET
    std_basket = float(np.std(baskets, ddof=1)) if len(baskets) > 1 else max(avg_basket * 0.3, 0.5)

    rev_obs = list(_cval(A, 'customer_revenues', []) or [])
    total_rev_calibrated = float(_cval(A, 'total_revenue', 0.0))
    if len(rev_obs) >= 2:
        rev_per_cust = float(np.mean(rev_obs))
        rev_std = float(np.std(rev_obs, ddof=1))
    else:
        rev_per_cust = total_rev_calibrated / completed if completed else 0.0
        rev_std = max(rev_per_cust * REV_STD_FRAC_OF_MEAN, 0.1)

    rev_std = max(rev_std, 0.05)

    # Impulse share of purchases. Without a calibration this is a LIVE-only
    # signal, with numerator and denominator both from the live counters:
    # dividing live impulse items by a calibrated invoice count would drive
    # the rate to ~0 as soon as a single live impulse sale is recorded.
    #
    # Once a dataset supplies the purchase side, the live counter is the
    # wrong source: transaction records cannot label impulse buys, so the
    # rate would sit at 0 until an agent happens to grab something at the
    # till, which switches the impulse term off entirely and makes the
    # numbers depend on how long the session ran. Use the literature
    # stand-in instead -- the same one the headless calibrated path uses,
    # so the two agree. A trajectory-only calibration carries no purchase
    # data at all, so it keeps the live rate, as the conversion and
    # abandonment rates above do. Omnichannel measures a per-family
    # ``mean_impulse_rate`` and wins.
    impulse_purchases = A.get('impulse_purchases', 0)
    live_completed = max(int(A.get('completed_purchases', 0) or 0), 1)
    impulse_rate = impulse_purchases / live_completed
    impulse_rate_source = 'live_counter'
    if 'conversion_rate' in cal:
        impulse_rate = DEFAULT_CALIBRATED_IMPULSE_RATE
        impulse_rate_source = 'literature_default'
    cal_imp = cal.get('mean_impulse_rate')
    if cal_imp is not None and not (isinstance(cal_imp, float) and np.isnan(cal_imp)):
        impulse_rate = float(cal_imp)
        impulse_rate_source = 'dataset'

    impulse_sales = A.get('impulse_item_sales', {})
    avg_impulse_val = 0.0
    if impulse_sales and impulse_purchases > 0:
        total_impulse_items = sum(impulse_sales.values())
        if total_impulse_items > 0:
            # Sales are keyed the way the analytics layer keys items, which
            # on a multi-floor shop means 'F<floor>:<name>' for duplicated
            # names -- so use the simulation's own resolver rather than a
            # lookup that only understands plain names on the floor the user
            # happens to be viewing.
            price_of = getattr(sim, '_get_item_price', None)
            if callable(price_of):
                total_impulse_rev = sum(float(price_of(name)) * cnt
                                        for name, cnt in impulse_sales.items())
            else:
                prices = getattr(sim.shop, 'prices', {}) or {}
                total_impulse_rev = sum(float(prices.get(name, 0.0)) * cnt
                                        for name, cnt in impulse_sales.items())
            avg_impulse_val = total_impulse_rev / total_impulse_items
    if avg_impulse_val <= 0.0:
        avg_impulse_val = max(rev_per_cust * IMPULSE_VALUE_FRAC_OF_GROSS, 0.01)

    imp_rev_obs = list(_cval(A, 'impulse_revenues', []) or [])
    if len(imp_rev_obs) >= 2:
        imp_std = float(np.std(imp_rev_obs, ddof=1))
    else:
        imp_std = max(avg_impulse_val * IMPULSE_VALUE_CV, MC_SD_FLOOR)

    rev_by_area = dict(_cval(A, 'revenue_by_area', {}) or {})
    total_area_rev = sum(rev_by_area.values()) or 1.0
    area_shares = {k: v / total_area_rev for k, v in rev_by_area.items()}

    queue_times = A.get('queue_wait_times', [])
    avg_queue = float(np.mean(queue_times)) if queue_times else DEFAULT_QUEUE_TIME_S

    dwell = A.get('dwell_times_by_zone', {})
    avg_dwell = {}
    for zone, times in dwell.items():
        if times:
            avg_dwell[zone] = float(np.mean(times))

    return {
        'customers_per_hour': customers_per_hour,
        'conversion_rate': max(MC_CONV_CLAMP_LO,
                               min(conversion_rate, MC_CONV_CLAMP_HI)),
        'abandonment_rate': abandonment_rate,
        'avg_basket_size': avg_basket,
        'std_basket_size': std_basket,
        'basket_sizes_observed': list(baskets) if baskets else [],
        # NET of the expected impulse component -- see net_base_revenue().
        'rev_per_converting_customer': net_base_revenue(
            rev_per_cust, min(max(impulse_rate, 0.0), 1.0), avg_impulse_val),
        'rev_per_customer_gross': rev_per_cust,
        'rev_std': rev_std,
        'impulse_rate': min(max(impulse_rate, 0.0), 1.0),
        'impulse_rate_source': impulse_rate_source,
        'avg_impulse_value': avg_impulse_val,
        'impulse_value_std': imp_std,
        'area_revenue_shares': area_shares,
        'avg_queue_time': avg_queue,
        'avg_dwell_by_zone': avg_dwell,
        'total_observed_customers': total,
        'total_observed_revenue': total_rev_calibrated,
        'run_hours': sim_hours(sim),
        'run_wall_hours': max(sim.run_time / 3600.0, 1e-6),
        'sim_time_seconds': getattr(sim, 'sim_time', 0.0),
        'simulation_speed': getattr(sim, 'simulation_speed', 1.0),
        'calibrated_source': bool(has_cal),
    }


# Trial counts up to this size get a full cumulative table in
# ``_binomial_icdf``; larger ones use a window around the mean. The cache
# bound keeps a pathological arrival rate (millions of trials per day,
# every iteration a different n) from filling memory with tables.
_BINOM_FULL_TABLE_MAX = 512
_BINOM_TABLE_CACHE_MAX = 4096
_BINOM_EMPTY_TABLE = np.ones(1, dtype=np.float64)   # zero trials -> zero successes

# A table only pays for itself when enough draws share its trial count.
# Building one costs about as much as evaluating the binomial quantile
# directly for _BINOM_TABLE_MIN_DRAWS draws, plus one more per
# _BINOM_TABLE_ENTRIES_PER_DRAW entries. At high arrival rates the Poisson
# spread and the day noise give almost every iteration a trial count of
# its own, and a table of several hundred entries built to answer one
# lookup costs tens of times more than the draw; those counts go to the
# direct quantile instead. Both routes compute the same inverse CDF, so
# the route changes the cost, not the draw.
_BINOM_TABLE_MIN_DRAWS = 6
_BINOM_TABLE_ENTRIES_PER_DRAW = 128

# Days of draws generated per block in ``mc_engine`` are capped so that a
# block holds about this many iterations x days; it bounds the block's
# temporaries to a few MB whatever n_iter and the horizon are.
_MC_BLOCK_DRAWS = 1 << 18


def lognormal_spend_total(n, mean, sd, z):
    """A day's total spend over ``n`` spenders, as a non-negative draw with
    the exact mean ``n * mean`` and SD ``sqrt(n) * sd`` of a sum of ``n``
    independent spends of mean ``mean`` and SD ``sd``.

    The total is lognormal with those two moments (Fenton 1960's
    moment-matched approximation to a sum of lognormal spends):

        X = n * mean * exp(s * z - s**2 / 2),   s**2 = log(1 + cv**2 / n),

    with ``cv = sd / mean`` and ``z`` a standard normal. E[exp(s z -
    s^2/2)] = 1 exactly, so E[X] = n * mean for every n and cv, and
    Var[X] = (n mean)^2 (exp(s^2) - 1) = n sd^2. It replaces the normal
    N(n mean, sqrt(n) sd) floored at zero: at Online Retail II's
    coefficient of variation (about 2.3 per invoice, 3.4 before the
    reversal pairs were removed) that floor bound on thin days and added
    to the mean, and the normal's lower tail made a day's spend FALL as a
    converter was added (d/dn < 0 for z < -2 sqrt(n) / cv).

    Common random numbers: ``z`` is the only randomness, so the draw is a
    deterministic, increasing function of it. A layout changes ``mean`` and
    ``sd`` by the same factor (``layout_objective.layout_mc_kwargs``), which
    scales X by that factor exactly. The draw rises with ``n`` except where
    z > 4.14 (probability 1.7e-5) on a thin day: for z <= 6 only from at
    most 72 spenders at cv 3.4 (56 at cv 3, 33 at cv 2.3), the bound
    growing with z. Under common numbers a higher conversion rate can
    therefore lower about one day in a million, never the mean
    (tests/test_spend_law.py). ``n`` may be an array (per iteration); a
    non-positive ``n`` or ``mean`` spends nothing."""
    n = np.asarray(n, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    out = np.zeros(np.broadcast(n, z).shape, dtype=np.float64)
    mean = float(mean)
    if not mean > 0.0:
        return out
    cv2 = (max(float(sd), 0.0) / mean) ** 2
    n_b = np.broadcast_to(n, out.shape)
    z_b = np.broadcast_to(z, out.shape)
    pos = n_b > 0
    if not pos.any():
        return out
    npos = n_b[pos]
    s2 = np.log1p(cv2 / npos)
    out[pos] = npos * mean * np.exp(np.sqrt(s2) * z_b[pos] - 0.5 * s2)
    return out


# The spend law ``mc_engine`` draws a day's base and impulse spend from,
# named by retail_literature.MC_SPEND_LAW so every sidecar records it.
_SPEND_TOTALS = {'lognormal_moment_matched': lognormal_spend_total}


def _binomial_window(n, p):
    """``(k_lo, k_hi)``: the range of success counts the cumulative table
    for ``n`` trials covers (``n`` may be an array)."""
    n = np.asarray(n, dtype=np.float64)
    sd = np.sqrt(n * p * (1.0 - p))
    full = n <= _BINOM_FULL_TABLE_MAX
    k_lo = np.where(full, 0.0,
                    np.maximum(0.0, np.floor(n * p - 12.0 * sd - 20.0)))
    k_hi = np.where(full, n,
                    np.minimum(n, np.ceil(n * p + 12.0 * sd + 20.0)))
    return k_lo.astype(np.int64), k_hi.astype(np.int64)


def _binomial_cdf_table(n_val, p, log_ratio, window=None):
    """Cumulative binomial probabilities for ``n_val`` trials, as
    ``(k_lo, cdf)`` where ``cdf[j]`` is F(k_lo + j).

    Built from the pmf recurrence pmf(k)/pmf(k-1) = (n-k+1)/k * p/(1-p) in
    log space, which stays finite for any n (a direct q**n underflows).
    For large n only a +-12 standard-deviation window around the mean is
    tabulated; the omitted tails carry far less mass than double precision
    can represent, and renormalizing the window absorbs them. ``window``
    is ``_binomial_window(n_val, p)`` when the caller already has it.
    """
    if window is None:
        window = _binomial_window(n_val, p)
    k_lo, k_hi = (int(v) for v in window)
    k = np.arange(k_lo + 1, k_hi + 1, dtype=np.float64)
    log_pmf = np.empty(k_hi - k_lo + 1, dtype=np.float64)
    log_pmf[0] = 0.0
    np.cumsum(np.log((n_val - k + 1.0) / k) + log_ratio, out=log_pmf[1:])
    log_pmf -= log_pmf.max()
    cdf = np.cumsum(np.exp(log_pmf))
    cdf /= cdf[-1]
    return k_lo, cdf


def _binomial_icdf(u, n, p, cache=None):
    """Binomial draws by inverse CDF: the smallest k with F(k; n, p) >= u.

    ``u`` holds one uniform per iteration, ``n`` the per-iteration trial
    count, ``p`` is a scalar. Reading the count off a single uniform makes
    it a monotone function of that uniform, so two parameter sets that are
    handed the same uniforms share nearly all of their sampling variation
    -- the property paired (common-random-number) comparisons rely on.
    numpy's binomial sampler has neither property at these sizes: it is not
    monotone in p, and it consumes a p-dependent number of uniforms, which
    also knocks every later draw out of step.

    Iterations are grouped by trial count so one cumulative table serves
    every iteration sharing an n. ``cache`` (a dict owned by the caller,
    one per probability) reuses those tables across calls, which is where
    most of the work would otherwise be repeated. A trial count shared by
    too few draws to pay for a table is answered by scipy's binomial
    quantile, which evaluates the same inverse CDF directly.
    """
    n = np.asarray(n)
    u = np.asarray(u)
    out = np.zeros(n.shape, dtype=np.int64)
    p = float(p)
    if p <= 0.0 or out.size == 0:
        return out
    if p >= 1.0:
        return n.astype(np.int64)
    if cache is None:
        cache = {}

    log_ratio = float(np.log(p) - np.log1p(-p))
    # Sort once so each trial count addresses a contiguous block, instead
    # of building a boolean mask over the whole sample per distinct n.
    order = np.argsort(n, kind='stable')
    n_sorted = n[order]
    u_sorted = u[order]
    uniq, starts = np.unique(n_sorted, return_index=True)
    sizes = np.append(starts[1:], n_sorted.size) - starts

    k_lo_w, k_hi_w = _binomial_window(uniq, p)
    in_cache = np.fromiter((int(v) in cache for v in uniq), dtype=bool,
                           count=uniq.size)
    tabulate = (in_cache | (uniq <= 0)
                | (sizes >= _BINOM_TABLE_MIN_DRAWS
                   + (k_hi_w - k_lo_w + 1) / _BINOM_TABLE_ENTRIES_PER_DRAW))

    res_sorted = np.zeros(n_sorted.size, dtype=np.int64)
    gid = np.repeat(np.arange(uniq.size), sizes)
    by_table = tabulate[gid]

    tab_groups = np.flatnonzero(tabulate)
    if tab_groups.size:
        tables, k_lo = [], np.zeros(tab_groups.size, dtype=np.int64)
        for j, g in enumerate(tab_groups):
            n_val = int(uniq[g])
            if n_val <= 0:
                # Binomial(0, p) is 0; a one-entry table keeps the lookup
                # below uniform over groups.
                tables.append(_BINOM_EMPTY_TABLE)
                continue
            entry = cache.get(n_val)
            if entry is None:
                entry = _binomial_cdf_table(n_val, p, log_ratio,
                                            (k_lo_w[g], k_hi_w[g]))
                if len(cache) < _BINOM_TABLE_CACHE_MAX:
                    cache[n_val] = entry
            k_lo[j] = entry[0]
            tables.append(entry[1])

        # Lay the per-n tables end to end, each block lifted by 2 * (its
        # position). Table values live in [0, 1], so the concatenation is
        # globally increasing and one search over it serves every iteration
        # -- far cheaper than one call per distinct trial count.
        lengths = np.fromiter((t.size for t in tables), dtype=np.int64,
                              count=len(tables))
        block_start = np.concatenate(([0], np.cumsum(lengths)[:-1]))
        big = np.concatenate(tables) + 2.0 * np.repeat(
            np.arange(tab_groups.size), lengths)
        slot = np.zeros(uniq.size, dtype=np.int64)
        slot[tab_groups] = np.arange(tab_groups.size)
        tg = slot[gid[by_table]]
        pos = np.searchsorted(big, u_sorted[by_table] + 2.0 * tg, side='left')
        res_sorted[by_table] = pos - block_start[tg] + k_lo[tg]

    direct = ~by_table
    if direct.any():
        from scipy.stats import binom
        # ppf(0) is -1 by scipy's convention for discrete laws; a uniform of
        # exactly 0 maps to the lowest count, as it does in a table.
        res_sorted[direct] = np.clip(
            binom.ppf(u_sorted[direct], n_sorted[direct], p),
            0, n_sorted[direct]).astype(np.int64)

    out[order] = res_sorted
    return np.minimum(out, n.astype(np.int64))


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
    day_noise_std=DEFAULT_DAY_NOISE_STD,
    rng=None,
):
    """
    Monte Carlo daily revenue with:
    - Poisson arrivals (per simulated day)
    - Binomial conversion, drawn by inverse CDF
    - The day's spend over its converters (and over its impulse buyers)
      drawn as one non-negative total with the exact mean and variance of
      a sum of independent per-converter spends of mean ``rev_mean`` and
      SD ``rev_std`` (``lognormal_spend_total``; retail_literature
      MC_SPEND_LAW), so no floor shifts the mean
    - Optional lognormal day-level traffic multiplier, drawn independently
      each day with log-factor ~ Normal(-sigma^2/2, sigma^2) so its mean is
      exactly 1 and its spread does not grow with the horizon

    Randomness is organised so that two parameter sets evaluated under the
    same seed walk the same sample path (common random numbers). Each
    family of draws -- day noise, arrivals, conversion uniforms, basket
    normals, revenue normals, impulse uniforms, impulse normals -- runs on
    its own child generator, and the counts are inverse-CDF functions of
    their uniforms, so a small change in conversion or impulse rate moves
    the result by a small amount instead of redrawing everything that
    follows. Paired differences between nearby layouts therefore measure
    the layouts, not the sampler.

    Basket size does not reach revenue here. ``avg_bsk``, ``std_bsk`` and
    ``observed_baskets`` (only its mean and SD are used, in a normal
    approximation to the day's item count) feed the tracked
    ``daily_baskets`` series and nothing else; a layout's basket effect
    reaches revenue only through ``rev_mean`` (and ``rev_std``), which the
    caller scales by b(s)/b0 (``layout_objective.layout_drivers``). The
    mean of the returned totals has a closed form,
    ``experiments.closed_form.mc_expected_total``, and it is exact: the
    spend draws are non-negative by construction and keep their means.

    Note: within-day non-homogeneity (the empirical hour-of-day arrival
    shape) is applied to the LIVE simulator's spawn loop via
    ``simulation.hourly_profile``, but NOT to this MC daily aggregator.
    A calibrated profile is scaled so that it sums to
    ``DEFAULT_OP_HOURS_PER_DAY`` over the open hours, so a live day driven
    at the calibrated visitor rate delivers the same expected daily volume
    as ``cph * op_hours`` here. Modelling within-day Jensen-effects on
    conversion or basket-size would require restructuring this engine to be
    hourly rather than daily and is out of scope for the current paper.
    """
    if rng is None:
        rng = np.random
    spend_total = _SPEND_TOTALS[MC_SPEND_LAW]

    # One integer from the caller's stream seeds every draw family below,
    # so seeding that stream (``np.random.seed(s)``, or a Generator handed
    # in) still determines the whole result and equal inputs still give
    # equal output. What it buys is independence between the families:
    # layout-dependent parameters can no longer shift the position of the
    # stream that later draws read from.
    if hasattr(rng, 'integers'):
        root_seed = int(rng.integers(0, 2 ** 63 - 1))
    else:
        root_seed = int(rng.randint(0, 2 ** 63 - 1, dtype=np.int64))
    (day_rng, arr_rng, conv_rng, bsk_rng, rev_rng, imp_u_rng,
     imp_z_rng) = [np.random.default_rng(s)
                   for s in np.random.SeedSequence(root_seed).spawn(7)]

    # Cumulative binomial tables, reused across blocks of days: the
    # conversion and impulse probabilities are fixed for the whole call, so
    # only the trial count changes from day to day.
    conv_tables, imp_tables = {}, {}

    if impulse_value_std is None:
        impulse_value_std = max(imp_val * IMPULSE_VALUE_CV, MC_SD_FLOOR)

    day_multipliers = np.ones(7)
    day_multipliers[5] = wknd_mult
    day_multipliers[6] = wknd_mult

    daily_rev = np.zeros((n_iter, n_days))
    daily_cust = np.zeros((n_iter, n_days))
    daily_converting = np.zeros((n_iter, n_days))
    daily_impulse_rev = np.zeros((n_iter, n_days))
    daily_baskets = np.zeros((n_iter, n_days))
    obs_baskets = (
        list(observed_baskets) if observed_baskets and len(observed_baskets) >= 5
        else None
    )
    # Pre-compute empirical mean/std of observed baskets ONCE. We use these
    # to draw the per-iteration basket total via a normal approximation
    # (sum of n i.i.d. ~ N(n*mu, sqrt(n)*sigma)) instead of materialising a
    # giant (n_iter, max_nc) sample matrix per day -- which made the
    # sensitivity loop allocate gigabytes and hang the GUI for realistic
    # customers_per_hour values.
    if obs_baskets is not None:
        _obs_arr = np.asarray(obs_baskets, dtype=np.float64)
        obs_mean = float(_obs_arr.mean())
        obs_std  = float(_obs_arr.std(ddof=1)) if _obs_arr.size > 1 else max(obs_mean * 0.3, 0.5)
        obs_std  = max(obs_std, 0.05)
    else:
        obs_mean = obs_std = None

    # Hard cap (MC_LAMBDA_CAP) to keep Poisson sampling numerically sane
    # even when the caller passes a pathological customers_per_hour (e.g.
    # sim_time floor was zero on first Optimize click before enough data
    # accumulated). The guards here are retail_literature's MC_* NUMERICAL
    # GUARDS, which experiments.closed_form applies too.

    # Sanitize the caller-supplied rate up-front so we never end up with
    # NaN/inf inside the loop.
    if not np.isfinite(cph) or cph < 0:
        cph = 0.0

    # Days are drawn a block at a time, as (days, n_iter) arrays. Each
    # family has a stream of its own, so a block yields exactly the values
    # a day-by-day loop would; what the block buys is that the binomial
    # lookups see every day in it at once, so a trial count that recurs
    # across days is tabulated and searched once.
    block_days = int(max(1, min(n_days, _MC_BLOCK_DRAWS // max(int(n_iter), 1))))
    for d0 in range(0, n_days, block_days):
        days = np.arange(d0, min(d0 + block_days, n_days))
        cols = slice(d0, d0 + days.size)
        shape = (days.size, n_iter)
        gf = 1.0 + monthly_growth * (days / 30.0)
        lam = np.maximum(cph * op_hours * day_multipliers[days % 7] * gf,
                         MC_LAMBDA_FLOOR)
        lam = np.minimum(np.where(np.isfinite(lam), lam, MC_LAMBDA_CAP),
                         MC_LAMBDA_CAP)[:, None]

        if day_noise_std > 0:
            # A fresh factor per day keeps the traffic noise stationary, so
            # its spread does not grow with the horizon. The -sigma^2/2 shift
            # gives E[exp(log_day_factor)] = 1. The clip at 8 sigma only
            # guards against overflow; its effect on the mean is far below
            # floating-point noise.
            half_var = 0.5 * day_noise_std ** 2
            log_day_factor = np.clip(
                day_rng.normal(0.0, day_noise_std, shape) - half_var,
                -8.0 * day_noise_std - half_var,
                8.0 * day_noise_std - half_var,
            )
            lam_vec = lam * np.exp(log_day_factor)
            lam_vec = np.clip(lam_vec, 0.0, MC_LAMBDA_CAP)
        else:
            lam_vec = np.repeat(lam, n_iter, axis=1)

        n_cust = arr_rng.poisson(lam_vec)
        daily_cust[:, cols] = n_cust.T
        n_conv = _binomial_icdf(conv_rng.random(shape).ravel(), n_cust.ravel(),
                                min(max(conv, MC_CONV_CLAMP_LO),
                                    MC_CONV_CLAMP_HI),
                                conv_tables).reshape(shape)
        daily_converting[:, cols] = n_conv.T

        nc_f = n_conv.astype(np.float64)
        nc_sqrt = np.sqrt(nc_f)
        # Sum of n_conv i.i.d. basket draws is approximately
        # N(n_conv*mu, sqrt(n_conv)*sigma) -- O(n_iter) per day instead of
        # O(n_iter * max(n_conv)), which is what made the sensitivity loop
        # allocate gigabytes. Writing it as mean + sd*z keeps the standard
        # normals themselves free of the basket parameters, so basket size
        # cannot shift any other draw.
        z_bsk = bsk_rng.standard_normal(shape)
        if obs_baskets is not None:
            total_items = nc_f * obs_mean + nc_sqrt * obs_std * z_bsk
        else:
            # Scaling one per-customer draw by n_conv would give an SD of
            # n_conv*sigma rather than sqrt(n_conv)*sigma.
            total_items = nc_f * avg_bsk + nc_sqrt * max(std_bsk, 0.5) * z_bsk
        total_items = np.maximum(total_items, 0.0)
        daily_baskets[:, cols] = total_items.T

        # The day's spend over n_conv converters: non-negative, with the
        # exact mean n*mu and SD sqrt(n)*sigma of n independent spends
        # (``lognormal_spend_total``), read off this family's standard
        # normals so CRN pairing is unchanged.
        rs = max(rev_std, MC_SD_FLOOR)
        base_rev = spend_total(nc_f, rev_mean, rs,
                               rev_rng.standard_normal(shape))

        n_imp = _binomial_icdf(imp_u_rng.random(shape).ravel(), n_conv.ravel(),
                               min(max(imp_rate, 0.0), 1.0),
                               imp_tables).reshape(shape)
        ni_f = n_imp.astype(np.float64)
        is_ = max(impulse_value_std, MC_SD_FLOOR)
        imp_rev = spend_total(ni_f, imp_val, is_,
                              imp_z_rng.standard_normal(shape))
        daily_impulse_rev[:, cols] = imp_rev.T

        daily_rev[:, cols] = (base_rev + imp_rev).T

    totals = daily_rev.sum(axis=1)
    return {
        'mean': float(totals.mean()),
        'std': float(totals.std()),
        'p5': float(np.percentile(totals, 5)),
        'p95': float(np.percentile(totals, 95)),
        # Per-iteration horizon totals. The optimize pipeline runs the
        # engine in chunks and concatenates these to form the baseline /
        # optimized samples -- without this key it silently fell back to
        # aggregating the CHUNK MEANS, which understated the dispersion
        # (std / p5 / p95 of 8 means instead of n_iter totals).
        'totals': totals,
        'daily_means': daily_rev.mean(axis=0),
        'daily_revenue': daily_rev,
        'daily_customers': daily_cust,
        'daily_converting': daily_converting,
        'daily_impulse_rev': daily_impulse_rev,
        'daily_baskets': daily_baskets,
    }
