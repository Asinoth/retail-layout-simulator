"""Exact expectation of the Monte Carlo fitness (Eq. (8) of the paper).

A layout reaches ``mc_engine`` only through the deterministic drivers of
``layout_objective.layout_drivers``, so the 'Monte Carlo objective' the
searches optimize has a closed-form mean. This module computes it exactly:

    E[R_D] = sum_d lambda_d * p * (mu_r' + q * mu_imp)

with lambda_d = max(cph * T_op * m_dow(d) * g(d), 0.1) the engine's day
rate (weekend multiplier on days d % 7 in {5, 6}, growth g), p and q the
conversion and impulse rate after the engine's own clamps, mu_r' the net
base spend per converter times b(s)/b0 and the queue factor, and mu_imp the
impulse value. Nothing else enters: the engine draws a day's spend as a
non-negative total that keeps its mean exactly (``MC_SPEND_LAW``,
``sim_calibration.lognormal_spend_total``), so there is no floor term.

The engine used to draw that total as a normal floored at zero, and the
floor added E[max(-X, 0)] to each day's mean -- negligible when spend per
converter is tight (the synthetic scenarios: below 1e-9 of the total), not
negligible on thin days of heavy-tailed invoices (Online Retail II).
``legacy_floor_bias`` still computes what it added, exactly -- the
converters on a day are Poisson(lambda_d F p) given the lognormal day
factor F (Poisson arrivals thinned by a binomial are Poisson), so the
correction is a Poisson sum integrated over F by Gauss-Hermite quadrature
-- so a report can state how far results made under the former law were
biased.

What is left out, as ``mc_engine``'s own documentation leaves it out: the
+-8 sigma clip on the day factor and the 1e6 cap on a day's rate (both
below floating-point resolution at any rate the runners use), and the
binomial table's +-12 SD window. ``mc_engine`` is therefore an unbiased
estimator of ``mc_expected_total`` for the same arguments, and a paired
difference of two layouts under common random numbers is an unbiased
estimator of the difference of their ``expected_revenue``.

Use ``expected_revenue`` to score a layout: it takes what ``_ga_fitness``
takes (a chromosome and item names on a shop, or a score and breakdown
already computed), the base parameters with their anchor, the horizon, and
optional elasticity overrides, and returns the value the fitness converges
to as its iteration count grows.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
from scipy.special import gammaln, ndtr

from layout_objective import layout_drivers, layout_mc_kwargs
from retail_literature import (DEFAULT_DAY_NOISE_STD, DEFAULT_OP_HOURS_PER_DAY,
                               DEFAULT_WEEKEND_MULTIPLIER, IMPULSE_VALUE_CV,
                               MC_CONV_CLAMP_HI, MC_CONV_CLAMP_LO,
                               MC_LAMBDA_CAP, MC_LAMBDA_FLOOR, MC_SD_FLOOR)

__all__ = ['expected_revenue', 'mc_expected_total', 'legacy_floor_bias',
           'spend_floor_correction']

# mc_engine's guards -- the floor and cap on a day's arrival rate and the
# clamps on the conversion probability and the spend SDs (the last read by
# ``legacy_floor_bias`` only) -- are
# retail_literature's MC_* NUMERICAL GUARDS, imported above by both modules,
# so the closed form cannot drift from the engine it describes.

# Gauss-Hermite rule for the lognormal day factor. 48 nodes integrate the
# Poisson sums below to machine precision for any day-noise SD the engine
# is run at.
_GH_X, _GH_W = np.polynomial.hermite.hermgauss(48)
_GH_W = _GH_W / np.sqrt(np.pi)

# A converter count n contributes to the former law's floor term only while
# n * (mean / sd)**2 / 2 is below this (exp(-_FLOOR_EXP) ~ 4e-18).
_FLOOR_EXP = 40.0


def _floor_terms(n: np.ndarray, mean: float, sd: float) -> np.ndarray:
    """E[max(X, 0)] - E[X] for X ~ N(n * mean, sqrt(n) * sd), elementwise
    in ``n`` (n >= 1): sqrt(n) sd phi(a) - n mean Phi(-a), a = sqrt(n) mean / sd.
    """
    rn = np.sqrt(n)
    a = rn * mean / sd
    phi = np.exp(-0.5 * a * a) / np.sqrt(2.0 * np.pi)
    return rn * sd * phi - n * mean * ndtr(-a)


def spend_floor_correction(mu: float, mean: float, sd: float,
                           day_noise_std: float) -> float:
    """What the FORMER day-spend law's zero floor added to one day's
    expected spend (the engine no longer floors; see ``legacy_floor_bias``).

    ``mu`` is the day's expected number of spenders before the day factor
    (converters for the base spend, impulse buyers for the impulse spend),
    each spending ``mean`` with SD ``sd``. Given the lognormal day factor F
    (log F ~ N(-s^2/2, s^2), s = ``day_noise_std``) the count is
    Poisson(mu F), and the former law drew the day's spend as
    N(n mean, sqrt(n) sd) floored at zero; the result is
    E_F[ sum_n Pois(n; mu F) (E[max(X_n, 0)] - n mean) ].
    """
    if mu <= 0.0 or sd <= 0.0:
        return 0.0
    if mean > 0.0:
        n_max = int(np.ceil(2.0 * _FLOOR_EXP * (sd / mean) ** 2)) + 1
    else:
        n_max = None                     # no decay; bounded by the Poisson window
    if day_noise_std > 0.0:
        z = -0.5 * day_noise_std ** 2 + np.sqrt(2.0) * day_noise_std * _GH_X
        mus = mu * np.exp(z)
        weights = _GH_W
    else:
        mus = np.array([mu])
        weights = np.array([1.0])
    hi_mu, lo_mu = float(mus.max()), float(mus.min())
    n_hi = int(np.ceil(hi_mu + 12.0 * np.sqrt(hi_mu) + 40.0))
    n_lo = max(1, int(np.floor(lo_mu - 12.0 * np.sqrt(lo_mu) - 40.0)))
    if n_max is not None:
        n_hi = min(n_hi, n_max)
    if n_hi < n_lo:
        return 0.0
    n = np.arange(n_lo, n_hi + 1, dtype=np.float64)
    c = _floor_terms(n, mean, sd)
    log_pmf = (n[None, :] * np.log(mus[:, None]) - mus[:, None]
               - gammaln(n + 1.0)[None, :])
    per_node = np.exp(log_pmf) @ c
    return float(weights @ per_node)


def _day_rates(cph: float, n_days: int, op_hours: float, wknd_mult: float,
               monthly_growth: float) -> np.ndarray:
    """The engine's expected arrivals on each day of the horizon, after its
    floor and cap."""
    if not np.isfinite(cph) or cph < 0:
        cph = 0.0
    days = np.arange(int(n_days))
    mult = np.where(np.isin(days % 7, (5, 6)), float(wknd_mult), 1.0)
    growth = 1.0 + monthly_growth * (days / 30.0)
    lam = np.maximum(cph * op_hours * mult * growth, MC_LAMBDA_FLOOR)
    return np.minimum(np.where(np.isfinite(lam), lam, MC_LAMBDA_CAP),
                      MC_LAMBDA_CAP)


def mc_expected_total(cph: float, conv: float, rev_mean: float,
                      rev_std: float, imp_rate: float, imp_val: float,
                      n_days: int,
                      op_hours: float = DEFAULT_OP_HOURS_PER_DAY,
                      wknd_mult: float = DEFAULT_WEEKEND_MULTIPLIER,
                      monthly_growth: float = 0.0,
                      impulse_value_std: Optional[float] = None,
                      day_noise_std: float = DEFAULT_DAY_NOISE_STD,
                      include_floors: bool = False,
                      **_unused: Any) -> float:
    """The mean ``mc_engine`` converges to for the same arguments.

    Takes ``mc_engine``'s keyword arguments (the ones that cannot move the
    mean -- ``rev_std``, ``impulse_value_std``, ``day_noise_std``,
    ``avg_bsk``, ``std_bsk``, ``observed_baskets``, ``n_iter``, ``rng`` --
    are accepted and ignored), so ``mc_expected_total(**kw)`` and
    ``mc_engine(**kw)`` describe the same model. A non-positive spend mean
    spends nothing, as in the engine. ``include_floors`` is kept for callers
    written against the former floored law and changes nothing: the engine
    has no floor now (``legacy_floor_bias`` gives what it used to add)."""
    lam = _day_rates(cph, n_days, op_hours, wknd_mult, monthly_growth)
    p = min(max(conv, MC_CONV_CLAMP_LO), MC_CONV_CLAMP_HI)
    q = min(max(imp_rate, 0.0), 1.0)
    return float(lam.sum()) * p * (max(rev_mean, 0.0)
                                   + q * max(imp_val, 0.0))


def legacy_floor_bias(cph: float, conv: float, rev_mean: float,
                      rev_std: float, imp_rate: float, imp_val: float,
                      n_days: int,
                      op_hours: float = DEFAULT_OP_HOURS_PER_DAY,
                      wknd_mult: float = DEFAULT_WEEKEND_MULTIPLIER,
                      monthly_growth: float = 0.0,
                      impulse_value_std: Optional[float] = None,
                      day_noise_std: float = DEFAULT_DAY_NOISE_STD,
                      **_unused: Any) -> float:
    """How much the FORMER day-spend law -- a normal floored at zero --
    raised the engine's mean above ``mc_expected_total`` for the same
    arguments (``mc_engine``'s keyword arguments, as there).

    Results computed before the switch to ``MC_SPEND_LAW`` carry this bias;
    the difference of two layouts' biases is how far a paired lift made
    under the former law was off."""
    lam = _day_rates(cph, n_days, op_hours, wknd_mult, monthly_growth)
    p = min(max(conv, MC_CONV_CLAMP_LO), MC_CONV_CLAMP_HI)
    q = min(max(imp_rate, 0.0), 1.0)
    rs = max(rev_std, MC_SD_FLOOR)
    if impulse_value_std is None:
        impulse_value_std = max(imp_val * IMPULSE_VALUE_CV, MC_SD_FLOOR)
    isd = max(impulse_value_std, MC_SD_FLOOR)
    total = 0.0
    rates, counts = np.unique(lam, return_counts=True)
    for rate, k in zip(rates, counts):
        total += k * spend_floor_correction(rate * p, rev_mean, rs,
                                            day_noise_std)
        if q > 0.0:
            total += k * spend_floor_correction(rate * p * q, imp_val, isd,
                                                day_noise_std)
    return float(total)


def expected_revenue(base_params: Mapping[str, Any], horizon_days: int = 30, *,
                     score: Optional[float] = None,
                     breakdown: Optional[Mapping[str, Any]] = None,
                     shop: Any = None,
                     chromosome: Any = None,
                     item_names: Optional[Sequence[str]] = None,
                     elasticities: Optional[Mapping[str, float]] = None,
                     anchor: Optional[Mapping[str, Any]] = None,
                     basket_exclude: Iterable[str] = (),
                     include_floors: bool = False,
                     allow_zero_anchor: bool = False) -> float:
    """Exact expected revenue of one layout over ``horizon_days``: the value
    ``shop._ga_fitness(chromosome, item_names, base_params, horizon_days,
    mc_iters)`` converges to as ``mc_iters`` grows.

    The layout is given either as ``score`` and ``breakdown`` (what
    ``_ga_compute_layout_score`` returns) or as ``chromosome`` and
    ``item_names`` on ``shop``, which is then scored here.

    ``base_params`` are the run's base parameters, anchor included
    (``experiments._common.anchor_base_params``); ``anchor`` overrides that
    anchor, which the weight sweeps use to re-score the anchor layout under
    the same weights as the layout. Parameters with no anchor, and no
    ``anchor`` argument, raise ``layout_objective.MissingAnchorError``:
    parameters rebuilt by ``base_params_for*`` carry none, and re-scoring
    with them would silently return the unanchored objective. Only a caller
    with no as-built layout passes ``allow_zero_anchor=True``. ``elasticities`` overrides any of
    ``'conv'``, ``'imp'``, ``'bsk'`` (default: the band midpoints the GA
    searches under). ``basket_exclude`` takes criteria out of the basket
    driver only (``layout_objective.SHORT_PATH_CRITERIA`` for the variant
    without the short-path criteria). The day is
    ``DEFAULT_OP_HOURS_PER_DAY`` long, day d of the horizon is weekday
    d % 7 with the cited weekend multiplier on days 5 and 6, and the day
    noise is the engine's default, all as ``_ga_fitness`` runs it.

    ``include_floors`` is accepted and changes nothing (the engine has no
    floor term any more; see ``legacy_floor_bias``).

    Stable signature: the later re-scoring of every final layout of Figures
    A-C and of the MC ground truth calls this function.
    """
    if score is None:
        if shop is None or chromosome is None or item_names is None:
            raise TypeError("expected_revenue needs score and breakdown, or "
                            "shop, chromosome and item_names")
        score, breakdown = shop._ga_compute_layout_score(
            np.asarray(chromosome, dtype=np.float64), list(item_names),
            base_params)
    elif breakdown is None:
        raise TypeError("expected_revenue needs the breakdown with the score")
    drivers = layout_drivers(score, breakdown, base_params,
                             elasticities=elasticities, anchor=anchor,
                             basket_exclude=basket_exclude,
                             allow_zero_anchor=allow_zero_anchor)
    kw = layout_mc_kwargs(drivers, base_params, n_days=int(horizon_days),
                          n_iter=1)
    return mc_expected_total(**kw)
