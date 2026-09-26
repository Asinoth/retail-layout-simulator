"""Paired comparison of two layouts under the layout model.

The interactive tools compare two layouts in three places: the Optimize
pipeline's projection (the as-built layout against the one it applied,
scored at optimize start), the same comparison re-scored after the live
POST window, and the What-If tab (any two saved snapshots). All three ask
one question -- how much more revenue does the model expect from layout B
than from layout A over the horizon -- and answer it here, one way:

* Both layouts reach ``mc_engine`` as keyword arguments that differ only
  in their layout drivers (``layout_objective.layout_mc_kwargs``), and
  both runs of a chunk start from ONE seed. The engine runs every family
  of draws on its own child stream and draws the conversion and impulse
  counts by inverse CDF, so the two runs walk the same sample path
  (common random numbers): iteration i of A and iteration i of B are a
  pair, and pairs are independent across iterations. The GUI used to draw
  both arms from a hand-rolled loop whose binomial draws consumed a
  parameter-dependent number of variates, so the streams drifted apart at
  the first differing draw and its 'paired' differences were close to
  independent (correlation about 0.01). Running the engine itself also
  gives both arms the engine's day-spend law (retail_literature
  MC_SPEND_LAW: ``sim_calibration.lognormal_spend_total``, non-negative
  with the exact mean and SD of a sum of independent spends), where the
  former loop drew a normal floored at zero, which was biased upward on
  thin days.

* The estimate is the mean paired difference B - A with a Student-t
  interval over the pairs, and the relative lift is that difference over
  A's mean, with a delta-method interval. The interval is Monte Carlo
  precision under the model: it narrows as the iteration count grows and
  says nothing about how uncertain the model itself is, so no p-value,
  significance flag or standardized effect size is computed, and nothing
  but the interval decides whether a direction is named. (A Welch p-value
  between two simulated distributions goes to zero with the iteration
  count, and Cohen's d grows with the horizon -- about sqrt(horizon),
  since the lift grows with the horizon and the spread of a total only
  with its square root -- so neither measures the layouts.)

* The exact expectation of each arm (``experiments.closed_form.
  mc_expected_total`` on the same keyword arguments) is reported beside
  the estimate: it is the value the Monte Carlo mean converges to, the
  same quantity the headless experiments score layouts with.

Tk-free, so the headless tests exercise exactly what the GUI shows.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np
from scipy import stats as sp_stats

from sim_calibration import mc_engine

#: Coverage of the reported intervals.
DEFAULT_LEVEL = 0.95


def _arm_summary(totals: np.ndarray, daily_means: np.ndarray,
                 exact: float) -> Dict[str, Any]:
    """Spread of one arm's simulated horizon totals. These percentiles
    describe the outcomes the model produces for that layout, which are far
    wider than the Monte Carlo uncertainty about its mean."""
    return {
        'mean': float(totals.mean()),
        'std': float(totals.std(ddof=1)) if totals.size > 1 else 0.0,
        'median': float(np.median(totals)),
        'p5': float(np.percentile(totals, 5)),
        'p95': float(np.percentile(totals, 95)),
        'totals': totals,
        'daily_means': daily_means,
        'exact_mean': float(exact),
    }


def summarize_pairs(totals_a: np.ndarray, totals_b: np.ndarray,
                    level: float = DEFAULT_LEVEL) -> Dict[str, Any]:
    """Paired-difference statistics of B - A over independent pairs.

    Returns the mean difference with its standard error and t interval,
    the relative lift (mean difference over A's mean) with a delta-method
    interval, the correlation of the paired totals (a check that the
    common random numbers took: independent draws give about zero) and the
    difference series itself."""
    a = np.asarray(totals_a, dtype=np.float64)
    b = np.asarray(totals_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or a.size < 2:
        raise ValueError('need two equal-length series of at least two pairs')
    n = a.size
    d = b - a
    mean_d = float(d.mean())
    se_d = float(d.std(ddof=1) / np.sqrt(n))
    tcrit = float(sp_stats.t.ppf(0.5 + level / 2.0, n - 1))
    mean_a = float(a.mean())
    # Relative lift R = mean(d) / mean(a). Delta method with the sample
    # covariance of the two means (both are means over the same pairs).
    if mean_a != 0.0:
        r = mean_d / mean_a
        var_d = float(d.var(ddof=1)) / n
        var_a = float(a.var(ddof=1)) / n
        cov_da = float(np.cov(d, a, ddof=1)[0, 1]) / n
        var_r = max(var_d - 2.0 * r * cov_da + r * r * var_a, 0.0) \
            / (mean_a * mean_a)
        se_r = float(np.sqrt(var_r))
        rel = (100.0 * r, (100.0 * (r - tcrit * se_r),
                           100.0 * (r + tcrit * se_r)))
    else:
        rel = (float('nan'), (float('nan'), float('nan')))
    if a.std() > 0 and b.std() > 0:
        corr = float(np.corrcoef(a, b)[0, 1])
    else:
        corr = float('nan')
    return {
        'n_pairs': int(n),
        'level': float(level),
        'lift_mean': mean_d,
        'lift_se': se_d,
        'lift_ci': (mean_d - tcrit * se_d, mean_d + tcrit * se_d),
        'lift_pct': rel[0],
        'lift_pct_ci': rel[1],
        'pair_correlation': corr,
        'diffs': d,
    }


def direction(lift_ci) -> Optional[str]:
    """'B' or 'A' when the interval of B - A lies wholly on one side of
    zero, else None: the only rule that names a layout as ahead."""
    lo, hi = lift_ci
    if lo > 0:
        return 'B'
    if hi < 0:
        return 'A'
    return None


def paired_comparison(kw_a: Mapping[str, Any], kw_b: Mapping[str, Any],
                      seed: int, n_chunks: int = 1,
                      level: float = DEFAULT_LEVEL,
                      between_chunks: Optional[Callable[[int], None]] = None
                      ) -> Dict[str, Any]:
    """Compare layouts A and B under common random numbers.

    ``kw_a`` / ``kw_b`` are ``mc_engine`` keyword arguments (``n_iter`` is
    per chunk), normally from ``layout_objective.layout_mc_kwargs``. Chunk c
    of both arms runs on ``RandomState(seed + c)``; ``between_chunks(c)``
    is called after each chunk (the GUI pumps its event loop there). The
    same ``seed`` reproduces the comparison exactly.

    Returns ``{'A': arm, 'B': arm, 'paired': summarize_pairs(...),
    'exact': {...}, 'direction': 'A' | 'B' | None, 'seed', 'n_chunks',
    'n_days'}``, where each arm carries its totals, their spread and its
    exact expectation, and ``exact`` holds the exact lift and its share of
    A's exact mean."""
    from experiments.closed_form import mc_expected_total

    if int(kw_a['n_days']) != int(kw_b['n_days']):
        raise ValueError('both layouts must be projected over one horizon')
    seed = int(seed) % (2 ** 32 - max(int(n_chunks), 1))
    tot = {'A': [], 'B': []}
    daily = {'A': [], 'B': []}
    for c in range(int(n_chunks)):
        for label, kw in (('A', kw_a), ('B', kw_b)):
            r = mc_engine(**kw, rng=np.random.RandomState(seed + c))
            tot[label].append(np.asarray(r['totals'], dtype=np.float64))
            daily[label].append(np.asarray(r['daily_means'],
                                           dtype=np.float64))
        if between_chunks is not None:
            between_chunks(c)
    exact_a = mc_expected_total(**kw_a)
    exact_b = mc_expected_total(**kw_b)
    arms = {label: _arm_summary(np.concatenate(tot[label]),
                                np.mean(daily[label], axis=0),
                                exact)
            for label, exact in (('A', exact_a), ('B', exact_b))}
    paired = summarize_pairs(arms['A']['totals'], arms['B']['totals'], level)
    exact_lift = exact_b - exact_a
    return {
        'A': arms['A'],
        'B': arms['B'],
        'paired': paired,
        'exact': {
            'A': float(exact_a),
            'B': float(exact_b),
            'lift': float(exact_lift),
            'lift_pct': (float(exact_lift / exact_a * 100.0)
                         if exact_a else float('nan')),
        },
        'direction': direction(paired['lift_ci']),
        'seed': int(seed),
        'n_chunks': int(n_chunks),
        'n_days': int(kw_a['n_days']),
        'n_iter': int(kw_a['n_iter']) * int(n_chunks),
    }


def comparison_lines(result: Mapping[str, Any], name_a: str = 'A',
                     name_b: str = 'B', money: str = '$') -> list:
    """The report lines every tool prints for a comparison, so the
    Optimize report and the What-If tab word the same result the same
    way."""
    p = result['paired']
    ex = result['exact']
    lvl = int(round(p['level'] * 100))
    lo, hi = p['lift_ci']
    plo, phi = p['lift_pct_ci']
    lines = [
        f"Paired lift ({name_b} - {name_a}), {result['n_days']}-day total:",
        f"  Monte Carlo:   {money}{p['lift_mean']:+,.2f}  "
        f"({p['lift_pct']:+.2f}%)",
        f"  {lvl}% interval:   {money}{lo:+,.2f} to {money}{hi:+,.2f}  "
        f"({plo:+.2f}% to {phi:+.2f}%)",
        f"  Exact (model): {money}{ex['lift']:+,.2f}  "
        f"({ex['lift_pct']:+.2f}%)",
        f"  {p['n_pairs']:,} pairs under common random numbers "
        f"(seed {result['seed']}; pair correlation "
        f"{p['pair_correlation']:.3f})",
    ]
    ahead = result['direction']
    if ahead == 'B':
        lines.append(f"  The model expects {name_b} to earn more than "
                     f"{name_a}.")
    elif ahead == 'A':
        lines.append(f"  The model expects {name_a} to earn more than "
                     f"{name_b}.")
    else:
        lines.append("  No difference distinguishable from Monte Carlo "
                     "noise at this iteration count.")
    lines.append("  The interval is Monte Carlo precision under the model;")
    lines.append("  it narrows with more iterations and does not cover")
    lines.append("  uncertainty in the model's own coefficients.")
    return lines
