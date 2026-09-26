"""Interval estimates for paired differences over a scenario x seed design.

Figure B pairs every method with the GA on the same (scenario, seed) runs.
The scenarios are a sample of shops, and the search seeds are shared across
scenarios -- seed j is seed j in every scenario -- so the paired differences
are a crossed two-way array, not independent draws. The manuscript's
headline interval (``cluster_boot_ci``, reproduced here from
``make_results_macros`` operation for operation, so a runner can report it
beside the others) resamples scenarios only, which ignores the seed effect
and under-covers when there is one. This module adds the alternatives the
comparison's conclusions are checked under:

  * ``crossed_boot_ci`` -- the pigeonhole bootstrap for a crossed array
    (McCullagh 2000; Owen 2007): scenarios and seeds resampled
    independently, the mean taken over the resampled grid. Conservative
    for a two-way random-effects mean;
  * ``twoway_cluster_t`` -- the two-way cluster-robust variance of the mean
    (Cameron, Gelbach & Miller 2011): V_scenario + V_seed - V_cell, with a
    Student-t interval on min(G_scenario, G_seed) - 1 degrees of freedom;
  * ``g1_t_interval`` -- the Student-t interval on G - 1 degrees of
    freedom over the scenario means (Cameron & Miller 2015 on few clusters);
  * ``smallest_equivalence_margin`` -- the smallest margin at which two
    one-sided tests at a given level would declare equivalence: the larger
    absolute end of the (1 - 2 alpha) interval.

Every function is deterministic (fixed bootstrap seeds).
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
from scipy import stats as sps

#: Bootstrap resamples, as ``make_results_macros.N_BOOT``.
N_BOOT = 4000

#: GA-vs-SA equivalence margin as a fraction of the annealer's mean paired
#: revenue, as ``make_results_macros.EQUIV_MARGIN_FRAC``: fixed after the
#: pilot run and never retuned. Recorded here so the runner can report the
#: decision it implies beside the smallest margin that would have sufficed.
EQUIV_MARGIN_FRAC = 0.001


def cluster_boot_ci(clustered: Sequence[Sequence[float]], n: int = N_BOOT,
                    seed: int = 0, alpha: float = 0.05
                    ) -> Tuple[float, float, float]:
    """Scenario-level cluster bootstrap of the grand mean: resample
    scenarios with replacement, carry all their seed-level differences.
    ``make_results_macros.cluster_boot_ci``, operation for operation.
    Returns (mean, lo, hi) at two-sided ``alpha``."""
    scen = [np.asarray(c, dtype=float) for c in clustered if len(c)]
    rng = np.random.default_rng(seed)
    k = len(scen)
    grand = np.concatenate(scen).mean()
    boots = np.empty(n)
    for b in range(n):
        idx = rng.integers(0, k, k)
        boots[b] = np.concatenate([scen[i] for i in idx]).mean()
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return (float(grand), float(np.percentile(boots, lo)),
            float(np.percentile(boots, hi)))


def _grid(table: Mapping[Tuple[object, object], float]
          ) -> Tuple[np.ndarray, list, list]:
    """``table`` {(scenario, seed): value} as a complete scenario x seed
    array; raises when a cell is missing (the crossed estimators need the
    full grid)."""
    scen = sorted({s for s, _ in table}, key=lambda v: (str(type(v)), v))
    seeds = sorted({j for _, j in table}, key=lambda v: (str(type(v)), v))
    arr = np.empty((len(scen), len(seeds)))
    for a, s in enumerate(scen):
        for b, j in enumerate(seeds):
            if (s, j) not in table:
                raise ValueError(f"crossed design is missing cell {(s, j)}")
            arr[a, b] = table[(s, j)]
    return arr, scen, seeds


def crossed_boot_ci(table: Mapping[Tuple[object, object], float],
                    n: int = N_BOOT, seed: int = 0, alpha: float = 0.05
                    ) -> Tuple[float, float, float]:
    """Pigeonhole bootstrap of the grand mean of a complete scenario x
    seed array: each resample draws scenarios and seeds with replacement,
    independently, and averages the resampled grid. Returns (mean, lo, hi)
    at two-sided ``alpha``."""
    arr, _, _ = _grid(table)
    g, h = arr.shape
    rng = np.random.default_rng(seed)
    boots = np.empty(n)
    for b in range(n):
        ws = np.bincount(rng.integers(0, g, g), minlength=g)
        wj = np.bincount(rng.integers(0, h, h), minlength=h)
        boots[b] = ws @ arr @ wj / (g * h)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return (float(arr.mean()), float(np.percentile(boots, lo)),
            float(np.percentile(boots, hi)))


def twoway_cluster_t(table: Mapping[Tuple[object, object], float],
                     alpha: float = 0.05) -> Dict[str, float]:
    """Two-way cluster-robust interval for the grand mean of a complete
    scenario x seed array (Cameron, Gelbach & Miller 2011): the variance is
    the scenario-clustered plus the seed-clustered minus the cell
    (heteroskedasticity-robust) variance, each with its small-sample
    factor G / (G - 1); a negative sum falls back to the larger one-way
    variance. t quantile on min(G_scenario, G_seed) - 1 df."""
    arr, _, _ = _grid(table)
    g, h = arr.shape
    n = g * h
    mean = float(arr.mean())
    e = arr - mean

    def _v(sums: np.ndarray, groups: int) -> float:
        return float((sums ** 2).sum()) / n ** 2 * groups / max(groups - 1, 1)

    v_s = _v(e.sum(axis=1), g)
    v_j = _v(e.sum(axis=0), h)
    v_c = _v(e.ravel(), n)
    var = v_s + v_j - v_c
    if var <= 0:
        var = max(v_s, v_j)
    df = max(min(g, h) - 1, 1)
    half = float(sps.t.ppf(1 - alpha / 2, df)) * var ** 0.5
    return {'mean': mean, 'lo': mean - half, 'hi': mean + half,
            'se': var ** 0.5, 'df': int(df), 'var_scenario': v_s,
            'var_seed': v_j, 'var_cell': v_c}


def g1_t_interval(clustered: Sequence[Sequence[float]],
                  alpha: float = 0.05) -> Dict[str, float]:
    """Student-t interval with G - 1 degrees of freedom over the G
    scenario means (equal cluster sizes make their mean the grand mean)."""
    means = np.array([np.mean(c) for c in clustered if len(c)], dtype=float)
    g = means.size
    m = float(means.mean())
    if g < 2:
        return {'mean': m, 'lo': float('nan'), 'hi': float('nan'),
                'df': 0, 'se': float('nan')}
    se = float(means.std(ddof=1)) / g ** 0.5
    half = float(sps.t.ppf(1 - alpha / 2, g - 1)) * se
    return {'mean': m, 'lo': m - half, 'hi': m + half, 'df': int(g - 1),
            'se': se}


def smallest_equivalence_margin(lo: float, hi: float) -> float:
    """Smallest margin m for which (-m, m) holds the interval [lo, hi]: two
    one-sided tests at the interval's level declare equivalence for any
    margin above it."""
    return float(max(abs(lo), abs(hi)))


def excludes_zero(lo: float, hi: float) -> bool:
    return bool(lo > 0 or hi < 0)


def paired_family_inference(by_run: Mapping[Tuple[object, object],
                                             Mapping[str, float]],
                            family: Sequence[str], reference: str = 'GA',
                            equiv_with: str = 'simulated_annealing',
                            alpha_family: float = 0.05,
                            margin_frac: float = EQUIV_MARGIN_FRAC
                            ) -> Dict[str, object]:
    """Every interval above for ``reference`` minus each member of
    ``family`` present in ``by_run`` ({(scenario, seed): {method: value}}),
    at the Bonferroni level alpha' = ``alpha_family`` / (members present),
    and the equivalence of ``reference`` and ``equiv_with``.

    Per comparison: the scenario cluster bootstrap (the manuscript's
    scheme), the crossed bootstrap, the two-way cluster-robust t and the
    G - 1 t, each with whether it excludes zero; the number of runs and of
    scenarios the reference won. Equivalence: the (1 - 2 alpha') interval
    under each scheme, the decision at ``margin_frac`` of ``equiv_with``'s
    mean, and the smallest margin at which each would declare it (absolute
    and as a fraction of that mean)."""
    present = [m for m in family
               if any(reference in v and m in v for v in by_run.values())]
    alpha = alpha_family / max(len(present), 1)
    out: Dict[str, object] = {
        'alpha_family': alpha_family, 'n_comparisons': len(present),
        'alpha_per_comparison': alpha,
        'level_per_comparison': 1 - alpha,
        'equivalence_level': 1 - 2 * alpha,
        'schemes': {
            'cluster_bootstrap': 'scenarios resampled, all seeds carried '
                                 f'({N_BOOT} resamples; the manuscript scheme)',
            'crossed_bootstrap': 'scenarios and seeds resampled '
                                 f'independently ({N_BOOT} resamples)',
            'twoway_cluster_t': 'Cameron-Gelbach-Miller two-way variance, '
                                't on min(G) - 1 df',
            'g1_t': 't on G - 1 df over scenario means'},
        'comparisons': {},
    }
    for m in present:
        table = {k: v[reference] - v[m] for k, v in by_run.items()
                 if reference in v and m in v}
        clusters: Dict[object, list] = {}
        for (s, _j), d in table.items():
            clusters.setdefault(s, []).append(d)
        clustered = [clusters[s] for s in clusters]
        cb = cluster_boot_ci(clustered, alpha=alpha)
        rec: Dict[str, object] = {
            'n_runs': len(table), 'n_scenarios': len(clustered),
            'mean': cb[0],
            'cluster_bootstrap': {'lo': cb[1], 'hi': cb[2],
                                  'excludes_zero': excludes_zero(cb[1], cb[2])},
            'win_runs': int(sum(d > 0 for d in table.values())),
            'win_scenarios': int(sum(np.mean(c) > 0 for c in clustered)),
        }
        try:
            xb = crossed_boot_ci(table, alpha=alpha)
            tw = twoway_cluster_t(table, alpha=alpha)
            rec['crossed_bootstrap'] = {'lo': xb[1], 'hi': xb[2],
                                        'excludes_zero':
                                            excludes_zero(xb[1], xb[2])}
            rec['twoway_cluster_t'] = {**tw, 'excludes_zero':
                                       excludes_zero(tw['lo'], tw['hi'])}
        except ValueError as exc:            # incomplete grid
            rec['crossed_bootstrap'] = rec['twoway_cluster_t'] = {
                'error': str(exc)}
        g1 = g1_t_interval(clustered, alpha=alpha)
        rec['g1_t'] = {**g1, 'excludes_zero': excludes_zero(g1['lo'],
                                                              g1['hi'])}
        out['comparisons'][m] = rec

    if equiv_with in present:
        table = {k: v[reference] - v[equiv_with] for k, v in by_run.items()
                 if reference in v and equiv_with in v}
        base = float(np.mean([v[equiv_with] for v in by_run.values()
                              if reference in v and equiv_with in v]))
        clusters = {}
        for (s, _j), d in table.items():
            clusters.setdefault(s, []).append(d)
        clustered = [clusters[s] for s in clusters]
        a2 = 2 * alpha
        ivs = {'cluster_bootstrap': cluster_boot_ci(clustered, alpha=a2)[1:]}
        try:
            ivs['crossed_bootstrap'] = crossed_boot_ci(table, alpha=a2)[1:]
            tw = twoway_cluster_t(table, alpha=a2)
            ivs['twoway_cluster_t'] = (tw['lo'], tw['hi'])
        except ValueError:
            pass
        g1 = g1_t_interval(clustered, alpha=a2)
        ivs['g1_t'] = (g1['lo'], g1['hi'])
        margin = margin_frac * base
        eq = {'with': equiv_with, 'margin_frac': margin_frac,
              'margin': margin, 'reference_mean': base, 'schemes': {}}
        for name, (lo, hi) in ivs.items():
            m_star = smallest_equivalence_margin(lo, hi)
            eq['schemes'][name] = {
                'lo': float(lo), 'hi': float(hi),
                'equivalent_at_margin': bool(-margin < lo and hi < margin),
                'smallest_margin': m_star,
                'smallest_margin_frac': m_star / max(base, 1e-12)}
        out['equivalence'] = eq
    return out
