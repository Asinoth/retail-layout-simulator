"""Interval estimates beside Figure B's scenario cluster bootstrap.

The runner reports the manuscript's interval (reproduced from the macro
generator, number for number), the crossed and two-way intervals that allow
for the search seeds shared across scenarios, the G - 1 t interval, and the
smallest margin at which the GA-SA equivalence would hold. These tests pin
the estimators on arrays whose answers are known.
"""

import numpy as np
import pytest
from scipy import stats as sps

import make_results_macros as MRM
from experiments import _inference as I
from experiments.run_baseline_comparison import (COMPARISON_FAMILY,
                                                 conclusions_unchanged)


def _grid(g=12, h=8, seed=0, seed_effect=0.0, mean=1.0):
    rng = np.random.default_rng(seed)
    s_eff = rng.normal(0, 1.0, g)
    j_eff = rng.normal(0, seed_effect, h)
    return {(s, j): mean + s_eff[s] + j_eff[j] + rng.normal(0, 0.3)
            for s in range(g) for j in range(h)}


def test_cluster_bootstrap_is_the_macro_generators():
    rng = np.random.default_rng(1)
    clustered = [rng.normal(5, 2, 10) for _ in range(30)]
    for alpha in (0.05, 0.05 / 7, 2 * 0.05 / 7):
        assert I.cluster_boot_ci(clustered, alpha=alpha) \
            == MRM.cluster_boot_ci(clustered, alpha=alpha)
    assert I.EQUIV_MARGIN_FRAC == MRM.EQUIV_MARGIN_FRAC
    assert I.N_BOOT == MRM.N_BOOT


def test_crossed_interval_widens_with_a_shared_seed_effect():
    """With a seed effect shared across scenarios, the scenario-only
    bootstrap is too narrow; the crossed bootstrap and the two-way t widen
    to cover it. Without one, all three agree roughly."""
    def widths(table):
        clusters = {}
        for (s, _), v in table.items():
            clusters.setdefault(s, []).append(v)
        _, lo, hi = I.cluster_boot_ci(list(clusters.values()), n=2000)
        _, xlo, xhi = I.crossed_boot_ci(table, n=2000)
        tw = I.twoway_cluster_t(table)
        return hi - lo, xhi - xlo, tw['hi'] - tw['lo']

    c0, x0, t0 = widths(_grid(seed_effect=0.0))
    c1, x1, t1 = widths(_grid(seed_effect=1.5))
    assert x1 > 1.5 * c1 and t1 > 1.5 * c1
    assert 0.6 < x0 / c0 < 1.8


def test_twoway_variance_is_the_cameron_gelbach_miller_sum():
    table = _grid(g=5, h=4, seed_effect=0.5)
    arr = np.array([[table[(s, j)] for j in range(4)] for s in range(5)])
    e = arr - arr.mean()
    n = arr.size
    v_s = (e.sum(1) ** 2).sum() / n ** 2 * 5 / 4
    v_j = (e.sum(0) ** 2).sum() / n ** 2 * 4 / 3
    v_c = (e ** 2).sum() / n ** 2 * n / (n - 1)
    tw = I.twoway_cluster_t(table, alpha=0.05)
    assert tw['var_scenario'] == pytest.approx(v_s)
    assert tw['var_seed'] == pytest.approx(v_j)
    assert tw['var_cell'] == pytest.approx(v_c)
    half = sps.t.ppf(0.975, 3) * np.sqrt(v_s + v_j - v_c)
    assert tw['hi'] - tw['mean'] == pytest.approx(half)
    with pytest.raises(ValueError):          # the crossed estimators need
        I.crossed_boot_ci({(0, 0): 1.0, (0, 1): 2.0, (1, 0): 3.0})


def test_g1_t_interval_and_smallest_margin():
    clustered = [[1.0, 3.0], [2.0, 4.0], [6.0, 2.0], [5.0, 5.0]]
    means = np.array([2.0, 3.0, 4.0, 5.0])
    g1 = I.g1_t_interval(clustered, alpha=0.1)
    half = sps.t.ppf(0.95, 3) * means.std(ddof=1) / 2
    assert g1['mean'] == pytest.approx(3.5)
    assert (g1['lo'], g1['hi']) == pytest.approx((3.5 - half, 3.5 + half))
    assert g1['df'] == 3
    assert I.smallest_equivalence_margin(-2.0, 1.5) == 2.0
    assert I.smallest_equivalence_margin(0.5, 3.0) == 3.0


def test_family_inference_and_the_closed_form_check():
    rng = np.random.default_rng(4)
    by_run = {}
    for s in range(10):
        for j in range(4):
            base = 1000.0 + 50 * s
            by_run[(s, j)] = {'GA': base + 20 + rng.normal(0, 2),
                              'simulated_annealing': base + rng.normal(0, 0.3),
                              'random_search': base - 30 + rng.normal(0, 5)}
    fam = [m for m in COMPARISON_FAMILY
           if m in ('simulated_annealing', 'random_search')]
    inf = I.paired_family_inference(by_run, fam)
    assert inf['n_comparisons'] == 2
    assert inf['alpha_per_comparison'] == pytest.approx(0.025)
    rs = inf['comparisons']['random_search']
    assert rs['mean'] > 0 and rs['cluster_bootstrap']['excludes_zero']
    assert rs['win_scenarios'] == 10
    eq = inf['equivalence']['schemes']['cluster_bootstrap']
    # GA - SA is about +20 on a base near 1225: outside a 0.1% margin
    # (~1.2), and the smallest margin that would hold covers the interval.
    assert not eq['equivalent_at_margin']
    assert eq['smallest_margin'] >= max(abs(eq['lo']), abs(eq['hi'])) - 1e-12
    same = conclusions_unchanged(inf, inf)
    assert same['all_unchanged']
    flipped = I.paired_family_inference(
        {k: {m: (-v if m == 'GA' else v) for m, v in d.items()}
         for k, d in by_run.items()}, fam)
    assert not conclusions_unchanged(inf, flipped)['all_unchanged']
