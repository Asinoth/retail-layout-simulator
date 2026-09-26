"""The operator sweeps: one budget per setting, noise-referenced spread.

``experiments.run_ga_sensitivity`` compares GA settings that all spend the
default setting's evaluations, judges their spread against what the noise
inside a setting alone would give, and sweeps the annealer's step, final
temperature and initial acceptance against the GA's default. These tests
pin the budget rule, the null range, the ANOVA, the annealer grid and that
the sweep does not depend on the worker count.

Every setting of a (scenario, seed) shares its search seed and held-out
block, so the null and the ANOVA must take the noise from the residual
after those seed blocks are removed; the tests below build data with
strong shared seed effects and check that they do.
"""
import math

import numpy as np
import pytest

from experiments import run_ga_sensitivity as S
from experiments._common import GA_N_FINAL_SEEDS


def test_every_ga_setting_spends_the_default_budget():
    budget = S.total_budget(30, 25)
    assert budget == 30 * (25 + GA_N_FINAL_SEEDS)
    settings = S.ga_settings(30, 25)
    assert len(settings) == len(S.MUT_RATES) * len(S.POP_SIZES)
    assert (0.18, 30, 25) in settings
    for _, p, g in settings:
        assert p * (g + GA_N_FINAL_SEEDS) == budget


def test_expected_range_of_normals():
    assert S.expected_range_of_normals(1) == 0.0
    assert S.expected_range_of_normals(2) == pytest.approx(2 / math.sqrt(math.pi))
    # tabulated d2 constants of the range of k normals
    assert S.expected_range_of_normals(9) == pytest.approx(2.970, abs=1e-3)
    rng = np.random.default_rng(0)
    sim = np.ptp(rng.normal(size=(20000, 5)), axis=1).mean()
    assert S.expected_range_of_normals(5) == pytest.approx(sim, rel=0.02)


def _crn_table(rng, n_scen, k, n, effect=0.0, block_sd=0.05,
               noise_sd=0.005, inter_sd=0.0):
    """(scenarios, settings, seeds) values with a seed effect shared by every
    setting of a (scenario, seed) -- common random numbers -- plus a setting
    effect, an optional scenario x setting interaction and small noise."""
    y = np.empty((n_scen, k, n))
    for s in range(n_scen):
        block = rng.normal(0, block_sd, n)
        inter = rng.normal(0, inter_sd, k) if inter_sd else np.zeros(k)
        for t in range(k):
            y[s, t] = (1.0 + 0.1 * s + effect * t + inter[t] + block
                       + rng.normal(0, noise_sd, n))
    return y


def test_blocked_residual_sd_removes_the_shared_seed_effect():
    """The blocked residual SD is the noise the setting means differ by;
    the SD within a setting also carries the shared seed effect."""
    rng = np.random.default_rng(1)
    y = _crn_table(rng, 1, 9, 200, block_sd=0.05, noise_sd=0.005)[0]
    assert S._blocked_residual_sd(y) == pytest.approx(0.005, rel=0.1)
    assert np.sqrt(np.mean(y.std(axis=1, ddof=1) ** 2)) > 0.04


def test_blocked_null_range_holds_its_level():
    """With no setting effect the observed range of the k setting means sits
    around the blocked null (ratio near 1), while the independent-draws null
    overstates it many times over when the seeds are shared."""
    rng = np.random.default_rng(2)
    ratios, ratios_ind, shares = [], [], []
    for _ in range(200):
        y = _crn_table(rng, 1, 9, 3)[0]
        results = [{'scenario': 0,
                    'ga': {f'{t}': list(y[t]) for t in range(9)}}]
        summ = S.ga_sweep_summary(results, '0')['per_scenario'][0]
        ratios.append(summ['spread_pct'] / summ['null_expected_range_pct'])
        ratios_ind.append(summ['spread_pct']
                          / summ['null_expected_range_pct_independent'])
        shares.append(summ['seed_block_share'])
    # The blocked SD has 16 degrees of freedom, so the mean ratio is near 1
    # (a little above: E[1/s] > 1/sigma); the independent null is far wider.
    assert 0.85 < np.mean(ratios) < 1.25
    assert np.mean(ratios_ind) < 0.3
    assert np.median(shares) > 0.8


def test_blocked_anova_separates_a_setting_effect_from_noise():
    rng = np.random.default_rng(3)
    null = S._blocked_anova(_crn_table(rng, 6, 4, 3))
    eff = S._blocked_anova(_crn_table(rng, 6, 4, 3, effect=0.01))
    assert null['setting_vs_residual']['p'] > 0.01
    assert null['setting_vs_interaction']['p'] > 0.01
    # A setting effect twice the residual noise but five times smaller than
    # the shared seed effect: an error term that kept the seed blocks would
    # bury it; the blocked test finds it.
    assert eff['setting_vs_residual']['p'] < 1e-6
    assert eff['setting_vs_interaction']['p'] < 1e-3
    assert eff['seed_blocks_vs_residual']['p'] < 1e-6
    assert eff['setting_vs_residual']['df'] == 3
    assert eff['setting_vs_residual']['df_error'] == 6 * 3 * 2
    assert eff['setting_vs_interaction']['df_error'] == 5 * 3


def test_random_scenario_test_does_not_count_an_interaction_as_an_effect():
    """Settings that rank differently by scenario with no effect on average:
    the interaction is detected, and the random-scenario test of the setting
    effect (against the interaction) does not call it a main effect."""
    rng = np.random.default_rng(4)
    y = _crn_table(rng, 8, 4, 3, inter_sd=0.02)
    an = S._blocked_anova(y)
    assert an['interaction']['p'] < 1e-6
    assert an['setting_vs_interaction']['p'] > 0.01


def test_annealer_grid():
    grid = S.sa_settings()
    assert len(grid) == len(S.SA_STEP_FRACS) * len(S.SA_FINAL_TEMP_FRACS) + 2
    assert tuple(map(float, S.SA_DEFAULT)) in {tuple(map(float, g))
                                               for g in grid}
    assert len(set(grid)) == len(grid)


def test_sweep_is_worker_count_invariant_and_summarizes():
    args = S.parse_args(['--n-scenarios', '2', '--n-seeds', '2',
                         '--n-gens', '2', '--pop-size', '4',
                         '--mc-iters', '10', '--mc-days', '3',
                         '--n-heldout-seeds', '2'])
    serial = S._map(S.run_scenario, 2, 1, args)
    pooled = S._map(S.run_scenario, 2, 2, args)
    assert serial == pooled
    default_key = f"{S.DEFAULT[0]}|{args.pop_size}|{args.n_gens}"
    ga = S.ga_sweep_summary(serial, default_key)
    for p in ga['per_scenario']:
        assert len(p['mean']) == len(p['sd_within']) == len(S.ga_settings(4, 2))
        assert p['null_expected_range_pct'] >= 0
    assert ga['null_design'] == S.GA_NULL_DESIGN
    assert set(ga['anova_setting']) >= {'setting_vs_residual',
                                        'setting_vs_interaction',
                                        'interaction'}
    sa = S.sa_sweep_summary(serial, default_key)
    assert sa['n_settings'] == len(S.sa_settings())
    assert sa['alpha_per_comparison'] == pytest.approx(0.05 / 7)
    assert sa['n_pairs'] == 4 and sa['figb_design_pairs'] == 300
    default = [k for k, v in sa['settings'].items() if v['is_default']]
    assert len(default) == 1
    assert sa['power_reference']['setting'] == default[0]
    assert sa['vs_default_sa_alpha'] == pytest.approx(
        0.05 / (len(S.sa_settings()) - 1))
    for v in sa['settings'].values():
        assert v['n_runs'] == 4
        assert v['equivalence']['smallest_margin'] >= 0
        # the annealer's own sensitivity, against the default annealer
        assert ('vs_default_sa' in v) == (not v['is_default'])
    # every setting's runs spent the same number of evaluations
    evals = {v for r in serial for k, v in r['evals'].items()}
    assert len(evals) == 1
