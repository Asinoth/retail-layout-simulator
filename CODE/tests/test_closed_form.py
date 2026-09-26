"""The exact mean of the Monte Carlo fitness (review R02, R27).

A layout reaches ``mc_engine`` only through deterministic drivers, so the
fitness has a closed-form mean, ``experiments.closed_form``. These tests
pin that the engine's mean over many iterations matches it within Monte
Carlo error -- on raw engine inputs (including a heavy-tailed, thin-day
case, where the former floored spend law was biased), through the
production ``_ga_fitness`` on a synthetic scenario, and on a small
calibrated store -- that the closed form carries no floor term, and that
the elasticity sweep uses this one function.
"""

import numpy as np
import pandas as pd
import pytest

import retail_literature as RL
from baselines import random_valid
from dataset_calibration import calibrate_transactional
from experiments._common import (anchor_base_params, base_params_for,
                                 base_params_for_calibration,
                                 build_headless_shop,
                                 build_headless_shop_from_calibration,
                                 feasible_layout, layout_score,
                                 layout_to_chromosome, zone_sampler)
from experiments.closed_form import (expected_revenue, legacy_floor_bias,
                                     mc_expected_total,
                                     spend_floor_correction)
from sim_calibration import mc_engine
from synthetic_shops import generate_synthetic_shop

Z_MAX = 4.0


def _engine_kw(**over):
    kw = dict(cph=20.0, conv=0.3, rev_mean=40.0, rev_std=14.0, imp_rate=0.2,
              imp_val=3.0, avg_bsk=3.0, std_bsk=1.0, observed_baskets=None,
              n_days=30, op_hours=RL.DEFAULT_OP_HOURS_PER_DAY,
              wknd_mult=RL.DEFAULT_WEEKEND_MULTIPLIER, monthly_growth=0.0)
    kw.update(over)
    return kw


@pytest.mark.parametrize('over', [
    {},                                                  # synthetic-like
    {'day_noise_std': 0.0},
    {'monthly_growth': 0.05, 'n_days': 45},
    {'cph': 0.4, 'rev_std': 120.0},                      # thin days, heavy tail
    {'cph': 0.05, 'conv': 0.5, 'rev_mean': 10.0, 'rev_std': 30.0,
     'imp_rate': 0.5},
    {'conv': 1.2, 'imp_rate': -0.1},                     # the engine's clamps
    {'conv': -0.3, 'cph': 0.0},             # lower conversion clamp, rate floor
    {'rev_std': 0.0, 'impulse_value_std': 0.0, 'cph': 0.4},   # SD floors
    # Online Retail II's invoice spread: CV 2.3 after the reversal pairs,
    # 3.4 before, on the calibrated store's thin days.
    {'cph': 1.6, 'rev_mean': 480.0, 'rev_std': 1100.0, 'imp_val': 75.0},
    {'cph': 0.5, 'rev_mean': 480.0, 'rev_std': 1630.0, 'imp_val': 75.0},
])
def test_engine_mean_matches_the_closed_form(over):
    kw = _engine_kw(**over)
    n = 60_000
    res = mc_engine(**kw, n_iter=n, rng=np.random.default_rng(5))
    exact = mc_expected_total(**kw)
    se = res['std'] / np.sqrt(n)
    assert abs(res['mean'] - exact) <= Z_MAX * se, (res['mean'], exact, se)


def test_the_closed_form_has_no_floor_term():
    """The engine's spend law keeps the mean exactly, so the closed form is
    the plain product, heavy tail and thin days included; the flag kept for
    old callers changes nothing."""
    for kw in (_engine_kw(), _engine_kw(cph=0.4, rev_std=120.0)):
        lam = sum(RL.DEFAULT_WEEKEND_MULTIPLIER if d % 7 in (5, 6) else 1.0
                  for d in range(kw['n_days'])) * kw['cph'] * kw['op_hours']
        plain = lam * kw['conv'] * (kw['rev_mean']
                                    + kw['imp_rate'] * kw['imp_val'])
        assert mc_expected_total(**kw) == pytest.approx(plain, rel=1e-12)
        assert mc_expected_total(**kw, include_floors=True) == \
            mc_expected_total(**kw)


def test_legacy_floor_bias_is_what_the_floored_law_added():
    """``legacy_floor_bias`` reports the former floored normal's bias: it
    vanishes for tight spend, is large on thin heavy-tailed days, and
    matches a direct simulation of the former law."""
    tight = _engine_kw()
    assert legacy_floor_bias(**tight) <= 1e-9 * mc_expected_total(**tight)
    thin = _engine_kw(cph=0.4, rev_std=120.0)
    assert legacy_floor_bias(**thin) > 0.2 * mc_expected_total(**thin)
    assert spend_floor_correction(5.0, 10.0, 30.0, 0.08) > 0.0
    assert spend_floor_correction(0.0, 10.0, 30.0, 0.08) == 0.0
    # One day at a fixed rate, no day noise, no impulse: the former law
    # drew max(N(n mu, sqrt(n) sd), 0) with n ~ Poisson(lam p).
    rng = np.random.default_rng(11)
    mu_n, mean, sd = 3.0, 40.0, 120.0
    n = rng.poisson(mu_n, 400_000).astype(float)
    x = np.maximum(n * mean + np.sqrt(n) * sd * rng.standard_normal(n.size),
                   0.0)
    bias = x.mean() - mu_n * mean
    se = x.std() / np.sqrt(n.size)
    assert abs(bias - spend_floor_correction(mu_n, mean, sd, 0.0)) \
        <= Z_MAX * se


def _fitness_mc(shop, names, lay, bp, days, iters, seed):
    real = shop._mc_engine
    seen = {}

    def spy(**kw):
        seen['res'] = real(**kw)
        return seen['res']
    shop._mc_engine = spy
    np.random.seed(seed)
    try:
        mean = shop._ga_fitness(layout_to_chromosome(lay, names), names, bp,
                                mc_days=days, mc_iters=iters)
    finally:
        del shop._mc_engine
    return mean, seen['res']['std'] / np.sqrt(iters)


def test_ga_fitness_converges_to_expected_revenue_on_a_synthetic_scenario():
    ss = generate_synthetic_shop(name='cf', seed=10_004, n_items=10)
    shop = build_headless_shop(ss)
    names = [it.name for it in ss.items]
    as_built = {n: tuple(shop.floors[1]['items'][n]['position'])
                for n in names}
    bp = anchor_base_params(shop, names, base_params_for(ss), as_built)
    layouts = [feasible_layout(shop, names, as_built),
               feasible_layout(shop, names, random_valid(ss, seed=2))]
    for lay in layouts:
        mean, se = _fitness_mc(shop, names, lay, bp, days=30, iters=20_000,
                               seed=17)
        exact = expected_revenue(bp, 30, shop=shop,
                                 chromosome=layout_to_chromosome(lay, names),
                                 item_names=names)
        assert abs(mean - exact) <= Z_MAX * se, (mean, exact, se)
    # The as-built layout's exact revenue is the calibrated one: visitors x
    # conversion x gross spend over the horizon.
    lam = bp['customers_per_hour'] * RL.DEFAULT_OP_HOURS_PER_DAY * sum(
        RL.DEFAULT_WEEKEND_MULTIPLIER if d % 7 in (5, 6) else 1.0
        for d in range(30))
    calibrated = lam * bp['conversion_rate'] * bp['rev_per_customer_gross']
    s0, b0 = layout_score(shop, names, layouts[0])
    assert expected_revenue(bp, 30, score=s0, breakdown=b0) == \
        pytest.approx(calibrated, rel=1e-9)


def _invoices(n_invoices=400, seed=0):
    rng = np.random.default_rng(seed)
    products = [(f'{cat[:3].upper()}{i}', cat, 1.5 + i)
                for cat in ('Bakery', 'Dairy', 'Garden', 'Kitchen')
                for i in range(4)]
    day0 = pd.Timestamp('2010-12-01 09:00')
    rows = []
    for k in range(n_invoices):
        ts = day0 + pd.Timedelta(days=k // 40,
                                 minutes=int(rng.integers(0, 480)))
        for j in rng.choice(len(products), size=int(rng.integers(1, 5)),
                            replace=False):
            pid, cat, price = products[j]
            rows.append((f'I{k:05d}', pid, f'ITEM {pid}',
                         int(rng.integers(1, 4)), ts, price, cat))
    return pd.DataFrame(rows, columns=['invoice_id', 'product_id',
                                       'product_name', 'quantity',
                                       'timestamp', 'unit_price', 'category'])


def test_ga_fitness_converges_to_expected_revenue_on_a_calibrated_store():
    params = calibrate_transactional(_invoices(), currency='GBP')
    shop = build_headless_shop_from_calibration(
        params, max_items_per_category=4, naive=True)
    names = list(shop.floors[1]['items'])
    as_built = {n: tuple(shop.floors[1]['items'][n]['position'])
                for n in names}
    bp = anchor_base_params(shop, names, base_params_for_calibration(params),
                            as_built)
    sample = zone_sampler(shop, names)
    rng = np.random.default_rng(3)
    layouts = [feasible_layout(shop, names, as_built),
               feasible_layout(shop, names, sample(rng))]
    for lay in layouts:
        mean, se = _fitness_mc(shop, names, lay, bp, days=28, iters=20_000,
                               seed=23)
        s, b = layout_score(shop, names, lay)
        exact = expected_revenue(bp, 28, score=s, breakdown=b)
        assert abs(mean - exact) <= Z_MAX * se, (mean, exact, se)


def test_score_and_chromosome_forms_agree():
    ss = generate_synthetic_shop(name='cf2', seed=10_001, n_items=8)
    shop = build_headless_shop(ss)
    names = [it.name for it in ss.items]
    as_built = {n: tuple(shop.floors[1]['items'][n]['position'])
                for n in names}
    bp = anchor_base_params(shop, names, base_params_for(ss), as_built)
    lay = feasible_layout(shop, names, random_valid(ss, seed=4))
    s, b = layout_score(shop, names, lay)
    by_score = expected_revenue(bp, 14, score=s, breakdown=b,
                                elasticities={'bsk': 0.02})
    by_chrom = expected_revenue(bp, 14, shop=shop,
                                chromosome=layout_to_chromosome(lay, names),
                                item_names=names, elasticities={'bsk': 0.02})
    assert by_score == by_chrom
    with pytest.raises(TypeError):
        expected_revenue(bp, 14)
    with pytest.raises(TypeError):
        expected_revenue(bp, 14, score=s)


def test_the_elasticity_sweep_has_no_private_revenue_formula():
    import experiments.run_elasticity_lhs as LHS
    assert not hasattr(LHS, 'projected_horizon_revenue')
    ss = generate_synthetic_shop(name='cf3', seed=10_002, n_items=8)
    shop = build_headless_shop(ss)
    names = [it.name for it in ss.items]
    as_built = {n: tuple(shop.floors[1]['items'][n]['position'])
                for n in names}
    bp = anchor_base_params(shop, names, base_params_for(ss), as_built)
    s, b = layout_score(shop, names,
                        feasible_layout(shop, names, random_valid(ss, seed=1)))
    e = {'conv': 0.3, 'imp': 0.6, 'bsk': 0.15}
    assert LHS.horizon_revenue(s, b, bp, e) == \
        expected_revenue(bp, LHS.MC_DAYS, score=s, breakdown=b,
                         elasticities=e)
