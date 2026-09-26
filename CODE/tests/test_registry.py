"""Every behaviour-shaping constant on the results path lives in
``retail_literature`` (review R18), with its value unchanged, and the code
reads it from there.

The objective's clamps and queue channel, the Monte Carlo engine's own
guards, the stand-in inputs of the headless runners, both cold-start
traffic priors, the synthetic scenarios' analytical ground truth and the
live agent's constants used to sit inline in ``_common.py``,
``viz_ga_run.py``, ``sim_calibration.py``, ``experiments/closed_form.py``,
``synthetic_shops.py``, ``customer.py`` and ``simulation.py``. These tests pin the registry values,
check that each consumer binds the registry's name (and that changing the
binding changes the behaviour), and that the old literals are gone.
"""

import inspect
import math

import numpy as np
import pytest

import retail_literature as RL

# name -> the value the code used inline before it moved.
REGISTRY = {
    # objective
    'CONV_CLAMP_LO': 0.01, 'CONV_CLAMP_HI': 0.99,
    'IMPULSE_RATE_CLAMP_HI': 0.99, 'ABANDON_RATE_CLAMP_HI': 0.5,
    'ABANDON_BASE_CAP': 0.95, 'NET_BASE_REVENUE_MIN_FRAC': 0.05,
    'QUEUE_BOTTLENECK_FACTOR': 0.3, 'QUEUE_PENALTY_SLOPE': 0.15,
    'QUEUE_PENALTY_FLOOR': 0.85, 'DEFAULT_QUEUE_TIME_S': 5.0,
    # the Monte Carlo engine's guards (mc_engine and its closed form)
    'MC_LAMBDA_FLOOR': 0.1, 'MC_LAMBDA_CAP': 1.0e6,
    'MC_CONV_CLAMP_LO': 0.001, 'MC_CONV_CLAMP_HI': 0.999,
    'MC_SD_FLOOR': 0.01,
    # the synthetic scenarios' analytical ground truth
    'SYNTH_DETOUR_ELASTICITY_RANGE': (0.30, 0.50),
    'SYNTH_DETOUR_ELASTICITY_DEFAULT': 0.40,
    'SYNTH_IMPULSE_ELASTICITY_RANGE': (0.40, 0.70),
    'SYNTH_IMPULSE_ELASTICITY_DEFAULT': 0.50,
    'SYNTH_PERIMETER_ELASTICITY_RANGE': (0.15, 0.30),
    'SYNTH_PERIMETER_ELASTICITY_DEFAULT': 0.20,
    'SYNTH_AFFINITY_RANGE': (0.10, 0.30), 'SYNTH_AFFINITY_DENSITY': 0.15,
    'SYNTH_DAILY_CUSTOMERS': 200.0,
    # stand-ins
    'ASSUMED_CONVERSION_RATE': 0.30, 'IMPULSE_RATE_STANDIN': 0.20,
    'IMPULSE_VALUE_FRAC_OF_GROSS': 0.15, 'IMPULSE_VALUE_CV': 0.3,
    'REV_STD_FRAC_OF_MEAN': 0.35, 'ABANDON_RATE_SYNTHETIC': 0.05,
    'ABANDON_FRAC_OF_NONCONVERTERS': 0.10,
    'STANDIN_AVG_BASKET': 3.0, 'STANDIN_STD_BASKET': 1.0,
    # heat priors
    'HEAT_PRIOR_SYNTH_ENTRANCE': 5.0, 'HEAT_PRIOR_SYNTH_CHECKOUT': 3.0,
    'HEAT_PRIOR_CAL_ENTRANCE': 3.0, 'HEAT_PRIOR_CAL_CHECKOUT': 2.0,
    'HEAT_PRIOR_CAL_WALL': 1.5, 'HEAT_PRIOR_WALL_MIN_D': 0.1,
    # live agent
    'AGENT_SPEED_RANGE_MPS': (0.8, 1.5), 'AGENT_SPEED_CLIP_MPS': (0.3, 2.5),
    'AGENT_SPEED_FALLBACK_SD': 0.2, 'AGENT_ABANDON_PROB_RANGE': (0.01, 0.05),
    'AGENT_ENTERING_TIME_RANGE_S': (0.25, 0.5),
    'AGENT_MAX_IMPULSE_ITEMS_RANGE': (1, 4),
    'AGENT_PURCHASE_INTENT_RANGE': (0.7, 0.95),
    'CUSTOMER_TYPES': ('quick', 'browser', 'thorough'),
    'CUSTOMER_TYPE_PROBS': (0.3, 0.4, 0.3),
    'LOYALTY_LEVELS': ('new', 'regular', 'vip'),
    'LOYALTY_PROBS': (0.5, 0.35, 0.15),
    'WC_PROBABILITY_BY_TYPE': {'quick': 0.15, 'browser': 0.35,
                               'thorough': 0.50},
    'DWELL_ITEM_RANGE_S': (5, 15), 'DWELL_IMPULSE_RANGE_S': (2, 5),
    'DWELL_WC_RANGE_S': (2, 8), 'SEPARATION_STRENGTH': 0.08,
    'SEPARATION_MIN_DIST_M': 0.35, 'WAYPOINT_RADIUS_M': 0.5,
    # already in the registry, now read everywhere
    'DEFAULT_DAY_NOISE_STD': 0.08, 'DEFAULT_OP_HOURS_PER_DAY': 10.0,
}


@pytest.mark.parametrize('name', sorted(REGISTRY))
def test_registry_holds_the_former_inline_value(name):
    assert hasattr(RL, name), name
    assert getattr(RL, name) == REGISTRY[name]


def test_restroom_share_is_the_documented_one():
    share = sum(p * RL.WC_PROBABILITY_BY_TYPE[t]
                for t, p in zip(RL.CUSTOMER_TYPES, RL.CUSTOMER_TYPE_PROBS))
    assert share == pytest.approx(0.335)


# --- the consumers bind the registry names --------------------------------

@pytest.mark.parametrize('module, names', [
    ('experiments._common', ('ASSUMED_CONVERSION_RATE', 'IMPULSE_RATE_STANDIN',
                             'IMPULSE_VALUE_FRAC_OF_GROSS',
                             'REV_STD_FRAC_OF_MEAN', 'ABANDON_RATE_SYNTHETIC',
                             'ABANDON_FRAC_OF_NONCONVERTERS',
                             'DEFAULT_QUEUE_TIME_S', 'HEAT_PRIOR_SYNTH_ENTRANCE',
                             'HEAT_PRIOR_SYNTH_CHECKOUT',
                             'HEAT_PRIOR_CAL_ENTRANCE',
                             'HEAT_PRIOR_CAL_CHECKOUT', 'HEAT_PRIOR_CAL_WALL',
                             'HEAT_PRIOR_WALL_MIN_D')),
    ('layout_objective', ('CONV_CLAMP_LO', 'CONV_CLAMP_HI',
                          'IMPULSE_RATE_CLAMP_HI', 'ABANDON_RATE_CLAMP_HI',
                          'ABANDON_BASE_CAP', 'QUEUE_BOTTLENECK_FACTOR',
                          'QUEUE_PENALTY_SLOPE', 'QUEUE_PENALTY_FLOOR',
                          'DEFAULT_QUEUE_TIME_S', 'ABANDON_FLOW_COEF',
                          'ABANDON_SECTION_COEF', 'ABANDON_BOTTLENECK_COEF',
                          'ABANDON_FLOOR_FRAC')),
    ('customer', ('AGENT_SPEED_RANGE_MPS', 'AGENT_ABANDON_PROB_RANGE',
                  'AGENT_ENTERING_TIME_RANGE_S',
                  'AGENT_MAX_IMPULSE_ITEMS_RANGE',
                  'AGENT_PURCHASE_INTENT_RANGE', 'CUSTOMER_TYPES',
                  'CUSTOMER_TYPE_PROBS', 'LOYALTY_LEVELS', 'LOYALTY_PROBS',
                  'WC_PROBABILITY_BY_TYPE', 'DWELL_ITEM_RANGE_S',
                  'DWELL_IMPULSE_RANGE_S', 'DWELL_WC_RANGE_S',
                  'WAYPOINT_RADIUS_M')),
    ('simulation', ('AGENT_SPEED_CLIP_MPS', 'AGENT_SPEED_FALLBACK_SD',
                    'SEPARATION_STRENGTH', 'SEPARATION_MIN_DIST_M')),
    ('sim_calibration', ('ASSUMED_CONVERSION_RATE', 'IMPULSE_RATE_STANDIN',
                         'IMPULSE_VALUE_FRAC_OF_GROSS', 'IMPULSE_VALUE_CV',
                         'REV_STD_FRAC_OF_MEAN',
                         'ABANDON_FRAC_OF_NONCONVERTERS',
                         'NET_BASE_REVENUE_MIN_FRAC', 'DEFAULT_QUEUE_TIME_S',
                         'DEFAULT_DAY_NOISE_STD', 'MC_LAMBDA_FLOOR',
                         'MC_LAMBDA_CAP', 'MC_CONV_CLAMP_LO',
                         'MC_CONV_CLAMP_HI', 'MC_SD_FLOOR', 'MC_SPEND_LAW')),
    ('experiments.closed_form', ('MC_LAMBDA_FLOOR', 'MC_LAMBDA_CAP',
                                 'MC_CONV_CLAMP_LO', 'MC_CONV_CLAMP_HI',
                                 'MC_SD_FLOOR')),
    ('synthetic_shops', ('SYNTH_DETOUR_ELASTICITY_RANGE',
                         'SYNTH_DETOUR_ELASTICITY_DEFAULT',
                         'SYNTH_IMPULSE_ELASTICITY_RANGE',
                         'SYNTH_IMPULSE_ELASTICITY_DEFAULT',
                         'SYNTH_PERIMETER_ELASTICITY_RANGE',
                         'SYNTH_PERIMETER_ELASTICITY_DEFAULT',
                         'SYNTH_AFFINITY_RANGE', 'SYNTH_AFFINITY_DENSITY',
                         'SYNTH_DAILY_CUSTOMERS')),
])
def test_consumers_bind_the_registry_objects(module, names):
    """Each consumer imports the registry's name. (Compared by value:
    tests/test_elasticity.py reloads retail_literature, which rebinds its
    names to fresh objects; the monkeypatch tests below show the code reads
    these bindings, and the literal scan that nothing else is left.)"""
    import importlib
    mod = importlib.import_module(module)
    src = inspect.getsource(mod)
    for n in names:
        assert getattr(mod, n) == getattr(RL, n), f'{module}.{n}'
        assert n in src, f'{module} does not name {n}'


def test_mc_engine_defaults_come_from_the_registry():
    from sim_calibration import (DEFAULT_CALIBRATED_IMPULSE_RATE, mc_engine)
    sig = inspect.signature(mc_engine)
    assert sig.parameters['day_noise_std'].default == RL.DEFAULT_DAY_NOISE_STD
    assert DEFAULT_CALIBRATED_IMPULSE_RATE == RL.IMPULSE_RATE_STANDIN


def test_structural_sweep_default_is_the_shipped_strength():
    from experiments import run_structural_sensitivity as RS
    assert RS.DEFAULT == RL.SEPARATION_STRENGTH
    assert RS.DEFAULT in RS.STRENGTHS


def test_synthetic_base_params_read_the_stand_ins(monkeypatch):
    from experiments import _common
    from synthetic_shops import generate_synthetic_shop
    ss = generate_synthetic_shop(name='reg', seed=10_000, n_items=10)
    bp = _common.base_params_for(ss)
    mean_rev = float(np.mean([it.base_revenue for it in ss.items]))
    assert bp['conversion_rate'] == RL.ASSUMED_CONVERSION_RATE
    assert bp['impulse_rate'] == RL.IMPULSE_RATE_STANDIN
    assert bp['rev_std'] == max(mean_rev * RL.REV_STD_FRAC_OF_MEAN, 0.5)
    assert bp['abandonment_rate'] == RL.ABANDON_RATE_SYNTHETIC
    assert bp['avg_queue_time'] == RL.DEFAULT_QUEUE_TIME_S
    assert bp['customers_per_hour'] == \
        ss.daily_customers / RL.DEFAULT_OP_HOURS_PER_DAY
    # Moving the binding moves the output: the code reads the name.
    monkeypatch.setattr(_common, 'REV_STD_FRAC_OF_MEAN', 0.5)
    assert _common.base_params_for(ss)['rev_std'] == max(mean_rev * 0.5, 0.5)


def test_synthetic_heat_prior_reads_the_registry():
    from experiments._common import build_headless_shop
    from synthetic_shops import generate_synthetic_shop
    ss = generate_synthetic_shop(name='reg_heat', seed=10_001, n_items=8)
    shop = build_headless_shop(ss)
    res = shop.customer_simulation.heat_map_resolution
    heat = shop.customer_simulation.heat_raw
    for ix, iy in ((0, 0), (37, 61), (heat.shape[0] - 1, heat.shape[1] - 1)):
        x, y = ix / res, iy / res
        want = (RL.HEAT_PRIOR_SYNTH_ENTRANCE
                / (1.0 + math.hypot(x - ss.entrance[0], y - ss.entrance[1]))
                + RL.HEAT_PRIOR_SYNTH_CHECKOUT
                / (1.0 + math.hypot(x - ss.checkout[0], y - ss.checkout[1])))
        assert heat[ix, iy] == np.float32(want)


def test_live_agent_reads_the_registry(monkeypatch):
    import customer as C
    monkeypatch.setattr(C, 'AGENT_SPEED_RANGE_MPS', (1.23, 1.23))
    monkeypatch.setattr(C, 'WC_PROBABILITY_BY_TYPE',
                        {'quick': 1.0, 'browser': 1.0, 'thorough': 1.0})
    agent = C.Customer(0, (1.0, 1.0), {}, (10.0, 10.0))
    assert agent.speed == 1.23
    assert agent.wc_probability == 1.0 and agent.needs_wc
    assert agent.customer_type in RL.CUSTOMER_TYPES
    agent.target_position = [1.0 + 0.9 * RL.WAYPOINT_RADIUS_M, 1.0]
    assert agent._reached_target()
    agent.target_position = [1.0 + 1.1 * RL.WAYPOINT_RADIUS_M, 1.0]
    assert not agent._reached_target()


def test_old_inline_literals_are_gone():
    import customer
    import simulation
    import layout_objective
    import viz_ga_run
    from experiments import _common
    src_customer = inspect.getsource(customer)
    for lit in ('uniform(0.8, 1.5)', 'uniform(0.01, 0.05)',
                'uniform(0.25,0.5)', "p=[0.3, 0.4, 0.3]",
                "'thorough': 0.50}", 'uniform(5, 15)', 'uniform(2, 5)',
                'uniform(2, 8)', '< 0.25'):
        assert lit not in src_customer, lit
    src_sep = inspect.getsource(simulation.CustomerFlowSimulation._apply_separation)
    assert '0.08' not in src_sep and '0.35' not in src_sep
    src_common = inspect.getsource(_common)
    for lit in ("'impulse_rate': 0.20", 'mean_rev * 0.35', 'gross * 0.15',
                '5.0 / (1.0 + d_ent)', '1.5 / (1.0 + max',
                "'abandonment_rate': 0.05"):
        assert lit not in src_common, lit
    src_fit = inspect.getsource(viz_ga_run.GARunMixin._ga_fitness)
    for lit in ('0.85', '0.15', '* 0.3', '0.99', '0.01'):
        assert lit not in src_fit, lit
    src_obj = inspect.getsource(layout_objective.layout_drivers)
    for lit in ('0.85', '0.15', '* 0.3', '0.99', '0.5)'):
        assert lit not in src_obj, lit


def test_stale_registry_comments_are_fixed():
    src = inspect.getsource(RL)
    assert '_UNCERTAINTY' not in src
    assert 'common factor across methods' not in src
    assert 'set so a fully-compliant' not in src
    assert 'TUNED' in src and 'STAND-IN' in src


# --- the Monte Carlo guards and the synthetic ground truth -----------------

def test_engine_and_closed_form_share_the_registry_guards(monkeypatch):
    """The closed form must apply mc_engine's guards to converge to its
    mean. Both read them from the registry, and neither keeps a private
    copy: moving the conversion clamp moves both."""
    import experiments.closed_form as CF
    import sim_calibration as SC
    for private in ('_LAM_FLOOR', '_LAM_CAP', '_CONV_LO', '_CONV_HI',
                    '_SD_MIN'):
        assert not hasattr(CF, private), private
    src_engine = inspect.getsource(SC.mc_engine)
    for lit in ('min(max(conv, 0.001), 0.999)', 'max(rev_std, 0.01)',
                'max(impulse_value_std, 0.01)', '_LAM_CAP = ',
                '* gf, 0.1)'):
        assert lit not in src_engine, lit
    for fn in (SC._observed_conversion_rate, SC.extract_simulation_parameters):
        assert 'min(rate, 0.999)' not in inspect.getsource(fn)
        assert '0.999))' not in inspect.getsource(fn)
    kw = dict(cph=20.0, conv=1.5, rev_mean=40.0, rev_std=14.0, imp_rate=0.2,
              imp_val=3.0, avg_bsk=3.0, std_bsk=1.0, observed_baskets=None,
              n_days=7, op_hours=RL.DEFAULT_OP_HOURS_PER_DAY,
              wknd_mult=RL.DEFAULT_WEEKEND_MULTIPLIER, monthly_growth=0.0,
              day_noise_std=0.0)
    at_registry = CF.mc_expected_total(**kw, include_floors=False)
    monkeypatch.setattr(CF, 'MC_CONV_CLAMP_HI', 0.5)
    monkeypatch.setattr(SC, 'MC_CONV_CLAMP_HI', 0.5)
    lowered = CF.mc_expected_total(**kw, include_floors=False)
    assert lowered == pytest.approx(at_registry * 0.5 / RL.MC_CONV_CLAMP_HI)
    res = SC.mc_engine(**kw, n_iter=4000, rng=np.random.default_rng(1))
    assert res['mean'] == pytest.approx(lowered, rel=0.02)


def test_objective_stamp_records_the_new_registry_entries():
    from experiments._common import OBJECTIVE_CONSTANTS, elasticity_snapshot
    for name in ('MC_LAMBDA_FLOOR', 'MC_LAMBDA_CAP', 'MC_CONV_CLAMP_LO',
                 'MC_CONV_CLAMP_HI', 'MC_SD_FLOOR', 'MC_SPEND_LAW',
                 'SYNTH_DETOUR_ELASTICITY_RANGE',
                 'SYNTH_IMPULSE_ELASTICITY_RANGE',
                 'SYNTH_PERIMETER_ELASTICITY_RANGE', 'SYNTH_AFFINITY_RANGE',
                 'SYNTH_AFFINITY_DENSITY', 'SYNTH_DAILY_CUSTOMERS'):
        assert name in OBJECTIVE_CONSTANTS, name
        assert elasticity_snapshot()['objective_constants'][name] == \
            getattr(RL, name)


def test_synthetic_generator_reads_the_registry_bands(monkeypatch):
    """The analytical objective's coefficients come from the registry's
    bands (values unchanged, so every scenario is the same draw), and
    moving a band moves the generated scenario."""
    import synthetic_shops as SS
    src = inspect.getsource(SS)
    for lit in ('uniform(0.30, 0.50)', 'uniform(0.40, 0.70)',
                'uniform(0.15, 0.30)', 'uniform(0.10, 0.30)',
                'float = 0.40', 'float = 0.50', 'float = 0.20',
                'affinity_density: float = 0.15',
                'daily_customers: float = 200.0'):
        assert lit not in src, lit
    ss = SS.generate_synthetic_shop(name='reg_syn', seed=10_000, n_items=10)
    for it in ss.items:
        lo, hi = RL.SYNTH_DETOUR_ELASTICITY_RANGE
        assert lo <= it.detour_elasticity <= hi
        lo, hi = RL.SYNTH_IMPULSE_ELASTICITY_RANGE
        assert lo <= it.impulse_elasticity <= hi
        lo, hi = RL.SYNTH_PERIMETER_ELASTICITY_RANGE
        assert lo <= it.perimeter_elasticity <= hi
    lo, hi = RL.SYNTH_AFFINITY_RANGE
    assert all(lo <= a <= hi for a in ss.affinity.values())
    assert ss.daily_customers == RL.SYNTH_DAILY_CUSTOMERS
    item = SS.SyntheticItem('x', 'c', (1.0, 1.0), 1.0)
    assert (item.detour_elasticity, item.impulse_elasticity,
            item.perimeter_elasticity) == (
        RL.SYNTH_DETOUR_ELASTICITY_DEFAULT, RL.SYNTH_IMPULSE_ELASTICITY_DEFAULT,
        RL.SYNTH_PERIMETER_ELASTICITY_DEFAULT)
    monkeypatch.setattr(SS, 'SYNTH_PERIMETER_ELASTICITY_RANGE', (0.9, 0.9))
    moved = SS.generate_synthetic_shop(name='reg_syn', seed=10_000, n_items=10)
    assert all(it.perimeter_elasticity == 0.9 for it in moved.items)


def test_synthetic_bands_carry_an_epistemic_label():
    src = inspect.getsource(RL)
    block = src[src.index("the synthetic scenarios' analytical objective"):
                src.index('SYNTH_DAILY_CUSTOMERS')]
    assert block.count('OPERATIONAL ASSUMPTION') >= 4
    # The cited papers give directions only; no band is read from one.
    assert 'CITED' not in block.replace('cited', '')
