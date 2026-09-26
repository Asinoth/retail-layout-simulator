"""Figure C's searches at several budgets, and the closed-form re-scoring.

``experiments.run_real_data_budget`` runs the GA, random search and both
annealers at multiples of Figure C's budget and measures each one's gap to
the best layout any of them found. These tests pin, on a small store
calibrated from a synthetic invoice frame (as tests/test_real_data_seeds.py
builds one):

  * the runs do not depend on the worker count, every final layout keeps the
    floor-plan invariants, and the four searches spend one budget;
  * the GA and random search at the larger budget extend the same run: the
    shorter run's trace is a prefix of the longer one's;
  * the reference is the layout with the highest closed-form revenue, its
    own gap is zero, and the shares are taken of the GA's Figure C lift;
  * ``run_real_data_example.closed_form_scan``: the lift at the band
    midpoints, the band box and its fixed basket points, and the
    conversion scan with the rate at which the 0.99 clamps start to bind.
"""
import numpy as np
import pandas as pd
import pytest

from dataset_calibration import calibrate_transactional
from experiments import run_real_data_budget as RDB
from experiments.closed_form import expected_revenue
from experiments._common import layout_score
from experiments.run_real_data_example import (build_store, closed_form_scan,
                                               _at_conversion, _clamps_bind)
from experiments.run_real_data_seeds import store_fingerprint
from layout_objective import ELASTICITY_BANDS

TINY = dict(max_items_per_category=4, n_gens=2, pop_size=4, mc_iters=20,
            mc_days=3, sa_initial_accept=0.8, n_mc_replicates=3, sa_k_move=2)
PLAN = [(1, 3), (1, 4), (2, 3), (2, 4)]


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


@pytest.fixture(scope='module')
def params():
    return calibrate_transactional(_invoices(), currency='GBP')


@pytest.fixture(scope='module')
def store(params):
    return build_store(params, TINY['max_items_per_category'], verbose=False)


@pytest.fixture(scope='module')
def design(store):
    return RDB.RunDesign(**TINY, store_fingerprint=store_fingerprint(store))


@pytest.fixture(scope='module')
def runs(params, design):
    return {w: RDB.map_runs(params, PLAN, design, w) for w in (1, 2)}


def _strip(rs):
    return [{k: v for k, v in r.items() if k != 'wall_seconds'} for r in rs]


def test_runs_do_not_depend_on_the_worker_count(runs):
    assert _strip(runs[1]) == _strip(runs[2])
    RDB.check_runs(runs[2])
    assert [(r['multiplier'], r['search_seed']) for r in runs[2]] == PLAN


def test_every_run_spends_one_budget_and_keeps_the_invariants(runs):
    for r in runs[1]:
        want = TINY['n_gens'] * r['multiplier'] * TINY['pop_size']
        assert set(r['evaluation_counts'].values()) == {want}
        assert set(r['final_evaluation_counts'].values()) == {
            TINY['pop_size'] * 5}
        assert set(r['evaluation_counts']) == set(RDB.METHODS)
        for inv in r['invariants'].values():
            assert inv['n_clearance'] == inv['n_unshoppable'] == 0
            assert inv['n_overlap'] == inv['n_outside_zone'] == 0


def test_the_longer_run_extends_the_shorter_one(runs):
    """Same search seed at every budget: the GA's first generations and
    random search's first blocks are the same in both runs."""
    by = {(r['multiplier'], r['search_seed']): r for r in runs[1]}
    for seed in (3, 4):
        short, long_ = by[(1, seed)], by[(2, seed)]
        g = len(short['traces']['GA'])
        assert long_['traces']['GA'][:g] == short['traces']['GA']
        b = len(short['traces']['random_search'])
        assert long_['traces']['random_search'][:b] == \
            short['traces']['random_search']


def test_reference_gap_and_shares(runs):
    s = RDB.summarize(runs[1], ga_seed=3)
    ref = s['reference']
    best = max(r['closed_form'][k] for r in runs[1]
               for k in RDB.METHODS.values())
    assert ref['closed_form'] == best
    row = s['per_method'][ref['method']][str(ref['multiplier'])]
    assert min(row['gap_cf']['per_seed']) == pytest.approx(0.0, abs=1e-9)
    for rows in s['per_method'].values():
        for r in rows.values():
            assert min(r['gap_cf']['per_seed']) >= -1e-9
    fig = next(r for r in runs[1] if r['multiplier'] == 1
               and r['search_seed'] == 3)
    assert s['figc_lift']['closed_form'] == pytest.approx(
        fig['closed_form']['optimized'] - fig['closed_form']['baseline'])


def test_closed_form_scan(store, runs):
    lay = {k: {n: tuple(v) for n, v in l.items()}
           for k, l in runs[1][0]['layouts'].items()}
    scan = closed_form_scan(store, lay, TINY['mc_days'])
    bp = store.base_params
    # Values are the exact means, and the lift is taken of them.
    for k, l in lay.items():
        s, b = layout_score(store.shop, store.item_names, l)
        assert scan['values'][k] == pytest.approx(
            expected_revenue(bp, TINY['mc_days'], score=s, breakdown=b))
    base = scan['values']['baseline']
    assert scan['lift_pct']['optimized'] == pytest.approx(
        (scan['values']['optimized'] - base) / base * 100)
    # Band box: 2 x 2 corners times the basket band's two ends and the two
    # fixed basket points.
    bb = scan['band_box']
    assert len(bb['points']) == 2 * 2 * 4
    assert {p['bsk'] for p in bb['points']} == set(ELASTICITY_BANDS['bsk']) \
        | {0.02, 0.0}
    # The lift is linear in each elasticity below the clamps, so the
    # midpoint lift lies inside the corners' range.
    rng_all = bb['range_band_corners']['optimized']
    assert rng_all['min'] - 1e-9 <= scan['lift_pct']['optimized'] \
        <= rng_all['max'] + 1e-9
    # Conversion scan: the reference rate is on the grid and reproduces the
    # headline lift; the clamps do not bind below the onset and do at it.
    cs = scan['conversion_scan']
    p0 = cs['reference_rate']
    assert cs['lift_pct']['optimized'][cs['grid'].index(p0)] == \
        pytest.approx(scan['lift_pct']['optimized'])
    onset = cs['clamp_onset']
    assert onset is not None and 0 < onset < 1
    scored = {k: layout_score(store.shop, store.item_names, l)
              for k, l in lay.items()}
    assert not any(_clamps_bind(*scored[k], _at_conversion(bp, onset - 1e-3))
                   for k in lay)
    assert any(_clamps_bind(*scored[k], _at_conversion(bp, onset + 1e-6))
               for k in lay)
    for p, b in zip(cs['grid'], cs['clamps_bind']):
        assert b == (p >= onset)
    # Recalibrating to another rate keeps the observed buyers.
    b2 = _at_conversion(bp, 0.6)
    assert b2['customers_per_hour'] * 0.6 == pytest.approx(
        bp['customers_per_hour'] * bp['conversion_rate'])


def test_closed_form_conclusions(runs):
    """Every conclusion the budget runner reports is checked in closed form
    as in Monte Carlo: gap signs, the ranking of the methods, and the sign
    and significance over seeds of the GA's lift and of the GA minus each
    other search; ``all_unchanged`` is their conjunction."""
    c = RDB.closed_form_conclusions(runs[1])
    assert set(c['per_multiplier']) == {'1', '2'}
    flags = []
    for m, blk in c['per_multiplier'].items():
        assert set(blk['gap_sign']) == set(RDB.METHODS)
        # The reference is the closed-form best, so no closed-form gap is
        # negative.
        assert all(v['cf'] >= 0 for v in blk['gap_sign'].values())
        assert sorted(blk['ranking']['cf']) == sorted(RDB.METHODS)
        assert set(blk['ga_minus']) == {'lift_over_asbuilt', 'random_search',
                                        'simulated_annealing',
                                        'simulated_annealing_k'}
        sel = [r for r in runs[1] if r['multiplier'] == int(m)]
        d = blk['ga_minus']['random_search']
        assert d['n_seeds'] == len(sel)
        assert d['cf']['per_seed'] == pytest.approx(
            [r['closed_form']['optimized'] - r['closed_form']['rs']
             for r in sel])
        assert d['mc']['per_seed'] == pytest.approx(
            [np.mean(np.asarray(r['revenues']['optimized'])
                     - np.asarray(r['revenues']['rs'])) for r in sel])
        assert d['sign_unchanged'] == (np.sign(d['cf']['mean'])
                                       == np.sign(d['mc']['mean']))
        flags.append(blk['all_unchanged'])
    assert c['all_unchanged'] == all(flags)
