"""The goodness-of-fit runner's cohort, list chain and year-shift split.

The runner analyses the window's arrivals followed until they leave (review
R46), records every visit's drawn list next to what it bought and tests the
category row along that chain (R29), splits the held-out year shift into a
fixed-assortment part and popularity turnover, and says on which side of
the replicas the simulator falls (R16). These tests pin the pieces on
hand-made records.
"""
import numpy as np
import pandas as pd
import pytest

from dataset_calibration import calibrate_transactional
from dataset_validation import cluster_permutation_chi2
from experiments import run_validation_gof as G


def _rec(spawn, drawn, bought=None, impulse=(), zones=('Dairy',)):
    r = {'spawn_t': spawn, 'paid': bought is not None, 'drawn': list(drawn),
         'impulse': list(impulse), 'zones': list(zones)}
    if bought is not None:
        r.update({'bought': list(bought), 'basket': float(len(bought)),
                  'revenue': 2.0 * len(bought)})
    return r


def test_cohort_analytics_keep_every_visit_and_split_the_paying_ones():
    cohort = [_rec(10, ['a', 'b'], ['a', 'b']),
              _rec(11, ['c'], None),                      # abandoned
              _rec(12, ['a', 'd'], ['a']),                # one item missed
              _rec(13, ['b'], ['b', 'gum'], impulse=['gum'])]
    win = G._cohort_analytics(cohort)
    assert win['basket_sizes'] == [2.0, 1.0, 2.0]
    assert win['visit_purchases'] == [['a', 'b'], ['a'], ['b', 'gum']]
    assert win['drawn_lists'] == [['a', 'b'], ['c'], ['a', 'd'], ['b']]
    assert win['item_conversion_rates'] == {'a': {'purchases': 2},
                                            'b': {'purchases': 2}}
    assert win['impulse_item_sales'] == {'gum': 1}
    assert win['area_visits'] == {'Dairy': 4}
    # One record per visit for visits.jsonl; an unpaid visit bought None.
    assert [v['drawn'] for v in win['visits']] == win['drawn_lists']
    assert [v['bought'] for v in win['visits']] == [
        ['a', 'b'], None, ['a'], ['b', 'gum']]
    assert win['visits'][1]['paid'] is False
    assert set(win['visits'][0]) == set(G.VISIT_FIELDS) - {'design', 'rep',
                                                           'seed'}
    assert win['chain'] == {'n_visits': 4, 'n_paying': 3, 'n_abandoned': 1,
                            'n_paying_whole_list': 2,
                            'items_drawn_by_paying': 5,
                            'items_bought_from_list': 4}
    tot = G._chain_summary([win['chain'], win['chain']])
    assert tot['n_visits'] == 8
    assert tot['share_paying_whole_list'] == pytest.approx(2 / 3)
    assert tot['share_listed_items_bought'] == pytest.approx(4 / 5)


class _Sim:
    def __init__(self, profile, start_hour=9.0):
        self.hourly_profile = np.asarray(profile, dtype=float)
        self.sim_clock_start_hour = start_hour


def test_offered_rate_integrates_the_hourly_multiplier():
    prof = np.zeros(24)
    prof[9], prof[10] = 0.5, 1.5
    sim = _Sim(prof)
    assert G._mean_profile_multiplier(sim, 1020.0, 2880.0) == pytest.approx(0.5)
    # Half the window in each hour.
    assert G._mean_profile_multiplier(sim, 3000.0, 4200.0) == pytest.approx(1.0)


def test_turnover_is_the_part_the_fixed_assortment_does_not_show():
    shift = {'basket': {'statistic': 0.10}, 'revenue': {'statistic': 0.08},
             'category': {'share_tv_distance': 0.05, 'statistic': 500.0}}
    fixed = {'basket': {'statistic': 0.02}, 'revenue': {'statistic': 0.05},
             'category': {'share_tv_distance': 0.01, 'statistic': 90.0}}
    t = G._turnover(shift, fixed)
    assert t['basket']['popularity_turnover'] == pytest.approx(0.08)
    assert t['category']['measure'] == 'share_tv_distance'
    assert t['category']['popularity_turnover'] == pytest.approx(0.04)
    assert G._turnover(shift, {'basket': None})['basket'] is None


def test_replica_summary_says_which_side_of_the_median():
    above = G._replica_summary([1.0, 2.0, 3.0], [0.5] * 3, 2.5, 0.05, 10, 1)
    below = G._replica_summary([1.0, 2.0, 3.0], [0.5] * 3, 1.5, 0.05, 10, 1)
    assert above['simulator_vs_median'] == 'above'
    assert above['simulator_minus_median'] == pytest.approx(0.5)
    assert below['simulator_vs_median'] == 'below'


def test_replicas_record_the_direction_of_the_mean_error():
    rng = np.random.default_rng(4)
    ref = {'sizes': rng.poisson(3.0, 3000) + 1.0,
           'revenues': rng.gamma(4.0, 2.0, 3000)}
    pooled = [{'test': 'Basket size (items/visit)', 'statistic': 0.05,
               'n_simulated': 300}]
    sim = {'basket': np.full(300, 6.0)}          # far too large baskets
    out = G._replicas(ref, ref, np.zeros((0, 2)), np.zeros((0, 2)), pooled,
                      0.05, n_rep=30, seed=3, sim_samples=sim)
    b = out['basket']
    assert b['simulator_mean_minus_reference'] == pytest.approx(
        6.0 - ref['sizes'].mean())
    assert b['simulator_mean_percentile'] == 1.0
    lo, hi = (b['replica_mean_minus_reference'][k] for k in ('lo', 'hi'))
    assert lo <= 0.0 <= hi


def test_effect_sizes_of_the_category_test():
    same = np.array([[2, 1, 0], [0, 1, 1], [1, 0, 2]] * 30)
    r = cluster_permutation_chi2(same, same, n_perm=50)
    assert r['statistic'] == pytest.approx(0.0)
    assert r['cramers_v'] == pytest.approx(0.0)
    assert r['share_tv_distance'] == pytest.approx(0.0)
    shifted = np.array([[3, 0, 0]] * 90)
    r = cluster_permutation_chi2(shifted, same, n_perm=50)
    n = shifted.sum() + same.sum()
    assert r['cramers_v'] == pytest.approx(np.sqrt(r['statistic'] / n))
    # Shares (1, 0, 0) against (3/8, 2/8, 3/8): half of 5/8 + 2/8 + 3/8.
    assert r['share_tv_distance'] == pytest.approx(0.625)


def test_balanced_cramers_v_removes_the_split():
    """The same share difference against a large reference: V grows with
    the simulated side's size, the share distance does not, and the
    balanced V equals V at an even split and moves far less than V."""
    ref = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]] * 400)      # 1200
    sim = np.array([[2, 1, 0]] * 800)       # 2400 purchases each side
    small = cluster_permutation_chi2(sim[:100], ref, n_perm=20)
    even = cluster_permutation_chi2(sim, ref, n_perm=20)
    assert small['share_tv_distance'] == pytest.approx(
        even['share_tv_distance'])
    assert even['sim_share_of_items'] == pytest.approx(0.5)
    assert even['cramers_v_balanced'] == pytest.approx(even['cramers_v'])
    assert small['cramers_v'] < 0.6 * even['cramers_v']
    drift_v = even['cramers_v'] / small['cramers_v']
    drift_b = even['cramers_v_balanced'] / small['cramers_v_balanced']
    assert abs(drift_b - 1.0) < 0.25 * abs(drift_v - 1.0)


def _chain_params():
    """Invoices over two categories of two products each."""
    rng = np.random.default_rng(0)
    products = [('B0', 'Bakery'), ('B1', 'Bakery'), ('D0', 'Dairy'),
                ('D1', 'Dairy')]
    day0 = pd.Timestamp('2010-12-01 09:00')
    rows = []
    for k in range(300):
        ts = day0 + pd.Timedelta(days=k // 30,
                                 minutes=int(rng.integers(0, 480)))
        for j in rng.choice(4, size=int(rng.integers(1, 3)), replace=False):
            pid, cat = products[j]
            rows.append((f'I{k:05d}', pid, f'ITEM {pid}', 1, ts, 2.0, cat))
    df = pd.DataFrame(rows, columns=['invoice_id', 'product_id',
                                     'product_name', 'quantity', 'timestamp',
                                     'unit_price', 'category'])
    return calibrate_transactional(df, currency='GBP')


def _visit(drawn, bought):
    return {'spawn_t': 1.0, 'paid': bought is not None, 'drawn': drawn,
            'bought': bought, 'impulse': [], 'basket': None,
            'revenue': None}


def test_list_chain_tests_disjoint_samples():
    params = _chain_params()
    shop_items = {1: {'items': {
        k: {'product_id': k, 'category': c} for k, c in
        (('B0', 'Bakery'), ('B1', 'Bakery'), ('D0', 'Dairy'),
         ('D1', 'Dairy'))}}}
    paying = [_visit(['B0', 'D0'], ['B0', 'D0'])] * 40 + \
        [_visit(['B1'], ['B1'])] * 40
    unpaid = [_visit(['D1'], None)] * 12
    rows, completion = G._list_chain_tests(params, shop_items,
                                           paying + unpaid, 0.05)
    assert [r['test'] for r in rows] == list(G.LIST_CHAIN_TESTS)
    drawn, selection = rows
    assert drawn['n_simulated'] == 92
    # Selection: the unpaid visits' drawn lists against the paying ones'.
    assert selection['n_simulated'] == 12 and selection['n_observed'] == 80
    assert selection['share_tv_distance'] > 0.5
    assert selection['effect_size_primary'] == 'share_tv_distance'
    # Every paying visit bought its whole list: completion distance 0.
    assert completion['share_tv_distance_drawn_vs_bought'] == \
        pytest.approx(0.0)
    assert completion['n_paying'] == 80
    partial = [_visit(['B0', 'D0'], ['B0'])] * 40 + \
        [_visit(['B1'], ['B1'])] * 40
    _, completion = G._list_chain_tests(params, shop_items,
                                        partial + unpaid, 0.05)
    assert completion['share_tv_distance_drawn_vs_bought'] > 0.1
    # No unpaid visit: the selection test cannot run, and says so.
    rows, _ = G._list_chain_tests(params, shop_items, paying, 0.05)
    assert rows[1]['p_value'] is None and rows[1]['decision'] == 'N/A'
    assert rows[1]['note'].startswith('Not run:')


def test_replicas_carry_effect_sizes_at_the_simulators_size():
    ref_clusters = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]] * 100)
    store_clusters = ref_clusters.copy()
    pooled = [{'test': 'Category purchase shares', 'statistic': 3.0,
               'n_simulated': 60, 'share_tv_distance': 0.02,
               'cramers_v': 0.03, 'cramers_v_balanced': 0.05}]
    empty = {'sizes': np.zeros(0), 'revenues': np.zeros(0)}
    out = G._replicas(empty, empty, store_clusters, ref_clusters, pooled,
                      0.05, n_rep=12, seed=5)
    eff = out['category']['effect_sizes']
    assert set(eff) == set(G.EFFECT_SIZES)
    for k in G.EFFECT_SIZES:
        assert eff[k]['lo'] <= eff[k]['median'] <= eff[k]['hi']
        assert eff[k]['simulator'] == pooled[0][k]
        assert 0.0 <= eff[k]['simulator_percentile'] <= 1.0


# --- Anonymous invoices in the list law (review R27) -------------------------

def _invoices_with_customers(n_invoices=300, seed=0):
    """Invoices over three categories of four products, every fourth one
    anonymous (no customer id) and larger than the rest."""
    rng = np.random.default_rng(seed)
    products = [(f'{cat[:3].upper()}{i}', cat, 1.5 + i)
                for cat in ('Bakery', 'Dairy', 'Garden') for i in range(4)]
    day0 = pd.Timestamp('2010-12-01 09:00')
    rows = []
    for k in range(n_invoices):
        anon = k % 4 == 0
        ts = day0 + pd.Timedelta(days=k // 40,
                                 minutes=int(rng.integers(0, 480)))
        size = int(rng.integers(3, 7)) if anon else int(rng.integers(1, 3))
        for j in rng.choice(len(products), size=size, replace=False):
            pid, cat, price = products[j]
            rows.append((f'I{k:05d}', pid, f'ITEM {pid}',
                         int(rng.integers(1, 4)), ts, price, cat,
                         np.nan if anon else f'C{k % 37:03d}'))
    return pd.DataFrame(rows, columns=['invoice_id', 'product_id',
                                       'product_name', 'quantity',
                                       'timestamp', 'unit_price', 'category',
                                       'customer_id'])


def test_design_block_records_the_anonymous_list_shares():
    """The runner reads the anonymous invoices' share of the store's lists
    (and of the items on them) from the store's own seeded calibration
    into every replication, and writes it into the design block beside the
    same shares among the reference invoices on the store's products."""
    from dataset_calibration import anonymous_placed_shares
    from dataset_validation import placed_product_ids
    params = calibrate_transactional(_invoices_with_customers(),
                                     currency='GBP')
    assert params.customer_id_available and params.n_anonymous_invoices > 0
    win = G.run_once(params, spawn=0.5, cap=30, seconds=30.0, warmup=10.0,
                     seed=3)
    shop = G.LS.build_live_store(params)
    cal = shop.customer_simulation.analytics['calibration']
    assert set(win['anonymous_lists']) == set(G.ANONYMOUS_LIST_KEYS)
    for k in G.ANONYMOUS_LIST_KEYS:
        assert win['anonymous_lists'][k] == pytest.approx(cal[k])
    # Anonymous invoices are a quarter of the lists and more of the items.
    share = win['anonymous_lists']['anonymous_list_share']
    assert 0.2 < share < 0.3
    assert win['anonymous_lists']['anonymous_list_item_share'] > share

    facts = G._store_facts('in_sample', win['shop_items'], params, params,
                           anonymous_lists=win['anonymous_lists'])
    for k in G.ANONYMOUS_LIST_KEYS:
        assert facts[k] == win['anonymous_lists'][k]
    ref = anonymous_placed_shares(
        params, placed_product_ids(G.ShopItems(win['shop_items'])))
    rec = facts['anonymous_lists_reference']
    assert rec['anonymous_list_share'] == pytest.approx(ref['invoice_share'])
    assert rec['anonymous_list_item_share'] == pytest.approx(
        ref['purchase_share'])
    # In sample the store's lists are the reference's invoices.
    assert rec['anonymous_list_share'] == pytest.approx(share)
    # A store whose calibration wrote no shares (a source without a
    # customer column) is recorded with nulls, not zeros.
    plain = G._store_facts('in_sample', win['shop_items'], params, params)
    assert all(plain[k] is None for k in G.ANONYMOUS_LIST_KEYS)
