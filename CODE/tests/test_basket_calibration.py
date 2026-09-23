"""Per-invoice product sets, the in-store portion of an invoice, and the
list-length law it calibrates.

A dataset-built shop stocks only the top products of each category, so a
simulated shopper can only reproduce the part of a real invoice that the
shop carries. The calibration keeps each invoice's distinct product set
(CSR arrays), ``placed_invoice_sample`` cuts every invoice to a given
assortment, ``seed_into`` hands that sample to the agents as their list
length, and the basket-size / per-visit-revenue goodness-of-fit rows use
the same sample as their reference.
"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats

import customer as customer_mod
from customer import Customer
from dataset_calibration import (calibrate_omnichannel,
                                 calibrate_transactional,
                                 placed_invoice_sample)
from dataset_validation import validate_against_simulation
from retail_literature import LIST_LENGTH_BY_TYPE

PRICES = {'P1': 1.0, 'P2': 2.0, 'P3': 4.0, 'P4': 8.0}


def _invoices_df(invoices, prices):
    """One line per (invoice, product) entry, a minute apart per invoice,
    in the shape ``calibrate_transactional`` takes. A product listed twice
    on an invoice is a repeated line, as in Online Retail II."""
    rows = []
    t0 = pd.Timestamp('2010-12-01 10:00')
    for i, (inv, pids) in enumerate(invoices):
        for pid in pids:
            rows.append({'invoice_id': inv, 'product_id': pid,
                         'product_name': f'NAME {pid}', 'quantity': 3,
                         'timestamp': t0 + pd.Timedelta(minutes=i),
                         'unit_price': prices[pid], 'category': 'General'})
    return pd.DataFrame(rows)


# A: P1 twice (one product), P2.  B: P3.  C: P1, P2, P3.  D: P4, P2 (no
# stocked product under {P1, P3}).  E: P1 again -- P1 is on three invoices.
HAND_INVOICES = [('A', ['P1', 'P1', 'P2']),
                 ('B', ['P3']),
                 ('C', ['P1', 'P2', 'P3']),
                 ('D', ['P4', 'P2']),
                 ('E', ['P1'])]
HAND_SETS = {'A': {'P1', 'P2'}, 'B': {'P3'}, 'C': {'P1', 'P2', 'P3'},
             'D': {'P2', 'P4'}, 'E': {'P1'}}


@pytest.fixture
def hand_params():
    return calibrate_transactional(_invoices_df(HAND_INVOICES, PRICES))


def _stub_sim(analytics=None, shop=None):
    return SimpleNamespace(analytics={} if analytics is None else analytics,
                           run_time=0.0, sim_time=0.0, simulation_speed=1.0,
                           shop=shop, _basket_struct_cache=None)


def _shop(floor_items):
    """A shop stand-in: ``{floor: {item name: product_id}}``."""
    return SimpleNamespace(floors={
        fid: {'items': {name: {'product_id': pid, 'category': 'General'}
                        for name, pid in items.items()}}
        for fid, items in floor_items.items()})


def _invoice_sets(params):
    idx, ptr, items = (params.invoice_product_index, params.invoice_ptr,
                       params.invoice_items)
    return [{idx[j] for j in items[ptr[i]:ptr[i + 1]]}
            for i in range(len(ptr) - 1)]


# --- CSR structure ------------------------------------------------------------

def test_csr_structure_matches_the_invoices(hand_params):
    p = hand_params
    assert p.invoice_ptr.dtype == np.int64
    assert p.invoice_items.dtype == np.int32
    assert len(p.invoice_ptr) == p.n_invoices + 1 == 5 + 1
    assert p.invoice_ptr[0] == 0 and p.invoice_ptr[-1] == len(p.invoice_items)
    # Same invoice order as the other per-invoice samples (A..E), and a
    # repeated line counted once.
    assert _invoice_sets(p) == [HAND_SETS[k] for k in 'ABCDE']
    assert np.diff(p.invoice_ptr).tolist() == p.basket_distinct_sizes.tolist()
    assert sorted(p.invoice_product_index) == ['P1', 'P2', 'P3', 'P4']
    assert p.has_invoice_structure


def test_aggregate_source_has_no_invoice_structure():
    fams = pd.DataFrame({
        'aisle_id': np.arange(1, 9),
        'product_family': [f'Family {i}' for i in range(8)],
        'zone_id': [i % 3 for i in range(8)],
        'avg_price': 3.0, 'purchase_pct_instore': 0.3,
        'daily_demand_instore': 10.0, 'dwell_s': 30.0, 'impulse_rate': 0.1,
    })
    params = calibrate_omnichannel(fams, None)
    assert not params.has_invoice_structure
    assert params.invoice_product_index == []
    assert params.invoice_ptr.size == 0 and params.invoice_items.size == 0
    out = placed_invoice_sample(params, [str(k) for k in params.item_prices])
    assert out['sizes'].size == 0 and out['revenues'].size == 0
    assert out['n_invoices'] == 0 and out['n_with_placed'] == 0


# --- placed_invoice_sample ----------------------------------------------------

def test_placed_sample_on_a_hand_computed_example(hand_params):
    # Stocked: P1, P3 (and an id the dataset never sold).
    out = placed_invoice_sample(hand_params, iter(['P1', 'P3', 'NOT_SOLD']))
    # A -> {P1}: 1 item, 1.0; B -> {P3}: 1, 4.0; C -> {P1, P3}: 2, 5.0;
    # D holds neither and is left out; E -> {P1}: 1, 1.0. One unit of each
    # stocked product, at the calibrated price -- not the 3 units a line
    # carries.
    assert out['sizes'].tolist() == [1, 1, 2, 1]
    assert np.allclose(out['revenues'], [1.0, 4.0, 5.0, 1.0])
    assert out['n_invoices'] == 5
    assert out['n_with_placed'] == 4
    assert out['sizes'].dtype.kind == 'i'


def test_placed_sample_with_the_whole_catalogue_is_the_distinct_sample(
        hand_params):
    out = placed_invoice_sample(hand_params, PRICES.keys())
    assert out['sizes'].tolist() == hand_params.basket_distinct_sizes.tolist()
    assert np.allclose(out['revenues'], hand_params.invoice_distinct_revenues)


def test_csr_order_survives_shuffled_rows():
    """Row order in the source does not matter: invoice i of the CSR arrays
    is invoice i of the other per-invoice samples."""
    invoices, prices = _larger_example()
    df = _invoices_df(invoices, prices).sample(frac=1.0, random_state=3)
    params = calibrate_transactional(df)
    by_id = dict(invoices)
    assert _invoice_sets(params) == [set(by_id[k]) for k in sorted(by_id)]
    out = placed_invoice_sample(params, prices.keys())
    assert out['sizes'].tolist() == params.basket_distinct_sizes.tolist()
    assert np.allclose(out['revenues'], params.invoice_distinct_revenues)


def test_placed_sample_prices_non_string_product_ids():
    """A source whose product ids are not strings still prices the placed
    products: the ids are matched, and priced, by their string form."""
    df = _invoices_df(HAND_INVOICES, PRICES)
    df['product_id'] = df['product_id'].str[1:].astype(int)
    params = calibrate_transactional(df)
    out = placed_invoice_sample(params, ['1', '3'])
    assert out['sizes'].tolist() == [1, 1, 2, 1]
    assert np.allclose(out['revenues'], [1.0, 4.0, 5.0, 1.0])


def test_placed_sample_with_nothing_stocked_is_empty(hand_params):
    for ids in ([], None, ['NOT_SOLD']):
        out = placed_invoice_sample(hand_params, ids)
        assert out['sizes'].size == 0 and out['revenues'].size == 0
        assert out['n_invoices'] == 5 and out['n_with_placed'] == 0


# --- seed_into ----------------------------------------------------------------

def test_seed_into_writes_list_length_only_with_a_shop(hand_params):
    sim = _stub_sim()
    hand_params.seed_into(sim)
    assert 'list_length_sample' not in sim.analytics['calibration']

    # P3 is stocked on the upper floor only; it still counts.
    shop = _shop({1: {'Mug': 'P1', 'Sign': None}, 2: {'Lamp': 'P3'}})
    hand_params.seed_into(sim, shop=shop)
    cal = sim.analytics['calibration']
    assert cal['list_length_sample'] == [1, 1, 2, 1]
    assert all(type(v) is int for v in cal['list_length_sample'])
    assert cal['list_length_source'] == 'placed-invoice empirical'
    assert cal['n_invoices_with_placed'] == 4
    for key in ('list_length_sample', 'list_length_source',
                'n_invoices_with_placed'):
        assert key in sim._calibration_seeded_keys


def test_reseeding_pops_the_list_length_sample(hand_params):
    sim = _stub_sim()
    shop = _shop({1: {'Mug': 'P1', 'Lamp': 'P3'}})
    hand_params.seed_into(sim, shop=shop)
    assert sim.analytics['calibration']['list_length_sample']

    hand_params.seed_into(sim)                     # no shop: no sample
    cal = sim.analytics['calibration']
    for key in ('list_length_sample', 'list_length_source',
                'n_invoices_with_placed'):
        assert key not in cal

    # A source without invoices, onto a shop: nothing to cut, nothing left.
    hand_params.seed_into(sim, shop=shop)
    no_invoices = replace(hand_params, invoice_product_index=[],
                          invoice_ptr=np.zeros(0, dtype=np.int64),
                          invoice_items=np.zeros(0, dtype=np.int32))
    no_invoices.seed_into(sim, shop=shop)
    assert 'list_length_sample' not in sim.analytics['calibration']


def test_seed_into_rekeys_as_before_with_the_shop(hand_params):
    """Collecting product ids over every floor leaves the floor-1 re-keying
    of the per-item dicts as it was."""
    sim = _stub_sim()
    shop = _shop({1: {'Mug': 'P1', 'Lamp': 'P3'}, 2: {'Vase': 'P2'}})
    hand_params.seed_into(sim, shop=shop)
    cal = sim.analytics['calibration']
    assert set(cal['popular_items']) == {'Mug', 'Lamp'}
    assert cal['popular_items']['Mug'] == 3        # P1 is on A, C, E


# --- the agent's list length --------------------------------------------------

def _regular_items(n):
    items = {f'Item{i}': {'category': 'General', 'price': 1.0 + i}
             for i in range(n)}
    items['Gum'] = {'category': 'Impulse', 'price': 0.5}
    items['Checkout'] = {'category': 'Checkout', 'price': 0.0}
    return items


def _spawn(sim, items, k):
    return Customer(k, [1.0, 1.0], items, (10.0, 10.0),
                    door_position=(5.0, 0.0), door_side='bottom',
                    simulation_ref=sim)


def test_calibrated_list_length_is_drawn_from_the_sample():
    np.random.seed(7)
    sim = _stub_sim({'calibration': {'list_length_sample': [3] * 40}})
    items = _regular_items(6)
    agents = [_spawn(sim, items, k) for k in range(60)]
    # Every type, including 'thorough' (6-12 under the type law), gets 3.
    assert {a.customer_type for a in agents} == set(LIST_LENGTH_BY_TYPE)
    assert all(len(a.shopping_list) == 3 for a in agents)
    assert all(len(set(a.shopping_list)) == 3 for a in agents)
    assert all(i not in ('Gum', 'Checkout')
               for a in agents for i in a.shopping_list)


def test_calibrated_list_length_is_clipped_to_the_assortment():
    np.random.seed(8)
    items = _regular_items(4)
    big = _stub_sim({'calibration': {'list_length_sample': [50]}})
    assert all(len(_spawn(big, items, k).shopping_list) == 4
               for k in range(10))
    zero = _stub_sim({'calibration': {'list_length_sample': [0]}})
    assert all(len(_spawn(zero, items, k).shopping_list) == 1
               for k in range(10))


def test_list_length_follows_the_seeded_placed_sample(hand_params):
    np.random.seed(9)
    sim = _stub_sim()
    hand_params.seed_into(sim, shop=_shop({1: {'Mug': 'P1', 'Lamp': 'P3'}}))
    items = _regular_items(8)
    lengths = {len(_spawn(sim, items, k).shopping_list) for k in range(80)}
    assert lengths == {1, 2}


def test_without_a_sample_the_type_law_applies():
    np.random.seed(10)
    items = _regular_items(20)
    for analytics in ({}, {'calibration': {'popular_items': {}}},
                      {'calibration': {'list_length_sample': []}},
                      # Live counters are never read for the list length.
                      {'list_length_sample': [3] * 40}):
        sim = _stub_sim(dict(analytics))
        for k in range(60):
            a = _spawn(sim, items, k)
            lo, hi = LIST_LENGTH_BY_TYPE[a.customer_type]
            assert lo <= len(a.shopping_list) <= hi
    # And it is clipped to the assortment as before.
    small = _regular_items(3)
    sim = _stub_sim()
    assert all(len(_spawn(sim, small, k).shopping_list) <= 3
               for k in range(30))


def test_list_length_costs_one_draw_either_way(monkeypatch):
    """The calibrated draw replaces the type law's randint one for one, so
    a calibration does not add or remove a call on the global stream."""
    calls = []
    real = np.random.randint

    def counting(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    items = _regular_items(8)
    counts = []
    for analytics in ({}, {'calibration': {'list_length_sample': [2, 3, 4]}}):
        np.random.seed(11)
        a = _spawn(_stub_sim(analytics), items, 0)
        monkeypatch.setattr(customer_mod.np.random, 'randint', counting)
        calls.clear()
        a._generate_shopping_list(items)
        monkeypatch.setattr(customer_mod.np.random, 'randint', real)
        counts.append(len(calls))
    assert counts == [1, 1]


# --- goodness-of-fit reference ------------------------------------------------

def _larger_example():
    prices = {f'P{k}': float(k + 1) for k in range(10)}
    invoices = []
    for i in range(80):
        pids = sorted({f'P{i % 10}', f'P{(3 * i) % 10}', f'P{(7 * i + 1) % 10}'})
        invoices.append((f'INV{i:03d}', pids))
    return invoices, prices


def test_validation_compares_against_the_placed_reference():
    invoices, prices = _larger_example()
    params = calibrate_transactional(_invoices_df(invoices, prices))
    stocked = {'P0', 'P1', 'P2', 'P3'}
    shop = _shop({1: {f'Item {p}': p for p in sorted(stocked)}})

    # Reference computed directly from the invoice lists.
    ref_sizes, ref_revs = [], []
    for _, pids in invoices:
        on_shelf = set(pids) & stocked
        if on_shelf:
            ref_sizes.append(len(on_shelf))
            ref_revs.append(sum(prices[p] for p in on_shelf))
    assert len(ref_sizes) < len(invoices)       # some invoices drop out

    sim_baskets = [1, 2, 2, 1, 3, 1, 2, 1, 1, 2, 1, 1]
    sim_revs = [1.0, 3.0, 5.0, 2.0, 6.0, 4.0, 3.0, 1.0, 2.0, 7.0, 3.0, 4.0]
    sim = _stub_sim({'basket_sizes': list(sim_baskets),
                     'customer_revenues': list(sim_revs)}, shop=shop)
    res = validate_against_simulation(params, sim)
    rows = {t.name: t for t in res.tests}

    basket = rows['Basket size (items/visit)']
    D, p = sp_stats.ks_2samp(ref_sizes, sim_baskets)
    assert basket.n_observed == len(ref_sizes)
    assert np.isclose(basket.statistic, D) and np.isclose(basket.p_value, p)
    assert 'distinct stocked products per invoice' in basket.note
    # Not the whole-catalogue sample.
    D_full, _ = sp_stats.ks_2samp(params.basket_distinct_sizes, sim_baskets)
    assert not np.isclose(basket.statistic, D_full)

    revenue = rows['Per-visit revenue']
    D, p = sp_stats.ks_2samp(ref_revs, sim_revs)
    assert revenue.n_observed == len(ref_revs)
    assert np.isclose(revenue.statistic, D) and np.isclose(revenue.p_value, p)
    assert 'one unit of each stocked product' in revenue.note


def test_validation_without_invoice_structure_reports_not_run():
    invoices, prices = _larger_example()
    params = calibrate_transactional(_invoices_df(invoices, prices))
    params = replace(params, invoice_product_index=[],
                     invoice_ptr=np.zeros(0, dtype=np.int64),
                     invoice_items=np.zeros(0, dtype=np.int32))
    shop = _shop({1: {'Item P0': 'P0', 'Item P1': 'P1'}})
    sim = _stub_sim({'basket_sizes': [1, 2, 1, 2, 1, 1],
                     'customer_revenues': [1.0, 3.0, 1.0, 2.0, 1.0, 2.0]},
                    shop=shop)
    res = validate_against_simulation(params, sim)
    rows = {t.name: t for t in res.tests}
    for name in ('Basket size (items/visit)', 'Per-visit revenue'):
        assert not rows[name].applicable()
        assert rows[name].verdict() == 'N/A'
        assert 'per-invoice product sets' in rows[name].note
