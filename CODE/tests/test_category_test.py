"""The category goodness-of-fit row: a chi-square on whole baskets.

Items bought on one trip are not independent draws, so the chi-square
distribution -- which counts every purchase as its own observation --
rejects a model that reproduces the data exactly far more often than its
level. ``cluster_permutation_chi2`` takes its p-value from reassigning
whole baskets between the simulated and the reference side instead. These
tests hold it to its size on a perfect model with clustered baskets, show
the item-level p over-rejecting on the same data, check its power against
a shifted category mix, and pin how the Validation tab and the headless
runner feed it: per-visit purchases recorded at exit, reference invoices
cut to the products placed in the shop, whole invoices on both sides.
"""
import json
from collections import defaultdict
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats

import dataset_validation as dv
from dataset_calibration import calibrate_transactional, placed_invoice_sample
from dataset_validation import (CATEGORY_ITEM_TEST_KIND, CATEGORY_N_PERM,
                                CATEGORY_PERM_SEED, CATEGORY_TEST_KIND,
                                CATEGORY_TEST_NAME, UNREFERENCED_CATEGORY,
                                cluster_permutation_chi2,
                                invoice_category_clusters,
                                merge_unreferenced_categories,
                                observed_category_counts,
                                placed_product_categories,
                                validate_against_simulation)
from sim_analytics import AnalyticsMixin


# --- A clustered basket distribution ---------------------------------------

BASE_MIX = np.array([0.30, 0.20, 0.15, 0.15, 0.10, 0.10])
# Share of a basket's items taken from the category the trip is for; the
# rest follow the store-wide mix. Several items per basket and this pull
# give the within-basket correlation of category that a real invoice
# shows -- shoppers buy several products of the department they came for:
# the item-level statistic comes out inflated by a factor of about 1.8,
# close to the design effect of the Online Retail II invoices (about 1.7).
MISSION_PULL = 0.45
ALPHA = 0.05
# Relabellings inside these tests. The level stays exact at any count --
# with 199, p <= 0.05 exactly when at most 9 relabelled statistics reach
# the observed one, a 10 in 200 chance under the null -- and the
# production count (CATEGORY_N_PERM) only narrows the Monte Carlo error.
TEST_N_PERM = 199


def _baskets(rng, n, mix=BASE_MIX):
    """``n`` baskets as per-category purchase counts: 1 + Poisson(3) items,
    each from the basket's own mission category with MISSION_PULL,
    otherwise from ``mix``; the mission itself is drawn from ``mix``."""
    k = len(mix)
    sizes = 1 + rng.poisson(3.0, n)
    mission = rng.choice(k, size=n, p=mix)
    row = np.repeat(np.arange(n), sizes)
    cat = np.where(rng.random(row.size) < MISSION_PULL, mission[row],
                   rng.choice(k, size=row.size, p=mix))
    return np.bincount(row * k + cat, minlength=n * k).reshape(n, k)


def _rejection_rates(n_reps, sim_mix=BASE_MIX, n_ref=2000, n_sim=400):
    perm, naive, inflation = [], [], []
    for rep in range(n_reps):
        rng = np.random.default_rng(10_000 + rep)
        ref = _baskets(rng, n_ref)
        sim = _baskets(rng, n_sim, mix=sim_mix)
        r = cluster_permutation_chi2(sim, ref, n_perm=TEST_N_PERM)
        perm.append(r['p_value'] <= ALPHA)
        naive.append(r['naive_p_value'] <= ALPHA)
        inflation.append(r['clustering_inflation'])
    return float(np.mean(perm)), float(np.mean(naive)), float(np.mean(inflation))


def test_permutation_test_holds_its_level_on_a_perfect_model():
    """Simulated visits drawn as random invoices from the reference's own
    basket distribution: the permutation test rejects at about alpha,
    while the item-level chi-square on the same statistic over-rejects."""
    perm, naive, inflation = _rejection_rates(200)
    assert 0.01 <= perm <= 0.09, perm
    assert naive >= 0.12, naive
    # The clustering is really there: relabelled statistics sit well above
    # the chi-square mean K - 1 an independent-items model would give.
    assert 1.5 < inflation < 2.1, inflation


def test_permutation_test_detects_a_shifted_category_mix():
    shifted = BASE_MIX + np.array([-0.08, 0.0, 0.0, 0.0, 0.0, 0.08])
    perm, _, _ = _rejection_rates(50, sim_mix=shifted)
    assert perm >= 0.9, perm


# --- The statistic and its p-value -----------------------------------------

def test_statistic_is_pearson_chi_square_of_the_category_totals():
    rng = np.random.default_rng(3)
    ref, sim = _baskets(rng, 300), _baskets(rng, 60)
    r = cluster_permutation_chi2(sim, ref, n_perm=TEST_N_PERM)
    table = np.vstack([sim.sum(axis=0), ref.sum(axis=0)])
    chi2, p, dof, _ = sp_stats.chi2_contingency(table, correction=False)
    assert np.isclose(r['statistic'], chi2)
    assert np.isclose(r['naive_p_value'], p) and r['df'] == dof
    assert r['n_sim_clusters'] == 60 and r['n_ref_clusters'] == 300
    assert r['n_items_sim'] == sim.sum() and r['n_items_ref'] == ref.sum()
    # The p-value sits on the permutation grid.
    reached = r['p_value'] * (TEST_N_PERM + 1) - 1
    assert np.isclose(reached, round(reached)) and 0 <= reached <= TEST_N_PERM


def test_identical_samples_give_zero_and_one():
    ref = _baskets(np.random.default_rng(4), 200)
    r = cluster_permutation_chi2(ref, ref.copy(), n_perm=TEST_N_PERM)
    assert r['statistic'] == 0.0 and r['p_value'] == 1.0


def test_production_count_seed_and_block_invariance(monkeypatch):
    """The default call runs CATEGORY_N_PERM relabellings under the fixed
    seed, repeats exactly, and does not depend on how the keys are cut
    into blocks."""
    rng = np.random.default_rng(5)
    ref, sim = _baskets(rng, 400), _baskets(rng, 80)
    r = cluster_permutation_chi2(sim, ref)
    assert r['n_permutations'] == CATEGORY_N_PERM == 2000
    assert r['permutation_seed'] == CATEGORY_PERM_SEED
    assert cluster_permutation_chi2(sim, ref) == r
    monkeypatch.setattr(dv, '_PERM_BLOCK_VALUES', 7 * (400 + 80))
    assert cluster_permutation_chi2(sim, ref) == r


def test_empty_baskets_and_columns_are_dropped_and_small_sides_refused():
    rng = np.random.default_rng(6)
    ref, sim = _baskets(rng, 200), _baskets(rng, 40)
    pad = lambda m: np.hstack([m, np.zeros((len(m), 1), dtype=m.dtype)])
    padded = cluster_permutation_chi2(
        np.vstack([pad(sim), np.zeros((3, 7), dtype=sim.dtype)]), pad(ref),
        n_perm=TEST_N_PERM)
    plain = cluster_permutation_chi2(sim, ref, n_perm=TEST_N_PERM)
    assert padded == plain
    few = cluster_permutation_chi2(sim[:4], ref, n_perm=TEST_N_PERM)
    assert np.isnan(few['p_value']) and 'Insufficient' in few['reason']


def test_unreferenced_categories_are_pooled_not_dropped():
    ref = np.array([[1, 0, 2, 0], [0, 0, 1, 0]])
    sim = np.array([[1, 1, 0, 0], [0, 2, 0, 1]])
    r, s, labels, pooled = merge_unreferenced_categories(
        ref, sim, ['A', 'B', 'C', 'D'])
    assert labels == ['A', 'C', UNREFERENCED_CATEGORY]
    assert pooled == ['B', 'D']
    assert r.tolist() == [[1, 2, 0], [0, 1, 0]]
    assert s.tolist() == [[1, 0, 1], [0, 0, 3]]
    # Nobody bought them: no column is kept for them.
    r, s, labels, pooled = merge_unreferenced_categories(
        ref, np.array([[1, 0, 1, 0]]), ['A', 'B', 'C', 'D'])
    assert labels == ['A', 'C'] and pooled == ['B', 'D']
    assert r.tolist() == [[1, 2], [0, 1]] and s.tolist() == [[1, 1]]


# --- Reference invoices and simulated visits ---------------------------------

PRODUCTS = [(f'{cat[:3].upper()}{i}', cat, 1.5 + i)
            for cat in ('Bakery', 'Dairy', 'Garden') for i in range(4)]
PLACED = [pid for pid, _, _ in PRODUCTS if not pid.endswith('3')]


def _invoice_df(n_invoices=300, seed=0):
    """Invoices that favour one department each, over three departments of
    four products."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp('2010-12-01 09:00')
    by_cat = defaultdict(list)
    for j, (_, cat, _) in enumerate(PRODUCTS):
        by_cat[cat].append(j)
    cats = sorted(by_cat)
    rows = []
    for k in range(n_invoices):
        home = by_cat[cats[int(rng.integers(len(cats)))]]
        picks = set()
        for _ in range(int(rng.integers(1, 5))):
            pool = home if rng.random() < 0.7 else range(len(PRODUCTS))
            picks.add(int(rng.choice(list(pool))))
        for j in sorted(picks):
            pid, cat, price = PRODUCTS[j]
            rows.append((f'I{k:05d}', pid, f'ITEM {pid}', 1,
                         t0 + pd.Timedelta(minutes=5 * k), price, cat))
    return pd.DataFrame(rows, columns=['invoice_id', 'product_id',
                                       'product_name', 'quantity',
                                       'timestamp', 'unit_price', 'category'])


@pytest.fixture(scope='module')
def params():
    return calibrate_transactional(_invoice_df())


def _shop():
    cat_of = {pid: cat for pid, cat, _ in PRODUCTS}
    items = {f'Item {pid}': {'product_id': pid, 'category': cat_of[pid]}
             for pid in PLACED}
    items['Gum'] = {'category': 'Impulse'}     # no product id
    return SimpleNamespace(floors={1: {'items': items}})


def _visits_as_invoices(params, n, seed):
    """Simulated visits drawn as random invoices of the reference, cut to
    the placed products: the model the category test's null describes."""
    rng = np.random.default_rng(seed)
    idx, ptr, items = (params.invoice_product_index, params.invoice_ptr,
                       params.invoice_items)
    visits = []
    while len(visits) < n:
        i = int(rng.integers(len(ptr) - 1))
        bought = [f'Item {idx[j]}' for j in items[ptr[i]:ptr[i + 1]]
                  if idx[j] in PLACED]
        if bought:
            visits.append(sorted(bought) + (['Gum'] if rng.random() < 0.2
                                            else []))
    return visits


def _stub_sim(analytics, shop=None):
    return SimpleNamespace(analytics=analytics,
                           shop=_shop() if shop is None else shop)


def test_invoice_clusters_are_the_placed_part_of_each_invoice(params):
    shop = _shop()
    pcat = placed_product_categories(params, shop)
    assert set(pcat) == set(PLACED)
    cats = sorted(set(pcat.values()))
    ref = invoice_category_clusters(params, pcat, cats)
    sample = placed_invoice_sample(params, PLACED)
    assert ref.shape == (sample['n_with_placed'], len(cats))
    assert ref.sum(axis=1).tolist() == sample['sizes'].tolist()
    # Column totals are the product-invoice touches the category panel draws.
    touches = observed_category_counts(params, product_ids=PLACED)
    assert dict(zip(cats, ref.sum(axis=0).tolist())) == \
        {c: int(v) for c, v in touches.items()}


def test_validation_row_tests_whole_visits_against_whole_invoices(params):
    visits = _visits_as_invoices(params, 150, seed=1)
    sim = _stub_sim({'basket_sizes': [len(v) for v in visits],
                     'customer_revenues': [1.0] * len(visits),
                     'visit_purchases': visits})
    res = validate_against_simulation(params, sim)
    row = {t.name: t for t in res.tests}[CATEGORY_TEST_NAME]
    assert row.test == CATEGORY_TEST_KIND
    assert row.applicable()
    assert row.n_observed == placed_invoice_sample(params, PLACED)['n_with_placed']
    assert row.n_simulated == len(visits)

    cat_of = {f'Item {pid}': cat for pid, cat, _ in PRODUCTS}
    cats = sorted({cat_of[f'Item {p}'] for p in PLACED})
    sim_tot = [sum(cat_of.get(i) == c for v in visits for i in v) for c in cats]
    ref_tot = observed_category_counts(params, product_ids=PLACED)
    table = np.array([sim_tot, [ref_tot[c] for c in cats]])
    chi2, naive_p, dof, _ = sp_stats.chi2_contingency(table, correction=False)
    assert np.isclose(row.statistic, chi2)
    assert np.isclose(row.extra['naive_p_value'], naive_p)
    assert row.extra['n_permutations'] == CATEGORY_N_PERM
    assert row.extra['permutation_seed'] == CATEGORY_PERM_SEED
    assert row.extra['n_categories'] == dof + 1 == len(cats)
    # The impulse pickups carry no product id and are counted as left out.
    assert row.extra['items_left_out'] == sum(v.count('Gum') for v in visits)
    assert 'naive' in row.note.lower()
    json.dumps(res.to_dict())


def test_old_record_without_visit_purchases_reports_not_run(params):
    analytics = {'basket_sizes': [2] * 6, 'customer_revenues': [3.0] * 6,
                 'item_conversion_rates': {'Item BAK0': {'purchases': 4}}}
    res = validate_against_simulation(params, _stub_sim(analytics))
    row = {t.name: t for t in res.tests}[CATEGORY_TEST_NAME]
    assert not row.applicable() and row.verdict() == 'N/A'
    assert 'visit_purchases' in row.note


def test_no_purchases_yet_leaves_the_row_out(params):
    for analytics in ({}, {'visit_purchases': []}):
        res = validate_against_simulation(params, _stub_sim(analytics))
        assert CATEGORY_TEST_NAME not in {t.name for t in res.tests}
        assert any('No simulated purchases yet' in s
                   for s in res.summary_lines)


def test_source_without_invoices_falls_back_to_the_item_level_test(params):
    bare = replace(params, invoice_product_index=[],
                   invoice_ptr=np.zeros(0, dtype=np.int64),
                   invoice_items=np.zeros(0, dtype=np.int32))
    analytics = {'item_conversion_rates': {f'Item {p}': {'purchases': 30}
                                           for p in PLACED}}
    res = validate_against_simulation(bare, _stub_sim(analytics))
    row = {t.name: t for t in res.tests}[CATEGORY_TEST_NAME]
    assert row.test == CATEGORY_ITEM_TEST_KIND
    assert 'independent' in row.note


# --- Per-visit purchases recorded at exit ------------------------------------

class _ExitSim(AnalyticsMixin):
    """Just enough of CustomerFlowSimulation for ``_process_customer_exit``."""

    def __init__(self):
        items = {'Milk': {'category': 'Dairy'}, 'Bread': {'category': 'Bakery'},
                 'Gum': {'category': 'Impulse'}}
        prices = {'Milk': 1.0, 'Bread': 2.0, 'Gum': 0.5}
        self.shop = SimpleNamespace(floors={1: {'items': items,
                                                'prices': prices}},
                                    items=items, prices=prices)
        self.sim_time = 120.0
        self.prev_state_by_cust = {}
        self.analytics = {
            'exit_traffic_by_minute': defaultdict(int),
            'return_customers': 0, 'total_revenue': 0.0,
            'completed_purchases': 0, 'abandoned_carts': 0,
            'basket_sizes': [], 'impulse_purchases': 0,
            'impulse_item_sales': defaultdict(int),
            'revenue_by_area': defaultdict(float),
            'item_conversion_rates': defaultdict(
                lambda: {'visits': 0, 'purchases': 0}),
            'popular_items': defaultdict(int),
            'cross_merchandising': defaultdict(int),
            'customer_lifetime_values': defaultdict(float),
            'dwell_times_by_zone': defaultdict(list),
            'customer_paths': [],
        }


def _customer(cid, visited, impulse, paid=True):
    return SimpleNamespace(
        id=cid, has_checked_out=paid, loyalty_level='new',
        state='exiting', state_history=['entering', 'moving', 'exiting'],
        start_shop_time=0.0, visited_items=set(visited),
        impulse_items=list(impulse), zone_dwell_times={},
        movement_path=[], visited_zones=set())


def test_exit_records_each_paying_visits_purchases():
    sim = _ExitSim()
    sim._process_customer_exit(_customer(1, {'Milk', 'Bread'}, ['Gum']))
    sim._process_customer_exit(_customer(2, {'Milk'}, [], paid=False))
    sim._process_customer_exit(_customer(3, {'Bread'}, []))
    A = sim.analytics
    assert A['visit_purchases'] == [['Bread', 'Milk', 'Gum'], ['Bread']]
    assert [len(v) for v in A['visit_purchases']] == A['basket_sizes']
    assert len(A['visit_purchases']) == len(A['customer_revenues'])
    json.dumps(A['visit_purchases'])
