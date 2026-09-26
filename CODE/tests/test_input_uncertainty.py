"""Input uncertainty of Figure C's lift (review R11), without the workbook.

``experiments.run_input_uncertainty`` resamples customers with
replacement, recalibrates, re-keys each replicate onto Figure C's fixtures
and re-scores the fixed layouts (as-built, GA, random search, annealing) in
closed form. These tests pin, on a
synthetic invoice frame with identified and anonymous buyers:

  * the resampling: whole customers, anonymous invoices as their own
    clusters, copies relabelled as distinct invoices and customers, and the
    same generator giving the same replicate;
  * re-keying the full data onto a fresh store reproduces the full-data
    scores exactly, and the stocked scale gives the whole scale's
    percentages (the lift and the GA's margins over the comparators);
  * the delta method's level SE is the clustered SE of the total, and the
    level is proportional to that total;
  * the runner end to end, and its summary does not depend on --workers;
    the margins get input intervals; the former law's bias is reported at
    the current and at adapter 1.1's inputs; a no-anonymous sensitivity
    run goes to a directory of its own.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import pytest

import retail_literature as RL
from dataset_calibration import calibrate_transactional
from experiments import _rekey as RK
from experiments import run_input_uncertainty as RIU
from experiments.run_real_data_example import build_store


def _invoices(n_invoices=360, seed=0):
    """Invoices over four categories from 40 customers and a stream of
    anonymous buyers (every sixth invoice)."""
    rng = np.random.default_rng(seed)
    products = [(f'{cat[:3].upper()}{i}', cat, 1.5 + i)
                for cat in ('Bakery', 'Dairy', 'Garden', 'Kitchen')
                for i in range(4)]
    day0 = pd.Timestamp('2010-12-01 09:00')
    rows = []
    for k in range(n_invoices):
        ts = day0 + pd.Timedelta(days=k // 30,
                                 minutes=int(rng.integers(0, 480)))
        cust = 'nan' if k % 6 == 0 else f'{12000 + int(rng.integers(0, 40))}.0'
        for j in rng.choice(len(products), size=int(rng.integers(1, 5)),
                            replace=False):
            pid, cat, price = products[j]
            rows.append((f'I{k:05d}', pid, f'ITEM {pid}',
                         int(rng.integers(1, 4)), ts, price, cat, cust))
    return pd.DataFrame(rows, columns=['invoice_id', 'product_id',
                                       'product_name', 'quantity',
                                       'timestamp', 'unit_price', 'category',
                                       'customer_id'])


@pytest.fixture(scope='module')
def frame():
    return _invoices()


def test_clusters_are_customers_and_anonymous_invoices(frame):
    cl = RIU.customer_clusters(frame)
    anon = frame['customer_id'] == 'nan'
    assert cl.n_anonymous_invoices == frame.loc[anon, 'invoice_id'].nunique()
    assert cl.n_customers == frame.loc[~anon, 'customer_id'].nunique()
    assert cl.n == cl.n_customers + cl.n_anonymous_invoices
    # Every cluster is one customer, or one anonymous invoice.
    for c in range(cl.n):
        rows = frame.iloc[cl.order[cl.starts[c]:cl.starts[c + 1]]]
        if (rows['customer_id'] == 'nan').all():
            assert rows['invoice_id'].nunique() == 1
        else:
            assert rows['customer_id'].nunique() == 1


def test_resample_keeps_whole_customers_and_relabels_copies(frame):
    cl = RIU.customer_clusters(frame)
    a = RIU.resample(frame, cl, np.random.default_rng([7, 0]))
    b = RIU.resample(frame, cl, np.random.default_rng([7, 0]))
    pd.testing.assert_frame_equal(a, b)
    # Undo the relabelling: every copy is a whole original invoice.
    base_inv = a['invoice_id'].str.split('#').str[0]
    for inv, g in a.groupby('invoice_id'):
        orig = frame[frame['invoice_id'] == inv.split('#')[0]]
        assert len(g) == len(orig)
    # Copies are distinct invoices; an identified customer's copy is a
    # distinct customer, an anonymous one stays anonymous.
    copies = a['invoice_id'].str.contains('#')
    assert copies.any()
    anon = a['customer_id'] == 'nan'
    assert a.loc[copies & ~anon, 'customer_id'].str.contains('#').all()
    assert (a.loc[anon, 'customer_id'] == 'nan').all()
    assert base_inv.nunique() <= frame['invoice_id'].nunique()
    # As many clusters as the data holds were drawn.
    n_drawn = (a.loc[~anon, 'customer_id'].nunique()
               + a.loc[anon, 'invoice_id'].nunique())
    assert n_drawn == cl.n


@pytest.fixture(scope='module')
def setup(frame):
    params = calibrate_transactional(frame, currency='GBP')
    store = build_store(params, 4, verbose=False)
    # Layouts other than the as-built one: the store's zone sampler.
    from experiments._common import feasible_layout, zone_sampler
    lays = {k: feasible_layout(store.shop, store.item_names,
                               zone_sampler(store.shop, store.item_names)(
                                   np.random.default_rng(seed)))
            for k, seed in (('ga', 3), ('rs', 4), ('sa', 5))}
    return params, store, lays


def test_rekeying_the_full_data_reproduces_the_full_data_scores(setup):
    params, store, lays = setup
    pids = RK.fixture_products(store)
    point = RIU.score_fixed_layouts(store, params, lays, pids, 14)
    fresh = build_store(params, 4, verbose=False)
    again = RIU.score_fixed_layouts(RK.reseed_store(fresh, params), params,
                                    lays, pids, 14)
    assert again == point
    # One replicate in between does not leak into the next.
    other = calibrate_transactional(_invoices(seed=5), currency='GBP')
    RK.reseed_store(fresh, other)
    assert RIU.score_fixed_layouts(RK.reseed_store(fresh, params), params,
                                   lays, pids, 14) == point


def test_the_stocked_scale_keeps_every_percentage(setup):
    params, store, lays = setup
    point = RIU.score_fixed_layouts(store, params, lays,
                                    RK.fixture_products(store), 14)
    assert point['stocked']['pct_lift'] == pytest.approx(
        point['whole']['pct_lift'], rel=1e-9, abs=1e-12)
    assert point['stocked']['level_asbuilt'] < point['whole']['level_asbuilt']
    for key in RIU.COMPARATORS:
        name = f'ga_minus_{key}'
        for v in ('pct', 'frac_of_asbuilt'):
            assert point['stocked'][name][v] == pytest.approx(
                point['whole'][name][v], rel=1e-9, abs=1e-12)


def test_margins_are_differences_of_the_levels(setup):
    params, store, lays = setup
    point = RIU.score_fixed_layouts(store, params, lays,
                                    RK.fixture_products(store), 14)
    for scale in RIU.SCALES:
        p = point[scale]
        assert p['lift'] == p['level_ga'] - p['level_asbuilt']
        for key in RIU.COMPARATORS:
            m = p[f'ga_minus_{key}']
            assert m['gbp'] == p['level_ga'] - p[f'level_{key}']
            assert m['frac_of_asbuilt'] * p['level_asbuilt'] == \
                pytest.approx(m['gbp'], rel=1e-12, abs=1e-9)
            assert m['pct'] == pytest.approx(
                m['gbp'] / p[f'level_{key}'] * 100.0, rel=1e-12)
    # Without comparators only the lift is scored.
    ga_only = RIU.score_fixed_layouts(store, params, {'ga': lays['ga']},
                                      RK.fixture_products(store), 14)
    assert 'ga_minus_rs' not in ga_only['whole']
    assert ga_only['whole']['lift'] == point['whole']['lift']


def test_level_is_proportional_to_the_total_and_its_se_is_clustered(
        frame, setup):
    params, store, lays = setup
    cl = RIU.customer_clusters(frame)
    pids = RK.fixture_products(store)
    totals = RIU.cluster_totals(frame, cl, params, pids)
    point = RIU.score_fixed_layouts(store, params, lays, pids, 14)
    days = np.arange(14)
    mult = np.where(days % 7 >= 5, RL.DEFAULT_WEEKEND_MULTIPLIER, 1.0).sum()
    mean_mult = (5.0 + 2.0 * RL.DEFAULT_WEEKEND_MULTIPLIER) / 7.0
    n_cal = params.calibration_extra['n_calendar_days']
    for scale in RIU.SCALES:
        assert point[scale]['level_asbuilt'] == pytest.approx(
            mult / mean_mult * totals[scale]['total'] / n_cal, rel=1e-9)
    # The clustered SE of a total is the bootstrap SD of the resampled
    # total (up to C / (C - 1)).
    rng = np.random.default_rng(1)
    x = rng.lognormal(0.0, 1.0, 300)
    boot = [x[rng.integers(0, x.size, x.size)].sum() for _ in range(4000)]
    assert RIU.cluster_total_se(x) == pytest.approx(np.std(boot), rel=0.06)


def _fake_io(monkeypatch, frame, tmp_path):
    book = tmp_path / 'book.xlsx'
    book.write_bytes(b'stand-in workbook')

    class _Report:
        rows_in = rows_kept = len(frame)

        def __init__(self, match_reversals):
            self.extra = {'currency': 'GBP',
                          'cleaning': {'reversal_pairs': 0,
                                       'reversal_matching': match_reversals}}

    def _resolve(p, args):
        args.retail_path = str(book)

    monkeypatch.setattr(RIU, 'resolve_workbook', _resolve)
    monkeypatch.setattr(RIU, 'read_workbook',
                        lambda args: (frame.copy(), {'rows': 1}))
    # The synthetic frame is already clean, so both cleanings return it.
    monkeypatch.setattr(RIU, 'adapt_rows',
                        lambda df, match_reversals=True: (
                            df.copy(), _Report(match_reversals)))


def _run(tmp_path, workers, extra=(), prefix='input_uncertainty_'):
    out = tmp_path / f'w{workers}{"".join(extra)}'
    assert RIU.main(['--mc-iters', '40', '--mc-days', '7', '--n-gens', '2',
                     '--pop-size', '4', '--n-boot', '4',
                     '--max-items-per-category', '4',
                     '--workers', str(workers), '--out-root', str(out),
                     *extra]) == 0
    (run,) = glob.glob(str(out / '*'))
    assert os.path.basename(run).startswith(prefix)
    with open(os.path.join(run, 'summary.json')) as f:
        summary = json.load(f)
    with open(os.path.join(run, 'sidecar.json')) as f:
        sidecar = json.load(f)
    return summary, sidecar


def test_runner_end_to_end_and_worker_count_invariant(monkeypatch, frame,
                                                       tmp_path):
    _fake_io(monkeypatch, frame, tmp_path)
    one, side = _run(tmp_path, 1)
    two, _ = _run(tmp_path, 2)
    assert json.dumps(one, sort_keys=True) == json.dumps(two, sort_keys=True)
    iu = one['input_uncertainty']
    assert iu['n_boot'] == 4
    for scale in RIU.SCALES:
        s = iu[scale]
        assert s['pct_lift']['covers'] == 'input'
        lo, hi = s['level_gbp']['ci']
        assert lo < s['level_gbp']['estimate'] < hi
        assert s['lift_gbp']['se'] >= 0.0
    assert set(one['uncertainty_layers']) == {'input', 'monte_carlo',
                                              'search_seed'}
    assert one['design']['n_boot'] == 4 and one['design']['boot_seed'] == \
        RIU.DEFAULT_BOOT_SEED
    assert one['anonymous']['n_anonymous_with_placed'] > 0
    assert one['experiment'] == 'input_uncertainty_uci'
    assert one['design']['exclude_anonymous'] is False
    # The GA's margins over the equal-budget comparators, searched once at
    # Figure C's budget from the as-built layout, carry input intervals.
    srch = one['design']['searches']
    assert set(srch['starts'].values()) == {'asbuilt'}
    # Every layout the run re-scores was checked against the floor-plan
    # invariants, and the operators and the aisle rule are recorded as
    # Figure C records them.
    feas = one['feasibility']
    assert set(feas['invariants']) == {'baseline', 'optimized', 'rs', 'sa'}
    for rec in feas['invariants'].values():
        assert all(v == 0 for k, v in rec.items() if k.startswith('n_'))
    assert 'n_zones_narrowed' in feas['aisle_rule']
    assert set(srch['operators']) >= {'space', 'GA', 'random_search',
                                      'simulated_annealing'}
    assert len(set(srch['evaluation_counts'].values())) == 1
    for scale in RIU.SCALES:
        for key in RIU.COMPARATORS:
            m = iu[scale][f'ga_minus_{key}']
            assert m['pct']['covers'] == m['gbp']['covers'] == 'input'
            assert m['pct']['pct_of'] == 'comparator_level'
            assert m['pct']['estimate'] == \
                one['point'][scale][f'ga_minus_{key}']['pct']
            lo, hi = m['gbp']['ci']
            assert lo <= m['gbp']['estimate'] <= hi
            assert 0 <= m['n_boot_ga_ahead'] <= 4
    # The former spend law's bias, at this run's inputs and at adapter
    # 1.1's (the same frame here, so the two agree).
    fb = one['legacy_floor_bias']
    for block in ('current_inputs', 'adapter_1_1_inputs'):
        b = fb[block]
        assert {'asbuilt', 'ga', 'rs', 'sa', 'lift', 'spend_cv',
                'frac_of_asbuilt_level', 'frac_of_lift'} <= set(b)
        assert b['lift'] == b['ga'] - b['asbuilt']
        assert b['asbuilt'] >= 0.0
    assert fb['adapter_1_1_inputs']['cleaning']['reversal_matching'] is False
    assert fb['adapter_1_1_inputs']['asbuilt'] == pytest.approx(
        fb['current_inputs']['asbuilt'], rel=1e-12)
    # The sidecar carries the dataset hash and the anchored base params.
    assert len(side['provenance']['source_sha256']) == 64
    assert side['base_params']['score_anchor']['source'] == \
        'as_built_repaired'
    assert side['elasticities']['objective_constants']['MC_SPEND_LAW'] == \
        RL.MC_SPEND_LAW


def test_runner_without_anonymous_invoices(monkeypatch, frame, tmp_path):
    """The sensitivity run: the calibration records the exclusion and the
    bootstrap resamples identified customers only."""
    _fake_io(monkeypatch, frame, tmp_path)
    # A sensitivity run: its own directory prefix and experiment name, so
    # the macro generator's newest-valid-run rule cannot take it for the
    # headline family.
    summary, side = _run(tmp_path, 1, ('--exclude-anonymous',),
                         prefix='noanon_input_uncertainty_')
    assert summary['experiment'] == 'input_uncertainty_uci_noanon'
    assert summary['design']['exclude_anonymous'] is True
    assert summary['design']['clusters']['n_anonymous_invoices'] == 0
    anon = side['provenance']['extra']['anonymous']
    assert anon['calibration'] == 'excluded'
    assert anon['n_anonymous_invoices_excluded'] == \
        frame.loc[frame['customer_id'] == 'nan', 'invoice_id'].nunique()
    assert summary['anonymous']['n_anonymous_with_placed'] == 0


@pytest.mark.parametrize('argv', [['--n-boot', '1'], ['--workers', '5'],
                                  ['--workers', '0'], ['--boot-seed', '-1']])
def test_command_line_refuses_bad_designs(monkeypatch, argv):
    monkeypatch.setattr(RIU, 'resolve_workbook',
                        lambda p, a: (_ for _ in ()).throw(AssertionError()))
    with pytest.raises(SystemExit):
        RIU.parse_args(argv)


def test_sensitivity_runs_get_a_family_of_their_own():
    """The macro generator takes the newest valid run whose directory
    starts with '<family>_'; a no-anonymous run's directory must match no
    headline family, or it could replace the headline's macros."""
    import argparse
    from experiments.run_real_data_example import (experiment_name,
                                                   run_dir_prefix)
    on = argparse.Namespace(exclude_anonymous=True)
    off = argparse.Namespace(exclude_anonymous=False)
    families = ('real_data_uci', 'real_data_seeds', 'input_uncertainty',
                'heldout_transfer')
    for base in families:
        assert run_dir_prefix(base, off) == base
        run_dir = run_dir_prefix(base, on) + '_20260925-120000'
        assert not any(run_dir.startswith(f + '_') for f in families)
    assert experiment_name('real_data_uci_figure_c', off) == \
        'real_data_uci_figure_c'
    assert experiment_name('real_data_uci_figure_c', on) == \
        'real_data_uci_figure_c_noanon'
