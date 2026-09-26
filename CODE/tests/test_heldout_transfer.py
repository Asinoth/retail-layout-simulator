"""Held-out transfer of the optimized layouts (review R12), without the
workbook.

``experiments.run_heldout_transfer`` searches Figure C's store built from
the prior period, re-keys the current period onto the same fixtures, and
scores the layouts found on the prior period under the current one against
a search re-run in hindsight. These tests pin, on two synthetic periods
(the current one lacking one stocked product):

  * the current store keeps the prior store's fixtures and as-built layout,
    and its as-built revenue is the current calibration's own level;
  * a product that did not sell in the current period is reported, and
    stays on the floor;
  * the scoring identities: a 'found' layout equal to the as-built one has
    zero promised, transferred and hindsight lift, and no transfer ratio;
  * the across-seed transfer gap is a paired comparison, and the summary
    is standard JSON even where a ratio is undefined;
  * the command line takes Figure C's defaults and refuses bad designs
    before looking for the workbook;
  * the runner end to end, and its summary does not depend on --workers.
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
from experiments import run_heldout_transfer as RHT


def _invoices(seed, day0, drop=()):
    rng = np.random.default_rng(seed)
    products = [(f'{cat[:3].upper()}{i}', cat, 1.5 + i)
                for cat in ('Bakery', 'Dairy', 'Garden', 'Kitchen')
                for i in range(4)]
    products = [p for p in products if p[0] not in drop]
    start = pd.Timestamp(day0)
    rows = []
    for k in range(300):
        ts = start + pd.Timedelta(days=k // 30,
                                  minutes=int(rng.integers(0, 480)))
        cust = 'nan' if k % 7 == 0 else f'{13000 + int(rng.integers(0, 30))}.0'
        for j in rng.choice(len(products), size=int(rng.integers(1, 5)),
                            replace=False):
            pid, cat, price = products[j]
            rows.append((f'{seed}-{k:05d}', pid, f'ITEM {pid}',
                         int(rng.integers(1, 4)), ts, price, cat, cust))
    return pd.DataFrame(rows, columns=['invoice_id', 'product_id',
                                       'product_name', 'quantity',
                                       'timestamp', 'unit_price', 'category',
                                       'customer_id'])


FRAMES = {'prior': _invoices(1, '2009-12-01 09:00'),
          'current': _invoices(2, '2010-12-01 09:00', drop=('GAR3',))}


def _record(period):
    df = FRAMES[period]
    return {'period': period, 'sheet': period, 'rows_in': len(df),
            'rows_calibrated': len(df),
            'first_date': str(df['timestamp'].min().date()),
            'last_date': str(df['timestamp'].max().date()),
            'n_invoices': int(df['invoice_id'].nunique()),
            'source_sha256': '0' * 64, 'adapter_version': 'test'}


@pytest.fixture(scope='module')
def params():
    return {p: calibrate_transactional(FRAMES[p], currency='GBP')
            for p in RHT.PERIODS}


@pytest.fixture(scope='module')
def stores(params):
    return {p: RHT.period_store(params, p, 4) for p in RHT.PERIODS}


def test_current_store_keeps_the_prior_fixtures(stores, params):
    prior, current = stores['prior'], stores['current']
    assert current.item_names == prior.item_names
    assert current.baseline_layout == prior.baseline_layout
    for n in prior.item_names:
        a = prior.shop.floors[1]['items'][n]
        b = current.shop.floors[1]['items'][n]
        assert (a['position'], a['size'], a.get('product_id')) == \
            (b['position'], b['size'], b.get('product_id'))
    # Anchored at the as-built layout under the current statistics, the
    # as-built store earns the current calibration's level exactly.
    days = np.arange(7)
    mult = np.where(days % 7 >= 5, RL.DEFAULT_WEEKEND_MULTIPLIER, 1.0).sum()
    bp = current.base_params
    level = (bp['customers_per_hour'] * RL.DEFAULT_OP_HOURS_PER_DAY * mult
             * bp['conversion_rate'] * bp['rev_per_customer_gross'])
    assert RK.layout_value(current, current.baseline_layout, 7) == \
        pytest.approx(level, rel=1e-9)
    assert bp['rev_per_customer_gross'] == pytest.approx(
        params['current'].invoice_revenues.mean())


def test_products_gone_from_the_current_period_are_reported(stores, params):
    facts = RHT.fixture_facts(stores, params, 4)
    assert facts['n_absent_in_current'] == 1
    (gone,) = facts['absent_in_current']
    assert gone['product_id'] == 'GAR3' and gone['prior_invoices'] > 0
    # It stays on the floor.
    assert gone['item'] in stores['current'].shop.floors[1]['items']
    assert facts['current_top_n_not_stocked'] == 0


def test_as_built_found_layouts_have_zero_lift(stores):
    base = {k: list(v) for k, v in stores['prior'].baseline_layout.items()}
    results = [{'period': p, 'seed': 0,
                'layouts': {m: base for m in RHT.METHODS}}
               for p in RHT.PERIODS]
    scored = RHT.score_transfer(stores, results, [0], mc_days=7, mc_iters=30,
                                n_replicates=3)
    (rec,) = scored['per_seed']
    for m in RHT.METHODS:
        for k in RHT.LIFTS:
            assert rec['methods'][m][k]['lift'] == pytest.approx(0.0,
                                                                abs=1e-9)
        for k in ('transferred', 'hindsight'):
            assert rec['methods'][m][k]['mc']['lift'] == 0.0
        # No achievable lift: the ratio is undefined, not NaN.
        assert rec['methods'][m]['transfer_ratio'] is None
        assert rec['methods'][m]['transfer_gap'] == pytest.approx(0.0,
                                                                 abs=1e-9)


def _per_seed(transferred, hindsight, base=1000.0):
    out = []
    for s, (t, h) in enumerate(zip(transferred, hindsight)):
        rec = {}
        for m in RHT.METHODS:
            rec[m] = {'promised': {'lift': h, 'pct': h / base * 100.0},
                      'transferred': {'lift': t, 'pct': t / base * 100.0,
                                      'mc': {'lift': t + 1.0}},
                      'hindsight': {'lift': h, 'pct': h / base * 100.0,
                                    'mc': {'lift': h + 3.0}},
                      'transfer_ratio': RHT.transfer_ratio(t, h),
                      'transfer_gap': h - t}
        out.append({'seed': s, 'methods': rec})
    return out


def test_transfer_gap_is_paired_within_seeds():
    from experiments.run_real_data_seeds import t_interval
    t = [4.0, 1.0, 6.0, 2.5]
    h = [5.0, 3.0, 6.5, 2.0]
    a = RHT.summarize_transfer(_per_seed(t, h))['methods']['GA']
    gap = a['transfer_gap']
    diffs = [hh - tt for tt, hh in zip(t, h)]
    mean, lo, hi, sd = t_interval(diffs, RHT.CI_LEVEL)
    assert gap['gbp']['mean'] == pytest.approx(mean)
    assert gap['gbp']['ci'] == pytest.approx([lo, hi])
    assert gap['gbp']['per_seed'] == pytest.approx(diffs)
    # Paired: the interval is on the differences, whose spread is smaller
    # here than either lift's own.
    assert gap['gbp']['sd'] == pytest.approx(sd)
    assert gap['gbp']['sd'] < a['transferred']['gbp']['sd']
    assert gap['pct']['mean'] == pytest.approx(mean / 1000.0 * 100.0)
    assert gap['mc_gbp']['mean'] == pytest.approx(mean + 2.0)
    assert gap['n_seeds_transferred_below_hindsight'] == 3
    assert a['transfer_ratio_of_means'] == pytest.approx(
        np.mean(t) / np.mean(h))
    assert 'transfer_ratio' not in a          # no t-summary of ratios
    assert a['n_seeds_transfer_ratio_defined'] == 4


def test_undefined_ratios_are_null_and_the_summary_is_standard_json():
    # A seed whose hindsight search found nothing (0) and one that lost
    # lift (-1): both ratios undefined, and so is the ratio of means.
    across = RHT.summarize_transfer(_per_seed([0.5, -2.0], [0.0, -1.0]))
    a = across['methods']['GA']
    assert a['transfer_ratio_of_means'] is None
    assert a['n_seeds_transfer_ratio_defined'] == 0
    json.dumps(across, allow_nan=False)


def _fake_io(monkeypatch, tmp_path, found=None):
    book = tmp_path / 'book.xlsx'
    book.write_bytes(b'stand-in workbook')

    def _resolve(p, args):
        if found is None:
            raise AssertionError("workbook looked up before the arguments "
                                 "were checked")
        found.append(True)
        args.retail_path = str(book)

    monkeypatch.setattr(RHT, 'resolve_workbook', _resolve)
    monkeypatch.setattr(RHT.LS, 'period_frame',
                        lambda path=None, period='current': FRAMES[period])
    monkeypatch.setattr(RHT.LS, 'period_record',
                        lambda path=None, period='current': _record(period))


@pytest.mark.parametrize('argv', [['--n-search-seeds', '1'],
                                  ['--workers', '5'], ['--workers', '0'],
                                  ['--ga-seed', '998'],
                                  ['--n-mc-replicates', '1']])
def test_command_line_refuses_bad_designs_before_the_workbook(
        monkeypatch, tmp_path, argv):
    _fake_io(monkeypatch, tmp_path)
    with pytest.raises(SystemExit):
        RHT.parse_args(argv)


def test_command_line_takes_figure_cs_defaults(monkeypatch, tmp_path):
    found = []
    _fake_io(monkeypatch, tmp_path, found)
    args = RHT.parse_args([])
    d = RHT._figure_c_defaults()
    for k in ('max_items_per_category', 'assumed_conversion', 'mc_iters',
              'mc_days', 'n_gens', 'pop_size', 'n_mc_replicates', 'ga_seed',
              'sa_initial_accept'):
        assert getattr(args, k) == d[k], k
    assert args.n_search_seeds == RHT.DEFAULT_N_SEARCH_SEEDS
    assert found == [True]


def _run(tmp_path, workers):
    out = tmp_path / f'w{workers}'
    assert RHT.main(['--n-search-seeds', '2', '--mc-iters', '30',
                     '--mc-days', '7', '--n-gens', '2', '--pop-size', '4',
                     '--n-mc-replicates', '3', '--max-items-per-category',
                     '4', '--workers', str(workers),
                     '--out-root', str(out)]) == 0
    (run,) = glob.glob(str(out / 'heldout_transfer_*'))
    with open(os.path.join(run, 'summary.json')) as f:
        # Standard JSON: no NaN or Infinity token anywhere.
        summary = json.load(f, parse_constant=lambda c: pytest.fail(
            f"non-standard JSON constant {c} in summary.json"))
    with open(os.path.join(run, 'sidecar.json')) as f:
        sidecar = json.load(f)
    return summary, sidecar


def test_runner_end_to_end_and_worker_count_invariant(monkeypatch, tmp_path):
    _fake_io(monkeypatch, tmp_path, [])
    one, side = _run(tmp_path, 1)
    two, _ = _run(tmp_path, 2)
    assert json.dumps(one, sort_keys=True) == json.dumps(two, sort_keys=True)
    assert [r['seed'] for r in one['per_seed']] == [0, 1]
    for m in RHT.METHODS:
        a = one['across_seeds']['methods'][m]
        assert set(a) >= {'promised', 'transferred', 'hindsight',
                          'transfer_gap', 'transfer_ratio_of_means',
                          'n_seeds_transferred_positive'}
        assert a['transfer_gap']['gbp']['per_seed'] == pytest.approx(
            [r['methods'][m]['transfer_gap'] for r in one['per_seed']])
    assert one['experiment'] == 'heldout_transfer_uci'
    assert set(one['across_seeds']['transferred_ga_vs']) == {
        'GA_minus_random_search', 'GA_minus_simulated_annealing'}
    assert one['fixtures']['n_absent_in_current'] == 1
    assert one['design']['search_start'] == 'asbuilt'
    counts = one['evaluation_counts']
    assert all(c[m] == 8 for c in counts.values() for m in RHT.METHODS)
    assert set(side['base_params']) == {'prior', 'current'}
    # Every reported layout of every (period, seed) task was checked
    # against the floor-plan invariants, and the operators and the aisle
    # rule are recorded as Figure C records them.
    feas = one['feasibility']
    assert set(feas['invariants']) == set(counts)
    for task in feas['invariants'].values():
        assert set(task) == {'baseline', 'optimized', 'rs', 'sa'}
        for rec in task.values():
            assert all(v == 0 for k, v in rec.items()
                       if k.startswith('n_'))
    assert set(feas['repair_stats']) == set(counts)
    assert 'n_zones_narrowed' in feas['aisle_rule']
    assert set(one['design']['operators']) >= {
        'space', 'GA', 'random_search', 'simulated_annealing'}
    assert side['provenance']['extra']['periods']['current'][
        'source_sha256'] == '0' * 64
