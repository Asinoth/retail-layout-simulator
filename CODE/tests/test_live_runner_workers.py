"""The live diagnostics write the same summary for any worker count.

Each live runner seeds every run explicitly and aggregates results in
design order, so running all replications in one process or spreading them
over worker processes must not change a single number. Tiny runs on a
small synthetic calibrated store keep this quick: the runners take their
store from ``experiments._live_store``, which is pointed at synthetic
calibrations here, one per period. The protocol block's ``workers`` entry
records the worker count and is expected to differ; so is a
``provenance`` block, which records the checkout the run started from and
changes whenever a source file is edited between the two runs.
"""
import csv
import glob
import json

import numpy as np
import pandas as pd
import pytest

from dataset_calibration import calibrate_transactional
from experiments import (_live_store, measure_queueing, run_abm_diagnostics,
                         run_structural_sensitivity, run_validation_gof)


TINY = ['--warmup', '30', '--seconds', '90', '--reps', '2']
WORKER_COUNTS = (1, 3)


def _invoices(n_invoices=400, seed=0, day0='2010-12-01 09:00'):
    """Invoices over three categories of four products each, enough for the
    layout builder to lay out a small store with two checkout lanes."""
    rng = np.random.default_rng(seed)
    products = [(f'{cat[:3].upper()}{i}', cat, 1.5 + i)
                for cat in ('Bakery', 'Dairy', 'Garden') for i in range(4)]
    day0 = pd.Timestamp(day0)
    rows = []
    for k in range(n_invoices):
        ts = day0 + pd.Timedelta(days=k // 40,
                                 minutes=int(rng.integers(0, 480)))
        picks = rng.choice(len(products), size=int(rng.integers(1, 5)),
                           replace=False)
        for j in picks:
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
def prior_params():
    """A calibration of an earlier, disjoint stretch of invoices."""
    return calibrate_transactional(
        _invoices(seed=1, day0='2009-12-01 09:00'), currency='GBP')


def _record(period, params):
    """A stand-in for ``_live_store.period_record`` with the fields the
    runners print."""
    return {'period': period, 'workbook': 'synthetic',
            'first_date': 'first', 'last_date': 'last',
            'n_invoices': int(params.n_invoices)}


def _use_store(monkeypatch, by_period):
    """Serve the runners' store from ``by_period`` instead of the workbook."""
    monkeypatch.setattr(_live_store, 'calibrate_live_store',
                        lambda retail_path=None, period='current':
                        by_period[period])
    monkeypatch.setattr(_live_store, 'period_record',
                        lambda retail_path=None, period='current':
                        _record(period, by_period[period]))


def _summary(pattern):
    paths = glob.glob(pattern)
    assert len(paths) == 1, paths
    with open(paths[0]) as f:
        return json.load(f)


def _canonical(summary, workers):
    """Summary text with the worker count and the run provenance removed;
    text rather than parsed objects so a NaN compares equal to itself."""
    assert summary['protocol'].pop('workers') == workers
    summary.pop('provenance', None)
    return json.dumps(summary, sort_keys=True)


def test_abm_diagnostics_summary_does_not_depend_on_worker_count(
        params, monkeypatch, tmp_path):
    _use_store(monkeypatch, {'current': params})
    texts = []
    for workers in WORKER_COUNTS:
        root = tmp_path / f'workers{workers}'
        run_abm_diagnostics.main(TINY + [
            '--spawn', '0.5', '--cap', '30',
            '--workers', str(workers), '--out-root', str(root)])
        s = _summary(str(root / 'abm_diagnostics_*' / 'summary.json'))
        assert s['protocol']['mode'] == 'fixed_step'
        assert s['protocol']['seeds'] == [1000, 1001]
        assert s['markov_order']['n_sequences'] > 0
        assert s['data'] == _record('current', params)
        em = s['emergence']
        assert em['perimeter_interior_ratio_geometric_null'] > 0
        rep0 = s['per_rep'][0]['emergence']
        assert set(rep0['band_sweep_geometric_null']) == set(rep0['band_sweep'])
        texts.append(_canonical(s, workers))
    assert texts[0] == texts[1]


def test_structural_sweep_summary_does_not_depend_on_worker_count(
        params, monkeypatch, tmp_path):
    _use_store(monkeypatch, {'current': params})
    texts = []
    for workers in WORKER_COUNTS:
        root = tmp_path / f'workers{workers}'
        run_structural_sensitivity.main(TINY + [
            '--spawn', '0.5', '--cap', '30',
            '--workers', str(workers), '--out-root', str(root)])
        s = _summary(str(root / 'structural_sensitivity_*' / 'summary.json'))
        seeds = [x for v in s['protocol']['seeds'].values() for x in v]
        assert len(set(seeds)) == len(seeds)
        assert sum(s['completed_per_setting']) > 0
        texts.append(_canonical(s, workers))
    assert texts[0] == texts[1]


def test_queueing_summary_does_not_depend_on_worker_count(
        params, monkeypatch, tmp_path):
    _use_store(monkeypatch, {'current': params})
    texts = []
    for workers in WORKER_COUNTS:
        root = tmp_path / f'workers{workers}'
        figs = root / 'figs'
        measure_queueing.main(TINY + [
            '--spawn', '0.5', '--cap', '30',
            '--stress-spawn', '1.5', '--stress-cap', '40',
            '--workers', str(workers), '--out-root', str(root),
            '--figs-dir', str(figs)])
        s = _summary(str(figs / 'queue_summary.json'))
        assert s == _summary(str(root / 'measure_queueing_*' / 'summary.json'))
        assert (figs / 'queue_lengths.pdf').exists()
        assert s['stress']['n_waits'] > 0
        texts.append(_canonical(s, workers))
    assert texts[0] == texts[1]


def test_validation_gof_summary_does_not_depend_on_worker_count(
        params, prior_params, monkeypatch, tmp_path):
    _use_store(monkeypatch, {'current': params, 'prior': prior_params})
    texts = []
    for workers in WORKER_COUNTS:
        root = tmp_path / f'workers{workers}'
        run_validation_gof.main(TINY + [
            '--spawn', '0.5', '--cap', '30',
            '--workers', str(workers), '--out-root', str(root)])
        s = _summary(str(root / 'validation_gof_*' / 'summary.json'))
        held = s['held_out']
        # The held-out replications continue the in-sample seed sequence.
        assert s['protocol']['seeds'] == [4000, 4001]
        assert held['protocol']['seeds'] == [4002, 4003]
        assert s['pooled'] and held['pooled']
        assert set(held) == set(s) - {'design', 'held_out'}
        assert s['design']['in_sample']['store_period'] == 'current'
        assert s['design']['held_out']['store_period'] == 'prior'
        assert s['design']['held_out']['reference_period'] == 'current'
        assert s['design']['held_out']['n_reference_invoices_with_placed'] > 0
        with open(glob.glob(str(root / 'validation_gof_*'
                                / 'results.csv'))[0]) as f:
            rows = list(csv.DictReader(f))
        assert {r['design'] for r in rows} == {'in_sample', 'held_out'}
        assert held['protocol'].pop('workers') == workers
        texts.append(_canonical(s, workers))
    assert texts[0] == texts[1]
