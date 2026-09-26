"""The real-data headline runs say whether anonymous invoices were in.

``--exclude-anonymous`` calibrates Figure C's store without the invoices
that carry no customer id -- a sensitivity analysis. The macro generator
takes, per family, the newest valid run whose directory starts with the
family's prefix, so a sensitivity run must never land under a headline
prefix, and every headline artifact must carry ``exclude_anonymous`` (False)
for a validator to check. These tests run Figure C and the budget sweep end
to end on a small synthetic store with a customer column, the workbook
reader replaced by the frame:

  * Figure C writes ``summary.data.exclude_anonymous = False`` and its
    adapter version under ``real_data_uci_*``;
  * the budget sweep does the same under ``real_data_budget_*``, and an
    ``--exclude-anonymous`` sweep goes to ``noanon_real_data_budget_*``
    with an ``_noanon`` experiment name and ``exclude_anonymous = True``.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

from dataset_adapters import OnlineRetailIIAdapter
from experiments import run_real_data_budget as RDB
from experiments import run_real_data_example as RDE


def _frame(n_invoices=400, seed=0):
    """Invoices over four categories of four products, every fifth one
    anonymous."""
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
                         int(rng.integers(1, 4)), ts, price, cat,
                         np.nan if k % 5 == 0 else f'C{k % 41:03d}'))
    return pd.DataFrame(rows, columns=['invoice_id', 'product_id',
                                       'product_name', 'quantity',
                                       'timestamp', 'unit_price', 'category',
                                       'customer_id'])


@pytest.fixture
def fake_workbook(monkeypatch, tmp_path):
    """Serve the synthetic frame wherever the runners read the workbook."""
    frame = _frame()
    book = tmp_path / 'book.xlsx'
    book.write_bytes(b'stand-in workbook')

    class _Report:
        rows_in = rows_kept = len(frame)
        extra = {'currency': 'GBP',
                 'cleaning': {'reversal_pairs': 0, 'reversal_matching': True}}

    def _resolve(p, args):
        args.retail_path = str(book)

    for mod in (RDE, RDB):
        monkeypatch.setattr(mod, 'resolve_workbook', _resolve)
    monkeypatch.setattr(RDE, 'read_workbook',
                        lambda args: (frame.copy(), {'rows': len(frame)}))
    monkeypatch.setattr(RDE, 'adapt_rows',
                        lambda df, match_reversals=True: (df.copy(),
                                                          _Report()))
    return frame


TINY = ['--max-items-per-category', '4', '--mc-iters', '20', '--mc-days',
        '3', '--n-gens', '2', '--pop-size', '4', '--n-mc-replicates', '3']


def _one_run(root, prefix):
    (run,) = glob.glob(os.path.join(str(root), '*'))
    assert os.path.basename(run).startswith(prefix), run
    with open(os.path.join(run, 'summary.json'), encoding='utf-8') as f:
        return json.load(f)


def test_figure_c_summary_carries_the_data_block(fake_workbook, monkeypatch,
                                                  tmp_path):
    out = tmp_path / 'figc'
    monkeypatch.setattr(sys, 'argv', ['run_real_data_example', *TINY,
                                      '--out-root', str(out)])
    assert RDE.main() == 0
    s = _one_run(out, 'real_data_uci_')
    assert s['experiment'] == 'real_data_uci_figure_c'
    assert s['data']['exclude_anonymous'] is False
    assert s['data']['adapter_version'] == OnlineRetailIIAdapter.version


@pytest.mark.parametrize('flag, prefix, experiment, excluded', [
    ((), 'real_data_budget_', 'real_data_uci_budget', False),
    (('--exclude-anonymous',), 'noanon_real_data_budget_',
     'real_data_uci_budget_noanon', True),
])
def test_budget_sweep_records_and_separates_the_sensitivity_run(
        fake_workbook, tmp_path, flag, prefix, experiment, excluded):
    out = tmp_path / ('budget' + ''.join(flag))
    assert RDB.main([*TINY, '--budget-multipliers', '1',
                     '--n-search-seeds', '2', '--sa-k-move', '2',
                     '--out-root', str(out), *flag]) == 0
    s = _one_run(out, prefix)
    assert s['experiment'] == experiment
    assert s['data']['exclude_anonymous'] is excluded
    assert s['design']['exclude_anonymous'] is excluded
    assert s['data']['adapter_version'] == OnlineRetailIIAdapter.version
    with open(glob.glob(os.path.join(str(out), '*', 'sidecar.json'))[0],
              encoding='utf-8') as f:
        side = json.load(f)
    anon = side['provenance']['extra']['anonymous']
    assert anon['calibration'] == ('excluded' if excluded else 'included')
    assert side['provenance']['extra']['exclude_anonymous'] is excluded


def test_no_sensitivity_run_matches_a_headline_family():
    """Every real-data family, the budget sweep included: a no-anonymous
    run's directory starts with no headline family's prefix."""
    import argparse
    on = argparse.Namespace(exclude_anonymous=True)
    families = ('real_data_uci', 'real_data_seeds', 'real_data_budget',
                'input_uncertainty', 'heldout_transfer')
    for base in families:
        run_dir = RDE.run_dir_prefix(base, on) + '_20260925-120000'
        assert not any(run_dir.startswith(f + '_') for f in families)
