"""The store the live diagnostics run on, and the perimeter ratio's
geometric null.

The workbook tests read Online Retail II in full, so they are skipped when
``dataset_paths`` cannot find it. The workbook's sheets are read through a
spy on the reader, which is how the tests see that the current period is
every row the reader returns and that each sheet is read once however many
times a period is asked for.
"""
import os

import numpy as np
import pytest

import dataset_adapters as DA
import dataset_paths
from experiments import _live_store as LS
from experiments.run_abm_diagnostics import (perimeter_ratio,
                                             perimeter_ratio_geometric_null,
                                             walkable_heat_cells)

GRID_RES = 0.25      # the simulator's pathfinding step, metres
HEAT_RES = 20        # the simulator's heat-map cells per metre


def _floor(W, H):
    blocked = np.zeros((int(W / GRID_RES) + 1, int(H / GRID_RES) + 1),
                       dtype=bool)
    heat_shape = (int(W * HEAT_RES) + 1, int(H * HEAT_RES) + 1)
    return blocked, heat_shape


def test_geometric_null_is_one_when_nothing_is_blocked():
    W, H = 6.0, 4.0
    blocked, heat_shape = _floor(W, H)
    walkable = walkable_heat_cells(blocked, GRID_RES, heat_shape, HEAT_RES)
    assert walkable.shape == heat_shape and walkable.all()
    for band in (0.5, 1.0, 1.5):
        assert perimeter_ratio_geometric_null(walkable, W, H, band) == 1.0
    # The null is perimeter_ratio itself, applied to the uniform map.
    assert perimeter_ratio(walkable.astype(float), W, H, 1.0)[
        'perimeter_interior_ratio'] == 1.0


def test_geometric_null_is_the_ratio_of_walkable_shares():
    W, H, band = 6.0, 4.0, 1.0
    blocked, heat_shape = _floor(W, H)
    # A run of shelving through the middle of the floor.
    blocked[int(2.0 / GRID_RES):int(4.0 / GRID_RES) + 1,
            int(1.5 / GRID_RES):int(2.5 / GRID_RES) + 1] = True
    walkable = walkable_heat_cells(blocked, GRID_RES, heat_shape, HEAT_RES)
    assert 0 < walkable.sum() < walkable.size

    xs = np.linspace(0, W, heat_shape[0])
    ys = np.linspace(0, H, heat_shape[1])
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    perim = (X < band) | (X > W - band) | (Y < band) | (Y > H - band)
    expected = walkable[perim].mean() / walkable[~perim].mean()
    null = perimeter_ratio_geometric_null(walkable, W, H, band)
    assert null == pytest.approx(expected, rel=1e-12)
    # Shelving in the interior alone raises the ratio above 1 with traffic
    # spread evenly over every spot an agent can stand on.
    assert null > 1.0


@pytest.fixture(scope='module')
def store():
    try:
        path = dataset_paths.uci_workbook()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    sheets = [name for name, _ in DA.list_excel_sheets(path)]
    reads = []
    real = DA.read_excel_sheets

    def spy(filename, sheet_names):
        df, notes = real(filename, sheet_names)
        reads.append((tuple(sheet_names), len(df)))
        return df, notes

    LS.clear_cache()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(DA, 'read_excel_sheets', spy)
        first = {p: LS.calibrate_live_store(path, p) for p in LS.PERIODS}
        again = {p: LS.calibrate_live_store(path, p) for p in LS.PERIODS}
        records = {p: LS.period_record(path, p) for p in LS.PERIODS}
    yield {'sheets': sheets, 'reads': reads, 'first': first, 'again': again,
           'records': records, 'path': path}
    LS.clear_cache()


def test_period_record_identifies_the_source(store):
    """Every live family records the workbook's SHA-256, the adapter and
    reader versions and what the cleaning removed (review R51, R27)."""
    import hashlib
    with open(store['path'], 'rb') as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    for period in LS.PERIODS:
        rec = store['records'][period]
        assert rec['source_sha256'] == sha
        assert rec['source_bytes'] == os.path.getsize(store['path'])
        assert rec['adapter_name'] == DA.OnlineRetailIIAdapter.name
        assert rec['adapter_version'] == DA.OnlineRetailIIAdapter.version
        assert rec['reader_version'] == DA.READER_VERSION
        c = rec['cleaning']
        assert c['reversal_pairs'] > 0
        assert c['cancellation_lines'] == (
            c['reversal_pairs'] + c['cancellation_lines_anonymous']
            + c['cancellation_lines_unmatched'])
        assert 0.0 < rec['anonymous_invoice_frac'] < 1.0
        assert rec['n_anonymous_invoices'] > 0
        # The live store keeps them, and says so where a headline
        # validator looks.
        assert rec['exclude_anonymous'] is False
        # The frame a runner may recalibrate is the one calibrated here.
        assert len(LS.period_frame(store['path'], period)) == \
            rec['rows_calibrated']
        # The calibrated rates the paper sets the live load against, and
        # the assortment rule (review R04, R34).
        params = store['first'][period]
        assert rec['arrivals_per_hour'] == params.arrivals_per_hour > 0
        assert rec['visitors_per_hour'] == pytest.approx(
            rec['arrivals_per_hour'] / rec['assumed_conversion_rate'])
        assert rec['max_items_per_category'] == LS.MAX_ITEMS_PER_CATEGORY
        assert rec['naive_layout'] is LS.NAIVE_LAYOUT


def test_source_fields_hash_the_file(tmp_path):
    import hashlib
    book = tmp_path / 'book.xlsx'
    book.write_bytes(b'not really a workbook')
    fields = LS._source_fields(str(book))
    assert fields['source_sha256'] == hashlib.sha256(
        b'not really a workbook').hexdigest()
    assert fields['source_bytes'] == len(b'not really a workbook')
    assert fields['adapter_version'] == DA.OnlineRetailIIAdapter.version
    LS._SOURCE_CACHE.pop(LS._key_path(str(book)), None)


def test_periods_are_disjoint_and_non_empty(store):
    cur, pri = store['records']['current'], store['records']['prior']
    assert cur['n_invoices'] > 0 and pri['n_invoices'] > 0
    assert store['first']['current'].n_invoices == cur['n_invoices']
    assert store['first']['prior'].n_invoices == pri['n_invoices']
    assert pri['sheet'] == store['sheets'][0]
    assert cur['sheet'] == store['sheets'][-1]
    # The cut is the current period's first calendar day, and every prior
    # invoice falls before it.
    assert pri['cut_before'] == cur['first_date']
    assert pri['last_timestamp'] < cur['first_timestamp']
    assert pri['last_date'] < cur['first_date']
    assert pri['n_invoices_shared_with_current'] == 0
    # The cut is made on the raw rows, before cleaning; the whole-invoice
    # guard after it removes nothing (one timestamp per invoice).
    assert pri['rows_before_cut'] < pri['rows_in']
    assert pri['rows_adapted'] <= pri['rows_before_cut']
    assert pri['rows_calibrated'] == pri['rows_adapted']
    assert pri['n_invoices_dropped_at_cut'] > 0
    # Purchases a current-period cancellation reverses stay in the prior
    # period, and are counted.
    across = pri['reversals_across_cut']
    assert across['purchase_lines'] > 0 and across['revenue'] > 0
    assert 'reversals_across_cut' not in cur


def test_current_period_is_the_whole_last_sheet(store):
    cur = store['records']['current']
    last = store['sheets'][-1]
    read = [n for names, n in store['reads'] if names == (last,)]
    assert read == [cur['rows_in']]
    # No row the adapter kept is left out of the calibration.
    assert cur['rows_calibrated'] == cur['rows_adapted']
    assert 'cut_before' not in cur


def test_each_sheet_is_read_once(store):
    assert sorted(names for names, _ in store['reads']) == sorted(
        (name,) for name in {store['sheets'][0], store['sheets'][-1]})
    for period in LS.PERIODS:
        assert store['again'][period] is store['first'][period]


def test_unknown_period_is_refused():
    with pytest.raises(ValueError):
        LS.calibrate_live_store(None, 'next_year')


# --- The prior period's cut comes before cleaning --------------------------

_COLS = ['Invoice', 'StockCode', 'Description', 'Quantity', 'InvoiceDate',
         'Price', 'Customer ID', 'Country']


def _sheet_rows(dates, first_invoice):
    """Two-product invoices of three customers, one per date."""
    rows = []
    for i, d in enumerate(dates):
        inv = str(first_invoice + i)
        rows.append([inv, '85123A', 'WHITE HANGING HEART', 6, d, 2.55,
                     17850 + i % 3, 'UK'])
        rows.append([inv, '22423', 'REGENCY CAKESTAND', 2, d, 12.75,
                     17850 + i % 3, 'UK'])
    return rows


def test_a_later_cancellation_does_not_reach_into_the_prior_period(tmp_path):
    """Cleaning the whole prior sheet before the cut let a cancellation
    dated in the current period remove the purchase it reverses from the
    prior period. The cut is now made on the raw rows, so the prior period
    keeps that purchase (it was visible at the cut) and counts it, while a
    reversal inside the prior period is still removed."""
    import pandas as pd

    current = _sheet_rows([f'2010-12-0{d} 10:00' for d in (1, 2, 3, 6)],
                          700000)
    prior = _sheet_rows([f'2010-11-{d} 10:00' for d in (15, 16, 17, 18)],
                        690000) + [
        # Bought before the cut, cancelled after it.
        ['690100', '21212', 'PACK OF CAKE CASES', 24, '2010-11-25 09:00',
         0.55, 12345, 'UK'],
        ['C700100', '21212', 'PACK OF CAKE CASES', -24, '2010-12-02 09:00',
         0.55, 12345, 'UK'],
        # Bought and cancelled before the cut.
        ['690200', '22423', 'REGENCY CAKESTAND', 5, '2010-11-20 09:00',
         12.75, 12346, 'UK'],
        ['C690201', '22423', 'REGENCY CAKESTAND', -5, '2010-11-22 09:00',
         12.75, 12346, 'UK'],
        # The sheets overlap: a current-period invoice repeated here.
        *[r for r in current if r[0] == '700000'],
    ]
    sheets = {'Year A': pd.DataFrame(prior, columns=_COLS),
              'Year B': pd.DataFrame(current, columns=_COLS)}
    book = tmp_path / 'book.xlsx'
    book.write_bytes(b'stand-in workbook')

    LS.clear_cache()
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(DA, 'list_excel_sheets',
                       lambda f: [(n, len(df)) for n, df in sheets.items()])
            mp.setattr(DA, 'read_excel_sheets',
                       lambda f, names: (pd.concat(
                           [sheets[n] for n in names], ignore_index=True),
                           []))
            rec = LS.period_record(str(book), 'prior')
            frame = LS.period_frame(str(book), 'prior')
    finally:
        LS.clear_cache()

    kept = set(zip(frame['invoice_id'], frame['product_id']))
    assert ('690100', '21212') in kept           # cancelled after the cut
    assert ('690200', '22423') not in kept       # cancelled before it
    assert not frame['invoice_id'].str.startswith('C').any()
    assert '700000' not in set(frame['invoice_id'])
    assert rec['cut_before'] == '2010-12-01'
    assert rec['rows_in'] == len(prior)
    assert rec['rows_before_cut'] == len(prior) - 3   # C700100 + overlap
    assert rec['rows_calibrated'] == rec['rows_adapted'] == len(frame)
    # Only the pair inside the prior period was cleaned out.
    assert rec['cleaning']['cancellation_lines'] == 1
    assert rec['cleaning']['reversal_pairs'] == 1
    assert rec['reversals_across_cut'] == {
        'purchase_lines': 1, 'invoices': 1, 'units': 24.0,
        'revenue': pytest.approx(24 * 0.55)}
    assert rec['n_invoices_dropped_at_cut'] == 1       # the overlap invoice
    assert rec['n_invoices_shared_with_current'] == 0
