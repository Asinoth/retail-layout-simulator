"""The store the live diagnostics run on, and the perimeter ratio's
geometric null.

The workbook tests read Online Retail II in full, so they are skipped when
``dataset_paths`` cannot find it. The workbook's sheets are read through a
spy on the reader, which is how the tests see that the current period is
every row the reader returns and that each sheet is read once however many
times a period is asked for.
"""
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
           'records': records}
    LS.clear_cache()


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
    assert pri['rows_calibrated'] < pri['rows_adapted']


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
