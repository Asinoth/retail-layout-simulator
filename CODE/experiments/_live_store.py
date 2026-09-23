"""The calibrated store every live diagnostic runs on.

``run_abm_diagnostics``, ``run_structural_sensitivity``,
``measure_queueing`` and ``run_validation_gof`` all measure the agent model
on one store: the UCI Online Retail II workbook, calibrated by the same
adapter and ``calibrate_transactional`` call, laid out by the same builder.
Defining it here keeps the four from drifting apart.

The workbook is read in full. Calibrating from a random sample of its rows
would split invoices apart -- sampling lines, not invoices, leaves a median
of 2 distinct products per invoice where the sheet has 15 -- and every
per-invoice law the agents and the goodness-of-fit references use would
describe the sample rather than the shop. A whole sheet calibrates in a
few seconds; reading it from the workbook is the slow part, which the
process-level cache below pays once.

Two disjoint calendar periods are available:

  * ``'current'`` -- the workbook's last sheet (Year 2010-2011 on Online
    Retail II), every row the reader returns;
  * ``'prior'`` -- the first sheet (Year 2009-2010), restricted to the
    invoices dated strictly before the first date the current period's
    invoices carry. The two sheets overlap by about nine days of the same
    invoices; the cut is computed from the data, and it is applied to
    whole invoices (an invoice is kept only if every line of it falls
    before the cut), so no invoice and no calendar day belongs to both
    periods.

``period_record`` reports what each period holds -- sheet, rows, date
range, invoice count, the workbook's path relative to the repository --
for the runners' summaries and sidecars.

``build_live_store`` lays a calibration out as the store the runners
measure. The calibration is seeded into it WITH the shop, which is what
writes ``list_length_sample``: the distinct stocked products per invoice
that the agents draw their list lengths from. A store without it would
send its agents out with the type-conditional list lengths instead, so
the builder refuses to hand one back.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Optional, Tuple

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import dataset_adapters as DA          # noqa: E402
import dataset_calibration as DC       # noqa: E402
import dataset_paths                   # noqa: E402
from experiments._common import build_headless_shop_from_calibration  # noqa: E402

PERIODS = ('current', 'prior')

CURRENCY = 'GBP'

# Layout of the live store: the top products of each category, in the
# naive (alphabetical, product-id-ordered) arrangement the GUI builds by
# default, so the live diagnostics measure the store the optimizer starts
# from.
MAX_ITEMS_PER_CATEGORY = 8
NAIVE_LAYOUT = True


@dataclass(frozen=True)
class _Period:
    params: DC.CalibratedParams
    record: Dict[str, Any]
    invoice_ids: FrozenSet[str]
    first_timestamp: pd.Timestamp


# (normalised workbook path, period) -> _Period. One process reads each
# sheet once, however many times a runner asks for the same period.
_CACHE: Dict[Tuple[str, str], _Period] = {}


def clear_cache() -> None:
    """Forget every period calibrated in this process."""
    _CACHE.clear()


def _key_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _check_period(period: str) -> None:
    if period not in PERIODS:
        raise ValueError(f"period must be one of {PERIODS}, got {period!r}")


def _read_adapted(path: str, sheet: str):
    """One sheet, every row the reader returns, through the UCI adapter."""
    df, _ = DA.read_excel_sheets(path, [sheet])
    norm, report = DA.OnlineRetailIIAdapter().adapt(df)
    return norm, int(report.rows_in)


def _describe(norm: pd.DataFrame) -> Dict[str, Any]:
    ts = pd.to_datetime(norm['timestamp'])
    first, last = ts.min(), ts.max()
    return {'first_date': first.date().isoformat(),
            'last_date': last.date().isoformat(),
            'first_timestamp': first.isoformat(),
            'last_timestamp': last.isoformat(),
            'n_invoices': int(norm['invoice_id'].nunique())}


def _load(retail_path: Optional[str], period: str) -> _Period:
    _check_period(period)
    path = dataset_paths.uci_workbook(retail_path)
    key = (_key_path(path), period)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit

    sheets = [name for name, _ in DA.list_excel_sheets(path)]
    if not sheets:
        raise ValueError(f"{path} holds no sheets")
    if period == 'current':
        sheet = sheets[-1]
        norm, rows_in = _read_adapted(path, sheet)
        if norm.empty:
            raise ValueError(f"sheet {sheet!r} of {path} holds no usable "
                             "invoice lines")
        kept = norm
        extra: Dict[str, Any] = {}
    else:
        if len(sheets) < 2:
            raise ValueError(f"{path} has one sheet; the prior period is "
                             "the first of at least two")
        current = _load(path, 'current')
        sheet = sheets[0]
        norm, rows_in = _read_adapted(path, sheet)
        # Whole invoices dated strictly before the current period's first
        # calendar day: the sheets overlap, and an overlap day's invoices
        # belong to the current period.
        cut = current.first_timestamp.normalize()
        inv_last = (pd.to_datetime(norm['timestamp'])
                    .groupby(norm['invoice_id']).transform('max'))
        kept = norm[inv_last < cut].reset_index(drop=True)
        if kept.empty:
            raise ValueError(f"no invoice of sheet {sheet!r} is dated "
                             f"before {cut.date().isoformat()}, where the "
                             "current period starts")
        ids = frozenset(kept['invoice_id'].astype(str))
        extra = {
            'cut_before': cut.date().isoformat(),
            'n_invoices_dropped_at_cut': int(
                norm['invoice_id'].nunique() - kept['invoice_id'].nunique()),
            # Zero whenever invoice numbers are not reused across periods;
            # recorded so the disjointness is visible, not assumed.
            'n_invoices_shared_with_current': len(ids & current.invoice_ids),
        }

    params = DC.calibrate_transactional(kept, currency=CURRENCY)
    record = {'period': period,
              'workbook': dataset_paths.repo_relative(path),
              'sheet': sheet,
              # Rows the reader returned for the sheet, all of which the
              # adapter received: nothing is sampled.
              'rows_in': rows_in,
              'rows_adapted': int(len(norm)),
              'rows_calibrated': int(len(kept)),
              **_describe(kept),
              **extra}
    out = _Period(params=params, record=record,
                  invoice_ids=frozenset(kept['invoice_id'].astype(str)),
                  first_timestamp=pd.to_datetime(kept['timestamp']).min())
    _CACHE[key] = out
    return out


def calibrate_live_store(retail_path: Optional[str] = None,
                         period: str = 'current') -> DC.CalibratedParams:
    """Calibrated parameters of the live store for ``period``.

    The workbook is ``dataset_paths.uci_workbook(retail_path)``. The same
    object is returned for every call with the same workbook and period in
    one process; callers treat it as read-only."""
    return _load(retail_path, period).params


def period_record(retail_path: Optional[str] = None,
                  period: str = 'current') -> Dict[str, Any]:
    """What ``period`` holds, for a run's summary or sidecar.

    ``workbook`` (relative to the repository root when the file is inside
    it), ``sheet``, ``rows_in`` (the sheet's rows as the reader returns
    them, all handed to the adapter), ``rows_adapted`` (after the adapter's
    cleaning), ``rows_calibrated`` (after the prior period's cut; equal to
    ``rows_adapted`` for the current period), the first and last date and
    timestamp, and ``n_invoices``. The prior period adds ``cut_before``,
    the invoices the cut removed and the invoices it shares with the
    current period. Also ``list_law``, how the store's shoppers form their
    lists (``LIST_LAW``). Calibrates the period if this process has not
    yet."""
    record = dict(_load(retail_path, period).record)
    record['list_law'] = LIST_LAW
    return record


# How the store's shoppers form their lists: each list is the stocked
# part of one empirical invoice of the calibration period. Recorded in
# every live run's period record so the macro validators can refuse runs
# made under an earlier list law.
LIST_LAW = 'stocked-invoice'


def build_live_store(params: DC.CalibratedParams):
    """The store a live diagnostic measures, built from ``params``.

    ``build_headless_shop_from_calibration`` lays the store out and seeds
    the calibration with the shop, which writes the invoices the agents
    draw their shopping lists from (``list_invoice_ptr`` and friends: the
    stocked part of each invoice holding a regular placed product). Only
    a seeding that knows the assortment can form them, so when the params
    carry per-invoice product sets and the seeded calibration has none,
    the store is refused rather than measured with the agents falling back
    to the uncalibrated list law."""
    shop = build_headless_shop_from_calibration(
        params, max_items_per_category=MAX_ITEMS_PER_CATEGORY,
        naive=NAIVE_LAYOUT)
    cal = shop.customer_simulation.analytics.get('calibration') or {}
    ptr = cal.get('list_invoice_ptr')
    if (getattr(params, 'has_invoice_structure', False)
            and (ptr is None or len(ptr) < 2)):
        raise RuntimeError(
            "the live store's seeded calibration has no stocked invoices "
            "(list_invoice_ptr), so its agents would not draw their lists "
            "from the invoices; the calibration must be seeded with the "
            "shop")
    return shop
