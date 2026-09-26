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
    periods. The cut is made on the sheet's RAW rows, before the adapter
    cleans them: the adapter removes each purchase together with the
    cancellation that reverses it, and that pairing looks forward in time,
    so cleaning the whole sheet first would let a cancellation dated in the
    current period remove a purchase from the prior one -- information a
    retailer at the cut did not have. The prior period therefore holds
    what was visible at the cut; the purchases a later cancellation would
    have reversed are counted in its record (``reversals_across_cut``).

``period_record`` reports what each period holds -- sheet, rows, date
range, invoice count, the workbook's path relative to the repository, its
SHA-256 and size, the adapter and reader versions, the adapter's cleaning
counts and the anonymous invoices' share -- for the runners' summaries
and sidecars. The source fields carry the names Figure C's provenance
record uses (``dataset_provenance``), so every family identifies its data
the same way.

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
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Optional, Tuple

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import dataset_adapters as DA          # noqa: E402
import dataset_calibration as DC       # noqa: E402
import dataset_paths                   # noqa: E402
import dataset_provenance              # noqa: E402
from dataset_schema import SCHEMA_VERSION  # noqa: E402
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
    # The adapted rows the period was calibrated from (``period_frame``).
    frame: Any = field(default=None, repr=False, compare=False)


# (normalised workbook path, period) -> _Period. One process reads each
# sheet once, however many times a runner asks for the same period.
_CACHE: Dict[Tuple[str, str], _Period] = {}

# normalised workbook path -> its source fields (``_source_fields``); the
# workbook is hashed once per process.
_SOURCE_CACHE: Dict[str, Dict[str, Any]] = {}


def clear_cache() -> None:
    """Forget every period calibrated in this process."""
    _CACHE.clear()
    _SOURCE_CACHE.clear()


def _source_fields(path: str) -> Dict[str, Any]:
    """The workbook's SHA-256 and size and the versions of the code that
    read it, under the field names of ``dataset_provenance.ProvenanceRecord``
    (what Figure C stamps), plus the reader's version."""
    key = _key_path(path)
    hit = _SOURCE_CACHE.get(key)
    if hit is None:
        hit = {**dataset_provenance.source_digest(path),
               'adapter_name': DA.OnlineRetailIIAdapter.name,
               'adapter_version': DA.OnlineRetailIIAdapter.version,
               'reader_version': DA.READER_VERSION,
               'schema_version': SCHEMA_VERSION}
        _SOURCE_CACHE[key] = hit
    return dict(hit)


def _key_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _check_period(period: str) -> None:
    if period not in PERIODS:
        raise ValueError(f"period must be one of {PERIODS}, got {period!r}")


def _read_sheet(path: str, sheet: str) -> pd.DataFrame:
    """One sheet, every row the reader returns."""
    df, _ = DA.read_excel_sheets(path, [sheet])
    return df


def _adapt(raw: pd.DataFrame):
    """Raw rows through the UCI adapter: ``(normalised frame, the adapter's
    cleaning counts)``. The frame keeps the raw rows' index labels."""
    norm, report = DA.OnlineRetailIIAdapter().adapt(raw)
    return norm, dict(report.extra.get('cleaning') or {})


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
        raw = _read_sheet(path, sheet)
        rows_in = int(len(raw))
        norm, cleaning = _adapt(raw)
        if norm.empty:
            raise ValueError(f"sheet {sheet!r} of {path} holds no usable "
                             "invoice lines")
        kept = norm.reset_index(drop=True)
        extra: Dict[str, Any] = {}
    else:
        if len(sheets) < 2:
            raise ValueError(f"{path} has one sheet; the prior period is "
                             "the first of at least two")
        current = _load(path, 'current')
        sheet = sheets[0]
        raw = _read_sheet(path, sheet)
        rows_in = int(len(raw))
        # Rows dated strictly before the current period's first calendar
        # day: the sheets overlap, and an overlap day's invoices belong to
        # the current period. The cut comes BEFORE cleaning, so that no
        # cancellation dated on or after it can pair with, and remove, a
        # purchase before it (the reversal pairing looks forward in time).
        cut = current.first_timestamp.normalize()
        before = (DA.OnlineRetailIIAdapter.raw_timestamps(raw)
                  < cut).to_numpy()
        norm, cleaning = _adapt(raw[before])
        # The whole sheet is cleaned as well, only to count what cleaning
        # it before cutting took from the prior period: purchases before
        # the cut that a cancellation on or after it reverses. They stay
        # in the prior period, which is what was visible at the cut.
        whole, _ = _adapt(raw)
        across = norm.loc[norm.index.difference(whole.index)]
        # Whole invoices before the cut. Online Retail II dates every line
        # of an invoice alike, so after the row cut this is a guard that
        # removes nothing; it keeps the rule whole-invoice on any sheet.
        inv_last = (pd.to_datetime(norm['timestamp'])
                    .groupby(norm['invoice_id']).transform('max'))
        kept = norm[(inv_last < cut).to_numpy()].reset_index(drop=True)
        if kept.empty:
            raise ValueError(f"no invoice of sheet {sheet!r} is dated "
                             f"before {cut.date().isoformat()}, where the "
                             "current period starts")
        ids = frozenset(kept['invoice_id'].astype(str))
        qty = pd.to_numeric(across['quantity'], errors='coerce')
        price = pd.to_numeric(across['unit_price'], errors='coerce')
        extra = {
            'cut_before': cut.date().isoformat(),
            'cut_applied_to': 'raw rows, before the adapter cleans them',
            'rows_before_cut': int(before.sum()),
            # Invoices of the cleaned sheet with a line dated on or after
            # the cut.
            'n_invoices_dropped_at_cut': int(
                (pd.to_datetime(whole['timestamp'])
                 .groupby(whole['invoice_id']).max() >= cut).sum()),
            # Zero whenever invoice numbers are not reused across periods;
            # recorded so the disjointness is visible, not assumed.
            'n_invoices_shared_with_current': len(ids & current.invoice_ids),
            # Kept purchases that a cancellation dated on or after the cut
            # reverses (on Online Retail II: 102 lines on 46 invoices,
            # GBP 19,287.52).
            'reversals_across_cut': {
                'purchase_lines': int(len(across)),
                'invoices': int(across['invoice_id'].nunique()),
                'units': float(qty.sum()),
                'revenue': float((qty * price).sum())},
        }

    params = DC.calibrate_transactional(kept, currency=CURRENCY)
    record = {'period': period,
              'workbook': dataset_paths.repo_relative(path),
              # SHA-256, size, adapter / reader / schema versions (R51).
              **_source_fields(path),
              'sheet': sheet,
              # Rows the reader returned for the sheet: nothing is sampled.
              # The current period hands all of them to the adapter, the
              # prior period those before the cut (``rows_before_cut``).
              'rows_in': rows_in,
              'rows_adapted': int(len(norm)),
              'rows_calibrated': int(len(kept)),
              # What the adapter removed from the rows it received:
              # cancellation lines and the purchase lines they reverse (R27).
              'cleaning': cleaning,
              # Anonymous invoices (no customer id) are kept; their share.
              # The live store never calibrates without them, and says
              # so in the field the headline validators read (it is
              # True only in a --exclude-anonymous sensitivity run).
              'exclude_anonymous': False,
              'n_anonymous_invoices': int(params.n_anonymous_invoices),
              'anonymous_invoice_frac': float(params.anonymous_invoice_frac),
              'anonymous_revenue_frac': float(params.anonymous_revenue_frac),
              # The arrival rates the calibration gives the period, in the
              # Monte Carlo engine's units (per open hour, calendar-day
              # basis): buyers from the invoices, visitors from the buyers
              # at the assumed conversion. The live runs are driven at
              # their protocol's door rate instead, which the paper states
              # beside these (review R04).
              'arrivals_per_hour': float(params.arrivals_per_hour),
              'visitors_per_hour': float(params.visitors_per_hour),
              'assumed_conversion_rate':
                  float(params.assumed_conversion_rate),
              **_describe(kept),
              **extra}
    out = _Period(params=params, record=record,
                  invoice_ids=frozenset(kept['invoice_id'].astype(str)),
                  first_timestamp=pd.to_datetime(kept['timestamp']).min(),
                  frame=kept)
    _CACHE[key] = out
    return out


def calibrate_live_store(retail_path: Optional[str] = None,
                         period: str = 'current') -> DC.CalibratedParams:
    """Calibrated parameters of the live store for ``period``.

    The workbook is ``dataset_paths.uci_workbook(retail_path)``. The same
    object is returned for every call with the same workbook and period in
    one process; callers treat it as read-only."""
    return _load(retail_path, period).params


def period_frame(retail_path: Optional[str] = None,
                 period: str = 'current') -> pd.DataFrame:
    """The adapted invoice lines ``period`` is calibrated from -- the
    whole last sheet, or the first sheet's invoices before the current
    period -- for a runner that calibrates the same period with other
    options (``run_heldout_transfer``: Figure C's assumed conversion, or
    without the anonymous invoices). The cached frame is returned; callers
    treat it as read-only."""
    return _load(retail_path, period).frame


def period_record(retail_path: Optional[str] = None,
                  period: str = 'current') -> Dict[str, Any]:
    """What ``period`` holds, for a run's summary or sidecar.

    ``workbook`` (relative to the repository root when the file is inside
    it), ``sheet``, ``rows_in`` (the sheet's rows as the reader returns
    them), ``rows_adapted`` (after the adapter's cleaning of the rows it
    received: all of them for the current period, those before the cut for
    the prior), ``rows_calibrated`` (after the whole-invoice cut; equal to
    ``rows_adapted`` on Online Retail II, whose invoices carry one
    timestamp), the first and last date and timestamp, and
    ``n_invoices``. The source: ``source_sha256`` and ``source_bytes`` of
    the workbook, ``adapter_name`` / ``adapter_version`` /
    ``reader_version`` / ``schema_version``, and the adapter's ``cleaning``
    counts for the rows it received (cancellation lines and the purchase
    lines they reverse). ``exclude_anonymous`` (always False: the live
    store keeps the anonymous invoices), ``n_anonymous_invoices`` and
    the anonymous invoices' shares of invoices and of line revenue. The prior period adds
    ``cut_before`` and ``cut_applied_to``, ``rows_before_cut`` (raw rows
    handed to the adapter), the invoices the cut removed, the invoices it
    shares with the current period, and ``reversals_across_cut``: the
    purchases before the cut that a cancellation on or after it reverses,
    kept because the cancellation was not visible at the cut (lines,
    invoices, units, revenue). The calibrated ``arrivals_per_hour`` (buyers)
    and ``visitors_per_hour`` with the ``assumed_conversion_rate`` between
    them. Also ``list_law``, how the store's shoppers form their lists
    (``LIST_LAW``), and how the store is laid out from the period:
    ``max_items_per_category`` (the top products of each category by
    invoice count) and ``naive_layout``. Calibrates the period if this
    process has not yet."""
    record = dict(_load(retail_path, period).record)
    record['list_law'] = LIST_LAW
    record['max_items_per_category'] = MAX_ITEMS_PER_CATEGORY
    record['naive_layout'] = NAIVE_LAYOUT
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
