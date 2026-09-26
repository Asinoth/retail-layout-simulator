"""What the UCI workbook's calendar looks like, as a run record.

The paper describes two features of the data that no calibration consumes:
the weekday mix (the Monte Carlo engine's day-of-week pattern is an
operational weekend multiplier, not the source's own mix) and the seasonal
concentration the worked example's span averages over. This script reads
the workbook exactly as Figure C does -- the same sheets, cross-sheet
repeats dropped, the same adapter and cleaning -- and records, over the
rows the adapter keeps:

  * invoices and trading days per weekday, and how many of each weekday
    the span holds (``weekday``);
  * revenue per calendar month and its share in October-December against
    the share of the span's calendar days those months take
    (``fourth_quarter``);
  * the span itself (first and last date, days, invoices), so the record
    can be matched to the Figure C run that reads the same rows.

Nothing is fitted or searched; the run takes as long as reading the
workbook. Writes ``summary.json`` and a sidecar under ``--out-root``.

    python -m experiments.profile_uci
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from dataset_paths import repo_relative, uci_workbook  # noqa: E402
from dataset_provenance import _sha256_path  # noqa: E402
from experiments._common import (make_run_dir, provenance_snapshot,  # noqa: E402
                                 write_json, write_sidecar)
from experiments.run_real_data_example import (adapt_rows,  # noqa: E402
                                               read_workbook)

WEEKDAYS = ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
            'Saturday', 'Sunday')


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--retail-path', type=str, default=None,
                   help="The UCI Online Retail II workbook. Default: found "
                        "by dataset_paths.uci_workbook()")
    p.add_argument('--sheets', type=str,
                   default="Year 2009-2010,Year 2010-2011",
                   help="Comma-separated sheets, as Figure C reads them")
    p.add_argument('--out-root', type=str,
                   default=os.path.join(_HERE, 'results'))
    args = p.parse_args(argv)
    try:
        args.retail_path = uci_workbook(args.retail_path)
    except FileNotFoundError as exc:
        p.error(str(exc))
    return args


def calendar_profile(rows: pd.DataFrame) -> dict:
    """The weekday and seasonal profile of adapted transaction rows
    (``invoice_id``, ``timestamp``, ``quantity``, ``unit_price``). An
    invoice is dated by its first line."""
    df = rows[['invoice_id', 'timestamp', 'quantity', 'unit_price']].copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df['revenue'] = df['quantity'].astype(float) * df['unit_price'].astype(float)
    inv = df.groupby('invoice_id').agg(t=('timestamp', 'min'),
                                       revenue=('revenue', 'sum'))
    dates = inv['t'].dt.normalize()
    first, last = dates.min(), dates.max()
    span = pd.date_range(first, last, freq='D')
    n_inv = int(len(inv))

    weekday = {}
    for k, name in enumerate(WEEKDAYS):
        on = inv['t'].dt.dayofweek == k
        weekday[name] = {
            'invoices': int(on.sum()),
            'invoice_share': float(on.sum() / n_inv) if n_inv else 0.0,
            'trading_days': int(dates[on].nunique()),
            'days_in_span': int((span.dayofweek == k).sum()),
        }

    month = inv.groupby(inv['t'].dt.to_period('M'))['revenue'].sum()
    total = float(inv['revenue'].sum())
    in_q4 = np.array([p.month >= 10 for p in month.index])
    q4_days = int((span.month >= 10).sum())
    return {
        'span': {'first_date': str(first.date()), 'last_date': str(last.date()),
                 'days': int(len(span)), 'invoices': n_inv,
                 'revenue': total},
        'weekday': weekday,
        'fourth_quarter': {
            'months': 'October-December',
            'revenue_share': float(month[in_q4].sum() / total) if total else 0.0,
            'day_share': float(q4_days / len(span)),
            'days': q4_days,
        },
        'revenue_by_month': {str(p): float(v) for p, v in month.items()},
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = make_run_dir(args.out_root, 'profile_uci')
    prov = provenance_snapshot()
    t0 = time.perf_counter()
    raw, reader = read_workbook(args)
    rows, report = adapt_rows(raw)
    profile = calendar_profile(rows)
    sha, nbytes = _sha256_path(args.retail_path)
    summary = {
        **profile,
        'data': {'workbook': repo_relative(args.retail_path),
                 'source_sha256': sha, 'source_bytes': nbytes,
                 'sheets': [s.strip() for s in args.sheets.split(',')
                            if s.strip()],
                 'rows_read': reader['rows_read'],
                 'rows_after_dedup': reader['rows_after_dedup'],
                 'rows_kept': int(report.rows_kept)},
    }
    write_sidecar(out_dir, {'experiment': 'profile_uci', 'args': vars(args),
                            'wall_seconds': time.perf_counter() - t0},
                  provenance=prov)
    # Written last: its presence marks a finished run.
    write_json(out_dir, 'summary.json', summary)
    sat = profile['weekday']['Saturday']
    q4 = profile['fourth_quarter']
    print(f"{profile['span']['invoices']:,} invoices over "
          f"{profile['span']['days']} days; Saturday: {sat['invoices']} "
          f"invoices on {sat['trading_days']} of {sat['days_in_span']} "
          f"Saturdays; October-December: {q4['revenue_share']:.1%} of revenue "
          f"on {q4['day_share']:.1%} of the days. -> {out_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
