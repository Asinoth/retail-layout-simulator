"""The UCI calendar profile (``experiments.profile_uci``) and its macros."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import make_results_macros as M  # noqa: E402
from experiments.profile_uci import calendar_profile  # noqa: E402


def _rows():
    # Four invoices: Fri 2010-10-01 (two lines), Sat 2010-10-02, Mon
    # 2010-11-01 and Tue 2011-03-01. Revenue 10+5, 7, 3, 25.
    return pd.DataFrame({
        'invoice_id': ['A', 'A', 'B', 'C', 'D'],
        'timestamp': pd.to_datetime(['2010-10-01 09:00', '2010-10-01 09:05',
                                     '2010-10-02 10:00', '2010-11-01 11:00',
                                     '2011-03-01 12:00']),
        'quantity': [1, 1, 1, 1, 5],
        'unit_price': [10.0, 5.0, 7.0, 3.0, 5.0],
    })


def test_weekday_counts_are_invoices_and_trading_days():
    p = calendar_profile(_rows())
    sat = p['weekday']['Saturday']
    assert sat['invoices'] == 1 and sat['trading_days'] == 1
    assert p['weekday']['Friday']['invoices'] == 1       # two lines, one invoice
    assert p['span']['invoices'] == 4
    # 2010-10-01 .. 2011-03-01 inclusive: 152 days, 22 Saturdays.
    assert p['span']['days'] == 152
    assert sat['days_in_span'] == 22
    assert abs(sat['invoice_share'] - 0.25) < 1e-12


def test_fourth_quarter_shares():
    p = calendar_profile(_rows())
    q4 = p['fourth_quarter']
    assert abs(q4['revenue_share'] - (15 + 7 + 3) / 50) < 1e-12
    assert q4['days'] == 92                  # October-December 2010
    assert abs(q4['day_share'] - 92 / 152) < 1e-12


def test_profile_record_passes_the_macro_check_and_formats():
    s = {**calendar_profile(_rows()),
         'data': {'sheets': ['Year 2009-2010', 'Year 2010-2011']}}
    assert M._profile_ok(s)
    assert not M._profile_ok({**s, 'data': {'sheets': ['Year 2010-2011']}})
