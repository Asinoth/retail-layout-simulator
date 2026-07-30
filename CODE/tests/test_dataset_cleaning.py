"""UCI Online Retail II cleaning regression tests (audit R7.1/R7.2).

Guards the data-quality fixes: non-merchandise stock codes (postage,
fees, manual adjustments, gift vouchers) and cancellation invoices are
dropped, and anonymous customer IDs do not corrupt the return rate.
"""

import numpy as np
import pandas as pd

from dataset_adapters import OnlineRetailIIAdapter
from dataset_calibration import calibrate_transactional

_COLS = ['Invoice', 'StockCode', 'Description', 'Quantity', 'InvoiceDate',
         'Price', 'Customer ID', 'Country']


def _raw(rows):
    return pd.DataFrame(rows, columns=_COLS)


def _base_rows():
    # A handful of real product lines across two invoices + customers.
    rows = []
    for i in range(6):
        rows.append([f'53640{i}', '85123A', 'WHITE HANGING HEART', 6,
                     f'2010-12-01 08:{20+i}', 2.55, 17850 + (i % 3),
                     'United Kingdom'])
        rows.append([f'53640{i}', '22423', 'REGENCY CAKESTAND', 3,
                     f'2010-12-01 08:{20+i}', 12.75, 17850 + (i % 3),
                     'United Kingdom'])
    return rows


def test_nonproduct_and_cancellations_dropped():
    rows = _base_rows() + [
        ['536500', 'POST', 'POSTAGE', 1, '2010-12-01 09:00', 18.0, 17850, 'UK'],
        ['536501', 'DOT', 'DOTCOM POSTAGE', 1, '2010-12-01 09:01', 20.0, 17851, 'UK'],
        ['536502', 'M', 'Manual', 1, '2010-12-01 09:02', 5.0, 17852, 'UK'],
        ['536503', 'BANK CHARGES', 'Bank Charges', 1, '2010-12-01 09:03', 15.0, 17850, 'UK'],
        ['536504', 'gift_0001_20', 'Gift Voucher', 1, '2010-12-01 09:04', 20.0, 17851, 'UK'],
        ['C536600', '85123A', 'WHITE HANGING HEART', -6, '2010-12-01 10:00', 2.55, 17850, 'UK'],
    ]
    norm, _ = OnlineRetailIIAdapter().adapt(_raw(rows))
    codes = set(norm['product_id'].astype(str).str.upper())
    assert codes <= {'85123A', '22423'}, f"non-products leaked: {codes}"
    assert not norm['invoice_id'].str.upper().str.startswith('C').any()


def test_anonymous_ids_excluded_from_return_rate():
    # Two known repeat customers + a pile of anonymous invoices that would,
    # if lumped into one "nan" customer, distort the return rate.
    rows = _base_rows()
    for i in range(8):
        rows.append([f'537{i:03d}', '85123A', 'WHITE HANGING HEART', 2,
                     f'2010-12-02 10:{10+i}', 2.55, np.nan, 'UK'])
    norm, _ = OnlineRetailIIAdapter().adapt(_raw(rows))
    params = calibrate_transactional(norm, currency='GBP')
    # Return rate is a proportion in [0,1]; anonymous invoices excluded so
    # n_unique_customers counts only the real (known-ID) customers (<= 3).
    assert 0.0 <= params.return_customer_rate <= 1.0
    assert params.n_unique_customers <= 3
    assert params.basket_distinct_median >= 1
