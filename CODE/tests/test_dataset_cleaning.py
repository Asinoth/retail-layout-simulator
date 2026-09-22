"""UCI Online Retail II cleaning regression tests (audit R7.1/R7.2).

Guards the data-quality fixes: non-merchandise stock codes (postage,
fees, manual adjustments, gift vouchers) and cancellation invoices are
dropped, and anonymous customer IDs do not corrupt the return rate.
"""

import numpy as np
import pandas as pd
import pytest

from dataset_adapters import OnlineRetailIIAdapter, read_excel_sheets
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


def test_stock_codes_case_normalized():
    # The same SKU under two spellings must be one product.
    rows = _base_rows() + [
        ['536700', '15056bl', 'EDWARDIAN PARASOL BLACK', 2, '2010-12-03 11:00', 5.95, 17850, 'UK'],
        ['536701', ' 15056BL', 'EDWARDIAN PARASOL BLACK', 1, '2010-12-03 11:05', 5.95, 17851, 'UK'],
    ]
    norm, _ = OnlineRetailIIAdapter().adapt(_raw(rows))
    parasol = norm[norm['product_name'].str.contains('PARASOL')]
    assert set(parasol['product_id']) == {'15056BL'}


def test_fee_descriptions_dropped_by_description():
    # Fee lines under ordinary numeric codes go; real products whose names
    # share a word with a fee ('CARRIAGE CLOCK', 'PIGGY BANK') stay.
    rows = _base_rows() + [
        ['536800', '23444', 'Next Day Carriage', 1, '2010-12-04 09:00', 15.0, 17850, 'UK'],
        ['536801', '23574', 'PACKING CHARGE', 1, '2010-12-04 09:01', 7.5, 17851, 'UK'],
        ['536802', '22016', 'Dotcomgiftshop Gift Voucher £100.00', 1, '2010-12-04 09:02', 83.33, 17852, 'UK'],
        ['536803', '21656', 'samples', 1, '2010-12-04 09:03', 1.0, 17850, 'UK'],
        ['536804', '85168B', 'BLACK BAROQUE CARRIAGE CLOCK', 1, '2010-12-04 09:04', 8.47, 17851, 'UK'],
        ['536805', '22637', 'PIGGY BANK RETROSPOT', 2, '2010-12-04 09:05', 2.55, 17852, 'UK'],
    ]
    norm, _ = OnlineRetailIIAdapter().adapt(_raw(rows))
    codes = set(norm['product_id'])
    assert not codes & {'23444', '23574', '22016', '21656'}, codes
    assert {'85168B', '22637'} <= codes


def test_category_whole_word_beats_substring_and_longest_wins():
    ad = OnlineRetailIIAdapter()
    cases = {
        # 'tin' inside 'bunting' must not shadow the 'bunting' entry.
        'PARTY BUNTING': 'Baking & Decor',
        # 'doll' inside 'dolly' must not win over the whole word 'lunch'.
        'DOLLY GIRL LUNCH BOX': 'Kitchen & Tableware',
        # Two whole-word keywords of equal length: map order decides.
        'WHITE HANGING HEART T-LIGHT HOLDER': 'Gifts & Ornaments',
        # Compounds still resolve through the substring tier.
        'REGENCY CAKESTAND 3 TIER': 'Baking & Decor',
        'ZZZ UNMATCHED THING': 'General Merchandise',
    }
    for desc, expected in cases.items():
        assert ad._infer_category(desc) == expected, desc
    vec = ad._infer_categories_vectorized(pd.Series(list(cases)))
    assert list(vec) == list(cases.values())


def test_read_excel_sheets_drops_cross_sheet_repeats(tmp_path):
    pytest.importorskip('openpyxl')
    cols = ['Invoice', 'StockCode', 'Quantity']
    a, b, c = ['1', 'X', 1], ['2', 'Y', 2], ['3', 'Z', 3]
    first = pd.DataFrame([a, b, b], columns=cols)       # b repeated in-sheet
    second = pd.DataFrame([b, c, c, a, a], columns=cols)
    path = tmp_path / 'two_sheets.xlsx'
    with pd.ExcelWriter(path) as xw:
        first.to_excel(xw, sheet_name='S1', index=False)
        second.to_excel(xw, sheet_name='S2', index=False)
    out, notes = read_excel_sheets(str(path), ['S1', 'S2'])
    rows = [tuple(r) for r in out.astype(str).itertuples(index=False)]
    # S1 is untouched; S2 loses its b (S1 holds two) and one of its two a's
    # (S1 holds one); its in-sheet repeat of c is kept.
    assert rows.count(('1', 'X', '1')) == 2
    assert rows.count(('2', 'Y', '2')) == 2
    assert rows.count(('3', 'Z', '3')) == 2
    assert len(out) == 6
    assert any('repeat rows of an earlier sheet' in n for n in notes)


def test_read_excel_sheets_matches_codes_parsed_as_numbers(tmp_path):
    # A sheet whose codes are all numeric reads them as integers; a sheet
    # that also holds an alphanumeric code keeps them as text. The shared
    # line must still be recognized as a repeat.
    pytest.importorskip('openpyxl')
    cols = ['Invoice', 'StockCode', 'Quantity']
    first = pd.DataFrame([['1', '85048', 1], ['2', '22087', 2]], columns=cols)
    second = pd.DataFrame([['1', '85048', 1], ['3', '85123A', 3]], columns=cols)
    path = tmp_path / 'typed_sheets.xlsx'
    with pd.ExcelWriter(path) as xw:
        first.to_excel(xw, sheet_name='S1', index=False)
        second.to_excel(xw, sheet_name='S2', index=False)
    out, _ = read_excel_sheets(str(path), ['S1', 'S2'])
    assert len(out) == 3
    assert sorted(out['StockCode'].astype(str)) == ['22087', '85048', '85123A']
