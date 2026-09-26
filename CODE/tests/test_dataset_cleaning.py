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


# --- reversal pairs and anonymous invoices (review R27) --------------------

def _reversal_rows():
    """Purchases and cancellations exercising each matching rule."""
    return [
        # Customer 1 buys 12 of A twice and cancels one order of 12: only
        # the more recent purchase before the cancellation goes.
        ['600001', 'A1', 'ALPHA MUG', 12, '2011-01-03 09:00', 2.0, 1, 'UK'],
        ['600002', 'A1', 'ALPHA MUG', 12, '2011-01-04 09:00', 2.0, 1, 'UK'],
        ['C600003', 'A1', 'ALPHA MUG', -12, '2011-01-05 09:00', 2.0, 1, 'UK'],
        ['600009', 'A1', 'ALPHA MUG', 12, '2011-01-06 09:00', 2.0, 1, 'UK'],
        # Customer 2: a cancellation dated before the purchase and a
        # partial cancellation (-2 of 6) match nothing.
        ['600004', 'B2', 'BETA CARD', 6, '2011-01-03 10:00', 1.0, 2, 'UK'],
        ['C600005', 'B2', 'BETA CARD', -6, '2011-01-02 10:00', 1.0, 2, 'UK'],
        ['C600006', 'B2', 'BETA CARD', -2, '2011-01-04 10:00', 1.0, 2, 'UK'],
        # Another customer's cancellation does not touch customer 2's line;
        # a same-minute cancellation matches.
        ['C600007', 'B2', 'BETA CARD', -6, '2011-01-04 11:00', 1.0, 3, 'UK'],
        ['600010', 'D4', 'DELTA BAG', 80995, '2011-01-07 09:15', 2.08, 3, 'UK'],
        ['C600011', 'D4', 'DELTA BAG', -80995, '2011-01-07 09:15', 2.08, 3, 'UK'],
        # An anonymous cancellation cannot be tied to a purchase.
        ['600012', 'E5', 'EPSILON TIN', 3, '2011-01-08 09:00', 4.0, np.nan, 'UK'],
        ['C600013', 'E5', 'EPSILON TIN', -3, '2011-01-08 09:30', 4.0, np.nan, 'UK'],
    ]


def test_cancellations_take_the_purchases_they_reverse():
    norm, report = OnlineRetailIIAdapter().adapt(_raw(_reversal_rows()))
    kept = set(zip(norm['invoice_id'], norm['product_id']))
    # Removed: the most recent earlier order of 12 (not the first, not the
    # later one) and the same-minute bulk line.
    assert ('600002', 'A1') not in kept and ('600010', 'D4') not in kept
    assert {('600001', 'A1'), ('600009', 'A1')} <= kept
    # Kept: a line only partially cancelled, cancelled before it was
    # bought, or cancelled by someone else; an anonymous purchase.
    assert ('600004', 'B2') in kept and ('600012', 'E5') in kept
    assert not norm['invoice_id'].str.startswith('C').any()
    c = report.extra['cleaning']
    assert c['cancellation_lines'] == 6
    assert c['reversal_pairs'] == c['reversed_purchase_lines_removed'] == 2
    assert c['cancellation_lines_anonymous'] == 1
    assert c['cancellation_lines_unmatched'] == 3
    assert c['reversed_units_removed'] == 12 + 80995
    assert c['reversed_revenue_removed'] == pytest.approx(24.0 + 80995 * 2.08)
    assert OnlineRetailIIAdapter.version == '1.2'


def test_reversal_matching_needs_a_customer_column():
    rows = [r[:6] for r in _reversal_rows()]
    df = pd.DataFrame(rows, columns=_COLS[:6])
    norm, report = OnlineRetailIIAdapter().adapt(df)
    c = report.extra['cleaning']
    assert c['reversal_pairs'] == 0
    assert c['cancellation_lines_anonymous'] == c['cancellation_lines'] == 6
    assert ('600002', 'A1') in set(zip(norm['invoice_id'], norm['product_id']))


def _anon_rows():
    rows = _base_rows()                       # identified, 6 invoices
    for i in range(3):                        # anonymous, larger baskets
        for code, desc, price in (('85123A', 'WHITE HANGING HEART', 2.55),
                                  ('22423', 'REGENCY CAKESTAND', 12.75),
                                  ('21212', 'PACK OF CAKE CASES', 0.55)):
            rows.append([f'537{i:03d}', code, desc, 4,
                         f'2010-12-02 10:{10 + i}', price, np.nan, 'UK'])
    return rows


def test_anonymous_invoices_are_flagged_and_disclosed():
    from dataset_calibration import anonymous_placed_shares
    norm, _ = OnlineRetailIIAdapter().adapt(_raw(_anon_rows()))
    params = calibrate_transactional(norm, currency='GBP')
    assert params.customer_id_available
    assert params.n_anonymous_invoices == 3
    assert params.anonymous_invoice_frac == pytest.approx(3 / 9)
    assert params.invoice_anonymous.size == params.basket_distinct_sizes.size
    # The flag lines up with the per-invoice samples: anonymous invoices
    # are the three-product ones.
    assert (params.basket_distinct_sizes[params.invoice_anonymous] == 3).all()
    assert (params.basket_distinct_sizes[~params.invoice_anonymous] == 2).all()
    line = norm['quantity'] * norm['unit_price']
    anon = norm['customer_id'].astype(str) == 'nan'
    assert params.anonymous_revenue_frac == pytest.approx(
        line[anon].sum() / line.sum())
    assert params.calibration_extra['anonymous_invoices'] == 'included'

    shares = anonymous_placed_shares(params, ['85123A', '21212'])
    assert shares['n_with_placed'] == 9
    assert shares['n_anonymous_with_placed'] == 3
    assert shares['invoice_share'] == pytest.approx(3 / 9)
    # Stocked purchases: 2 per anonymous invoice, 1 per identified one.
    assert shares['purchase_share'] == pytest.approx(6 / 12)
    assert shares['mean_size_anonymous'] == 2.0
    assert shares['mean_size_identified'] == 1.0


def test_seeding_with_the_shop_records_the_anonymous_list_shares():
    from experiments._common import build_headless_shop_from_calibration
    norm, _ = OnlineRetailIIAdapter().adapt(_raw(_anon_rows()))
    params = calibrate_transactional(norm, currency='GBP')
    shop = build_headless_shop_from_calibration(params, naive=True)
    cal = shop.customer_simulation.analytics['calibration']
    assert cal['n_invoices_with_placed'] == 9
    assert cal['anonymous_list_share'] == pytest.approx(3 / 9)
    # Items on the stored lists: 3 per anonymous invoice, 2 per identified.
    assert cal['anonymous_list_item_share'] == pytest.approx(9 / 21)


def test_calibration_without_anonymous_invoices():
    norm, _ = OnlineRetailIIAdapter().adapt(_raw(_anon_rows()))
    full = calibrate_transactional(norm, currency='GBP')
    ident = calibrate_transactional(norm, currency='GBP',
                                    exclude_anonymous=True)
    assert ident.n_invoices == full.n_invoices - 3
    assert ident.n_anonymous_invoices == 0
    assert not ident.invoice_anonymous.any()
    assert '21212' not in ident.item_visit_counts
    extra = ident.calibration_extra
    assert extra['anonymous_invoices'] == 'excluded'
    assert extra['n_anonymous_invoices_excluded'] == 3
    assert extra['n_anonymous_rows_excluded'] == 9
    assert ident.n_unique_customers == full.n_unique_customers
    # No customer column: anonymity is unknown, and cannot be excluded.
    bare = norm.drop(columns=['customer_id'])
    unknown = calibrate_transactional(bare, currency='GBP')
    assert not unknown.customer_id_available
    assert unknown.calibration_extra['anonymous_invoices'] == 'unknown'
    with pytest.raises(ValueError):
        calibrate_transactional(bare, currency='GBP', exclude_anonymous=True)


def test_legacy_cleaning_keeps_the_reversed_purchases():
    """``match_reversals=False`` is adapter 1.1's rule, kept only to
    measure what results made under it carried: the cancellation lines
    go, the purchases they reverse stay, and the report says so."""
    new, rep_new = OnlineRetailIIAdapter().adapt(_raw(_reversal_rows()))
    old, rep_old = OnlineRetailIIAdapter(match_reversals=False).adapt(
        _raw(_reversal_rows()))
    assert rep_new.extra['cleaning']['reversal_matching'] is True
    c = rep_old.extra['cleaning']
    assert c['reversal_matching'] is False
    assert c['reversal_pairs'] == c['reversed_purchase_lines_removed'] == 0
    assert c['cancellation_lines'] == 6
    assert len(old) == len(new) + 2
    kept = set(zip(old['invoice_id'], old['product_id']))
    assert {('600002', 'A1'), ('600010', 'D4')} <= kept
    assert not old['invoice_id'].str.startswith('C').any()
