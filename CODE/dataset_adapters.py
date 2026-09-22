"""Adapters that map arbitrary dataset files onto the normalized schema.

Each adapter implements:

  - ``CAN_HANDLE_HINTS`` (class attr) -> list of substrings that, if present
    in the source filename or column names, make the adapter a strong
    candidate for the auto-detector.

  - ``adapt(df) -> (normalized_df, ValidationReport)`` -> returns a DataFrame
    whose columns match the schema, plus a ``ValidationReport`` describing
    which source columns mapped onto which schema fields and what was
    inferred or dropped.

The auto-detector tries every registered adapter, scores them on
``ValidationReport.is_blocking()`` and the number of ``inferred`` vs
``present`` fields, and returns the best one. Specific adapters
(``OnlineRetailIIAdapter``) always score higher than the generic
catch-all when their column signature is present.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

from dataset_schema import (
    TRANSACTIONAL_FIELDS,
    TRAJECTORY_FIELDS,
    FieldStatus,
    ValidationReport,
    SCHEMA_VERSION,
)


# Version of the readers in this module (sheet concatenation and
# cross-sheet de-duplication, separator/encoding detection, the choice of
# files inside the Omnichannel bundle). Callers stamp it into provenance
# next to the adapter version so a record identifies the whole ingestion
# path, not the adapter alone. Bump it, like an adapter's own ``version``,
# whenever the rows a source yields change.
READER_VERSION = "1.1"


# --- Helpers --------------------------------------------------------------

def _norm(s: str) -> str:
    """lowercase, strip spaces/underscores -- for fuzzy column matching."""
    return re.sub(r"[\s_\-]+", "", str(s).lower())


def _col_tokens(s) -> set:
    """Words of a column name, split on separators and camelCase
    ('agentID' -> {'agent', 'id'})."""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(s))
    return {t for t in re.split(r"[^a-z0-9]+", s.lower()) if t}


# Everyday words that must match a whole word of the column name. As raw
# substrings they bind unrelated columns: 'count' sits inside 'Country' and
# 'Discount', 'item' inside 'Itemized tax', 'order' inside 'Reorder level'.
_WHOLE_WORD_PATTERNS = {"type", "class", "count", "name", "item", "order",
                        "cost", "amount"}


def _exact_col(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    """Column whose normalized name equals one of ``patterns`` -- the
    equality pass of ``_find_col`` on its own, for fields where a near miss
    is worse than no match at all."""
    norm_to_orig = {_norm(c): c for c in df.columns}
    for p in patterns:
        hit = norm_to_orig.get(_norm(p))
        if hit is not None:
            return hit
    return None


def _find_col(df: pd.DataFrame, patterns: List[str],
              exclude: Tuple[str, ...] = (),
              skip: Tuple[str, ...] = ()) -> Optional[str]:
    """Return the first column whose normalized name matches any pattern.

    Match rule: equality first, then substring. In the substring pass a
    pattern shorter than 3 characters, or one of the everyday words in
    ``_WHOLE_WORD_PATTERNS``, must equal a whole word of the column name (a
    raw substring test lets 't' hit 'agent_id', 'x' hit 'max_qty' and
    'count' hit 'Country'). Columns whose normalized name contains an
    ``exclude`` needle are skipped (e.g. an invoice field must not bind
    'InvoiceDate'), and so are the columns named in ``skip``, which callers
    use to keep one column from being bound to two fields.
    """
    skipped = {str(c) for c in skip}
    norm_to_orig = {_norm(c): c for c in df.columns if str(c) not in skipped}
    pats = [_norm(p) for p in patterns]
    for p in pats:
        if p in norm_to_orig:
            return norm_to_orig[p]
    excl = [_norm(x) for x in exclude]
    for p in pats:
        for n, orig in norm_to_orig.items():
            if any(x in n for x in excl):
                continue
            if len(p) < 3 or p in _WHOLE_WORD_PATTERNS:
                if p in _col_tokens(orig):
                    return orig
            elif p in n:
                return orig
    return None


def _names_all_exact(df: pd.DataFrame, pattern_groups: List[List[str]]) -> bool:
    """True if every pattern group has a column whose normalized name equals
    one of its patterns (the equality pass of ``_find_col`` only)."""
    cols = {_norm(c) for c in df.columns}
    return all(cols & {_norm(p) for p in group} for group in pattern_groups)


def _filename_has_hint(filename: str, hints: List[str]) -> bool:
    """True if one of ``hints`` names the file itself, as a whole word.

    Only the basename is checked (a folder called 'Students' says nothing
    about the file inside it), and a hint may not be flanked by letters, so
    'atc' matches 'atc-20121024.csv' but not 'batch_orders.csv'. Spaces,
    underscores and hyphens inside a hint are interchangeable."""
    base = os.path.basename(str(filename)).lower()
    for h in hints:
        words = [w for w in re.split(r"[\s_\-]+", h.lower()) if w]
        if not words:
            continue
        pat = (r"(?<![a-z])" + r"[\s_\-]*".join(map(re.escape, words))
               + r"(?![a-z])")
        if re.search(pat, base):
            return True
    return False


def _str_or_na(s: pd.Series) -> pd.Series:
    """``astype(str)`` that keeps missing cells missing.

    Plain ``astype(str)`` turns NaN/None into the strings 'nan'/'None',
    which then slip past ``fillna`` and ``dropna``. Blank or
    whitespace-only cells count as missing too."""
    text = s.astype(str)
    return text.where(s.notna() & (text.str.strip() != ""))


def _all_numeric_labels(cols) -> bool:
    """True if every column label parses as a number -- the tell-tale sign
    that ``read_csv`` consumed a headerless file's first data row as a
    header (e.g. ETH/UCY ``obsmat.txt`` or an ATC daily CSV).

    Tolerates pandas' duplicate-column dedup suffix: a headerless row with
    two identical values (e.g. two ``0.0`` columns) becomes labels like
    ``'0.0'`` and ``'0.0.1'`` -- the trailing ``.<n>`` is stripped before
    the numeric test."""
    labels = list(cols)
    if not labels:
        return False
    for c in labels:
        s = str(c)
        try:
            float(s)
            continue
        except (TypeError, ValueError):
            pass
        m = re.match(r"^(.*)\.\d+$", s)   # strip pandas dedup suffix
        if m:
            try:
                float(m.group(1))
                continue
            except (TypeError, ValueError):
                pass
        return False
    return True


# --- Base -----------------------------------------------------------------

class BaseTransactionalAdapter:
    name: str = "base"
    version: str = "0.0"
    CAN_HANDLE_HINTS: List[str] = []

    def confidence(self, df: pd.DataFrame, filename: str = "") -> float:
        """0..1 score for the auto-detector. Higher = better match."""
        return 0.0

    def adapt(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, ValidationReport]:
        raise NotImplementedError


# --- UCI Online Retail II -------------------------------------------------
# Source: https://archive.ics.uci.edu/dataset/502/online+retail+ii
# Two .xlsx sheets, columns:
#   Invoice, StockCode, Description, Quantity, InvoiceDate, Price,
#   Customer ID, Country
# (The 2010-2011 sheet uses 'InvoiceNo' and 'UnitPrice' -- handle both.)

# Online Retail II non-merchandise stock codes (postage, carriage, fees,
# manual adjustments, samples, bad debt, gift vouchers, test rows). These
# carry positive quantity/price but are not products, so they must be
# excluded before calibrating baskets, revenue, and categories (audit R7.1).
_UCI_NONPRODUCT_CODES = {
    "POST", "DOT", "D", "M", "C2", "S", "B", "CRUK", "PADS",
    "BANK CHARGES", "AMAZONFEE", "ADJUST", "ADJUST2",
}
_UCI_NONPRODUCT_RE = r"^(GIFT|TEST|BANK)"

# Fee and non-merchandise lines that carry ordinary product codes as well
# (23444 'Next Day Carriage', 23574 'PACKING CHARGE', 22016 'Dotcomgiftshop
# Gift Voucher', numeric-coded 'samples' and 'adjustment' notes). Matched on
# the trimmed, lower-cased, space-collapsed description. Words that also
# occur in real product names are anchored to the whole description:
# 'FRENCH CARRIAGE LANTERN', 'BAROQUE CARRIAGE CLOCK' and 'PIGGY BANK' stay.
_UCI_NONPRODUCT_DESC_RE = re.compile(
    r"^(?:next day )?carriage$"
    r"|^(?:dotcom )?postage$"
    r"|^packing charge$"
    r"|gift voucher"
    r"|^manual$"
    r"|^discount$"
    r"|\badjust(?:ment)?\b"
    r"|^bank charges?$"
    r"|^amazon fee$"
    r"|\bsamples?\b"
)


class OnlineRetailIIAdapter(BaseTransactionalAdapter):
    name = "uci_online_retail_ii"
    version = "1.1"
    CAN_HANDLE_HINTS = ["online_retail", "online retail", "onlineretail", "retail_ii"]

    UCI_KEYWORDS_TO_CATEGORY = {
        # Crude keyword map -- keeps a flat dataset usable without an
        # external taxonomy. Reviewers see exactly what categorization
        # the report was built against because the map is here in source.
        "bag":           "Bags & Accessories",
        "card":          "Cards & Stationery",
        "book":          "Cards & Stationery",
        "candle":        "Home & Candles",
        "mug":           "Kitchen & Tableware",
        "cup":           "Kitchen & Tableware",
        "bowl":          "Kitchen & Tableware",
        "plate":         "Kitchen & Tableware",
        "jar":           "Kitchen & Tableware",
        "bottle":        "Kitchen & Tableware",
        "tin":           "Kitchen & Tableware",
        "lunch":         "Kitchen & Tableware",
        "cake":          "Baking & Decor",
        "decoration":    "Baking & Decor",
        "garland":       "Baking & Decor",
        "bunting":       "Baking & Decor",
        "heart":         "Gifts & Ornaments",
        "ornament":      "Gifts & Ornaments",
        "christmas":     "Seasonal",
        "easter":        "Seasonal",
        "valentine":     "Seasonal",
        "doll":          "Toys & Games",
        "toy":           "Toys & Games",
        "game":          "Toys & Games",
        "pen":           "Cards & Stationery",
        "pencil":        "Cards & Stationery",
        "frame":         "Home & Candles",
        "light":         "Home & Candles",
        "cushion":       "Home & Candles",
        "clock":         "Home & Candles",
        "vase":          "Home & Candles",
        "umbrella":      "Bags & Accessories",
        "scarf":         "Bags & Accessories",
    }

    def confidence(self, df: pd.DataFrame, filename: str = "") -> float:
        if _filename_has_hint(filename, self.CAN_HANDLE_HINTS):
            return 1.0
        # Column-signature match
        cols = {_norm(c) for c in df.columns}
        signature = {"invoice", "stockcode", "description",
                     "quantity", "invoicedate"}
        # Allow either 'unitprice', 'price' for the price column.
        has_price = bool(cols & {"unitprice", "price"})
        match = len(signature & cols)
        if match >= 4 and has_price:
            return 0.9
        if match >= 3 and has_price:
            return 0.6
        return 0.0

    @classmethod
    def _category_for_text(cls, text: str) -> str:
        """Category for one lower-cased description.

        A map keyword that appears as a whole word (optionally plural)
        takes precedence over one found only inside a longer word, so
        'tin' inside 'bunting' or 'doll' inside 'dolly' cannot shadow the
        map's own 'bunting' / 'lunch' entries. Within each of the two
        tiers the longest matching keyword wins, and map order breaks
        ties. The substring tier still catches compounds such as
        'cakestand' or 'teacup'."""
        ranked = sorted(enumerate(cls.UCI_KEYWORDS_TO_CATEGORY.items()),
                        key=lambda t: (-len(t[1][0]), t[0]))
        for _, (kw, cat) in ranked:
            if re.search(r"\b" + re.escape(kw) + r"(?:e?s)?\b", text):
                return cat
        for _, (kw, cat) in ranked:
            if kw in text:
                return cat
        return "General Merchandise"

    def _infer_category(self, description: str) -> str:
        """Scalar version (kept for unit-testability). The bulk path is
        ``_infer_categories_vectorized`` below; both use
        ``_category_for_text``."""
        return self._category_for_text(str(description).lower())

    @classmethod
    def _infer_categories_vectorized(cls, names: pd.Series) -> pd.Series:
        """Keyword inference over the whole Series.

        Equivalent to ``names.apply(cls._infer_category)``, but each
        distinct description is categorized once and the result mapped
        back: UCI-scale data has ~1M rows but only a few thousand distinct
        descriptions."""
        text = names.fillna("").astype(str).str.lower()
        codes, uniques = pd.factorize(text)
        cats = np.asarray([cls._category_for_text(t) for t in uniques],
                          dtype=object)
        if cats.size == 0:
            return pd.Series("General Merchandise", index=names.index,
                             dtype=object)
        return pd.Series(cats[codes], index=names.index, dtype=object)

    def adapt(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, ValidationReport]:
        report = ValidationReport(
            schema_kind="transactional",
            adapter_name=self.name,
            schema_version=SCHEMA_VERSION,
            rows_in=len(df),
        )

        # Column resolution
        col_invoice = _find_col(df, ["invoice", "invoiceno"])
        col_stock = _find_col(df, ["stockcode", "stock_code", "sku"])
        col_desc = _find_col(df, ["description", "product_name", "name"])
        col_qty = _find_col(df, ["quantity", "qty"])
        col_date = _find_col(df, ["invoicedate", "invoice_date", "date"])
        col_price = _find_col(df, ["unitprice", "unit_price", "price"])
        col_cust = _find_col(df, ["customerid", "customer_id", "customer id"])
        col_country = _find_col(df, ["country"])

        def fs(name, src, status="present", note=""):
            report.fields.append(FieldStatus(name=name, status=status,
                                             source_column=src, note=note))

        if not all([col_invoice, col_stock, col_qty, col_date, col_price]):
            for name, c in [("invoice_id", col_invoice),
                            ("product_id", col_stock),
                            ("quantity", col_qty),
                            ("timestamp", col_date),
                            ("unit_price", col_price)]:
                if c is None:
                    fs(name, None, status="missing",
                       note=f"Required column not found.")
                else:
                    fs(name, c)
            fs("product_name", col_desc,
               status="present" if col_desc else "missing")
            fs("customer_id", col_cust,
               status="present" if col_cust else "missing")
            fs("category", None, status="missing",
               note="Will be inferred from product_name if present.")
            return pd.DataFrame(), report

        # Build normalized DataFrame. Stock codes are trimmed and upper-cased
        # once here: Online Retail II records some SKUs under both spellings
        # ('15056BL' / '15056bl', same description), which would otherwise
        # split one product's visit counts, prices and co-purchase pairs.
        # Missing ids stay missing so the dropna below removes them.
        out = pd.DataFrame({
            "invoice_id":   _str_or_na(df[col_invoice]),
            "product_id":   _str_or_na(df[col_stock]).str.strip().str.upper(),
            "product_name": (df[col_desc].astype(str)
                             if col_desc else df[col_stock].astype(str)),
            "quantity":     pd.to_numeric(df[col_qty], errors="coerce"),
            "timestamp":    pd.to_datetime(df[col_date], errors="coerce"),
            "unit_price":   pd.to_numeric(df[col_price], errors="coerce"),
        })
        if col_cust is not None:
            out["customer_id"] = df[col_cust].astype(str)
        if col_country is not None:
            report.extra["country_col"] = col_country
            report.info.append(
                f"Country column '{col_country}' present — "
                f"top: {df[col_country].value_counts().head(3).to_dict()}"
            )

        # Drop returns (UCI uses 'C' prefix on InvoiceNo for cancellations)
        returns = out["invoice_id"].str.upper().str.startswith("C", na=False)
        n_returns = int(returns.sum())
        if n_returns:
            out = out[~returns]
            report.info.append(f"Dropped {n_returns:,} cancellation rows "
                               f"(InvoiceNo starts with 'C').")

        # Drop non-merchandise stock codes (postage/fees/adjustments/
        # vouchers/tests) so baskets, revenue, and categories reflect actual
        # products (audit R7.1).
        code_u = out["product_id"]
        nonprod = (code_u.isin(_UCI_NONPRODUCT_CODES)
                   | code_u.str.match(_UCI_NONPRODUCT_RE, na=False))
        n_nonprod = int(nonprod.sum())
        if n_nonprod:
            out = out[~nonprod]
            report.info.append(
                f"Dropped {n_nonprod:,} non-merchandise stock-code rows "
                f"(postage, fees, adjustments, vouchers, tests).")

        # The same kinds of line also appear under ordinary product codes;
        # catch them by description, whatever the code.
        if col_desc:
            desc = (out["product_name"].str.strip().str.lower()
                    .str.replace(r"\s+", " ", regex=True))
            fee_desc = desc.str.contains(_UCI_NONPRODUCT_DESC_RE, na=False)
            n_fee_desc = int(fee_desc.sum())
            if n_fee_desc:
                out = out[~fee_desc]
                report.info.append(
                    f"Dropped {n_fee_desc:,} fee / non-merchandise rows by "
                    f"description (carriage, postage, packing charge, gift "
                    f"voucher, manual, discount, adjustment, bank charges, "
                    f"Amazon fee, samples).")

        # Drop bad rows
        before = len(out)
        out = out.dropna(subset=["invoice_id", "product_id",
                                  "quantity", "timestamp", "unit_price"])
        out = out[out["quantity"] > 0]
        out = out[out["unit_price"] > 0]
        n_dropped = before - len(out)
        if n_dropped:
            report.warnings.append(
                f"Dropped {n_dropped:,} rows with NaN, zero, or negative qty/price."
            )

        # Category inference from description (vectorized: a single compiled
        # str.contains pass per keyword vs. a Python loop per row).
        out["category"] = self._infer_categories_vectorized(out["product_name"])
        n_general = int((out["category"] == "General Merchandise").sum())
        report.info.append(
            f"Inferred {out['category'].nunique()} categories via keyword map. "
            f"{n_general:,} rows fell into 'General Merchandise' fallback."
        )

        # Field statuses
        fs("invoice_id", col_invoice)
        fs("timestamp", col_date)
        fs("product_id", col_stock)
        fs("product_name", col_desc,
           status="present" if col_desc else "inferred",
           note="" if col_desc else "Filled from product_id.")
        fs("quantity", col_qty)
        fs("unit_price", col_price)
        fs("customer_id", col_cust,
           status="present" if col_cust else "missing",
           note="" if col_cust else "Return-customer / LTV stats unavailable.")
        fs("category", col_desc,
           status="inferred",
           note="Derived from product_name via UCI keyword map. "
                "Edit dataset_adapters.UCI_KEYWORDS_TO_CATEGORY to customize.")

        report.rows_kept = len(out)
        report.extra["currency"] = "GBP"   # UCI Online Retail II is UK retailer
        return out, report


# --- Generic transactional ------------------------------------------------

class GenericTransactionalAdapter(BaseTransactionalAdapter):
    name = "generic_transactional"
    version = "1.1"
    CAN_HANDLE_HINTS: List[str] = []

    # Column-name patterns for the required fields. The everyday words
    # ('order', 'item', 'cost', ...) only match a whole word of a column
    # name, so their common glued spellings are listed explicitly.
    INVOICE_COLS = ["invoice", "order", "transaction", "basket", "receipt",
                    "orderid", "orderno", "ordernumber"]
    STOCK_COLS = ["stockcode", "sku", "productid", "product_id", "itemid"]
    DESC_COLS = ["description", "productname", "product_name",
                 "product", "name", "item", "itemname"]
    QTY_COLS = ["quantity", "qty", "units", "count"]
    DATE_COLS = ["date", "datetime", "timestamp", "time"]
    PRICE_COLS = ["unitprice", "unit_price", "price", "amount", "cost",
                  "unitcost"]
    CAT_COLS = ["category", "productcategory", "productline", "department",
                "type", "class"]

    def confidence(self, df: pd.DataFrame, filename: str = "") -> float:
        # The fallback after specific adapters (which score 0 or >= 0.55).
        # A file naming every required field exactly scores 0.3 so that
        # detect_schema_kind can still separate it from a plain trajectory
        # file, whose generic adapter gets the same bump.
        if _names_all_exact(df, [self.INVOICE_COLS, self.DATE_COLS,
                                 self.STOCK_COLS + self.DESC_COLS,
                                 self.QTY_COLS, self.PRICE_COLS]):
            return 0.3
        return 0.1

    def adapt(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, ValidationReport]:
        report = ValidationReport(
            schema_kind="transactional",
            adapter_name=self.name,
            schema_version=SCHEMA_VERSION,
            rows_in=len(df),
        )

        # One column may only stand for one field: on a header like
        # 'Invoice ID, Customer type, Product line, Unit price, Quantity'
        # the category patterns would otherwise re-use a column already
        # taken, or bind 'Customer type' and section the shop by membership
        # status.
        claimed: List[str] = []

        def take(patterns, exclude: Tuple[str, ...] = ()):
            col = _find_col(df, patterns, exclude=exclude, skip=tuple(claimed))
            if col is not None:
                claimed.append(col)
            return col

        # 'order'/'invoice' also occur in 'OrderDate'/'InvoiceDate'.
        col_invoice = take(self.INVOICE_COLS, exclude=("date", "time"))
        col_stock = take(self.STOCK_COLS)
        # 'name' is also a word of 'Customer Name'; shop items must not be
        # labelled with who bought them.
        col_desc = take(self.DESC_COLS, exclude=("customer", "payment", "user"))
        # 'Country', 'Discount' and 'Account' are not quantities.
        col_qty = take(self.QTY_COLS,
                       exclude=("country", "discount", "account"))
        col_date = take(self.DATE_COLS)
        col_price = take(self.PRICE_COLS)
        col_cust = take(["customerid", "customer_id", "userid", "user_id"])
        # A product taxonomy, not a customer or payment taxonomy.
        col_cat = take(self.CAT_COLS, exclude=("customer", "payment", "user"))

        # Allow product_id OR product_name as the SKU key.
        product_col = col_stock or col_desc

        def fs(name, src, status="present", note=""):
            report.fields.append(FieldStatus(name=name, status=status,
                                             source_column=src, note=note))

        missing_required = []
        for req_name, col in [("invoice_id", col_invoice),
                              ("timestamp", col_date),
                              ("product_id", product_col),
                              ("quantity", col_qty),
                              ("unit_price", col_price)]:
            if col is None:
                missing_required.append(req_name)

        if missing_required:
            for name, src in [("invoice_id", col_invoice),
                              ("timestamp", col_date),
                              ("product_id", product_col),
                              ("quantity", col_qty),
                              ("unit_price", col_price)]:
                if src is None:
                    fs(name, None, status="missing",
                       note="Required — add this column to proceed.")
                else:
                    fs(name, src)
            fs("product_name", col_desc,
               status="present" if col_desc else "missing")
            fs("customer_id", col_cust,
               status="present" if col_cust else "missing")
            fs("category", col_cat,
               status="present" if col_cat else "missing",
               note="If missing we fall back to a single 'General' category.")
            report.warnings.append(
                "Required columns missing: " + ", ".join(missing_required)
            )
            return pd.DataFrame(), report

        out = pd.DataFrame({
            "invoice_id":   _str_or_na(df[col_invoice]),
            "product_id":   _str_or_na(df[product_col]),
            "product_name": (df[col_desc].astype(str)
                             if col_desc else df[product_col].astype(str)),
            "quantity":     pd.to_numeric(df[col_qty], errors="coerce"),
            "timestamp":    pd.to_datetime(df[col_date], errors="coerce"),
            "unit_price":   pd.to_numeric(df[col_price], errors="coerce"),
        })
        if col_cust is not None:
            out["customer_id"] = df[col_cust].astype(str)
        if col_cat is not None:
            out["category"] = _str_or_na(df[col_cat]).fillna("General")
            cat_status, cat_note = "present", ""
        else:
            out["category"] = "General"
            cat_status, cat_note = "inferred", "No category column found — single 'General' bucket."

        coerced = out
        before = len(out)
        out = out.dropna(subset=["invoice_id", "product_id",
                                  "quantity", "timestamp", "unit_price"])
        out = out[out["quantity"] > 0]
        out = out[out["unit_price"] > 0]
        n_dropped = before - len(out)
        if n_dropped:
            report.warnings.append(
                f"Dropped {n_dropped:,} rows with NaN, zero, or negative qty/price."
            )

        # Nothing survived: a column was almost certainly mapped onto the
        # wrong field (text where a quantity was expected, say). Mark those
        # fields invalid so the report blocks instead of handing the
        # calibrator an empty frame under a clean bill of health.
        invalid: Dict[str, str] = {}
        if len(out) == 0:
            for name, src in [("invoice_id", col_invoice),
                              ("timestamp", col_date),
                              ("product_id", product_col),
                              ("quantity", col_qty),
                              ("unit_price", col_price)]:
                if coerced[name].isna().all():
                    invalid[name] = (f"Column '{src}' held no usable "
                                     f"{name} values.")
            if not invalid:
                invalid["quantity"] = invalid["unit_price"] = (
                    "No row had both a positive quantity and a positive price.")
            report.warnings.append(
                "No rows survived cleaning — check the column mapping above.")

        fs("invoice_id", col_invoice,
           status="invalid" if "invoice_id" in invalid else "present",
           note=invalid.get("invoice_id", ""))
        fs("timestamp", col_date,
           status="invalid" if "timestamp" in invalid else "present",
           note=invalid.get("timestamp", ""))
        fs("product_id", product_col,
           status="invalid" if "product_id" in invalid else "present",
           note=invalid.get("product_id", ""))
        fs("product_name", col_desc,
           status="present" if col_desc else "inferred",
           note="" if col_desc else "Filled from product_id.")
        fs("quantity", col_qty,
           status="invalid" if "quantity" in invalid else "present",
           note=invalid.get("quantity", ""))
        fs("unit_price", col_price,
           status="invalid" if "unit_price" in invalid else "present",
           note=invalid.get("unit_price", ""))
        fs("customer_id", col_cust,
           status="present" if col_cust else "missing",
           note="" if col_cust else "Return-customer / LTV stats unavailable.")
        fs("category", col_cat, status=cat_status, note=cat_note)

        report.rows_kept = len(out)
        return out, report


# --- Omnichannel Retail (aggregated behavioral data) ----------------------
# Source: https://github.com/JoyjitBhowmick/Omnichannel-Retail-Datasets
# This is NOT transactional: there are no invoices/SKUs. It is a bundle of
# four CSVs aggregated per *product family* and joined on ``Aisle ID``:
#   - Demand and Shopping Behavior.csv  (purchase %, daily demand, dwell,
#                                        impulse rate, discrete probabilities)
#   - Product Information.csv           (avg/sd price, dimensions, weight)
#   - Product Family Mapping.csv        (Aisle ID -> in-store Zone ID)
#   - In-store Customer and Online Order Arrivals.csv (hour x weekday traffic)
# We map the genuine *measured* aggregates straight into CalibratedParams via
# ``dataset_calibration.calibrate_omnichannel`` -- no fabricated transactions.

def _omni_col(df: pd.DataFrame, *needles: str, exclude: Tuple[str, ...] = ()):
    """First column whose normalized name contains ALL ``needles`` and none
    of ``exclude``. Tolerant to the trailing spaces / curly apostrophes in
    the Omnichannel headers."""
    for c in df.columns:
        n = _norm(c)
        if all(k in n for k in needles) and not any(x in n for x in exclude):
            return c
    return None


def _pct(v) -> float:
    """Parse a percentage cell ('12.70%' -> 0.127) or a bare fraction/number.
    Values written as percents (with '%' or > 1) are divided by 100.

    Tolerant of dataset-source weirdness: curly quotes, non-breaking spaces,
    leading / trailing non-numeric punctuation, and stray whitespace inside
    the number. Anything that isn't a valid float character (digits, '.',
    sign, exponent) is stripped before the float() call; '%' is detected
    BEFORE stripping so the scale decision still fires."""
    if v is None:
        return float("nan")
    raw = str(v).strip()
    if raw == "" or raw.lower() in ("nan", "none"):
        return float("nan")
    had_pct = "%" in raw
    raw = re.sub(r"[^\d.,+\-eE]", "", raw)
    if not raw:
        return float("nan")
    raw = raw.replace(",", "")
    try:
        x = float(raw)
    except ValueError:
        return float("nan")
    if had_pct or x > 1.0:
        return x / 100.0
    return x


OMNICHANNEL_FILE_HINTS = [
    "demand and shopping", "product family mapping",
    "product information", "in-store customer and online",
]


def looks_like_omnichannel(df: pd.DataFrame, filename: str = "") -> bool:
    """Recognize the Omnichannel bundle from any one of its CSVs (by file
    name or distinctive aggregate columns)."""
    fn = os.path.basename(filename).lower()
    if any(h in fn for h in OMNICHANNEL_FILE_HINTS):
        return True
    cols = [_norm(c) for c in df.columns]
    probes = ["impulsepurchaserate", "averagedailydemand",
              "discreteprobabilitydistribution", "purchasepercentage"]
    hits = sum(1 for p in probes if any(p in c for c in cols))
    return hits >= 2


def load_omnichannel_bundle(path: str):
    """Read + merge the Omnichannel CSV bundle into a per-family DataFrame
    plus an arrivals DataFrame.

    ``path`` may be the folder, or any one CSV inside it (its siblings are
    discovered automatically). Returns ``(families_df, arrivals_df, notes)``
    where ``families_df`` has normalized columns:
        aisle_id, product_family, zone_id, avg_price, avg_len_in, avg_wid_in,
        purchase_pct_instore, discrete_prob_instore, impulse_rate,
        daily_demand_instore, dwell_s
    and ``arrivals_df`` (or None) has: weekday, from_hour, instore_arrivals.
    """
    folder = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
    csvs = {f: os.path.join(folder, f)
            for f in os.listdir(folder) if f.lower().endswith(".csv")}

    def pick(*needles, exclude=()):
        # Sorted so the choice doesn't depend on the filesystem's listdir
        # order.
        for f, full in sorted(csvs.items()):
            lf = f.lower()
            if all(n in lf for n in needles) and not any(x in lf for x in exclude):
                return full
        return None

    f_demand = pick("demand")
    # The bundle also ships 'Web Scraped Product Information.csv', which has
    # no Avg price column; picking it would silently drop every price.
    f_info   = pick("information", exclude=("scraped",))
    f_map    = pick("mapping")
    f_arr    = pick("arrival")
    if not f_demand:
        raise ValueError(
            "Omnichannel bundle incomplete: could not find a "
            "'Demand and Shopping Behavior.csv' in " + folder)

    notes = [f"Loaded {os.path.basename(f_demand)}"]
    demand = pd.read_csv(f_demand)

    c_aisle = _omni_col(demand, "aisleid")
    c_pf    = _omni_col(demand, "productfamily")
    c_buy   = _omni_col(demand, "instore", "purchasepercentage")
    c_disc  = _omni_col(demand, "instore", "discreteprobability")
    c_imp   = _omni_col(demand, "impulse")
    c_dem   = _omni_col(demand, "dailydemand", "instore")
    c_dwell = _omni_col(demand, "dwell")
    if c_aisle is None or c_pf is None:
        raise ValueError("Omnichannel demand CSV missing 'Aisle ID' / "
                         "'Product family' columns.")

    fam_df = pd.DataFrame({
        "aisle_id":              pd.to_numeric(demand[c_aisle], errors="coerce"),
        "product_family":        demand[c_pf].astype(str),
        "purchase_pct_instore":  demand[c_buy].map(_pct) if c_buy else float("nan"),
        "discrete_prob_instore": demand[c_disc].map(_pct) if c_disc else float("nan"),
        "impulse_rate":          demand[c_imp].map(_pct) if c_imp else float("nan"),
        "daily_demand_instore":  pd.to_numeric(demand[c_dem], errors="coerce") if c_dem else float("nan"),
        "dwell_s":               pd.to_numeric(demand[c_dwell], errors="coerce") if c_dwell else float("nan"),
    })

    if f_info:
        info = pd.read_csv(f_info)
        ci_aisle = _omni_col(info, "aisleid")
        ci_price = _omni_col(info, "avgprice")
        ci_len   = _omni_col(info, "avglength")
        ci_wid   = _omni_col(info, "avgwidth")
        if ci_aisle is not None and ci_price is not None:
            price_df = pd.DataFrame({
                "aisle_id":   pd.to_numeric(info[ci_aisle], errors="coerce"),
                "avg_price":  pd.to_numeric(info[ci_price], errors="coerce"),
                "avg_len_in": pd.to_numeric(info[ci_len], errors="coerce") if ci_len else float("nan"),
                "avg_wid_in": pd.to_numeric(info[ci_wid], errors="coerce") if ci_wid else float("nan"),
            })
            fam_df = fam_df.merge(price_df, on="aisle_id", how="left")
            notes.append(f"Merged prices/dimensions from {os.path.basename(f_info)}")
    if "avg_price" not in fam_df.columns:
        fam_df["avg_price"] = float("nan")
        fam_df["avg_len_in"] = float("nan")
        fam_df["avg_wid_in"] = float("nan")

    if f_map:
        fam = pd.read_csv(f_map)
        cm_aisle = _omni_col(fam, "aisleid")
        cm_zone  = _omni_col(fam, "zoneid", exclude=("alternativ",))
        if cm_aisle is not None and cm_zone is not None:
            zone_df = pd.DataFrame({
                "aisle_id": pd.to_numeric(fam[cm_aisle], errors="coerce"),
                "zone_id":  pd.to_numeric(fam[cm_zone], errors="coerce"),
            })
            fam_df = fam_df.merge(zone_df, on="aisle_id", how="left")
            notes.append(f"Merged in-store zones from {os.path.basename(f_map)}")
    if "zone_id" not in fam_df.columns:
        fam_df["zone_id"] = float("nan")

    arrivals = None
    if f_arr:
        araw = pd.read_csv(f_arr)
        a_day  = _omni_col(araw, "day")
        a_from = _omni_col(araw, "fromtime")
        a_in   = _omni_col(araw, "instore", "arrival")
        if a_day is not None and a_from is not None and a_in is not None:
            arrivals = pd.DataFrame({
                "weekday":          araw[a_day].astype(str),
                "from_hour":        pd.to_numeric(araw[a_from], errors="coerce"),
                "instore_arrivals": pd.to_numeric(araw[a_in], errors="coerce"),
            })
            notes.append(f"Loaded arrivals from {os.path.basename(f_arr)}")

    fam_df = fam_df.dropna(subset=["aisle_id", "product_family"])
    fam_df = fam_df[fam_df["product_family"].astype(str).str.strip() != ""]
    return fam_df, arrivals, notes


class OmnichannelRetailAdapter(BaseTransactionalAdapter):
    """Detection + validation-report shim for the Omnichannel bundle. The
    real ingestion is ``load_omnichannel_bundle`` + ``calibrate_omnichannel``;
    the normal single-file ``adapt`` path is not used (the dataset is a
    4-CSV bundle that must be merged)."""
    name = "omnichannel_retail"
    version = "1.0"
    CAN_HANDLE_HINTS = ["omnichannel", "demand and shopping",
                        "product family mapping"]

    def confidence(self, df: pd.DataFrame, filename: str = "") -> float:
        return 0.95 if looks_like_omnichannel(df, filename) else 0.0

    def adapt(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, ValidationReport]:
        report = ValidationReport(
            schema_kind="transactional", adapter_name=self.name,
            schema_version=SCHEMA_VERSION, rows_in=len(df))
        report.info.append(
            "Omnichannel is an AGGREGATE multi-CSV bundle. Pick the folder "
            "(or any CSV inside it) so all four files merge; calibration "
            "maps the measured family-level aggregates directly.")
        report.extra["aggregate_bundle"] = True
        return df, report

    def report_for_bundle(self, families: pd.DataFrame, arrivals) -> ValidationReport:
        report = ValidationReport(
            schema_kind="transactional", adapter_name=self.name,
            schema_version=SCHEMA_VERSION, rows_in=len(families))

        def fs(name, src, status="present", note=""):
            report.fields.append(FieldStatus(name=name, status=status,
                                             source_column=src, note=note))

        has_price = bool(families["avg_price"].notna().any())
        has_zone  = bool("zone_id" in families and families["zone_id"].notna().any())
        fs("product_id", "aisle_id")
        fs("product_name", "product_family")
        fs("unit_price", "avg_price",
           status="present" if has_price else "inferred",
           note="" if has_price else "No price file; median fallback.")
        fs("category", "zone_id" if has_zone else None,
           status="present" if has_zone else "inferred",
           note="In-store Zone ID becomes the section." if has_zone
                else "No zone mapping; product family used as section.")
        fs("quantity", "daily_demand_instore", status="inferred",
           note="Daily demand is used as a popularity weight, not a count.")
        fs("invoice_id", None, status="inferred",
           note="Aggregate source: no invoices. Baskets are a parametric "
                "realization of the measured per-family purchase probabilities.")
        fs("timestamp", "arrivals" if arrivals is not None else None,
           status="inferred",
           note="Hour x weekday arrival profile drives the NHPP."
                if arrivals is not None else
                "No arrivals CSV; uniform hourly fallback.")
        report.rows_kept = len(families)
        report.extra["currency"] = "USD"
        report.extra["aggregate_bundle"] = True
        report.info.append(f"Product families: {len(families):,}")
        if has_zone:
            report.info.append(
                f"In-store zones: {int(families['zone_id'].nunique()):,}")
        if arrivals is not None:
            report.info.append(f"Arrival rows: {len(arrivals):,}")
        return report


# --- Trajectory adapters --------------------------------------------------

class BaseTrajectoryAdapter:
    """Same interface as the transactional adapters, but the output df has
    columns track_id, time_s, x_m, y_m, [floor]."""
    name: str = "base_trajectory"
    version: str = "0.0"
    CAN_HANDLE_HINTS: List[str] = []

    def confidence(self, df: pd.DataFrame, filename: str = "") -> float:
        return 0.0

    def adapt(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, ValidationReport]:
        raise NotImplementedError


class ATCShoppingMallAdapter(BaseTrajectoryAdapter):
    """Brscic et al. (IROS 2013) ATC Shopping Mall format.

    Per the dataset README (https://dil.atr.jp/sets/ATC/):
      columns: time [s since midnight, float],
               person_id [int],
               position_x [mm, float], position_y [mm, float],
               position_z (height) [mm, float],
               velocity [mm/s, float],
               angle of motion [rad, float],
               facing angle [rad, float]

    Files are CSV with no header, one per recording day.
    """
    name = "atc_shopping_mall"
    version = "1.0"
    CAN_HANDLE_HINTS = ["atc", "atc-", "shopping_mall"]

    def confidence(self, df: pd.DataFrame, filename: str = "") -> float:
        if _filename_has_hint(filename, self.CAN_HANDLE_HINTS):
            return 0.95
        # Headerless 8-column numeric DF with second column int-like and
        # x/y columns in ~mm range (i.e. |value| >> 100) is a strong hint.
        cols = df.columns.tolist()
        if cols == list(range(len(cols))) and len(cols) == 8:
            try:
                x = pd.to_numeric(df.iloc[:, 2], errors="coerce")
                y = pd.to_numeric(df.iloc[:, 3], errors="coerce")
                if (x.abs().median() > 1000) and (y.abs().median() > 1000):
                    return 0.85
            except Exception:
                pass
        return 0.0

    def adapt(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, ValidationReport]:
        report = ValidationReport(
            schema_kind="trajectory",
            adapter_name=self.name,
            schema_version=SCHEMA_VERSION,
            rows_in=len(df),
        )

        def fs(name, src, status="present", note=""):
            report.fields.append(FieldStatus(name=name, status=status,
                                             source_column=src, note=note))

        # ATC files have no header; assign column names by position.
        ncols = len(df.columns)
        if ncols < 4:
            fs("track_id", None, status="missing",
               note="ATC files must have at least 4 columns (time, id, x, y).")
            return pd.DataFrame(), report

        # Permit either a headered or headerless source.
        if df.columns.tolist() == list(range(ncols)):
            df = df.copy()
            atc_cols = ["time_s", "person_id", "pos_x_mm", "pos_y_mm",
                        "pos_z_mm", "velocity_mm_s",
                        "angle_motion_rad", "facing_angle_rad"][:ncols]
            df.columns = atc_cols + list(df.columns[len(atc_cols):])

        col_time = _find_col(df, ["time_s", "time", "t"])
        col_id   = _find_col(df, ["person_id", "track_id", "id", "ped_id"])
        col_x    = _find_col(df, ["pos_x_mm", "pos_x", "x_mm", "x"])
        col_y    = _find_col(df, ["pos_y_mm", "pos_y", "y_mm", "y"])
        col_v    = _find_col(df, ["velocity_mm_s", "velocity", "v"])

        for need, c in [("track_id", col_id), ("time_s", col_time),
                        ("x_m", col_x), ("y_m", col_y)]:
            if c is None:
                fs(need, None, status="missing",
                   note="Required for trajectory calibration.")
        if any(fsx.status == "missing" for fsx in report.fields):
            return pd.DataFrame(), report

        # Build normalized df. Convert mm -> m. Drop NaN rows.
        out = pd.DataFrame({
            "track_id": pd.to_numeric(df[col_id], errors="coerce").astype("Int64"),
            "time_s":   pd.to_numeric(df[col_time], errors="coerce"),
            "x_m":      pd.to_numeric(df[col_x], errors="coerce") / 1000.0,
            "y_m":      pd.to_numeric(df[col_y], errors="coerce") / 1000.0,
        })
        if col_v is not None:
            out["velocity_m_s"] = (
                pd.to_numeric(df[col_v], errors="coerce") / 1000.0
            )

        before = len(out)
        out = out.dropna(subset=["track_id", "time_s", "x_m", "y_m"])
        if before - len(out):
            report.warnings.append(
                f"Dropped {before - len(out):,} rows with NaN required fields."
            )

        # Stationary outliers: ATC sometimes records a static reading when
        # the tracker briefly lost a person. Filter velocities > 5 m/s
        # (running) which are almost always tracker glitches.
        if "velocity_m_s" in out.columns:
            n_fast = int((out["velocity_m_s"] > 5.0).sum())
            if n_fast:
                out = out[out["velocity_m_s"] <= 5.0]
                report.warnings.append(
                    f"Dropped {n_fast:,} rows with velocity > 5 m/s "
                    f"(tracker glitch, not human walking)."
                )

        fs("track_id", col_id)
        fs("time_s", col_time)
        fs("x_m", col_x, note="Converted from mm to m.")
        fs("y_m", col_y, note="Converted from mm to m.")
        fs("floor", None, status="inferred",
           note="ATC is single-floor (ground); all tracks placed on floor 1.")

        report.rows_kept = len(out)
        report.info.append(
            f"Tracks:          {out['track_id'].nunique():,}\n"
            f"  Time span:       {out['time_s'].max()-out['time_s'].min():.0f} s"
        )
        report.extra["coord_units"] = "mm -> m"
        return out, report


class OpenTrajAdapter(BaseTrajectoryAdapter):
    """OpenTraj pedestrian-trajectory collection (Amirian et al.).

    https://github.com/crowdbotp/OpenTraj

    Handles the two on-disk shapes the project ships / exports:

      * ETH / UCY *world-coordinate* raw files (e.g. ``seq_eth/obsmat.txt``):
        whitespace-delimited, headerless, 8 columns in the BIWI order
            frame_id, agent_id, pos_x, pos_z, pos_y, v_x, v_z, v_y
        Positions are already in METRES (world frame); the pos_z height
        column is dropped. Time is derived from the frame number and the
        dataset annotation rate (2.5 fps for ETH+UCY), reproducing
        OpenTraj's own ``load_eth`` post-processing.

      * OpenTraj *unified* CSV exported from ``TrajDataset.data``: named
        columns ``frame_id, agent_id, pos_x, pos_y [, vel_x, vel_y,
        timestamp, scene_id, label]``.

    Separation from ATC: ETH/UCY world coords are in metres (|median| < 100),
    whereas ATC is in millimetres (|median| > 1000), so the structural
    detector never confuses the two.
    """
    name = "opentraj"
    version = "1.1"
    # NB: bare 'eth'/'ucy' are deliberately NOT hints -- 'eth' is a substring
    # of innocent words ('method'). We use the distinctive scene/file names.
    CAN_HANDLE_HINTS = ["opentraj", "obsmat", "biwi", "seq_eth", "seq_hotel",
                        "zara", "students", "crowds", "univ_examples"]

    # ETH/UCY positions are annotated at 2.5 fps (one sample / 0.4 s).
    ANNOT_FPS = 2.5

    def _is_unified(self, df: pd.DataFrame) -> bool:
        cols = {_norm(c) for c in df.columns}
        return ("posx" in cols and "posy" in cols
                and ("agentid" in cols or "frameid" in cols))

    def _is_world_obsmat(self, df: pd.DataFrame) -> bool:
        cols = df.columns.tolist()
        if cols != list(range(len(cols))) or len(cols) < 5:
            return False
        try:
            x = pd.to_numeric(df.iloc[:, 2], errors="coerce")
            y = pd.to_numeric(df.iloc[:, 4], errors="coerce")
        except Exception:
            return False
        return bool(x.abs().median() < 100) and bool(y.abs().median() < 100)

    def confidence(self, df: pd.DataFrame, filename: str = "") -> float:
        hinted = _filename_has_hint(filename, self.CAN_HANDLE_HINTS)
        if self._is_unified(df):
            return 0.95 if hinted else 0.85
        if self._is_world_obsmat(df):
            return 0.9 if hinted else 0.55
        return 0.6 if hinted else 0.0

    def _frame_to_seconds(self, frames) -> Tuple[pd.Series, float]:
        """time_s = frame_id / fps, fps = frame_stride * 2.5 (this exactly
        reproduces OpenTraj's ``load_eth``: a stride-6 frame index at the
        common 2.5 fps annotation rate yields 0.4 s between samples)."""
        f = pd.to_numeric(pd.Series(frames).reset_index(drop=True), errors="coerce")
        uniq = np.unique(f.dropna().to_numpy())
        stride = 1.0
        if uniq.size >= 2:
            d = np.diff(uniq)
            d = d[d > 0]
            if d.size:
                stride = float(np.min(d))
        fps = stride * self.ANNOT_FPS
        if fps <= 0:
            fps = self.ANNOT_FPS
        return f / fps, fps

    def adapt(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, ValidationReport]:
        report = ValidationReport(
            schema_kind="trajectory",
            adapter_name=self.name,
            schema_version=SCHEMA_VERSION,
            rows_in=len(df),
        )

        def fs(name, src, status="present", note=""):
            report.fields.append(FieldStatus(name=name, status=status,
                                             source_column=src, note=note))

        df = df.copy()
        unified = self._is_unified(df)
        vel = None

        if unified:
            # A unified CSV may carry frame_id and scene_id without
            # agent_id, and both of those contain the word 'id'. Only an
            # exact agent/track name may become the track id: binding a
            # frame or a scene column instead would merge every pedestrian
            # of the scene into one track and turn the distances between
            # different people into walking speed.
            col_id    = _exact_col(df, ["agent_id", "track_id", "id", "ped_id"])
            col_x     = _find_col(df, ["pos_x", "x_m", "x"])
            col_y     = _find_col(df, ["pos_y", "y_m", "y"])
            col_t     = _find_col(df, ["timestamp", "time_s", "time", "t"])
            col_frame = _find_col(df, ["frame_id", "frame"])
            col_vx    = _find_col(df, ["vel_x", "vx"])
            col_vy    = _find_col(df, ["vel_y", "vy"])
            if col_id is None:
                fs("track_id", None, status="missing",
                   note="No agent/track id column in unified CSV.")
                return pd.DataFrame(), report
            track = pd.to_numeric(df[col_id], errors="coerce")
            x = pd.to_numeric(df[col_x], errors="coerce")
            y = pd.to_numeric(df[col_y], errors="coerce")
            if col_t is not None:
                time_s = pd.to_numeric(df[col_t], errors="coerce")
                tnote = f"From '{col_t}'."
                src_t = col_t
            elif col_frame is not None:
                time_s, fps = self._frame_to_seconds(df[col_frame])
                tnote = f"frame/{fps:g} fps."
                src_t = col_frame
            else:
                fs("time_s", None, status="missing",
                   note="No timestamp or frame column in unified CSV.")
                return pd.DataFrame(), report
            if col_vx is not None and col_vy is not None:
                vx = pd.to_numeric(df[col_vx], errors="coerce")
                vy = pd.to_numeric(df[col_vy], errors="coerce")
                vel = np.sqrt(vx * vx + vy * vy)
            src_id, src_x, src_y = col_id, col_x, col_y
            y_note = "World metres."
        else:
            ncols = len(df.columns)
            if ncols < 5 or df.columns.tolist() != list(range(ncols)):
                fs("track_id", None, status="missing",
                   note="Expected an OpenTraj unified CSV or an 8-column "
                        "ETH/UCY world file (obsmat); got an unrecognized "
                        "shape. Try the generic trajectory adapter.")
                return pd.DataFrame(), report
            track = pd.to_numeric(df.iloc[:, 1], errors="coerce")
            x = pd.to_numeric(df.iloc[:, 2], errors="coerce")
            y = pd.to_numeric(df.iloc[:, 4], errors="coerce")
            time_s, fps = self._frame_to_seconds(df.iloc[:, 0])
            tnote = f"col0 / {fps:g} fps (ETH/UCY 2.5 fps annotation)."
            if ncols >= 8:
                vx = pd.to_numeric(df.iloc[:, 5], errors="coerce")
                vy = pd.to_numeric(df.iloc[:, 7], errors="coerce")
                vel = np.sqrt(vx * vx + vy * vy)
            src_id, src_x, src_y, src_t = "col1", "col2", "col4", "col0"
            y_note = "World metres; ETH/UCY pos_z (height, col3) dropped."

        out = pd.DataFrame({
            "track_id": track.astype("Int64"),
            "time_s":   pd.to_numeric(time_s, errors="coerce"),
            "x_m":      x,
            "y_m":      y,
        })
        if vel is not None:
            out["velocity_m_s"] = pd.to_numeric(vel, errors="coerce")

        before = len(out)
        out = out.dropna(subset=["track_id", "time_s", "x_m", "y_m"])
        if before - len(out):
            report.warnings.append(
                f"Dropped {before - len(out):,} rows with NaN required fields.")

        if "velocity_m_s" in out.columns:
            n_fast = int((out["velocity_m_s"] > 5.0).sum())
            if n_fast:
                out = out[out["velocity_m_s"] <= 5.0]
                report.warnings.append(
                    f"Dropped {n_fast:,} rows with velocity > 5 m/s "
                    f"(tracker glitch, not human walking).")

        fs("track_id", src_id)
        fs("time_s", src_t, note=tnote)
        fs("x_m", src_x, note="World metres (no unit conversion).")
        fs("y_m", src_y, note=y_note)
        fs("floor", None, status="inferred",
           note="OpenTraj scenes are single-level; all tracks on floor 1.")
        report.rows_kept = len(out)
        report.extra["coord_units"] = "m (world)"
        report.extra["annotation_fps"] = self.ANNOT_FPS
        if len(out):
            report.info.append(
                f"Tracks: {int(out['track_id'].nunique()):,}; "
                f"time span {float(out['time_s'].max() - out['time_s'].min()):.0f} s.")
        return out, report


class GenericTrajectoryAdapter(BaseTrajectoryAdapter):
    """Fallback adapter: any CSV/TSV with columns track_id, time_s, x_m, y_m
    (or close fuzzy matches). Units are assumed to be SI (seconds, metres);
    if a heuristic finds large values it auto-scales mm -> m."""
    name = "generic_trajectory"
    version = "1.1"

    # Column-name patterns for the required fields. The glued spellings
    # ('xpos', 'pid', 'ts') are listed explicitly rather than reached by a
    # looser match rule, which would also bind bounding-box corners (x1/y1)
    # as positions.
    TRACK_COLS = ["track_id", "person_id", "id", "ped_id", "pid", "oid",
                  "objectid", "agent", "pedestrian"]
    TIME_COLS = ["time_s", "time", "t", "timestamp", "ts", "frame"]
    X_COLS = ["x_m", "x", "pos_x", "position_x", "xpos", "posx", "px"]
    Y_COLS = ["y_m", "y", "pos_y", "position_y", "ypos", "posy", "py"]

    def confidence(self, df: pd.DataFrame, filename: str = "") -> float:
        # Still the fallback (specific adapters score 0 or >= 0.55), but a
        # file whose track/time/x/y columns all resolve -- to four different
        # columns -- must beat the transactional fallback's 0.1, or
        # detect_schema_kind never routes a plain 'agent_id,timestamp,x,y'
        # export here. The resolution uses the same calls adapt() makes, so
        # the score and the mapping cannot disagree.
        cols = [_find_col(df, self.TRACK_COLS, exclude=("frame",)),
                _find_col(df, self.TIME_COLS),
                _find_col(df, self.X_COLS),
                _find_col(df, self.Y_COLS)]
        if all(c is not None for c in cols) and len(set(cols)) == len(cols):
            return 0.3
        return 0.1

    def adapt(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, ValidationReport]:
        report = ValidationReport(
            schema_kind="trajectory",
            adapter_name=self.name,
            schema_version=SCHEMA_VERSION,
            rows_in=len(df),
        )

        def fs(name, src, status="present", note=""):
            report.fields.append(FieldStatus(name=name, status=status,
                                             source_column=src, note=note))

        # 'frame' is a time pattern; keep 'id' from binding 'frame_id' too.
        col_id   = _find_col(df, self.TRACK_COLS, exclude=("frame",))
        col_time = _find_col(df, self.TIME_COLS)
        col_x    = _find_col(df, self.X_COLS)
        col_y    = _find_col(df, self.Y_COLS)

        # A frame index is the last resort for time; it counts annotation
        # frames, not seconds, so say so instead of reporting frame numbers
        # as a duration.
        time_note = ""
        if col_time is not None and "frame" in _norm(col_time):
            time_note = (f"'{col_time}' looks like a frame index; time_s is in "
                         f"frames, not seconds — divide by the source's frame "
                         f"rate before comparing dwell times.")
            report.warnings.append(time_note)

        for need, c in [("track_id", col_id), ("time_s", col_time),
                        ("x_m", col_x), ("y_m", col_y)]:
            fs(need, c, status="present" if c else "missing",
               note=(time_note if need == "time_s" else "") if c else "Required.")
        if report.is_blocking():
            return pd.DataFrame(), report

        out = pd.DataFrame({
            "track_id": pd.to_numeric(df[col_id], errors="coerce").astype("Int64"),
            "time_s":   pd.to_numeric(df[col_time], errors="coerce"),
            "x_m":      pd.to_numeric(df[col_x], errors="coerce"),
            "y_m":      pd.to_numeric(df[col_y], errors="coerce"),
        })

        before = len(out)
        out = out.dropna(subset=["track_id", "time_s", "x_m", "y_m"])

        # Auto-scale mm -> m if positions look like millimetres.
        if out["x_m"].abs().median() > 200 and out["y_m"].abs().median() > 200:
            out["x_m"] = out["x_m"] / 1000.0
            out["y_m"] = out["y_m"] / 1000.0
            report.info.append("Detected mm-scale coordinates; converted to m.")

        if before - len(out):
            report.warnings.append(
                f"Dropped {before - len(out):,} rows with NaN required fields."
            )

        fs("floor", None, status="inferred",
           note="No floor column; all tracks placed on floor 1.")
        report.rows_kept = len(out)
        return out, report


# --- Registries + auto-detect ---------------------------------------------

TRANSACTIONAL_ADAPTERS: List[BaseTransactionalAdapter] = [
    OnlineRetailIIAdapter(),
    OmnichannelRetailAdapter(),
    GenericTransactionalAdapter(),
]

TRAJECTORY_ADAPTERS: List[BaseTrajectoryAdapter] = [
    ATCShoppingMallAdapter(),
    OpenTrajAdapter(),
    GenericTrajectoryAdapter(),
]


def best_transactional_adapter(df: pd.DataFrame, filename: str = ""
                               ) -> BaseTransactionalAdapter:
    """Pick the transactional adapter with the highest confidence score."""
    scored = [(a.confidence(df, filename), a) for a in TRANSACTIONAL_ADAPTERS]
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[0][1]


def best_trajectory_adapter(df: pd.DataFrame, filename: str = ""
                            ) -> BaseTrajectoryAdapter:
    """Pick the trajectory adapter with the highest confidence score."""
    scored = [(a.confidence(df, filename), a) for a in TRAJECTORY_ADAPTERS]
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[0][1]


def detect_schema_kind(df: pd.DataFrame, filename: str = "") -> str:
    """Decide whether this source is transactional or trajectory by
    asking both adapter families and seeing which gives a higher score."""
    tx_score = max(a.confidence(df, filename) for a in TRANSACTIONAL_ADAPTERS)
    tj_score = max(a.confidence(df, filename) for a in TRAJECTORY_ADAPTERS)
    return "trajectory" if tj_score > tx_score else "transactional"


def list_excel_sheets(filename: str) -> List[Tuple[str, int]]:
    """Return [(sheet_name, n_rows), ...] for the sheets in an Excel workbook.

    Used by the UI to let the user pick (and to merge across) sheets when a
    workbook like UCI Online Retail II ships one sheet per calendar year.

    ``n_rows`` excludes the header. For openpyxl workbooks it is the sheet's
    recorded extent, which needs no cell parsing; other engines, and sheets
    without a usable extent, fall back to reading the first column.
    """
    try:
        xls = pd.ExcelFile(filename)
    except Exception as e:
        raise ValueError(f"Could not open Excel workbook: {e}") from e
    out: List[Tuple[str, int]] = []
    for name in xls.sheet_names:
        nrows = -1
        if xls.engine == "openpyxl":
            try:
                # pandas opens the workbook read-only, where max_row comes
                # from the <dimension> tag (~0.1 s) instead of parsing the
                # whole sheet (~20 s per UCI sheet). It is None when the tag
                # is missing; 1 may be a placeholder 'A1' extent, so recount.
                max_row = xls.book[name].max_row
                if isinstance(max_row, int) and max_row > 1:
                    nrows = max_row - 1
            except Exception:
                pass
        if nrows < 0:
            try:
                # Read just the index column to count rows
                nrows = len(pd.read_excel(filename, sheet_name=name, usecols=[0]))
            except Exception:
                nrows = -1
        out.append((name, nrows))
    return out


def read_excel_sheets(filename: str, sheet_names: List[str]
                      ) -> Tuple[pd.DataFrame, List[str]]:
    """Read one or more named sheets from an Excel workbook and concatenate
    them row-wise. Used after the user picks sheets in the load dialog.

    Rows of a later sheet that repeat rows of an earlier sheet are dropped
    (see below); repeated rows within one sheet are kept."""
    notes: List[str] = []
    frames: List[pd.DataFrame] = []
    for name in sheet_names:
        df = pd.read_excel(filename, sheet_name=name)
        frames.append(df)
        notes.append(f"Read sheet '{name}' ({len(df):,} rows).")
    if not frames:
        raise ValueError("No sheets selected.")
    if len(frames) == 1:
        return frames[0], notes
    # Concatenate; pandas aligns by column name. If column names differ
    # across sheets the union is taken and missing cells are NaN -- the
    # adapter's dropna step will filter those.
    out = pd.concat(frames, ignore_index=True, sort=False)

    # Sheets of one workbook can overlap in time: Online Retail II's
    # 'Year 2009-2010' runs to 2010-12-09 and 'Year 2010-2011' starts on
    # 2010-12-01, so the same invoices appear in both and concatenation
    # doubles every line of them. A later-sheet row is dropped when it
    # equals, on every column (missing cells equal), a row kept from an
    # earlier sheet. Copies are paired one to one, so a line repeated within
    # a single sheet more often than earlier sheets hold it keeps its extra
    # copies, and repeats confined to one sheet are never touched.
    if len(out) and len(out.columns):
        sheet_idx = np.repeat(np.arange(len(frames)), [len(f) for f in frames])
        # Text columns are compared as strings: read_excel parses a column of
        # numeric-looking codes as integers in one sheet but keeps them as
        # text where the sheet also holds codes like '85123A', so the same
        # line would otherwise not match across sheets.
        key_frame = out.copy(deep=False)
        for col in key_frame.columns:
            if key_frame[col].dtype == object:
                cells = key_frame[col]
                key_frame[col] = cells.astype(str).where(cells.notna())
        row_key = (key_frame.groupby(list(key_frame.columns), dropna=False,
                                     sort=False)
                   .ngroup().to_numpy())
        occurrence = (pd.DataFrame({"sheet": sheet_idx, "key": row_key})
                      .groupby(["sheet", "key"], sort=False)
                      .cumcount().to_numpy())
        kept_count = np.zeros(int(row_key.max()) + 1, dtype=np.int64)
        keep = np.ones(len(out), dtype=bool)
        for i, name in enumerate(sheet_names):
            rows = np.flatnonzero(sheet_idx == i)
            keys = row_key[rows]
            repeat = occurrence[rows] < kept_count[keys]
            keep[rows[repeat]] = False
            kept_count += np.bincount(keys[~repeat], minlength=kept_count.size)
            if repeat.any():
                notes.append(f"Dropped {int(repeat.sum()):,} rows of sheet "
                             f"'{name}' that repeat rows of an earlier sheet.")
        if not keep.all():
            out = out[keep].reset_index(drop=True)
    notes.append(f"Concatenated {len(frames)} sheets -> {len(out):,} rows total.")
    return out, notes


def read_any(filename: str) -> Tuple[pd.DataFrame, List[str]]:
    """Read CSV / Excel / TSV / TXT into a DataFrame, returning notes about
    encoding + separator chosen. Raises ValueError on truly unreadable input.

    For multi-sheet Excel workbooks this reads only the first sheet -- call
    ``list_excel_sheets`` + ``read_excel_sheets`` from the UI instead when
    the user should be allowed to pick.
    """
    notes: List[str] = []
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if ext in ("xlsx", "xls"):
        try:
            xls = pd.ExcelFile(filename)
            if len(xls.sheet_names) > 1:
                notes.append(
                    f"Workbook has {len(xls.sheet_names)} sheets "
                    f"({', '.join(xls.sheet_names)}); read only the first."
                )
            df = pd.read_excel(filename, sheet_name=xls.sheet_names[0])
            notes.append(f"Read as Excel ({ext}), sheet '{xls.sheet_names[0]}'.")
            return df, notes
        except Exception as e:
            raise ValueError(f"Excel read failed: {e}") from e

    # Text-based: try a few encodings + separators.
    last_err = None
    for sep, sep_name in [(",", "comma"), ("\t", "tab"),
                          (";", "semicolon"), (r"\s+", "whitespace")]:
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                # The C engine supports all four separators ('\s+' included)
                # and is several times faster than the python engine.
                # low_memory=False infers each column's dtype over the whole
                # file, as the python engine does, rather than per chunk
                # (which can mix '536365' and 536365.0 in one id column).
                df = pd.read_csv(filename, sep=sep, encoding=enc,
                                 engine="c", low_memory=False,
                                 on_bad_lines="skip")
                if len(df.columns) >= 2 and len(df) > 0:
                    # Headerless numeric files (ETH/UCY obsmat, ATC daily CSVs)
                    # get their first DATA row consumed as a header by
                    # read_csv. Detect that (all column labels parse as
                    # numbers) and re-read with header=None so the row is
                    # preserved and columns become a positional RangeIndex
                    # (0..n) -- exactly what the trajectory adapters expect.
                    if _all_numeric_labels(df.columns):
                        df = pd.read_csv(filename, sep=sep, encoding=enc,
                                         engine="c", low_memory=False,
                                         on_bad_lines="skip", header=None)
                        notes.append(f"Read as headerless text "
                                     f"(sep={sep_name}, encoding={enc}).")
                    else:
                        notes.append(f"Read as text (sep={sep_name}, encoding={enc}).")
                    return df, notes
            except Exception as e:
                last_err = e
                continue
            # Decoded without error, but this separator doesn't split the
            # file. The separators are ASCII, so another encoding yields the
            # same columns; move on to the next separator.
            break

    raise ValueError(f"Could not parse {filename!r}: {last_err}")
