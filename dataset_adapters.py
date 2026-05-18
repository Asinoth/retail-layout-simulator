"""Adapters that map arbitrary dataset files onto the normalized schema.

Each adapter implements:

  - ``CAN_HANDLE_HINTS`` (class attr) → list of substrings that, if present
    in the source filename or column names, make the adapter a strong
    candidate for the auto-detector.

  - ``adapt(df) -> (normalized_df, ValidationReport)`` → returns a DataFrame
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


# ─── Helpers ──────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    """lowercase, strip spaces/underscores — for fuzzy column matching."""
    return re.sub(r"[\s_\-]+", "", str(s).lower())


def _find_col(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    """Return the first column whose normalized name matches any pattern.

    Match rule: equality first, then substring.
    """
    norm_to_orig = {_norm(c): c for c in df.columns}
    pats = [_norm(p) for p in patterns]
    for p in pats:
        if p in norm_to_orig:
            return norm_to_orig[p]
    for p in pats:
        for n, orig in norm_to_orig.items():
            if p in n:
                return orig
    return None


# ─── Base ─────────────────────────────────────────────────────────────────

class BaseTransactionalAdapter:
    name: str = "base"
    version: str = "0.0"
    CAN_HANDLE_HINTS: List[str] = []

    def confidence(self, df: pd.DataFrame, filename: str = "") -> float:
        """0..1 score for the auto-detector. Higher = better match."""
        return 0.0

    def adapt(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, ValidationReport]:
        raise NotImplementedError


# ─── UCI Online Retail II ─────────────────────────────────────────────────
# Source: https://archive.ics.uci.edu/dataset/502/online+retail+ii
# Two .xlsx sheets, columns:
#   Invoice, StockCode, Description, Quantity, InvoiceDate, Price,
#   Customer ID, Country
# (The 2010-2011 sheet uses 'InvoiceNo' and 'UnitPrice' — handle both.)

class OnlineRetailIIAdapter(BaseTransactionalAdapter):
    name = "uci_online_retail_ii"
    version = "1.0"
    CAN_HANDLE_HINTS = ["online_retail", "online retail", "onlineretail", "retail_ii"]

    UCI_KEYWORDS_TO_CATEGORY = {
        # Crude keyword map — keeps a flat dataset usable without an
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
        fname = _norm(filename)
        for h in self.CAN_HANDLE_HINTS:
            if _norm(h) in fname:
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

    def _infer_category(self, description: str) -> str:
        text = str(description).lower()
        for kw, cat in self.UCI_KEYWORDS_TO_CATEGORY.items():
            if kw in text:
                return cat
        return "General Merchandise"

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

        # Build normalized DataFrame
        out = pd.DataFrame({
            "invoice_id":   df[col_invoice].astype(str),
            "product_id":   df[col_stock].astype(str),
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
        n_returns = int(out["invoice_id"].str.upper().str.startswith("C").sum())
        if n_returns:
            out = out[~out["invoice_id"].str.upper().str.startswith("C")]
            report.info.append(f"Dropped {n_returns:,} cancellation rows "
                               f"(InvoiceNo starts with 'C').")

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

        # Category inference from description
        out["category"] = out["product_name"].apply(self._infer_category)
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


# ─── Generic transactional ────────────────────────────────────────────────

class GenericTransactionalAdapter(BaseTransactionalAdapter):
    name = "generic_transactional"
    version = "1.0"
    CAN_HANDLE_HINTS: List[str] = []

    def confidence(self, df: pd.DataFrame, filename: str = "") -> float:
        # Always 0.1 — strictly the fallback after specific adapters.
        return 0.1

    def adapt(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, ValidationReport]:
        report = ValidationReport(
            schema_kind="transactional",
            adapter_name=self.name,
            schema_version=SCHEMA_VERSION,
            rows_in=len(df),
        )

        col_invoice = _find_col(df, ["invoice", "order", "transaction", "basket", "receipt"])
        col_stock = _find_col(df, ["stockcode", "sku", "productid", "product_id", "itemid"])
        col_desc = _find_col(df, ["description", "productname", "product_name",
                                  "product", "name", "item"])
        col_qty = _find_col(df, ["quantity", "qty", "units", "count"])
        col_date = _find_col(df, ["date", "datetime", "timestamp", "time"])
        col_price = _find_col(df, ["unitprice", "unit_price", "price", "amount", "cost"])
        col_cust = _find_col(df, ["customerid", "customer_id", "userid", "user_id"])
        col_cat = _find_col(df, ["category", "department", "type", "class"])

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
            "invoice_id":   df[col_invoice].astype(str),
            "product_id":   df[product_col].astype(str),
            "product_name": (df[col_desc].astype(str)
                             if col_desc else df[product_col].astype(str)),
            "quantity":     pd.to_numeric(df[col_qty], errors="coerce"),
            "timestamp":    pd.to_datetime(df[col_date], errors="coerce"),
            "unit_price":   pd.to_numeric(df[col_price], errors="coerce"),
        })
        if col_cust is not None:
            out["customer_id"] = df[col_cust].astype(str)
        if col_cat is not None:
            out["category"] = df[col_cat].astype(str).fillna("General")
            cat_status, cat_note = "present", ""
        else:
            out["category"] = "General"
            cat_status, cat_note = "inferred", "No category column found — single 'General' bucket."

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

        fs("invoice_id", col_invoice)
        fs("timestamp", col_date)
        fs("product_id", product_col)
        fs("product_name", col_desc,
           status="present" if col_desc else "inferred",
           note="" if col_desc else "Filled from product_id.")
        fs("quantity", col_qty)
        fs("unit_price", col_price)
        fs("customer_id", col_cust,
           status="present" if col_cust else "missing",
           note="" if col_cust else "Return-customer / LTV stats unavailable.")
        fs("category", col_cat, status=cat_status, note=cat_note)

        report.rows_kept = len(out)
        return out, report


# ─── Trajectory adapters ──────────────────────────────────────────────────

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
        fname = _norm(filename)
        for h in self.CAN_HANDLE_HINTS:
            if _norm(h) in fname:
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


class GenericTrajectoryAdapter(BaseTrajectoryAdapter):
    """Fallback adapter: any CSV/TSV with columns track_id, time_s, x_m, y_m
    (or close fuzzy matches). Units are assumed to be SI (seconds, metres);
    if a heuristic finds large values it auto-scales mm -> m."""
    name = "generic_trajectory"
    version = "1.0"

    def confidence(self, df: pd.DataFrame, filename: str = "") -> float:
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

        col_id   = _find_col(df, ["track_id", "person_id", "id", "ped_id",
                                  "agent", "pedestrian"])
        col_time = _find_col(df, ["time_s", "time", "t", "timestamp", "frame"])
        col_x    = _find_col(df, ["x_m", "x", "pos_x", "position_x", "px"])
        col_y    = _find_col(df, ["y_m", "y", "pos_y", "position_y", "py"])

        for need, c in [("track_id", col_id), ("time_s", col_time),
                        ("x_m", col_x), ("y_m", col_y)]:
            fs(need, c, status="present" if c else "missing",
               note="" if c else "Required.")
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


# ─── Registries + auto-detect ─────────────────────────────────────────────

TRANSACTIONAL_ADAPTERS: List[BaseTransactionalAdapter] = [
    OnlineRetailIIAdapter(),
    GenericTransactionalAdapter(),
]

TRAJECTORY_ADAPTERS: List[BaseTrajectoryAdapter] = [
    ATCShoppingMallAdapter(),
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
    """
    try:
        xls = pd.ExcelFile(filename)
    except Exception as e:
        raise ValueError(f"Could not open Excel workbook: {e}") from e
    out: List[Tuple[str, int]] = []
    for name in xls.sheet_names:
        try:
            # Read just the index column to count rows cheaply
            nrows = len(pd.read_excel(filename, sheet_name=name, usecols=[0]))
        except Exception:
            nrows = -1
        out.append((name, nrows))
    return out


def read_excel_sheets(filename: str, sheet_names: List[str]
                      ) -> Tuple[pd.DataFrame, List[str]]:
    """Read one or more named sheets from an Excel workbook and concatenate
    them row-wise. Used after the user picks sheets in the load dialog."""
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
    # across sheets the union is taken and missing cells are NaN — the
    # adapter's dropna step will filter those.
    out = pd.concat(frames, ignore_index=True, sort=False)
    notes.append(f"Concatenated {len(frames)} sheets -> {len(out):,} rows total.")
    return out, notes


def read_any(filename: str) -> Tuple[pd.DataFrame, List[str]]:
    """Read CSV / Excel / TSV / TXT into a DataFrame, returning notes about
    encoding + separator chosen. Raises ValueError on truly unreadable input.

    For multi-sheet Excel workbooks this reads only the first sheet — call
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
                df = pd.read_csv(filename, sep=sep, encoding=enc,
                                 engine="python", on_bad_lines="skip")
                if len(df.columns) >= 2 and len(df) > 0:
                    notes.append(f"Read as text (sep={sep_name}, encoding={enc}).")
                    return df, notes
            except Exception as e:
                last_err = e
                continue

    raise ValueError(f"Could not parse {filename!r}: {last_err}")
