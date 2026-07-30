"""Typed contracts for datasets the simulation can ingest.

A *transactional* dataset describes purchases (one row per item-in-an-invoice).
A *trajectory* dataset describes pedestrian positions over time.

Adapters in ``dataset_adapters.py`` are responsible for mapping arbitrary
source files onto these contracts. The simulation calibration pipeline in
``dataset_calibration.py`` consumes only the normalized columns defined here,
so swapping in a new data source never requires touching downstream code.

Versioning rule: bump ``SCHEMA_VERSION`` whenever a required column is added
or renamed. The version is stamped into ``analytics['provenance']`` so an
old simulation report can be paired back to the schema it was calibrated
against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

SCHEMA_VERSION = "1.0.0"


# --- Field descriptors ----------------------------------------------------
# Each schema is a list of FieldSpec. ``required=True`` means calibration
# cannot proceed without it; ``required=False`` means the adapter may infer
# or skip it, optionally adding a note to ValidationReport.warnings.

@dataclass(frozen=True)
class FieldSpec:
    name: str
    description: str
    dtype: str          # 'int', 'float', 'datetime', 'str', 'id'
    unit: str           # 'count', 'GBP', 'seconds', 'meters', '' for ids/strings
    required: bool
    notes: str = ""


TRANSACTIONAL_FIELDS: List[FieldSpec] = [
    FieldSpec(
        name="invoice_id",
        description="Unique identifier per customer visit/basket.",
        dtype="id", unit="", required=True,
        notes="UCI Online Retail II calls this InvoiceNo. One visit = one invoice.",
    ),
    FieldSpec(
        name="timestamp",
        description="When the invoice closed (sub-second resolution preferred).",
        dtype="datetime", unit="seconds since epoch", required=True,
        notes="Used to derive empirical inter-arrival distribution.",
    ),
    FieldSpec(
        name="product_id",
        description="Stable identifier for the SKU.",
        dtype="id", unit="", required=True,
        notes="UCI calls this StockCode; many datasets use SKU/UPC.",
    ),
    FieldSpec(
        name="product_name",
        description="Human-readable product name (used in the visualizer label).",
        dtype="str", unit="", required=False,
        notes="If missing, product_id is used verbatim.",
    ),
    FieldSpec(
        name="quantity",
        description="Number of units of this product on the invoice.",
        dtype="int", unit="count", required=True,
        notes="Negative values are treated as returns and excluded from calibration.",
    ),
    FieldSpec(
        name="unit_price",
        description="Per-unit price in the dataset's currency.",
        dtype="float", unit="currency", required=True,
        notes="Currency is stored in ProvenanceRecord.currency.",
    ),
    FieldSpec(
        name="customer_id",
        description="Stable identifier for the shopping party.",
        dtype="id", unit="", required=False,
        notes="When present, enables return-customer and LTV calibration.",
    ),
    FieldSpec(
        name="category",
        description="Coarse product taxonomy (e.g. 'Beverages', 'Snacks').",
        dtype="str", unit="", required=False,
        notes="If missing, the calibrator runs a keyword-based inference. "
              "Section walls are built per inferred category.",
    ),
]


TRAJECTORY_FIELDS: List[FieldSpec] = [
    FieldSpec(
        name="track_id",
        description="Stable identifier per pedestrian track.",
        dtype="id", unit="", required=True,
        notes="ATC Shopping Mall calls this person_id.",
    ),
    FieldSpec(
        name="time_s",
        description="Time in seconds (relative or absolute).",
        dtype="float", unit="seconds", required=True,
        notes="Frame numbers must be converted via the source's frame rate.",
    ),
    FieldSpec(
        name="x_m",
        description="X position in metres in shop-local coordinates.",
        dtype="float", unit="meters", required=True,
    ),
    FieldSpec(
        name="y_m",
        description="Y position in metres in shop-local coordinates.",
        dtype="float", unit="meters", required=True,
    ),
    FieldSpec(
        name="floor",
        description="Floor index (1 = ground).",
        dtype="int", unit="", required=False,
        notes="If missing, all tracks are placed on floor 1.",
    ),
]


# --- Validation report ----------------------------------------------------

@dataclass
class FieldStatus:
    """Per-column outcome of validating a source against a schema."""
    name: str
    status: str         # 'present', 'inferred', 'missing', 'invalid'
    source_column: Optional[str] = None
    note: str = ""


@dataclass
class ValidationReport:
    """Structured outcome of running an adapter over a source DataFrame.

    A report is *blocking* when it has any field with ``status='missing'``
    AND that field is marked ``required=True`` in the schema. The UI uses
    this to decide whether to offer 'Calibrate' or to ask the user to fix
    the source file.
    """
    schema_kind: str                       # 'transactional' or 'trajectory'
    adapter_name: str
    schema_version: str = SCHEMA_VERSION
    fields: List[FieldStatus] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    rows_in: int = 0
    rows_kept: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def field_specs(self) -> List[FieldSpec]:
        return (TRANSACTIONAL_FIELDS if self.schema_kind == 'transactional'
                else TRAJECTORY_FIELDS)

    def is_blocking(self) -> bool:
        spec_required = {f.name for f in self.field_specs() if f.required}
        for fs in self.fields:
            if fs.name in spec_required and fs.status in ('missing', 'invalid'):
                return True
        return False

    def missing_required(self) -> List[FieldStatus]:
        spec_required = {f.name for f in self.field_specs() if f.required}
        return [fs for fs in self.fields
                if fs.name in spec_required
                and fs.status in ('missing', 'invalid')]

    def to_text(self) -> str:
        """Human-readable summary for the validation dialog."""
        lines = [
            f"Adapter:        {self.adapter_name}",
            f"Schema:         {self.schema_kind} (v{self.schema_version})",
            f"Rows in/kept:   {self.rows_in:,} → {self.rows_kept:,}",
            "",
            "COLUMNS",
            "-" * 60,
        ]
        for fs in self.fields:
            marker = {
                'present':  '[OK]',
                'inferred': '[~ ]',
                'missing':  '[XX]',
                'invalid':  '[!!]',
            }.get(fs.status, '[??]')
            src = f"  <- {fs.source_column}" if fs.source_column else ""
            lines.append(f"  {marker} {fs.name:<14} [{fs.status}]{src}")
            if fs.note:
                lines.append(f"      {fs.note}")
        if self.warnings:
            lines.append("")
            lines.append("WARNINGS")
            lines.append("-" * 60)
            for w in self.warnings:
                lines.append(f"  • {w}")
        if self.info:
            lines.append("")
            lines.append("NOTES")
            lines.append("-" * 60)
            for note in self.info:
                lines.append(f"  • {note}")
        return "\n".join(lines)
