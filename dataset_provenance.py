"""Provenance metadata stamped onto every dataset-calibrated simulation run.

Stored at ``simulation.analytics['provenance']`` and surfaced in optimization
reports + the Validation tab. A reviewer with the source file should be able
to recompute the SHA-256, confirm the schema version, and re-seed RNGs to
reproduce a reported number exactly.
"""

from __future__ import annotations

import hashlib
import os
import platform
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any

from dataset_schema import SCHEMA_VERSION


def _sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            blk = f.read(chunk)
            if not blk:
                break
            h.update(blk)
    return h.hexdigest()


@dataclass
class ProvenanceRecord:
    source_path: str
    source_bytes: int
    source_sha256: str
    adapter_name: str
    adapter_version: str
    schema_version: str
    rows_in: int
    rows_kept: int
    currency: Optional[str] = None
    seed: Optional[int] = None
    loaded_at: float = field(default_factory=time.time)
    loaded_at_iso: str = field(default_factory=lambda:
                               time.strftime("%Y-%m-%dT%H:%M:%S",
                                             time.localtime()))
    python: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=lambda: platform.platform())
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_text(self) -> str:
        return (
            f"Source:          {os.path.basename(self.source_path)}\n"
            f"  bytes:         {self.source_bytes:,}\n"
            f"  sha256:        {self.source_sha256}\n"
            f"Adapter:         {self.adapter_name} v{self.adapter_version}\n"
            f"Schema:          v{self.schema_version}\n"
            f"Rows in / kept:  {self.rows_in:,} / {self.rows_kept:,}\n"
            f"Currency:        {self.currency or '-'}\n"
            f"Seed:            {self.seed if self.seed is not None else '-'}\n"
            f"Loaded:          {self.loaded_at_iso}\n"
            f"Python:          {self.python}\n"
            f"Platform:        {self.platform}\n"
        )


def stamp(source_path: str,
          adapter_name: str,
          adapter_version: str,
          rows_in: int,
          rows_kept: int,
          currency: Optional[str] = None,
          seed: Optional[int] = None,
          extra: Optional[Dict[str, Any]] = None) -> ProvenanceRecord:
    """Build a ProvenanceRecord by hashing the source file on disk."""
    return ProvenanceRecord(
        source_path=source_path,
        source_bytes=os.path.getsize(source_path),
        source_sha256=_sha256_file(source_path),
        adapter_name=adapter_name,
        adapter_version=adapter_version,
        schema_version=SCHEMA_VERSION,
        rows_in=rows_in,
        rows_kept=rows_kept,
        currency=currency,
        seed=seed,
        extra=extra or {},
    )
