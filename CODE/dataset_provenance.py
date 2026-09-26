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


def _sha256_path(path: str) -> tuple:
    """(sha256, total_bytes) for a file OR a directory source.

    Directory sources (e.g. the Omnichannel multi-CSV bundle) are hashed
    deterministically: every regular file, sorted by its relative path,
    contributes its path string and content to one running digest. A
    reviewer with the folder can recompute the hash exactly."""
    if not os.path.isdir(path):
        return _sha256_file(path), os.path.getsize(path)
    h = hashlib.sha256()
    total = 0
    entries = []
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for f in sorted(files):
            full = os.path.join(root, f)
            entries.append((os.path.relpath(full, path).replace(os.sep, '/'),
                            full))
    for rel, full in sorted(entries):
        h.update(rel.encode('utf-8', errors='replace'))
        h.update(b'\x00')
        try:
            with open(full, 'rb') as fh:
                while True:
                    blk = fh.read(1 << 20)
                    if not blk:
                        break
                    h.update(blk)
            total += os.path.getsize(full)
        except OSError:
            # Unreadable member (locked etc.) -- record its name only so
            # the digest is still deterministic and the load never dies.
            h.update(b'<unreadable>')
    return h.hexdigest(), total


def source_digest(path: str) -> Dict[str, Any]:
    """``{'source_sha256', 'source_bytes'}`` of a file or directory source,
    hashed as ``stamp`` hashes it -- for records that describe the source
    without a whole ``ProvenanceRecord`` (the live runners' period
    record), under the same field names."""
    sha, nbytes = _sha256_path(path)
    return {'source_sha256': sha, 'source_bytes': int(nbytes)}


def _pkg_versions() -> Dict[str, str]:
    """Resolved numerics-stack versions (audit R5.1): stamped into every
    provenance record so a reviewer can confirm the exact computation
    environment, not just the input hash."""
    from importlib.metadata import version, PackageNotFoundError
    out: Dict[str, str] = {}
    for pkg in ('numpy', 'scipy', 'pandas', 'matplotlib', 'openpyxl'):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = 'not-installed'
    return out


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
    packages: Dict[str, str] = field(default_factory=lambda: _pkg_versions())
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
    """Build a ProvenanceRecord by hashing the source on disk. The source
    may be a single file or a directory bundle (multi-CSV datasets)."""
    sha, nbytes = _sha256_path(source_path)
    return ProvenanceRecord(
        source_path=source_path,
        source_bytes=nbytes,
        source_sha256=sha,
        adapter_name=adapter_name,
        adapter_version=adapter_version,
        schema_version=SCHEMA_VERSION,
        rows_in=rows_in,
        rows_kept=rows_kept,
        currency=currency,
        seed=seed,
        extra=extra or {},
    )
