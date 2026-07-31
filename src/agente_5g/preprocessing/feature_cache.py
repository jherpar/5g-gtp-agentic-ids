"""Parquet-based caching for parsed packets and engineered features.

Parsing a single large pcapng file (up to ~660MB) takes minutes, and this
project's iterative validation/diagnostic scripts
(scripts/validate_labeling.py, validate_labeling_all.py,
diagnose_confidence_system.py) have already re-parsed the same 9 files
repeatedly across investigation rounds. Caching to parquet after each
expensive step -- packet parsing (the dominant cost) and TEID/session
feature extraction -- lets repeat runs skip straight to the cheap analysis
step instead of re-reading gigabytes of pcapng.

Not wired into the pipeline automatically: callers decide when to read/write
the cache (see `cached_or_parse` for the common "load if present, else
compute and save" pattern).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TypeVar

import pandas as pd
from pydantic import BaseModel

from agente_5g.models.packet import GTPPacketRecord
from agente_5g.models.session import PDUSessionRecord
from agente_5g.models.teid_features import TEIDFeatureRecord

ModelT = TypeVar("ModelT", bound=BaseModel)


def save_records(records: Iterable[BaseModel], path: Path) -> None:
    """Write any flat Pydantic model list to parquet (JSON-safe field
    encoding, so enums/Literals round-trip through `model_validate`)."""
    rows = [r.model_dump(mode="json") for r in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # An empty file with no schema is ambiguous to reload; callers
        # should treat "no cache file" and "cached empty result" the same
        # way, so we simply don't write one.
        return
    pd.DataFrame(rows).to_parquet(path, index=False)


def load_records(path: Path, model: type[ModelT]) -> list[ModelT]:
    df = pd.read_parquet(path)
    return [model.model_validate(row) for row in df.to_dict(orient="records")]


def save_packets(records: Iterable[GTPPacketRecord], path: Path) -> None:
    save_records(records, path)


def load_packets(path: Path) -> list[GTPPacketRecord]:
    return load_records(path, GTPPacketRecord)


def save_teid_features(records: Iterable[TEIDFeatureRecord], path: Path) -> None:
    save_records(records, path)


def load_teid_features(path: Path) -> list[TEIDFeatureRecord]:
    return load_records(path, TEIDFeatureRecord)


def save_sessions(records: Iterable[PDUSessionRecord], path: Path) -> None:
    save_records(records, path)


def load_sessions(path: Path) -> list[PDUSessionRecord]:
    return load_records(path, PDUSessionRecord)


def cached_or_compute(
    path: Path, model: type[ModelT], compute: Callable[[], list[ModelT]]
) -> list[ModelT]:
    """Load `path` if it exists, else call `compute()` and cache the result."""
    if path.exists():
        return load_records(path, model)
    records = compute()
    save_records(records, path)
    return records
