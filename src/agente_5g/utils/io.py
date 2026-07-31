"""Run reproducibility helpers: config snapshots and content hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from agente_5g.settings import Settings


def config_hash(settings: Settings) -> str:
    """Stable sha256 of the resolved, canonicalized settings."""
    canonical = json.dumps(settings.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def snapshot_config(settings: Settings, run_id: str) -> Path:
    """Write outputs/reports/<run_id>/config.yaml with the resolved settings."""
    out_dir = settings.resolve_path(settings.paths.outputs) / "reports" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = out_dir / "config.yaml"
    with snapshot_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(settings.model_dump(mode="json"), fh, sort_keys=True)
    return snapshot_path


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        manifest: dict[str, Any] = json.load(fh)
        return manifest


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True, default=str)
