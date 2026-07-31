"""Typed, reproducible configuration loading.

Settings are assembled from `configs/base.yaml` plus a mode-specific overlay
(`configs/sample.yaml` or `configs/full.yaml`), then environment variables
(prefix `AGENTE5G_`) may override any field. Every pipeline run should build
its `Settings` once via `Settings.load(...)` and pass it explicitly rather
than reading YAML ad hoc, so a run's resolved configuration can be snapshotted
for reproducibility (see `utils/io.py::snapshot_config`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"


class PathsConfig(BaseModel):
    data_raw: Path = Path("data/raw")
    data_processed: Path = Path("data/processed")
    data_features: Path = Path("data/features")
    outputs: Path = Path("outputs")


class SampleConfig(BaseModel):
    max_packets_per_file: int | None = 20000
    max_files: int | None = None
    max_duration_s: float | None = 60.0


class FullRunConfig(BaseModel):
    checkpointing: bool = True
    max_workers: int = 2


class SessionConfig(BaseModel):
    window_sizes_s: list[int] = Field(default_factory=lambda: [1, 5, 10, 30])


class TEIDConfig(BaseModel):
    idle_gap_reuse_threshold_s: float = 30.0


class LabelingConfig(BaseModel):
    schedule_config: Path = Path("configs/attack_schedule.yaml")
    pattern_config: Path = Path("configs/label_patterns.yaml")


class RandomForestConfig(BaseModel):
    n_estimators: int = 200
    max_depth: int | None = None
    random_state: int = 42


class XGBoostConfig(BaseModel):
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.1
    random_state: int = 42


class MLConfig(BaseModel):
    random_forest: RandomForestConfig = RandomForestConfig()
    xgboost: XGBoostConfig = XGBoostConfig()
    split_strategy: str = "chronological_per_file"
    test_size: float = 0.3


class LLMConfig(BaseModel):
    enabled: bool = False
    endpoint: str = "http://localhost:11434/api/generate"
    model: str = "gemma4:latest"
    fallback_model: str = "llama3.2:1b"
    cache_path: Path = Path("outputs/logs/llm_explanations_cache.jsonl")
    timeout_s: int = 30


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGENTE5G_", env_nested_delimiter="__")

    seed: int = 42
    mode: Literal["sample", "full"] = "sample"
    base_stations: list[str] = Field(default_factory=lambda: ["BS1", "BS2"])
    attack_types: list[str] = Field(default_factory=list)

    paths: PathsConfig = Field(default_factory=PathsConfig)
    sample: SampleConfig = Field(default_factory=SampleConfig)
    full: FullRunConfig = Field(default_factory=FullRunConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    teid: TEIDConfig = Field(default_factory=TEIDConfig)
    labeling: LabelingConfig = Field(default_factory=LabelingConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    @classmethod
    def load(cls, mode: Literal["sample", "full"] | None = None) -> Settings:
        """Load base.yaml, merge the mode-specific overlay, then env vars."""
        base = _read_yaml(CONFIGS_DIR / "base.yaml")
        resolved_mode = mode or base.get("mode", "sample")
        overlay_path = CONFIGS_DIR / f"{resolved_mode}.yaml"
        overlay = _read_yaml(overlay_path)
        merged = _deep_merge(base, overlay)
        merged["mode"] = resolved_mode
        return cls(**merged)

    def resolve_path(self, relative: Path | str) -> Path:
        """Resolve a config-relative path against the project root."""
        path = Path(relative)
        return path if path.is_absolute() else PROJECT_ROOT / path


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
