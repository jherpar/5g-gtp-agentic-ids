"""Typed loaders for configs/attack_schedule.yaml and configs/label_patterns.yaml.

Kept as plain data models (not pydantic-settings) since these are reference
data checked into version control and cited in the thesis, not runtime
overridable settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


class AttackScheduleEntry(BaseModel):
    official_name: str
    benign_only: bool = False
    date: str | None = None
    session_window: tuple[str, str] | None = None
    attack_window: dict[Literal["BS1", "BS2"], tuple[str, str]] | None = None


class AttackSchedule(BaseModel):
    calibrated: bool
    timezone: str
    year: int | None
    victim_ip: str
    attacker_ip_hints: dict[Literal["BS1", "BS2"], str] = {}
    attacks: dict[str, AttackScheduleEntry]

    @classmethod
    def load(cls, path: Path) -> AttackSchedule:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls(**data)


class FloodPatternConfig(BaseModel):
    min_sustained_packets_per_s: float
    min_window_s: float
    max_packet_size_entropy: float
    max_unique_dst_ports: int


class ScanPatternConfig(BaseModel):
    min_unique_dst_ports_per_source: int
    max_window_s: float


class SlowratePatternConfig(BaseModel):
    min_connection_duration_s: float
    max_bytes_per_s: float
    min_concurrent_connections: int


class LabelPatternsConfig(BaseModel):
    flood_pattern: FloodPatternConfig
    scan_pattern: ScanPatternConfig
    slowrate_pattern: SlowratePatternConfig

    @classmethod
    def load(cls, path: Path) -> LabelPatternsConfig:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls(**data)
