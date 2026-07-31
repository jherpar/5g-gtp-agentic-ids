"""Typed loader for configs/thresholds.yaml (agent detection rule thresholds).

Kept as a plain data model (not pydantic-settings) since this is reference
data checked into version control and cited in the thesis, not a runtime
overridable setting. Physically separate from
`models/schedule_config.py::LabelPatternsConfig` (Level-3 label validation)
so label quality and agent detection never share thresholds/code.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class FloodRuleConfig(BaseModel):
    min_packets_per_s: float
    max_teid_entropy: float
    max_unique_dst_ports: int


class SynFloodRuleConfig(BaseModel):
    min_syn_count: int
    max_ack_to_syn_ratio: float


class ScanRuleConfig(BaseModel):
    min_unique_dst_ports: int
    max_packets_per_dst_port: float


class TEIDAgentConfig(BaseModel):
    flood: FloodRuleConfig
    syn_flood: SynFloodRuleConfig
    scan: ScanRuleConfig


class StateMachineConfig(BaseModel):
    normal_max: float
    watch_max: float
    suspicious_max: float


class PDUSessionAgentConfig(BaseModel):
    state_machine: StateMachineConfig
    high_state_transition_rate: float
    low_temporal_entropy: float
    high_diversity: int


class FusionWeightsConfig(BaseModel):
    teid_agent: float
    pdu_session_agent: float


class SupervisorAgentConfig(BaseModel):
    fusion_weights: FusionWeightsConfig
    attack_decision_threshold: float


class ThresholdsConfig(BaseModel):
    teid_agent: TEIDAgentConfig
    pdu_session_agent: PDUSessionAgentConfig
    supervisor_agent: SupervisorAgentConfig

    @classmethod
    def load(cls, path: Path) -> ThresholdsConfig:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls(**data)
