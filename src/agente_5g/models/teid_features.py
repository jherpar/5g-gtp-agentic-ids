"""TEID-level engineered features (src/agente_5g/preprocessing/teid_extractor.py).

This is the feature set arm B (GTP-U ML baseline) and arm C (the agentic
system) both consume — arm A (the official baseline) instead uses
data/processed/Encoded.csv, which never sees any of these fields.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from agente_5g.models.labels import LabelConfidence


class TEIDFeatureRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    teid: int
    base_station: Literal["BS1", "BS2"]
    capture_file: str
    source_attack_type: str
    window_start: float
    window_end: float

    packet_count: int
    byte_count: int
    duration_s: float
    packets_per_s: float
    bytes_per_s: float
    unique_dst_ips: int
    unique_dst_ports: int
    avg_packet_size: float
    std_packet_size: float
    interarrival_mean: float
    interarrival_std: float

    syn_count: int
    ack_count: int
    rst_count: int
    fin_count: int

    teid_lifetime_s: float
    teid_reuse_count: int
    teid_entropy: float
    teid_burstiness: float
    teid_fanout: int
    teid_directionality: float  # uplink_bytes / (uplink_bytes + downlink_bytes)

    # Populated by preprocessing/labeling.py in a separate enrichment pass
    # (Phase 4) that runs after extraction — None here means "not yet
    # labeled", not "known benign". Use `.model_copy(update=...)` to attach
    # labels post hoc since the model is frozen.
    label: str | None = None
    is_attack: bool | None = None
    label_confidence: LabelConfidence | None = None
    label_evidence: list[str] = []
