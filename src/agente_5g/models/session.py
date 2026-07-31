"""Inferred PDU session record (src/agente_5g/preprocessing/session_builder.py).

NSA doesn't expose PDU sessions explicitly at the GTP-U layer we can observe,
so a session is inferred by grouping UE_IP + TEID over a temporal window;
`session_id` is deterministic so re-runs are reproducible.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from agente_5g.models.labels import LabelConfidence


class PDUSessionRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str  # sha256(f"{ue_ip}|{teid}|{start_time}")
    ue_ip: str
    teid: int
    window_size_s: Literal[1, 5, 10, 30]
    start_time: float
    end_time: float
    duration_s: float

    traffic_volume_bytes: int
    flow_diversity: int
    port_diversity: int
    destination_diversity: int
    state_transition_rate: float
    temporal_entropy: float

    state_sequence: list[str]
    final_state: Literal["NORMAL", "WATCH", "SUSPICIOUS", "ATTACK"]

    label: str
    is_attack: bool
    label_confidence: LabelConfidence
    label_evidence: list[str] = []
