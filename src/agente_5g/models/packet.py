"""Packet-level record produced by the parsers (src/agente_5g/parsers/*)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class GTPPacketRecord(BaseModel):
    """One parsed packet from a raw BS1/BS2 pcapng capture.

    `outer_src_ip`/`outer_dst_ip` are always the IP-layer endpoints of the
    captured packet. `inner_*` fields describe the actual transport-layer
    traffic: for GTP-U packets (`is_gtp=True`) these come from the
    encapsulated (tunneled) IP packet and `teid`/`gtp_message_type` are set;
    for non-GTP packets `is_gtp=False`, `teid`/`gtp_message_type` are None,
    and `inner_*` simply describes that packet's own (untunneled)
    protocol/ports. Provenance fields (`capture_file`, `base_station`,
    `source_attack_type`) come from the filename, not packet content, and are
    the Level-1 input to the labeling pipeline.
    """

    model_config = ConfigDict(frozen=True)

    packet_id: int
    capture_file: str
    base_station: Literal["BS1", "BS2"]
    source_attack_type: str

    timestamp: float

    is_gtp: bool
    teid: int | None = None
    gtp_message_type: int | None = None

    outer_src_ip: str
    outer_dst_ip: str
    inner_src_ip: str | None = None
    inner_dst_ip: str | None = None
    inner_src_port: int | None = None
    inner_dst_port: int | None = None
    inner_proto: Literal["tcp", "udp", "icmp", "other"] | None = None

    packet_size: int

    tcp_syn: bool = False
    tcp_ack: bool = False
    tcp_rst: bool = False
    tcp_fin: bool = False

    ue_ip: str | None = None
    direction: Literal["uplink", "downlink", "unknown"] = "unknown"
