"""Hand-built GTPPacketRecord instances for preprocessing unit tests.

Unlike synthetic_packets.py (raw Scapy packets, for parser tests), these
skip packet bytes entirely and construct the parsed model directly, since
teid_extractor/session_builder tests want full control over specific field
combinations without going through packet dissection.
"""

from __future__ import annotations

from typing import Literal

from agente_5g.models.packet import GTPPacketRecord


def make_packet(
    packet_id: int,
    teid: int | None,
    timestamp: float,
    packet_size: int = 100,
    inner_src_ip: str | None = "12.0.0.5",
    inner_dst_ip: str | None = "93.184.216.34",
    inner_src_port: int | None = 51000,
    inner_dst_port: int | None = 80,
    inner_proto: str | None = "tcp",
    tcp_syn: bool = False,
    tcp_ack: bool = False,
    tcp_rst: bool = False,
    tcp_fin: bool = False,
    is_gtp: bool = True,
    base_station: Literal["BS1", "BS2"] = "BS1",
    capture_file: str = "TestAttack_BS1.pcapng",
    source_attack_type: str = "TestAttack",
) -> GTPPacketRecord:
    return GTPPacketRecord(
        packet_id=packet_id,
        capture_file=capture_file,
        base_station=base_station,
        source_attack_type=source_attack_type,
        timestamp=timestamp,
        is_gtp=is_gtp,
        teid=teid if is_gtp else None,
        gtp_message_type=255 if is_gtp else None,
        outer_src_ip="10.0.0.1",
        outer_dst_ip="10.0.0.2",
        inner_src_ip=inner_src_ip,
        inner_dst_ip=inner_dst_ip,
        inner_src_port=inner_src_port,
        inner_dst_port=inner_dst_port,
        inner_proto=inner_proto,
        packet_size=packet_size,
        tcp_syn=tcp_syn,
        tcp_ack=tcp_ack,
        tcp_rst=tcp_rst,
        tcp_fin=tcp_fin,
    )
