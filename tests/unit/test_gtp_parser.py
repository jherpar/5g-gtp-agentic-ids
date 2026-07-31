from __future__ import annotations

from pathlib import Path

import pytest

from agente_5g.parsers.scapy_parser import ScapyPacketParser
from tests.fixtures import synthetic_packets as sp


@pytest.fixture
def parser() -> ScapyPacketParser:
    return ScapyPacketParser()


def _parse_single(parser: ScapyPacketParser, tmp_path: Path, packet, filename: str = "test.pcapng"):
    pcap_path = sp.write_pcap([packet], tmp_path / filename)
    records = list(
        parser.parse_file(pcap_path, base_station="BS1", source_attack_type="TestAttack")
    )
    assert len(records) == 1
    return records[0]


def test_gtp_tcp_syn_extracts_teid_and_inner_flow(parser: ScapyPacketParser, tmp_path: Path):
    pkt = sp.gtp_tcp_syn_packet(teid=0x11223344, inner_sport=51000, inner_dport=80)
    record = _parse_single(parser, tmp_path, pkt)

    assert record.is_gtp is True
    assert record.teid == 0x11223344
    assert record.gtp_message_type == sp.G_PDU_MESSAGE_TYPE
    assert record.inner_src_ip == "12.0.0.5"
    assert record.inner_dst_ip == "93.184.216.34"
    assert record.inner_src_port == 51000
    assert record.inner_dst_port == 80
    assert record.inner_proto == "tcp"
    assert record.tcp_syn is True
    assert record.tcp_ack is False
    assert record.tcp_rst is False
    assert record.tcp_fin is False
    assert record.outer_src_ip == "10.0.0.1"
    assert record.outer_dst_ip == "10.0.0.2"


def test_gtp_udp_flood_extracts_teid_and_inner_flow(parser: ScapyPacketParser, tmp_path: Path):
    pkt = sp.gtp_udp_flood_packet(teid=0xAABBCCDD, inner_sport=40000, inner_dport=53)
    record = _parse_single(parser, tmp_path, pkt)

    assert record.is_gtp is True
    assert record.teid == 0xAABBCCDD
    assert record.inner_proto == "udp"
    assert record.inner_src_port == 40000
    assert record.inner_dst_port == 53
    assert record.tcp_syn is False


def test_gtp_icmp_extracts_inner_icmp(parser: ScapyPacketParser, tmp_path: Path):
    pkt = sp.gtp_icmp_packet(teid=0x55667788)
    record = _parse_single(parser, tmp_path, pkt)

    assert record.is_gtp is True
    assert record.teid == 0x55667788
    assert record.inner_proto == "icmp"
    assert record.inner_src_port is None


def test_non_gtp_tcp_packet_is_flagged_and_still_extracted(
    parser: ScapyPacketParser, tmp_path: Path
):
    pkt = sp.non_gtp_tcp_packet(sport=1234, dport=443, flags="A")
    record = _parse_single(parser, tmp_path, pkt)

    assert record.is_gtp is False
    assert record.teid is None
    assert record.gtp_message_type is None
    assert record.inner_proto == "tcp"
    assert record.inner_src_port == 1234
    assert record.inner_dst_port == 443
    assert record.tcp_ack is True
    assert record.tcp_syn is False


def test_malformed_gtp_header_does_not_crash_and_yields_no_teid(
    parser: ScapyPacketParser, tmp_path: Path
):
    pkt = sp.malformed_gtp_packet()
    record = _parse_single(parser, tmp_path, pkt)

    assert record.is_gtp is True
    assert record.teid is None
    assert record.gtp_message_type is None
    assert record.inner_src_ip is None


def test_gtp_echo_request_has_no_inner_ip_but_header_parses(
    parser: ScapyPacketParser, tmp_path: Path
):
    pkt = sp.gtp_echo_request_packet()
    record = _parse_single(parser, tmp_path, pkt)

    assert record.is_gtp is True
    assert record.gtp_message_type == 1
    assert record.inner_src_ip is None
    assert record.inner_proto is None


def test_multiple_packets_streamed_in_order(parser: ScapyPacketParser, tmp_path: Path):
    packets = [
        sp.gtp_tcp_syn_packet(teid=1),
        sp.gtp_udp_flood_packet(teid=2),
        sp.non_gtp_tcp_packet(),
    ]
    for i, pkt in enumerate(packets):
        pkt.time = 1000.0 + i

    pcap_path = sp.write_pcap(packets, tmp_path / "multi.pcapng")
    records = list(
        ScapyPacketParser().parse_file(pcap_path, base_station="BS2", source_attack_type="UDPflood")
    )

    assert [r.packet_id for r in records] == [0, 1, 2]
    assert all(r.base_station == "BS2" for r in records)
    assert all(r.source_attack_type == "UDPflood" for r in records)
    assert records[0].teid == 1
    assert records[1].teid == 2
    assert records[2].is_gtp is False


def test_max_packets_truncates_stream(parser: ScapyPacketParser, tmp_path: Path):
    packets = [sp.gtp_udp_flood_packet(teid=i) for i in range(5)]
    for i, pkt in enumerate(packets):
        pkt.time = 2000.0 + i

    pcap_path = sp.write_pcap(packets, tmp_path / "many.pcapng")
    records = list(
        parser.parse_file(
            pcap_path, base_station="BS1", source_attack_type="UDPflood", max_packets=2
        )
    )
    assert len(records) == 2


def test_max_duration_truncates_stream(parser: ScapyPacketParser, tmp_path: Path):
    packets = [sp.gtp_udp_flood_packet(teid=i) for i in range(5)]
    for i, pkt in enumerate(packets):
        pkt.time = 3000.0 + i * 10.0  # 10s apart

    pcap_path = sp.write_pcap(packets, tmp_path / "spread.pcapng")
    records = list(
        parser.parse_file(
            pcap_path, base_station="BS1", source_attack_type="UDPflood", max_duration_s=15.0
        )
    )
    # packets at t=0,10,20,30,40 relative to first (0s); duration cutoff 15s -> keep t=0,10
    assert len(records) == 2
