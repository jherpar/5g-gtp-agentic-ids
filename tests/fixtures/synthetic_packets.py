"""Hand-crafted Scapy packets for parser unit tests.

Never commit real pcap binaries — these are built at test time and written
to `tmp_path` via `wrpcap` by the tests that need a file on disk.
"""

from __future__ import annotations

from pathlib import Path

from scapy.contrib import gtp
from scapy.layers.inet import ICMP, IP, TCP, UDP
from scapy.layers.l2 import Ether
from scapy.packet import Raw
from scapy.utils import wrpcap

GTP_U_PORT = 2152
G_PDU_MESSAGE_TYPE = 255


def gtp_tcp_syn_packet(
    teid: int = 0x11223344,
    outer_src: str = "10.0.0.1",
    outer_dst: str = "10.0.0.2",
    inner_src: str = "12.0.0.5",
    inner_dst: str = "93.184.216.34",
    inner_sport: int = 51000,
    inner_dport: int = 80,
):
    """A GTP-U-encapsulated TCP SYN packet (uplink connection attempt)."""
    inner = IP(src=inner_src, dst=inner_dst) / TCP(sport=inner_sport, dport=inner_dport, flags="S")
    return (
        Ether()
        / IP(src=outer_src, dst=outer_dst)
        / UDP(sport=GTP_U_PORT, dport=GTP_U_PORT)
        / gtp.GTP_U_Header(teid=teid, gtp_type=G_PDU_MESSAGE_TYPE)
        / inner
    )


def gtp_udp_flood_packet(
    teid: int = 0xAABBCCDD,
    outer_src: str = "10.0.0.1",
    outer_dst: str = "10.0.0.2",
    inner_src: str = "12.0.0.9",
    inner_dst: str = "93.184.216.34",
    inner_sport: int = 40000,
    inner_dport: int = 53,
    payload: bytes = b"X" * 32,
):
    """A GTP-U-encapsulated UDP packet, representative of a UDP flood."""
    inner = (
        IP(src=inner_src, dst=inner_dst)
        / UDP(sport=inner_sport, dport=inner_dport)
        / Raw(load=payload)
    )
    return (
        Ether()
        / IP(src=outer_src, dst=outer_dst)
        / UDP(sport=GTP_U_PORT, dport=GTP_U_PORT)
        / gtp.GTP_U_Header(teid=teid, gtp_type=G_PDU_MESSAGE_TYPE)
        / inner
    )


def gtp_icmp_packet(
    teid: int = 0x55667788,
    outer_src: str = "10.0.0.1",
    outer_dst: str = "10.0.0.2",
    inner_src: str = "12.0.0.9",
    inner_dst: str = "93.184.216.34",
):
    """A GTP-U-encapsulated ICMP echo request."""
    inner = IP(src=inner_src, dst=inner_dst) / ICMP(type=8)
    return (
        Ether()
        / IP(src=outer_src, dst=outer_dst)
        / UDP(sport=GTP_U_PORT, dport=GTP_U_PORT)
        / gtp.GTP_U_Header(teid=teid, gtp_type=G_PDU_MESSAGE_TYPE)
        / inner
    )


def non_gtp_tcp_packet(
    src: str = "10.0.0.5",
    dst: str = "10.0.0.6",
    sport: int = 1234,
    dport: int = 443,
    flags: str = "A",
):
    """Plain (non-tunneled) TCP traffic seen at the capture point."""
    return Ether() / IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, flags=flags)


def malformed_gtp_packet(outer_src: str = "10.0.0.1", outer_dst: str = "10.0.0.2"):
    """UDP/2152 payload too short to contain the mandatory 8-byte GTP-U header."""
    return (
        Ether()
        / IP(src=outer_src, dst=outer_dst)
        / UDP(sport=GTP_U_PORT, dport=GTP_U_PORT)
        / Raw(load=b"\x30\xff\x00")
    )


def gtp_echo_request_packet(outer_src: str = "10.0.0.1", outer_dst: str = "10.0.0.2"):
    """A GTP-U control message (Echo Request, type 1) — no inner IP payload."""
    return (
        Ether()
        / IP(src=outer_src, dst=outer_dst)
        / UDP(sport=GTP_U_PORT, dport=GTP_U_PORT)
        / gtp.GTP_U_Header(teid=0, gtp_type=1)
    )


def write_pcap(packets: list, path: Path) -> Path:
    wrpcap(str(path), packets)
    return path
