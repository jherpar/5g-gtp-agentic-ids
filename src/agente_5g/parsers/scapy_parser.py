"""Primary PacketParser backend: pure-Python Scapy, streaming.

Deliberately uses `PcapReader` (streaming) rather than `rdpcap` (loads the
whole file into memory) — the largest captures in this dataset are ~660MB,
and per-packet Python objects can multiply that several times over in
memory if loaded all at once (see plan risk #2).

GTP-U dissection: tries scapy's `contrib.gtp` layer first; if that layer
isn't present on the packet (contrib unavailable, or scapy didn't recognize
it) it falls back to `gtp_layers.decode_gtp_u_header_fallback`, which decodes
the fixed 8-byte GTP-U header directly from bytes. Either way, the payload
following the GTP-U header is re-dissected as an inner IP packet; GTP-U
control messages (echo request/response, error indication) carry no inner IP
payload, so `_parse_inner_ip` simply returns None for those and the record's
inner_* fields stay unset rather than raising.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from agente_5g.models.packet import GTPPacketRecord
from agente_5g.parsers.base import PacketParser
from agente_5g.parsers.gtp_layers import (
    GTP_U_PORT,
    GtpUHeader,
    decode_gtp_u_header_fallback,
    ensure_gtp_contrib_loaded,
)
from agente_5g.utils.logging import get_logger

logger = get_logger(__name__)


class ScapyPacketParser(PacketParser):
    def __init__(self) -> None:
        self._gtp_contrib_available = ensure_gtp_contrib_loaded()

    def parse_file(
        self,
        path: Path,
        base_station: Literal["BS1", "BS2"],
        source_attack_type: str,
        max_packets: int | None = None,
        max_duration_s: float | None = None,
    ) -> Iterator[GTPPacketRecord]:
        from scapy.utils import PcapReader

        packet_id = 0
        first_ts: float | None = None
        skipped = 0

        with PcapReader(str(path)) as reader:
            for scapy_pkt in reader:
                if max_packets is not None and packet_id >= max_packets:
                    break

                ts = float(scapy_pkt.time)
                if first_ts is None:
                    first_ts = ts
                if max_duration_s is not None and (ts - first_ts) > max_duration_s:
                    break

                try:
                    record = self._parse_packet(
                        scapy_pkt, packet_id, path.name, base_station, source_attack_type, ts
                    )
                except Exception:
                    logger.debug("Skipping unparseable packet #%d in %s", packet_id, path.name)
                    record = None
                    skipped += 1

                packet_id += 1
                if record is not None:
                    yield record

        if skipped:
            logger.info(
                "%s: skipped %d unparseable/non-IP packets out of %d", path.name, skipped, packet_id
            )

    def _parse_packet(
        self,
        scapy_pkt: Any,
        packet_id: int,
        capture_file: str,
        base_station: Literal["BS1", "BS2"],
        source_attack_type: str,
        ts: float,
    ) -> GTPPacketRecord | None:
        from scapy.layers.inet import ICMP, IP, TCP, UDP

        if IP not in scapy_pkt:
            return None  # out of scope: ARP and other non-IP link traffic

        ip_layer = scapy_pkt[IP]
        outer_src_ip = str(ip_layer.src)
        outer_dst_ip = str(ip_layer.dst)
        packet_size = len(scapy_pkt)

        is_gtp = False
        teid: int | None = None
        gtp_message_type: int | None = None
        inner_src_ip = inner_dst_ip = None
        inner_src_port = inner_dst_port = None
        inner_proto = None
        tcp_syn = tcp_ack = tcp_rst = tcp_fin = False

        udp_layer = scapy_pkt[UDP] if UDP in scapy_pkt else None
        if udp_layer is not None and GTP_U_PORT in (udp_layer.sport, udp_layer.dport):
            is_gtp = True
            header, inner = self._dissect_gtp(scapy_pkt)
            if header is not None:
                teid = header.teid
                gtp_message_type = header.message_type
            if inner is not None:
                inner_src_ip = inner.get("src")
                inner_dst_ip = inner.get("dst")
                inner_proto = inner.get("proto")
                inner_src_port = inner.get("sport")
                inner_dst_port = inner.get("dport")
                tcp_syn = bool(inner.get("syn", False))
                tcp_ack = bool(inner.get("ack", False))
                tcp_rst = bool(inner.get("rst", False))
                tcp_fin = bool(inner.get("fin", False))
        elif TCP in scapy_pkt:
            tcp_layer = scapy_pkt[TCP]
            flags = int(tcp_layer.flags)
            inner_proto = "tcp"
            inner_src_port = int(tcp_layer.sport)
            inner_dst_port = int(tcp_layer.dport)
            tcp_syn = bool(flags & 0x02)
            tcp_ack = bool(flags & 0x10)
            tcp_rst = bool(flags & 0x04)
            tcp_fin = bool(flags & 0x01)
        elif UDP in scapy_pkt:
            inner_proto = "udp"
            inner_src_port = int(scapy_pkt[UDP].sport)
            inner_dst_port = int(scapy_pkt[UDP].dport)
        elif ICMP in scapy_pkt:
            inner_proto = "icmp"
        else:
            inner_proto = "other"

        return GTPPacketRecord(
            packet_id=packet_id,
            capture_file=capture_file,
            base_station=base_station,
            source_attack_type=source_attack_type,
            timestamp=ts,
            is_gtp=is_gtp,
            teid=teid,
            gtp_message_type=gtp_message_type,
            outer_src_ip=outer_src_ip,
            outer_dst_ip=outer_dst_ip,
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

    def _dissect_gtp(self, scapy_pkt: Any) -> tuple[GtpUHeader | None, dict[str, Any] | None]:
        from scapy.layers.inet import UDP

        udp_payload = bytes(scapy_pkt[UDP].payload)
        header: GtpUHeader | None = None
        remaining = udp_payload

        if self._gtp_contrib_available:
            try:
                from scapy.contrib import gtp

                if gtp.GTP_U_Header in scapy_pkt:
                    gtp_layer = scapy_pkt[gtp.GTP_U_Header]
                    payload_bytes = bytes(gtp_layer.payload)
                    header = GtpUHeader(
                        version=int(gtp_layer.version),
                        protocol_type=int(gtp_layer.PT),
                        message_type=int(gtp_layer.gtp_type),
                        length=int(gtp_layer.length),
                        teid=int(gtp_layer.teid),
                        header_len=len(bytes(gtp_layer)) - len(payload_bytes),
                    )
                    remaining = payload_bytes
            except Exception:
                header = None

        if header is None:
            header = decode_gtp_u_header_fallback(udp_payload)
            if header is None:
                return None, None
            remaining = udp_payload[header.header_len :]

        return header, self._parse_inner_ip(remaining)

    def _parse_inner_ip(self, raw: bytes) -> dict[str, Any] | None:
        if len(raw) < 20:
            return None  # e.g. GTP echo/error-indication: no inner IP payload

        from scapy.layers.inet import ICMP, IP, TCP, UDP

        try:
            inner_pkt = IP(raw)
        except Exception:
            return None

        info: dict[str, Any] = {"src": str(inner_pkt.src), "dst": str(inner_pkt.dst)}
        if TCP in inner_pkt:
            tcp_layer = inner_pkt[TCP]
            flags = int(tcp_layer.flags)
            info.update(
                proto="tcp",
                sport=int(tcp_layer.sport),
                dport=int(tcp_layer.dport),
                syn=bool(flags & 0x02),
                ack=bool(flags & 0x10),
                rst=bool(flags & 0x04),
                fin=bool(flags & 0x01),
            )
        elif UDP in inner_pkt:
            udp_layer = inner_pkt[UDP]
            info.update(proto="udp", sport=int(udp_layer.sport), dport=int(udp_layer.dport))
        elif ICMP in inner_pkt:
            info.update(proto="icmp")
        else:
            info.update(proto="other")
        return info
