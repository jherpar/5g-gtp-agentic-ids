"""Optional PyShark backend.

Only usable when `tshark` is on PATH (see `factory.get_parser`, which is the
only place this class should be instantiated from). `pyshark` itself is not
a hard dependency of this package — the import is lazy so the project
installs and runs fully without it (see pyproject.toml's `[project.optional-dependencies]`).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from agente_5g.models.packet import GTPPacketRecord
from agente_5g.parsers.base import PacketParser
from agente_5g.utils.logging import get_logger

logger = get_logger(__name__)


class PySharkPacketParser(PacketParser):
    def __init__(self) -> None:
        try:
            import pyshark  # noqa: F401
        except ImportError as exc:  # pragma: no cover - exercised only when pyshark is absent
            raise RuntimeError(
                "pyshark is not installed; install the 'pyshark' extra or use ScapyPacketParser"
            ) from exc

    def parse_file(
        self,
        path: Path,
        base_station: Literal["BS1", "BS2"],
        source_attack_type: str,
        max_packets: int | None = None,
        max_duration_s: float | None = None,
    ) -> Iterator[GTPPacketRecord]:
        import pyshark

        packet_id = 0
        first_ts: float | None = None
        cap = pyshark.FileCapture(str(path), keep_packets=False)
        try:
            for pkt in cap:
                if max_packets is not None and packet_id >= max_packets:
                    break

                ts = float(pkt.sniff_timestamp)
                if first_ts is None:
                    first_ts = ts
                if max_duration_s is not None and (ts - first_ts) > max_duration_s:
                    break

                try:
                    record = self._parse_packet(
                        pkt, packet_id, path.name, base_station, source_attack_type, ts
                    )
                except Exception:
                    logger.debug("Skipping unparseable packet #%d in %s", packet_id, path.name)
                    record = None

                packet_id += 1
                if record is not None:
                    yield record
        finally:
            cap.close()

    def _parse_packet(
        self, pkt, packet_id, capture_file, base_station, source_attack_type, ts
    ) -> GTPPacketRecord | None:
        if not hasattr(pkt, "ip"):
            return None

        is_gtp = hasattr(pkt, "gtp")
        teid = int(pkt.gtp.teid, 16) if is_gtp and hasattr(pkt.gtp, "teid") else None
        gtp_message_type = (
            int(pkt.gtp.message_type) if is_gtp and hasattr(pkt.gtp, "message_type") else None
        )

        inner_src_ip = inner_dst_ip = None
        inner_src_port = inner_dst_port = None
        inner_proto = None
        tcp_syn = tcp_ack = tcp_rst = tcp_fin = False

        # When GTP-U encapsulated, pyshark exposes the inner IP as a second
        # "ip" layer; otherwise pkt.ip *is* the inner/only IP layer.
        inner_ip_layer = pkt.ip
        if is_gtp:
            ip_layers = [layer for layer in pkt.layers if layer.layer_name == "ip"]
            if len(ip_layers) > 1:
                inner_ip_layer = ip_layers[-1]

        inner_src_ip = getattr(inner_ip_layer, "src", None)
        inner_dst_ip = getattr(inner_ip_layer, "dst", None)

        if hasattr(pkt, "tcp"):
            inner_proto = "tcp"
            inner_src_port = int(pkt.tcp.srcport)
            inner_dst_port = int(pkt.tcp.dstport)
            tcp_syn = pkt.tcp.flags_syn == "1"
            tcp_ack = pkt.tcp.flags_ack == "1"
            tcp_rst = pkt.tcp.flags_reset == "1"
            tcp_fin = pkt.tcp.flags_fin == "1"
        elif hasattr(pkt, "udp") and not is_gtp:
            inner_proto = "udp"
            inner_src_port = int(pkt.udp.srcport)
            inner_dst_port = int(pkt.udp.dstport)
        elif hasattr(pkt, "icmp"):
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
            outer_src_ip=pkt.ip.src,
            outer_dst_ip=pkt.ip.dst,
            inner_src_ip=inner_src_ip,
            inner_dst_ip=inner_dst_ip,
            inner_src_port=inner_src_port,
            inner_dst_port=inner_dst_port,
            inner_proto=inner_proto,
            packet_size=int(pkt.length),
            tcp_syn=tcp_syn,
            tcp_ack=tcp_ack,
            tcp_rst=tcp_rst,
            tcp_fin=tcp_fin,
        )
