"""Dump complete packet-level evidence for the victim-IP-corroborated
SYNflood and Goldeneye instances (BS1), per explicit user request, before
designing a new Level-3 corroboration rule for these two attack types.

The calibration run (scripts/calibrate_flood_pattern.py) and evidence
quantification (scripts/quantify_evidence_sources.py) both point at SYNflood
and Goldeneye behaving differently from ICMPflood/UDPflood -- their few
victim-IP-corroborated instances show high entropy and very high
destination-port counts, the opposite of a concentrated volumetric flood.
This script does NOT propose or apply a threshold; it only reports raw
connection-level facts for the n=3 SYNflood / n=2 Goldeneye corroborated
instances: packet count, SYN/ACK counts and ratio, unique src/dst ports,
unique src IPs, connection tuples (4-tuple flows), flow duration, and bytes
transferred -- plus a same-size sample of NON-corroborated instances from
the same files for contrast.

Usage:
    poetry run python scripts/inspect_flood_evidence.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from agente_5g.models.packet import GTPPacketRecord  # noqa: E402
from agente_5g.models.schedule_config import AttackSchedule, LabelPatternsConfig  # noqa: E402
from scripts.validate_labeling import process_file  # noqa: E402

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports" / "flood_evidence_inspection"
TYPES = ["SYNflood", "Goldeneye"]
MAX_DURATION_S = 2200.0


def _instance_packets_by_teid(packets: list[GTPPacketRecord]) -> dict[int, list[GTPPacketRecord]]:
    by_teid: dict[int, list[GTPPacketRecord]] = defaultdict(list)
    for p in packets:
        if p.is_gtp and p.teid is not None:
            by_teid[p.teid].append(p)
    return by_teid


def _instance_stats(feat: Any, packets: list[GTPPacketRecord]) -> dict[str, Any]:
    n = len(packets)
    syn = sum(1 for p in packets if p.tcp_syn)
    ack = sum(1 for p in packets if p.tcp_ack)
    rst = sum(1 for p in packets if p.tcp_rst)
    fin = sum(1 for p in packets if p.tcp_fin)
    src_ports = {p.inner_src_port for p in packets if p.inner_src_port is not None}
    dst_ports = {p.inner_dst_port for p in packets if p.inner_dst_port is not None}
    src_ips = {p.inner_src_ip for p in packets if p.inner_src_ip is not None}
    dst_ips = {p.inner_dst_ip for p in packets if p.inner_dst_ip is not None}
    tuples = {(p.inner_src_ip, p.inner_dst_ip, p.inner_src_port, p.inner_dst_port) for p in packets}
    timestamps = [p.timestamp for p in packets]
    duration = max(timestamps) - min(timestamps) if timestamps else 0.0
    total_bytes = sum(p.packet_size for p in packets)
    return {
        "teid": feat.teid,
        "label_evidence": feat.label_evidence,
        "window_start": feat.window_start,
        "window_end": feat.window_end,
        "n_packets": n,
        "syn_count": syn,
        "ack_count": ack,
        "rst_count": rst,
        "fin_count": fin,
        "syn_ack_ratio": round(syn / ack, 3) if ack else None,
        "unique_src_ports": len(src_ports),
        "unique_dst_ports": len(dst_ports),
        "unique_src_ips": len(src_ips),
        "unique_dst_ips": len(dst_ips),
        "unique_conn_tuples": len(tuples),
        "duration_s": round(duration, 3),
        "total_bytes": total_bytes,
        "bytes_per_s": round(total_bytes / duration, 2) if duration > 0 else None,
        "sample_dst_ports": sorted(dst_ports)[:15],
        "sample_src_ports": sorted(src_ports)[:15],
        "sample_dst_ips": sorted(dst_ips)[:5],
    }


def render(attack_type: str, corroborated: list[dict], not_corroborated: list[dict]) -> str:
    lines = [f"\n## {attack_type}\n"]
    lines.append(f"Victim-IP corroborated instances: {len(corroborated)}\n")
    for i, s in enumerate(corroborated, start=1):
        lines.append(f"### Corroborated instance {i} (TEID={s['teid']})\n")
        lines.append(f"- evidence: {s['label_evidence']}")
        lines.append(
            f"- window: [{s['window_start']:.1f}, {s['window_end']:.1f}] "
            f"(duration {s['duration_s']}s)"
        )
        lines.append(f"- packets: {s['n_packets']}")
        lines.append(
            f"- SYN={s['syn_count']}, ACK={s['ack_count']}, RST={s['rst_count']}, "
            f"FIN={s['fin_count']}, SYN/ACK ratio={s['syn_ack_ratio']}"
        )
        lines.append(
            f"- unique src ports={s['unique_src_ports']}, "
            f"unique dst ports={s['unique_dst_ports']}, "
            f"unique src IPs={s['unique_src_ips']}, unique dst IPs={s['unique_dst_ips']}"
        )
        lines.append(f"- unique connection tuples (4-tuple flows): {s['unique_conn_tuples']}")
        lines.append(f"- total bytes: {s['total_bytes']}, bytes/s: {s['bytes_per_s']}")
        lines.append(f"- sample dst ports: {s['sample_dst_ports']}")
        lines.append(f"- sample src ports: {s['sample_src_ports']}")
        lines.append(f"- sample dst IPs: {s['sample_dst_ips']}\n")

    lines.append(
        f"### Non-corroborated instances, for contrast "
        f"(n={len(not_corroborated)}, showing up to 5)\n"
    )
    lines.append(
        "| TEID | packets | SYN | ACK | SYN/ACK | uniq src_port | uniq dst_port | "
        "uniq src_ip | conn tuples | duration_s | bytes/s |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for s in not_corroborated[:5]:
        lines.append(
            f"| {s['teid']} | {s['n_packets']} | {s['syn_count']} | {s['ack_count']} | "
            f"{s['syn_ack_ratio']} | {s['unique_src_ports']} | {s['unique_dst_ports']} | "
            f"{s['unique_src_ips']} | {s['unique_conn_tuples']} | {s['duration_s']} | "
            f"{s['bytes_per_s']} |"
        )
    return "\n".join(lines)


def main() -> None:
    schedule = AttackSchedule.load(PROJECT_ROOT / "configs" / "attack_schedule.yaml")
    patterns = LabelPatternsConfig.load(PROJECT_ROOT / "configs" / "label_patterns.yaml")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    sections = [
        "# Packet-Level Evidence: Victim-IP Corroborated SYNflood / Goldeneye Instances (BS1)\n"
    ]
    sections.append(
        "Raw connection-level facts only -- no threshold proposed here. Generated by "
        "`scripts/inspect_flood_evidence.py`.\n"
    )

    for attack_type in TYPES:
        path = PROJECT_ROOT / "data" / "raw" / "BS1" / f"{attack_type}_BS1.pcapng"
        data = process_file(
            path,
            base_station="BS1",
            attack_type=attack_type,
            schedule=schedule,
            patterns=patterns,
            max_duration_s=MAX_DURATION_S,
        )
        packets_by_teid = _instance_packets_by_teid(data["packets"])
        features = data["features"]

        corroborated_stats = []
        not_corroborated_stats = []
        for feat in features:
            instance_packets = [
                p
                for p in packets_by_teid[feat.teid]
                if feat.window_start <= p.timestamp <= feat.window_end
            ]
            if not instance_packets:
                continue
            stats = _instance_stats(feat, instance_packets)
            if "VICTIM_IP" in feat.label_evidence:
                corroborated_stats.append(stats)
            elif feat.is_attack:
                not_corroborated_stats.append(stats)

        section = render(attack_type, corroborated_stats, not_corroborated_stats)
        sections.append(section)
        print(
            f"[{attack_type}] corroborated={len(corroborated_stats)} "
            f"not_corroborated_attack={len(not_corroborated_stats)}"
        )

    report_path = REPORT_DIR / "report.md"
    report_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
