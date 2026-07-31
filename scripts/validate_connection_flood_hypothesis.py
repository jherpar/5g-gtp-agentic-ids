"""Validate the "connection-oriented flood" hypothesis for SYNflood/Goldeneye
across BOTH base stations, before defining any new Level-3 corroboration rule.

Prior steps (scripts/inspect_flood_evidence.py, BS1 only) found that the few
victim-IP-corroborated SYNflood/Goldeneye instances share a shape entropy/
port-concentration checks miss entirely: one concentrated destination IP,
extreme asymmetry between unique src-port and dst-port counts, and heavy
connection churn (SYN/RST/FIN volume) -- structurally different from
ICMPflood/UDPflood's uniform-payload, few-ports-both-sides signature. One of
the 3 BS1 SYNflood corroborated instances looked like incidental background
traffic (many destination IPs, no concentration) rather than genuine attack
backscatter.

This script does NOT propose thresholds. It computes three candidate metrics
for EVERY attack-labeled SYNflood/Goldeneye instance (BS1 + BS2), split by
victim-IP corroboration, and reports full-population percentile
distributions so the corroborated-vs-not separation (or lack of it) can be
inspected honestly before any rule is written:

  1. dst_ip_concentration  -- unique_dst_ips (lower = more concentrated)
  2. port_cardinality_asymmetry -- max(uniq_src_ports, uniq_dst_ports) /
     max(1, min(uniq_src_ports, uniq_dst_ports))  (higher = more asymmetric)
  3. connection_churn -- unique 4-tuple flows per second of instance duration

Usage:
    poetry run python scripts/validate_connection_flood_hypothesis.py
"""

from __future__ import annotations

import statistics
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

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports" / "connection_flood_hypothesis"
TYPES = ["SYNflood", "Goldeneye"]
BASE_STATIONS = ["BS1", "BS2"]
MAX_DURATION_S = 2200.0


def _instance_packets_by_teid(packets: list[GTPPacketRecord]) -> dict[int, list[GTPPacketRecord]]:
    by_teid: dict[int, list[GTPPacketRecord]] = defaultdict(list)
    for p in packets:
        if p.is_gtp and p.teid is not None:
            by_teid[p.teid].append(p)
    return by_teid


def _instance_stats(feat: Any, base_station: str, packets: list[GTPPacketRecord]) -> dict[str, Any]:
    n = len(packets)
    syn = sum(1 for p in packets if p.tcp_syn)
    ack = sum(1 for p in packets if p.tcp_ack)
    rst = sum(1 for p in packets if p.tcp_rst)
    fin = sum(1 for p in packets if p.tcp_fin)
    src_ports = {p.inner_src_port for p in packets if p.inner_src_port is not None}
    dst_ports = {p.inner_dst_port for p in packets if p.inner_dst_port is not None}
    dst_ips = {p.inner_dst_ip for p in packets if p.inner_dst_ip is not None}
    tuples = {(p.inner_src_ip, p.inner_dst_ip, p.inner_src_port, p.inner_dst_port) for p in packets}
    timestamps = [p.timestamp for p in packets]
    duration = max(timestamps) - min(timestamps) if timestamps else 0.0

    n_src_ports = len(src_ports)
    n_dst_ports = len(dst_ports)
    port_asymmetry = max(n_src_ports, n_dst_ports) / max(1, min(n_src_ports, n_dst_ports))
    churn = len(tuples) / duration if duration > 0 else float(len(tuples))

    return {
        "base_station": base_station,
        "teid": feat.teid,
        "corroborated": "VICTIM_IP" in feat.label_evidence,
        "n_packets": n,
        "duration_s": round(duration, 3),
        "syn": syn,
        "ack": ack,
        "rst": rst,
        "fin": fin,
        "unique_src_ports": n_src_ports,
        "unique_dst_ports": n_dst_ports,
        "unique_dst_ips": len(dst_ips),
        "port_cardinality_asymmetry": round(port_asymmetry, 2),
        "connection_churn_per_s": round(churn, 3),
    }


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "n": 0,
            "min": None,
            "p10": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "max": None,
        }
    sv = sorted(values)

    def pct(p: float) -> float:
        idx = min(len(sv) - 1, int(round(p * (len(sv) - 1))))
        return round(sv[idx], 3)

    return {
        "n": len(sv),
        "min": round(sv[0], 3),
        "p10": pct(0.10),
        "p25": pct(0.25),
        "median": round(statistics.median(sv), 3),
        "p75": pct(0.75),
        "p90": pct(0.90),
        "max": round(sv[-1], 3),
    }


def render(attack_type: str, corroborated: list[dict], not_corroborated: list[dict]) -> str:
    lines = [f"\n## {attack_type} (BS1 + BS2 combined)\n"]
    lines.append(
        f"Corroborated: n={len(corroborated)}. "
        f"Non-corroborated attack-labeled: n={len(not_corroborated)}.\n"
    )

    lines.append("### Per-instance detail: victim-IP corroborated\n")
    lines.append(
        "| BS | TEID | packets | dur_s | SYN | ACK | RST | FIN | uniq_src_p | uniq_dst_p | "
        "uniq_dst_ip | port_asym | churn/s |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in corroborated:
        lines.append(
            f"| {s['base_station']} | {s['teid']} | {s['n_packets']} | {s['duration_s']} | "
            f"{s['syn']} | {s['ack']} | {s['rst']} | {s['fin']} | {s['unique_src_ports']} | "
            f"{s['unique_dst_ports']} | {s['unique_dst_ips']} | "
            f"{s['port_cardinality_asymmetry']} | {s['connection_churn_per_s']} |"
        )

    metric_labels = [
        ("unique_dst_ips", "1. Destination-IP concentration (lower=more concentrated)"),
        ("port_cardinality_asymmetry", "2. Port-cardinality asymmetry (max/min of src vs dst)"),
        ("connection_churn_per_s", "3. Connection churn (unique 4-tuple flows / s)"),
    ]
    for metric, label in metric_labels:
        lines.append(f"\n### {label}\n")
        corro_vals = [s[metric] for s in corroborated]
        not_corro_vals = [s[metric] for s in not_corroborated]
        cd = _percentiles(corro_vals)
        nd = _percentiles(not_corro_vals)
        lines.append("| Group | n | min | p10 | p25 | median | p75 | p90 | max |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        lines.append(
            f"| Corroborated | {cd['n']} | {cd['min']} | {cd['p10']} | {cd['p25']} | "
            f"{cd['median']} | {cd['p75']} | {cd['p90']} | {cd['max']} |"
        )
        lines.append(
            f"| Not corroborated | {nd['n']} | {nd['min']} | {nd['p10']} | {nd['p25']} | "
            f"{nd['median']} | {nd['p75']} | {nd['p90']} | {nd['max']} |"
        )

    return "\n".join(lines)


def main() -> None:
    schedule = AttackSchedule.load(PROJECT_ROOT / "configs" / "attack_schedule.yaml")
    patterns = LabelPatternsConfig.load(PROJECT_ROOT / "configs" / "label_patterns.yaml")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    sections = ["# Connection-Oriented Flood Hypothesis Validation (SYNflood/Goldeneye, BS1+BS2)\n"]
    sections.append(
        "No thresholds proposed here -- full-population percentile distributions for the 3 "
        "candidate metrics (destination-IP concentration, port-cardinality asymmetry, connection "
        "churn), corroborated vs. non-corroborated, across both base stations. Generated by "
        "`scripts/validate_connection_flood_hypothesis.py`.\n"
    )

    for attack_type in TYPES:
        corroborated: list[dict] = []
        not_corroborated: list[dict] = []
        for bs in BASE_STATIONS:
            path = PROJECT_ROOT / "data" / "raw" / bs / f"{attack_type}_{bs}.pcapng"
            print(f"[{attack_type}_{bs}] parsing/loading ...")
            data = process_file(
                path,
                base_station=bs,  # type: ignore[arg-type]
                attack_type=attack_type,
                schedule=schedule,
                patterns=patterns,
                max_duration_s=MAX_DURATION_S,
            )
            packets_by_teid = _instance_packets_by_teid(data["packets"])
            features = data["features"]

            for feat in features:
                if not feat.is_attack:
                    continue
                instance_packets = [
                    p
                    for p in packets_by_teid[feat.teid]
                    if feat.window_start <= p.timestamp <= feat.window_end
                ]
                if not instance_packets:
                    continue
                stats = _instance_stats(feat, bs, instance_packets)
                if stats["corroborated"]:
                    corroborated.append(stats)
                else:
                    not_corroborated.append(stats)
            print(
                f"[{attack_type}_{bs}] done: corroborated so far={len(corroborated)}, "
                f"non_corroborated so far={len(not_corroborated)}"
            )

        section = render(attack_type, corroborated, not_corroborated)
        sections.append(section)

    report_path = REPORT_DIR / "report.md"
    report_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
