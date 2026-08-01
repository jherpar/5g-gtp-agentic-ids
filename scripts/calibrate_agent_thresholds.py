"""Phase 5B: calibrate agents/rules.py thresholds against real distributions.

The agent validation report (outputs/reports/agent_validation/report.md)
found high false-positive rates on HIGH-confidence (trustworthy) benign
traffic for the DoS-type attacks -- e.g. 44-73% FPR -- driven mainly by
PDUSessionAgent's `low_temporal_entropy` and `high_state_transition` rules,
which were thresholded during Phase 5 implementation against synthetic test
cases, never against real traffic. This script computes real distribution
statistics (p50/p75/p90/p95/p99) for every rule's underlying metric, split
by three groups:

  - HIGH-confidence benign  (trustworthy negative)
  - HIGH-confidence attack  (trustworthy positive)
  - MEDIUM-confidence attack (trustworthy positive, weaker corroboration)

pooled across all 9 attack types, since the agents are attack-type-blind in
production (they only see TEID/session features, never `source_attack_type`).
Does NOT add new rules or change rule logic -- only reports the data needed
to pick better threshold VALUES for the existing rules in
`configs/thresholds.yaml`.

Usage:
    poetry run python scripts/calibrate_agent_thresholds.py
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

from agente_5g.models.schedule_config import AttackSchedule, LabelPatternsConfig  # noqa: E402
from scripts.validate_labeling import process_file  # noqa: E402
from scripts.validate_labeling_all import _session_window_s  # noqa: E402

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports" / "agent_threshold_calibration"
ATTACK_TYPES = [
    "ICMPflood",
    "UDPflood",
    "SYNflood",
    "Goldeneye",
    "Slowloris",
    "Torshammer",
    "SYNScan",
    "TCPConnect",
    "UDPScan",
]
MAX_DURATION_S = 2200.0
GROUPS = ["high_benign", "high_attack", "medium_attack"]

TEID_METRICS = [
    "packets_per_s",
    "teid_entropy",
    "unique_dst_ports",
    "syn_count",
    "ack_to_syn_ratio",
    "packets_per_port",
]
SESSION_METRICS = [
    "state_transition_rate",
    "temporal_entropy",
    "port_diversity",
    "destination_diversity",
    "peak_diversity",
]

RULE_METRIC = {
    "flood (packets_per_s)": ("packets_per_s", "min_packets_per_s", 0.8),
    "flood (teid_entropy)": ("teid_entropy", "max_teid_entropy", 1.5),
    "flood (unique_dst_ports)": ("unique_dst_ports", "max_unique_dst_ports", 3),
    "syn_flood (syn_count)": ("syn_count", "min_syn_count", 20),
    "syn_flood (ack_to_syn_ratio)": ("ack_to_syn_ratio", "max_ack_to_syn_ratio", 0.1),
    "scan (unique_dst_ports)": ("unique_dst_ports", "min_unique_dst_ports", 15),
    "scan (packets_per_port)": ("packets_per_port", "max_packets_per_dst_port", 3.0),
    "high_state_transition": ("state_transition_rate", "threshold", 0.5),
    "low_temporal_entropy": ("temporal_entropy", "threshold", 0.5),
    "high_diversity": ("peak_diversity", "threshold", 15),
}


def _group(is_attack: bool | None, confidence_value: str | None) -> str | None:
    if is_attack is False and confidence_value == "HIGH":
        return "high_benign"
    if is_attack is True and confidence_value == "HIGH":
        return "high_attack"
    if is_attack is True and confidence_value == "MEDIUM":
        return "medium_attack"
    return None


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"n": 0, "p50": None, "p75": None, "p90": None, "p95": None, "p99": None}
    sv = sorted(values)

    def pct(p: float) -> float:
        idx = min(len(sv) - 1, int(round(p * (len(sv) - 1))))
        return round(sv[idx], 4)

    return {
        "n": len(sv),
        "p50": round(statistics.median(sv), 4),
        "p75": pct(0.75),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "p99": pct(0.99),
    }


def collect(all_features: list[Any], all_sessions: list[Any]) -> dict[str, dict[str, list[float]]]:
    metrics: dict[str, dict[str, list[float]]] = {
        m: defaultdict(list) for m in TEID_METRICS + SESSION_METRICS
    }

    for feat in all_features:
        conf = feat.label_confidence.value if feat.label_confidence else None
        group = _group(feat.is_attack, conf)
        if group is None:
            continue
        metrics["packets_per_s"][group].append(feat.packets_per_s)
        metrics["teid_entropy"][group].append(feat.teid_entropy)
        metrics["unique_dst_ports"][group].append(float(feat.unique_dst_ports))
        metrics["syn_count"][group].append(float(feat.syn_count))
        ack_to_syn = feat.ack_count / feat.syn_count if feat.syn_count > 0 else 0.0
        metrics["ack_to_syn_ratio"][group].append(ack_to_syn)
        pkts_per_port = (
            feat.packet_count / feat.unique_dst_ports if feat.unique_dst_ports > 0 else 0.0
        )
        metrics["packets_per_port"][group].append(pkts_per_port)

    for session in all_sessions:
        conf = session.label_confidence.value if session.label_confidence else None
        group = _group(session.is_attack, conf)
        if group is None:
            continue
        metrics["state_transition_rate"][group].append(session.state_transition_rate)
        metrics["temporal_entropy"][group].append(session.temporal_entropy)
        metrics["port_diversity"][group].append(float(session.port_diversity))
        metrics["destination_diversity"][group].append(float(session.destination_diversity))
        metrics["peak_diversity"][group].append(
            float(max(session.port_diversity, session.destination_diversity))
        )

    return metrics


def render(metrics: dict[str, dict[str, list[float]]]) -> str:
    lines = ["# Phase 5B: Agent Rule Threshold Calibration (real BS1 data, all 9 attack types)\n"]
    lines.append(
        "Generated by `scripts/calibrate_agent_thresholds.py`. Distribution statistics "
        "(p50/p75/p90/p95/p99) for each rule's metric, pooled across all 9 attack types, split "
        "by HIGH-confidence benign / HIGH-confidence attack / MEDIUM-confidence attack. No new "
        "rules, no architecture changes -- informs threshold VALUES only.\n"
    )

    for rule_label, (metric, field, current) in RULE_METRIC.items():
        lines.append(f"\n## Rule: {rule_label}\n")
        lines.append(f"Current threshold (`{field}`): `{current}`\n")
        lines.append("| Group | n | p50 | p75 | p90 | p95 | p99 |")
        lines.append("|---|---|---|---|---|---|---|")
        for group in GROUPS:
            p = _percentiles(metrics[metric][group])
            lines.append(
                f"| {group} | {p['n']} | {p['p50']} | {p['p75']} | {p['p90']} | "
                f"{p['p95']} | {p['p99']} |"
            )

    return "\n".join(lines)


def main() -> None:
    schedule = AttackSchedule.load(PROJECT_ROOT / "configs" / "attack_schedule.yaml")
    patterns = LabelPatternsConfig.load(PROJECT_ROOT / "configs" / "label_patterns.yaml")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    all_features: list[Any] = []
    all_sessions: list[Any] = []
    for attack_type in ATTACK_TYPES:
        path = PROJECT_ROOT / "data" / "raw" / "BS1" / f"{attack_type}_BS1.pcapng"
        data = process_file(
            path,
            base_station="BS1",
            attack_type=attack_type,
            schedule=schedule,
            patterns=patterns,
            max_duration_s=MAX_DURATION_S,
            session_window_s=_session_window_s(attack_type),
        )
        all_features.extend(data["features"])
        all_sessions.extend(data["sessions"])
        print(
            f"[{attack_type}] accumulated: {len(all_features)} features, "
            f"{len(all_sessions)} sessions"
        )

    metrics = collect(all_features, all_sessions)
    report_path = REPORT_DIR / "report.md"
    report_path.write_text(render(metrics), encoding="utf-8")
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
