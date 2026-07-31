"""Diagnose WHY HIGH+MEDIUM confidence is low across attack types.

The aggregate report (scripts/validate_labeling_all.py) found every one of
the 9 real attack types below a 30% HIGH+MEDIUM threshold (worst: UDPflood
at 4.4% TEID / 3.7% session). This script investigates *why*, per attack
type, rather than guessing at a threshold fix:

  1. Evidence breakdown: among attack-labeled (Level-1-fired) instances,
     what fraction also have Level 2 (victim IP) and/or Level 3 (pattern)
     evidence? Critically, `preprocessing/labeling.py::_classify` GATES on
     Level 2 -- an instance can only reach MEDIUM/HIGH if Level 2 fired at
     all, regardless of Level 3 -- so a Level-3-only match ("pattern fired,
     victim IP didn't") currently contributes NOTHING to confidence. This
     script surfaces how often that gated-out case happens, since it's
     invisible in the confidence tiers alone.
  2. Metric distributions: for the category-relevant raw signal (packets_per_s
     for flood, unique_dst_ports for scan, duration_s/bytes_per_s proxies for
     slowrate), compare victim-IP-corroborated instances (best available
     proxy for "genuinely real attack traffic") against non-corroborated
     ones, at BOTH TEID and session (5s) granularity -- this directly tests
     the "session window sizing" hypothesis found for scan types in the
     aggregate report.
  3. Candidate threshold: the value at which ~90% of victim-IP-corroborated
     instances would clear the rule (10th percentile of that group), shown
     against the current configured threshold.
  4. Root-cause classification per type: "attack schedule mismatch" (most
     attack-labeled instances never touch the victim IP at all -- Level 1's
     window is catching mostly unrelated traffic), "pattern threshold too
     strict" (victim-corroborated instances exist but don't clear Level 3),
     "session window sizing" (TEID-level evidence rate is much higher than
     session-level), or "victim-IP corroboration failure" (pattern evidence
     exists on instances that never get victim-IP corroboration, suggesting
     real attack traffic the victim-IP check is missing).

Writes outputs/reports/confidence_diagnosis/report.md +
outputs/reports/confidence_diagnosis/raw_stats.json (checkpointed
per-attack-type as it runs, since this re-parses the full pcap files again).

Usage:
    poetry run python scripts/diagnose_confidence_system.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from agente_5g.models.schedule_config import AttackSchedule, LabelPatternsConfig  # noqa: E402
from scripts.validate_labeling import process_file  # noqa: E402

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports" / "confidence_diagnosis"

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

CATEGORY = {
    "ICMPflood": "flood",
    "UDPflood": "flood",
    "SYNflood": "flood",
    "Goldeneye": "flood",
    "Slowloris": "slowrate",
    "Torshammer": "slowrate",
    "SYNScan": "scan",
    "TCPConnect": "scan",
    "UDPScan": "scan",
}

CURRENT_THRESHOLDS = {
    "flood": ("packets_per_s", 0.9),
    "scan": ("unique_dst_ports", 15),
    "slowrate": ("duration_s", 20.0),  # proxy: min_connection_duration_s
}


def _pct(part: int, total: int) -> float:
    return round(100 * part / total, 1) if total else 0.0


def _metric_value(feat: Any, category: str) -> float:
    if category == "flood":
        return feat.packets_per_s
    if category == "scan":
        return float(feat.unique_dst_ports)
    return feat.duration_s  # slowrate proxy


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p10": None, "p25": None, "median": None, "p75": None, "max": None}
    sorted_v = sorted(values)

    def pct(p: float) -> float:
        idx = min(len(sorted_v) - 1, int(round(p * (len(sorted_v) - 1))))
        return sorted_v[idx]

    return {
        "min": round(sorted_v[0], 3),
        "p10": round(pct(0.10), 3),
        "p25": round(pct(0.25), 3),
        "median": round(statistics.median(sorted_v), 3),
        "p75": round(pct(0.75), 3),
        "max": round(sorted_v[-1], 3),
    }


def evidence_breakdown(records: list[Any]) -> dict[str, Any]:
    attack = [r for r in records if r.is_attack]
    n = len(attack)
    has_victim = sum(1 for r in attack if "VICTIM_IP" in r.label_evidence)
    has_pattern = sum(1 for r in attack if "PATTERN" in r.label_evidence)
    has_both = sum(
        1 for r in attack if "VICTIM_IP" in r.label_evidence and "PATTERN" in r.label_evidence
    )
    has_neither = sum(
        1
        for r in attack
        if "VICTIM_IP" not in r.label_evidence and "PATTERN" not in r.label_evidence
    )
    pattern_without_victim = sum(
        1 for r in attack if "PATTERN" in r.label_evidence and "VICTIM_IP" not in r.label_evidence
    )
    return {
        "n_attack": n,
        "victim_ip_pct": _pct(has_victim, n),
        "pattern_pct": _pct(has_pattern, n),
        "both_pct": _pct(has_both, n),
        "neither_pct": _pct(has_neither, n),
        "pattern_without_victim_pct": _pct(pattern_without_victim, n),
    }


def metric_distribution(features: list[Any], category: str) -> dict[str, Any]:
    attack = [f for f in features if f.is_attack]
    corroborated = [_metric_value(f, category) for f in attack if "VICTIM_IP" in f.label_evidence]
    not_corroborated = [
        _metric_value(f, category) for f in attack if "VICTIM_IP" not in f.label_evidence
    ]
    metric_name, current_threshold = CURRENT_THRESHOLDS[category]
    candidate = _percentiles(corroborated)["p10"] if corroborated else None
    return {
        "metric_name": metric_name,
        "current_threshold": current_threshold,
        "corroborated_n": len(corroborated),
        "corroborated_dist": _percentiles(corroborated),
        "not_corroborated_n": len(not_corroborated),
        "not_corroborated_dist": _percentiles(not_corroborated),
        "candidate_threshold_p10_of_corroborated": candidate,
    }


def classify_root_cause(teid_ev: dict, sess_ev: dict, teid_dist: dict) -> str:
    reasons = []
    if teid_ev["victim_ip_pct"] < 20.0:
        reasons.append(
            "attack schedule mismatch (Level 1 window mostly catches non-victim traffic)"
        )
    if teid_ev["pattern_without_victim_pct"] >= 10.0:
        reasons.append("victim-IP corroboration failure (pattern matches traffic victim-IP misses)")
    if teid_ev["victim_ip_pct"] >= 20.0 and teid_ev["both_pct"] < teid_ev["victim_ip_pct"] * 0.5:
        reasons.append("pattern threshold too strict for victim-corroborated traffic")
    if teid_ev["victim_ip_pct"] > 0 and sess_ev["victim_ip_pct"] < teid_ev["victim_ip_pct"] * 0.5:
        reasons.append("session window sizing (evidence rate collapses at 5s granularity)")
    return "; ".join(reasons) if reasons else "no dominant single cause identified"


def render_report(rows: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# Confidence System Diagnosis (all 9 attack types, BS1)\n")
    lines.append(
        "Generated by `scripts/diagnose_confidence_system.py`. Investigates *why* "
        "HIGH+MEDIUM confidence is low (found universally < 30% in "
        "`outputs/reports/labeling_validation_all/report.md`) rather than guessing "
        "at a threshold fix. Victim-IP corroboration is used throughout as the best "
        'available proxy for "genuinely real attack traffic" among Level-1-flagged '
        "instances.\n"
    )

    for row in rows:
        t = row["attack_type"]
        teid_ev = row["teid_evidence"]
        sess_ev = row["session_evidence"]
        dist = row["metric_distribution"]

        lines.append(f"\n## {t} ({row['category']})\n")
        lines.append(
            f"Attack-labeled TEIDs: {teid_ev['n_attack']}, sessions: {sess_ev['n_attack']}\n"
        )

        lines.append("### 1. Which evidence fails most often\n")
        lines.append("| Level | TEID % | Session % |")
        lines.append("|---|---|---|")
        v_teid, v_sess = teid_ev["victim_ip_pct"], sess_ev["victim_ip_pct"]
        p_teid, p_sess = teid_ev["pattern_pct"], sess_ev["pattern_pct"]
        b_teid, b_sess = teid_ev["both_pct"], sess_ev["both_pct"]
        n_teid, n_sess = teid_ev["neither_pct"], sess_ev["neither_pct"]
        g_teid, g_sess = (
            teid_ev["pattern_without_victim_pct"],
            sess_ev["pattern_without_victim_pct"],
        )
        lines.append(f"| Touches victim IP (Level 2) | {v_teid}% | {v_sess}% |")
        lines.append(f"| Matches pattern (Level 3) | {p_teid}% | {p_sess}% |")
        lines.append(f"| Both (-> HIGH) | {b_teid}% | {b_sess}% |")
        lines.append(f"| Neither (schedule only, -> LOW) | {n_teid}% | {n_sess}% |")
        lines.append(
            f"| Pattern fired but victim IP didn't (gated out, still LOW) | {g_teid}% | {g_sess}% |"
        )

        lines.append("\n### 2. Distribution of pattern metrics (TEID level)\n")
        lines.append(
            f"Metric: `{dist['metric_name']}`, current threshold: `{dist['current_threshold']}`\n"
        )
        lines.append("| Group | n | min | p10 | p25 | median | p75 | max |")
        lines.append("|---|---|---|---|---|---|---|---|")
        cd = dist["corroborated_dist"]
        nd = dist["not_corroborated_dist"]
        lines.append(
            f"| Victim-IP corroborated | {dist['corroborated_n']} | {cd['min']} | {cd['p10']} "
            f"| {cd['p25']} | {cd['median']} | {cd['p75']} | {cd['max']} |"
        )
        lines.append(
            f"| Not corroborated | {dist['not_corroborated_n']} | {nd['min']} | {nd['p10']} "
            f"| {nd['p25']} | {nd['median']} | {nd['p75']} | {nd['max']} |"
        )

        lines.append("\n### 3. Candidate threshold recalibration\n")
        lines.append(
            f"- Current: `{dist['current_threshold']}`\n"
            f"- Candidate (p10 of victim-IP-corroborated group): "
            f"`{dist['candidate_threshold_p10_of_corroborated']}`"
        )

        lines.append("\n### 4. Root cause\n")
        lines.append(f"- {row['root_cause']}")

    lines.append("\n## Summary: root cause by attack type\n")
    lines.append("| Attack type | Category | Victim IP % (TEID) | Pattern % (TEID) | Root cause |")
    lines.append("|---|---|---|---|---|")
    for row in rows:
        teid_ev = row["teid_evidence"]
        lines.append(
            f"| {row['attack_type']} | {row['category']} | {teid_ev['victim_ip_pct']}% "
            f"| {teid_ev['pattern_pct']}% | {row['root_cause']} |"
        )

    return "\n".join(lines)


def main() -> None:
    schedule = AttackSchedule.load(PROJECT_ROOT / "configs" / "attack_schedule.yaml")
    patterns = LabelPatternsConfig.load(PROJECT_ROOT / "configs" / "label_patterns.yaml")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    raw_stats_path = REPORT_DIR / "raw_stats.json"

    rows: list[dict] = []
    for attack_type in ATTACK_TYPES:
        path = PROJECT_ROOT / "data" / "raw" / "BS1" / f"{attack_type}_BS1.pcapng"
        t0 = time.time()
        data = process_file(
            path,
            base_station="BS1",
            attack_type=attack_type,
            schedule=schedule,
            patterns=patterns,
            max_duration_s=MAX_DURATION_S,
        )
        category = CATEGORY[attack_type]
        teid_ev = evidence_breakdown(data["features"])
        sess_ev = evidence_breakdown(data["sessions"])
        dist = metric_distribution(data["features"], category)
        root_cause = classify_root_cause(teid_ev, sess_ev, dist)

        row = {
            "attack_type": attack_type,
            "category": category,
            "teid_evidence": teid_ev,
            "session_evidence": sess_ev,
            "metric_distribution": dist,
            "root_cause": root_cause,
        }
        rows.append(row)
        elapsed = time.time() - t0
        print(f"[{attack_type}] done in {elapsed:.1f}s: {root_cause}")

        raw_stats_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")

    report_path = REPORT_DIR / "report.md"
    report_path.write_text(render_report(rows), encoding="utf-8")
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
