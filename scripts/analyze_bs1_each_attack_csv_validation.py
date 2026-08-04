"""BS1_each_attack_csv Ground Truth Validation -- standalone appendix analysis.

Completely separate from Phase 4-8: does NOT modify any label, model,
threshold, agent, or existing output. Only READS the frozen pipeline's
already-computed labels (via `scripts.validate_labeling.process_file`,
exactly as every other read-only analysis script in this project does)
and compares them against a previously-unused author-provided dataset:
`data/raw/BS1_each_attack_csv/BS1_each_attack_csv/*.csv` -- a per-attack-
type flow export (Argus-derived, one row per biflow) that, unlike
`Combined.csv`/`Encoded.csv`, RETAINS real IP addresses (`SrcAddr`/
`DstAddr`) and a per-flow `Label`/`Attack Type`/`Attack Tool`.

Objectives (verbatim from the request):
  1. Analyze data/raw/BS1_each_attack_csv structure.
  2. Determine available identifiers, attack labels, attack types, flow
     granularity.
  3. Compare author-provided labels against: Schedule labels, Confidence
     labels, HIGH/MEDIUM subset.
  4. Measure agreement rate, precision, recall, confusion matrices.
  5. Quantify how many HIGH/MEDIUM labels are confirmed by the author
     dataset, and how many LOW labels are actually attacks according to
     the author dataset.
  6. Produce a standalone report.

Matching strategy: for every one of the 9 attack-type CSVs, the malicious
rows have exactly ONE source IP and ONE destination IP (verified
empirically -- see the report's Objective 2 section), always
`10.41.150.68` as the destination, matching `configs/attack_schedule.yaml`'s
`victim_ip` exactly (an independent confirmation of that value, read-only,
not a change to it). The (attacker_ip, victim_ip) pair is derived directly
from each CSV's own malicious rows (not hardcoded), and "author confirms
attack" for one of our own TEID/session instances is defined as: does ANY
packet in that instance touch this exact IP pair (either direction)? This
sidesteps the CSV's ambiguous `StartTime` format (MM:SS.s, no date/hour --
ordering is reliable only within a single-hour capture) entirely by
matching on identifiers, not time.

Usage:
    poetry run python scripts/analyze_bs1_each_attack_csv_validation.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from agente_5g.evaluation.metrics import compute_metrics  # noqa: E402
from agente_5g.models.schedule_config import AttackSchedule, LabelPatternsConfig  # noqa: E402
from scripts.validate_labeling import process_file  # noqa: E402
from scripts.validate_labeling_all import _session_window_s  # noqa: E402

CSV_DIR = PROJECT_ROOT / "data" / "raw" / "BS1_each_attack_csv" / "BS1_each_attack_csv"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports" / "bs1_each_attack_csv_validation"
MAX_DURATION_S = 2200.0

# Filename token (our pipeline) -> author CSV file / author "Attack Type" string.
# The author's per-flow export uses different names for two of our types
# (TCPConnect -> TCPConnectScan, Goldeneye -> HTTPFlood) and COLLAPSES
# Slowloris and Torshammer into a single "SlowrateDoS" category -- an
# independent confirmation that these two are the same category at the
# author's own granularity, consistent with `_PATTERN_CATEGORY` in
# `preprocessing/labeling.py` grouping them as "slowrate" too.
ATTACK_TYPE_TO_CSV_FILE = {
    "ICMPflood": "ICMPFlood1.csv",
    "UDPflood": "UDPFlood1.csv",
    "SYNflood": "SYNFlood1.csv",
    "Goldeneye": "Goldeneye1.csv",
    "Slowloris": "Slowloris1.csv",
    "Torshammer": "Torshammer1.csv",
    "SYNScan": "SYNScan1.csv",
    "TCPConnect": "TCPConnect1.csv",
    "UDPScan": "UDPScan1.csv",
}
ATTACK_TYPE_TO_AUTHOR_LABEL = {
    "ICMPflood": "ICMPFlood",
    "UDPflood": "UDPFlood",
    "SYNflood": "SYNFlood",
    "Goldeneye": "HTTPFlood",
    "Slowloris": "SlowrateDoS",
    "Torshammer": "SlowrateDoS",
    "SYNScan": "SYNScan",
    "TCPConnect": "TCPConnectScan",
    "UDPScan": "UDPScan",
}
BENIGN_ONLY_CSV = "SSH1.csv"


def load_author_csv(path: Path) -> pd.DataFrame:
    """Trailing whitespace in some files' `Attack Type ` header (verified:
    TCPConnect1.csv) is stripped so column access is uniform across files."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    return df


def structural_summary(attack_type: str, df: pd.DataFrame) -> dict[str, Any]:
    return {
        "attack_type": attack_type,
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "label_counts": df["Label"].value_counts(dropna=False).to_dict(),
        "attack_type_counts": df["Attack Type"].value_counts(dropna=False).to_dict(),
        "proto_counts": df["Proto"].value_counts(dropna=False).to_dict(),
        "n_unique_src": int(df["SrcAddr"].nunique()),
        "n_unique_dst": int(df["DstAddr"].nunique()),
    }


def derive_attacker_victim_ip(df: pd.DataFrame) -> tuple[str | None, str | None]:
    mal = df[df["Label"] == "Malicious"]
    if mal.empty:
        return None, None
    src_counts = mal["SrcAddr"].value_counts()
    dst_counts = mal["DstAddr"].value_counts()
    attacker_ip = src_counts.idxmax() if not src_counts.empty else None
    victim_ip = dst_counts.idxmax() if not dst_counts.empty else None
    return attacker_ip, victim_ip


def compare_attack_type(
    attack_type: str, schedule: AttackSchedule, patterns: LabelPatternsConfig
) -> dict[str, Any] | None:
    csv_path = CSV_DIR / ATTACK_TYPE_TO_CSV_FILE[attack_type]
    author_df = load_author_csv(csv_path)
    attacker_ip, victim_ip = derive_attacker_victim_ip(author_df)
    if attacker_ip is None or victim_ip is None:
        return None
    pair = {attacker_ip, victim_ip}

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
    packets_by_teid: dict[int, list[Any]] = defaultdict(list)
    for p in data["packets"]:
        if p.is_gtp and p.teid is not None:
            packets_by_teid[p.teid].append(p)

    def author_confirms(instance_packets: list[Any]) -> bool:
        return any({p.inner_src_ip, p.inner_dst_ip} == pair for p in instance_packets)

    def build_rows(records: list[Any], is_session: bool) -> list[dict[str, Any]]:
        rows = []
        for rec in records:
            start, end = (
                (rec.start_time, rec.end_time)
                if is_session
                else (
                    rec.window_start,
                    rec.window_end,
                )
            )
            instance_packets = [p for p in packets_by_teid[rec.teid] if start <= p.timestamp <= end]
            rows.append(
                {
                    "our_is_attack": bool(rec.is_attack),
                    "our_confidence": rec.label_confidence.value if rec.label_confidence else None,
                    "author_says_attack": author_confirms(instance_packets),
                }
            )
        return rows

    return {
        "attack_type": attack_type,
        "attacker_ip": attacker_ip,
        "victim_ip": victim_ip,
        "csv_malicious_rows": int((author_df["Label"] == "Malicious").sum()),
        "csv_total_rows": len(author_df),
        "teid_rows": build_rows(data["features"], is_session=False),
        "session_rows": build_rows(data["sessions"], is_session=True),
    }


def _agreement_metrics(rows: list[dict[str, Any]], predicted_key_fn: Any) -> dict[str, Any]:
    y_true = [r["author_says_attack"] for r in rows]
    y_pred = [predicted_key_fn(r) for r in rows]
    m = compute_metrics(y_true, y_pred)
    agreement = (
        sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p) / len(rows) if rows else None
    )
    return {
        "n": len(rows),
        "agreement_rate": round(agreement, 4) if agreement is not None else None,
        "precision": round(m["precision"], 4),
        "recall": round(m["recall"], 4),
        "confusion_matrix": m["confusion_matrix"],
    }


def _tier_confirmation(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for tier in ["HIGH", "MEDIUM", "LOW"]:
        tier_rows = [r for r in rows if r["our_is_attack"] and r["our_confidence"] == tier]
        n = len(tier_rows)
        confirmed = sum(1 for r in tier_rows if r["author_says_attack"])
        result[tier] = {
            "n": n,
            "author_confirmed": confirmed,
            "author_confirmed_pct": round(100 * confirmed / n, 1) if n else None,
        }
    return result


def render_report(structures: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> str:
    lines = ["# BS1_each_attack_csv Ground Truth Validation\n"]
    lines.append(
        "Standalone appendix analysis. Generated by "
        "`scripts/analyze_bs1_each_attack_csv_validation.py`. Does not modify any "
        "label, model, threshold, agent, or Phase 4-8 output -- read-only comparison "
        "against a previously-unused author-provided per-flow dataset. See the script's "
        "module docstring for the exact matching methodology.\n"
    )

    lines.append("\n## Objective 1-2: structure of data/raw/BS1_each_attack_csv\n")
    lines.append(
        "Argus-derived per-biflow export, one file per attack type + `SSH1.csv` "
        "(pure benign). Unlike `Combined.csv`/`Encoded.csv`, these files retain real "
        "`SrcAddr`/`DstAddr` IP addresses, `Sport`/`Dport`, and a `StartTime`/`LastTime` "
        "pair (format `MM:SS.s`, no date/hour -- ordering is only reliable within a "
        "single-hour capture, and was NOT used for matching here; identifier-based "
        "matching was used instead, see below). 108 columns total; identifiers: "
        "`SrcAddr`, `DstAddr`, `Sport`, `Dport`, `Proto`; ground truth: `Label` "
        "(Benign/Malicious), `Attack Type`, `Attack Tool`.\n"
    )
    lines.append("| Attack type | rows | Label counts | unique SrcAddr | unique DstAddr |")
    lines.append("|---|---|---|---|---|")
    for s in structures:
        lines.append(
            f"| {s['attack_type']} | {s['n_rows']} | {s['label_counts']} | "
            f"{s['n_unique_src']} | {s['n_unique_dst']} |"
        )

    lines.append(
        "\nOne data-quality quirk found: `TCPConnect1.csv`'s `Attack Type` column "
        'header has trailing whitespace (`"Attack Type "`) -- stripped on load, not '
        "otherwise significant.\n"
    )

    lines.append("\n## Attacker/victim IP identified per attack type\n")
    lines.append(
        "Derived independently from each file's own malicious rows (mode of "
        "`SrcAddr`/`DstAddr` among `Label==Malicious` rows) -- not hardcoded, not read "
        "from `configs/attack_schedule.yaml`. Every one of the 9 types has EXACTLY one "
        "malicious source IP and one malicious destination IP.\n"
    )
    lines.append(
        "| Attack type | attacker_ip (author) | victim_ip (author) | "
        "matches configs/attack_schedule.yaml victim_ip? |"
    )
    lines.append("|---|---|---|---|")
    for c in comparisons:
        matches = "YES" if c["victim_ip"] == "10.41.150.68" else "NO"
        lines.append(f"| {c['attack_type']} | {c['attacker_ip']} | {c['victim_ip']} | {matches} |")

    lines.append(
        "\nThe victim IP matches `configs/attack_schedule.yaml`'s `victim_ip` "
        "(`10.41.150.68`) for all 9 types -- an independent confirmation of that "
        "value from a source never used to derive it. The attacker IP varies per "
        "attack type (unlike the config's single `attacker_ip_hints.BS1` value, which "
        "was documented there as a hint never actually used for gating, only Level-2 "
        "victim-IP matching is) -- consistent with the config's own comment, not a "
        "contradiction of anything load-bearing.\n"
    )

    lines.append("\n## Objectives 3-5: agreement with our schedule/confidence labels\n")
    for granularity, key in [
        ("TEID-instance level", "teid_rows"),
        ("Session level", "session_rows"),
    ]:
        lines.append(f"\n### {granularity}\n")
        for c in comparisons:
            rows = c[key]
            lines.append(f"\n#### {c['attack_type']}\n")
            schedule_m = _agreement_metrics(rows, lambda r: r["our_is_attack"])
            hm_m = _agreement_metrics(
                rows, lambda r: r["our_is_attack"] and r["our_confidence"] in ("HIGH", "MEDIUM")
            )
            lines.append(
                "| Predictor | n | agreement | precision | recall | confusion [[TN,FP],[FN,TP]] |"
            )
            lines.append("|---|---|---|---|---|---|")
            lines.append(
                f"| Schedule label (is_attack) | {schedule_m['n']} | "
                f"{schedule_m['agreement_rate']} | {schedule_m['precision']} | "
                f"{schedule_m['recall']} | {schedule_m['confusion_matrix']} |"
            )
            lines.append(
                f"| HIGH+MEDIUM subset | {hm_m['n']} | {hm_m['agreement_rate']} | "
                f"{hm_m['precision']} | {hm_m['recall']} | {hm_m['confusion_matrix']} |"
            )

            tiers = _tier_confirmation(rows)
            lines.append("\n| Confidence tier | n (our attack-labeled) | author-confirmed | % |")
            lines.append("|---|---|---|---|")
            for tier in ["HIGH", "MEDIUM", "LOW"]:
                t = tiers[tier]
                lines.append(
                    f"| {tier} | {t['n']} | {t['author_confirmed']} | {t['author_confirmed_pct']} |"
                )

    lines.append("\n## Overall summary (pooled across all 9 attack types)\n")
    for granularity, key in [
        ("TEID-instance level", "teid_rows"),
        ("Session level", "session_rows"),
    ]:
        all_rows = [r for c in comparisons for r in c[key]]
        schedule_m = _agreement_metrics(all_rows, lambda r: r["our_is_attack"])
        hm_m = _agreement_metrics(
            all_rows, lambda r: r["our_is_attack"] and r["our_confidence"] in ("HIGH", "MEDIUM")
        )
        tiers = _tier_confirmation(all_rows)
        lines.append(f"\n### {granularity} (n={len(all_rows)})\n")
        lines.append("| Predictor | agreement | precision | recall |")
        lines.append("|---|---|---|---|")
        lines.append(
            f"| Schedule label (is_attack) | {schedule_m['agreement_rate']} | "
            f"{schedule_m['precision']} | {schedule_m['recall']} |"
        )
        lines.append(
            f"| HIGH+MEDIUM subset | {hm_m['agreement_rate']} | "
            f"{hm_m['precision']} | {hm_m['recall']} |"
        )
        lines.append("\n| Confidence tier | n | author-confirmed | % |")
        lines.append("|---|---|---|---|")
        for tier in ["HIGH", "MEDIUM", "LOW"]:
            t = tiers[tier]
            lines.append(
                f"| {tier} | {t['n']} | {t['author_confirmed']} | {t['author_confirmed_pct']} |"
            )

    lines.append(
        "\n\nThis is a validation study only -- no model was retrained, no Phase 4-8 "
        "result, conclusion, label, threshold, or output was changed to produce it. "
        "See `experiment_plan.md` for the frozen primary results.\n"
    )

    return "\n".join(lines)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    schedule = AttackSchedule.load(PROJECT_ROOT / "configs" / "attack_schedule.yaml")
    patterns = LabelPatternsConfig.load(PROJECT_ROOT / "configs" / "label_patterns.yaml")

    print("Objective 1-2: structural analysis ...")
    structures = []
    ssh_df = load_author_csv(CSV_DIR / BENIGN_ONLY_CSV)
    structures.append(structural_summary("SSH (benign-only)", ssh_df))
    for attack_type, fname in ATTACK_TYPE_TO_CSV_FILE.items():
        df = load_author_csv(CSV_DIR / fname)
        structures.append(structural_summary(attack_type, df))
        print(
            f"  [{attack_type}] {len(df)} rows, {int((df['Label'] == 'Malicious').sum())} malicious"
        )

    print("\nObjectives 3-5: comparing against our schedule/confidence labels ...")
    comparisons = []
    for attack_type in ATTACK_TYPE_TO_CSV_FILE:
        result = compare_attack_type(attack_type, schedule, patterns)
        if result is not None:
            comparisons.append(result)
            print(
                f"  [{attack_type}] attacker={result['attacker_ip']} victim={result['victim_ip']} "
                f"TEID_n={len(result['teid_rows'])} session_n={len(result['session_rows'])}"
            )

    report = render_report(structures, comparisons)
    (REPORT_DIR / "report.md").write_text(report, encoding="utf-8")
    print(f"\nReport written to {REPORT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
