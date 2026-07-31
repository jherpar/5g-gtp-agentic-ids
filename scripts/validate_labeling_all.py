"""Aggregate labeling validation report across all 9 real attack types (BS1).

Addresses a methodological gap in the single-file report
(scripts/validate_labeling.py): SYNScan_BS1 showed 97.9% of attack-labeled
sessions at LOW confidence (schedule-only evidence, since SYNScan's schedule
row has no separate benign-only collection period -- see the KNOWN
LIMITATION note in preprocessing/labeling.py). Before training any model on
these labels, we need to know whether that's a SYNScan-specific quirk or
affects other attack types too, and how many HIGH/MEDIUM-confidence samples
would actually remain per type if LOW-confidence attack labels are excluded
from the supervised training set (kept only for evaluation / uncertainty
analysis / qualitative discussion, per the user's explicit request).

Writes:
  - outputs/reports/labeling_validation_all/report.md   (per-type + summary table)
  - outputs/reports/labeling_validation_all/summary.csv (machine-readable,
    for the Phase 6 training split to consume directly)

Runs on the FULL real pcapng files for BS1 (capped at max_duration_s=2200s,
which exceeds the documented 30-minute DoS session length with margin and
doesn't truncate the shorter ~10-minute scan sessions at all) -- this is a
long-running batch job (the largest files are ~660MB), so results are
checkpointed to summary.csv after each attack type in case of interruption.

Usage:
    poetry run python scripts/validate_labeling_all.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from agente_5g.models.schedule_config import AttackSchedule, LabelPatternsConfig  # noqa: E402
from scripts.validate_labeling import process_file  # noqa: E402

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports" / "labeling_validation_all"

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

# Covers the documented 30-minute DoS session length with margin; the
# ~10-minute scan sessions finish well before this cap and are never
# truncated by it.
MAX_DURATION_S = 2200.0

# Priority 2 fix (per confidence_diagnosis/report.md): 5s session windows
# fragment a port scan's signature (needs 15+ distinct ports to corroborate,
# rarely reached within any single 5s slice even though the full ~500s+ scan
# clearly shows it) -- scan-type files use 30s windows instead, matching the
# TEID-level granularity that was already shown to corroborate cleanly.
SCAN_TYPES = {"SYNScan", "TCPConnect", "UDPScan"}


def _session_window_s(attack_type: str) -> int:
    return 30 if attack_type in SCAN_TYPES else 5


HIGH_MEDIUM_MIN_PCT = 30.0  # below this, flag the attack type as unreliable for training

CSV_FIELDS = [
    "attack_type",
    "session_window_s",
    "n_packets",
    "span_s",
    "n_teid",
    "n_sess",
    "n_teid_attack",
    "n_sess_attack",
    "teid_attack_pct",
    "teid_benign_pct",
    "sess_attack_pct",
    "sess_benign_pct",
    "teid_attack_high_pct",
    "teid_attack_medium_pct",
    "teid_attack_low_pct",
    "sess_attack_high_pct",
    "sess_attack_medium_pct",
    "sess_attack_low_pct",
    "teid_attack_high_medium_pct",
    "sess_attack_high_medium_pct",
    "teid_usable_for_training",
    "sess_usable_for_training",
    "teid_usable_pct",
    "sess_usable_pct",
]


def _pct(part: int, total: int) -> float:
    return 100 * part / total if total else 0.0


def _confidence_counts(records: list) -> dict[str, int]:
    counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for r in records:
        if r.label_confidence is not None:
            counts[r.label_confidence.value] += 1
    return counts


def compute_stats(data: dict) -> dict:
    features = data["features"]
    sessions = data["sessions"]
    n_teid, n_sess = len(features), len(sessions)

    teid_attack = [f for f in features if f.is_attack]
    teid_benign = [f for f in features if not f.is_attack]
    sess_attack = [s for s in sessions if s.is_attack]
    sess_benign = [s for s in sessions if not s.is_attack]

    teid_attack_conf = _confidence_counts(teid_attack)
    sess_attack_conf = _confidence_counts(sess_attack)
    teid_benign_conf = _confidence_counts(teid_benign)
    sess_benign_conf = _confidence_counts(sess_benign)

    # "Usable for training" = HIGH or MEDIUM confidence, on EITHER side
    # (attack or benign) -- LOW-confidence labels of either class are kept
    # only for evaluation/uncertainty analysis/qualitative discussion, per
    # the user's explicit request, not for the supervised training set.
    teid_usable = (
        teid_attack_conf["HIGH"]
        + teid_attack_conf["MEDIUM"]
        + teid_benign_conf["HIGH"]
        + teid_benign_conf["MEDIUM"]
    )
    sess_usable = (
        sess_attack_conf["HIGH"]
        + sess_attack_conf["MEDIUM"]
        + sess_benign_conf["HIGH"]
        + sess_benign_conf["MEDIUM"]
    )

    n_teid_attack, n_sess_attack = len(teid_attack), len(sess_attack)

    return {
        "attack_type": data["attack_type"],
        "session_window_s": _session_window_s(data["attack_type"]),
        "n_packets": data["n_packets"],
        "span_s": round(data["span_s"], 1),
        "n_teid": n_teid,
        "n_sess": n_sess,
        "n_teid_attack": n_teid_attack,
        "n_sess_attack": n_sess_attack,
        "teid_attack_pct": round(_pct(n_teid_attack, n_teid), 1),
        "teid_benign_pct": round(_pct(len(teid_benign), n_teid), 1),
        "sess_attack_pct": round(_pct(n_sess_attack, n_sess), 1),
        "sess_benign_pct": round(_pct(len(sess_benign), n_sess), 1),
        "teid_attack_high_pct": round(_pct(teid_attack_conf["HIGH"], n_teid_attack), 1),
        "teid_attack_medium_pct": round(_pct(teid_attack_conf["MEDIUM"], n_teid_attack), 1),
        "teid_attack_low_pct": round(_pct(teid_attack_conf["LOW"], n_teid_attack), 1),
        "sess_attack_high_pct": round(_pct(sess_attack_conf["HIGH"], n_sess_attack), 1),
        "sess_attack_medium_pct": round(_pct(sess_attack_conf["MEDIUM"], n_sess_attack), 1),
        "sess_attack_low_pct": round(_pct(sess_attack_conf["LOW"], n_sess_attack), 1),
        "teid_attack_high_medium_pct": round(
            _pct(teid_attack_conf["HIGH"] + teid_attack_conf["MEDIUM"], n_teid_attack), 1
        ),
        "sess_attack_high_medium_pct": round(
            _pct(sess_attack_conf["HIGH"] + sess_attack_conf["MEDIUM"], n_sess_attack), 1
        ),
        "teid_usable_for_training": teid_usable,
        "sess_usable_for_training": sess_usable,
        "teid_usable_pct": round(_pct(teid_usable, n_teid), 1),
        "sess_usable_pct": round(_pct(sess_usable, n_sess), 1),
    }


def write_csv(rows: list[dict]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "summary.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def render_report(rows: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# Aggregate Labeling Validation Report (all 9 attack types, BS1)\n")
    lines.append(
        "Generated by `scripts/validate_labeling_all.py`. Answers a methodological "
        "question raised after the single-file report "
        "(`outputs/reports/labeling_validation/report.md`) showed SYNScan_BS1 at "
        "97.9% LOW-confidence sessions: does that generalize across attack types, "
        "and how many HIGH/MEDIUM-confidence samples remain per type if LOW-"
        "confidence attack labels are excluded from supervised training (kept only "
        "for evaluation / uncertainty analysis / qualitative discussion)?\n"
    )

    lines.append("## Summary table\n")
    header = (
        "| Attack type | Window(s) | TEIDs | Sessions | Attack % "
        "| TEID HIGH % | MEDIUM % | LOW % | HIGH+MEDIUM % (TEID) | HIGH+MEDIUM % (session) "
        "| Usable TEIDs | Usable Sessions |"
    )
    sep = "|---" * 12 + "|"
    lines.append(header)
    lines.append(sep)
    for r in rows:
        flag = " ⚠️" if r["teid_attack_high_medium_pct"] < HIGH_MEDIUM_MIN_PCT else ""
        lines.append(
            f"| {r['attack_type']}{flag} | {r['session_window_s']} | {r['n_teid']} | {r['n_sess']} "
            f"| {r['teid_attack_pct']}% "
            f"| {r['teid_attack_high_pct']}% | {r['teid_attack_medium_pct']}% "
            f"| {r['teid_attack_low_pct']}% | {r['teid_attack_high_medium_pct']}% "
            f"| {r['sess_attack_high_medium_pct']}% "
            f"| {r['teid_usable_for_training']} ({r['teid_usable_pct']}%) "
            f"| {r['sess_usable_for_training']} ({r['sess_usable_pct']}%) |"
        )
    lines.append(
        "\n_HIGH/MEDIUM/LOW % are computed **within attack-labeled TEID instances "
        "only** (the metric that matters for training ground-truth quality), not "
        'over all instances. "Window(s)" is the session window size used for that '
        "type -- 30s for scan types (SYNScan/TCPConnect/UDPScan, per the Priority 2 "
        "fix in confidence_diagnosis/report.md), 5s for everything else. "
        '"Usable TEIDs/Sessions" = HIGH+MEDIUM confidence instances of either class '
        "(attack or benign) -- the estimated size of a supervised training set that "
        "excludes LOW-confidence labels entirely._\n"
    )

    flagged = [r for r in rows if r["teid_attack_high_medium_pct"] < HIGH_MEDIUM_MIN_PCT]
    lines.append(f"## Attack types below the {HIGH_MEDIUM_MIN_PCT:.0f}% HIGH+MEDIUM threshold\n")
    if flagged:
        for r in flagged:
            lines.append(
                f"- **{r['attack_type']}**: only {r['teid_attack_high_medium_pct']}% of "
                f"attack-labeled TEID instances are HIGH+MEDIUM confidence "
                f"({r['n_teid_attack']} attack-labeled TEIDs total). Recommendation: "
                f"exclude this type's LOW-confidence attack labels from the supervised "
                f"training set; use them only for evaluation, uncertainty analysis, or "
                f"qualitative discussion in the thesis, per the KNOWN LIMITATION note "
                f"in `preprocessing/labeling.py`."
            )
    else:
        lines.append("None -- every attack type clears the threshold.")

    lines.append("\n## Per-type detail\n")
    lines.append("| Attack type | Packets | Span (s) | TEIDs (attack) | Sessions (attack) |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['attack_type']} | {r['n_packets']:,} | {r['span_s']} "
            f"| {r['n_teid']} ({r['n_teid_attack']}) | {r['n_sess']} ({r['n_sess_attack']}) |"
        )

    total_usable_teid = sum(r["teid_usable_for_training"] for r in rows)
    total_teid = sum(r["n_teid"] for r in rows)
    total_usable_sess = sum(r["sess_usable_for_training"] for r in rows)
    total_sess = sum(r["n_sess"] for r in rows)
    lines.append(
        f"\n## Overall (all 9 types combined, BS1 only)\n\n"
        f"- Total TEID instances: {total_teid}, usable (HIGH+MEDIUM) for training: "
        f"{total_usable_teid} ({_pct(total_usable_teid, total_teid):.1f}%)\n"
        f"- Total sessions (5s window): {total_sess}, usable (HIGH+MEDIUM) for "
        f"training: {total_usable_sess} ({_pct(total_usable_sess, total_sess):.1f}%)\n\n"
        f"This is BS1 only; BS2 has not been run yet (see follow-up note below)."
    )

    lines.append(
        "\n## Recommendation for Phase 6 (ML/agentic training)\n\n"
        "1. Build the supervised training set from HIGH+MEDIUM confidence labels "
        "only (both attack and benign side), per attack type -- do not blindly pool "
        "LOW-confidence attack labels in as positives.\n"
        "2. For any attack type flagged above the threshold list, treat its "
        "LOW-confidence attack instances as a held-out uncertainty/qualitative set: "
        "report detection behavior on them separately, don't count them in "
        "precision/recall/F1.\n"
        "3. Re-run this script for BS2 before finalizing the training set, and "
        "consider re-running with the full (uncapped) `max_duration_s` for any type "
        "whose captured span turns out to exceed 2200s, to rule out truncation bias."
    )
    return "\n".join(lines)


def main() -> None:
    schedule = AttackSchedule.load(PROJECT_ROOT / "configs" / "attack_schedule.yaml")
    patterns = LabelPatternsConfig.load(PROJECT_ROOT / "configs" / "label_patterns.yaml")

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
            session_window_s=_session_window_s(attack_type),
        )
        stats = compute_stats(data)
        rows.append(stats)
        elapsed = time.time() - t0
        hm_pct = stats["teid_attack_high_medium_pct"]
        print(
            f"[{attack_type}] done in {elapsed:.1f}s: {stats['n_teid']} TEIDs "
            f"({stats['n_teid_attack']} attack), {stats['n_sess']} sessions "
            f"({stats['n_sess_attack']} attack), HIGH+MEDIUM={hm_pct}%"
        )
        # Checkpoint after every file so a long run's partial progress survives
        # an interruption.
        write_csv(rows)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "report.md"
    report_path.write_text(render_report(rows), encoding="utf-8")
    print(f"\nReport written to {report_path}")
    print(f"CSV written to {REPORT_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()
