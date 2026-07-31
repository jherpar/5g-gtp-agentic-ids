"""Phase 5 validation: run the agentic detectors on real labeled data.

Phase 5's agents (TEIDAgent/PDUSessionAgent/SupervisorAgent) were
implemented and unit-tested against synthetic data only. Before starting
Phase 6 (ML baselines), this script runs them on the real labeled TEID
features / PDU sessions for all 9 BS1 attack types and compares agent
decisions against the three label views established in `experiment_plan.md`:

  A) Schedule-based labels    -- `is_attack` (Level 1 only, full population)
  B) Corroborated labels      -- same, restricted to label_confidence != LOW
  C) Confidence-tier metadata -- recall/FPR broken down BY confidence tier,
                                  not filtered by it

Reports, per attack type and combined:
  1. Rule trigger rates (TEIDAgent: flood/syn_flood/scan; PDUSessionAgent:
     high_state_transition/low_temporal_entropy/high_diversity)
  2. Precision/recall/F1 for the fused SupervisorAgent decision, views A and B
  3. Recall (attack) / FPR (benign) by confidence tier (view C)
  4. State transition (final_state) distribution, cross-tabulated with the
     true label
  5. A handful of concrete false-positive / false-negative examples per type

SupervisorAgent fusion needs both a TEID-level and a session-level decision
for the same entity. TEID features (idle-gap-split instances) and PDU
sessions (fixed-window slices) don't share a key, so each session is paired
with the TEID feature instance for the same TEID whose window contains the
session's window; sessions with no such match are excluded from fused
(SupervisorAgent) metrics but still counted in the standalone PDUSessionAgent
section.

Usage:
    poetry run python scripts/validate_agents.py
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from agente_5g.agents.pdu_session_agent import PDUSessionAgent  # noqa: E402
from agente_5g.agents.supervisor_agent import SupervisorAgent  # noqa: E402
from agente_5g.agents.teid_agent import TEIDAgent  # noqa: E402
from agente_5g.models.agent_decision import AgentDecision  # noqa: E402
from agente_5g.models.agent_thresholds import ThresholdsConfig  # noqa: E402
from agente_5g.models.schedule_config import AttackSchedule, LabelPatternsConfig  # noqa: E402
from agente_5g.models.session import PDUSessionRecord  # noqa: E402
from agente_5g.models.teid_features import TEIDFeatureRecord  # noqa: E402
from scripts.validate_labeling import process_file  # noqa: E402
from scripts.validate_labeling_all import _session_window_s  # noqa: E402

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports" / "agent_validation"
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


def _match_feature(
    session: PDUSessionRecord, feats_by_teid: dict[int, list[TEIDFeatureRecord]]
) -> TEIDFeatureRecord | None:
    candidates = feats_by_teid.get(session.teid, [])
    contained = [
        f
        for f in candidates
        if f.window_start <= session.start_time and session.end_time <= f.window_end
    ]
    if contained:
        return min(contained, key=lambda f: f.window_end - f.window_start)
    if not candidates:
        return None
    mid = (session.start_time + session.end_time) / 2
    return min(candidates, key=lambda f: abs((f.window_start + f.window_end) / 2 - mid))


def _confusion(pairs: list[tuple[bool, bool]]) -> dict[str, Any]:
    tp = sum(1 for pred, true in pairs if pred and true)
    fp = sum(1 for pred, true in pairs if pred and not true)
    tn = sum(1 for pred, true in pairs if not pred and not true)
    fn = sum(1 for pred, true in pairs if not pred and true)
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _fmt(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else "n/a"


def process_attack_type(
    attack_type: str,
    schedule: AttackSchedule,
    patterns: LabelPatternsConfig,
    thresholds: ThresholdsConfig,
) -> dict[str, Any]:
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
    features: list[TEIDFeatureRecord] = data["features"]
    sessions: list[PDUSessionRecord] = data["sessions"]

    teid_agent = TEIDAgent(thresholds.teid_agent)
    pdu_agent = PDUSessionAgent(thresholds.pdu_session_agent)
    supervisor = SupervisorAgent(thresholds.supervisor_agent)

    # --- TEIDAgent: standalone ---
    teid_decisions: dict[int, AgentDecision] = {}
    teid_rule_counts: Counter[str] = Counter()
    teid_pairs_a: list[tuple[bool, bool]] = []
    teid_pairs_b: list[tuple[bool, bool]] = []
    feats_by_teid: dict[int, list[TEIDFeatureRecord]] = defaultdict(list)
    for idx, feat in enumerate(features):
        decision = teid_agent.evaluate(feat)
        teid_decisions[idx] = decision
        feats_by_teid[feat.teid].append(feat)
        for rule in decision.rule_triggers:
            teid_rule_counts[rule] += 1
        predicted = bool(decision.rule_triggers)
        true = bool(feat.is_attack)
        teid_pairs_a.append((predicted, true))
        if feat.label_confidence is not None and feat.label_confidence.value != "LOW":
            teid_pairs_b.append((predicted, true))

    # --- PDUSessionAgent: annotate_series per (ue_ip, teid), then decide ---
    sessions_by_key: dict[tuple[str, int], list[PDUSessionRecord]] = defaultdict(list)
    for s in sessions:
        sessions_by_key[(s.ue_ip, s.teid)].append(s)
    annotated: list[PDUSessionRecord] = []
    for group in sessions_by_key.values():
        annotated.extend(pdu_agent.annotate_series(group))

    session_decisions: dict[str, AgentDecision] = {}
    session_rule_counts: Counter[str] = Counter()
    session_pairs_a: list[tuple[bool, bool]] = []
    session_pairs_b: list[tuple[bool, bool]] = []
    state_dist: Counter[tuple[str, bool]] = Counter()
    confidence_tier_stats: dict[str, dict[str, int]] = {
        "HIGH": {"attack_n": 0, "attack_detected": 0, "benign_n": 0, "benign_flagged": 0},
        "MEDIUM": {"attack_n": 0, "attack_detected": 0, "benign_n": 0, "benign_flagged": 0},
        "LOW": {"attack_n": 0, "attack_detected": 0, "benign_n": 0, "benign_flagged": 0},
    }

    for session in annotated:
        decision = pdu_agent.decide(session)
        session_decisions[session.session_id] = decision
        for rule in decision.rule_triggers:
            session_rule_counts[rule] += 1
        predicted = bool(decision.rule_triggers)
        true = bool(session.is_attack)
        session_pairs_a.append((predicted, true))
        if session.label_confidence is not None and session.label_confidence.value != "LOW":
            session_pairs_b.append((predicted, true))
        state_dist[(session.final_state or "NORMAL", true)] += 1

        tier = session.label_confidence.value if session.label_confidence else None
        if tier in confidence_tier_stats:
            if true:
                confidence_tier_stats[tier]["attack_n"] += 1
                if predicted:
                    confidence_tier_stats[tier]["attack_detected"] += 1
            else:
                confidence_tier_stats[tier]["benign_n"] += 1
                if predicted:
                    confidence_tier_stats[tier]["benign_flagged"] += 1

    # --- SupervisorAgent: fuse each session with its matched TEID feature ---
    fused_pairs_a: list[tuple[bool, bool]] = []
    fused_pairs_b: list[tuple[bool, bool]] = []
    unmatched = 0
    false_positives: list[dict[str, Any]] = []
    false_negatives: list[dict[str, Any]] = []
    for session in annotated:
        matched_feat = _match_feature(session, feats_by_teid)
        if matched_feat is None:
            unmatched += 1
            continue
        feat_idx = features.index(matched_feat)
        teid_decision = teid_decisions[feat_idx]
        session_decision = session_decisions[session.session_id]
        sup_decision = supervisor.fuse(
            entity_id=session.session_id,
            teid_decision=teid_decision,
            session_decision=session_decision,
            predicted_attack_type=attack_type,
        )
        predicted = sup_decision.final_label == "Attack"
        true = bool(session.is_attack)
        fused_pairs_a.append((predicted, true))
        if session.label_confidence is not None and session.label_confidence.value != "LOW":
            fused_pairs_b.append((predicted, true))

        if predicted and not true and len(false_positives) < 3:
            false_positives.append(
                {"session_id": session.session_id[:12], "reason": sup_decision.explanation[:200]}
            )
        if not predicted and true and len(false_negatives) < 3:
            false_negatives.append(
                {
                    "session_id": session.session_id[:12],
                    "confidence": (
                        session.label_confidence.value if session.label_confidence else None
                    ),
                    "reason": sup_decision.explanation[:200],
                }
            )

    return {
        "attack_type": attack_type,
        "n_features": len(features),
        "n_sessions": len(sessions),
        "teid_rule_counts": dict(teid_rule_counts),
        "session_rule_counts": dict(session_rule_counts),
        "teid_view_a": _confusion(teid_pairs_a),
        "teid_view_b": _confusion(teid_pairs_b),
        "session_view_a": _confusion(session_pairs_a),
        "session_view_b": _confusion(session_pairs_b),
        "fused_view_a": _confusion(fused_pairs_a),
        "fused_view_b": _confusion(fused_pairs_b),
        "fused_unmatched": unmatched,
        "state_dist": dict(state_dist),
        "confidence_tier_stats": confidence_tier_stats,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def render(rows: list[dict[str, Any]]) -> str:
    lines = ["# Phase 5 Agent Validation Report (real BS1 labeled data)\n"]
    lines.append(
        "Generated by `scripts/validate_agents.py`. Compares TEIDAgent, "
        "PDUSessionAgent, and the fused SupervisorAgent decision against "
        "view A (schedule labels, full population), view B (HIGH+MEDIUM "
        "corroborated only), and view C (per-confidence-tier breakdown). "
        "See `experiment_plan.md` for what these views mean and why both "
        "exist.\n"
    )

    agg_fused_a: list[tuple[bool, bool]] = []
    agg_fused_b: list[tuple[bool, bool]] = []

    for row in rows:
        t = row["attack_type"]
        lines.append(f"\n## {t}\n")
        lines.append(f"TEID instances: {row['n_features']}, sessions: {row['n_sessions']}\n")

        lines.append("### 1. Rule trigger counts (real traffic)\n")
        lines.append(f"- TEIDAgent: {row['teid_rule_counts'] or 'none triggered'}")
        lines.append(f"- PDUSessionAgent: {row['session_rule_counts'] or 'none triggered'}\n")

        lines.append("### 2. Precision/recall/F1 -- fused SupervisorAgent decision\n")
        lines.append("| View | n | TP | FP | TN | FN | Precision | Recall | F1 |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for label, key in [("A (schedule)", "fused_view_a"), ("B (corroborated)", "fused_view_b")]:
            c = row[key]
            n = c["tp"] + c["fp"] + c["tn"] + c["fn"]
            lines.append(
                f"| {label} | {n} | {c['tp']} | {c['fp']} | {c['tn']} | {c['fn']} | "
                f"{_fmt(c['precision'])} | {_fmt(c['recall'])} | {_fmt(c['f1'])} |"
            )
        lines.append(
            f"\n(sessions unmatched to any TEID feature, excluded above: "
            f"{row['fused_unmatched']})\n"
        )

        lines.append("### 3. Detection rate / false-positive rate by confidence tier (view C)\n")
        lines.append("| Tier | attack n | detected | recall | benign n | flagged | FPR |")
        lines.append("|---|---|---|---|---|---|---|")
        for tier in ["HIGH", "MEDIUM", "LOW"]:
            s = row["confidence_tier_stats"][tier]
            recall = s["attack_detected"] / s["attack_n"] if s["attack_n"] else None
            fpr = s["benign_flagged"] / s["benign_n"] if s["benign_n"] else None
            lines.append(
                f"| {tier} | {s['attack_n']} | {s['attack_detected']} | {_fmt(recall)} | "
                f"{s['benign_n']} | {s['benign_flagged']} | {_fmt(fpr)} |"
            )

        lines.append("\n### 4. Session final_state distribution (state, is_attack) -> count\n")
        for (state, is_attack), count in sorted(row["state_dist"].items()):
            lines.append(f"- {state}, is_attack={is_attack}: {count}")

        if row["false_positives"]:
            lines.append("\n### 5a. Example false positives\n")
            for fp in row["false_positives"]:
                lines.append(f"- session {fp['session_id']}: {fp['reason']}")
        if row["false_negatives"]:
            lines.append("\n### 5b. Example false negatives\n")
            for fn in row["false_negatives"]:
                lines.append(
                    f"- session {fn['session_id']} (confidence={fn['confidence']}): {fn['reason']}"
                )

        agg_fused_a.extend(
            [(True, True)] * row["fused_view_a"]["tp"]
            + [(True, False)] * row["fused_view_a"]["fp"]
            + [(False, False)] * row["fused_view_a"]["tn"]
            + [(False, True)] * row["fused_view_a"]["fn"]
        )
        agg_fused_b.extend(
            [(True, True)] * row["fused_view_b"]["tp"]
            + [(True, False)] * row["fused_view_b"]["fp"]
            + [(False, False)] * row["fused_view_b"]["tn"]
            + [(False, True)] * row["fused_view_b"]["fn"]
        )

    lines.append("\n## Aggregate across all 9 attack types (fused SupervisorAgent decision)\n")
    lines.append("| View | n | TP | FP | TN | FN | Precision | Recall | F1 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for label, pairs in [("A (schedule)", agg_fused_a), ("B (corroborated)", agg_fused_b)]:
        c = _confusion(pairs)
        n = c["tp"] + c["fp"] + c["tn"] + c["fn"]
        lines.append(
            f"| {label} | {n} | {c['tp']} | {c['fp']} | {c['tn']} | {c['fn']} | "
            f"{_fmt(c['precision'])} | {_fmt(c['recall'])} | {_fmt(c['f1'])} |"
        )

    return "\n".join(lines)


def main() -> None:
    schedule = AttackSchedule.load(PROJECT_ROOT / "configs" / "attack_schedule.yaml")
    patterns = LabelPatternsConfig.load(PROJECT_ROOT / "configs" / "label_patterns.yaml")
    thresholds = ThresholdsConfig.load(PROJECT_ROOT / "configs" / "thresholds.yaml")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for attack_type in ATTACK_TYPES:
        row = process_attack_type(attack_type, schedule, patterns, thresholds)
        rows.append(row)
        fa = row["fused_view_a"]
        print(
            f"[{attack_type}] fused view A: precision={_fmt(fa['precision'])} "
            f"recall={_fmt(fa['recall'])} f1={_fmt(fa['f1'])}"
        )

    report_path = REPORT_DIR / "report.md"
    report_path.write_text(render(rows), encoding="utf-8")
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
