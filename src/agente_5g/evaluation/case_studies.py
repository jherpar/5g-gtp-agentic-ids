"""Qualitative case studies: representative worked examples per attack
type, replacing a single scalar "explainability score" (see the plan's
"Explainability as qualitative case studies" decision). Answers RQ3 by
contrasting the agentic system's per-instance, human-readable `explanation`
string (`agents/supervisor_agent.py::SupervisorDecision.explanation`)
against classical ML's coarser, model-level feature importances -- not by
forcing them onto the same numeric scale.

Deliberately generic (works on plain tuples, not tied to any one arm) so
the same functions serve arm C's per-session `SupervisorDecision`
explanations and could equally serve any future per-instance text.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Literal

Outcome = Literal["TP", "FP", "FN", "TN"]

# (identifier, attack_type, true_label, predicted_label, risk_score, explanation)
CaseRecord = tuple[str, str, bool, bool, float, str]


@dataclass(frozen=True)
class CaseStudyExample:
    identifier: str
    attack_type: str
    outcome: Outcome
    true_label: bool
    predicted_label: bool
    risk_score: float
    explanation: str


def _outcome(true_label: bool, predicted_label: bool) -> Outcome:
    if true_label and predicted_label:
        return "TP"
    if not true_label and predicted_label:
        return "FP"
    if true_label and not predicted_label:
        return "FN"
    return "TN"


def select_case_studies(
    records: list[CaseRecord],
    outcomes: tuple[Outcome, ...] = ("TP", "FN", "FP"),
) -> dict[str, dict[Outcome, CaseStudyExample | None]]:
    """One representative example per (attack_type, outcome): the
    highest-risk_score match for TP/FP (the most "confident" catch or
    mistake), the lowest-risk_score match for FN (the most obviously
    missed case) -- FN examples are the most illustrative when they show
    the score wasn't even close. None if no example of that outcome
    exists for that attack type."""
    by_type: dict[str, list[CaseRecord]] = defaultdict(list)
    for record in records:
        by_type[record[1]].append(record)

    result: dict[str, dict[Outcome, CaseStudyExample | None]] = {}
    for attack_type, items in by_type.items():
        result[attack_type] = {}
        for outcome in outcomes:
            matching = [r for r in items if _outcome(r[2], r[3]) == outcome]
            if not matching:
                result[attack_type][outcome] = None
                continue
            pick = min if outcome == "FN" else max
            identifier, _at, true_label, predicted_label, risk_score, explanation = pick(
                matching, key=lambda r: r[4]
            )
            result[attack_type][outcome] = CaseStudyExample(
                identifier=identifier,
                attack_type=attack_type,
                outcome=outcome,
                true_label=true_label,
                predicted_label=predicted_label,
                risk_score=risk_score,
                explanation=explanation,
            )
    return result


def error_analysis_summary(records: list[CaseRecord]) -> list[dict[str, object]]:
    """Per-attack-type TP/FP/FN/TN counts and the resulting recall/FPR."""
    by_type: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for record in records:
        by_type[record[1]].append((record[2], record[3]))

    rows: list[dict[str, object]] = []
    for attack_type, pairs in sorted(by_type.items()):
        counts = Counter(_outcome(t, p) for t, p in pairs)
        tp, fp, fn, tn = counts["TP"], counts["FP"], counts["FN"], counts["TN"]
        recall = tp / (tp + fn) if (tp + fn) else None
        fpr = fp / (fp + tn) if (fp + tn) else None
        rows.append(
            {
                "attack_type": attack_type,
                "n": len(pairs),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "recall": recall,
                "fpr": fpr,
            }
        )
    return rows
