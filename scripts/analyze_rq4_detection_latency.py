"""RQ4 measurement study: does the agentic architecture detect attacks
earlier than a traditional ML classifier?

Pure measurement, no optimization. Explicitly does NOT retrain, recalibrate,
change any threshold/rule/label/split/config, or modify any Phase 4-7
artifact. Every model/threshold/rule used here is the exact frozen one:

  - `configs/thresholds.yaml` (agent rules) -- unmodified, loaded read-only.
  - The ML "model" is the identical frozen fit Phase 6/7 used (same
    RandomForest/XGBoost hyperparameters, same seed, same train split via
    `ml.dataset.build_gtp_session_dataset`) -- re-DERIVING that exact fit
    (verified bit-for-bit reproducible in Phase 7) is not retraining in
    any sense that changes results; no data, hyperparameter, or seed
    differs from Phase 6. XGBoost is used as the single representative ML
    baseline (it had the higher view-B ROC-AUC in Phase 6/7).
  - "Attack start" uses the existing Level-1 ground truth
    (`preprocessing.labeling._approximate_attack_subwindow`), not a new
    label definition.
  - The ML decision threshold is the existing frozen 0.5 default
    (`predict_proba >= 0.5`, matching how `RandomForestModel`/
    `XGBoostModel.predict()` already decide internally) -- no new
    threshold introduced.

Unit of measurement: per (ue_ip, teid) conversation group within an
attack-type file (not per attack-type file as a whole), since a single
file has exactly one attack_start but typically many independent
conversation groups -- this is what makes a per-attack-type
mean/median/min/max distribution meaningful without touching the frozen
BS1-only scope (see the honest discussion of why this, and not something
else, is the unit of analysis).

For the AGENT: `PDUSessionAgent.annotate_series` is run over each group's
FULL chronological session sequence for that file (no train/test split
applies -- the agent is deterministic and rule-based, never trained, so
there is no leakage concern).

For the ML MODEL: scored over the SAME full chronological sequence using
the re-derived frozen fit. This means some scored sessions were part of
that fit's own training set for attack types whose groups mostly fall in
the Phase 6 train split (an explicit, documented limitation below -- not
hidden). A stricter test-split-only measurement is also reported per type
where it is defined (i.e. where at least one attack group exists in the
test split); for most types it is NOT defined, since Phase 7's own error
analysis already found most attack types have zero attack sessions in
the held-out test split.

Usage:
    poetry run python scripts/analyze_rq4_detection_latency.py
"""

from __future__ import annotations

import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from agente_5g.agents.pdu_session_agent import STATE_ORDER, PDUSessionAgent  # noqa: E402
from agente_5g.ml.dataset import (  # noqa: E402
    GTP_SESSION_FEATURE_COLUMNS,
    build_gtp_session_dataset,
    to_gtp_matrix,
)
from agente_5g.ml.xgboost_model import XGBoostModel  # noqa: E402
from agente_5g.models.agent_thresholds import ThresholdsConfig  # noqa: E402
from agente_5g.models.schedule_config import AttackSchedule, LabelPatternsConfig  # noqa: E402
from agente_5g.preprocessing.labeling import _approximate_attack_subwindow  # noqa: E402
from scripts.validate_labeling import process_file  # noqa: E402
from scripts.validate_labeling_all import _session_window_s  # noqa: E402

_LINE_COLOR = "#1565C0"
_ML_MARKER_COLOR = "#EF6C00"

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports" / "rq4_detection_latency"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures" / "rq4_detection_latency"
TEST_FRACTION = 0.3
MAX_DURATION_S = 2200.0
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
# One representative attack type per category for the timeline figures.
REPRESENTATIVE = {"flood": "ICMPflood", "scan": "SYNScan", "slow-rate": "Slowloris"}


def _first_time_at_or_above(
    sessions: list[Any], state_sequence: list[str], target_state: str
) -> float | None:
    target_rank = STATE_ORDER.index(target_state)
    for session, state in zip(sessions, state_sequence, strict=True):
        if STATE_ORDER.index(state) >= target_rank:
            return session.end_time
    return None


def _first_ml_positive_time(
    sessions: list[Any], model: XGBoostModel, test_session_ids: set[str]
) -> tuple[float | None, float | None]:
    """Returns (full_timeline_time, test_only_time). test_only_time is None
    if no session in this group is in the test split."""
    x = pd.DataFrame(
        [{col: getattr(s, col) for col in GTP_SESSION_FEATURE_COLUMNS} for s in sessions]
    )
    proba = model.predict_proba(x)
    full_time = None
    test_time = None
    for session, p in zip(sessions, proba, strict=True):
        if p >= 0.5:
            if full_time is None:
                full_time = session.end_time
            if test_time is None and session.session_id in test_session_ids:
                test_time = session.end_time
    return full_time, test_time


def measure_attack_type(
    attack_type: str,
    schedule: AttackSchedule,
    patterns: LabelPatternsConfig,
    pdu_agent: PDUSessionAgent,
    ml_model: XGBoostModel,
    test_session_ids: set[str],
) -> list[dict[str, Any]]:
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
    packets = data["packets"]
    file_first_ts = min(p.timestamp for p in packets)
    file_last_ts = max(p.timestamp for p in packets)
    subwindow = _approximate_attack_subwindow(
        schedule, attack_type, "BS1", file_first_ts, file_last_ts
    )
    if subwindow is None:
        return []
    attack_start = subwindow[0]

    sessions = data["sessions"]
    groups: dict[tuple[str, int], list[Any]] = defaultdict(list)
    for s in sessions:
        if s.is_attack:
            groups[(s.ue_ip, s.teid)].append(s)
    # a group qualifies if ANY of its sessions is labeled attack, but we
    # score/annotate the group's FULL session history (not just the
    # attack-labeled sessions) so the state machine sees real context
    all_sessions_by_group: dict[tuple[str, int], list[Any]] = defaultdict(list)
    for s in sessions:
        all_sessions_by_group[(s.ue_ip, s.teid)].append(s)

    rows: list[dict[str, Any]] = []
    for key in groups:
        group_sessions = sorted(all_sessions_by_group[key], key=lambda s: s.start_time)
        annotated = pdu_agent.annotate_series(group_sessions)
        state_sequence = [s.final_state for s in annotated]

        t_watch = _first_time_at_or_above(annotated, state_sequence, "WATCH")
        t_suspicious = _first_time_at_or_above(annotated, state_sequence, "SUSPICIOUS")
        t_attack = _first_time_at_or_above(annotated, state_sequence, "ATTACK")
        t_ml_full, t_ml_test = _first_ml_positive_time(annotated, ml_model, test_session_ids)

        rows.append(
            {
                "attack_type": attack_type,
                "ue_ip": key[0],
                "teid": key[1],
                "attack_start": attack_start,
                "time_to_first_watch": (t_watch - attack_start) if t_watch is not None else None,
                "time_to_first_suspicious": (
                    (t_suspicious - attack_start) if t_suspicious is not None else None
                ),
                "time_to_first_attack": (t_attack - attack_start) if t_attack is not None else None,
                "time_to_first_ml_detection_full": (
                    (t_ml_full - attack_start) if t_ml_full is not None else None
                ),
                "time_to_first_ml_detection_test_only": (
                    (t_ml_test - attack_start) if t_ml_test is not None else None
                ),
                "lead_time_watch": (
                    (t_ml_full - t_watch) if t_ml_full is not None and t_watch is not None else None
                ),
                "lead_time_suspicious": (
                    (t_ml_full - t_suspicious)
                    if t_ml_full is not None and t_suspicious is not None
                    else None
                ),
                "lead_time_attack": (
                    (t_ml_full - t_attack)
                    if t_ml_full is not None and t_attack is not None
                    else None
                ),
                "_state_sequence": state_sequence,
                "_session_times": [s.end_time - attack_start for s in annotated],
                "_ml_proba": None,  # filled below only for representative groups if needed
            }
        )
    return rows


def _stats(values: list[float]) -> dict[str, float | None]:
    clean = [v for v in values if v is not None]
    if not clean:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(clean),
        "mean": round(statistics.mean(clean), 3),
        "median": round(statistics.median(clean), 3),
        "min": round(min(clean), 3),
        "max": round(max(clean), 3),
    }


METRIC_LABELS = [
    ("time_to_first_watch", "Time to first WATCH (s)"),
    ("time_to_first_suspicious", "Time to first SUSPICIOUS (s)"),
    ("time_to_first_attack", "Time to first ATTACK (s)"),
    ("time_to_first_ml_detection_full", "Time to first ML detection, full timeline (s)"),
    ("time_to_first_ml_detection_test_only", "Time to first ML detection, test-split only (s)"),
    ("lead_time_watch", "Detection lead time: ML - WATCH (s)"),
    ("lead_time_suspicious", "Detection lead time: ML - SUSPICIOUS (s)"),
    ("lead_time_attack", "Detection lead time: ML - ATTACK (s)"),
]


def render_report(all_rows: list[dict[str, Any]]) -> str:
    lines = ["# RQ4 Detection-Latency Measurement Study\n"]
    lines.append(
        "Generated by `scripts/analyze_rq4_detection_latency.py`. Pure measurement -- no "
        "model/threshold/rule/label/split/config was changed to produce this. See the module "
        "docstring for exactly what was reused frozen vs. re-derived (the ML fit) and why. "
        "Positive `lead_time_*` means the agent event happened BEFORE ML's first detection "
        "(agent earlier); negative means ML detected first.\n"
    )

    lines.append("\n## Per-attack-type summary\n")
    for attack_type in ATTACK_TYPES:
        type_rows = [r for r in all_rows if r["attack_type"] == attack_type]
        lines.append(
            f"\n### {attack_type} (n={len(type_rows)} attack-labeled conversation groups)\n"
        )
        if not type_rows:
            lines.append("No attack-labeled (ue_ip, teid) groups found in this file. n/a.\n")
            continue
        lines.append("| Metric | n | mean | median | min | max |")
        lines.append("|---|---|---|---|---|---|")
        for key, label in METRIC_LABELS:
            s = _stats([r[key] for r in type_rows])
            lines.append(
                f"| {label} | {s['n']} | {s['mean']} | {s['median']} | {s['min']} | {s['max']} |"
            )

    lines.append("\n## Overall summary (pooled across all 9 attack types)\n")
    lines.append("| Metric | n | mean | median | min | max |")
    lines.append("|---|---|---|---|---|---|")
    for key, label in METRIC_LABELS:
        s = _stats([r[key] for r in all_rows])
        lines.append(
            f"| {label} | {s['n']} | {s['mean']} | {s['median']} | {s['min']} | {s['max']} |"
        )

    n_test_only_defined = sum(
        1 for r in all_rows if r["time_to_first_ml_detection_test_only"] is not None
    )
    lines.append(
        f"\nTest-split-only ML detection time is defined for {n_test_only_defined}/{len(all_rows)} "
        "groups -- see Limitations.\n"
    )

    return "\n".join(lines)


def plot_timeline(attack_type: str, row: dict[str, Any], category: str) -> go.Figure:
    state_ranks = [STATE_ORDER.index(s) for s in row["_state_sequence"]]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=row["_session_times"],
            y=state_ranks,
            mode="lines+markers",
            name="Agent state",
            line={"shape": "hv", "color": _LINE_COLOR},
        )
    )
    fig.add_vline(x=0, line={"dash": "dash", "color": "#C62828"}, annotation_text="attack_start")
    if row["time_to_first_ml_detection_full"] is not None:
        fig.add_vline(
            x=row["time_to_first_ml_detection_full"],
            line={"dash": "dot", "color": _ML_MARKER_COLOR},
            annotation_text="ML first detection",
        )
    fig.update_layout(
        title=f"State timeline -- {attack_type} ({category}), UE {row['ue_ip']}",
        xaxis_title="Time relative to attack_start (s)",
        yaxis={"tickmode": "array", "tickvals": [0, 1, 2, 3], "ticktext": STATE_ORDER},
        template="plotly_white",
        margin={"t": 60, "b": 40},
    )
    return fig


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    schedule = AttackSchedule.load(PROJECT_ROOT / "configs" / "attack_schedule.yaml")
    patterns = LabelPatternsConfig.load(PROJECT_ROOT / "configs" / "label_patterns.yaml")
    thresholds = ThresholdsConfig.load(PROJECT_ROOT / "configs" / "thresholds.yaml")
    pdu_agent = PDUSessionAgent(thresholds.pdu_session_agent)

    print("Re-deriving the frozen arm-B XGBoost fit (identical to Phase 6/7, not retraining) ...")
    sessions_by_type = {}
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
        sessions_by_type[attack_type] = data["sessions"]
    train_df, test_df = build_gtp_session_dataset(sessions_by_type, test_fraction=TEST_FRACTION)
    x_train, y_train = to_gtp_matrix(train_df)
    ml_model = XGBoostModel()
    ml_model.fit(x_train, y_train)
    test_session_ids = set(test_df["session_id"])

    print("Measuring detection latency per attack type ...")
    all_rows: list[dict[str, Any]] = []
    for attack_type in ATTACK_TYPES:
        rows = measure_attack_type(
            attack_type, schedule, patterns, pdu_agent, ml_model, test_session_ids
        )
        all_rows.extend(rows)
        print(f"  [{attack_type}] {len(rows)} attack-labeled conversation groups measured")

    report = render_report(all_rows)
    (REPORT_DIR / "report.md").write_text(report, encoding="utf-8")
    print(f"Report written to {REPORT_DIR / 'report.md'}")

    print("Building representative timeline figures ...")
    for category, attack_type in REPRESENTATIVE.items():
        candidates = [r for r in all_rows if r["attack_type"] == attack_type]
        # prefer a group that reached ATTACK, for the most complete trajectory
        reached_attack = [r for r in candidates if r["time_to_first_attack"] is not None]
        pick = reached_attack[0] if reached_attack else (candidates[0] if candidates else None)
        if pick is None:
            print(f"  [{category}/{attack_type}] no qualifying group -- skipped")
            continue
        fig = plot_timeline(attack_type, pick, category)
        safe_name = f"timeline_{category}_{attack_type}".replace(" ", "_")
        fig.write_image(FIGURE_DIR / f"{safe_name}.png", scale=2)
        print(f"  [{category}/{attack_type}] {safe_name}.png")

    print(f"\nFigures written to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
