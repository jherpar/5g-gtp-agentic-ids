"""Phase 7: comparative evaluation, visualization, case studies, and the
RQ-oriented writeup -- built entirely on the FROZEN Phase 6 configuration
(same seed, same `configs/thresholds.yaml`, same model hyperparameters).
No threshold or hyperparameter is tuned here; `evaluation.compare
.threshold_sensitivity` reports how metrics WOULD look across a sweep
purely for discussion, it never selects or applies a value.

`ml/train.py`'s public functions only return summary `EvaluationResult`
rows (Phase 6's committed numbers, `outputs/reports/phase6_training/`),
not the raw per-sample prediction arrays curves need. This script reuses
the same frozen primitives (model classes, dataset builders, agent
classes) to re-derive predictions for plotting, then cross-checks the
resulting confusion matrices against the committed Phase 6 CSV as a
reproducibility guard -- if they don't match bit-for-bit, that's a bug,
and the check prints a loud warning rather than silently disagreeing with
the "Phase 6 results are the primary reported results" instruction.

Usage:
    poetry run python scripts/run_phase7_analysis.py
"""

from __future__ import annotations

import ast
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from agente_5g.agents.pdu_session_agent import PDUSessionAgent  # noqa: E402
from agente_5g.agents.supervisor_agent import SupervisorAgent  # noqa: E402
from agente_5g.agents.teid_agent import TEIDAgent  # noqa: E402
from agente_5g.evaluation.case_studies import (  # noqa: E402
    CaseRecord,
    error_analysis_summary,
    select_case_studies,
)
from agente_5g.evaluation.compare import (  # noqa: E402
    pr_curve_data,
    roc_curve_data,
    threshold_sensitivity,
)
from agente_5g.evaluation.metrics import compute_metrics  # noqa: E402
from agente_5g.evaluation.visualize import (  # noqa: E402
    plot_confusion_matrix,
    plot_pr_curves,
    plot_roc_curves,
    plot_threshold_sensitivity,
)
from agente_5g.ml.dataset import (  # noqa: E402
    build_gtp_session_dataset,
    load_combined_csv,
    load_encoded_csv,
    per_group_chronological_split,
    to_arm_a_matrix,
    to_gtp_matrix,
)
from agente_5g.ml.random_forest import RandomForestModel  # noqa: E402
from agente_5g.ml.train import _match_feature  # noqa: E402
from agente_5g.ml.xgboost_model import XGBoostModel  # noqa: E402
from agente_5g.models.agent_thresholds import ThresholdsConfig  # noqa: E402
from agente_5g.models.schedule_config import AttackSchedule, LabelPatternsConfig  # noqa: E402
from scripts.validate_labeling import process_file  # noqa: E402
from scripts.validate_labeling_all import _session_window_s  # noqa: E402

FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures" / "phase7"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports" / "phase7_analysis"
PHASE6_RESULTS_CSV = PROJECT_ROOT / "outputs" / "reports" / "phase6_training" / "results.csv"

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
TEST_FRACTION = 0.3


def _verify_against_phase6(arm: str, model_name: str, y_true: Any, y_pred: Any) -> None:
    """Cross-checks a recomputed confusion matrix against the committed
    Phase 6 results.csv. Same seed/data/model -> must match exactly."""
    if not PHASE6_RESULTS_CSV.exists():
        return
    phase6 = pd.read_csv(PHASE6_RESULTS_CSV)
    row = phase6[(phase6["arm"] == arm) & (phase6["model_name"] == model_name)]
    if row.empty:
        return
    stored_cm = ast.literal_eval(row.iloc[0]["confusion_matrix"])
    recomputed_cm = compute_metrics(y_true, y_pred)["confusion_matrix"]
    status = "OK" if recomputed_cm == stored_cm else "MISMATCH"
    print(f"  [reproducibility check: {arm}/{model_name}] {status}")
    if status == "MISMATCH":
        print(f"    stored={stored_cm} recomputed={recomputed_cm}")


def run_arm_a(df: pd.DataFrame, arm: str) -> dict[str, dict[str, Any]]:
    train_df, test_df = per_group_chronological_split(
        df, group_column="Attack Type", order_column="Seq", test_fraction=TEST_FRACTION
    )
    x_train, y_train = to_arm_a_matrix(train_df)
    x_test, y_test = to_arm_a_matrix(test_df)

    predictions: dict[str, dict[str, Any]] = {}
    for model_name, model in [("RandomForest", RandomForestModel()), ("XGBoost", XGBoostModel())]:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_proba = model.predict_proba(x_test)
        _verify_against_phase6(arm, model_name, y_test, y_pred)
        predictions[f"{arm}/{model_name}"] = {
            "y_true": y_test.tolist(),
            "y_proba": y_proba.tolist(),
        }
    return predictions


def run_arm_b(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    x_train, y_train = to_gtp_matrix(train_df)
    x_test, y_test = to_gtp_matrix(test_df)
    corroborated_mask = test_df["label_confidence"].isin(["HIGH", "MEDIUM"]).to_numpy()

    predictions: dict[str, dict[str, Any]] = {}
    for model_name, model in [("RandomForest", RandomForestModel()), ("XGBoost", XGBoostModel())]:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_proba = model.predict_proba(x_test)
        _verify_against_phase6("B_gtp_ml", f"{model_name} (view A)", y_test, y_pred)
        _verify_against_phase6(
            "B_gtp_ml",
            f"{model_name} (view B)",
            y_test[corroborated_mask],
            y_pred[corroborated_mask],
        )
        predictions[f"B_gtp_ml/{model_name} (view A)"] = {
            "y_true": y_test.tolist(),
            "y_proba": y_proba.tolist(),
        }
        predictions[f"B_gtp_ml/{model_name} (view B)"] = {
            "y_true": y_test[corroborated_mask].tolist(),
            "y_proba": y_proba[corroborated_mask].tolist(),
        }
    return predictions


def run_arm_c(
    test_sessions_by_type: dict[str, list[Any]],
    features_by_type: dict[str, list[Any]],
    thresholds: ThresholdsConfig,
) -> tuple[dict[str, dict[str, Any]], list[CaseRecord]]:
    teid_agent = TEIDAgent(thresholds.teid_agent)
    pdu_agent = PDUSessionAgent(thresholds.pdu_session_agent)
    supervisor = SupervisorAgent(thresholds.supervisor_agent)

    case_records: list[CaseRecord] = []
    y_true: list[bool] = []
    y_pred: list[bool] = []
    y_proba: list[float] = []
    mask: list[bool] = []

    for attack_type, sessions in test_sessions_by_type.items():
        features = features_by_type.get(attack_type, [])
        feats_by_teid: dict[int, list[Any]] = defaultdict(list)
        for f in features:
            feats_by_teid[f.teid].append(f)

        sessions_by_key: dict[tuple[str, int], list[Any]] = defaultdict(list)
        for s in sessions:
            sessions_by_key[(s.ue_ip, s.teid)].append(s)
        annotated: list[Any] = []
        for group in sessions_by_key.values():
            annotated.extend(pdu_agent.annotate_series(group))

        for session in annotated:
            matched_feat = _match_feature(session, feats_by_teid)
            if matched_feat is None:
                continue
            teid_decision = teid_agent.evaluate(matched_feat)
            session_decision = pdu_agent.decide(session)
            sup = supervisor.fuse(
                entity_id=session.session_id,
                teid_decision=teid_decision,
                session_decision=session_decision,
                predicted_attack_type=attack_type,
            )
            predicted = sup.final_label == "Attack"
            true = bool(session.is_attack)
            y_true.append(true)
            y_pred.append(predicted)
            y_proba.append(sup.fused_risk_score)
            confidence = session.label_confidence.value if session.label_confidence else None
            mask.append(confidence in ("HIGH", "MEDIUM"))
            case_records.append(
                (
                    session.session_id[:12],
                    attack_type,
                    true,
                    predicted,
                    sup.fused_risk_score,
                    sup.explanation,
                )
            )

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    y_proba_arr = np.array(y_proba)
    mask_arr = np.array(mask)

    _verify_against_phase6(
        "C_agentic", "Agentic (TEID+PDU+Supervisor) (view A)", y_true_arr, y_pred_arr
    )
    _verify_against_phase6(
        "C_agentic",
        "Agentic (TEID+PDU+Supervisor) (view B)",
        y_true_arr[mask_arr],
        y_pred_arr[mask_arr],
    )

    predictions = {
        "C_agentic/Agentic (view A)": {
            "y_true": y_true_arr.tolist(),
            "y_proba": y_proba_arr.tolist(),
        },
        "C_agentic/Agentic (view B)": {
            "y_true": y_true_arr[mask_arr].tolist(),
            "y_proba": y_proba_arr[mask_arr].tolist(),
        },
    }
    return predictions, case_records


def render_writeup(
    phase6_results: pd.DataFrame,
    error_rows: list[dict[str, Any]],
    case_studies: dict[str, dict[str, Any]],
) -> str:
    lines = ["# Phase 7: Comparative Evaluation and Research-Question Writeup\n"]
    lines.append(
        "Generated by `scripts/run_phase7_analysis.py`. Phase 6 results (frozen configuration, "
        "`outputs/reports/phase6_training/results.csv`) are the primary reported results; this "
        "report adds ROC/PR curves, threshold-sensitivity discussion, case studies, and error "
        "analysis on top of them, without changing any configured threshold or hyperparameter.\n"
    )

    lines.append("\n## Primary results (Phase 6, frozen)\n")
    if not phase6_results.empty:
        cols = [c for c in phase6_results.columns if c != "confusion_matrix"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * len(cols))
        for _, row in phase6_results.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")

    lines.append(
        "\n\n## RQ1: Does GTP-U/TEID information improve detection over the "
        "official flow dataset?\n"
    )
    lines.append(
        "Arms A1/A2 (official Combined.csv/Encoded.csv, no GTP-U info) reach F1 ~0.95-0.96 with "
        "ROC-AUC 0.96-0.99. Arm B (GTP-U session features) reaches F1 only 0.23-0.42 at the "
        "default 0.5 cutoff, but ROC-AUC 0.94-0.97 -- comparable ranking ability to arm A. On raw "
        "classification metrics the answer is NO for this dataset as constructed; the ROC-AUC gap "
        "is far smaller, indicating the GTP-U features carry real discriminative signal that a "
        "~20x smaller, noisier (~14.5% independently corroborated) training set and an "
        "uncalibrated "
        "default threshold fail to convert into strong precision/recall at a fixed cutoff. This is "
        "a data-scale and label-quality confound, not evidence that GTP-U/TEID information itself "
        "is uninformative.\n"
    )

    lines.append(
        "\n## RQ2: Does the agentic architecture match/exceed ML given identical features?\n"
    )
    lines.append(
        "Mixed, not a clean win or loss. Arm C has far better precision (0.34-0.49 vs 0.13-0.27) "
        "and FPR (0.018 vs 0.38-0.43) than arm B at their respective operating points, but far "
        "worse recall (0.12-0.15 vs 0.95-0.99). By ROC-AUC alone, arm C's view-B score (0.947) is "
        "competitive with arm B's trained models (0.94-0.97) -- the agent's risk score ranks "
        "attacks about as well; it is simply evaluated at a far more conservative fixed decision "
        "threshold (Phase 5B calibrated `attack_decision_threshold` for low false positives, not "
        "for recall). See the threshold-sensitivity plots for how each arm's precision/recall "
        "trade off across the full threshold range -- shown for discussion only, no threshold "
        "was changed.\n"
    )

    lines.append("\n## RQ3: Does TEID/session-level reasoning improve explainability?\n")
    lines.append(
        "Qualitatively yes: the agentic system produces a concrete, human-readable, per-decision "
        "explanation (see case studies below) built from named triggered rules and their measured "
        'values (e.g. "flood (packets_per_s=X >= Y)"). Classical ML (arms A/B) offers only a '
        "model-level, non-per-instance explanation (global feature importances) unless a "
        "post-hoc method (e.g. SHAP) were added, which this project does not implement. This is "
        "the intended comparison from the plan -- a capability difference, not something to "
        "'fix' on the ML side.\n"
    )

    lines.append("\n## RQ4: Can attacks be detected earlier via TEID/session reasoning?\n")
    lines.append(
        "Partial answer from available evidence. Both arm B and arm C have sub-millisecond "
        "per-sample inference latency (Phase 6 results.csv) once trained/configured, so raw "
        "wall-clock speed is not the differentiator. The qualitative distinction is architectural: "
        "`PDUSessionAgent`'s state machine (NORMAL->WATCH->SUSPICIOUS->ATTACK) produces staged "
        "early-warning signal across a session's own timeline that a single-shot flow classifier "
        "does not have by construction -- a session can surface as WATCH/SUSPICIOUS before ever "
        "reaching the final ATTACK verdict. This project does not yet measure time-to-first-flag "
        "empirically (would require replaying sessions in temporal order and recording when "
        "state first elevates); left as a concrete Phase 8 extension.\n"
    )

    lines.append("\n## Error analysis (arm C, per attack type)\n")
    lines.append("| Attack type | n | TP | FP | FN | TN | Recall | FPR |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in error_rows:
        recall = f"{row['recall']:.3f}" if row["recall"] is not None else "n/a"
        fpr = f"{row['fpr']:.3f}" if row["fpr"] is not None else "n/a"
        lines.append(
            f"| {row['attack_type']} | {row['n']} | {row['tp']} | {row['fp']} | {row['fn']} | "
            f"{row['tn']} | {recall} | {fpr} |"
        )

    lines.append("\n## Per-attack-type case studies (arm C)\n")
    for attack_type, outcomes in sorted(case_studies.items()):
        lines.append(f"\n### {attack_type}\n")
        for outcome_name in ("TP", "FN", "FP"):
            example = outcomes.get(outcome_name)
            if example is None:
                lines.append(f"- **{outcome_name}**: none in test set\n")
                continue
            lines.append(
                f"- **{outcome_name}** (session `{example.identifier}`, "
                f"risk={example.risk_score:.3f}): {example.explanation}\n"
            )

    return "\n".join(lines)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    schedule = AttackSchedule.load(PROJECT_ROOT / "configs" / "attack_schedule.yaml")
    patterns = LabelPatternsConfig.load(PROJECT_ROOT / "configs" / "label_patterns.yaml")
    thresholds = ThresholdsConfig.load(PROJECT_ROOT / "configs" / "thresholds.yaml")

    all_predictions: dict[str, dict[str, Any]] = {}

    print("[A1_combined] re-deriving predictions ...")
    t0 = time.time()
    df_a1 = load_combined_csv(PROJECT_ROOT / "data" / "processed" / "Combined" / "Combined.csv")
    all_predictions.update(run_arm_a(df_a1, "A1_combined"))
    print(f"[A1_combined] done in {time.time() - t0:.1f}s")

    print("[A2_encoded] re-deriving predictions ...")
    t0 = time.time()
    df_a2 = load_encoded_csv(PROJECT_ROOT / "data" / "processed" / "Encoded" / "Encoded.csv")
    all_predictions.update(run_arm_a(df_a2, "A2_encoded"))
    print(f"[A2_encoded] done in {time.time() - t0:.1f}s")

    sessions_by_type = {}
    features_by_type = {}
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
        features_by_type[attack_type] = data["features"]

    train_df, test_df = build_gtp_session_dataset(sessions_by_type, test_fraction=TEST_FRACTION)
    print("[B_gtp_ml] re-deriving predictions ...")
    all_predictions.update(run_arm_b(train_df, test_df))

    test_ids = set(test_df["session_id"])
    test_sessions_by_type = {
        attack_type: [s for s in sessions if s.session_id in test_ids]
        for attack_type, sessions in sessions_by_type.items()
    }
    print("[C_agentic] re-deriving predictions ...")
    arm_c_predictions, case_records = run_arm_c(test_sessions_by_type, features_by_type, thresholds)
    all_predictions.update(arm_c_predictions)

    print("Building ROC/PR curves ...")
    roc_curves = {
        name: roc_curve_data(p["y_true"], p["y_proba"]) for name, p in all_predictions.items()
    }
    pr_curves = {
        name: pr_curve_data(p["y_true"], p["y_proba"]) for name, p in all_predictions.items()
    }

    view_a_names = [n for n in all_predictions if "view A" in n or ("A1" in n or "A2" in n)]
    view_b_names = [n for n in all_predictions if "view B" in n]

    plot_roc_curves(
        {n: roc_curves[n] for n in view_a_names}, "ROC -- view A / official baselines"
    ).write_image(FIGURE_DIR / "roc_view_a.png", scale=2)
    plot_roc_curves(
        {n: roc_curves[n] for n in view_b_names}, "ROC -- view B (corroborated)"
    ).write_image(FIGURE_DIR / "roc_view_b.png", scale=2)
    plot_pr_curves(
        {n: pr_curves[n] for n in view_a_names}, "Precision-Recall -- view A"
    ).write_image(FIGURE_DIR / "pr_view_a.png", scale=2)
    plot_pr_curves(
        {n: pr_curves[n] for n in view_b_names}, "Precision-Recall -- view B"
    ).write_image(FIGURE_DIR / "pr_view_b.png", scale=2)

    print("Building confusion matrix figures for headline models ...")
    headline_models = [
        "A1_combined/XGBoost",
        "B_gtp_ml/XGBoost (view B)",
        "C_agentic/Agentic (view A)",
        "C_agentic/Agentic (view B)",
    ]
    for name in headline_models:
        p = all_predictions[name]
        y_proba_arr = np.array(p["y_proba"])
        y_pred_at_default_threshold = y_proba_arr >= 0.5
        cm = compute_metrics(p["y_true"], y_pred_at_default_threshold)["confusion_matrix"]
        safe_name = name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
        plot_confusion_matrix(cm, f"Confusion matrix -- {name}").write_image(
            FIGURE_DIR / f"confusion_matrix_{safe_name}.png", scale=2
        )

    print("Building threshold sensitivity plots (discussion only) ...")
    for name in ["B_gtp_ml/XGBoost (view B)", "C_agentic/Agentic (view B)"]:
        p = all_predictions[name]
        rows = threshold_sensitivity(p["y_true"], p["y_proba"])
        safe_name = name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
        plot_threshold_sensitivity(
            rows, f"Threshold sensitivity (discussion only) -- {name}"
        ).write_image(FIGURE_DIR / f"threshold_sensitivity_{safe_name}.png", scale=2)

    print("Building case studies and error analysis ...")
    case_studies = select_case_studies(case_records)
    error_rows = error_analysis_summary(case_records)

    phase6_results = (
        pd.read_csv(PHASE6_RESULTS_CSV) if PHASE6_RESULTS_CSV.exists() else pd.DataFrame()
    )
    report = render_writeup(phase6_results, error_rows, case_studies)
    (REPORT_DIR / "report.md").write_text(report, encoding="utf-8")
    print(f"\nReport written to {REPORT_DIR / 'report.md'}")
    print(f"Figures written to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
