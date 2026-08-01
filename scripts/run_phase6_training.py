"""Phase 6: train and evaluate all four arms, real data end to end.

  A1_combined: RF + XGBoost on data/processed/Combined/Combined.csv (our
    own documented preprocessing) -- primary official baseline.
  A2_encoded:  RF + XGBoost on data/processed/Encoded/Encoded.csv (authors'
    own pre-encoded columns, near-verbatim) -- secondary reproducibility
    check only.
  B_gtp_ml:    RF + XGBoost on real labeled GTP/TEID/session features
    (the 9 BS1 attack-type files), TEID-safe split.
  C_agentic:   the existing Phase 5B-calibrated agentic pipeline, scored
    on the IDENTICAL test sessions arm B was evaluated on.

Uses the packet cache (outputs/cache/packets/) populated by earlier
validation scripts, so this reuses already-parsed data rather than
re-reading the raw pcapng files.

Writes outputs/reports/phase6_training/results.csv +
outputs/reports/phase6_training/report.md.

Usage:
    poetry run python scripts/run_phase6_training.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from agente_5g.ml.dataset import (  # noqa: E402
    build_gtp_session_dataset,
    load_combined_csv,
    load_encoded_csv,
)
from agente_5g.ml.train import (  # noqa: E402
    evaluate_arm_c,
    train_and_evaluate_arm_a,
    train_and_evaluate_arm_b,
)
from agente_5g.models.agent_thresholds import ThresholdsConfig  # noqa: E402
from agente_5g.models.evaluation import EvaluationResult  # noqa: E402
from agente_5g.models.schedule_config import AttackSchedule, LabelPatternsConfig  # noqa: E402
from scripts.validate_labeling import process_file  # noqa: E402
from scripts.validate_labeling_all import _session_window_s  # noqa: E402

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports" / "phase6_training"
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


def _result_row(r: EvaluationResult) -> dict[str, Any]:
    return {
        "arm": r.arm,
        "model_name": r.model_name,
        "accuracy": round(r.accuracy, 4),
        "precision": round(r.precision, 4),
        "recall": round(r.recall, 4),
        "f1": round(r.f1, 4),
        "roc_auc": round(r.roc_auc, 4) if r.roc_auc is not None else None,
        "fpr": round(r.fpr, 4),
        "fit_time_ms": round(r.detection_time_ms, 2),
        "inference_time_ms_per_sample": round(r.inference_time_ms, 5),
        "confusion_matrix": r.confusion_matrix,
    }


def run_arm_a1() -> list[EvaluationResult]:
    print("[A1_combined] loading Combined.csv ...")
    t0 = time.time()
    df = load_combined_csv(PROJECT_ROOT / "data" / "processed" / "Combined" / "Combined.csv")
    print(f"[A1_combined] loaded {len(df)} rows in {time.time() - t0:.1f}s, training ...")
    t0 = time.time()
    results = train_and_evaluate_arm_a(df, arm="A1_combined", test_fraction=TEST_FRACTION)
    print(f"[A1_combined] done in {time.time() - t0:.1f}s")
    return results


def run_arm_a2() -> list[EvaluationResult]:
    print("[A2_encoded] loading Encoded.csv ...")
    t0 = time.time()
    df = load_encoded_csv(PROJECT_ROOT / "data" / "processed" / "Encoded" / "Encoded.csv")
    print(f"[A2_encoded] loaded {len(df)} rows in {time.time() - t0:.1f}s, training ...")
    t0 = time.time()
    results = train_and_evaluate_arm_a(df, arm="A2_encoded", test_fraction=TEST_FRACTION)
    print(f"[A2_encoded] done in {time.time() - t0:.1f}s")
    return results


def run_arm_b_and_c(
    schedule: AttackSchedule, patterns: LabelPatternsConfig, thresholds: ThresholdsConfig
) -> list[EvaluationResult]:
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
        n_feat, n_sess = len(data["features"]), len(data["sessions"])
        print(f"[{attack_type}] {n_feat} TEID features, {n_sess} sessions")

    train_df, test_df = build_gtp_session_dataset(sessions_by_type, test_fraction=TEST_FRACTION)
    print(f"[B_gtp_ml] train={len(train_df)} sessions, test={len(test_df)} sessions")
    arm_b_results, _models = train_and_evaluate_arm_b(train_df, test_df)

    test_ids = set(test_df["session_id"])
    test_sessions_by_type = {
        attack_type: [s for s in sessions if s.session_id in test_ids]
        for attack_type, sessions in sessions_by_type.items()
    }
    arm_c_results = evaluate_arm_c(test_sessions_by_type, features_by_type, thresholds)

    return arm_b_results + arm_c_results


def render_report(rows: list[dict[str, Any]]) -> str:
    lines = ["# Phase 6: ML Baseline Results (all four arms)\n"]
    lines.append(
        "Generated by `scripts/run_phase6_training.py`. A1_combined is the primary official "
        "baseline; A2_encoded is a secondary reproducibility check only. B/C use view A (full "
        "schedule-labeled population) and view B (HIGH/MEDIUM-confidence corroborated only) -- "
        "see `experiment_plan.md`.\n"
    )
    lines.append(
        "| Arm | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | "
        "Fit(ms) | Infer(ms/sample) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        lines.append(
            f"| {row['arm']} | {row['model_name']} | {row['accuracy']} | {row['precision']} | "
            f"{row['recall']} | {row['f1']} | {row['roc_auc']} | {row['fpr']} | "
            f"{row['fit_time_ms']} | {row['inference_time_ms_per_sample']} |"
        )
    return "\n".join(lines)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    schedule = AttackSchedule.load(PROJECT_ROOT / "configs" / "attack_schedule.yaml")
    patterns = LabelPatternsConfig.load(PROJECT_ROOT / "configs" / "label_patterns.yaml")
    thresholds = ThresholdsConfig.load(PROJECT_ROOT / "configs" / "thresholds.yaml")

    all_results: list[EvaluationResult] = []
    all_results.extend(run_arm_a1())
    all_results.extend(run_arm_a2())
    all_results.extend(run_arm_b_and_c(schedule, patterns, thresholds))

    rows = [_result_row(r) for r in all_results]
    pd.DataFrame(rows).to_csv(REPORT_DIR / "results.csv", index=False)
    (REPORT_DIR / "report.md").write_text(render_report(rows), encoding="utf-8")
    print(f"\nResults written to {REPORT_DIR}")

    for row in rows:
        print(
            f"[{row['arm']}] {row['model_name']}: acc={row['accuracy']} prec={row['precision']} "
            f"rec={row['recall']} f1={row['f1']}"
        )


if __name__ == "__main__":
    main()
