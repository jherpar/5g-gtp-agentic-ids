"""Training/evaluation orchestration for Phase 6's four arms, run under a
shared seed and scored through `evaluation/metrics.py`'s single code path
so results are directly comparable. Real-file I/O (reading CSVs/pcapng,
writing reports) lives in `scripts/run_phase6_training.py`; this module
takes already-loaded data and returns `EvaluationResult` rows.

Views A/B (see `experiment_plan.md`) apply to arms B and C, which are
built from our own multi-level-confidence labels: view A is the full
schedule-labeled population, view B restricts to HIGH/MEDIUM-confidence
(corroborated) instances only. Arms A1/A2 have no such confidence tiers
(the dataset authors' Label column is a flat binary), so they report one
view.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from collections import defaultdict
from typing import Literal

import pandas as pd

from agente_5g.agents.pdu_session_agent import PDUSessionAgent
from agente_5g.agents.supervisor_agent import SupervisorAgent
from agente_5g.agents.teid_agent import TEIDAgent
from agente_5g.evaluation.metrics import compute_metrics
from agente_5g.ml.dataset import (
    DEFAULT_TEST_FRACTION,
    SEED,
    per_group_chronological_split,
    to_arm_a_matrix,
    to_gtp_matrix,
)
from agente_5g.ml.random_forest import RandomForestModel
from agente_5g.ml.xgboost_model import XGBoostModel
from agente_5g.models.agent_thresholds import ThresholdsConfig
from agente_5g.models.evaluation import EvaluationResult
from agente_5g.models.session import PDUSessionRecord
from agente_5g.models.teid_features import TEIDFeatureRecord

ArmA = Literal["A1_combined", "A2_encoded"]
ArmBC = Literal["B_gtp_ml", "C_agentic"]


def _make_result(
    model_name: str,
    arm: ArmA | ArmBC,
    y_true: pd.Series,
    y_pred: object,
    y_proba: object,
    fit_s: float,
    infer_s: float,
) -> EvaluationResult:
    metrics = compute_metrics(y_true, y_pred, y_proba)
    n = len(y_true)
    config_hash = hashlib.sha256(f"{model_name}|{arm}|seed={SEED}".encode()).hexdigest()[:16]
    return EvaluationResult(
        model_name=model_name,
        arm=arm,
        accuracy=metrics["accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        roc_auc=metrics["roc_auc"],
        fpr=metrics["fpr"],
        detection_time_ms=fit_s * 1000,
        inference_time_ms=(infer_s * 1000 / n) if n else 0.0,
        confusion_matrix=metrics["confusion_matrix"],
        run_id=str(uuid.uuid4()),
        config_hash=config_hash,
    )


def train_and_evaluate_arm_a(
    df: pd.DataFrame,
    arm: ArmA,
    group_column: str = "Attack Type",
    order_column: str = "Seq",
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> list[EvaluationResult]:
    """Per-attack-type chronological split (no GLOBAL session/TEID/timestamp
    identifier exists in this data -- `Seq`/`RunTime` both reset per source
    capture file, verified empirically; a naive global sort by either drops
    most attack types from the test set entirely, see
    `dataset.chronological_split`'s docstring), train RF + XGBoost, evaluate
    once (arms A1/A2 have no confidence-tier concept)."""
    train_df, test_df = per_group_chronological_split(
        df, group_column=group_column, order_column=order_column, test_fraction=test_fraction
    )
    x_train, y_train = to_arm_a_matrix(train_df)
    x_test, y_test = to_arm_a_matrix(test_df)

    results: list[EvaluationResult] = []
    candidates: list[tuple[str, RandomForestModel | XGBoostModel]] = [
        ("RandomForest", RandomForestModel()),
        ("XGBoost", XGBoostModel()),
    ]
    for model_name, model in candidates:
        t0 = time.perf_counter()
        model.fit(x_train, y_train)
        fit_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_pred = model.predict(x_test)
        infer_s = time.perf_counter() - t0
        y_proba = model.predict_proba(x_test)

        results.append(_make_result(model_name, arm, y_test, y_pred, y_proba, fit_s, infer_s))
    return results


def train_and_evaluate_arm_b(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[list[EvaluationResult], dict[str, RandomForestModel | XGBoostModel]]:
    """Trains on `train_df` (from `dataset.build_gtp_session_dataset`),
    evaluates on `test_df` under both view A (full) and view B
    (HIGH/MEDIUM-confidence only). Returns the fitted models too, so
    `evaluate_arm_c` can be scored on the identical test_df for a like-for-
    like comparison."""
    x_train, y_train = to_gtp_matrix(train_df)
    x_test, y_test = to_gtp_matrix(test_df)
    corroborated_mask = test_df["label_confidence"].isin(["HIGH", "MEDIUM"])

    results: list[EvaluationResult] = []
    models: dict[str, RandomForestModel | XGBoostModel] = {}
    candidates: list[tuple[str, RandomForestModel | XGBoostModel]] = [
        ("RandomForest", RandomForestModel()),
        ("XGBoost", XGBoostModel()),
    ]
    for model_name, model in candidates:
        t0 = time.perf_counter()
        model.fit(x_train, y_train)
        fit_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_pred = model.predict(x_test)
        infer_s = time.perf_counter() - t0
        y_proba = model.predict_proba(x_test)

        results.append(
            _make_result(
                f"{model_name} (view A)", "B_gtp_ml", y_test, y_pred, y_proba, fit_s, infer_s
            )
        )
        results.append(
            _make_result(
                f"{model_name} (view B)",
                "B_gtp_ml",
                y_test[corroborated_mask],
                y_pred[corroborated_mask.to_numpy()],
                y_proba[corroborated_mask.to_numpy()],
                fit_s,
                infer_s,
            )
        )
        models[model_name] = model
    return results, models


def _match_feature(
    session: PDUSessionRecord, feats_by_teid: dict[int, list[TEIDFeatureRecord]]
) -> TEIDFeatureRecord | None:
    """Pairs a session with the TEID feature instance for the same TEID
    whose window contains the session's window (or, failing that, the
    temporally closest one) -- TEID features (idle-gap-split instances)
    and PDU sessions (fixed-window slices) don't share a key."""
    candidates = feats_by_teid.get(session.teid, [])
    if not candidates:
        return None
    contained = [
        f
        for f in candidates
        if f.window_start <= session.start_time and session.end_time <= f.window_end
    ]
    if contained:
        return min(contained, key=lambda f: f.window_end - f.window_start)
    mid = (session.start_time + session.end_time) / 2
    return min(candidates, key=lambda f: abs((f.window_start + f.window_end) / 2 - mid))


def evaluate_arm_c(
    test_sessions_by_attack_type: dict[str, list[PDUSessionRecord]],
    features_by_attack_type: dict[str, list[TEIDFeatureRecord]],
    thresholds: ThresholdsConfig,
) -> list[EvaluationResult]:
    """Runs the existing (Phase 5B-calibrated, not trained) agentic
    pipeline on the SAME test sessions arm B was evaluated on -- a like-
    for-like B-vs-C comparison. The state machine (`PDUSessionAgent.
    annotate_series`) only sees the test-period sessions for each
    (ue_ip, teid), not the full history including the training period, a
    known limitation of scoring against a held-out split rather than a
    live stream (noted in experiment_plan.md, not hidden)."""
    teid_agent = TEIDAgent(thresholds.teid_agent)
    pdu_agent = PDUSessionAgent(thresholds.pdu_session_agent)
    supervisor = SupervisorAgent(thresholds.supervisor_agent)

    y_true: list[bool] = []
    y_pred: list[bool] = []
    y_proba: list[float] = []
    confidences: list[str | None] = []

    t0 = time.perf_counter()
    for attack_type, sessions in test_sessions_by_attack_type.items():
        features = features_by_attack_type.get(attack_type, [])
        feats_by_teid: dict[int, list[TEIDFeatureRecord]] = defaultdict(list)
        for f in features:
            feats_by_teid[f.teid].append(f)

        sessions_by_key: dict[tuple[str, int], list[PDUSessionRecord]] = defaultdict(list)
        for s in sessions:
            sessions_by_key[(s.ue_ip, s.teid)].append(s)
        annotated: list[PDUSessionRecord] = []
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
            y_true.append(bool(session.is_attack))
            y_pred.append(sup.final_label == "Attack")
            y_proba.append(sup.fused_risk_score)
            confidences.append(session.label_confidence.value if session.label_confidence else None)
    infer_s = time.perf_counter() - t0

    y_true_s = pd.Series(y_true)
    y_pred_s = pd.Series(y_pred)
    y_proba_s = pd.Series(y_proba)
    mask = pd.Series([c in ("HIGH", "MEDIUM") for c in confidences])

    model_name = "Agentic (TEID+PDU+Supervisor)"
    return [
        _make_result(
            f"{model_name} (view A)", "C_agentic", y_true_s, y_pred_s, y_proba_s, 0.0, infer_s
        ),
        _make_result(
            f"{model_name} (view B)",
            "C_agentic",
            y_true_s[mask],
            y_pred_s[mask],
            y_proba_s[mask],
            0.0,
            infer_s,
        ),
    ]
