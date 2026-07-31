from __future__ import annotations

from agente_5g.models.evaluation import EvaluationResult


def test_evaluation_result_round_trips_through_json():
    result = EvaluationResult(
        model_name="RandomForest",
        arm="B_gtp_ml",
        accuracy=0.95,
        precision=0.9,
        recall=0.92,
        f1=0.91,
        roc_auc=0.97,
        fpr=0.02,
        detection_time_ms=1.5,
        inference_time_ms=0.3,
        confusion_matrix=[[100, 5], [3, 92]],
        run_id="run-1",
        config_hash="abc123",
    )

    dumped = result.model_dump(mode="json")
    restored = EvaluationResult.model_validate(dumped)

    assert restored == result
    assert restored.arm == "B_gtp_ml"


def test_evaluation_result_allows_none_roc_auc():
    result = EvaluationResult(
        model_name="TEIDAgent",
        arm="C_agentic",
        accuracy=0.9,
        precision=0.85,
        recall=0.88,
        f1=0.86,
        roc_auc=None,
        fpr=0.05,
        detection_time_ms=0.5,
        inference_time_ms=0.1,
        confusion_matrix=[[10, 1], [2, 9]],
        run_id="run-2",
        config_hash="def456",
    )
    assert result.roc_auc is None
