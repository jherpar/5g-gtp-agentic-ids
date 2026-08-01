from __future__ import annotations

from agente_5g.evaluation.metrics import compute_metrics


def test_compute_metrics_perfect_classification():
    y_true = [False, False, True, True]
    y_pred = [False, False, True, True]

    result = compute_metrics(y_true, y_pred)

    assert result["accuracy"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert result["fpr"] == 0.0
    assert result["confusion_matrix"] == [[2, 0], [0, 2]]


def test_compute_metrics_all_false_negatives():
    y_true = [True, True, False, False]
    y_pred = [False, False, False, False]

    result = compute_metrics(y_true, y_pred)

    assert result["recall"] == 0.0
    assert result["precision"] == 0.0  # zero_division=0, no positive predictions at all
    assert result["fpr"] == 0.0
    assert result["confusion_matrix"] == [[2, 0], [2, 0]]


def test_compute_metrics_false_positive_rate():
    y_true = [False, False, False, False, True]
    y_pred = [True, True, False, False, True]

    result = compute_metrics(y_true, y_pred)

    # 2 false positives out of 4 true negatives -> FPR 0.5
    assert result["fpr"] == 0.5


def test_compute_metrics_roc_auc_none_without_proba():
    result = compute_metrics([False, True], [False, True])
    assert result["roc_auc"] is None


def test_compute_metrics_roc_auc_none_for_single_class_ground_truth():
    result = compute_metrics([False, False, False], [False, False, True], y_proba=[0.1, 0.2, 0.6])
    assert result["roc_auc"] is None


def test_compute_metrics_roc_auc_computed_when_both_classes_present():
    y_true = [False, False, True, True]
    y_proba = [0.1, 0.2, 0.8, 0.9]

    result = compute_metrics(y_true, [False, False, True, True], y_proba=y_proba)

    assert result["roc_auc"] == 1.0
