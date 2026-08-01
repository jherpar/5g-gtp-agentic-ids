from __future__ import annotations

from agente_5g.evaluation.compare import (
    pr_curve_data,
    roc_curve_data,
    threshold_sensitivity,
)


def test_roc_curve_data_perfect_separation_reaches_top_left_corner():
    y_true = [False, False, True, True]
    y_proba = [0.1, 0.2, 0.8, 0.9]

    curve = roc_curve_data(y_true, y_proba)

    assert 0.0 in curve["fpr"]
    assert 1.0 in curve["tpr"]
    # perfect separation: some point has fpr=0 and tpr=1 simultaneously
    assert any(f == 0.0 and t == 1.0 for f, t in zip(curve["fpr"], curve["tpr"], strict=True))


def test_pr_curve_data_shapes_match_sklearn_convention():
    y_true = [False, False, True, True]
    y_proba = [0.1, 0.2, 0.8, 0.9]

    curve = pr_curve_data(y_true, y_proba)

    assert len(curve["precision"]) == len(curve["recall"])
    assert len(curve["thresholds"]) == len(curve["precision"]) - 1
    assert curve["precision"][-1] == 1.0
    assert curve["recall"][-1] == 0.0


def test_threshold_sensitivity_default_sweep_has_nine_points():
    y_true = [False, False, True, True]
    y_proba = [0.1, 0.4, 0.6, 0.9]

    rows = threshold_sensitivity(y_true, y_proba)

    assert len(rows) == 9
    assert [round(r["threshold"], 1) for r in rows] == [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
    ]


def test_threshold_sensitivity_recall_decreases_as_threshold_rises():
    y_true = [False, False, True, True]
    y_proba = [0.1, 0.4, 0.6, 0.9]

    rows = threshold_sensitivity(y_true, y_proba, thresholds=[0.2, 0.5, 0.8])

    # threshold=0.2: everything >= 0.2 predicted positive -> both attacks caught
    # threshold=0.8: only 0.9 predicted positive -> only one attack caught
    assert rows[0]["recall"] >= rows[-1]["recall"]


def test_threshold_sensitivity_custom_thresholds_used_verbatim():
    y_true = [False, True]
    y_proba = [0.3, 0.7]

    rows = threshold_sensitivity(y_true, y_proba, thresholds=[0.5])

    assert len(rows) == 1
    assert rows[0]["threshold"] == 0.5
    assert rows[0]["precision"] == 1.0
    assert rows[0]["recall"] == 1.0
