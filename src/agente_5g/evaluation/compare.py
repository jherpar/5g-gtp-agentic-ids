"""ROC/PR curve data and threshold-sensitivity analysis, shared across all
four Phase 6 arms.

`threshold_sensitivity` is diagnostic/discussion material ONLY -- Phase 6's
configured decision thresholds (`configs/thresholds.yaml`'s
`attack_decision_threshold`, the RF/XGBoost default 0.5 cutoff) are frozen
per explicit instruction; this module never selects or applies a new
threshold, it only reports how precision/recall/F1/FPR would look across a
sweep, for the thesis discussion of the precision/recall trade-off.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from agente_5g.evaluation.metrics import compute_metrics


def roc_curve_data(y_true: Any, y_proba: Any) -> dict[str, list[float]]:
    """Returns {"fpr": [...], "tpr": [...], "thresholds": [...]} for
    plotting a ROC curve."""
    fpr, tpr, thresholds = roc_curve(np.asarray(y_true, dtype=bool), np.asarray(y_proba))
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()}


def pr_curve_data(y_true: Any, y_proba: Any) -> dict[str, list[float]]:
    """Returns {"precision": [...], "recall": [...], "thresholds": [...]}
    for plotting a precision-recall curve. `thresholds` is one shorter than
    `precision`/`recall` (sklearn convention: the last precision/recall
    point has no corresponding threshold)."""
    precision, recall, thresholds = precision_recall_curve(
        np.asarray(y_true, dtype=bool), np.asarray(y_proba)
    )
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
    }


def threshold_sensitivity(
    y_true: Any, y_proba: Any, thresholds: list[float] | None = None
) -> list[dict[str, float]]:
    """Metrics at a sweep of candidate decision thresholds, for discussion
    of the precision/recall trade-off -- NOT used to pick or change any
    configured threshold. Default sweep: 0.1 to 0.9 in steps of 0.1."""
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.1, 1.0, 0.1)]
    y_true_arr = np.asarray(y_true, dtype=bool)
    y_proba_arr = np.asarray(y_proba)

    rows = []
    for t in thresholds:
        y_pred = y_proba_arr >= t
        metrics = compute_metrics(y_true_arr, y_pred, y_proba_arr)
        rows.append(
            {
                "threshold": t,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "fpr": metrics["fpr"],
            }
        )
    return rows
