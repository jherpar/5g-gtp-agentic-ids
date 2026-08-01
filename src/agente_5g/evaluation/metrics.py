"""Shared metric computation for every Phase 6 arm (A1/A2/B/C), so all four
are scored through the identical code path -- comparisons across arms
reflect real performance differences, not differences in how metrics were
computed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true: Any, y_pred: Any, y_proba: Any | None = None) -> dict[str, Any]:
    """Returns accuracy/precision/recall/f1/roc_auc/fpr/confusion_matrix.

    `roc_auc` is None if `y_proba` isn't supplied or `y_true` has only one
    class (undefined in that case, not zero)."""
    y_true_arr = np.asarray(y_true, dtype=bool)
    y_pred_arr = np.asarray(y_pred, dtype=bool)
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[False, True])
    tn, fp, _fn, _tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    roc_auc = None
    if y_proba is not None and len(np.unique(y_true_arr)) > 1:
        roc_auc = float(roc_auc_score(y_true_arr, y_proba))

    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        "roc_auc": roc_auc,
        "fpr": float(fpr),
        "confusion_matrix": cm.tolist(),
    }
