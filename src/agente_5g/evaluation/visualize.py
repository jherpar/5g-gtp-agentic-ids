"""Plotly figure builders for the Phase 7 comparative report: ROC curves,
precision-recall curves, confusion matrices, and threshold-sensitivity
plots across all four arms. Functions return `go.Figure` objects (never
write files themselves) -- callers decide where/whether to
`fig.write_image(...)`, matching `evaluation/label_validation.py`'s
convention.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

_PALETTE = ["#1565C0", "#2E7D32", "#EF6C00", "#6A1B9A", "#C62828", "#00838F", "#F9A825"]


def plot_roc_curves(curves: dict[str, dict[str, list[float]]], title: str) -> go.Figure:
    """`curves`: {series_name: {"fpr": [...], "tpr": [...]}} (extra keys
    ignored, e.g. "thresholds" from `evaluation.compare.roc_curve_data`)."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"dash": "dash", "color": "#9E9E9E"},
            name="Random (AUC=0.5)",
        )
    )
    for i, (name, curve) in enumerate(curves.items()):
        fig.add_trace(
            go.Scatter(
                x=curve["fpr"],
                y=curve["tpr"],
                mode="lines",
                name=name,
                line={"color": _PALETTE[i % len(_PALETTE)]},
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        margin={"t": 60, "b": 40},
    )
    return fig


def plot_pr_curves(curves: dict[str, dict[str, list[float]]], title: str) -> go.Figure:
    """`curves`: {series_name: {"precision": [...], "recall": [...]}}."""
    fig = go.Figure()
    for i, (name, curve) in enumerate(curves.items()):
        fig.add_trace(
            go.Scatter(
                x=curve["recall"],
                y=curve["precision"],
                mode="lines",
                name=name,
                line={"color": _PALETTE[i % len(_PALETTE)]},
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Recall",
        yaxis_title="Precision",
        template="plotly_white",
        margin={"t": 60, "b": 40},
        yaxis={"range": [0, 1.05]},
    )
    return fig


def plot_confusion_matrix(cm: list[list[int]], title: str) -> go.Figure:
    """`cm` is [[TN, FP], [FN, TP]] (sklearn's `labels=[False, True]`
    convention, matching `evaluation.metrics.compute_metrics`)."""
    labels = ["Benign", "Attack"]
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            text=cm,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        template="plotly_white",
        margin={"t": 60, "b": 40},
    )
    return fig


def plot_threshold_sensitivity(rows: list[dict[str, Any]], title: str) -> go.Figure:
    """`rows`: `evaluation.compare.threshold_sensitivity`'s output. Purely
    illustrative -- does not mark or recommend any threshold, since Phase
    6's configured thresholds are frozen."""
    fig = go.Figure()
    thresholds = [r["threshold"] for r in rows]
    for metric, color in [
        ("precision", _PALETTE[0]),
        ("recall", _PALETTE[1]),
        ("f1", _PALETTE[2]),
        ("fpr", _PALETTE[4]),
    ]:
        fig.add_trace(
            go.Scatter(
                x=thresholds,
                y=[r[metric] for r in rows],
                mode="lines+markers",
                name=metric,
                line={"color": color},
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Decision threshold (discussion only, not applied)",
        yaxis_title="Metric value",
        template="plotly_white",
        margin={"t": 60, "b": 40},
        yaxis={"range": [0, 1.05]},
    )
    return fig
