from __future__ import annotations

import plotly.graph_objects as go

from agente_5g.evaluation.visualize import (
    plot_confusion_matrix,
    plot_pr_curves,
    plot_roc_curves,
    plot_threshold_sensitivity,
)


def test_plot_roc_curves_returns_figure_with_diagonal_and_series():
    curves = {"RandomForest": {"fpr": [0.0, 0.1, 1.0], "tpr": [0.0, 0.8, 1.0]}}

    fig = plot_roc_curves(curves, "Test ROC")

    assert isinstance(fig, go.Figure)
    # diagonal reference line + 1 series = 2 traces
    assert len(fig.data) == 2
    assert fig.data[1].name == "RandomForest"


def test_plot_roc_curves_supports_multiple_series():
    curves = {
        "RandomForest": {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]},
        "XGBoost": {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]},
    }

    fig = plot_roc_curves(curves, "Test ROC")

    names = {trace.name for trace in fig.data}
    assert {"RandomForest", "XGBoost"}.issubset(names)


def test_plot_pr_curves_returns_figure():
    curves = {"RandomForest": {"precision": [1.0, 0.5], "recall": [0.0, 1.0]}}

    fig = plot_pr_curves(curves, "Test PR")

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert list(fig.data[0].x) == [0.0, 1.0]  # recall on x-axis
    assert list(fig.data[0].y) == [1.0, 0.5]  # precision on y-axis


def test_plot_confusion_matrix_returns_heatmap():
    fig = plot_confusion_matrix([[10, 2], [3, 5]], "Test CM")

    assert isinstance(fig, go.Figure)
    assert isinstance(fig.data[0], go.Heatmap)


def test_plot_threshold_sensitivity_includes_all_four_metrics():
    rows = [
        {"threshold": 0.3, "precision": 0.5, "recall": 0.9, "f1": 0.6, "fpr": 0.4},
        {"threshold": 0.7, "precision": 0.8, "recall": 0.5, "f1": 0.6, "fpr": 0.1},
    ]

    fig = plot_threshold_sensitivity(rows, "Test sensitivity")

    names = {trace.name for trace in fig.data}
    assert names == {"precision", "recall", "f1", "fpr"}
