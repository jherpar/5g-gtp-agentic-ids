"""Charting/formatting helpers for the labeling ground-truth validation report.

This is deliberately separate from `preprocessing/labeling.py` (which
computes labels) -- this module only summarizes and visualizes already-
labeled records, for `scripts/validate_labeling.py` to assemble into a
report citable as methodology evidence in the thesis.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

import plotly.graph_objects as go

CONFIDENCE_ORDER = ["HIGH", "MEDIUM", "LOW"]
CONFIDENCE_COLORS = {"HIGH": "#2E7D32", "MEDIUM": "#F9A825", "LOW": "#C62828"}


def confidence_counts(records: Iterable[Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for r in records:
        if r.label_confidence is not None:
            counts[r.label_confidence.value] += 1
    return counts


def attack_counts(records: Iterable[Any]) -> Counter[bool]:
    counts: Counter[bool] = Counter()
    for r in records:
        counts[bool(r.is_attack)] += 1
    return counts


def confidence_bar_figure(counts: Counter[str], title: str) -> go.Figure:
    values = [counts.get(tier, 0) for tier in CONFIDENCE_ORDER]
    total = sum(values) or 1
    percentages = [100 * v / total for v in values]
    fig = go.Figure(
        go.Bar(
            x=CONFIDENCE_ORDER,
            y=values,
            marker_color=[CONFIDENCE_COLORS[t] for t in CONFIDENCE_ORDER],
            text=[f"{v} ({p:.1f}%)" for v, p in zip(values, percentages, strict=True)],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Label confidence",
        yaxis_title="Count",
        template="plotly_white",
        showlegend=False,
        margin={"t": 60, "b": 40},
    )
    return fig


def temporal_distribution_figure(
    instances: list[tuple[float, bool, str | None]],
    file_first_ts: float,
    title: str,
    bin_width_s: float = 60.0,
) -> go.Figure:
    """`instances` is a list of (window_start, is_attack, confidence_value)."""
    fig = go.Figure()
    for tier in [*CONFIDENCE_ORDER, None]:
        xs = [
            (ts - file_first_ts) / 60.0
            for ts, is_attack, conf in instances
            if is_attack and conf == tier
        ]
        if not xs:
            continue
        label = f"Attack ({tier})" if tier else "Attack (unlabeled)"
        color = CONFIDENCE_COLORS.get(tier, "#9E9E9E") if tier is not None else "#9E9E9E"
        fig.add_trace(
            go.Histogram(
                x=xs,
                name=label,
                marker_color=color,
                xbins={"size": bin_width_s / 60.0},
                opacity=0.85,
            )
        )
    benign_xs = [(ts - file_first_ts) / 60.0 for ts, is_attack, _ in instances if not is_attack]
    if benign_xs:
        fig.add_trace(
            go.Histogram(
                x=benign_xs,
                name="Benign",
                marker_color="#90A4AE",
                xbins={"size": bin_width_s / 60.0},
                opacity=0.6,
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Minutes since file start",
        yaxis_title="TEID instances",
        barmode="stack",
        template="plotly_white",
        legend_title="Label",
        margin={"t": 60, "b": 40},
    )
    return fig


def destination_distribution_figure(
    ip_counts: Counter[str], title: str, victim_ip: str | None = None, top_n: int = 10
) -> go.Figure:
    top = ip_counts.most_common(top_n)
    ips = [ip for ip, _ in top]
    values = [v for _, v in top]
    colors = ["#C62828" if ip == victim_ip else "#1565C0" for ip in ips]
    fig = go.Figure(go.Bar(x=values, y=ips, orientation="h", marker_color=colors))
    fig.update_layout(
        title=title,
        xaxis_title="Packets",
        yaxis_title="Destination IP",
        template="plotly_white",
        margin={"t": 60, "b": 40, "l": 140},
        yaxis={"autorange": "reversed"},
    )
    return fig
