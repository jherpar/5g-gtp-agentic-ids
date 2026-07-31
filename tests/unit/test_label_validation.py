from __future__ import annotations

import plotly.graph_objects as go

from agente_5g.evaluation.label_validation import (
    attack_counts,
    confidence_bar_figure,
    confidence_counts,
    destination_distribution_figure,
    temporal_distribution_figure,
)
from agente_5g.models.labels import LabelConfidence
from tests.fixtures.feature_records import make_teid_feature


def _with_label(feat, is_attack: bool, confidence: LabelConfidence):
    return feat.model_copy(
        update={"is_attack": is_attack, "label_confidence": confidence, "label": "x"}
    )


def test_confidence_counts_tallies_by_tier():
    records = [
        _with_label(make_teid_feature(1), True, LabelConfidence.HIGH),
        _with_label(make_teid_feature(2), True, LabelConfidence.HIGH),
        _with_label(make_teid_feature(3), False, LabelConfidence.LOW),
    ]
    counts = confidence_counts(records)
    assert counts["HIGH"] == 2
    assert counts["LOW"] == 1
    assert counts["MEDIUM"] == 0


def test_confidence_counts_skips_unlabeled_records():
    records = [make_teid_feature(1)]  # label_confidence is None by default
    assert confidence_counts(records) == {}


def test_attack_counts_tallies_bool():
    records = [
        _with_label(make_teid_feature(1), True, LabelConfidence.HIGH),
        _with_label(make_teid_feature(2), False, LabelConfidence.HIGH),
    ]
    counts = attack_counts(records)
    assert counts[True] == 1
    assert counts[False] == 1


def test_confidence_bar_figure_returns_plotly_figure():
    fig = confidence_bar_figure({"HIGH": 5, "MEDIUM": 2, "LOW": 1}, "Test title")
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Test title"


def test_temporal_distribution_figure_handles_mixed_confidence():
    instances = [
        (100.0, True, "HIGH"),
        (160.0, True, "LOW"),
        (220.0, False, None),
    ]
    fig = temporal_distribution_figure(instances, file_first_ts=100.0, title="Timeline")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_temporal_distribution_figure_handles_no_instances():
    fig = temporal_distribution_figure([], file_first_ts=0.0, title="Empty")
    assert isinstance(fig, go.Figure)


def test_destination_distribution_figure_highlights_victim_ip():
    from collections import Counter

    counts = Counter({"10.41.150.68": 100, "10.155.15.1": 90, "8.8.8.8": 5})
    fig = destination_distribution_figure(counts, "Victims", victim_ip="10.41.150.68")
    assert isinstance(fig, go.Figure)
    colors = fig.data[0].marker.color
    assert "#C62828" in colors  # victim highlighted in red
