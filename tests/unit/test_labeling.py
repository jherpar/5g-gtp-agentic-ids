from __future__ import annotations

import math

import pytest

from agente_5g.models.labels import LabelConfidence
from agente_5g.models.schedule_config import (
    AttackSchedule,
    AttackScheduleEntry,
    ConnectionFloodPatternConfig,
    FloodPatternConfig,
    LabelPatternsConfig,
    ScanPatternConfig,
    SlowratePatternConfig,
)
from agente_5g.preprocessing.labeling import (
    _approximate_attack_subwindow,
    _classify,
    _level3_pattern_matches,
    label_sessions,
    label_teid_features,
)
from agente_5g.preprocessing.session_builder import SessionBuilder
from agente_5g.preprocessing.teid_extractor import TEIDFeatureExtractor
from tests.fixtures.packet_records import make_packet

VICTIM_IP = "10.41.150.68"
ATTACKER_IP = "10.155.15.1"


@pytest.fixture
def schedule() -> AttackSchedule:
    return AttackSchedule(
        calibrated=True,
        timezone="Europe/Helsinki",
        year=2022,
        victim_ip=VICTIM_IP,
        attacker_ip_hints={},
        attacks={
            "SSH": AttackScheduleEntry(official_name="Benign", benign_only=True),
            "SYNScan": AttackScheduleEntry(
                official_name="SYN Scan",
                session_window=("12:20", "12:30"),
                attack_window={"BS1": ("12:20", "12:30"), "BS2": ("12:20", "12:30")},
            ),
            "ICMPflood": AttackScheduleEntry(
                official_name="ICMP Flood",
                session_window=("14:45", "15:15"),  # 30 min
                # rel_start = 5/30 = 0.1667, rel_end = 15/30 = 0.5
                attack_window={"BS1": ("14:50", "15:00"), "BS2": ("14:55", "15:05")},
            ),
        },
    )


@pytest.fixture
def patterns() -> LabelPatternsConfig:
    return LabelPatternsConfig(
        flood_pattern=FloodPatternConfig(
            min_sustained_packets_per_s=10.0,
            min_window_s=1.0,
            max_packet_size_entropy=2.0,
            max_unique_dst_ports=5,
        ),
        connection_flood_pattern=ConnectionFloodPatternConfig(min_port_cardinality_asymmetry=200.0),
        scan_pattern=ScanPatternConfig(min_unique_dst_ports_per_source=3, max_window_s=30.0),
        slowrate_pattern=SlowratePatternConfig(
            min_connection_duration_s=5.0, max_bytes_per_s=1000.0, min_concurrent_connections=2
        ),
    )


def test_relative_window_mapping_preserves_schedule_fraction(schedule):
    # ICMPflood BS1: rel_start=5/30, rel_end=15/30 over a 1800s file -> (300, 900)
    subwindow = _approximate_attack_subwindow(schedule, "ICMPflood", "BS1", 0.0, 1800.0)
    assert subwindow is not None
    start, end = subwindow
    assert math.isclose(start, 300.0, rel_tol=1e-9)
    assert math.isclose(end, 900.0, rel_tol=1e-9)


def test_relative_window_mapping_returns_none_for_benign_only(schedule):
    assert _approximate_attack_subwindow(schedule, "SSH", "BS1", 0.0, 100.0) is None


def test_ssh_always_benign_high_confidence(schedule, patterns):
    # Even with victim-IP-touching packets, SSH files have no attack window at all.
    packets = [
        make_packet(0, teid=1, timestamp=0.0, inner_src_ip=ATTACKER_IP, inner_dst_ip=VICTIM_IP)
    ]
    label, is_attack, confidence, evidence = _classify(
        source_attack_type="SSH",
        base_station="BS1",
        instance_start=0.0,
        instance_end=1.0,
        instance_packets=packets,
        schedule=schedule,
        patterns=patterns,
        file_first_ts=0.0,
        file_last_ts=100.0,
    )
    assert label == "Benign"
    assert is_attack is False
    assert confidence == LabelConfidence.HIGH
    assert evidence == []


def test_outside_window_no_corroboration_is_benign_high_confidence(schedule, patterns):
    # SYNScan window == full file span (0, 600); instance at (700, 700) is outside it.
    packets = [make_packet(0, teid=1, timestamp=700.0)]
    label, is_attack, confidence, evidence = _classify(
        source_attack_type="SYNScan",
        base_station="BS1",
        instance_start=700.0,
        instance_end=700.0,
        instance_packets=packets,
        schedule=schedule,
        patterns=patterns,
        file_first_ts=0.0,
        file_last_ts=600.0,
    )
    assert label == "Benign"
    assert is_attack is False
    assert confidence == LabelConfidence.HIGH


def test_outside_window_with_victim_ip_is_benign_low_confidence(schedule, patterns):
    packets = [
        make_packet(0, teid=1, timestamp=700.0, inner_src_ip=ATTACKER_IP, inner_dst_ip=VICTIM_IP)
    ]
    label, is_attack, confidence, evidence = _classify(
        source_attack_type="SYNScan",
        base_station="BS1",
        instance_start=700.0,
        instance_end=700.0,
        instance_packets=packets,
        schedule=schedule,
        patterns=patterns,
        file_first_ts=0.0,
        file_last_ts=600.0,
    )
    assert label == "Benign"
    assert is_attack is False
    assert confidence == LabelConfidence.LOW
    assert "VICTIM_IP" in evidence


def test_inside_window_no_corroboration_is_attack_low_confidence(schedule, patterns):
    # Inside SYNScan's full-file window, but no victim IP / pattern match.
    packets = [make_packet(0, teid=1, timestamp=100.0)]
    label, is_attack, confidence, evidence = _classify(
        source_attack_type="SYNScan",
        base_station="BS1",
        instance_start=100.0,
        instance_end=100.0,
        instance_packets=packets,
        schedule=schedule,
        patterns=patterns,
        file_first_ts=0.0,
        file_last_ts=600.0,
    )
    assert label == "SYNScan"
    assert is_attack is True
    assert confidence == LabelConfidence.LOW
    assert evidence == ["SCHEDULE"]


def test_inside_window_with_victim_ip_only_is_medium_confidence(schedule, patterns):
    packets = [
        make_packet(
            i, teid=1, timestamp=100.0 + i, inner_src_ip=ATTACKER_IP, inner_dst_ip=VICTIM_IP
        )
        for i in range(2)  # only 2 dst ports touched -> below scan pattern threshold of 3
    ]
    label, is_attack, confidence, evidence = _classify(
        source_attack_type="SYNScan",
        base_station="BS1",
        instance_start=100.0,
        instance_end=102.0,
        instance_packets=packets,
        schedule=schedule,
        patterns=patterns,
        file_first_ts=0.0,
        file_last_ts=600.0,
    )
    assert is_attack is True
    assert confidence == LabelConfidence.MEDIUM
    assert set(evidence) == {"SCHEDULE", "VICTIM_IP"}


def test_inside_window_with_victim_ip_and_pattern_is_high_confidence(schedule, patterns):
    # 5 distinct dst ports (>= scan threshold of 3) all touching the victim IP.
    packets = [
        make_packet(
            i,
            teid=1,
            timestamp=100.0 + i * 0.1,
            inner_src_ip=ATTACKER_IP,
            inner_dst_ip=VICTIM_IP,
            inner_dst_port=1000 + i,
        )
        for i in range(5)
    ]
    label, is_attack, confidence, evidence = _classify(
        source_attack_type="SYNScan",
        base_station="BS1",
        instance_start=100.0,
        instance_end=100.4,
        instance_packets=packets,
        schedule=schedule,
        patterns=patterns,
        file_first_ts=0.0,
        file_last_ts=600.0,
    )
    assert is_attack is True
    assert confidence == LabelConfidence.HIGH
    assert set(evidence) == {"SCHEDULE", "VICTIM_IP", "PATTERN"}


def test_unknown_attack_type_defaults_to_benign(schedule, patterns):
    label, is_attack, confidence, evidence = _classify(
        source_attack_type="SomeUnknownType",
        base_station="BS1",
        instance_start=0.0,
        instance_end=1.0,
        instance_packets=[],
        schedule=schedule,
        patterns=patterns,
        file_first_ts=0.0,
        file_last_ts=600.0,
    )
    assert label == "Benign"
    assert is_attack is False


def test_label_teid_features_end_to_end_attaches_labels(schedule, patterns):
    packets = [
        make_packet(
            i,
            teid=42,
            timestamp=100.0 + i * 0.1,
            inner_src_ip=ATTACKER_IP,
            inner_dst_ip=VICTIM_IP,
            inner_dst_port=1000 + i,
            source_attack_type="SYNScan",
        )
        for i in range(5)
    ]
    features = list(TEIDFeatureExtractor().extract(packets))
    assert features[0].label is None  # unlabeled before Phase 4 enrichment

    labeled = list(label_teid_features(features, packets, schedule, patterns))
    assert len(labeled) == 1
    assert labeled[0].is_attack is True
    assert labeled[0].label_confidence == LabelConfidence.HIGH
    assert labeled[0].label == "SYNScan"
    # original record is untouched (frozen models -> model_copy produces a new instance)
    assert features[0].label is None


def test_label_sessions_end_to_end_attaches_labels(schedule, patterns):
    packets = [
        make_packet(
            i,
            teid=42,
            timestamp=100.0 + i * 0.1,
            inner_src_ip=ATTACKER_IP,
            inner_dst_ip=VICTIM_IP,
            inner_dst_port=1000 + i,
        )
        for i in range(5)
    ]
    sessions = list(SessionBuilder(window_size_s=1).build(packets))
    assert sessions[0].label is None

    labeled = list(
        label_sessions(
            sessions,
            packets,
            source_attack_type="SYNScan",
            base_station="BS1",
            schedule=schedule,
            patterns=patterns,
        )
    )
    assert len(labeled) == 1
    assert labeled[0].is_attack is True
    assert labeled[0].label_confidence == LabelConfidence.HIGH


FLOOD_PATTERN_CFG = FloodPatternConfig(
    min_sustained_packets_per_s=0.9,
    min_window_s=5.0,
    max_packet_size_entropy=1.0,
    max_unique_dst_ports=2,
)
CONNECTION_FLOOD_PATTERN_CFG = ConnectionFloodPatternConfig(min_port_cardinality_asymmetry=200.0)
FLOOD_PATTERNS = LabelPatternsConfig(
    flood_pattern=FLOOD_PATTERN_CFG,
    connection_flood_pattern=CONNECTION_FLOOD_PATTERN_CFG,
    scan_pattern=ScanPatternConfig(min_unique_dst_ports_per_source=15, max_window_s=30.0),
    slowrate_pattern=SlowratePatternConfig(
        min_connection_duration_s=20.0, max_bytes_per_s=5.0, min_concurrent_connections=5
    ),
)


def test_flood_pattern_matches_uniform_concentrated_traffic():
    # High rate, identical packet sizes (zero entropy), single destination
    # port -- a genuine flood signature.
    packets = [
        make_packet(i, teid=1, timestamp=100.0 + i * 0.1, packet_size=64, inner_dst_port=53)
        for i in range(60)
    ]
    assert _level3_pattern_matches("ICMPflood", packets, FLOOD_PATTERNS) is True


def test_flood_pattern_does_not_match_high_rate_diverse_traffic():
    # Regression test for the bug found in confidence_diagnosis/report.md:
    # a rate-only check matched ordinary sustained conversations (up to
    # ~13,000 pkt/s in the non-victim-corroborated group) just as easily as
    # real floods. High rate + varied packet sizes + many destination ports
    # is NOT a flood signature and must not match, however high the rate.
    packets = [
        make_packet(
            i,
            teid=1,
            timestamp=100.0 + i * 0.01,
            packet_size=64 + (i % 500),  # highly varied sizes -> high entropy
            inner_dst_port=1000 + (i % 50),  # many distinct destination ports
        )
        for i in range(600)
    ]
    assert _level3_pattern_matches("ICMPflood", packets, FLOOD_PATTERNS) is False


def test_flood_pattern_does_not_match_below_rate_floor():
    packets = [make_packet(0, teid=1, timestamp=100.0, packet_size=64)]
    assert _level3_pattern_matches("ICMPflood", packets, FLOOD_PATTERNS) is False


def test_flood_pattern_does_not_match_below_min_window():
    # High rate but the whole burst lasts under min_window_s.
    packets = [
        make_packet(i, teid=1, timestamp=100.0 + i * 0.01, packet_size=64) for i in range(10)
    ]
    assert _level3_pattern_matches("ICMPflood", packets, FLOOD_PATTERNS) is False


def test_connection_flood_pattern_matches_high_port_asymmetry():
    # SYNflood/Goldeneye signature: one side (source ports) fans out into
    # the hundreds while the other (destination port) stays fixed -- the
    # real shape found in outputs/reports/connection_flood_hypothesis/report.md.
    packets = [
        make_packet(
            i, teid=1, timestamp=100.0 + i * 0.01, inner_src_port=2000 + i, inner_dst_port=80
        )
        for i in range(250)
    ]
    assert _level3_pattern_matches("SYNflood", packets, FLOOD_PATTERNS) is True
    assert _level3_pattern_matches("Goldeneye", packets, FLOOD_PATTERNS) is True


def test_connection_flood_pattern_does_not_match_balanced_port_counts():
    # Regression test for the bug found in
    # outputs/reports/evidence_quantification/report.md: SYNflood's true
    # positives had ZERO entropy/port-concentration pattern matches under
    # the old shared flood_pattern rule. Ordinary traffic with a handful of
    # ports on both sides (no asymmetry) must not match either.
    packets = [
        make_packet(
            i,
            teid=1,
            timestamp=100.0 + i * 0.1,
            inner_src_port=51000 + (i % 3),
            inner_dst_port=80 + (i % 3),
        )
        for i in range(20)
    ]
    assert _level3_pattern_matches("SYNflood", packets, FLOOD_PATTERNS) is False
    assert _level3_pattern_matches("Goldeneye", packets, FLOOD_PATTERNS) is False
