from __future__ import annotations

import math

from agente_5g.agents.rules import (
    flood_rule,
    high_diversity_rule,
    high_state_transition_rule,
    low_temporal_entropy_rule,
    scan_rule,
    syn_flood_rule,
)
from agente_5g.models.agent_thresholds import FloodRuleConfig, ScanRuleConfig, SynFloodRuleConfig
from tests.fixtures.feature_records import make_teid_feature

FLOOD_CFG = FloodRuleConfig(min_packets_per_s=100.0, max_teid_entropy=1.0, max_unique_dst_ports=2)
SYN_FLOOD_CFG = SynFloodRuleConfig(min_syn_count=50, max_ack_to_syn_ratio=0.1)
SCAN_CFG = ScanRuleConfig(min_unique_dst_ports=15, max_packets_per_dst_port=3.0)


def test_flood_rule_triggers_on_high_rate_uniform_concentrated_traffic():
    feat = make_teid_feature(packets_per_s=500.0, teid_entropy=0.0, unique_dst_ports=1)
    result = flood_rule(feat, FLOOD_CFG)
    assert result.triggered is True
    assert 0.0 < result.intensity <= 1.0


def test_flood_rule_does_not_trigger_on_low_rate_traffic():
    feat = make_teid_feature(packets_per_s=2.0, teid_entropy=0.0, unique_dst_ports=1)
    result = flood_rule(feat, FLOOD_CFG)
    assert result.triggered is False
    assert result.intensity == 0.0


def test_flood_rule_does_not_trigger_when_traffic_diverse():
    # high rate but many distinct ports and varied sizes -> not flood-like
    feat = make_teid_feature(packets_per_s=500.0, teid_entropy=4.0, unique_dst_ports=50)
    result = flood_rule(feat, FLOOD_CFG)
    assert result.triggered is False


def test_flood_rule_intensity_scales_with_rate():
    low = flood_rule(
        make_teid_feature(packets_per_s=100.0, teid_entropy=0.0, unique_dst_ports=1), FLOOD_CFG
    )
    high = flood_rule(
        make_teid_feature(packets_per_s=1000.0, teid_entropy=0.0, unique_dst_ports=1), FLOOD_CFG
    )
    assert low.triggered and high.triggered
    assert high.intensity > low.intensity
    assert high.intensity == 1.0  # clipped


def test_flood_rule_intensity_is_not_saturated_right_at_the_threshold():
    # Regression test: a naive `value / threshold` clips to exactly 1.0 the
    # instant a rule triggers, since triggering requires value >= threshold
    # -- discarding the whole point of a continuous intensity. A
    # just-barely-triggering case must read below the maximum.
    borderline = flood_rule(
        make_teid_feature(packets_per_s=100.0, teid_entropy=0.0, unique_dst_ports=1), FLOOD_CFG
    )
    assert borderline.triggered is True
    assert borderline.intensity < 1.0


def test_syn_flood_rule_triggers_on_high_syn_low_ack():
    feat = make_teid_feature(syn_count=200, ack_count=2)
    result = syn_flood_rule(feat, SYN_FLOOD_CFG)
    assert result.triggered is True


def test_syn_flood_rule_does_not_trigger_on_completed_handshakes():
    feat = make_teid_feature(syn_count=200, ack_count=195)  # ~1.0 ratio, normal
    result = syn_flood_rule(feat, SYN_FLOOD_CFG)
    assert result.triggered is False


def test_syn_flood_rule_does_not_trigger_below_min_syn_count():
    feat = make_teid_feature(syn_count=5, ack_count=0)
    result = syn_flood_rule(feat, SYN_FLOOD_CFG)
    assert result.triggered is False


def test_syn_flood_rule_handles_zero_syn_count_without_error():
    feat = make_teid_feature(syn_count=0, ack_count=0)
    result = syn_flood_rule(feat, SYN_FLOOD_CFG)
    assert result.triggered is False
    assert result.intensity == 0.0


def test_scan_rule_triggers_on_many_ports_few_packets_each():
    feat = make_teid_feature(unique_dst_ports=30, packet_count=30)  # 1 packet/port
    result = scan_rule(feat, SCAN_CFG)
    assert result.triggered is True


def test_scan_rule_does_not_trigger_on_sustained_conversation():
    feat = make_teid_feature(unique_dst_ports=1, packet_count=500)  # 500 packets/port
    result = scan_rule(feat, SCAN_CFG)
    assert result.triggered is False


def test_scan_rule_does_not_trigger_below_min_ports():
    feat = make_teid_feature(unique_dst_ports=3, packet_count=3)
    result = scan_rule(feat, SCAN_CFG)
    assert result.triggered is False


def test_scan_rule_handles_zero_unique_ports_without_error():
    feat = make_teid_feature(unique_dst_ports=0, packet_count=0)
    result = scan_rule(feat, SCAN_CFG)
    assert result.triggered is False
    assert result.intensity == 0.0


def test_high_state_transition_rule_triggers_above_threshold():
    result = high_state_transition_rule(state_transition_rate=2.0, threshold=0.5)
    assert result.triggered is True
    assert result.intensity == 1.0  # clipped


def test_high_state_transition_rule_does_not_trigger_below_threshold():
    result = high_state_transition_rule(state_transition_rate=0.1, threshold=0.5)
    assert result.triggered is False


def test_low_temporal_entropy_rule_triggers_below_threshold():
    result = low_temporal_entropy_rule(temporal_entropy=0.0, threshold=0.5)
    assert result.triggered is True
    assert result.intensity == 1.0


def test_low_temporal_entropy_rule_does_not_trigger_above_threshold():
    result = low_temporal_entropy_rule(temporal_entropy=3.0, threshold=0.5)
    assert result.triggered is False
    assert result.intensity == 0.0


def test_low_temporal_entropy_rule_partial_intensity():
    result = low_temporal_entropy_rule(temporal_entropy=0.25, threshold=0.5)
    assert result.triggered is True
    assert math.isclose(result.intensity, 0.5)


def test_high_diversity_rule_triggers_on_port_diversity():
    result = high_diversity_rule(port_diversity=50, destination_diversity=1, threshold=15)
    assert result.triggered is True


def test_high_diversity_rule_triggers_on_destination_diversity():
    result = high_diversity_rule(port_diversity=1, destination_diversity=50, threshold=15)
    assert result.triggered is True


def test_high_diversity_rule_does_not_trigger_below_threshold():
    result = high_diversity_rule(port_diversity=2, destination_diversity=2, threshold=15)
    assert result.triggered is False
