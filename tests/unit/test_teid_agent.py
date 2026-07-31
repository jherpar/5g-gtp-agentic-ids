from __future__ import annotations

from agente_5g.agents.teid_agent import TEIDAgent
from agente_5g.models.agent_thresholds import (
    FloodRuleConfig,
    ScanRuleConfig,
    SynFloodRuleConfig,
    TEIDAgentConfig,
)
from tests.fixtures.feature_records import make_teid_feature

CONFIG = TEIDAgentConfig(
    flood=FloodRuleConfig(min_packets_per_s=100.0, max_teid_entropy=1.0, max_unique_dst_ports=2),
    syn_flood=SynFloodRuleConfig(min_syn_count=50, max_ack_to_syn_ratio=0.1),
    scan=ScanRuleConfig(min_unique_dst_ports=15, max_packets_per_dst_port=3.0),
)


def test_normal_traffic_scores_zero_risk_with_no_triggers():
    agent = TEIDAgent(CONFIG)
    feat = make_teid_feature(packets_per_s=2.0, syn_count=1, ack_count=1, unique_dst_ports=1)
    decision = agent.evaluate(feat)

    assert decision.risk_score == 0.0
    assert decision.rule_triggers == []
    assert "No rule triggered" in decision.reason
    assert decision.agent_name == "TEIDAgent"


def test_flood_traffic_is_detected():
    agent = TEIDAgent(CONFIG)
    feat = make_teid_feature(packets_per_s=500.0, teid_entropy=0.0, unique_dst_ports=1)
    decision = agent.evaluate(feat)

    assert decision.risk_score > 0.0
    assert "flood" in decision.rule_triggers
    assert "flood" in decision.reason


def test_syn_flood_traffic_is_detected():
    agent = TEIDAgent(CONFIG)
    feat = make_teid_feature(syn_count=200, ack_count=1)
    decision = agent.evaluate(feat)

    assert "syn_flood" in decision.rule_triggers
    assert decision.risk_score > 0.0


def test_scan_traffic_is_detected():
    agent = TEIDAgent(CONFIG)
    feat = make_teid_feature(unique_dst_ports=30, packet_count=30)
    decision = agent.evaluate(feat)

    assert "scan" in decision.rule_triggers


def test_risk_score_is_max_of_triggered_rule_intensities():
    agent = TEIDAgent(CONFIG)
    # Only flood barely triggers (low intensity); syn_flood triggers hard.
    feat = make_teid_feature(
        packets_per_s=100.0,
        teid_entropy=0.0,
        unique_dst_ports=1,
        syn_count=5000,
        ack_count=1,
    )
    decision = agent.evaluate(feat)

    assert set(decision.rule_triggers) == {"flood", "syn_flood"}
    assert decision.risk_score == 1.0  # syn_flood's intensity dominates (clipped)


def test_entity_id_and_timestamp_derived_from_feature():
    agent = TEIDAgent(CONFIG)
    feat = make_teid_feature(
        teid=777, capture_file="ICMPflood_BS1.pcapng", window_start=10.0, window_end=25.0
    )
    decision = agent.evaluate(feat)

    assert decision.entity_id == "teid:777:ICMPflood_BS1.pcapng:10.0"
    assert decision.timestamp == 25.0
