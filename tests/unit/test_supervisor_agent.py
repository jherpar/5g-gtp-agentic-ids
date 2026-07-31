from __future__ import annotations

import math

from agente_5g.agents.supervisor_agent import SupervisorAgent
from agente_5g.models.agent_decision import AgentDecision
from agente_5g.models.agent_thresholds import FusionWeightsConfig, SupervisorAgentConfig

CONFIG = SupervisorAgentConfig(
    fusion_weights=FusionWeightsConfig(teid_agent=0.6, pdu_session_agent=0.4),
    attack_decision_threshold=0.5,
)


def _decision(agent_name: str, risk_score: float, reason: str = "reason") -> AgentDecision:
    return AgentDecision(
        entity_id="x",
        agent_name=agent_name,
        risk_score=risk_score,
        reason=reason,
        rule_triggers=[],
        timestamp=10.0,
    )


def test_fuse_computes_weighted_average_of_both_decisions():
    agent = SupervisorAgent(CONFIG)
    teid_decision = _decision("TEIDAgent", risk_score=1.0)
    session_decision = _decision("PDUSessionAgent", risk_score=0.0)

    result = agent.fuse("entity-1", teid_decision, session_decision)

    assert math.isclose(result.fused_risk_score, 0.6)  # 0.6*1.0 + 0.4*0.0


def test_fuse_falls_back_to_teid_only_when_no_session_decision():
    agent = SupervisorAgent(CONFIG)
    teid_decision = _decision("TEIDAgent", risk_score=0.7)

    result = agent.fuse("entity-1", teid_decision, None)

    assert result.fused_risk_score == 0.7
    assert result.session_decision is None


def test_final_label_attack_when_fused_risk_at_or_above_threshold():
    agent = SupervisorAgent(CONFIG)
    result = agent.fuse("e", _decision("TEIDAgent", 0.5), None)
    assert result.final_label == "Attack"


def test_final_label_benign_when_fused_risk_below_threshold():
    agent = SupervisorAgent(CONFIG)
    result = agent.fuse("e", _decision("TEIDAgent", 0.49), None)
    assert result.final_label == "Benign"


def test_predicted_attack_type_only_set_when_label_is_attack():
    agent = SupervisorAgent(CONFIG)
    attack = agent.fuse("e", _decision("TEIDAgent", 0.9), None, predicted_attack_type="SYNflood")
    benign = agent.fuse("e", _decision("TEIDAgent", 0.1), None, predicted_attack_type="SYNflood")
    assert attack.predicted_attack_type == "SYNflood"
    assert benign.predicted_attack_type is None


def test_explanation_concatenates_both_agent_reasons():
    agent = SupervisorAgent(CONFIG)
    result = agent.fuse(
        "e",
        _decision("TEIDAgent", 0.8, reason="flood detected"),
        _decision("PDUSessionAgent", 0.6, reason="state=ATTACK"),
    )
    assert "TEIDAgent: flood detected" in result.explanation
    assert "PDUSessionAgent: state=ATTACK" in result.explanation


def test_explanation_omits_session_agent_when_absent():
    agent = SupervisorAgent(CONFIG)
    result = agent.fuse("e", _decision("TEIDAgent", 0.8, reason="flood detected"), None)
    assert "PDUSessionAgent" not in result.explanation


def test_decision_latency_is_nonnegative_and_llm_explanation_unset():
    agent = SupervisorAgent(CONFIG)
    result = agent.fuse("e", _decision("TEIDAgent", 0.1), None)
    assert result.decision_latency_ms >= 0.0
    assert result.llm_explanation is None
