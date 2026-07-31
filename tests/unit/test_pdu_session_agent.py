from __future__ import annotations

from agente_5g.agents.pdu_session_agent import PDUSessionAgent
from agente_5g.models.agent_thresholds import PDUSessionAgentConfig, StateMachineConfig
from tests.fixtures.feature_records import make_session

CONFIG = PDUSessionAgentConfig(
    state_machine=StateMachineConfig(normal_max=0.25, watch_max=0.5, suspicious_max=0.75),
    high_state_transition_rate=1.0,
    low_temporal_entropy=0.5,
    high_diversity=15,
)


def _normal_session(session_id: str, start_time: float) -> object:
    return make_session(
        session_id,
        start_time=start_time,
        end_time=start_time + 5.0,
        state_transition_rate=0.0,
        temporal_entropy=3.0,
        port_diversity=1,
        destination_diversity=1,
    )


def _high_risk_session(session_id: str, start_time: float) -> object:
    return make_session(
        session_id,
        start_time=start_time,
        end_time=start_time + 5.0,
        state_transition_rate=2.0,  # >= 1.0 threshold -> triggers, intensity 1.0
        temporal_entropy=0.0,  # <= 0.5 threshold -> triggers, intensity 1.0
        port_diversity=50,  # >= 15 threshold -> triggers, intensity 1.0
        destination_diversity=1,
    )


def test_single_normal_session_stays_normal():
    agent = PDUSessionAgent(CONFIG)
    (annotated,) = agent.annotate_series([_normal_session("s1", 100.0)])
    assert annotated.final_state == "NORMAL"
    assert annotated.state_sequence == ["NORMAL"]


def test_escalation_is_rate_limited_to_one_level_per_window():
    agent = PDUSessionAgent(CONFIG)
    # Even though this window's risk maps to ATTACK, starting from NORMAL
    # it can only step up to WATCH on the first observation.
    (annotated,) = agent.annotate_series([_high_risk_session("s1", 100.0)])
    assert annotated.final_state == "WATCH"


def test_sustained_high_risk_escalates_through_all_levels():
    agent = PDUSessionAgent(CONFIG)
    sessions = [_high_risk_session(f"s{i}", 100.0 + i * 5) for i in range(4)]
    annotated = agent.annotate_series(sessions)

    states = [s.final_state for s in annotated]
    assert states == ["WATCH", "SUSPICIOUS", "ATTACK", "ATTACK"]
    assert annotated[-1].state_sequence == ["WATCH", "SUSPICIOUS", "ATTACK", "ATTACK"]


def test_state_de_escalates_one_level_when_risk_subsides():
    agent = PDUSessionAgent(CONFIG)
    sessions = [_high_risk_session(f"s{i}", 100.0 + i * 5) for i in range(3)]  # -> ATTACK by s2
    sessions.append(_normal_session("s3", 115.0))
    annotated = agent.annotate_series(sessions)

    states = [s.final_state for s in annotated]
    assert states == ["WATCH", "SUSPICIOUS", "ATTACK", "SUSPICIOUS"]


def test_annotate_series_sorts_by_start_time_regardless_of_input_order():
    agent = PDUSessionAgent(CONFIG)
    sessions = [_high_risk_session(f"s{i}", 100.0 + i * 5) for i in range(4)]
    shuffled = [sessions[2], sessions[0], sessions[3], sessions[1]]

    annotated = agent.annotate_series(shuffled)

    assert [s.start_time for s in annotated] == [100.0, 105.0, 110.0, 115.0]
    assert [s.final_state for s in annotated] == ["WATCH", "SUSPICIOUS", "ATTACK", "ATTACK"]


def test_decide_on_unannotated_session_computes_state_in_isolation():
    agent = PDUSessionAgent(CONFIG)
    session = _high_risk_session("s1", 100.0)
    assert session.final_state is None  # not yet run through annotate_series

    decision = agent.decide(session)
    assert decision.risk_score == 1.0
    assert "state=ATTACK" in decision.reason  # isolated target, no rate limiting applied
    assert set(decision.rule_triggers) == {
        "high_state_transition",
        "low_temporal_entropy",
        "high_diversity",
    }


def test_decide_on_annotated_session_honors_rate_limited_final_state():
    agent = PDUSessionAgent(CONFIG)
    (annotated,) = agent.annotate_series([_high_risk_session("s1", 100.0)])
    assert annotated.final_state == "WATCH"

    decision = agent.decide(annotated)
    assert "state=WATCH" in decision.reason


def test_decide_entity_id_and_timestamp():
    agent = PDUSessionAgent(CONFIG)
    session = _normal_session("session-xyz", 100.0)
    decision = agent.decide(session)

    assert decision.entity_id == "session:session-xyz"
    assert decision.timestamp == session.end_time
    assert decision.agent_name == "PDUSessionAgent"
