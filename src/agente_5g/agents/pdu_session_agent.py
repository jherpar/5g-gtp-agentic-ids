"""PDUSessionAgent: temporal reasoning over a UE+TEID's session history.

Deterministic and rule-based (see `agents/rules.py`), same reproducibility
guarantee as TEIDAgent. Where TEIDAgent scores a single instance in
isolation, PDUSessionAgent's state machine (NORMAL -> WATCH -> SUSPICIOUS ->
ATTACK) reasons over a *chronologically-ordered sequence* of session windows
for the same (ue_ip, teid), rate-limiting transitions to at most one level
per observed window -- a single noisy window can't jump straight from
NORMAL to ATTACK; sustained elevated risk across consecutive windows is
required to escalate all the way, and the state also de-escalates one level
at a time once risk subsides.
"""

from __future__ import annotations

from agente_5g.agents.rules import (
    RuleResult,
    high_diversity_rule,
    high_state_transition_rule,
    low_temporal_entropy_rule,
)
from agente_5g.models.agent_decision import AgentDecision
from agente_5g.models.agent_thresholds import PDUSessionAgentConfig
from agente_5g.models.session import PDUSessionRecord

STATE_ORDER = ["NORMAL", "WATCH", "SUSPICIOUS", "ATTACK"]


class PDUSessionAgent:
    def __init__(self, config: PDUSessionAgentConfig) -> None:
        self.config = config

    def _risk_and_rules(self, session: PDUSessionRecord) -> tuple[float, dict[str, RuleResult]]:
        results = {
            "high_state_transition": high_state_transition_rule(
                session.state_transition_rate, self.config.high_state_transition_rate
            ),
            "low_temporal_entropy": low_temporal_entropy_rule(
                session.temporal_entropy, self.config.low_temporal_entropy
            ),
            "high_diversity": high_diversity_rule(
                session.port_diversity, session.destination_diversity, self.config.high_diversity
            ),
        }
        triggered = {name: r for name, r in results.items() if r.triggered}
        risk = max((r.intensity for r in triggered.values()), default=0.0)
        return risk, triggered

    def _target_state(self, risk: float) -> str:
        sm = self.config.state_machine
        if risk <= sm.normal_max:
            return "NORMAL"
        if risk <= sm.watch_max:
            return "WATCH"
        if risk <= sm.suspicious_max:
            return "SUSPICIOUS"
        return "ATTACK"

    def annotate_series(self, sessions: list[PDUSessionRecord]) -> list[PDUSessionRecord]:
        """Run the state machine over sessions for the SAME (ue_ip, teid),
        in whatever order they're given -- they are re-sorted by
        `start_time` internally. Returns new records (frozen model) with
        `state_sequence`/`final_state` populated."""
        ordered = sorted(sessions, key=lambda s: s.start_time)
        state_sequence: list[str] = []
        current_index = 0

        annotated: list[PDUSessionRecord] = []
        for session in ordered:
            risk, _ = self._risk_and_rules(session)
            target_index = STATE_ORDER.index(self._target_state(risk))
            if target_index > current_index:
                current_index += 1
            elif target_index < current_index:
                current_index -= 1
            state_sequence.append(STATE_ORDER[current_index])
            annotated.append(
                session.model_copy(
                    update={
                        "state_sequence": list(state_sequence),
                        "final_state": STATE_ORDER[current_index],
                    }
                )
            )
        return annotated

    def decide(self, session: PDUSessionRecord) -> AgentDecision:
        """Produce a fusable AgentDecision for a single session. Uses the
        session's own `final_state` if already annotated (via
        `annotate_series`); otherwise computes the target state for this
        window in isolation (useful for one-off scoring/tests)."""
        risk, triggered = self._risk_and_rules(session)
        state = session.final_state or self._target_state(risk)

        if triggered:
            reason = "; ".join(f"{name} ({r.detail})" for name, r in triggered.items())
        else:
            reason = "No rule triggered; session traffic within normal thresholds."

        return AgentDecision(
            entity_id=f"session:{session.session_id}",
            agent_name="PDUSessionAgent",
            risk_score=risk,
            reason=f"state={state}; {reason}",
            rule_triggers=list(triggered.keys()),
            timestamp=session.end_time,
        )
