"""SupervisorAgent: fuses TEIDAgent and PDUSessionAgent decisions.

Deterministic weighted average of the two agents' risk scores (weights from
`configs/thresholds.yaml`'s `supervisor_agent.fusion_weights`) -- no LLM in
the decision path. `explain.py`'s optional LLM prose (if enabled) is
attached separately, after this fusion, and never feeds back into
`fused_risk_score` or `final_label`.
"""

from __future__ import annotations

import time

from agente_5g.models.agent_decision import AgentDecision, SupervisorDecision
from agente_5g.models.agent_thresholds import SupervisorAgentConfig


class SupervisorAgent:
    def __init__(self, config: SupervisorAgentConfig) -> None:
        self.config = config

    def fuse(
        self,
        entity_id: str,
        teid_decision: AgentDecision,
        session_decision: AgentDecision | None,
        predicted_attack_type: str | None = None,
    ) -> SupervisorDecision:
        t0 = time.perf_counter()
        weights = self.config.fusion_weights

        if session_decision is not None:
            total_weight = weights.teid_agent + weights.pdu_session_agent
            fused_risk = (
                weights.teid_agent * teid_decision.risk_score
                + weights.pdu_session_agent * session_decision.risk_score
            ) / total_weight
        else:
            # No session context available (e.g. TEID couldn't be attributed
            # to a UE/session) -- fall back to the TEID-only signal.
            fused_risk = teid_decision.risk_score

        final_label = "Attack" if fused_risk >= self.config.attack_decision_threshold else "Benign"

        explanation_parts = [f"TEIDAgent: {teid_decision.reason}"]
        if session_decision is not None:
            explanation_parts.append(f"PDUSessionAgent: {session_decision.reason}")
        explanation = " | ".join(explanation_parts)

        decision_latency_ms = (time.perf_counter() - t0) * 1000

        return SupervisorDecision(
            entity_id=entity_id,
            teid_decision=teid_decision,
            session_decision=session_decision,
            fused_risk_score=fused_risk,
            final_label=final_label,
            predicted_attack_type=predicted_attack_type if final_label == "Attack" else None,
            explanation=explanation,
            llm_explanation=None,
            decision_latency_ms=decision_latency_ms,
        )
