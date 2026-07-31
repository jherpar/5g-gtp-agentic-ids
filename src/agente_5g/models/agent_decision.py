"""Agent output records (src/agente_5g/agents/*)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class AgentDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    entity_id: str
    agent_name: str
    risk_score: float  # in [0, 1]
    reason: str
    rule_triggers: list[str]
    timestamp: float


class SupervisorDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    entity_id: str
    teid_decision: AgentDecision
    session_decision: AgentDecision | None
    fused_risk_score: float
    final_label: str  # "Benign" | "Attack"
    predicted_attack_type: str | None
    explanation: str
    llm_explanation: str | None = None
    decision_latency_ms: float
