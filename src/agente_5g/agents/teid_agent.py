"""TEIDAgent: local anomaly detection over a single TEID instance's features.

Deterministic and rule-based (see `agents/rules.py`) -- no LLM in the
decision path, so results are reproducible given the same features and
`configs/thresholds.yaml`. Detects flood-like traffic (ICMP/UDP/generic),
SYN floods, and port scans.
"""

from __future__ import annotations

from agente_5g.agents.rules import RuleResult, flood_rule, scan_rule, syn_flood_rule
from agente_5g.models.agent_decision import AgentDecision
from agente_5g.models.agent_thresholds import TEIDAgentConfig
from agente_5g.models.teid_features import TEIDFeatureRecord


class TEIDAgent:
    def __init__(self, config: TEIDAgentConfig) -> None:
        self.config = config

    def evaluate(self, feat: TEIDFeatureRecord) -> AgentDecision:
        results: dict[str, RuleResult] = {
            "flood": flood_rule(feat, self.config.flood),
            "syn_flood": syn_flood_rule(feat, self.config.syn_flood),
            "scan": scan_rule(feat, self.config.scan),
        }

        triggered = {name: r for name, r in results.items() if r.triggered}
        risk_score = max((r.intensity for r in triggered.values()), default=0.0)

        if triggered:
            reason = "; ".join(f"{name} ({r.detail})" for name, r in triggered.items())
        else:
            reason = "No rule triggered; traffic within normal thresholds."

        return AgentDecision(
            entity_id=f"teid:{feat.teid}:{feat.capture_file}:{feat.window_start}",
            agent_name="TEIDAgent",
            risk_score=risk_score,
            reason=reason,
            rule_triggers=list(triggered.keys()),
            timestamp=feat.window_end,
        )
