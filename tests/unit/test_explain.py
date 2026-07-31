from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
import requests

from agente_5g.agents.explain import generate_explanation
from agente_5g.models.agent_decision import AgentDecision, SupervisorDecision
from agente_5g.settings import LLMConfig


def _decision(entity_id: str = "teid:1:file:0.0") -> SupervisorDecision:
    teid_decision = AgentDecision(
        entity_id=entity_id,
        agent_name="TEIDAgent",
        risk_score=0.9,
        reason="flood detected",
        rule_triggers=["flood"],
        timestamp=10.0,
    )
    return SupervisorDecision(
        entity_id=entity_id,
        teid_decision=teid_decision,
        session_decision=None,
        fused_risk_score=0.9,
        final_label="Attack",
        predicted_attack_type="ICMPflood",
        explanation="TEIDAgent: flood detected",
        decision_latency_ms=1.0,
    )


@pytest.fixture
def llm_config(tmp_path) -> LLMConfig:
    return LLMConfig(
        enabled=True,
        endpoint="http://localhost:11434/api/generate",
        model="gemma4:latest",
        fallback_model="llama3.2:1b",
        cache_path=tmp_path / "llm_cache.jsonl",
        timeout_s=5,
    )


def test_disabled_returns_none_without_network_call(llm_config):
    llm_config.enabled = False
    with patch("agente_5g.agents.explain.requests.post") as mock_post:
        result = generate_explanation(_decision(), llm_config)
    assert result is None
    mock_post.assert_not_called()


def test_enabled_calls_ollama_and_returns_text(llm_config):
    mock_response = Mock()
    mock_response.json.return_value = {"response": "This looks like a flood attack."}
    mock_response.raise_for_status = Mock()
    with patch("agente_5g.agents.explain.requests.post", return_value=mock_response) as mock_post:
        result = generate_explanation(_decision(), llm_config)

    assert result == "This looks like a flood attack."
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["json"]["model"] == "gemma4:latest"
    assert call_kwargs["timeout"] == 5


def test_result_is_cached_and_second_call_skips_the_network(llm_config):
    mock_response = Mock()
    mock_response.json.return_value = {"response": "cached explanation"}
    mock_response.raise_for_status = Mock()
    decision = _decision()

    with patch("agente_5g.agents.explain.requests.post", return_value=mock_response) as mock_post:
        first = generate_explanation(decision, llm_config)
        second = generate_explanation(decision, llm_config)

    assert first == second == "cached explanation"
    mock_post.assert_called_once()  # second call hit the cache, not the network


def test_request_exception_returns_none_without_raising(llm_config):
    with patch(
        "agente_5g.agents.explain.requests.post",
        side_effect=requests.ConnectionError("ollama not running"),
    ):
        result = generate_explanation(_decision(), llm_config)
    assert result is None


def test_empty_response_text_returns_none(llm_config):
    mock_response = Mock()
    mock_response.json.return_value = {"response": "   "}
    mock_response.raise_for_status = Mock()
    with patch("agente_5g.agents.explain.requests.post", return_value=mock_response):
        result = generate_explanation(_decision(), llm_config)
    assert result is None
