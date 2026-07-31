"""Optional LLM-generated natural-language explanations via local Ollama.

STRICTLY POST-HOC: never used for scoring or classification -- the
Supervisor's `fused_risk_score`/`final_label` are fully deterministic
before this module is even consulted (see `supervisor_agent.py`). Disabled
by default (`configs/base.yaml`'s `llm.enabled: false`) since text
generation isn't reproducible run-to-run and the project's reproducibility
mandate requires metrics to never depend on it. When enabled, outputs are
cached to `llm.cache_path` (JSON-lines, keyed by entity_id) so re-running
the same pipeline doesn't regenerate text or re-hit the LLM.
"""

from __future__ import annotations

import json
from pathlib import Path

import requests

from agente_5g.models.agent_decision import SupervisorDecision
from agente_5g.settings import LLMConfig
from agente_5g.utils.logging import JsonlWriter, get_logger

logger = get_logger(__name__)


def _read_cache(cache_path: Path) -> dict[str, str]:
    if not cache_path.exists():
        return {}
    cache: dict[str, str] = {}
    with cache_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            cache[record["entity_id"]] = record["llm_explanation"]
    return cache


def _build_prompt(decision: SupervisorDecision) -> str:
    return (
        "You are assisting a 5G network security analyst. In 2-3 concise "
        "sentences, explain in plain language why this traffic was "
        f"classified as {decision.final_label} (fused risk score "
        f"{decision.fused_risk_score:.2f}). Base your explanation only on "
        f"the evidence given, do not invent details.\n\nEvidence: {decision.explanation}"
    )


def generate_explanation(decision: SupervisorDecision, config: LLMConfig) -> str | None:
    """Return LLM-generated prose for `decision`, or None if disabled,
    cached-and-empty, or the LLM call fails. Never raises: a failed or
    unavailable LLM degrades to None, since explanation prose must never
    block or alter the deterministic pipeline."""
    if not config.enabled:
        return None

    cache_path = Path(config.cache_path)
    cache = _read_cache(cache_path)
    if decision.entity_id in cache:
        return cache[decision.entity_id]

    try:
        response = requests.post(
            config.endpoint,
            json={"model": config.model, "prompt": _build_prompt(decision), "stream": False},
            timeout=config.timeout_s,
        )
        response.raise_for_status()
        text = str(response.json().get("response", "")).strip()
    except requests.RequestException as exc:
        logger.warning("Ollama explanation generation failed (model=%s): %s", config.model, exc)
        return None

    if not text:
        return None

    JsonlWriter(cache_path).write({"entity_id": decision.entity_id, "llm_explanation": text})
    return text
