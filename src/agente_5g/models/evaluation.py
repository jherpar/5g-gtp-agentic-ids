"""Evaluation result record (src/agente_5g/evaluation/metrics.py).

`arm` distinguishes the three comparison arms (see plan's "Three-way
comparative evaluation design"): A_official trains only on the dataset
authors' Encoded.csv (no GTP-U info), B_gtp_ml trains classical ML on the
same TEID/session features the agentic system uses, C_agentic is the
TEIDAgent/PDUSessionAgent/SupervisorAgent pipeline itself. There is
deliberately no scalar "explainability score" field here — qualitative
explainability is handled by evaluation/case_studies.py instead.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class EvaluationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_name: str
    arm: Literal["A_official", "B_gtp_ml", "C_agentic"]

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    fpr: float

    detection_time_ms: float
    inference_time_ms: float
    confusion_matrix: list[list[int]]

    run_id: str
    config_hash: str
