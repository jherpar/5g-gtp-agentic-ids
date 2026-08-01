"""Evaluation result record (src/agente_5g/evaluation/metrics.py).

`arm` distinguishes the four comparison arms (see plan's "Three-way
comparative evaluation design", extended to four per the Phase 6 data
review): A1_combined trains on the dataset authors' Combined.csv with our
own documented preprocessing (no GTP-U info, clean/interpretable columns),
A2_encoded trains on the authors' own pre-encoded Encoded.csv verbatim (kept
as a secondary reproducibility check only, since many of its column names
are uninterpretable), B_gtp_ml trains classical ML on the same TEID/session
features the agentic system uses, C_agentic is the
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
    arm: Literal["A1_combined", "A2_encoded", "B_gtp_ml", "C_agentic"]

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
