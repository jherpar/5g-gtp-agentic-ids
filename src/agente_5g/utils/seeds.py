"""Deterministic seeding, per the project's reproducibility mandate.

Scikit-learn/XGBoost determinism is enforced by passing `random_state`
explicitly at every model construction site (see `ml/random_forest.py`,
`ml/xgboost_model.py`, `ml/baseline_official.py`) rather than relying solely
on global state — this function seeds the global generators used elsewhere
(e.g. sampling, shuffling) but is not a substitute for that.
"""

from __future__ import annotations

import random

import numpy as np


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
