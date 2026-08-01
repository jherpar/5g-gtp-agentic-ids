"""Thin, seeded wrapper around xgboost's XGBClassifier.

`scale_pos_weight` is computed from the training labels at `fit()` time
(standard XGBoost imbalance handling, since it has no `class_weight="balanced"`
equivalent) rather than fixed up front, since the attack/benign ratio
differs substantially between arms (e.g. Combined.csv is roughly balanced,
the GTP session data is heavily benign-skewed).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from xgboost import XGBClassifier

from agente_5g.ml.dataset import SEED


class XGBoostModel:
    def __init__(self, n_estimators: int = 200, max_depth: int = 6, seed: int = SEED) -> None:
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=seed,
            eval_metric="logloss",
            n_jobs=-1,
        )

    def fit(self, x: pd.DataFrame, y: pd.Series) -> XGBoostModel:
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        self.model.set_params(scale_pos_weight=scale_pos_weight)
        self.model.fit(x, y)
        return self

    def predict(self, x: pd.DataFrame) -> npt.NDArray[np.bool_]:
        return np.asarray(self.model.predict(x))

    def predict_proba(self, x: pd.DataFrame) -> npt.NDArray[np.float64]:
        return np.asarray(self.model.predict_proba(x)[:, 1])

    @property
    def feature_importances(self) -> dict[str, float]:
        return dict(zip(self.model.feature_names_in_, self.model.feature_importances_, strict=True))
