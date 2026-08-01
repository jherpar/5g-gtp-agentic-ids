"""Thin, seeded wrapper around sklearn's RandomForestClassifier.

Used for arms A1/A2 (official-baseline data) and arm B (GTP-U/session
features) under the same class, so `ml/train.py` can treat all three the
same way -- only the input DataFrame differs.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from agente_5g.ml.dataset import SEED


class RandomForestModel:
    def __init__(
        self, n_estimators: int = 200, max_depth: int | None = None, seed: int = SEED
    ) -> None:
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
        )

    def fit(self, x: pd.DataFrame, y: pd.Series) -> RandomForestModel:
        self.model.fit(x, y)
        return self

    def predict(self, x: pd.DataFrame) -> npt.NDArray[np.bool_]:
        return np.asarray(self.model.predict(x))

    def predict_proba(self, x: pd.DataFrame) -> npt.NDArray[np.float64]:
        return np.asarray(self.model.predict_proba(x)[:, 1])

    @property
    def feature_importances(self) -> dict[str, float]:
        return dict(zip(self.model.feature_names_in_, self.model.feature_importances_, strict=True))
