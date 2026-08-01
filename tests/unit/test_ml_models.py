from __future__ import annotations

import pandas as pd

from agente_5g.ml.random_forest import RandomForestModel
from agente_5g.ml.xgboost_model import XGBoostModel


def _separable_dataset() -> tuple[pd.DataFrame, pd.Series]:
    # feature "x" >= 10 -> attack, < 10 -> benign: trivially separable so a
    # handful of trees/estimators is enough to fit it exactly.
    x = pd.DataFrame({"x": [0, 1, 2, 3, 10, 11, 12, 13], "y_noise": [5, 4, 6, 5, 5, 6, 4, 5]})
    y = pd.Series([False, False, False, False, True, True, True, True])
    return x, y


def test_random_forest_fits_and_predicts_separable_data():
    x, y = _separable_dataset()
    model = RandomForestModel(n_estimators=20, seed=42).fit(x, y)

    preds = model.predict(x)

    assert list(preds) == list(y)


def test_random_forest_feature_importances_cover_all_columns():
    x, y = _separable_dataset()
    model = RandomForestModel(n_estimators=20, seed=42).fit(x, y)

    importances = model.feature_importances

    assert set(importances.keys()) == {"x", "y_noise"}
    assert importances["x"] > importances["y_noise"]  # x is the real signal


def test_random_forest_is_deterministic_given_same_seed():
    x, y = _separable_dataset()
    preds_a = RandomForestModel(n_estimators=20, seed=42).fit(x, y).predict(x)
    preds_b = RandomForestModel(n_estimators=20, seed=42).fit(x, y).predict(x)

    assert list(preds_a) == list(preds_b)


def test_xgboost_fits_and_predicts_separable_data():
    x, y = _separable_dataset()
    model = XGBoostModel(n_estimators=20, seed=42).fit(x, y)

    preds = model.predict(x)

    assert list(preds) == list(y)


def test_xgboost_predict_proba_in_valid_range():
    x, y = _separable_dataset()
    model = XGBoostModel(n_estimators=20, seed=42).fit(x, y)

    proba = model.predict_proba(x)

    assert all(0.0 <= p <= 1.0 for p in proba)


def test_xgboost_is_deterministic_given_same_seed():
    x, y = _separable_dataset()
    preds_a = XGBoostModel(n_estimators=20, seed=42).fit(x, y).predict(x)
    preds_b = XGBoostModel(n_estimators=20, seed=42).fit(x, y).predict(x)

    assert list(preds_a) == list(preds_b)
