from __future__ import annotations

import pandas as pd

from agente_5g.ml.dataset import (
    GTP_SESSION_FEATURE_COLUMNS,
    build_gtp_session_dataset,
    chronological_split,
    load_combined_csv,
    to_arm_a_matrix,
    to_gtp_matrix,
)
from agente_5g.models.labels import LabelConfidence
from tests.fixtures.feature_records import make_session

COMBINED_CSV_ROWS = """,Seq,Dur,Proto,sTos,dTos,sDSb,dDSb,Cause,State,Label,Attack Type,Attack Tool
0,1,0.1,udp,0.0,,cs0,,Start,CON,Benign,Benign,Benign
1,2,0.2,tcp,0.0,0.0,cs0,cs0,Start,FIN,Malicious,SYNFlood,hping3
2,3,0.3,icmp,,,,,Status,ECO,Benign,Benign,Benign
"""


def test_load_combined_csv_encodes_categoricals_and_target(tmp_path):
    csv_path = tmp_path / "combined.csv"
    csv_path.write_text(COMBINED_CSV_ROWS, encoding="utf-8")

    df = load_combined_csv(csv_path)

    assert df["is_attack"].tolist() == [False, True, False]
    # one-hot columns exist for the observed categories
    assert "Proto_udp" in df.columns
    assert "Proto_tcp" in df.columns
    assert "State_ECO" in df.columns


def test_to_arm_a_matrix_imputes_nan_and_drops_leakage_columns(tmp_path):
    csv_path = tmp_path / "combined.csv"
    csv_path.write_text(COMBINED_CSV_ROWS, encoding="utf-8")
    df = load_combined_csv(csv_path)

    x, y = to_arm_a_matrix(df)

    assert list(y) == [False, True, False]
    assert "Attack Type" not in x.columns
    assert "Label" not in x.columns
    assert "is_attack" not in x.columns
    assert x.isna().sum().sum() == 0  # dTos NaN imputed to -1
    assert (x["dTos"] == -1).sum() == 2  # rows 0 and 2 had NaN dTos


def test_chronological_split_is_ordered_not_shuffled():
    df = pd.DataFrame({"Seq": [3, 1, 2, 5, 4], "value": ["c", "a", "b", "e", "d"]})

    train, test = chronological_split(df, order_column="Seq", test_fraction=0.4)

    assert train["Seq"].tolist() == [1, 2, 3]
    assert test["Seq"].tolist() == [4, 5]


def test_build_gtp_session_dataset_never_splits_a_teid_group():
    sessions = [
        make_session(
            f"s{i}", teid=1, start_time=float(i * 5), end_time=float(i * 5 + 5)
        ).model_copy(update={"is_attack": True, "label_confidence": LabelConfidence.HIGH})
        for i in range(10)
    ]

    train_df, test_df = build_gtp_session_dataset({"TestAttack": sessions}, test_fraction=0.3)

    train_teids = set(zip(train_df.get("session_id", []), [1] * len(train_df), strict=False))
    # every session shares the same (ue_ip, teid) -> must all land on the same side
    assert len(train_df) == 0 or len(test_df) == 0
    assert len(train_df) + len(test_df) == 10
    del train_teids


def test_build_gtp_session_dataset_splits_by_group_chronologically():
    early_group = [
        make_session(f"early-{i}", teid=1, start_time=float(i), end_time=float(i + 1)).model_copy(
            update={"is_attack": False, "label_confidence": LabelConfidence.HIGH}
        )
        for i in range(3)
    ]
    late_group = [
        make_session(
            f"late-{i}", teid=2, start_time=float(1000 + i), end_time=float(1001 + i)
        ).model_copy(update={"is_attack": True, "label_confidence": LabelConfidence.HIGH})
        for i in range(3)
    ]

    train_df, test_df = build_gtp_session_dataset(
        {"TestAttack": early_group + late_group}, test_fraction=0.5
    )

    assert set(train_df["session_id"]) == {s.session_id for s in early_group}
    assert set(test_df["session_id"]) == {s.session_id for s in late_group}


def test_build_gtp_session_dataset_excludes_unlabeled_sessions():
    unlabeled = make_session("unlabeled", teid=1)  # is_attack defaults to None
    labeled = make_session("labeled", teid=2).model_copy(
        update={"is_attack": True, "label_confidence": LabelConfidence.HIGH}
    )

    train_df, test_df = build_gtp_session_dataset(
        {"TestAttack": [unlabeled, labeled]}, test_fraction=0.5
    )

    all_ids = set(train_df["session_id"]) | set(test_df["session_id"])
    assert all_ids == {"labeled"}


def test_to_gtp_matrix_selects_feature_columns_only():
    sessions = [
        make_session("s1").model_copy(
            update={"is_attack": True, "label_confidence": LabelConfidence.HIGH}
        )
    ]
    train_df, _ = build_gtp_session_dataset({"TestAttack": sessions}, test_fraction=0.0)

    x, y = to_gtp_matrix(train_df)

    assert list(x.columns) == GTP_SESSION_FEATURE_COLUMNS
    assert list(y) == [True]
