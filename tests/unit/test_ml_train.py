from __future__ import annotations

import pandas as pd

from agente_5g.ml.train import (
    _match_feature,
    evaluate_arm_c,
    train_and_evaluate_arm_a,
    train_and_evaluate_arm_b,
)
from agente_5g.models.agent_thresholds import (
    FloodRuleConfig,
    PDUSessionAgentConfig,
    ScanRuleConfig,
    StateMachineConfig,
    SupervisorAgentConfig,
    SynFloodRuleConfig,
    TEIDAgentConfig,
    ThresholdsConfig,
)
from agente_5g.models.labels import LabelConfidence
from tests.fixtures.feature_records import make_session, make_teid_feature

THRESHOLDS = ThresholdsConfig(
    teid_agent=TEIDAgentConfig(
        flood=FloodRuleConfig(
            min_packets_per_s=100.0, max_teid_entropy=0.5, max_unique_dst_ports=2
        ),
        syn_flood=SynFloodRuleConfig(min_syn_count=1000, max_ack_to_syn_ratio=0.05),
        scan=ScanRuleConfig(min_unique_dst_ports=50, max_packets_per_dst_port=1.0),
    ),
    pdu_session_agent=PDUSessionAgentConfig(
        state_machine=StateMachineConfig(normal_max=0.25, watch_max=0.5, suspicious_max=0.75),
        high_state_transition_rate=50.0,
        low_temporal_entropy=-1.0,
        high_diversity=500,
    ),
    supervisor_agent=SupervisorAgentConfig(
        fusion_weights={"teid_agent": 0.6, "pdu_session_agent": 0.4},
        attack_decision_threshold=0.5,
    ),
)


def _arm_a_frame(n_benign: int, n_attack: int) -> pd.DataFrame:
    rows = []
    seq = 1
    for _ in range(n_benign):
        rows.append({"Seq": seq, "feat1": 1.0, "feat2": 5.0, "is_attack": False})
        seq += 1
    for _ in range(n_attack):
        rows.append({"Seq": seq, "feat1": 100.0, "feat2": 5.0, "is_attack": True})
        seq += 1
    return pd.DataFrame(rows)


def test_train_and_evaluate_arm_a_returns_rf_and_xgboost_results():
    df = _arm_a_frame(n_benign=40, n_attack=40)

    results = train_and_evaluate_arm_a(df, arm="A1_combined", test_fraction=0.3)

    assert {r.model_name for r in results} == {"RandomForest", "XGBoost"}
    assert all(r.arm == "A1_combined" for r in results)
    assert all(0.0 <= r.accuracy <= 1.0 for r in results)


def _gtp_frame(n_benign: int, n_attack: int, confidence: str) -> pd.DataFrame:
    rows = []
    for i in range(n_benign):
        rows.append(
            {
                "window_size_s": 5,
                "duration_s": 5.0,
                "traffic_volume_bytes": 500,
                "flow_diversity": 1,
                "port_diversity": 1,
                "destination_diversity": 1,
                "state_transition_rate": 0.1,
                "temporal_entropy": 3.0,
                "is_attack": False,
                "label_confidence": "HIGH",
                "attack_type": "Test",
                "session_id": f"benign-{i}",
            }
        )
    for i in range(n_attack):
        rows.append(
            {
                "window_size_s": 5,
                "duration_s": 5.0,
                "traffic_volume_bytes": 500000,
                "flow_diversity": 50,
                "port_diversity": 50,
                "destination_diversity": 50,
                "state_transition_rate": 90.0,
                "temporal_entropy": 0.1,
                "is_attack": True,
                "label_confidence": confidence,
                "attack_type": "Test",
                "session_id": f"attack-{i}",
            }
        )
    return pd.DataFrame(rows)


def test_train_and_evaluate_arm_b_returns_view_a_and_view_b_results():
    train_df = _gtp_frame(30, 30, confidence="HIGH")
    test_df = _gtp_frame(10, 10, confidence="MEDIUM")

    results, models = train_and_evaluate_arm_b(train_df, test_df)

    names = {r.model_name for r in results}
    assert names == {
        "RandomForest (view A)",
        "RandomForest (view B)",
        "XGBoost (view A)",
        "XGBoost (view B)",
    }
    assert all(r.arm == "B_gtp_ml" for r in results)
    assert set(models.keys()) == {"RandomForest", "XGBoost"}


def test_train_and_evaluate_arm_b_view_b_excludes_low_confidence():
    train_df = _gtp_frame(20, 20, confidence="HIGH")
    test_df = _gtp_frame(5, 5, confidence="LOW")

    results, _ = train_and_evaluate_arm_b(train_df, test_df)

    view_b = next(r for r in results if r.model_name == "RandomForest (view B)")
    # every attack row in test_df is LOW confidence -> view B has 0 positives
    tp_plus_fn = view_b.confusion_matrix[1][0] + view_b.confusion_matrix[1][1]
    assert tp_plus_fn == 0


def test_match_feature_prefers_containing_window():
    session = make_session("s1", teid=1, start_time=10.0, end_time=15.0)
    contained = make_teid_feature(teid=1, window_start=0.0, window_end=20.0)
    other = make_teid_feature(teid=1, window_start=100.0, window_end=110.0)

    matched = _match_feature(session, {1: [other, contained]})

    assert matched is contained


def test_match_feature_falls_back_to_nearest_when_no_containment():
    session = make_session("s1", teid=1, start_time=1000.0, end_time=1005.0)
    near = make_teid_feature(teid=1, window_start=990.0, window_end=995.0)
    far = make_teid_feature(teid=1, window_start=0.0, window_end=5.0)

    matched = _match_feature(session, {1: [far, near]})

    assert matched is near


def test_match_feature_returns_none_for_unknown_teid():
    session = make_session("s1", teid=99)
    assert _match_feature(session, {1: [make_teid_feature(teid=1)]}) is None


def test_evaluate_arm_c_detects_obvious_attack_and_ignores_normal_traffic():
    benign_session = make_session("benign-1", teid=1, start_time=0.0, end_time=5.0).model_copy(
        update={"is_attack": False, "label_confidence": LabelConfidence.HIGH}
    )
    attack_session = make_session(
        "attack-1",
        teid=2,
        start_time=0.0,
        end_time=5.0,
        state_transition_rate=200.0,
        port_diversity=1000,
        destination_diversity=1000,
    ).model_copy(update={"is_attack": True, "label_confidence": LabelConfidence.HIGH})

    benign_feat = make_teid_feature(teid=1, window_start=0.0, window_end=5.0)
    attack_feat = make_teid_feature(
        teid=2,
        window_start=0.0,
        window_end=5.0,
        packets_per_s=10000.0,
        teid_entropy=0.0,
        unique_dst_ports=1,
    )

    results = evaluate_arm_c(
        {"Test": [benign_session, attack_session]},
        {"Test": [benign_feat, attack_feat]},
        THRESHOLDS,
    )

    view_a = next(r for r in results if "view A" in r.model_name)
    assert view_a.confusion_matrix == [[1, 0], [0, 1]]  # 1 TN, 1 TP, no errors
