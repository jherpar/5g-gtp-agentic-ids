from __future__ import annotations

from agente_5g.models.labels import LabelConfidence
from agente_5g.models.session import PDUSessionRecord
from agente_5g.models.teid_features import TEIDFeatureRecord
from agente_5g.preprocessing.feature_cache import (
    cached_or_compute,
    load_packets,
    load_sessions,
    load_teid_features,
    save_packets,
    save_sessions,
    save_teid_features,
)
from tests.fixtures.feature_records import make_session, make_teid_feature
from tests.fixtures.packet_records import make_packet


def test_packets_round_trip(tmp_path):
    records = [make_packet(0, teid=1, timestamp=100.0), make_packet(1, teid=1, timestamp=100.1)]
    path = tmp_path / "packets.parquet"

    save_packets(records, path)
    loaded = load_packets(path)

    assert loaded == records


def test_teid_features_round_trip_including_labeled_fields(tmp_path):
    feat = make_teid_feature(1).model_copy(
        update={
            "label": "ICMPflood",
            "is_attack": True,
            "label_confidence": LabelConfidence.HIGH,
            "label_evidence": ["SCHEDULE", "VICTIM_IP", "PATTERN"],
        }
    )
    path = tmp_path / "features.parquet"

    save_teid_features([feat], path)
    (loaded,) = load_teid_features(path)

    assert loaded == feat
    assert loaded.label_confidence == LabelConfidence.HIGH


def test_teid_features_round_trip_unlabeled_defaults(tmp_path):
    feat = make_teid_feature(1)  # label/is_attack/label_confidence all None
    path = tmp_path / "features.parquet"

    save_teid_features([feat], path)
    (loaded,) = load_teid_features(path)

    assert loaded.label is None
    assert loaded.label_confidence is None


def test_sessions_round_trip_including_state_sequence(tmp_path):
    session = make_session("s1").model_copy(
        update={"state_sequence": ["NORMAL", "WATCH", "ATTACK"], "final_state": "ATTACK"}
    )
    path = tmp_path / "sessions.parquet"

    save_sessions([session], path)
    (loaded,) = load_sessions(path)

    assert loaded == session
    assert loaded.state_sequence == ["NORMAL", "WATCH", "ATTACK"]


def test_save_empty_list_does_not_write_a_file(tmp_path):
    path = tmp_path / "empty.parquet"
    save_teid_features([], path)
    assert not path.exists()


def test_cached_or_compute_only_calls_compute_once(tmp_path):
    path = tmp_path / "cache.parquet"
    call_count = 0

    def compute() -> list[TEIDFeatureRecord]:
        nonlocal call_count
        call_count += 1
        return [make_teid_feature(1)]

    first = cached_or_compute(path, TEIDFeatureRecord, compute)
    second = cached_or_compute(path, TEIDFeatureRecord, compute)

    assert call_count == 1
    assert first == second


def test_cached_or_compute_loads_correct_model_type(tmp_path):
    path = tmp_path / "sessions_cache.parquet"
    result = cached_or_compute(path, PDUSessionRecord, lambda: [make_session("s1")])
    assert isinstance(result[0], PDUSessionRecord)
