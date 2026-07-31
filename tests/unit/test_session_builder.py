from __future__ import annotations

import math

import pytest

from agente_5g.preprocessing.session_builder import SessionBuilder
from tests.fixtures.packet_records import make_packet

PRIVATE_UE_IP = "10.155.15.5"
PUBLIC_PEER_IP = "93.184.216.34"


def _up(packet_id: int, teid: int, timestamp: float, **kwargs):
    """An uplink packet: UE (private) -> peer (public)."""
    kwargs.setdefault("inner_src_ip", PRIVATE_UE_IP)
    kwargs.setdefault("inner_dst_ip", PUBLIC_PEER_IP)
    return make_packet(packet_id, teid, timestamp, **kwargs)


def _down(packet_id: int, teid: int, timestamp: float, **kwargs):
    """A downlink packet: peer (public) -> UE (private)."""
    kwargs.setdefault("inner_src_ip", PUBLIC_PEER_IP)
    kwargs.setdefault("inner_dst_ip", PRIVATE_UE_IP)
    return make_packet(packet_id, teid, timestamp, **kwargs)


def test_two_uplink_packets_same_window_form_one_session():
    records = [
        _up(0, 1, 100.0, packet_size=100),
        _up(1, 1, 100.5, packet_size=150),
    ]
    (session,) = list(SessionBuilder(window_size_s=1).build(records))

    assert session.ue_ip == PRIVATE_UE_IP
    assert session.teid == 1
    assert session.traffic_volume_bytes == 250
    assert math.isclose(session.duration_s, 0.5)
    assert session.start_time == 100.0
    assert session.end_time == 101.0


def test_downlink_packet_infers_ue_ip_as_private_destination():
    (session,) = list(SessionBuilder(window_size_s=1).build([_down(0, 2, 100.0)]))
    assert session.ue_ip == PRIVATE_UE_IP


def test_unclassifiable_direction_packets_are_excluded():
    both_private = make_packet(0, 3, 100.0, inner_src_ip="10.0.0.5", inner_dst_ip="10.0.0.6")
    records = [both_private, _up(1, 3, 100.1)]
    (session,) = list(SessionBuilder(window_size_s=1).build(records))
    assert session.traffic_volume_bytes == 100  # only the classifiable packet counted


def test_packets_in_different_windows_form_separate_sessions():
    records = [_up(0, 1, 100.0), _up(1, 1, 105.0)]
    sessions = sorted(SessionBuilder(window_size_s=1).build(records), key=lambda s: s.start_time)

    assert len(sessions) == 2
    assert sessions[0].start_time == 100.0
    assert sessions[1].start_time == 105.0
    assert sessions[0].session_id != sessions[1].session_id


def test_window_boundary_uses_floor_division():
    # window_size_s=5: a packet at t=105.0 belongs to window [105, 110), not [100, 105)
    (session,) = list(SessionBuilder(window_size_s=5).build([_up(0, 1, 105.0)]))
    assert session.start_time == 105.0
    assert session.end_time == 110.0


def test_session_id_is_deterministic_and_window_sensitive():
    (s1,) = list(SessionBuilder(window_size_s=1).build([_up(0, 1, 100.0)]))
    (s2,) = list(SessionBuilder(window_size_s=1).build([_up(0, 1, 100.5)]))  # same window
    (s3,) = list(SessionBuilder(window_size_s=1).build([_up(0, 1, 101.0)]))  # next window

    assert s1.session_id == s2.session_id
    assert s1.session_id != s3.session_id


def test_flow_port_destination_diversity_counts():
    records = [
        _up(i, 1, 100.0 + i * 0.01, inner_dst_ip=f"93.184.216.{i}", inner_dst_port=1000 + i)
        for i in range(5)
    ]
    (session,) = list(SessionBuilder(window_size_s=1).build(records))

    assert session.destination_diversity == 5
    assert session.port_diversity == 5
    assert session.flow_diversity == 5


def test_state_transition_rate_hand_computed():
    # states in order: SYN, SYN, ACK, FIN -> transitions SYN->ACK, ACK->FIN = 2
    records = [
        _up(0, 1, 100.0, tcp_syn=True),
        _up(1, 1, 100.1, tcp_syn=True),
        _up(2, 1, 100.2, tcp_ack=True),
        _up(3, 1, 100.4, tcp_fin=True),
    ]
    (session,) = list(SessionBuilder(window_size_s=1).build(records))

    expected_rate = 2 / (100.4 - 100.0)
    assert math.isclose(session.state_transition_rate, expected_rate, rel_tol=1e-9)


def test_temporal_entropy_zero_when_all_packets_in_one_instant():
    records = [_up(i, 1, 100.0) for i in range(5)]
    (session,) = list(SessionBuilder(window_size_s=1).build(records))
    assert session.temporal_entropy == 0.0


def test_temporal_entropy_positive_when_spread_across_subbins():
    # window_size_s=1, 10 sub-bins -> spread one packet per 0.1s sub-bin
    records = [_up(i, 1, 100.0 + i * 0.1) for i in range(10)]
    (session,) = list(SessionBuilder(window_size_s=1).build(records))
    assert math.isclose(session.temporal_entropy, math.log2(10), rel_tol=1e-6)


def test_non_gtp_packets_are_ignored():
    sessions = list(SessionBuilder(window_size_s=1).build([_up(0, 1, 100.0, is_gtp=False)]))
    assert sessions == []


def test_invalid_window_size_raises():
    with pytest.raises(ValueError, match="window_size_s"):
        SessionBuilder(window_size_s=7)  # type: ignore[arg-type]


@pytest.mark.parametrize("window_size_s", [1, 5, 10, 30])
def test_all_configured_window_sizes_are_accepted(window_size_s: int):
    (session,) = list(SessionBuilder(window_size_s=window_size_s).build([_up(0, 1, 100.0)]))  # type: ignore[arg-type]
    assert session.window_size_s == window_size_s
