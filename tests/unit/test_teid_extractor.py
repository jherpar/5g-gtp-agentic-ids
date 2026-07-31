from __future__ import annotations

import math
from collections import Counter

from agente_5g.preprocessing.teid_extractor import (
    TEIDFeatureExtractor,
    classify_direction,
    shannon_entropy,
)
from tests.fixtures.packet_records import make_packet as _pkt


def test_single_packet_instance_has_zero_duration_and_defaults():
    records = [_pkt(0, teid=1, timestamp=100.0, packet_size=200)]
    (feat,) = list(TEIDFeatureExtractor().extract(records))

    assert feat.teid == 1
    assert feat.packet_count == 1
    assert feat.byte_count == 200
    assert feat.duration_s == 0.0
    assert feat.packets_per_s == 1.0
    assert feat.bytes_per_s == 200.0
    assert feat.std_packet_size == 0.0
    assert feat.interarrival_mean == 0.0
    assert feat.interarrival_std == 0.0
    assert feat.teid_burstiness == 0.0
    assert feat.teid_entropy == 0.0  # single unique size -> zero entropy
    assert feat.teid_reuse_count == 0


def test_uniform_udp_flood_packets_have_low_entropy_and_high_rate():
    records = [
        _pkt(
            i,
            teid=42,
            timestamp=100.0 + i * 0.01,
            packet_size=64,
            inner_proto="udp",
            inner_src_port=40000,
            inner_dst_port=53,
        )
        for i in range(100)
    ]
    (feat,) = list(TEIDFeatureExtractor().extract(records))

    assert feat.packet_count == 100
    assert feat.byte_count == 6400
    assert math.isclose(feat.duration_s, 0.99, rel_tol=1e-6)
    assert feat.teid_entropy == 0.0  # every packet is the same size
    assert feat.packets_per_s > 90  # ~100 packets in ~1s
    assert feat.unique_dst_ips == 1
    assert feat.unique_dst_ports == 1
    assert feat.teid_fanout == 1


def test_scan_like_traffic_has_high_fanout_but_low_unique_ips():
    # One destination IP, many distinct destination ports: classic port scan.
    records = [
        _pkt(
            i,
            teid=7,
            timestamp=100.0 + i * 0.1,
            inner_dst_ip="10.41.150.68",
            inner_dst_port=1000 + i,
        )
        for i in range(20)
    ]
    (feat,) = list(TEIDFeatureExtractor().extract(records))

    assert feat.unique_dst_ips == 1
    assert feat.unique_dst_ports == 20
    assert feat.teid_fanout == 20


def test_entropy_matches_hand_computed_shannon_entropy():
    sizes = [64, 64, 64, 128]  # 3x64, 1x128
    records = [_pkt(i, teid=9, timestamp=100.0 + i, packet_size=s) for i, s in enumerate(sizes)]
    (feat,) = list(TEIDFeatureExtractor().extract(records))

    expected = shannon_entropy(Counter(sizes))
    # hand check: p(64)=0.75, p(128)=0.25 -> -0.75*log2(0.75) - 0.25*log2(0.25)
    hand_expected = -0.75 * math.log2(0.75) - 0.25 * math.log2(0.25)
    assert math.isclose(expected, hand_expected, rel_tol=1e-9)
    assert math.isclose(feat.teid_entropy, hand_expected, rel_tol=1e-9)


def test_teid_reused_after_idle_gap_yields_two_instances():
    records = [
        _pkt(0, teid=5, timestamp=100.0),
        _pkt(1, teid=5, timestamp=101.0),
        # gap of 40s > default 30s threshold -> new instance, reuse_count 1
        _pkt(2, teid=5, timestamp=141.0),
        _pkt(3, teid=5, timestamp=142.0),
    ]
    features = list(TEIDFeatureExtractor(idle_gap_reuse_threshold_s=30.0).extract(records))

    assert len(features) == 2
    assert features[0].teid_reuse_count == 0
    assert features[0].packet_count == 2
    assert features[1].teid_reuse_count == 1
    assert features[1].packet_count == 2


def test_gap_within_threshold_stays_one_instance():
    records = [
        _pkt(0, teid=5, timestamp=100.0),
        _pkt(1, teid=5, timestamp=120.0),  # 20s gap, under 30s threshold
    ]
    features = list(TEIDFeatureExtractor(idle_gap_reuse_threshold_s=30.0).extract(records))

    assert len(features) == 1
    assert features[0].packet_count == 2
    assert features[0].teid_reuse_count == 0


def test_interleaved_teids_are_kept_separate():
    records = [
        _pkt(0, teid=1, timestamp=100.0, packet_size=50),
        _pkt(1, teid=2, timestamp=100.1, packet_size=60),
        _pkt(2, teid=1, timestamp=100.2, packet_size=50),
        _pkt(3, teid=2, timestamp=100.3, packet_size=60),
    ]
    features = {f.teid: f for f in TEIDFeatureExtractor().extract(records)}

    assert set(features) == {1, 2}
    assert features[1].packet_count == 2
    assert features[2].packet_count == 2


def test_non_gtp_and_teidless_packets_are_ignored():
    records = [
        _pkt(0, teid=1, timestamp=100.0, is_gtp=False),  # non-GTP, teid forced None
        _pkt(1, teid=1, timestamp=100.1),
    ]
    features = list(TEIDFeatureExtractor().extract(records))

    assert len(features) == 1
    assert features[0].packet_count == 1


def test_tcp_flag_counts():
    records = [
        _pkt(0, teid=3, timestamp=100.0, tcp_syn=True),
        _pkt(1, teid=3, timestamp=100.1, tcp_syn=True),
        _pkt(2, teid=3, timestamp=100.2, tcp_ack=True, tcp_fin=True),
        _pkt(3, teid=3, timestamp=100.3, tcp_rst=True),
    ]
    (feat,) = list(TEIDFeatureExtractor().extract(records))

    assert feat.syn_count == 2
    assert feat.ack_count == 1
    assert feat.rst_count == 1
    assert feat.fin_count == 1


def test_directionality_uplink_when_src_private_dst_public():
    records = [
        _pkt(
            i,
            teid=11,
            timestamp=100.0 + i,
            inner_src_ip="10.155.15.5",
            inner_dst_ip="93.184.216.34",
        )
        for i in range(3)
    ]
    (feat,) = list(TEIDFeatureExtractor().extract(records))
    assert feat.teid_directionality == 1.0


def test_directionality_downlink_when_src_public_dst_private():
    records = [
        _pkt(
            i,
            teid=12,
            timestamp=100.0 + i,
            inner_src_ip="93.184.216.34",
            inner_dst_ip="10.155.15.5",
        )
        for i in range(3)
    ]
    (feat,) = list(TEIDFeatureExtractor().extract(records))
    assert feat.teid_directionality == 0.0


def test_directionality_neutral_when_unclassifiable():
    # both private (e.g. tunnel-internal addressing) -> "unknown" direction
    records = [_pkt(0, teid=13, timestamp=100.0, inner_src_ip="10.0.0.5", inner_dst_ip="10.0.0.6")]
    (feat,) = list(TEIDFeatureExtractor().extract(records))
    assert feat.teid_directionality == 0.5


def test_classify_direction_helper_directly():
    assert classify_direction("10.155.15.5", "93.184.216.34") == "uplink"
    assert classify_direction("93.184.216.34", "10.155.15.5") == "downlink"
    assert classify_direction("10.0.0.5", "10.0.0.6") == "unknown"
    assert classify_direction(None, "93.184.216.34") == "unknown"


def test_shannon_entropy_of_uniform_distribution_is_maximal():
    # 4 equally likely distinct values -> entropy = log2(4) = 2 bits
    counter = Counter({1: 1, 2: 1, 3: 1, 4: 1})
    assert math.isclose(shannon_entropy(counter), 2.0)


def test_shannon_entropy_of_single_value_is_zero():
    assert shannon_entropy(Counter({1: 5})) == 0.0
