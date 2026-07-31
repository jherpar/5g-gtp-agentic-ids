"""Hand-built TEIDFeatureRecord/PDUSessionRecord instances for agent tests.

Defaults describe unremarkable, clearly-benign traffic; individual tests
override just the fields relevant to the rule under test.
"""

from __future__ import annotations

from typing import Literal

from agente_5g.models.session import PDUSessionRecord
from agente_5g.models.teid_features import TEIDFeatureRecord


def make_teid_feature(
    teid: int = 1,
    *,
    base_station: Literal["BS1", "BS2"] = "BS1",
    capture_file: str = "Test_BS1.pcapng",
    source_attack_type: str = "Test",
    window_start: float = 100.0,
    window_end: float = 110.0,
    packet_count: int = 20,
    byte_count: int = 2000,
    packets_per_s: float = 2.0,
    bytes_per_s: float = 200.0,
    unique_dst_ips: int = 1,
    unique_dst_ports: int = 1,
    avg_packet_size: float = 100.0,
    std_packet_size: float = 5.0,
    interarrival_mean: float = 0.5,
    interarrival_std: float = 0.1,
    syn_count: int = 1,
    ack_count: int = 1,
    rst_count: int = 0,
    fin_count: int = 1,
    teid_lifetime_s: float = 10.0,
    teid_reuse_count: int = 0,
    teid_entropy: float = 3.0,
    teid_burstiness: float = 0.0,
    teid_fanout: int = 1,
    teid_directionality: float = 1.0,
) -> TEIDFeatureRecord:
    return TEIDFeatureRecord(
        teid=teid,
        base_station=base_station,
        capture_file=capture_file,
        source_attack_type=source_attack_type,
        window_start=window_start,
        window_end=window_end,
        packet_count=packet_count,
        byte_count=byte_count,
        duration_s=window_end - window_start,
        packets_per_s=packets_per_s,
        bytes_per_s=bytes_per_s,
        unique_dst_ips=unique_dst_ips,
        unique_dst_ports=unique_dst_ports,
        avg_packet_size=avg_packet_size,
        std_packet_size=std_packet_size,
        interarrival_mean=interarrival_mean,
        interarrival_std=interarrival_std,
        syn_count=syn_count,
        ack_count=ack_count,
        rst_count=rst_count,
        fin_count=fin_count,
        teid_lifetime_s=teid_lifetime_s,
        teid_reuse_count=teid_reuse_count,
        teid_entropy=teid_entropy,
        teid_burstiness=teid_burstiness,
        teid_fanout=teid_fanout,
        teid_directionality=teid_directionality,
    )


def make_session(
    session_id: str = "session-1",
    *,
    ue_ip: str = "10.155.15.5",
    teid: int = 1,
    window_size_s: Literal[1, 5, 10, 30] = 5,
    start_time: float = 100.0,
    end_time: float = 105.0,
    traffic_volume_bytes: int = 1000,
    flow_diversity: int = 1,
    port_diversity: int = 1,
    destination_diversity: int = 1,
    state_transition_rate: float = 0.1,
    temporal_entropy: float = 2.0,
) -> PDUSessionRecord:
    return PDUSessionRecord(
        session_id=session_id,
        ue_ip=ue_ip,
        teid=teid,
        window_size_s=window_size_s,
        start_time=start_time,
        end_time=end_time,
        duration_s=end_time - start_time,
        traffic_volume_bytes=traffic_volume_bytes,
        flow_diversity=flow_diversity,
        port_diversity=port_diversity,
        destination_diversity=destination_diversity,
        state_transition_rate=state_transition_rate,
        temporal_entropy=temporal_entropy,
    )
