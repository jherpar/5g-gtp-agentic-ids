"""Infers PDU sessions from GTP-U packets (NSA doesn't expose them explicitly).

A session groups packets by (ue_ip, teid) within a fixed-size, epoch-aligned
temporal window (`window_size_s`, one of {1, 5, 10, 30} per
`configs/base.yaml`'s `session.window_sizes_s` -- callers build a separate
session dataset at each granularity, e.g. to compare how detection speed
changes with window size for RQ4). UE IP is inferred via flow-initiation
(same convention as `teid_extractor.infer_initiator_ip`: whoever sent a
TEID's earliest packet owns that tunnel), not read from
`GTPPacketRecord.ue_ip` (left unset by the parser) and not from public/
private IP classification -- this dataset's victim/MEC server also sits on a
private address, so a public/private split can't tell attacker and victim
apart. The initiator is resolved once per TEID over the *whole* file (a
single pass before windowing) since GTP-U TEIDs are protocol-unidirectional,
then reused for every window that TEID appears in.

`session_id = sha256(f"{ue_ip}|{teid}|{window_start}")` uses the window's
epoch-aligned boundary rather than any packet's actual arrival time, so it's
stable across re-runs regardless of minor timing jitter or stream order.

Feature definitions worth calling out:
  - state_transition_rate: rate (per second) of changes in a coarse
    per-packet TCP-flag-derived pseudo-state (SYN/SYN-ACK/ACK/FIN/RST/OTHER)
    across consecutive packets in the window. This is a raw traffic signal,
    computed independently of the PDUSessionAgent's
    NORMAL/WATCH/SUSPICIOUS/ATTACK reasoning state
    (`final_state`/`state_sequence`), which is populated later in Phase 5 by
    a different module using different logic.
  - temporal_entropy: Shannon entropy (bits) of packet arrivals across 10
    equal sub-bins of the window -- evenly spread traffic reads as high
    entropy, a burst concentrated in one sub-bin reads as low entropy.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from typing import Literal

from agente_5g.models.packet import GTPPacketRecord
from agente_5g.models.session import PDUSessionRecord
from agente_5g.preprocessing.teid_extractor import infer_initiator_ip, shannon_entropy

_SUB_BINS = 10
_VALID_WINDOW_SIZES = (1, 5, 10, 30)


def _packet_flow_state(record: GTPPacketRecord) -> str:
    if record.tcp_rst:
        return "RST"
    if record.tcp_fin:
        return "FIN"
    if record.tcp_syn and record.tcp_ack:
        return "SYNACK"
    if record.tcp_syn:
        return "SYN"
    if record.tcp_ack:
        return "ACK"
    return "OTHER"


class SessionBuilder:
    def __init__(self, window_size_s: Literal[1, 5, 10, 30]) -> None:
        if window_size_s not in _VALID_WINDOW_SIZES:
            msg = f"window_size_s must be one of {_VALID_WINDOW_SIZES}, got {window_size_s}"
            raise ValueError(msg)
        self.window_size_s: Literal[1, 5, 10, 30] = window_size_s

    def build(self, records: Iterable[GTPPacketRecord]) -> Iterator[PDUSessionRecord]:
        gtp_records = [r for r in records if r.is_gtp and r.teid is not None]

        packets_by_teid: dict[int, list[GTPPacketRecord]] = defaultdict(list)
        for record in gtp_records:
            teid = record.teid
            assert teid is not None  # guaranteed by the filter above; narrows for mypy
            packets_by_teid[teid].append(record)
        initiator_by_teid = {
            teid: infer_initiator_ip(packets) for teid, packets in packets_by_teid.items()
        }

        buckets: dict[tuple[str, int, int], list[GTPPacketRecord]] = defaultdict(list)
        for record in gtp_records:
            teid = record.teid
            assert teid is not None
            ue_ip = initiator_by_teid.get(teid)
            if ue_ip is None:
                continue
            window_index = int(record.timestamp // self.window_size_s)
            buckets[(ue_ip, teid, window_index)].append(record)

        for (ue_ip, teid, window_index), packets in buckets.items():
            yield self._build_record(ue_ip, teid, window_index, packets)

    def _build_record(
        self, ue_ip: str, teid: int, window_index: int, packets: list[GTPPacketRecord]
    ) -> PDUSessionRecord:
        packets = sorted(packets, key=lambda r: r.timestamp)
        window_start = window_index * self.window_size_s
        window_end = window_start + self.window_size_s

        session_id = hashlib.sha256(f"{ue_ip}|{teid}|{window_start}".encode()).hexdigest()

        first_ts = packets[0].timestamp
        last_ts = packets[-1].timestamp
        duration = last_ts - first_ts

        traffic_volume_bytes = sum(p.packet_size for p in packets)

        flows = {
            (p.inner_src_ip, p.inner_dst_ip, p.inner_src_port, p.inner_dst_port, p.inner_proto)
            for p in packets
        }
        dst_ports = {p.inner_dst_port for p in packets if p.inner_dst_port is not None}
        dst_ips = {p.inner_dst_ip for p in packets if p.inner_dst_ip}

        states = [_packet_flow_state(p) for p in packets]
        transitions = sum(1 for a, b in zip(states, states[1:], strict=False) if a != b)
        state_transition_rate = transitions / duration if duration > 0 else float(transitions)

        # A small epsilon guards against float cancellation (e.g. offsets
        # computed from `100.7 - 100.0` landing a hair under the next
        # sub-bin boundary and truncating down) pushing samples into the
        # wrong sub-bin at exact-looking boundaries.
        sub_bin_counts: Counter[int] = Counter()
        for p in packets:
            offset = p.timestamp - window_start
            sub_bin = min(int(offset / self.window_size_s * _SUB_BINS + 1e-9), _SUB_BINS - 1)
            sub_bin_counts[sub_bin] += 1
        temporal_entropy = shannon_entropy(sub_bin_counts)

        return PDUSessionRecord(
            session_id=session_id,
            ue_ip=ue_ip,
            teid=teid,
            window_size_s=self.window_size_s,
            start_time=float(window_start),
            end_time=float(window_end),
            duration_s=duration,
            traffic_volume_bytes=traffic_volume_bytes,
            flow_diversity=len(flows),
            port_diversity=len(dst_ports),
            destination_diversity=len(dst_ips),
            state_transition_rate=state_transition_rate,
            temporal_entropy=temporal_entropy,
        )
