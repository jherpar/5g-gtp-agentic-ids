"""Groups parsed GTP-U packets by TEID into feature records.

A TEID's packets are split into distinct "instances" whenever the gap
between consecutive packets on that TEID exceeds
`idle_gap_reuse_threshold_s` (see `configs/base.yaml`'s `teid` section) --
GTP-U tunnel endpoints reuse TEID values over time, and treating a
long-idle reappearance as a fresh instance avoids conflating unrelated
traffic bursts into one inflated feature row. `teid_reuse_count` is the
0-indexed instance number for that TEID value within its capture file.

Feature definitions worth calling out (not obvious from the field name
alone):
  - teid_lifetime_s: the active span of THIS instance (currently equal to
    duration_s). Kept as a separate field because the project's feature
    spec treats "lifetime" and "duration" as conceptually distinct, even
    though they coincide at the single-instance granularity computed here.
  - teid_entropy: Shannon entropy (bits) of the packet-size distribution
    within the instance -- floods of near-identical packets read as low
    entropy, heterogeneous traffic as higher entropy.
  - teid_burstiness: the Goh & Barabasi burstiness parameter
    (std-mean)/(std+mean) of interarrival times, in [-1, 1]; -1 is
    perfectly periodic, 0 is Poisson-like, +1 is maximally bursty. 0.0 when
    fewer than 2 interarrival samples exist (insufficient data).
  - teid_fanout: count of distinct (dst_ip, dst_port) endpoints touched --
    deliberately distinct from unique_dst_ips (IP-only), so a scan hitting
    many ports on one IP is visible here even when unique_dst_ips is low.
  - teid_directionality: uplink_bytes / (uplink_bytes + downlink_bytes),
    where direction is inferred by classifying the inner src/dst as
    private vs. public IPs (UEs get private/NAT'd addresses; the wider
    internet doesn't), rather than a hardcoded outer-IP topology, since
    which literal outer IP is the eNB side vs. the core side is not fixed
    across capture files. 0.5 (neutral) when nothing could be classified.

Labeling (`label`/`is_attack`/`label_confidence`) is intentionally left
unset here -- that's `preprocessing/labeling.py`'s job (Phase 4), applied as
a separate enrichment pass so this module has no knowledge of the attack
schedule/labeling heuristics.
"""

from __future__ import annotations

import ipaddress
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from typing import Literal

from agente_5g.models.packet import GTPPacketRecord
from agente_5g.models.teid_features import TEIDFeatureRecord

Direction = Literal["uplink", "downlink", "unknown"]


def _is_private_ip(ip: str | None) -> bool | None:
    if not ip:
        return None
    try:
        return ipaddress.ip_address(ip).is_private
    except ValueError:
        return None


def classify_direction(inner_src_ip: str | None, inner_dst_ip: str | None) -> Direction:
    src_private = _is_private_ip(inner_src_ip)
    dst_private = _is_private_ip(inner_dst_ip)
    if src_private is True and dst_private is False:
        return "uplink"
    if dst_private is True and src_private is False:
        return "downlink"
    return "unknown"


def shannon_entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


class TEIDFeatureExtractor:
    def __init__(self, idle_gap_reuse_threshold_s: float = 30.0) -> None:
        self.idle_gap_reuse_threshold_s = idle_gap_reuse_threshold_s

    def extract(self, records: Iterable[GTPPacketRecord]) -> Iterator[TEIDFeatureRecord]:
        """Consume packets in capture (timestamp) order; yield one
        TEIDFeatureRecord per (teid, instance).

        Instances only close (and get yielded) once a later packet on the
        same TEID arrives past the idle-gap threshold, or the stream ends --
        so this is a streaming-friendly single pass, not a full sort+groupby.
        """
        open_instances: dict[int, list[GTPPacketRecord]] = defaultdict(list)
        reuse_counts: dict[int, int] = defaultdict(int)
        last_ts: dict[int, float] = {}

        for record in records:
            if not record.is_gtp or record.teid is None:
                continue

            teid = record.teid
            gap = record.timestamp - last_ts[teid] if teid in last_ts else 0.0
            if gap > self.idle_gap_reuse_threshold_s:
                yield self._build_record(teid, open_instances.pop(teid), reuse_counts[teid])
                reuse_counts[teid] += 1

            open_instances[teid].append(record)
            last_ts[teid] = record.timestamp

        for teid, packets in open_instances.items():
            yield self._build_record(teid, packets, reuse_counts[teid])

    def _build_record(
        self, teid: int, packets: list[GTPPacketRecord], reuse_count: int
    ) -> TEIDFeatureRecord:
        packets = sorted(packets, key=lambda r: r.timestamp)
        first = packets[0]
        n = len(packets)

        timestamps = [p.timestamp for p in packets]
        sizes = [p.packet_size for p in packets]
        duration = timestamps[-1] - timestamps[0]

        byte_count = sum(sizes)
        avg_size = statistics.fmean(sizes)
        std_size = statistics.pstdev(sizes) if n > 1 else 0.0

        interarrivals = [t2 - t1 for t1, t2 in zip(timestamps, timestamps[1:], strict=False)]
        interarrival_mean = statistics.fmean(interarrivals) if interarrivals else 0.0
        interarrival_std = statistics.pstdev(interarrivals) if len(interarrivals) > 1 else 0.0

        denom = interarrival_std + interarrival_mean
        burstiness = (interarrival_std - interarrival_mean) / denom if denom > 0 else 0.0

        dst_ips = {p.inner_dst_ip for p in packets if p.inner_dst_ip}
        dst_ports = {p.inner_dst_port for p in packets if p.inner_dst_port is not None}
        fanout_endpoints = {
            (p.inner_dst_ip, p.inner_dst_port)
            for p in packets
            if p.inner_dst_ip and p.inner_dst_port is not None
        }

        size_entropy = shannon_entropy(Counter(sizes))

        uplink_bytes = downlink_bytes = 0
        for p in packets:
            direction = classify_direction(p.inner_src_ip, p.inner_dst_ip)
            if direction == "uplink":
                uplink_bytes += p.packet_size
            elif direction == "downlink":
                downlink_bytes += p.packet_size
        total_directional = uplink_bytes + downlink_bytes
        directionality = uplink_bytes / total_directional if total_directional > 0 else 0.5

        packets_per_s = n / duration if duration > 0 else float(n)
        bytes_per_s = byte_count / duration if duration > 0 else float(byte_count)

        return TEIDFeatureRecord(
            teid=teid,
            base_station=first.base_station,
            capture_file=first.capture_file,
            source_attack_type=first.source_attack_type,
            window_start=timestamps[0],
            window_end=timestamps[-1],
            packet_count=n,
            byte_count=byte_count,
            duration_s=duration,
            packets_per_s=packets_per_s,
            bytes_per_s=bytes_per_s,
            unique_dst_ips=len(dst_ips),
            unique_dst_ports=len(dst_ports),
            avg_packet_size=avg_size,
            std_packet_size=std_size,
            interarrival_mean=interarrival_mean,
            interarrival_std=interarrival_std,
            syn_count=sum(1 for p in packets if p.tcp_syn),
            ack_count=sum(1 for p in packets if p.tcp_ack),
            rst_count=sum(1 for p in packets if p.tcp_rst),
            fin_count=sum(1 for p in packets if p.tcp_fin),
            teid_lifetime_s=duration,
            teid_reuse_count=reuse_count,
            teid_entropy=size_entropy,
            teid_burstiness=burstiness,
            teid_fanout=len(fanout_endpoints),
            teid_directionality=directionality,
        )
