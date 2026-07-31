"""Multi-level ground-truth labeling with an explicit confidence score.

The descriptor paper states benign traffic keeps flowing *during* attack
windows, so a timestamp window alone cannot safely label an individual
TEID-instance/session -- it only bounds when an attack *could* be present.
This module combines three independent evidence levels into an explicit
confidence tier rather than a silent binary label:

  Level 1 -- Attack Schedule (`configs/attack_schedule.yaml`, Tables III/IV
    of the descriptor paper). The Phase 1 calibration run found the
    published per-minute clock windows drift by several minutes from the
    real capture timestamps (planned vs. actual execution), so this module
    does NOT trust the schedule's literal wall-clock cutoffs. Instead it
    computes the attack sub-window as a *fraction* of the schedule's own
    session duration (`(attack_start - session_start) / session_duration`),
    then applies that fraction to the FILE's own observed
    [first_ts, last_ts] span -- preserving the schedule's relative
    structure (e.g. "attack starts 1/6 of the way into the session") while
    sidestepping the unreliable absolute clock alignment.
  Level 2 -- Victim IP (`schedule.victim_ip`, empirically confirmed
    identical across BS1/BS2 SYN Scan captures during calibration): does
    this instance's traffic touch the shared victim/MEC server?
  Level 3 -- Traffic pattern validation (`configs/label_patterns.yaml`):
    a coarse, attack-type-specific structural check (sustained high rate
    for floods, port fan-out for scans, many long-lived low-throughput
    connections for slow-rate attacks) computed with thresholds and code
    entirely separate from `agents/rules.py`'s detection thresholds, so
    label quality is never validated using the same logic being evaluated
    against these labels.

Confidence:
  HIGH   -- Level 1 + Level 2 + Level 3 all agree the instance is an attack.
  MEDIUM -- Level 1 + Level 2 agree, Level 3 inconclusive/didn't fire.
  LOW    -- only Level 1 fired (plausibly the concurrent benign traffic the
            descriptor paper documents), OR the instance falls outside the
            approximate attack sub-window yet Level 2/3 unexpectedly fired
            (ambiguous, given the schedule's known imprecision).
An instance outside the approximate attack sub-window with no Level 2/3
signal either is labeled Benign at HIGH confidence (unremarkable background
traffic). Files with no attack schedule row (`SSH_BS{1,2}.pcapng`) are
always Benign at HIGH confidence.

Level 1 alone is NEVER sufficient to call something an attack -- it only
identifies *when an attack could plausibly appear*; Level 2/3 must
corroborate before MEDIUM/HIGH confidence is assigned. Instances outside the
approximate attack window are always labeled Benign, regardless of Level 2/3
evidence (surprising corroboration there just lowers confidence in that
Benign label rather than flipping it to Attack), since a single fixed
victim IP or a coarse pattern check on its own is far too weak a signal to
overturn the schedule's structural evidence about *where in the file* an
attack was staged.

KNOWN LIMITATION, verified against real data: for the port-scan attack types
(SYNScan/TCPConnect/UDPScan), Table III of the descriptor paper gives no
separate "collection period" distinct from the attack window (unlike Table
IV's DoS attacks) -- `session_window` and `attack_window` are identical, so
Level 1 fires for effectively the entire file. In practice this means every
instance in a scan-type file is labeled `is_attack=True`, and it's the
confidence tier (mostly LOW, since ordinary background traffic won't
corroborate via Level 2/3) rather than the label itself that separates real
scan traffic from routine background traffic during that session. Consumers
that need a cleaner binary signal for scan-type files should filter to
MEDIUM/HIGH confidence rather than trusting `is_attack` alone. DoS-type
files (ICMPflood/UDPflood/SYNflood/...) don't have this issue, since their
schedule rows include a genuinely wider session window around a narrower
attack sub-window.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from typing import Literal

from agente_5g.models.labels import (
    BENIGN_ONLY_TOKENS,
    FILENAME_TOKEN_TO_ATTACK_TYPE,
    LabelConfidence,
    LabelSource,
)
from agente_5g.models.packet import GTPPacketRecord
from agente_5g.models.schedule_config import AttackSchedule, LabelPatternsConfig
from agente_5g.models.session import PDUSessionRecord
from agente_5g.models.teid_features import TEIDFeatureRecord
from agente_5g.preprocessing.teid_extractor import shannon_entropy


def _hhmm_to_minutes(value: str) -> float:
    hours, minutes = value.split(":")
    return int(hours) * 60 + int(minutes)


def _approximate_attack_subwindow(
    schedule: AttackSchedule,
    source_attack_type: str,
    base_station: Literal["BS1", "BS2"],
    file_first_ts: float,
    file_last_ts: float,
) -> tuple[float, float] | None:
    """Map the schedule's *relative* attack position onto this file's own
    observed timestamp span. Returns None if this attack type has no
    schedule row, is benign-only, or the schedule data is malformed."""
    entry = schedule.attacks.get(source_attack_type)
    if (
        entry is None
        or entry.benign_only
        or entry.session_window is None
        or entry.attack_window is None
    ):
        return None

    session_start = _hhmm_to_minutes(entry.session_window[0])
    session_end = _hhmm_to_minutes(entry.session_window[1])
    session_span = session_end - session_start
    if session_span <= 0:
        return None

    bs_window = entry.attack_window.get(base_station)
    if bs_window is None:
        return None
    attack_start = _hhmm_to_minutes(bs_window[0])
    attack_end = _hhmm_to_minutes(bs_window[1])

    rel_start = max(0.0, min(1.0, (attack_start - session_start) / session_span))
    rel_end = max(0.0, min(1.0, (attack_end - session_start) / session_span))

    file_span = file_last_ts - file_first_ts
    return (
        file_first_ts + rel_start * file_span,
        file_first_ts + rel_end * file_span,
    )


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return a_start <= b_end and a_end >= b_start


_PATTERN_CATEGORY: dict[str, str] = {
    "ICMPflood": "flood",
    "UDPflood": "flood",
    "SYNflood": "flood",
    "Goldeneye": "flood",
    "Slowloris": "slowrate",
    "Torshammer": "slowrate",
    "SYNScan": "scan",
    "TCPConnect": "scan",
    "UDPScan": "scan",
}


def _level3_pattern_matches(
    source_attack_type: str,
    packets: list[GTPPacketRecord],
    patterns: LabelPatternsConfig,
) -> bool:
    category = _PATTERN_CATEGORY.get(source_attack_type)
    if category is None or not packets:
        return False

    timestamps = [p.timestamp for p in packets]
    duration = max(timestamps) - min(timestamps)
    n = len(packets)

    if category == "flood":
        flood_cfg = patterns.flood_pattern
        if duration < flood_cfg.min_window_s:
            return False
        rate = n / duration if duration > 0 else float(n)
        if rate < flood_cfg.min_sustained_packets_per_s:
            return False
        # Rate alone barely discriminates real floods from ordinary sustained
        # traffic: the confidence diagnosis (outputs/reports/confidence_diagnosis)
        # found packets that clear a 0.9 pkt/s floor ranging up to ~13,000 pkt/s
        # among traffic that never touches the victim IP -- nearly any real
        # conversation qualifies. Floods are also characteristically uniform
        # (near-identical packet sizes -> low entropy) and concentrated on very
        # few destination ports, unlike an ordinary multi-service conversation.
        size_entropy = shannon_entropy(Counter(p.packet_size for p in packets))
        dst_ports = {p.inner_dst_port for p in packets if p.inner_dst_port is not None}
        return (
            size_entropy <= flood_cfg.max_packet_size_entropy
            and len(dst_ports) <= flood_cfg.max_unique_dst_ports
        )

    if category == "scan":
        scan_cfg = patterns.scan_pattern
        ports_by_src: dict[str, set[int]] = defaultdict(set)
        for p in packets:
            if p.inner_src_ip and p.inner_dst_port is not None:
                ports_by_src[p.inner_src_ip].add(p.inner_dst_port)
        return any(
            len(ports) >= scan_cfg.min_unique_dst_ports_per_source
            for ports in ports_by_src.values()
        )

    if category == "slowrate":
        slowrate_cfg = patterns.slowrate_pattern
        flows: dict[tuple[str | None, str | None, int | None, int | None], list[float]] = (
            defaultdict(list)
        )
        for p in packets:
            key = (p.inner_src_ip, p.inner_dst_ip, p.inner_src_port, p.inner_dst_port)
            flows[key].append(p.timestamp)
        long_lived = [
            ts
            for ts in flows.values()
            if (max(ts) - min(ts)) >= slowrate_cfg.min_connection_duration_s
        ]
        bytes_per_s = sum(p.packet_size for p in packets) / duration if duration > 0 else float(n)
        return (
            len(long_lived) >= slowrate_cfg.min_concurrent_connections
            and bytes_per_s <= slowrate_cfg.max_bytes_per_s
        )

    return False


def _classify(
    *,
    source_attack_type: str,
    base_station: Literal["BS1", "BS2"],
    instance_start: float,
    instance_end: float,
    instance_packets: list[GTPPacketRecord],
    schedule: AttackSchedule,
    patterns: LabelPatternsConfig,
    file_first_ts: float,
    file_last_ts: float,
) -> tuple[str, bool, LabelConfidence, list[str]]:
    if source_attack_type in BENIGN_ONLY_TOKENS:
        return "Benign", False, LabelConfidence.HIGH, []

    subwindow = _approximate_attack_subwindow(
        schedule, source_attack_type, base_station, file_first_ts, file_last_ts
    )
    level1 = subwindow is not None and _overlaps(instance_start, instance_end, *subwindow)

    victim_ip = schedule.victim_ip
    level2 = any(
        p.inner_src_ip == victim_ip or p.inner_dst_ip == victim_ip for p in instance_packets
    )

    level3 = _level3_pattern_matches(source_attack_type, instance_packets, patterns)

    evidence: list[str] = []
    if level1:
        evidence.append(LabelSource.SCHEDULE.value)
    if level2:
        evidence.append(LabelSource.VICTIM_IP.value)
    if level3:
        evidence.append(LabelSource.PATTERN.value)

    if not level1:
        confidence = LabelConfidence.LOW if (level2 or level3) else LabelConfidence.HIGH
        return "Benign", False, confidence, evidence

    attack_type = FILENAME_TOKEN_TO_ATTACK_TYPE.get(source_attack_type)
    label = attack_type.value if attack_type is not None else source_attack_type
    if level2 and level3:
        confidence = LabelConfidence.HIGH
    elif level2:
        confidence = LabelConfidence.MEDIUM
    else:
        confidence = LabelConfidence.LOW
    return label, True, confidence, evidence


def label_teid_features(
    features: Iterable[TEIDFeatureRecord],
    file_packets: list[GTPPacketRecord],
    schedule: AttackSchedule,
    patterns: LabelPatternsConfig,
) -> Iterator[TEIDFeatureRecord]:
    """Attach labels to already-built TEIDFeatureRecords.

    `file_packets` must be the FULL parsed packet stream for the capture
    file these features came from (used both to recover each instance's own
    packets by TEID + window bounds, and to establish the file's own
    [first_ts, last_ts] span for Level 1's relative-window mapping).
    """
    if not file_packets:
        return

    file_first_ts = min(p.timestamp for p in file_packets)
    file_last_ts = max(p.timestamp for p in file_packets)

    packets_by_teid: dict[int, list[GTPPacketRecord]] = defaultdict(list)
    for p in file_packets:
        if p.is_gtp and p.teid is not None:
            packets_by_teid[p.teid].append(p)

    for feat in features:
        instance_packets = [
            p
            for p in packets_by_teid[feat.teid]
            if feat.window_start <= p.timestamp <= feat.window_end
        ]
        label, is_attack, confidence, evidence = _classify(
            source_attack_type=feat.source_attack_type,
            base_station=feat.base_station,
            instance_start=feat.window_start,
            instance_end=feat.window_end,
            instance_packets=instance_packets,
            schedule=schedule,
            patterns=patterns,
            file_first_ts=file_first_ts,
            file_last_ts=file_last_ts,
        )
        yield feat.model_copy(
            update={
                "label": label,
                "is_attack": is_attack,
                "label_confidence": confidence,
                "label_evidence": evidence,
            }
        )


def label_sessions(
    sessions: Iterable[PDUSessionRecord],
    file_packets: list[GTPPacketRecord],
    source_attack_type: str,
    base_station: Literal["BS1", "BS2"],
    schedule: AttackSchedule,
    patterns: LabelPatternsConfig,
) -> Iterator[PDUSessionRecord]:
    """Attach labels to already-built PDUSessionRecords.

    Unlike TEID features, session records don't carry `source_attack_type`/
    `base_station` directly (they're keyed on ue_ip/teid/window), so callers
    pass them explicitly -- one call should cover all sessions built from a
    single capture file.
    """
    if not file_packets:
        return

    file_first_ts = min(p.timestamp for p in file_packets)
    file_last_ts = max(p.timestamp for p in file_packets)

    packets_by_teid: dict[int, list[GTPPacketRecord]] = defaultdict(list)
    for p in file_packets:
        if p.is_gtp and p.teid is not None:
            packets_by_teid[p.teid].append(p)

    for session in sessions:
        instance_packets = [
            p
            for p in packets_by_teid[session.teid]
            if session.start_time <= p.timestamp <= session.end_time
        ]
        label, is_attack, confidence, evidence = _classify(
            source_attack_type=source_attack_type,
            base_station=base_station,
            instance_start=session.start_time,
            instance_end=session.end_time,
            instance_packets=instance_packets,
            schedule=schedule,
            patterns=patterns,
            file_first_ts=file_first_ts,
            file_last_ts=file_last_ts,
        )
        yield session.model_copy(
            update={
                "label": label,
                "is_attack": is_attack,
                "label_confidence": confidence,
                "label_evidence": evidence,
            }
        )
