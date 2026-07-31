"""Pure, unit-testable detection rules used by TEIDAgent and PDUSessionAgent.

Every rule returns a `RuleResult(triggered, intensity, detail)`:
  - `triggered`: whether the rule fired at all.
  - `intensity`: how far past the threshold, clipped to [0, 1] -- 0.0 when
    not triggered, up to 1.0 the further the observation exceeds the
    threshold. Agents take the max intensity across triggered rules as
    their risk_score (see teid_agent.py/pdu_session_agent.py), so a
    barely-over-threshold case and a wildly-over-threshold case are
    distinguishable rather than both just reading 1.0.
  - `detail`: a short human-readable fragment (measured values), used to
    build the agent's `reason` string.

These thresholds (`configs/thresholds.yaml`, loaded via
`models/agent_thresholds.py`) are deliberately a separate file and separate
code path from `preprocessing/labeling.py`'s Level-3 pattern-validation
thresholds (`configs/label_patterns.yaml`) -- label quality must never be
validated using the same logic being evaluated as a detector.
"""

from __future__ import annotations

from typing import NamedTuple

from agente_5g.models.agent_thresholds import (
    FloodRuleConfig,
    ScanRuleConfig,
    SynFloodRuleConfig,
)
from agente_5g.models.teid_features import TEIDFeatureRecord


class RuleResult(NamedTuple):
    triggered: bool
    intensity: float
    detail: str


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _over_threshold_intensity(value: float, threshold: float) -> float:
    """Intensity for a "higher is worse" rule, given it already triggered
    (value >= threshold).

    A naive `value / threshold` always lands in [1.0, inf) the moment the
    rule triggers -- by definition triggering requires value >= threshold,
    so the ratio can never read below 1.0 and every triggered case clips to
    the same 1.0, discarding exactly the gradation `intensity` exists to
    capture. Dividing by `2 * threshold` instead reads 0.5 right at the
    threshold (a bare pass) and saturates to 1.0 at 2x the threshold (a
    clearly severe case), giving triggered cases a real [0.5, 1.0] spread.
    """
    if threshold <= 0:
        return 0.0
    return _clip01(value / (2 * threshold))


def flood_rule(feat: TEIDFeatureRecord, cfg: FloodRuleConfig) -> RuleResult:
    """Sustained, near-uniform-size traffic concentrated on very few
    destination ports -- generic signature for ICMP/UDP/SYN-style floods
    regardless of protocol (packet rate + uniformity is the signal)."""
    triggered = (
        feat.packets_per_s >= cfg.min_packets_per_s
        and feat.teid_entropy <= cfg.max_teid_entropy
        and feat.unique_dst_ports <= cfg.max_unique_dst_ports
    )
    intensity = (
        _over_threshold_intensity(feat.packets_per_s, cfg.min_packets_per_s) if triggered else 0.0
    )
    detail = (
        f"packets_per_s={feat.packets_per_s:.2f} (>= {cfg.min_packets_per_s}), "
        f"teid_entropy={feat.teid_entropy:.2f} (<= {cfg.max_teid_entropy}), "
        f"unique_dst_ports={feat.unique_dst_ports} (<= {cfg.max_unique_dst_ports})"
    )
    return RuleResult(triggered, intensity, detail)


def syn_flood_rule(feat: TEIDFeatureRecord, cfg: SynFloodRuleConfig) -> RuleResult:
    """High SYN volume with a low ACK-to-SYN ratio: many half-open
    connections, the classic SYN-flood signature."""
    ack_to_syn_ratio = feat.ack_count / feat.syn_count if feat.syn_count > 0 else 0.0
    triggered = feat.syn_count >= cfg.min_syn_count and ack_to_syn_ratio <= cfg.max_ack_to_syn_ratio
    intensity = _over_threshold_intensity(feat.syn_count, cfg.min_syn_count) if triggered else 0.0
    detail = (
        f"syn_count={feat.syn_count} (>= {cfg.min_syn_count}), "
        f"ack/syn={ack_to_syn_ratio:.3f} (<= {cfg.max_ack_to_syn_ratio})"
    )
    return RuleResult(triggered, intensity, detail)


def scan_rule(feat: TEIDFeatureRecord, cfg: ScanRuleConfig) -> RuleResult:
    """Many distinct destination ports touched with few packets per port:
    a port scan rather than a sustained conversation."""
    packets_per_port = (
        feat.packet_count / feat.unique_dst_ports if feat.unique_dst_ports > 0 else 0.0
    )
    triggered = (
        feat.unique_dst_ports >= cfg.min_unique_dst_ports
        and packets_per_port <= cfg.max_packets_per_dst_port
    )
    intensity = (
        _over_threshold_intensity(feat.unique_dst_ports, cfg.min_unique_dst_ports)
        if triggered
        else 0.0
    )
    detail = (
        f"unique_dst_ports={feat.unique_dst_ports} (>= {cfg.min_unique_dst_ports}), "
        f"packets_per_port={packets_per_port:.2f} (<= {cfg.max_packets_per_dst_port})"
    )
    return RuleResult(triggered, intensity, detail)


def high_state_transition_rule(state_transition_rate: float, threshold: float) -> RuleResult:
    """Rapid protocol-state churn (SYN/ACK/FIN/RST flapping) within a
    session window -- unstable, chaotic traffic rather than a normal flow."""
    triggered = state_transition_rate >= threshold
    intensity = _over_threshold_intensity(state_transition_rate, threshold) if triggered else 0.0
    detail = f"state_transition_rate={state_transition_rate:.2f}/s (>= {threshold})"
    return RuleResult(triggered, intensity, detail)


def low_temporal_entropy_rule(temporal_entropy: float, threshold: float) -> RuleResult:
    """Traffic concentrated in a few sub-bins of the window rather than
    spread evenly -- bursty, flood-like arrival pattern."""
    triggered = temporal_entropy <= threshold
    intensity = _clip01(1.0 - temporal_entropy / threshold) if triggered and threshold > 0 else 0.0
    detail = f"temporal_entropy={temporal_entropy:.2f} bits (<= {threshold})"
    return RuleResult(triggered, intensity, detail)


def high_diversity_rule(
    port_diversity: int, destination_diversity: int, threshold: int
) -> RuleResult:
    """Session-level scan signature: many distinct ports/destinations
    touched within one short window."""
    peak = max(port_diversity, destination_diversity)
    triggered = peak >= threshold
    intensity = _over_threshold_intensity(peak, threshold) if triggered else 0.0
    detail = (
        f"port_diversity={port_diversity}, destination_diversity={destination_diversity} "
        f"(peak {peak} >= {threshold})"
    )
    return RuleResult(triggered, intensity, detail)
