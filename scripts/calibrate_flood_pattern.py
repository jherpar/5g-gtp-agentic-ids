"""One-off calibration helper: distributions of packet-size entropy and
destination-port count for the flood-category attack types, split by
victim-IP corroboration, to pick `max_packet_size_entropy`/
`max_unique_dst_ports` for the Priority-1 fix in
preprocessing/labeling.py::_level3_pattern_matches (see
outputs/reports/confidence_diagnosis/report.md).

Not a permanent pipeline component -- prints candidate values to stdout and
exits; configs/label_patterns.yaml is updated by hand afterward based on
this output.

Usage:
    poetry run python scripts/calibrate_flood_pattern.py
"""

from __future__ import annotations

import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from agente_5g.models.packet import GTPPacketRecord  # noqa: E402
from agente_5g.models.schedule_config import AttackSchedule, LabelPatternsConfig  # noqa: E402
from agente_5g.preprocessing.teid_extractor import shannon_entropy  # noqa: E402
from scripts.validate_labeling import process_file  # noqa: E402

FLOOD_TYPES = ["ICMPflood", "UDPflood", "SYNflood", "Goldeneye"]
MAX_DURATION_S = 2200.0


def _instance_packets_by_teid(packets: list[GTPPacketRecord]) -> dict[int, list[GTPPacketRecord]]:
    by_teid: dict[int, list[GTPPacketRecord]] = defaultdict(list)
    for p in packets:
        if p.is_gtp and p.teid is not None:
            by_teid[p.teid].append(p)
    return by_teid


def _percentiles(values: list[float]) -> str:
    if not values:
        return "n=0"
    sv = sorted(values)

    def pct(p: float) -> float:
        idx = min(len(sv) - 1, int(round(p * (len(sv) - 1))))
        return sv[idx]

    return (
        f"n={len(sv)} min={sv[0]:.3f} p10={pct(0.10):.3f} p25={pct(0.25):.3f} "
        f"median={statistics.median(sv):.3f} p75={pct(0.75):.3f} p90={pct(0.90):.3f} "
        f"max={sv[-1]:.3f}"
    )


def main() -> None:
    schedule = AttackSchedule.load(PROJECT_ROOT / "configs" / "attack_schedule.yaml")
    patterns = LabelPatternsConfig.load(PROJECT_ROOT / "configs" / "label_patterns.yaml")

    all_corroborated_entropy: list[float] = []
    all_corroborated_ports: list[float] = []
    all_not_corroborated_entropy: list[float] = []
    all_not_corroborated_ports: list[float] = []

    for attack_type in FLOOD_TYPES:
        path = PROJECT_ROOT / "data" / "raw" / "BS1" / f"{attack_type}_BS1.pcapng"
        data = process_file(
            path,
            base_station="BS1",
            attack_type=attack_type,
            schedule=schedule,
            patterns=patterns,
            max_duration_s=MAX_DURATION_S,
        )
        packets_by_teid = _instance_packets_by_teid(data["packets"])
        print(f"\n=== {attack_type} ===")
        features = data["features"]
        attack_feats = [f for f in features if f.is_attack]

        corroborated_entropy, corroborated_ports = [], []
        not_corroborated_entropy, not_corroborated_ports = [], []

        for feat in attack_feats:
            instance_packets = [
                p
                for p in packets_by_teid[feat.teid]
                if feat.window_start <= p.timestamp <= feat.window_end
            ]
            if not instance_packets:
                continue
            entropy = shannon_entropy(Counter(p.packet_size for p in instance_packets))
            ports = len(
                {p.inner_dst_port for p in instance_packets if p.inner_dst_port is not None}
            )
            is_corroborated = "VICTIM_IP" in feat.label_evidence
            if is_corroborated:
                corroborated_entropy.append(entropy)
                corroborated_ports.append(ports)
            else:
                not_corroborated_entropy.append(entropy)
                not_corroborated_ports.append(ports)

        print(f"  corroborated entropy:     {_percentiles(corroborated_entropy)}")
        print(f"  not corroborated entropy: {_percentiles(not_corroborated_entropy)}")
        print(f"  corroborated ports:       {_percentiles(corroborated_ports)}")
        print(f"  not corroborated ports:   {_percentiles(not_corroborated_ports)}")

        all_corroborated_entropy.extend(corroborated_entropy)
        all_corroborated_ports.extend(corroborated_ports)
        all_not_corroborated_entropy.extend(not_corroborated_entropy)
        all_not_corroborated_ports.extend(not_corroborated_ports)

    print("\n=== Combined across all 4 flood types ===")
    print(f"  corroborated entropy:     {_percentiles(all_corroborated_entropy)}")
    print(f"  not corroborated entropy: {_percentiles(all_not_corroborated_entropy)}")
    print(f"  corroborated ports:       {_percentiles(all_corroborated_ports)}")
    print(f"  not corroborated ports:   {_percentiles(all_not_corroborated_ports)}")


if __name__ == "__main__":
    main()
