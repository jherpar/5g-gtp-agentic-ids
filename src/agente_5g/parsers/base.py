"""Abstract packet parser interface.

Two backends implement this: `scapy_parser.ScapyPacketParser` (primary —
pure Python, no external binary required) and `pyshark_parser.PySharkPacketParser`
(optional — richer GTP-U dissection, but requires tshark on PATH). Callers
should go through `factory.get_parser()` rather than instantiating a backend
directly, so backend selection stays centralized.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from agente_5g.models.packet import GTPPacketRecord


class PacketParser(ABC):
    """Streams `GTPPacketRecord`s out of a single pcapng capture file."""

    @abstractmethod
    def parse_file(
        self,
        path: Path,
        base_station: Literal["BS1", "BS2"],
        source_attack_type: str,
        max_packets: int | None = None,
        max_duration_s: float | None = None,
    ) -> Iterator[GTPPacketRecord]:
        """Yield one GTPPacketRecord per packet, in capture order.

        Implementations must stream (never load the whole file into memory)
        so 660MB captures don't blow up process memory, and must not raise on
        a single malformed/unsupported packet — log and skip it instead,
        continuing the stream.
        """
        raise NotImplementedError
