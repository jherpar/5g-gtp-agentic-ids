"""Backend selection for PacketParser.

Scapy is always used unless the caller explicitly asks for PyShark AND
`tshark` is actually on PATH — this environment doesn't have tshark
installed, so the default must keep working without it.
"""

from __future__ import annotations

import shutil
from typing import Literal

from agente_5g.parsers.base import PacketParser
from agente_5g.utils.logging import get_logger

logger = get_logger(__name__)


def get_parser(backend: Literal["scapy", "pyshark", "auto"] = "scapy") -> PacketParser:
    if backend == "scapy":
        from agente_5g.parsers.scapy_parser import ScapyPacketParser

        return ScapyPacketParser()

    if backend == "pyshark":
        from agente_5g.parsers.pyshark_parser import PySharkPacketParser

        return PySharkPacketParser()

    # "auto": prefer pyshark only if tshark is actually available, else scapy.
    if shutil.which("tshark") is not None:
        try:
            from agente_5g.parsers.pyshark_parser import PySharkPacketParser

            return PySharkPacketParser()
        except RuntimeError:
            logger.info("tshark found but pyshark not installed; using scapy backend")

    from agente_5g.parsers.scapy_parser import ScapyPacketParser

    return ScapyPacketParser()
