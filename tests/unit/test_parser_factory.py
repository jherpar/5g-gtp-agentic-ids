from __future__ import annotations

from agente_5g.parsers.factory import get_parser
from agente_5g.parsers.scapy_parser import ScapyPacketParser


def test_scapy_backend_explicit():
    assert isinstance(get_parser("scapy"), ScapyPacketParser)


def test_auto_backend_falls_back_to_scapy_when_no_tshark(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _name: None)
    assert isinstance(get_parser("auto"), ScapyPacketParser)
