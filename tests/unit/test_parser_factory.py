from __future__ import annotations

import pytest

from agente_5g.parsers.factory import get_parser
from agente_5g.parsers.scapy_parser import ScapyPacketParser


def test_scapy_backend_explicit():
    assert isinstance(get_parser("scapy"), ScapyPacketParser)


def test_auto_backend_falls_back_to_scapy_when_no_tshark(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _name: None)
    assert isinstance(get_parser("auto"), ScapyPacketParser)


def test_pyshark_backend_raises_when_pyshark_not_installed():
    # This environment genuinely has no pyshark installed (it's an optional
    # extra) -- get_parser("pyshark") should surface that as a clear error
    # rather than a confusing ImportError deep in pyshark_parser.py.
    with pytest.raises(RuntimeError, match="pyshark is not installed"):
        get_parser("pyshark")


def test_auto_backend_falls_back_to_scapy_when_tshark_found_but_pyshark_missing(monkeypatch):
    # tshark present on PATH, but pyshark itself still isn't installed --
    # factory should catch the RuntimeError and degrade to scapy rather
    # than propagating it.
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/tshark")
    assert isinstance(get_parser("auto"), ScapyPacketParser)
