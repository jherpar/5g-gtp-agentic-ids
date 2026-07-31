from __future__ import annotations

from agente_5g.parsers.gtp_layers import decode_gtp_u_header_fallback


def test_fallback_decodes_minimal_8_byte_header():
    # flags=0x30 (v1, PT=1, no E/S/PN), msg_type=255 (G-PDU), length=20, teid=0x11223344
    raw = bytes([0x30, 0xFF, 0x00, 0x14, 0x11, 0x22, 0x33, 0x44]) + b"\x00" * 20
    header = decode_gtp_u_header_fallback(raw)

    assert header is not None
    assert header.version == 1
    assert header.protocol_type == 1
    assert header.message_type == 255
    assert header.length == 20
    assert header.teid == 0x11223344
    assert header.header_len == 8


def test_fallback_detects_extended_header_from_flags():
    # flags=0x32 (v1, PT=1, S bit set) -> 12-byte header (4 extra optional-field octets)
    raw = bytes([0x32, 0x01, 0x00, 0x04, 0xAA, 0xBB, 0xCC, 0xDD]) + b"\x00" * 4
    header = decode_gtp_u_header_fallback(raw)

    assert header is not None
    assert header.teid == 0xAABBCCDD
    assert header.header_len == 12


def test_fallback_returns_none_for_truncated_payload():
    assert decode_gtp_u_header_fallback(b"\x30\xff\x00") is None
    assert decode_gtp_u_header_fallback(b"") is None
