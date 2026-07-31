"""Scapy GTP-U layer wiring + a manual fallback decoder.

Scapy's `contrib.gtp` module dissects GTP-U reasonably well but is not
guaranteed to cover every extension header (e.g. the PDU Session Container
carrying QFI) or malformed/truncated packets seen in real captures. Rather
than let a dissection gap crash the whole parse of a 660MB file, every
caller should: try scapy's layer first, and on failure/`is None` fall back to
`decode_gtp_u_header_fallback`, which only decodes the fixed 8-byte GTP-U
header (enough to recover TEID and message type) directly from raw bytes.

GTP-U header layout (3GPP TS 29.281 sec. 5.1):
  octet 1   : Version(3) | PT(1) | Spare(1) | E(1) | S(1) | PN(1)
  octet 2   : Message Type
  octets 3-4: Length (payload length, excludes this 8-byte mandatory header)
  octets 5-8: TEID
  if E|S|PN set: 4 more octets (Seq Number(2), N-PDU Number(1), Next Ext Hdr Type(1))
"""

from __future__ import annotations

from dataclasses import dataclass

from agente_5g.utils.logging import get_logger

logger = get_logger(__name__)

GTP_U_PORT = 2152

_gtp_contrib_loaded = False


def ensure_gtp_contrib_loaded() -> bool:
    """Load scapy's GTP contrib layer and bind it to UDP/2152, once.

    Returns True if scapy's native GTP dissection is available, False if the
    contrib module couldn't be imported (callers should then rely on the
    manual fallback decoder for every packet).
    """
    global _gtp_contrib_loaded
    if _gtp_contrib_loaded:
        return True
    try:
        from scapy.config import conf
        from scapy.layers.inet import UDP
        from scapy.layers.l2 import bind_layers

        conf.load_layers.append("gtp") if "gtp" not in conf.load_layers else None
        from scapy.contrib import gtp  # noqa: F401  (import triggers layer registration)

        bind_layers(UDP, gtp.GTP_U_Header, dport=GTP_U_PORT)
        bind_layers(UDP, gtp.GTP_U_Header, sport=GTP_U_PORT)
        _gtp_contrib_loaded = True
        return True
    except Exception:  # pragma: no cover - exercised only if scapy build lacks contrib
        logger.warning("scapy.contrib.gtp unavailable; falling back to manual GTP-U decoding")
        return False


@dataclass(frozen=True)
class GtpUHeader:
    version: int
    protocol_type: int
    message_type: int
    length: int
    teid: int
    header_len: int  # total bytes consumed by the GTP-U header (8 or 12+)


def decode_gtp_u_header_fallback(payload: bytes) -> GtpUHeader | None:
    """Manually decode the fixed GTP-U header from raw UDP payload bytes.

    Returns None if `payload` is too short to contain even the mandatory
    8-byte header (a truncated/malformed packet) — callers should log and
    skip such packets rather than treat this as fatal.
    """
    if len(payload) < 8:
        return None

    first_octet = payload[0]
    version = (first_octet >> 5) & 0b111
    protocol_type = (first_octet >> 4) & 0b1
    has_optional_fields = bool(first_octet & 0b0000_0111)  # E | S | PN

    message_type = payload[1]
    length = int.from_bytes(payload[2:4], "big")
    teid = int.from_bytes(payload[4:8], "big")

    header_len = 8
    if has_optional_fields:
        header_len = 12

    return GtpUHeader(
        version=version,
        protocol_type=protocol_type,
        message_type=message_type,
        length=length,
        teid=teid,
        header_len=header_len,
    )
