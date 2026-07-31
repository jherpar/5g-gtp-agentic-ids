"""Enums shared by the multi-level labeling pipeline (see preprocessing/labeling.py)."""

from __future__ import annotations

from enum import Enum


class AttackType(str, Enum):
    BENIGN = "Benign"
    ICMP_FLOOD = "ICMPflood"
    UDP_FLOOD = "UDPflood"
    SYN_FLOOD = "SYNflood"
    HTTP_FLOOD = "Goldeneye"
    SLOWLORIS = "Slowloris"
    TORSHAMMER = "Torshammer"
    SYN_SCAN = "SYNScan"
    TCP_CONNECT_SCAN = "TCPConnect"
    UDP_SCAN = "UDPScan"


# Filename token (data/raw/BS{1,2}/<token>_BS{1,2}.pcapng) -> AttackType.
# SSH is the pure-benign capture session and maps to BENIGN directly.
FILENAME_TOKEN_TO_ATTACK_TYPE: dict[str, AttackType] = {
    "SSH": AttackType.BENIGN,
    "ICMPflood": AttackType.ICMP_FLOOD,
    "UDPflood": AttackType.UDP_FLOOD,
    "SYNflood": AttackType.SYN_FLOOD,
    "Goldeneye": AttackType.HTTP_FLOOD,
    "Slowloris": AttackType.SLOWLORIS,
    "Torshammer": AttackType.TORSHAMMER,
    "SYNScan": AttackType.SYN_SCAN,
    "TCPConnect": AttackType.TCP_CONNECT_SCAN,
    "UDPScan": AttackType.UDP_SCAN,
}

# Files that contain no attack window at all (pure benign traffic generation).
BENIGN_ONLY_TOKENS: frozenset[str] = frozenset({"SSH"})


class LabelConfidence(str, Enum):
    """How much independent evidence corroborates a packet/flow's Attack label.

    HIGH   : schedule + victim-IP + traffic-pattern evidence all agree.
    MEDIUM : schedule + victim-IP agree, pattern evidence inconclusive/unused.
    LOW    : only the schedule window matched (plausibly the concurrent
             benign traffic the descriptor paper documents during attacks).
    """

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class LabelSource(str, Enum):
    """Which evidence level(s) contributed to a label (see label_evidence)."""

    SCHEDULE = "SCHEDULE"
    VICTIM_IP = "VICTIM_IP"
    PATTERN = "PATTERN"
