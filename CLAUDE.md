# Project Overview

Agentic Intrusion Detection System for 5G User Plane.

Goal:
Compare Agentic AI against traditional ML using
the 5G-NIDD dataset.

Architecture:

1. GTP-U Parser
2. TEID Agent
3. PDU Session Agent
4. Supervisor Agent
5. ML Baseline

See `architecture.md` for the actual module layout and `experiment_plan.md`
for methodology/results. Key conventions, not obvious from the code alone:

- Scapy is the primary parser backend (streaming `PcapReader`, never
  `rdpcap`); PyShark is optional, only auto-selected if `tshark` is on
  PATH.
- Detection is fully deterministic and rule-based (`agents/rules.py`,
  `configs/thresholds.yaml`) — no LLM in the decision path. Local Ollama
  LLM use (`agents/explain.py`) is optional, disabled by default, and
  limited to post-hoc natural-language explanation text only; it never
  feeds back into any score.
- `preprocessing/labeling.py` (ground-truth label quality) and
  `agents/rules.py` (detection) are physically separate files/constants
  by design (`configs/label_patterns.yaml` vs. `configs/thresholds.yaml`),
  enforced by `tests/unit/test_architecture_boundaries.py`, so label
  quality is never validated against the same logic being evaluated as a
  detector.
- Ground-truth labels carry an explicit HIGH/MEDIUM/LOW confidence tier
  (`label_confidence`/`label_evidence` fields), not a single binary label
  — see `experiment_plan.md`'s labeling-methodology section for why, and
  its "three preserved label views" policy that downstream code must
  respect.
- Evaluation compares four arms, not a single model: A1 (official
  Combined.csv, our preprocessing), A2 (official Encoded.csv, authors'
  own encoding, secondary check only), B (GTP-U/session ML), C (agentic).

## Research Questions

- **RQ1**: Can GTP-U TEID features improve intrusion detection performance
  over the official 5G-NIDD flow dataset? *(arm A1/A2 vs. arm B)*
- **RQ2**: Can an agentic architecture achieve comparable or better
  detection performance than traditional ML when using the same GTP-U
  information? *(arm B vs. arm C)*
- **RQ3**: Does TEID-level and session-level reasoning provide improved
  explainability for 5G intrusion detection? *(qualitative — agent
  `reason`/`rule_triggers` output vs. ML feature-importance output, see
  `evaluation/case_studies.py`)*
- **RQ4**: Can attacks be detected earlier by TEID/session reasoning than
  by traditional flow-based ML methods? *(qualitative — `PDUSessionAgent`'s
  state-transition timestamps as an early-warning signal vs. single-shot
  flow classification; no empirical time-to-first-flag measurement was
  implemented)*

Findings for all four are in `experiment_plan.md`'s "Research Questions —
findings" section, frozen as of the Phase 7 commit.

Coding Standards

- Python 3.11+
- Type hints mandatory
- Pydantic models
- pytest coverage >80%
- Ruff
- Black
- MyPy

Research Constraints

- Reproducibility mandatory
- Deterministic seeds
- All experiments logged

Dataset

5G-NIDD

Inputs:

`data/raw/BS{1,2}/<AttackType>_BS{1,2}.pcapng` — 10 attack-type files ×
2 base stations = 20 files (~3GB total): `Goldeneye`, `ICMPflood`,
`Slowloris`, `SSH` (benign-only, not an attack), `SYNflood`, `SYNScan`,
`TCPConnect`, `Torshammer`, `UDPflood`, `UDPScan`. Reported results
(Phase 4–7) use BS1 only — see `experiment_plan.md`'s Limitations.

Also `data/processed/{Combined,Encoded}/*.csv` (the dataset authors' own
processed export, used only for arm A1/A2, never for the GTP-U pipeline).

Primary protocol:

GTP-U

Outputs:

Features
Agent decisions
ML predictions
Evaluation reports

Deliverables

- Research-ready code
- Reproducible pipeline
- Thesis figures
- Comparative evaluation