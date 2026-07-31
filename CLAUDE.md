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

BS1.pcapng
BS2.pcapng

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