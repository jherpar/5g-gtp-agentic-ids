# agente_5g — Agentic Intrusion Detection System for the 5G User Plane

Master's Thesis (TFM) research codebase comparing a rule-based **agentic**
detection architecture (TEID Agent → PDU Session Agent → Supervisor Agent)
against classical ML (RandomForest, XGBoost) for intrusion detection on the
[5G-NIDD](https://ieee-dataport.org/10203) dataset — built directly from the
**raw** GTP-U pcapng captures rather than the dataset authors' GTP-stripped
Argus/CSV export, in order to exploit TEID/tunnel/PDU-session information
that classical ML on the official dataset never has access to.

See [`architecture.md`](architecture.md) for the system design and
[`experiment_plan.md`](experiment_plan.md) for the research questions and
evaluation methodology. Project conventions for coding agents are in
[`CLAUDE.md`](CLAUDE.md).

## Status

All planned phases are implemented and results are frozen (no further
model tuning or threshold recalibration after Phase 7): GTP-U parsing,
TEID/session feature extraction, multi-level ground-truth labeling,
agentic detectors, four-arm ML baseline comparison, and the RQ1–4
evaluation writeup. See [`experiment_plan.md`](experiment_plan.md) for
findings, limitations, and threats to validity, and
[`architecture.md`](architecture.md) for what was actually built and run
(some scripts/config paths differ from the original plan — documented
there, not glossed over). All Phase 4–7 results are BS1-only; a BS1+BS2
run was scoped out, not attempted silently — see `experiment_plan.md`'s
Limitations section.

## Setup

Requires Python 3.11 and [Poetry](https://python-poetry.org/).

```bash
poetry install
```

`pyshark` (optional PyShark parser backend) requires `tshark`/Wireshark on
PATH and is not installed by default; the primary parser backend (Scapy)
has no such external dependency:

```bash
poetry install --extras pyshark
```

## Data

Place the raw 5G-NIDD captures under `data/raw/BS1/` and `data/raw/BS2/`
(not committed to git — see `.gitignore`), named `<AttackType>_BS{1,2}.pcapng`:
`Goldeneye`, `ICMPflood`, `Slowloris`, `SSH` (benign-only), `SYNflood`,
`SYNScan`, `TCPConnect`, `Torshammer`, `UDPflood`, `UDPScan`.

The dataset authors' own processed export (`data/processed/Combined/Combined.csv`,
`data/processed/Encoded/Encoded.csv`) is used only as the arm A1/A2
baselines in evaluation (see `experiment_plan.md`) — it is never an input
to the GTP-U/agentic pipeline (arms B/C).

## Running the pipeline

The original plan called for `scripts/run_sample_pipeline.py`/
`run_full_pipeline.py` driven by `configs/base.yaml`; in practice the
labeling investigation and ML baselines were built and run with a
sequence of purpose-built scripts instead (see `architecture.md`'s
"Scripts actually used"). To reproduce the committed results
(`experiment_plan.md`'s Reproducibility section has the full list):

```bash
# Labeling ground-truth validation (all 9 BS1 attack types)
poetry run python scripts/validate_labeling_all.py

# Phase 5 agentic detectors against real labeled data
poetry run python scripts/validate_agents.py

# Train/evaluate all four ML baseline arms (A1/A2/B/C)
poetry run python scripts/run_phase6_training.py

# ROC/PR curves, case studies, error analysis, RQ1-4 writeup
poetry run python scripts/run_phase7_analysis.py
```

Each step caches parsed packets under `outputs/cache/packets/`
(gitignored), so re-runs after the first are fast. `outputs/reports/` and
`outputs/figures/` (also gitignored) hold the generated reports/plots;
`experiment_plan.md` inlines every number load-bearing for the reported
findings so the thesis narrative doesn't depend on those artifacts being
present.

## Tests

```bash
poetry run pytest            # 198 tests, ~93% coverage, no real pcap data needed
poetry run ruff check .
poetry run black --check .
poetry run mypy src
```

An `integration` pytest marker and `tests/integration/` package exist for
a future real-file smoke test but none was written — real-data validation
happened through the `scripts/` above instead, run directly against the
raw pcapng files.
