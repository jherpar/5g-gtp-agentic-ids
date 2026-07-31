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

Early scaffolding — pipeline stages are being implemented incrementally
(parsers → features → agents → ML baselines → evaluation). See task list /
git history for current progress.

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
`data/processed/Encoded/Encoded.csv`) is used only as the arm-A baseline in
evaluation (see `experiment_plan.md`) — it is never a pipeline input.

## Running the pipeline

```bash
# Fast dev iteration over a small sample of packets/files
poetry run python scripts/run_sample_pipeline.py

# Full thesis-scale run over all 20 raw pcapng files (~3GB, long-running,
# checkpointed/resumable — intended to run in the background)
poetry run python scripts/run_full_pipeline.py
```

## Tests

```bash
poetry run pytest            # fast unit suite (excludes @pytest.mark.integration)
poetry run pytest -m integration   # also exercises one small real pcapng file
poetry run ruff check .
poetry run black --check .
poetry run mypy src
```
