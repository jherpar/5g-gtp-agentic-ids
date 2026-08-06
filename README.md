# 5G-GTP Agentic IDS

**An Agentic Intrusion Detection Architecture for the 5G User Plane**

Master's Thesis (TFM) research codebase comparing a rule-based **agentic**
detection architecture (TEID Agent → PDU Session Agent → Supervisor Agent)
against classical ML (RandomForest, XGBoost) for intrusion detection on the
[5G-NIDD](https://ieee-dataport.org/10203) dataset — built directly from the
**raw** GTP-U pcapng captures rather than the dataset authors' GTP-stripped
Argus/CSV export, in order to exploit TEID/tunnel/PDU-session information
that classical ML on the official dataset never has access to.

**Status: experimental phase closed.** All reported results are frozen (no
further model tuning, retraining, threshold recalibration, or labeling
changes after Phase 7 + the RQ4/ground-truth-validation follow-ups). This
repository is published for transparency, reproducibility, and reference
during thesis defense — see [`CONTRIBUTING.md`](CONTRIBUTING.md) for what
that means for contributions.

## Project overview

5G-NIDD is normally consumed as a GTP-stripped flow export (Argus-derived
`Combined.csv`/`Encoded.csv`, no IP/port columns) — the dataset as most
classical-ML intrusion-detection work sees it. This project instead builds
a pipeline directly from the raw pcapng captures, preserving the GTP-U
tunnel layer, to ask whether **TEID/PDU-session-level information the
official dataset discards** helps detection, and whether a deterministic,
rule-based **agentic** architecture reasoning over that information is
competitive with classical ML trained on the same features.

```
raw pcapng → GTP-U parser → TEID/session features → multi-level labeling
    → agentic detectors (TEID Agent → PDU Session Agent → Supervisor Agent)
    → four-arm comparative ML evaluation
```

See [`docs/architecture.md`](docs/architecture.md) for the full system
design and [`docs/experiment_plan.md`](docs/experiment_plan.md) for
methodology, all findings, limitations, and threats to validity — this
README summarizes both.

## Research questions

- **RQ1**: Can GTP-U/TEID features improve detection over the official
  5G-NIDD flow dataset?
- **RQ2**: Can an agentic architecture match or exceed traditional ML given
  identical GTP-U information?
- **RQ3**: Does TEID/session-level reasoning improve explainability?
- **RQ4**: Can attacks be detected earlier via TEID/session reasoning than
  by traditional flow-based ML?

## System architecture

| Stage | Module | Role |
|---|---|---|
| Parsing | `src/agente_5g/parsers/` | Streaming Scapy GTP-U packet parser (primary backend; optional PyShark backend, gated on `tshark`) |
| Feature extraction | `src/agente_5g/preprocessing/` | Per-TEID-instance and per-session (`ue_ip`, `teid`, window) engineered features |
| Ground-truth labeling | `src/agente_5g/preprocessing/labeling.py` | 3-level evidence model (schedule / victim-IP / traffic pattern) → HIGH/MEDIUM/LOW confidence tiers, never a silent binary label |
| Agentic detection (arm C) | `src/agente_5g/agents/` | Deterministic, rule-based `TEIDAgent` → `PDUSessionAgent` (NORMAL→WATCH→SUSPICIOUS→ATTACK state machine) → `SupervisorAgent` fusion — **no LLM in the decision path** |
| ML baselines (arms A1/A2/B) | `src/agente_5g/ml/` | RandomForest/XGBoost on the official CSV (A1, A2) and on GTP-U session features (B) |
| Evaluation | `src/agente_5g/evaluation/` | Shared metrics, ROC/PR curves, per-attack-type case studies, error analysis |

Detection is fully deterministic (`configs/thresholds.yaml`). An optional
local-LLM post-hoc explanation component (`agents/explain.py`) exists and
is unit-tested but was **never activated** in producing any reported
result — see `docs/architecture.md`.

## Experimental design

Four comparison arms, not one model vs. one baseline:

| Arm | Data | What it tests |
|---|---|---|
| **A1** (primary official baseline) | `Combined.csv`, our own documented preprocessing | Classical ML, no GTP-U info, clean/interpretable features |
| **A2** (secondary reproducibility check) | `Encoded.csv`, authors' own pre-encoded columns near-verbatim | Same, using the dataset authors' own (partly uninterpretable) encoding |
| **B** | Real labeled GTP-U/TEID/session features | Classical ML *with* GTP-U info |
| **C** | Same features as B | The agentic system (untrained, rule-based) |

Ground truth carries an explicit **HIGH/MEDIUM/LOW confidence tier**, not
a single binary label (only ~14.5% of TEID-level / ~12.5% of session-level
attack-window traffic is independently corroborated — a property of the
dataset's concurrent-benign-traffic design, not a labeling defect; see
`docs/experiment_plan.md`). Arms B/C are evaluated under both the full
population (view A) and the HIGH+MEDIUM-only subset (view B).

## Main findings

Full tables, per-attack-type breakdowns, and the complete RQ1–4 discussion
are in [`docs/experiment_plan.md`](docs/experiment_plan.md#research-questions--findings-phase-67-frozen).
Headline results:

| Arm | Model | F1 | ROC-AUC |
|---|---|---|---|
| A1 (Combined, ours) | XGBoost | 0.963 | 0.994 |
| A2 (Encoded, authors') | XGBoost | 0.963 | 0.994 |
| B (GTP-ML), view B | XGBoost | 0.227 | 0.972 |
| C (Agentic), view B | — | 0.208 | 0.947 |

- **RQ1 — no** on raw metrics for this dataset as constructed, but the
  ROC-AUC gap is far smaller than the F1 gap: arm B's models discriminate
  about as well as arm A's once ranking is separated from an uncalibrated
  fixed threshold on a ~20x smaller, noisier training set.
- **RQ2 — a genuine trade-off, not a win/loss**: the agentic system has
  dramatically better precision/FPR than arm B at their respective
  operating points, but far worse recall. By ROC-AUC, arm C's ranking
  ability is competitive with arm B's trained models.
- **RQ3 — qualitatively yes**: the agentic system gives a concrete,
  per-decision, human-readable explanation; classical ML offers only
  global feature importances.
- **RQ4 — measured empirically, and the result is negative**: a
  post-Phase-7 detection-latency study found the ML classifier typically
  detects *before* the agent's own WATCH/SUSPICIOUS/ATTACK escalation, not
  after — reported as found, not softened. See `docs/experiment_plan.md`'s
  "RQ4 follow-up" subsection.
- **Appendix**: an independent ground-truth validation against a
  previously-unused author-provided per-flow CSV (real IP addresses)
  confirms the HIGH+MEDIUM confidence subset 94–100% of the time and LOW
  confidence 0% of the time, across all 9 attack types — external support
  for the confidence-tier design. See
  [`docs/appendix_ground_truth_validation.md`](docs/appendix_ground_truth_validation.md).

## Reproducibility

```bash
poetry install
poetry run pytest        # 198 tests, ~93% coverage, no real data needed
```

Full environment setup (Poetry / pip / conda), dataset placement, and the
exact script sequence to regenerate every reported result are in
[`reproducibility/run_order.md`](reproducibility/run_order.md). Minimal
path to reproduce Phase 6/7 specifically:

```bash
poetry run python scripts/run_phase6_training.py   # primary results: outputs/reports/phase6_training/results.csv
poetry run python scripts/run_phase7_analysis.py    # ROC/PR curves, RQ1-4 writeup; bit-for-bit verifies Phase 6
```

## Repository structure

```
README.md                  This file
LICENSE                     MIT (code only — see data/dataset_instructions.md for the dataset's own terms)
CONTRIBUTING.md              Contribution policy (experimental phase is closed)
CITATION.cff                   Machine-readable citation metadata

docs/
├── README.md                    Docs index
├── architecture.md               System design by module
├── experiment_plan.md             Methodology, RQ1-4 findings, limitations, threats to validity
└── appendix_ground_truth_validation.md   Standalone external label-validation study

src/agente_5g/               Library code (parsers, preprocessing, agents, ml, evaluation, models)
configs/                       Frozen YAML configuration (thresholds, label patterns, attack schedule)
scripts/                        Executable analysis/training/reporting entry points
tests/                            pytest suite (198 tests, ~93% coverage, synthetic data only)

data/
├── README.md                   Directory layout (not committed — large binaries)
└── dataset_instructions.md      How to obtain/place the 5G-NIDD dataset

outputs/                     Regenerated reports/figures (gitignored, not committed)
├── figures/
└── reports/

reproducibility/
├── environment.yml            Conda environment (convenience alternative to Poetry)
├── requirements.txt            Pip requirements, exact versions from poetry.lock
└── run_order.md                 Full setup + exact script execution order
```

## Limitations

Full discussion (BS1-only scope, small/noisy arm B/C training data,
mismatched units of analysis across arms, unmatched decision thresholds,
an unexplained inverted-entropy rule finding, a structural scan-type
labeling limitation) is in
[`docs/experiment_plan.md`'s Limitations section](docs/experiment_plan.md#limitations)
and [Threats to Validity section](docs/experiment_plan.md#threats-to-validity).
Briefly: all reported results use BS1 only (20 raw captures exist across
BS1+BS2; a full run was scoped out, not silently skipped), and arm B/C's
training data (~19,500 GTP sessions, ~14.5% independently corroborated) is
far smaller and noisier than arm A's (~1.2M mature flow records) — a
confound RQ1's finding is explicitly read through, not glossed over.

## Citation

See [`CITATION.cff`](CITATION.cff). If you use this code or its
methodology:

```bibtex
@mastersthesis{hernandez2026_5gnidd_agentic_ids,
  author = {Hernández, Justo},
  title  = {5G-GTP Agentic IDS: An Agentic Intrusion Detection Architecture for the 5G User Plane},
  school = {IT BUSSINESS SCHOOL (ENIIT)},
  year   = {2026},
  type   = {Master's Thesis (TFM)}
}
```

The underlying dataset ([5G-NIDD](https://ieee-dataport.org/documents/5g-nidd-comprehensive-network-intrusion-detection-dataset-generated-over-5g-wireless)) is not redistributed here and has its own citation requirements — see
[`data/dataset_instructions.md`](data/dataset_instructions.md).
## REFERENCE
Y. Siriwardhana et al., "Descriptor: 5G Wireless Network Intrusion Detection Dataset (5G-NIDD)," in IEEE Data Descriptions, doi: 10.1109/IEEEDATA.2025.3592888.https://ieeexplore.ieee.org/document/11098458/references#references