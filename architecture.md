# Architecture

System design for the agentic 5G-NIDD intrusion detection pipeline. See
[`experiment_plan.md`](experiment_plan.md) for methodology, research
questions, and results; this document covers code structure only.

## Pipeline stages

```
raw pcapng (data/raw/BS{1,2}/*.pcapng)
  -> parsers/          GTPPacketRecord stream (Scapy, streaming PcapReader)
  -> preprocessing/     TEIDFeatureRecord (per-TEID-instance features)
                         PDUSessionRecord (per (ue_ip, teid, window) features)
                         multi-level labeling (Level 1/2/3 -> HIGH/MEDIUM/LOW)
  -> agents/             TEIDAgent, PDUSessionAgent -> SupervisorAgent (arm C)
  -> ml/                  RandomForest/XGBoost on official CSVs (arm A1/A2)
                           and on GTP session features (arm B)
  -> evaluation/           metrics, ROC/PR curves, case studies, writeup
```

## Data models (`src/agente_5g/models/`)

Pydantic models, frozen where they represent an immutable observation:

- `packet.py::GTPPacketRecord` — one parsed packet (inner/outer IP-port,
  TCP flags, GTP TEID).
- `teid_features.py::TEIDFeatureRecord` — engineered features for one
  TEID instance (idle-gap-split), plus `label`/`is_attack`/
  `label_confidence`/`label_evidence` (populated later by labeling).
- `session.py::PDUSessionRecord` — engineered features for one
  (ue_ip, teid, fixed-window) slice, plus the same label fields and
  `state_sequence`/`final_state` (populated later by `agents/`).
- `agent_decision.py::AgentDecision`/`SupervisorDecision` — agent outputs.
- `evaluation.py::EvaluationResult` — one row per (arm, model) evaluation.
- `labels.py` — `LabelConfidence` (HIGH/MEDIUM/LOW), `LabelSource`
  (SCHEDULE/VICTIM_IP/PATTERN), filename-token-to-attack-type mapping.
- `schedule_config.py` — typed loaders for `configs/attack_schedule.yaml`
  and `configs/label_patterns.yaml`.
- `agent_thresholds.py` — typed loader for `configs/thresholds.yaml`.

## Parsing (`src/agente_5g/parsers/`)

`base.py::PacketParser` is the abstract interface. `scapy_parser.py` is
the primary, always-available implementation (streaming `PcapReader`,
never `rdpcap`, so memory stays bounded on 500MB+ files); `gtp_layers.py`
registers Scapy's GTP-U contrib layer plus a manual byte-offset fallback
decoder for packets the contrib layer doesn't cover. `pyshark_parser.py`
is an optional second backend, only selected by `factory.py::get_parser`
if `tshark` is found on `PATH` — the project installs and runs fully
without it.

## Preprocessing (`src/agente_5g/preprocessing/`)

- `teid_extractor.py::TEIDFeatureExtractor` — groups packets by TEID,
  splits into instances on an idle-gap threshold, computes rate/entropy/
  burstiness/directionality features. Also hosts `infer_initiator_ip`
  (flow-initiation convention: whoever sent a TEID's first packet owns
  it) and `shannon_entropy`, both reused by `session_builder.py` and (for
  entropy) `labeling.py`.
- `session_builder.py::SessionBuilder` — groups packets into fixed-size
  windows keyed by `(ue_ip, teid)`, computing `state_transition_rate`
  (TCP-flag-derived pseudo-state churn) and `temporal_entropy` (packet
  arrival uniformity across 10 sub-bins).
- `labeling.py` — the ground-truth labeling pipeline. See
  `experiment_plan.md`'s "Ground-truth labeling methodology" section for
  the full Level 1/2/3 design and the Phase 4C confidence-model rationale;
  this module's own docstring is the authoritative in-code reference.
- `feature_cache.py` — Parquet round-trip caching for parsed packets/
  features/sessions (`outputs/cache/`, gitignored), used throughout
  development to avoid re-parsing the same large pcapng files repeatedly.

## Agentic detectors (`src/agente_5g/agents/`) — arm C

Deterministic, rule-based, no LLM in the decision path (reproducible
given the same features and `configs/thresholds.yaml`):

- `rules.py` — pure threshold functions (`RuleResult(triggered, intensity,
  detail)`), physically separate file/constants from
  `preprocessing/labeling.py`'s Level-3 pattern checks (enforced by
  `tests/unit/test_architecture_boundaries.py`) so label quality is never
  validated against the same logic being evaluated as a detector.
- `teid_agent.py::TEIDAgent` — flood/syn_flood/scan rules over a single
  `TEIDFeatureRecord`.
- `pdu_session_agent.py::PDUSessionAgent` — a NORMAL→WATCH→SUSPICIOUS→
  ATTACK state machine over a chronologically-ordered session sequence
  for one `(ue_ip, teid)`, rate-limited to one level per observed window.
- `supervisor_agent.py::SupervisorAgent` — deterministic weighted average
  of the two agents' risk scores (`configs/thresholds.yaml`'s
  `fusion_weights`), producing the final Attack/Benign label.
- `explain.py` — optional local-Ollama HTTP client for post-hoc natural-
  language explanation text only; disabled by default, never feeds back
  into any score, JSON-lines cached.

## ML baselines (`src/agente_5g/ml/`) — arms A1/A2/B

- `dataset.py` — loaders/preprocessing for all three ML arms; see its
  module docstring for the full arm A1 vs. A2 vs. B/C split-strategy
  rationale (including the per-attack-type chronological split fix for
  Combined.csv/Encoded.csv, whose `Seq`/`RunTime` columns reset per
  source capture file rather than forming a global timeline).
- `random_forest.py` / `xgboost_model.py` — thin seeded wrappers
  (`RandomForestModel`, `XGBoostModel`) sharing one fit/predict/
  predict_proba/feature_importances interface, used identically for
  arms A1, A2, and B.
- `train.py` — orchestration: `train_and_evaluate_arm_a`,
  `train_and_evaluate_arm_b`, `evaluate_arm_c` (runs the existing,
  untrained agentic pipeline on the identical test split arm B was
  scored on, pairing sessions to TEID features via window containment/
  nearest-match — `_match_feature`).

## Evaluation (`src/agente_5g/evaluation/`)

- `metrics.py::compute_metrics` — the one code path every arm's
  accuracy/precision/recall/f1/roc_auc/fpr/confusion_matrix is computed
  through.
- `compare.py` — ROC/PR curve data, `threshold_sensitivity` (a metrics
  sweep for discussion only, never selects a threshold).
- `case_studies.py` — per-attack-type representative TP/FP/FN examples
  and per-type error-rate summaries, replacing a single scalar
  "explainability score."
- `visualize.py` — Plotly figure builders (ROC/PR curves, confusion
  matrix heatmaps, threshold-sensitivity plots); `label_validation.py`
  has the earlier, labeling-specific figure builders (confidence-tier
  bars, temporal/victim distributions) used by the Phase 4 validation
  reports.

## Configuration

- `configs/thresholds.yaml` — agent detection rule thresholds.
- `configs/label_patterns.yaml` — Level-3 label-validation pattern
  thresholds. Physically separate from `thresholds.yaml` by design (see
  `agents/rules.py`'s docstring and the architecture-boundary test).
- `configs/attack_schedule.yaml` — Tables III/IV of the 5G-NIDD
  descriptor paper, plus the victim IP and the relative-window mapping
  rationale.
- `configs/base.yaml` + `sample.yaml`/`full.yaml` — loaded by
  `settings.py::Settings` (seed, paths, window sizes, ML hyperparameter
  defaults). **Not** what the scripts actually used to produce the
  results in `experiment_plan.md` — see "Scripts actually used" below.

## Scripts actually used (`scripts/`)

The original plan called for `scripts/run_sample_pipeline.py` and
`scripts/run_full_pipeline.py` driven by `Settings`/`configs/base.yaml`.
In practice, the labeling investigation and Phase 6/7 work were done with
a sequence of purpose-built scripts, each with its own local constants
(attack-type list, `max_duration_s`, etc.) rather than going through
`Settings` — a real gap between the originally-planned unified CLI and
what was actually run, noted here rather than glossed over (see
`experiment_plan.md`'s Limitations section). The scripts that produced
the committed results, in the order they matter:

- `validate_labeling.py` / `validate_labeling_all.py` — per-file and
  9-type-aggregate labeling validation reports; `process_file()` here is
  imported by nearly every later script for its packet-cache-backed
  parse+label+build pipeline.
- `calibrate_flood_pattern.py`, `diagnose_confidence_system.py`,
  `quantify_evidence_sources.py`, `inspect_flood_evidence.py`,
  `validate_connection_flood_hypothesis.py` — the Phase 4 label-quality
  investigation trail (see `experiment_plan.md`).
- `calibrate_agent_thresholds.py`, `validate_agents.py` — Phase 5B agent
  threshold calibration and validation against real data.
- `run_phase6_training.py` — trains/evaluates all four arms, writes
  `outputs/reports/phase6_training/results.csv` (the primary reported
  results).
- `run_phase7_analysis.py` — re-derives per-sample predictions from the
  identical frozen configuration to build ROC/PR curves, threshold-
  sensitivity plots, case studies, and the RQ1-4 writeup; cross-checks
  every recomputed confusion matrix against `results.csv` as a
  reproducibility guard.

All `outputs/**` (reports, figures, cache) are gitignored — they are
regenerable artifacts, not source. `experiment_plan.md` inlines the
essential final numbers so the thesis narrative doesn't depend on them
being present.

## Testing

`tests/unit/` (fast, no real pcap data — Scapy-constructed synthetic
packets or hand-built Pydantic records) is the default `pytest` run,
coverage-gated at 80% (`pyproject.toml`, currently ~93%; 198 tests as of
the Phase 7 commit). An
`integration` pytest marker and `tests/integration/` package exist
(intended for a real-file smoke test, e.g. `SSH_BS1.pcapng`) but no test
was ever written there — the real-data validation instead happened
through the `scripts/` listed above, run directly against the raw
pcapng files rather than as part of the pytest suite.
