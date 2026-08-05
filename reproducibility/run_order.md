# Reproducibility guide: environment, data, and exact run order

This is the complete, self-contained path from a fresh clone to every
number/figure reported in [`docs/experiment_plan.md`](../docs/experiment_plan.md).
All scripts are deterministic (seed 42, `src/agente_5g/ml/dataset.py::SEED`,
propagated everywhere a model is fit) and read/write only
`configs/*.yaml` (read-only, frozen) and `outputs/` (regenerable,
gitignored).

## 1. Environment setup

Requires Python 3.11 (3.12 not supported — see `pyproject.toml`'s
`requires-python`).

**Option A — Poetry (primary, authoritative path):**

```bash
poetry install
```

**Option B — pip + venv:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r reproducibility/requirements.txt
pip install -e .
```

**Option C — conda:**

```bash
conda env create -f reproducibility/environment.yml
conda activate agente-5g
pip install -e .
```

Optional PyShark parser backend (not needed — the primary Scapy backend
has no external dependency):

```bash
poetry install --extras pyshark    # requires tshark/Wireshark on PATH
```

Verify the install:

```bash
poetry run pytest        # 198 tests, ~93% coverage, no real data needed
poetry run ruff check .
poetry run black --check .
poetry run mypy src
```

## 2. Dataset preparation

See [`../data/dataset_instructions.md`](../data/dataset_instructions.md)
for where to obtain the 5G-NIDD dataset and exactly how to place it.
Minimum for the primary reported results: `data/raw/BS1/*.pcapng` (10
files) + `data/processed/{Combined,Encoded}/*.csv`.

## 3. Run order

Every step below caches its parsed-packet intermediate under
`outputs/cache/packets/` (gitignored) — the FIRST run of each attack-type
file is slow (large pcapng parsing, see per-file timing below), every
subsequent script that touches the same file reuses the cache and is
fast (seconds).

| # | Command | Produces | Approx. time (cold / cached) |
|---|---|---|---|
| 1 | `poetry run python scripts/validate_labeling_all.py` | `outputs/reports/labeling_validation_all/` — HIGH/MEDIUM/LOW distribution, all 9 BS1 types | ~25-30 min / ~1 min |
| 2 | `poetry run python scripts/validate_agents.py` | `outputs/reports/agent_validation/` — Phase 5 agents vs. real data | ~1 min (cached from step 1) |
| 3 | `poetry run python scripts/run_phase6_training.py` | `outputs/reports/phase6_training/results.csv` — **primary reported results, all 4 arms** | ~5-7 min (A1/A2 training on ~1.2M rows dominates: ~90s each, regardless of cache) |
| 4 | `poetry run python scripts/run_phase7_analysis.py` | `outputs/reports/phase7_analysis/`, `outputs/figures/phase7/` — ROC/PR curves, case studies, RQ1-4 writeup; independently re-derives and bit-for-bit verifies every Phase 6 confusion matrix | ~5-7 min |
| 5 | `poetry run python scripts/analyze_rq4_detection_latency.py` | `outputs/reports/rq4_detection_latency/`, `outputs/figures/rq4_detection_latency/` — RQ4 detection-latency measurement study | ~2-3 min (cached) |
| 6 | `poetry run python scripts/analyze_bs1_each_attack_csv_validation.py` | `outputs/reports/bs1_each_attack_csv_validation/` — ground-truth validation appendix (requires `data/raw/BS1_each_attack_csv/`, optional) | ~2-3 min (cached) |

Steps 1–2 are prerequisite investigation/calibration reports (Phase
4/4C/5B) already reflected in the frozen `configs/*.yaml` values — they
are not needed to reproduce Phase 6/7's numbers, only to see the
evidence trail that led to those config values. **To regenerate Phase 6
and Phase 7 specifically, steps 3 and 4 are sufficient** on top of a
working environment and dataset.

Earlier investigation scripts (`calibrate_flood_pattern.py`,
`diagnose_confidence_system.py`, `quantify_evidence_sources.py`,
`inspect_flood_evidence.py`, `validate_connection_flood_hypothesis.py`,
`calibrate_agent_thresholds.py`) reproduce the Phase 4C/5B calibration
evidence trail referenced in `docs/experiment_plan.md`, but their output
values are already baked into the committed `configs/*.yaml` files —
re-running them is optional and only useful to inspect the underlying
evidence, not to reproduce any reported metric.

## 4. What's deterministic vs. what to expect to vary

- **Deterministic, bit-for-bit**: every metric in `results.csv`, every
  confusion matrix, every ROC-AUC — verified in step 4 (10/10
  reproducibility checks against step 3's output).
- **Not deterministic, and not load-bearing**: wall-clock timing numbers
  (`fit_time_ms`/`inference_time_ms_per_sample` columns) will differ by
  hardware; only the ranking/order-of-magnitude claims in
  `docs/experiment_plan.md` should be treated as portable, not exact
  values.
- **Requires `data/raw/BS2/`** (not needed for the primary results): only
  the Phase 4C connection-oriented-flood cross-base-station validation
  (`validate_connection_flood_hypothesis.py`) reads BS2 data.
