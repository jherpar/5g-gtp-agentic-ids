# data/

Not committed to git (large binaries — see `.gitignore`). This directory
is populated locally by placing the 5G-NIDD dataset files as described in
[`dataset_instructions.md`](dataset_instructions.md).

```
data/
├── raw/
│   ├── BS1/                     10 attack-type pcapng captures (base station 1)
│   ├── BS2/                     10 attack-type pcapng captures (base station 2)
│   └── BS1_each_attack_csv/     per-attack-type flow CSVs with real IPs (appendix validation only)
├── processed/
│   ├── Combined/Combined.csv    dataset authors' flow export, clean categorical columns (arm A1 input)
│   └── Encoded/Encoded.csv      dataset authors' pre-encoded export (arm A2 input, secondary check)
└── features/                    parquet cache written by preprocessing.feature_cache (git-tracked
                                  directory, empty by default — see .gitkeep)
```

Reported results (`docs/experiment_plan.md`) use **BS1 only**. `BS2` is
only needed to reproduce the Phase 4C connection-oriented-flood
hypothesis validation. `BS1_each_attack_csv` is only needed to reproduce
the ground-truth-validation appendix
(`docs/appendix_ground_truth_validation.md`) — it is not used by the
main Phase 4–8 pipeline.

`data/raw/**/*.pcapng`, `data/raw/**/*.csv`, and `data/processed/**/*.csv`
are gitignored (large binaries, not source). `data/features/` is also
gitignored except for a `.gitkeep` placeholder, since it's a regenerable
cache, not an input.
