# docs/

| File | Contents |
|---|---|
| [`architecture.md`](architecture.md) | System design by module: parsing, feature extraction, labeling, agentic detection, ML baselines, evaluation — what was actually built and run, including where it diverges from the original plan (documented, not hidden). |
| [`experiment_plan.md`](experiment_plan.md) | The full thesis-facing methodology and results document: ground-truth labeling design and its Phase 4/4C findings, Phase 5B agent calibration, the four-arm ML baseline design, RQ1–4 findings (including the RQ4 detection-latency follow-up), Limitations, Threats to Validity, and Reproducibility. **This is the primary source of truth for every reported number.** |
| [`appendix_ground_truth_validation.md`](appendix_ground_truth_validation.md) | Standalone appendix: an independent cross-check of the labeling methodology against a previously-unused author-provided per-flow CSV with real IP addresses. Does not modify any Phase 4–8 result or conclusion. |
| `memoria.pdf` *(not yet added)* | The full thesis document (TFM memoria). |
| `defense_slides.pdf` *(not yet added)* | Thesis defense presentation slides. |

`memoria.pdf` and `defense_slides.pdf` are excluded from this repository's
default `*.pdf` gitignore rule (see `../.gitignore`'s `!docs/*.pdf`
exception) so they can be committed here once finalized.

Read order for a first pass: `architecture.md` for what the system is,
then `experiment_plan.md` for what was found and why it's trustworthy
(or isn't — the Limitations/Threats to Validity sections are as load-bearing
as the results tables).
