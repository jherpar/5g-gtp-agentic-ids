# Contributing

This repository is the code release for a Master's Thesis (TFM). It is
published for transparency, reproducibility, and reuse — not as an
actively-developed product.

## Project status: experimental phase closed

The experimental phase concluded with the Phase 7 evaluation and the
RQ4 follow-up / ground-truth validation appendix (see
[`docs/experiment_plan.md`](docs/experiment_plan.md)). **No further
models, thresholds, labels, or analysis will be added to the reported
results by the author.** Pull requests that change reported numbers,
retrain models, or alter any frozen configuration
(`configs/thresholds.yaml`, `configs/label_patterns.yaml`,
`configs/attack_schedule.yaml`) will not be merged, since that would
invalidate the thesis's reported findings.

## What contributions ARE welcome

- Bug reports and fixes that don't change any reported metric (e.g. a
  crash on an edge case, a broken reproduction step, a documentation
  error).
- Documentation improvements, typo fixes, clearer reproduction
  instructions.
- Extensions clearly built *on top of* the frozen baseline — e.g. a BS2
  run, a different dataset, new attack types — as long as they are
  presented as a fork or a clearly-separated addition, not a change to
  the existing Phase 4–8 results.
- Questions and discussion via GitHub Issues.

## Development setup

```bash
poetry install
poetry run pytest        # 198 tests, ~93% coverage required
poetry run ruff check .
poetry run black --check .
poetry run mypy src
```

See [`reproducibility/run_order.md`](reproducibility/run_order.md) for
the full pipeline reproduction sequence.

## Code style

- Python 3.11+, type hints mandatory, Pydantic models for data records.
- Formatted with Black, linted with Ruff, type-checked with mypy
  (`--strict` on `src/`).
- `pytest` coverage must stay ≥80% (currently ~93%).
- No comments explaining *what* code does (names should do that) —
  comments are reserved for non-obvious *why* (a constraint, a bug
  workaround, a design decision that would otherwise be surprising).

## Reporting issues

Please include: what you ran, what you expected, what happened instead,
and your environment (`poetry run python --version`, OS). If it's about
a reported result or figure, please reference the exact section of
[`docs/experiment_plan.md`](docs/experiment_plan.md) or the exact script
in `scripts/`.
