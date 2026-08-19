# Preprint reproduction

The canonical manuscript is [preprint.md](preprint.md). Tables and figures are generated from committed result JSON, not transcribed as the source of truth.

## Rebuild

From the repository root in the project virtual environment:

```powershell
python -m pip install -r paper/requirements.txt
python paper/analyze.py
python paper/build_pdf.py
```

The PDF is written to `output/pdf/relational_activation_distillation_preprint.pdf`.

## Source-of-truth results

- `experiments/E001-trajectory-sufficiency/results/pilot.json`
- `experiments/E001-trajectory-sufficiency/results/p1-dev-budget.json`
- `experiments/E001-trajectory-sufficiency/results/p1-dev-arch-4x128.json`
- `experiments/E001-trajectory-sufficiency/results/p1-dev-arch-4x256.json`
- `experiments/E002-contextual-transition-executor/results/confirm-affine.json`
- `experiments/E002-contextual-transition-executor/results/confirm-bitwise.json`

`paper/analyze.py` regenerates `paper/generated/statistics.json`, `paper/generated/tables.md`, and all four figures. The manuscript's compact tables are checked against that generated output during validation.

## Interpretation boundary

Only the affine E002 cell passed every registered positive-control gate. Bitwise measurements are retained for transparency but are descriptive. The affine result supports a correspondence-preserving learned-teacher layer-geometry effect, not yet teacher-specific activation content or a portable-program interpretation. The task-factored executor is a compact existence comparison; its semantically selected table is not realized as sparse compute by the current implementation. The paper makes no claim about language models, general lossless compression, or energy efficiency.
