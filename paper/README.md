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
- `experiments/E003-label-geometry-specificity/results/confirm-label-geometry.json`
- `experiments/E003-label-geometry-specificity/results/analysis.json`
- `experiments/E004-tiny-language-model/results/smoke.json`
- `experiments/E004-tiny-language-model/results/smoke-answer-weighted.json`

`paper/analyze.py` regenerates `paper/generated/statistics.json`, `paper/generated/tables.md`, and all four figures. The manuscript's compact tables are checked against that generated output during validation.

## Interpretation boundary

Only the affine E002 cell passed every registered positive-control gate. Bitwise measurements are retained for transparency but are descriptive. E003 prospectively shows that E002's learned-teacher layer-geometry effect exceeds terminal-label geometry, while portable-program and broader causal interpretations remain unresolved. E004 is a failed tiny causal-LM capability bridge: answer weighting supports a shallow loss-allocation effect, but no teacher/student gate passed and no KD/geometry condition ran. The task-factored executor is a compact existence comparison; its semantically selected table is not realized as sparse compute by the current implementation. The paper makes no positive language-model, general lossless compression, or energy-efficiency claim.
