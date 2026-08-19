# E003 — Label-geometry specificity

**Status:** preregistered; not yet run.

## Decision

Does learned teacher layer geometry provide a held-out-composition advantage beyond the simpler geometry induced by examples sharing the same terminal label?

This is the smallest discriminating follow-up to the external council's audit of E001–E002. It adds one frozen condition to the valid affine task and compares it with already committed E002 results. It is not a rescue run and does not alter the E002 decision rules.

Read the [preregistration](preregister.md) before running the frozen [configuration](config/confirm-label-geometry.json).

## Run

From the repository root:

```powershell
.\.venv\Scripts\python.exe experiments\E002-contextual-transition-executor\src\run_study.py `
  --config experiments\E003-label-geometry-specificity\config\confirm-label-geometry.json `
  --output experiments\E003-label-geometry-specificity\results\confirm-label-geometry.json `
  --device cuda
```

