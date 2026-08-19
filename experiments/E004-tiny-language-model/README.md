# E004 — Tiny causal-language-model bridge

E004 is the first language-model test of [H-010](../../hypothesis.md#h-010--activation-specific-distillation-survives-next-token-controls).
It asks whether a learned teacher's hidden relation geometry supplies a useful
training signal beyond next-token labels, output distillation, target-token
equivalence, and teacher-logit geometry.

The experiment is intentionally dependency-light: the causal Transformer,
tokenizer, controlled language, training loop, and analysis live in this
directory and require only PyTorch plus the Python standard library.

## Status

**Smoke protocol drafted; no E004 result yet.** Read [the prospective
protocol](preregister.md) before interpreting any artifact. The fixed paired
comparison must not run until the teacher and labels-only student clear the
smoke gates and any amendment is committed.

## Local data

The natural-language component is a pinned TinyStories source artifact. Fetch
and verify it with:

```powershell
& C:\path\to\python.exe experiments/E004-tiny-language-model/src/fetch_data.py
```

The 22.5 MB source file is stored under `data/raw/`, which is ignored by Git.
Its revision, byte count, SHA-256, and CDLA-Sharing-1.0 dataset license are
frozen in the configuration and preregistration. Repository code and original
documentation are MIT licensed; this does not relicense TinyStories.

## Run the smoke phase

```powershell
& C:\path\to\python.exe -m unittest discover `
  -s experiments/E004-tiny-language-model/tests -v

& C:\path\to\python.exe experiments/E004-tiny-language-model/src/run.py `
  --config experiments/E004-tiny-language-model/config/smoke.json `
  --output experiments/E004-tiny-language-model/results/smoke.json `
  --device cuda
```

The result records the Git commit, worktree status, configuration and corpus
hashes, environment, parameter counts, token exposures, wall time, peak GPU
memory, teacher gates, and the single labels-only student gate.

