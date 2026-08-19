# E004 — Tiny causal-language-model bridge

E004 is the first language-model test of [H-010](../../hypothesis.md#h-010--activation-specific-distillation-survives-next-token-controls).
It asks whether a learned teacher's hidden relation geometry supplies a useful
training signal beyond next-token labels, output distillation, target-token
equivalence, and teacher-logit geometry.

The experiment is intentionally dependency-light: the causal Transformer,
tokenizer, controlled language, training loop, and analysis live in this
directory and require only PyTorch plus the Python standard library.

## Status

**Capability gate failed; the controlled benchmark is retired.** Read the
[prospective protocol](preregister.md), [initial smoke report](report-smoke.md),
and [answer-weighted diagnostic](report-answer-weighted.md). Weighting rare
answer positions created strong shallow ID gains, but neither teacher nor
labels-only student passed the frozen gates. The fixed paired comparison was not
run, so H-010 remains untested.

The diagnostic changed only the controlled answer-token loss weight and was
frozen in [its own preregistration](preregister-answer-weighted.md).

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
