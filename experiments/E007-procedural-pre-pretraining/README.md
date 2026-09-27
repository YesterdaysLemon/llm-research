# E007 — Procedural pre-pretraining for a tiny student

E007 is the first test of [H-012](../../hypothesis.md#h-012--procedural-pre-pretraining-supplies-tiny-model-computation). It trains E004's 1.58M student on synthetic data first, then on TinyStories. There are two synthetic sources:
- random context-free grammars;
- outputs of programs sampled from arXiv:2609.30063's universal prior.

It asks whether that head start beats three alternatives:
- training from scratch;
- training from scratch with the synthetic stage's compute spent on extra real text;
- a control with the same weight statistics but no learned structure.

Background is in the [reading note](../../research/self-play-pretraining-2026-09-26.md).

## Status

**Registered; smoke pending on the RTX 4060 host.** Read the [preregistration](preregister.md). Nothing has run on real data. The unit tests pass on CPU. The program interpreter reproduces all five of the paper's discovered sequences.

## Local data

Use E004's pinned artifact:

```powershell
& C:\path\to\python.exe experiments/E004-tiny-language-model/src/fetch_data.py
```

## Run

```powershell
& C:\path\to\python.exe -m unittest discover `
  -s experiments/E007-procedural-pre-pretraining/tests -v

# 1. Smoke: validity only (seed 8700). The first run also builds the
#    128,000-row program bank, a one-time CPU job cached under results/cache/.
& C:\path\to\python.exe experiments/E007-procedural-pre-pretraining/src/train.py `
  --config experiments/E007-procedural-pre-pretraining/config/smoke.json `
  --output-dir experiments/E007-procedural-pre-pretraining/results/smoke `
  --device cuda

# 2. Fixed comparison: 25 runs (5 seeds x 5 arms)
& C:\path\to\python.exe experiments/E007-procedural-pre-pretraining/src/train.py `
  --config experiments/E007-procedural-pre-pretraining/config/fixed.json `
  --output-dir experiments/E007-procedural-pre-pretraining/results/fixed `
  --device cuda

# 3. Registered analysis
& C:\path\to\python.exe experiments/E007-procedural-pre-pretraining/analyze.py
```

The runner writes one file per (arm, seed) and skips files that already exist. The synthetic stage is saved as a checkpoint, so the shuffled control reuses it. Checkpoints, the program bank and their hashes are recorded, and the checkpoints and bank are git-ignored.

The bank build uses every CPU core by default; set `--bank-processes` to change that. The bank's contents do not depend on the process count.
