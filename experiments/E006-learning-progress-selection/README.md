# E006 — Learning-progress selection of natural text

E006 is the first test of [H-013](../../hypothesis.md#h-013--learning-progress-selection-of-natural-data). It transfers the generator reward of arXiv:2609.30063 to natural text. At every step, it picks 16 of 64 candidate TinyStories windows by the absolute alignment between each window's gradient and the learner's recent AdamW-preconditioned parameter movement. It compares that choice against uniform, loss, gradient-norm, direction-only and signed selection, and against uniform training given the scoring compute. See the [reading note](../../research/self-play-pretraining-2026-09-26.md) for where the hypothesis comes from.

The model, tokenizer and data split are imported unchanged from [E004](../E004-tiny-language-model/README.md). The runner verifies that E004's vocabulary and token counts reproduce before it trains.

## Status

**Registered; smoke pending on the RTX 4060 host.** Read the [preregistration](preregister.md). Nothing has run on real data. The unit tests pass on CPU, including the checks that the JVP scores equal explicit per-example gradient dot products.

## Local data

Use E004's pinned artifact:

```powershell
& C:\path\to\python.exe experiments/E004-tiny-language-model/src/fetch_data.py
```

## Run

```powershell
& C:\path\to\python.exe -m unittest discover `
  -s experiments/E006-learning-progress-selection/tests -v

# 1. Smoke: validity only (seed 8600, 5% budget, all arms, both conditions)
& C:\path\to\python.exe experiments/E006-learning-progress-selection/src/train.py `
  --config experiments/E006-learning-progress-selection/config/smoke.json `
  --output-dir experiments/E006-learning-progress-selection/results/smoke `
  --device cuda

# 2. Fixed comparison: 60 runs (5 seeds x 2 conditions x 6 arms)
& C:\path\to\python.exe experiments/E006-learning-progress-selection/src/train.py `
  --config experiments/E006-learning-progress-selection/config/fixed.json `
  --output-dir experiments/E006-learning-progress-selection/results/fixed `
  --device cuda

# 3. Registered analysis
& C:\path\to\python.exe experiments/E006-learning-progress-selection/analyze.py
```

The runner writes one file per run and skips files that already exist. An interrupted fixed run therefore resumes where it stopped, and no completed run is overwritten.

Each file records:
- the Git commit and worktree status;
- the configuration, corpus and vocabulary hashes;
- the environment;
- the evaluation curve with nominal FLOPs and wall time;
- selected-noise counts;
- diagnostic score correlations and JVP consistency;
- peak CUDA memory.

The analysis requires SciPy for exact t quantiles, and falls back to a built-in table otherwise.
