# E006 — Heart-wall agent sampling

This bounded side experiment measures how fresh subscription-backed coding-agent
sessions interpret a deliberately underspecified visual prompt. It is separate
from the repository's active E005 tiny-language-model decision.

The fixed prompt is:

> Print one heart emoji. Then, on the next line, print a wall made only of heart emojis. Do not explain your choices and do not use code fences.

The registered comparison targets 50 independent responses at every supported
reasoning-effort level for one OpenAI Codex model and one Claude Code model.
Effort labels are compared within a provider only; similarly named levels are
not assumed to be equivalent across providers.

## Commands

```powershell
python experiments/E006-heart-wall-agent-sampling/src/run_study.py `
  --config experiments/E006-heart-wall-agent-sampling/config/fixed-50.json `
  --output experiments/E006-heart-wall-agent-sampling/results/fixed-50.jsonl

python experiments/E006-heart-wall-agent-sampling/src/analyze.py `
  --input experiments/E006-heart-wall-agent-sampling/results/fixed-50.jsonl `
          experiments/E006-heart-wall-agent-sampling/results/fixed-50-claude.jsonl `
  --json experiments/E006-heart-wall-agent-sampling/results/fixed-50-summary.json `
  --csv experiments/E006-heart-wall-agent-sampling/results/fixed-50-runs.csv

python -m pytest experiments/E006-heart-wall-agent-sampling/tests
```

The runner is resumable by job ID and never retries a recorded observation.
Transport failures stay in the dataset. Authentication is inherited from the
locally installed subscription CLIs, while API-key environment variables are
removed from child processes.

See [preregister.md](preregister.md) for the frozen design and limitations.
