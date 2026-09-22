# Post-primary diagnostic: answer formatting and the token cap

Frozen after the original 792 rows completed and their summaries/raw recency
outputs were inspected, before any diagnostic run. This is explicitly
exploratory and does not replace the primary endpoint or its results.

Observation: recency outputs often begin "The current access color for
archive-..." and hit the 12-token cap before naming a color. The primary
first-word endpoint conflates instruction retention, output format, truncation,
and factual retrieval. A zero score does not establish loss of the target fact.

Run the same 24 held-out prompts at 8 MiB / INT8 for sink-plus-recency, seeded
random, and both learned arms at the first preselected policy seed (41), plus
full BF16 cache. Change only the maximum generated tokens from 12 to 64.
No retraining, task replacement, policy selection, or modified prompt.

Report the original first-word endpoint, whether the first mentioned color
matches the target, EOS/cap counts, and every raw output. The relaxed color
metric is post-hoc and can credit an unsupported guess or an early color in an
ambiguous answer; it is not a definitive semantic correctness score.

Any claim that learned retention outperforms recency on factual retrieval
must acknowledge this confound and the lack of independent strong baselines.
