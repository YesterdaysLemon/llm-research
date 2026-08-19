# NEXT

Last updated: 2026-08-19

## Current evidence

The E001–E003 synthetic arc is complete. In the valid affine cell, adding correspondence-preserving learned-teacher encoder-layer geometry to labels and logits improved held-out-pair accuracy from 11.30% to 23.31% across ten paired seeds.

An independent Opus 5 and Qwen 3.8 council found that the earlier mechanistic wording exceeded the design. The target is indexed by encoder layer, not task-computation step; a target-label Gram was missing; the generic student had only 8.15% in-distribution accuracy at depth four; and the task-factored executor is a compact existence comparison rather than a causal context-selection result or a swept efficiency frontier.

E003 ran the council's cheapest decisive control once from a clean preregistration commit. Learned-teacher geometry beat target-label geometry by 17.44 percentage points, 95% CI [15.65, 19.23]. Target-label geometry underperformed logits alone by 5.43 points [-6.66, -4.19]. Terminal-answer equivalence therefore does not explain the learned-teacher gain under this setup. The [council record](research/council-2026-08-19.md), [E003 report](experiments/E003-label-geometry-specificity/report.md), and corrected [preprint](paper/preprint.md) are the current interpretive authorities.

The task-factored affine executor reached 100% with 17,672 stored parameters versus 1,082,927 for the relational Transformer. Its relation token semantically selects one table, but the reference implementation computes all table softmaxes before indexing. Stored count, semantic selection, realized compute, wall time, and energy must remain separate.

The bitwise cell remains descriptive because its teacher missed the registered positive-control gate.

## Active decision

Preregister H-010 as E004, the first tiny causal language-model bridge. Do not run a fixed comparison until a smoke phase has established a teacher and labels-only student capability floor and frozen the corpus/tokenizer artifacts.

The experiment must decide whether learned hidden-relation geometry improves genuine next-token learning beyond both output distillation and next-token target geometry at matched student size, tokens, initialization, and optimizer steps.

The minimum core is:

1. ordinary next-token training;
2. output distillation;
3. output plus learned hidden-relation geometry;
4. output plus next-token target geometry;
5. an ordinary extra-data or extra-compute baseline.

Prefer a 1M–10M-parameter student and a 10M–40M teacher that can both run on the RTX 4060 and M2 host. Use at least five paired seeds, a dominant simple-English stream, and a smaller controlled compositional stream. Freeze natural validation NLL, controlled accuracy by depth, teacher and student capability gates, token counts, model bytes, wall time, and peak memory before the fixed run. No activation-specific claim is allowed unless learned hidden geometry beats both output distillation and target-token geometry. No efficiency claim is allowed unless it survives the ordinary compute/data control.

Candidate public foundations include TinyStories for language that small models can learn, WikiText-2 as a small natural-text anchor, and Pythia's openly documented small checkpoints for calibration or architecture references. Dataset versions, licenses, tokenizer training data, and exact checksums must be frozen before use.

## Start-of-session checklist

1. Run `git status` and preserve unrelated work.
2. Read this file, the active preregistration, and the latest report.
3. State the one decision the run can change.
4. Freeze configuration and code in a clean commit before viewing a new result.
5. Run deterministic tests and record the configuration hash and Git commit.
6. Update the report and hypothesis ledger for positive, null, invalid, or negative outcomes.
7. Inspect the complete diff and secret scan before publishing.
