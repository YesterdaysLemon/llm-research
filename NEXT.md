# NEXT

Last updated: 2026-08-19

## Current evidence

The E001–E002 experimental arc is complete. The valid affine cell supports a narrow behavioral statement: adding correspondence-preserving learned-teacher encoder-layer geometry to labels and logits improved held-out-pair accuracy from 11.30% to 23.31% across ten paired seeds.

An independent Opus 5 and Qwen 3.8 council found that the earlier mechanistic wording exceeded the design. The target is indexed by encoder layer, not task-computation step; a target-label Gram was missing; the generic student had only 8.15% in-distribution accuracy at depth four; and the task-factored executor is a compact existence comparison rather than a causal context-selection result or a swept efficiency frontier. The [council record](research/council-2026-08-19.md) and corrected [preprint](paper/preprint.md) are the current interpretive authorities.

The task-factored affine executor reached 100% with 17,672 stored parameters versus 1,082,927 for the relational Transformer. Its relation token semantically selects one table, but the reference implementation computes all table softmaxes before indexing. Stored count, semantic selection, realized compute, wall time, and energy must remain separate.

The bitwise cell remains descriptive because its teacher missed the registered positive-control gate.

## Active experiment

Run [E003 label-geometry specificity](experiments/E003-label-geometry-specificity/preregister.md) exactly once from its clean preregistration commit. It adds one condition: labels plus logits plus a teacher-free terminal-label Gram. The primary paired contrast is learned-teacher geometry minus target-label geometry.

The result changes one decision:

- positive lower CI: learned teacher geometry provides benefit beyond answer-class equivalence under this setup;
- CI includes zero: teacher specificity remains inconclusive;
- negative upper CI: target-label geometry outperforms learned-teacher geometry.

Do not tune the auxiliary weight or add controls after seeing E003.

## Immediate bridge to tiny language models

After E003 is recorded, preregister H-010 as a genuine causal next-token study. The minimum core is:

1. ordinary next-token training;
2. output distillation;
3. output plus learned hidden-relation geometry;
4. output plus next-token target geometry;
5. an ordinary extra-data or extra-compute baseline.

Use a 1M–10M-parameter student, a small teacher, at least five paired seeds, a dominant simple-English stream, and a smaller controlled compositional stream. Freeze natural validation NLL, controlled accuracy by depth, teacher and student capability gates, token counts, model bytes, wall time, and peak memory before the fixed run. No activation-specific claim is allowed unless learned hidden geometry beats both output distillation and target-token geometry. No efficiency claim is allowed unless it survives the ordinary compute/data control.

Candidate public foundations include TinyStories for language that small models can learn, WikiText-2 as a small natural-text anchor, and Pythia's openly documented small checkpoints for calibration or architecture references. Dataset versions, licenses, tokenizer training data, and exact checksums must be frozen before use.

## Start-of-session checklist

1. Run `git status` and preserve unrelated work.
2. Read this file, the active preregistration, and the latest report.
3. State the one decision the run can change.
4. Freeze configuration and code in a clean commit before viewing a new result.
5. Run deterministic tests and record the configuration hash and Git commit.
6. Update the report and hypothesis ledger for positive, null, invalid, or negative outcomes.
7. Inspect the complete diff and secret scan before publishing.
