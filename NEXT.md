# NEXT

Last updated: 2026-08-19

## Current evidence

The E001–E003 synthetic arc is complete. In the valid affine cell, adding correspondence-preserving learned-teacher encoder-layer geometry to labels and logits improved held-out-pair accuracy from 11.30% to 23.31% across ten paired seeds.

An independent Opus 5 and Qwen 3.8 council found that the earlier mechanistic wording exceeded the design. The target is indexed by encoder layer, not task-computation step; a target-label Gram was missing; the generic student had only 8.15% in-distribution accuracy at depth four; and the task-factored executor is a compact existence comparison rather than a causal context-selection result or a swept efficiency frontier.

E003 ran the council's cheapest decisive control once from a clean preregistration commit. Learned-teacher geometry beat target-label geometry by 17.44 percentage points, 95% CI [15.65, 19.23]. Target-label geometry underperformed logits alone by 5.43 points [-6.66, -4.19]. Terminal-answer equivalence therefore does not explain the learned-teacher gain under this setup. The [council record](research/council-2026-08-19.md), [E003 report](experiments/E003-label-geometry-specificity/report.md), and corrected [preprint](paper/preprint.md) are the current interpretive authorities.

The task-factored affine executor reached 100% with 17,672 stored parameters versus 1,082,927 for the relational Transformer. Its relation token semantically selects one table, but the reference implementation computes all table softmaxes before indexing. Stored count, semantic selection, realized compute, wall time, and energy must remain separate.

The bitwise cell remains descriptive because its teacher missed the registered positive-control gate.

## Active decision

The first E004 capability ladder is complete and failed: neither the teacher nor
labels-only student selected a passing rung through 21,600 steps. Do not run the
fixed comparison. Register and run one answer-weighted capability diagnostic
from a new clean commit; if its teacher still fails, retire this controlled
benchmark for H-010 rather than adding another rescue.

The pre-smoke Opus 5 audit and local replay found five blocking defects in the
first draft; all were repaired before training. Qwen 3.8 timed out twice without
a report, so no Qwen claim enters the design. The repaired protocol uses a
non-commutative six-operator affine language modulo 11, crosses every held-out
pair with every evaluation depth, starts controlled training at complete record
boundaries, evaluates in float32, and replaces the guessed 400/300-step budget
with a capability-only ladder at 400, 2,400, 7,200, and 21,600 steps. See the
[E004 protocol](experiments/E004-tiny-language-model/preregister.md) and
[design audit](research/council-2026-08-19-e004.md).

The [smoke report](experiments/E004-tiny-language-model/report-smoke.md) records
the result. Natural NLL improved strongly, but final teacher ID/held-out answer
accuracy was only 16.80%/12.89% and final student accuracy was 6.84%/7.62%
against 9.09% chance. Answer positions were 0.706% of all supervised positions.
The single diagnostic changes only their CE weight to `25`, approximately the
mean controlled-record length; every model, split, seed, gate, and ladder rung
stays frozen, and no auxiliary condition may run.

The experiment must decide whether learned hidden-relation geometry improves genuine next-token learning beyond both output distillation and next-token target geometry at matched student size, tokens, initialization, and optimizer steps.

The fixed comparison, if licensed by smoke, has this minimum core:

1. ordinary next-token training;
2. output distillation;
3. output plus learned hidden-relation geometry;
4. output plus next-token target geometry;
5. an ordinary extra-data or extra-compute baseline.

It also requires teacher-logit geometry and frozen untrained-teacher geometry
controls before a learned-hidden-specificity interpretation.

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
