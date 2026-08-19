# E002 report — context-selected transition execution

Date: 2026-08-18

Status: stopped at the preregistered boundary; affine confirmation valid, bitwise replication invalid; interpretation narrowed after external audit on 2026-08-19

## Decision

Correspondence-preserving learned-teacher layer geometry improves the generic Transformer's held-out affine behavior relative to output distillation. The registered strong algorithm-transfer criterion is not supported, but the student's low in-distribution depth-four capability prevents a mechanistic conclusion about a learned-but-nonportable algorithm. A task-factored transition executor solves the affine task exactly with 61.3 times fewer stored parameters. The fixed bitwise cell is descriptive only because its teacher missed the registered positive-control gate.

Read the full [preprint](../../paper/preprint.md) for methods, related work, uncertainty intervals, figures, and limitations.

## Registered affine confirmation

The teacher passed its gate with 99.25% held-out-pair accuracy overall and 98.45% at depth four.

Across ten paired student seeds:

- relational Transformer: 23.31% +/- 2.02%;
- logits-only Transformer: 11.30% +/- 1.82%;
- shuffled-geometry Transformer: 6.65% +/- 1.45%;
- transition-table executor: 100.00% with 17,672 parameters;
- relational Transformer: 1,082,927 parameters.

The paired relational-minus-logits effect was +12.01 percentage points, 95% CI [10.16, 13.86]. Relational minus shuffled was +16.66 [15.04, 18.28]. Both registered intervals are positive, so the affine relational-transfer claim is supported.

Depth-four relational accuracy was only 3.51% against 2.13% chance. It missed the registered chance-plus-ten-point absolute floor, so strong algorithm transfer is not supported. A post hoc audit found only 8.15% in-distribution depth-four accuracy for this student. The result therefore does not distinguish missing algorithmic information from a capacity, optimization, or architectural ceiling.

The original controls establish that learned, example-aligned geometry matters, but they do not distinguish teacher-specific hidden structure from terminal-label similarity. E003 prospectively registers that missing control.

## Bitwise replication gate

The frozen bitwise teacher reached 94.46% overall but only 89.21% at depth four. The protocol required at least 95% overall and at every depth. The cell is therefore invalid, despite descriptively favorable student results. No rescue run or retuning was performed.

## Efficiency interpretation

The affine transition table uses 61.28 times fewer stored parameters than the relational Transformer. One table containing 2,209 parameters is semantically selected per relation step, but the current forward pass computes the softmax of all eight tables before indexing. The earlier approximately 490-fold comparison is therefore not a realized-compute result. The executor trained about 24 times faster in this implementation; neither wall time nor parameter count is an energy measurement.

The executor is deliberately privileged with the correct finite-state factorization and scales quadratically in the number of states. Its result is an existence proof about task representation, not a swept frontier, causal context-selection result, or general architecture claim.

## Next hypothesis

First run E003's target-label Gram specificity control. Then move the geometry question into a tiny causal language model with output-only, target-token geometry, learned hidden geometry, and ordinary extra-compute/data baselines. A separate architecture study can ask whether a learner discovers a compact operator library and reusable execution rule. None has yet been tested.
