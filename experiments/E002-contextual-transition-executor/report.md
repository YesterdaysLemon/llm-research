# E002 report — context-selected transition execution

Date: 2026-08-18

Status: stopped at the preregistered boundary; affine confirmation valid, bitwise replication invalid

## Decision

Relational teacher geometry transfers local compositional structure to the generic Transformer in the affine task, but it does not transfer a depth-general iteration algorithm. A task-aligned context-selected transition executor solves the affine task exactly with 61.3 times fewer stored parameters. The fixed bitwise cell is descriptive only because its teacher missed the registered positive-control gate.

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

Depth-four relational accuracy was only 3.51% against 2.13% chance. It missed the registered chance-plus-ten-point absolute floor, so strong algorithm transfer is not supported.

## Bitwise replication gate

The frozen bitwise teacher reached 94.46% overall but only 89.21% at depth four. The protocol required at least 95% overall and at every depth. The cell is therefore invalid, despite descriptively favorable student results. No rescue run or retuning was performed.

## Efficiency interpretation

The affine transition table uses 61.28 times fewer stored parameters and approximately 490 times fewer context-selected parameters per step than the relational Transformer. It trained about 24 times faster in this implementation. These are not energy measurements.

The executor is deliberately privileged with the correct finite-state factorization and scales quadratically in the number of states. Its result is an existence proof about task representation, not a general architecture claim.

## Next hypothesis

A learner that discovers a compact operator library and reusable execution rule may outperform direct trajectory imitation. Relational geometry could help discover the states or routing assignments, while recurrence supplies the missing algorithmic depth. This has not been tested.
