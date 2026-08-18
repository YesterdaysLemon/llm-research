# E001-P1 prospective protocol

Registered: 2026-08-18, before any P1 model was evaluated

Status: development procedure frozen; confirmatory hyperparameters pending development-only selection

## Paper-shaped question

Does learned, basis-invariant teacher geometry transfer compositional behavior to a smaller Transformer beyond output distillation, coordinate matching, fake or untrained geometry, generic self-regularization, and additional ordinary labels at matched example exposures?

P1 is designed to distinguish a narrow data-fixed regularization effect from transferable algorithmic information. A negative or shallow-only result remains reportable.

## Task and split

The main task composes eight affine permutations over 47 entities. Every student-training example excludes both of two disjoint pair sets:

- development pairs: `(1,0)`, `(3,2)`, `(5,4)`, `(7,6)`;
- confirmatory pairs: `(0,1)`, `(2,3)`, `(4,5)`, `(6,7)`.

Development evaluation requires at least one development pair and excludes confirmatory pairs. Confirmatory evaluation requires at least one confirmatory pair and excludes development pairs. Both cover depths two, three, and four. In-distribution evaluation excludes both sets.

The teacher receives no pair exclusions. It must reach at least 95% on in-distribution, development, and confirmatory evaluations or the positive control fails and no student conclusion is drawn.

## Development-only choices

Development seeds are `201`, `202`, and `203`. They will never enter confirmatory summaries.

1. Evaluate logits-only students at 24, 48, and 72 epochs.
2. Choose the smallest budget reaching at least 80% mean in-distribution accuracy. If none reaches 80%, use 72 epochs and record the failed target.
3. At that budget, evaluate auxiliary weights `0.1`, `0.3`, `1.0`, and `3.0` for relational and pointwise matching.
4. Select each objective's weight by mean development-pair accuracy, breaking ties toward the smaller weight.
5. Apply the selected relational weight unchanged to shuffled, random, untrained-teacher, and self-relation controls because their losses share the same normalized-Gram scale.

No confirmatory metric may be generated before the selected budget and weights are written to the confirmatory configuration with a new filename and hash.

### Prospective amendment after the budget run

The registered two-layer student failed the 80% learnability target at every budget: its mean in-distribution accuracy plateaued near 44%, with depth-four accuracy near 5%, while the teacher exceeded 99%. These development results are preserved in `results/p1-dev-budget.json`.

Before any confirmatory evaluation, the architecture-selection rule is amended as follows:

1. test a four-layer, width-128 student at 72 epochs;
2. if it misses 80% mean in-distribution accuracy, test a four-layer, width-256 student;
3. use the first architecture that reaches 80%; if neither does, stop E001-P1 as an invalid student-capability design rather than interpreting trajectory effects.

This search changes depth while keeping the smaller candidate below P0's student parameter count. It is motivated by the observed depth-specific failure and uses development pairs only. Pointwise matching will be omitted if the selected width differs from the teacher because a learned projection would change the trainable-parameter budget; P0 remains the same-width pointwise comparison.

## Main confirmatory comparison

Confirmatory student seeds are `1101` through `1110`. Every ordinary condition begins from the same initialization within a seed and processes the same number of example exposures.

1. labels only;
2. output logits plus labels;
3. logits plus pointwise teacher states;
4. logits plus learned teacher relational geometry;
5. logits plus example-shuffled teacher geometry;
6. logits plus fixed random features of matched shape;
7. logits plus relational geometry from an untrained teacher;
8. logits plus a student layer-consistency relation with no teacher target;
9. labels on a four-times-larger training set, with epochs divided by four to match example exposures;
10. labels on the ordinary dataset with a predeclared longer budget approximating relational wall time.

Random, shuffled, and untrained targets remain fixed per example except that shuffled example correspondence is deterministically resampled per minibatch, as in P0. The report will distinguish this stochastic control from a single frozen permutation.

## Outcomes and decision rules

The primary outcome is accuracy on confirmatory-pair examples, paired by initialization seed.

Report paired mean differences with two-sided 95% Student-t confidence intervals. The relational target supplies confirmatory transfer evidence only if the lower confidence bound is above zero against both logits and shuffled geometry. The stronger algorithm-transfer interpretation additionally requires:

- positive paired lower bounds at depths three and four against both controls; and
- mean depth-four accuracy at least ten percentage points above chance.

The relational method supplies an efficiency advantage over ordinary supervision only if it beats the data-rich labels condition at matched example exposures after reporting actual tensor bytes and wall time. Parameter count, supervision bytes, peak allocated GPU memory, wall time, and energy are separate quantities. No joule claim will be made without power telemetry.

## Replications fixed in advance

Regardless of the main outcome, run two five-seed replications using seeds `2101` through `2105`:

1. the same affine task with a narrower, differently layered student;
2. a 64-entity bitwise-permutation task with the standard student.

Each replication includes at least logits, relational, shuffled, random, untrained-teacher, and data-rich-label conditions. Pointwise matching may be omitted when widths differ.

## Required ablations and diagnostics

- accuracy by composition depth;
- basis-invariance numerical check under independent orthogonal rotations;
- fixed-permutation versus per-minibatch shuffled targets;
- target and dataset byte accounting;
- peak GPU allocation and wall time;
- teacher positive-control performance;
- loss-scale and selected-weight table;
- complete seed-level results, including failures.

## Stopping rule

Stop experimental expansion after the main comparison, both fixed replications, and required ablations. Write the strongest claim justified by all cells. Do not add model families merely to rescue a preferred conclusion. A shallow-only, null, or efficiency-negative result is a valid endpoint for the preprint.
