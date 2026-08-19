# E002 — Context-selected transition execution

**Status:** complete at the preregistered stopping boundary. Read the [final report](report.md) and [preprint](../../paper/preprint.md).

## Question

Does explicit reuse of a context-selected transition operator produce more compositional capability per parameter than a generic Transformer, and does relational teacher supervision still help once the student architecture can execute the task?

## Motivation

E001-P1 found that Transformer students from 0.81M to 2.14M parameters all plateaued near 44% in-distribution accuracy while a 4.77M teacher exceeded 99%. The invariant depth-specific failure suggests that the encoder students never learned a reusable transition rule.

E002 separates three notions that an ordinary parameter count conflates:

1. **stored parameters** — every learned scalar in the model;
2. **context-selected parameters** — the relation operator chosen for the current step;
3. **reused computation** — applying the same execution rule once per relation token.

## Development models

- a generic Transformer encoder;
- a weight-tied GRU executor with no explicit transition-table structure;
- a finite-state executor with one learned entity-transition matrix per relation and a shared iteration rule.

The transition-table model is deliberately strong structure. It is not presented as a general language architecture. Its role is to test whether the apparent need for trajectory supervision vanishes when the task decomposition is represented directly.

## Development gate

Labels-only training includes single-relation examples as primitive supervision and compositions through depth four. All student training excludes the development and future confirmatory ordered pairs.

At least one compact executor must reach 90% accuracy on development-pair compositions at depths three and four before E002 proceeds to a frozen comparison of labels, logits, relational targets, and controls. Otherwise this task family remains unsuitable for the paper's efficiency claim.

The no-single-relation-supervision ablation is mandatory after a capable executor is identified.
