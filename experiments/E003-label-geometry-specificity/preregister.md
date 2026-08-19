# E003 preregistration — target-label geometry specificity

Registered: 2026-08-19, before executing the frozen configuration.

## Motivation and exact decision

E002 showed that logits plus learned teacher layer geometry beat logits alone on held-out affine relation pairs. The strongest unresolved rival is that the teacher Gram matrices mainly provide answer-class similarity: examples with the same terminal state are pulled together and examples with different terminal states are separated. Such a signal would still be useful, but it would not show that teacher activations contain distinctive transferable structure beyond labels.

E003 asks one question: **does learned teacher geometry outperform a target-label Gram constructed without teacher hidden states?**

## Frozen data and reference conditions

E003 reuses the affine data generator, split definitions, data seed `5150`, teacher seed `801`, and student seeds `5101` through `5110` from E002. It does not modify or rerun the committed E002 reference conditions. The frozen references are:

- `transformer-logits`;
- `transformer-relational`;
- the affine teacher positive control.

The configuration SHA-256 and clean Git commit will be recorded automatically in the new result artifact.

## New condition

The only new condition is `transformer-label-geometry`. It uses the same two-layer width-256 Transformer, initialization seeds, 48 epochs, examples, minibatch order, optimizer, learning rate, labels-plus-logits base loss, auxiliary weight `3.0`, and relational-loss implementation as `transformer-relational`.

For each training example with terminal target `y`, construct the one-hot vector `e_y` over the 47 entity classes. Repeat `e_y` at both student layers. Within each minibatch and layer, center and normalize the Gram matrix exactly as in E002. The auxiliary loss matches the student's layer Gram to this target-label Gram.

This target contains terminal-label equivalence and minibatch composition, but no teacher hidden state. It is not information-matched in bytes to the learned-teacher cache, and no byte-efficiency claim will be made.

## Gates

The deterministic affine teacher must reproduce the E002 positive-control rule: at least 95% overall and at every confirmatory depth. If it fails, E003 is invalid and no student inference is drawn.

No new student capability gate is introduced. In-distribution accuracy overall and by depth is mandatory context for every comparison, and no algorithm-transfer inference may be drawn at a depth where the new student is below 50% in-distribution accuracy.

## Statistics and decisions

All contrasts use paired seed-level accuracy differences and two-sided 95% Student-t confidence intervals.

The primary contrast is:

```text
E002 transformer-relational minus E003 transformer-label-geometry
```

- If its lower confidence bound is above zero, record evidence that learned teacher layer geometry supplies a behavioral advantage beyond terminal-label equivalence under this objective.
- If its interval contains zero, record the teacher-specificity question as inconclusive.
- If its upper confidence bound is below zero, record that target-label geometry outperformed learned teacher geometry under the frozen setup.

The secondary contrast is label geometry minus E002 logits. A positive lower bound shows that explicit batchwise target-class geometry improves held-out composition beyond the same labels-plus-logits base loss. It does not establish a teacher-activation mechanism.

Report the same contrasts at depths two, three, and four as secondary analyses. Also report in-distribution accuracy by depth and the descriptive normalized-transfer ratio `(heldout - chance) / (in_distribution - chance)` where the denominator is positive. No multiplicity adjustment or cross-task generalization claim is registered.

## Stop rule

Run the single condition for all ten seeds once. After validation, write the result regardless of sign. Do not tune the auxiliary weight, add another target, change the non-capability gate, or rerun selected seeds. A future tiny-language-model experiment receives its own preregistration and may use this outcome to choose which controls are mandatory.

