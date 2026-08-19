# E003 report — label-geometry specificity

Date: 2026-08-19

Status: complete; teacher gate passed; registered learned-teacher-specificity decision supported on the affine task

## Prediction and decision rule

E003 tested whether E002's learned-teacher layer-geometry gain could be explained by the simpler relation “examples with the same terminal answer should be nearby.” The only new condition received the same labels-plus-logits base loss and the same auxiliary weight as E002 relational training, but its target Gram came from one-hot terminal labels repeated at both student layers.

The primary contrast was paired E002 learned-teacher geometry minus E003 target-label geometry across the frozen seeds `5101`–`5110`. A positive 95% Student-t lower confidence bound supports a learned-teacher advantage beyond terminal-label equivalence under this setup.

## Validity and provenance

- Run commit: `5701ce36dab29aa63ebda324e467a8ca489dd984`.
- Recorded pre-run Git status: clean.
- Configuration SHA-256: `a78736731718cbe4c610d75e0474261ea768d6df5c6310187307759761ed944c`.
- E003 exactly preserved E002's affine data, teacher, training settings, and ten student seeds.
- The teacher gate passed at 99.25% overall, with 100.00%, 99.88%, and 98.44% at depths two, three, and four.
- All ten registered student runs completed; no seed was excluded or rerun.

## Absolute results

| Condition | ID overall | Held-out overall | Depth 2 | Depth 3 | Depth 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| E002 learned-teacher geometry | 68.55% | 23.31% | 77.41% | 18.39% | 3.51% |
| E003 target-label geometry | 60.03% | 5.87% | 16.53% | 4.17% | 2.52% |
| E002 logits only | 61.77% | 11.30% | 41.69% | 5.61% | 2.39% |

Target-label geometry scored 5.87% +/- 0.93% held-out accuracy across seeds. Its in-distribution accuracy was 60.03% +/- 0.25%. In-distribution accuracy also collapsed with depth: 87.27%, 26.53%, and 5.20% at depths two, three, and four.

## Registered contrasts

Learned-teacher minus target-label geometry was:

- **overall: +17.44 percentage points, 95% CI [15.65, 19.23];**
- depth two: +60.88 [53.28, 68.47];
- depth three: +14.22 [12.69, 15.75];
- depth four: +0.99 [0.57, 1.42].

The primary lower bound is positive, so E003 supports the registered statement: **learned teacher layer geometry supplies a behavioral advantage beyond terminal-label equivalence under this frozen objective and task.**

The secondary target-label-geometry-minus-logits contrast was -5.43 points, 95% CI [-6.66, -4.19]. Target-label geometry therefore harmed overall transfer relative to the same labels-plus-logits base loss. Its depth-two and depth-three effects were also negative; the depth-four interval included zero.

## Interpretation

E003 rules out the council's simplest rival. The E002 gain is not reproduced by merely grouping training examples according to their terminal answer. Together with E002's positive learned-teacher-minus-logits effect, this makes the useful information more specific to learned, example-aligned teacher representations than the corrected pre-E003 manuscript could claim.

The experiment does **not** prove that the Gram target contains an iterative program. Both E002 and E003 students remain incapable at in-distribution depth four. Nor does it prove a unique causal mechanism: learned teacher geometry could expose task difficulty, intermediate features, input neighborhoods, or other information not reducible to terminal-label identity. The label Gram is also a harmful rather than neutral control, so its low absolute score must not inflate the cleaner learned-teacher-minus-logits evidence.

## Efficiency and runtime boundary

Mean E003 student training time was 86.55 seconds with a 76.62-second sample SD. The first runs experienced severe host/GPU contention, while later runs returned near 40–50 seconds. Optimizer steps and example exposures were unchanged, so the behavioral comparison remains valid. Cross-run wall-time or energy comparisons are invalid for E003 and no efficiency inference is drawn.

## Decision and next test

Update H-002: an affine learned-teacher geometry advantage beyond logits and terminal-label equivalence is supported. Cross-family generality, algorithm content, causal mediation, and compute efficiency remain inconclusive.

Move next to a tiny causal language model. Its mandatory core conditions are ordinary next-token learning, output distillation, learned hidden-relation geometry, next-token target geometry, and an ordinary extra-data or extra-compute baseline. The tiny-LM design must gate both the teacher and labels-only student before interpreting controlled compositional transfer.

