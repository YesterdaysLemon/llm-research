# E001-P0 report — relational trajectories transfer shallow structure

Date: 2026-08-18

Status: completed pilot; suggestive, not confirmatory

## Question

Can basis-invariant relations among a teacher's hidden states help a smaller student generalize to excluded relation compositions beyond labels, output logits, or pointwise state matching?

This pilot tests a narrow mechanism. It does not test natural-language intelligence, lossless model compression, consciousness, or measured energy efficiency.

## Design

The task composes synthetic affine relations over 47 entities. The teacher sees all ordered relation pairs. Student training excludes selected ordered pairs, while the out-of-distribution test requires those pairs at familiar depths two through four.

- teacher: six layers, 4,766,767 parameters;
- student: two layers, 1,082,415 parameters;
- parameter ratio: 4.40 to 1;
- student seeds: 11, 22, and 33;
- fixed configuration: [`config/pilot.json`](config/pilot.json);
- configuration SHA-256: `e78918acb39a84360a47e2933374282abac4a2bdaab404ad0877d4915266e1fd`;
- raw results: [`results/pilot.json`](results/pilot.json) and [`results/pilot.csv`](results/pilot.csv).

The five student conditions were:

1. ground-truth labels;
2. teacher output logits;
3. logits plus pointwise hidden-state matching;
4. logits plus normalized centered Gram matrices of teacher states;
5. logits plus the same relational targets shuffled across examples.

The Gram target preserves pairwise geometry under orthogonal changes of basis. The shuffled condition asks whether a real correspondence between examples and teacher geometry matters.

## Amendment before the fixed run

The first smoke split withheld longer depths, but the teacher achieved only 3.8% there. That made the intended positive control invalid. The split was amended to withheld ordered relation pairs at familiar depths before the fixed comparison. Both [the failed smoke](results/smoke.json) and [the amended smoke](results/smoke-v2.json) remain available.

The failed smoke's original configuration bytes were accidentally overwritten while amending the split. Its raw result retains the original hash, and [a reconstructed effective configuration](config/smoke-invalid-depth-reconstructed.json) records the known settings without pretending to match that hash. This is a provenance failure in P0, not a scientific result; future amendments must use new config filenames.

The fixed teacher then reached 98.72% in-distribution accuracy and 98.90% on excluded pairs, so failure of the students cannot be blamed on an incapable teacher.

## Results

Chance accuracy is 2.13%. Values are mean plus or minus sample standard deviation across three seeds.

| Condition | In-distribution accuracy | Excluded-pair accuracy | Training time |
| --- | ---: | ---: | ---: |
| Labels | 47.84% ± 0.16 | 3.21% ± 0.36 | 7.29 s |
| Logits | 48.30% ± 0.36 | 4.23% ± 0.61 | 7.63 s |
| Pointwise | 49.75% ± 0.65 | 5.33% ± 0.31 | 8.24 s |
| Relational | **53.20% ± 0.43** | **8.77% ± 1.33** | 9.26 s |
| Shuffled relational | 48.17% ± 0.33 | 3.88% ± 0.64 | 9.36 s |

Relational minus shuffled excluded-pair accuracy was +3.80, +5.32, and +5.54 percentage points for seeds 11, 22, and 33. Relational minus logits was +2.60, +4.54, and +6.48 points. The direction therefore held for every seed in this pilot.

The relational condition took about 21% longer to train than logits. That is wall time, not energy. GPU power draw was unavailable through the current Windows WDDM telemetry path, so this experiment supports no joule-efficiency claim.

### Where the effect lived

| Condition | Depth 2 | Depth 3 | Depth 4 |
| --- | ---: | ---: | ---: |
| Labels | 5.6% | 2.5% | 2.0% |
| Logits | 8.9% | 3.7% | 2.2% |
| Pointwise | 12.2% | 4.7% | 2.6% |
| Relational | **23.8%** | **8.8%** | **3.4%** |
| Shuffled relational | 8.9% | 3.3% | 2.1% |

The relational advantage is strongest at depth two, smaller at depth three, and nearly gone at depth four. This is evidence for transfer of shallow pair structure, not for transfer of a general composition algorithm.

## What we learned

1. **The real teacher relation mattered.** Shuffling the same kind of target removed most of the gain, so the result is not explained merely by adding an auxiliary loss of that shape.
2. **Relations beat coordinates in this setup.** The relational condition outperformed pointwise state matching despite using an invariant summary rather than exact teacher coordinates.
3. **The transferred information was local.** Performance by depth argues against the exciting interpretation that the student inherited a recursive algorithm.
4. **Absolute generalization remained poor.** An 8.77% mean is above chance and the controls, but it still represents failure on more than nine out of ten excluded-pair examples.
5. **The efficiency question remains open.** The teacher-to-student parameter ratio is meaningful, but P0 neither matches supervision bytes nor measures energy.

## What this does not rule out

The teacher target may be a useful task-specific regularizer rather than a compact carrier of an algorithm. It may also reveal information that extra ordinary labels could teach more cheaply. P0 lacks matched-dimensional random targets, an untrained-teacher target, a student-self-relation control, and additional labels matched to the stored trajectory bytes.

Other limitations include the synthetic task, three seeds, one teacher/student family, the shared hidden width, and invariance only to orthogonal basis transformations. No causal intervention was performed.

## Hypothesis update

H-002 remains **proposed**. P0 supports only the weaker hypothesis:

> On this task, teacher-derived relational geometry transfers some local composition structure more effectively than logits, pointwise coordinates, or shuffled geometry at the tested training budget.

It does not establish general relational-trajectory distillation, lossless compression, capability per parameter across tasks, or capability per joule.

## Next decisive test

E001-P1 should keep the task and primary split fixed while adding:

1. matched-dimensional random targets;
2. relations from an untrained teacher;
3. student self-relations without teacher information;
4. extra labeled examples matched to the relational target's stored bytes;
5. a development-only auxiliary-weight sweep frozen before evaluation.

Scaling to a pretrained language model should wait until the relational condition survives these controls and improves depths three and four.
