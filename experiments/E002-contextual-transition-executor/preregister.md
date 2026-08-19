# E002 prospective protocol

Registered: 2026-08-18 after one development smoke seed and before confirmatory evaluation

## Development observation motivating confirmation

On one affine development seed, a 17,672-parameter transition-table executor reached 100% on held-out ordered pairs through depth four. A 9,647-parameter GRU reached 7.46%, and a 1,082,927-parameter Transformer reached 8.62%. These are development observations, not confirmatory evidence.

## Development-only selections

1. Evaluate the table executor at 5, 10, 20, 40, and 80 epochs on seeds 401–403. Select the smallest budget with at least 95% mean development accuracy at both depths three and four.
2. Evaluate relational auxiliary weights `0.1`, `0.3`, `1.0`, and `3.0` for the Transformer on the same development seeds. Select mean development accuracy, breaking ties toward the smaller weight.
3. Apply the selected relational weight unchanged to shuffled, random, and untrained-teacher geometry controls.
4. Run the table executor without depth-one training examples at the selected budget. This is an ablation and cannot change the confirmatory configuration.

The selected five-epoch no-depth-one ablation failed on all three development seeds. An explicitly exploratory extension at 10, 20, 40, and 80 epochs will test whether primitive transitions are unidentifiable or merely slower to infer. This post-observation diagnostic cannot alter the five-epoch confirmatory budget or the registered decision rules.

Development selected five epochs for the transition table and auxiliary weight `3.0` for relational objectives. The confirmatory datasets contain 20,000 student-training examples, 80,000 teacher-training examples, 10,000 in-distribution evaluation examples, and 10,000 confirmatory-pair examples per task family. These counts were fixed before either confirmatory configuration was executed.

## Main confirmatory cell

- family: affine permutations over 47 entities;
- new data seed: 5150;
- student seeds: 5101–5110;
- training depths: one through four;
- evaluation depths: two through four;
- training excludes development pairs `(1,0)`, `(3,2)`, `(5,4)`, `(7,6)` and confirmatory pairs `(0,1)`, `(2,3)`, `(4,5)`, `(6,7)`;
- confirmatory evaluation requires the confirmatory pairs and excludes the development pairs.

Conditions:

1. transition-table executor, labels;
2. comparably sized weight-tied GRU, labels;
3. generic Transformer, labels;
4. Transformer, labels plus teacher logits;
5. Transformer, logits plus learned teacher relational geometry;
6. Transformer, logits plus per-minibatch shuffled geometry;
7. Transformer, logits plus fixed random geometry of matched shape;
8. Transformer, logits plus geometry from an untrained teacher.

The six-layer teacher must exceed 95% overall and at every confirmatory depth. Otherwise the cell is invalid.

## Fixed replication

Repeat all eight conditions with five seeds, 6101–6105, on 64 entities transformed by bit rotations followed by XOR masks. Use data seed 27182 and the same pair split. No hyperparameter may be retuned on the replication family.

## Primary outcomes

- absolute confirmatory-pair accuracy;
- accuracy at depths three and four;
- stored parameters;
- context-selected active parameters per composition step;
- accuracy per stored parameter, reported only alongside the absolute capability floor.

Wall time, peak allocated GPU memory, example exposures, supervision tensor bytes, and unavailable energy telemetry are secondary and remain separate.

## Decision rules

The structured-efficiency claim is supported only if the transition executor:

1. exceeds 95% at depths three and four in both task families;
2. beats both generic baselines in absolute held-out accuracy;
3. uses at least 20 times fewer stored parameters than the Transformer.

The relational-transfer claim is supported in the main cell only if paired 95% Student-t confidence intervals for relational minus logits and relational minus shuffled accuracy both exclude zero on the positive side. The stronger algorithm-transfer claim additionally requires positive intervals at depths three and four and mean depth-four accuracy at least ten percentage points above chance.

## Mandatory limitations

The table executor encodes the finite-state factorization and scales quadratically with the state count. Success cannot be generalized to language models, unknown latent states, energy efficiency, or continual learning. Direct primitive supervision and task-structure dependence must be discussed even if the registered numerical criteria pass.

## Stopping rule

After development selection, the main cell, fixed bitwise replication, no-depth-one ablation, and prespecified statistics, stop expanding E002. Write the preprint around the joint E001–E002 evidence without adding rescue architectures or datasets.
