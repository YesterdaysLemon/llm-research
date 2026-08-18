# E001 — Trajectory sufficiency

## Identity

- Hypothesis: H-002
- Date registered: 2026-08-18
- Status: P0 pilot complete; full E001 hypothesis remains open
- Primary host: Ryzen 7 7700, 64 GB RAM, RTX 4060 8 GB
- Reduced host: CPU-only or Apple M2 with 16 GB unified memory

The completed [P0 report](report.md) and [raw metrics](results/pilot.json) are the current evidence. P0 found a consistent shallow-transfer signal, not general compositional transfer.

## P0 implementation

The first bounded pilot uses a synthetic affine-composition task. Each example applies a sequence of learned permutation-like relations to an entity. The teacher is trained on all ordered relation pairs. The student omits selected ordered pairs from training, and its out-of-distribution test uses those omitted pairs at familiar composition depths two through four.

- teacher: six-layer Transformer, 4,766,767 parameters;
- student: two-layer Transformer, 1,082,415 parameters;
- conditions: labels, logits, pointwise states, relational states, and example-shuffled relational states;
- seeds: 11, 22, and 33;
- relational target: normalized centered Gram matrices over examples at selected teacher layers;
- primary P0 metric: accuracy on excluded ordered relation pairs.

P0 deliberately does not claim matched target-byte budgets or complete coverage of the controls below. Those remain obligations for P1.

### Reproduce locally

From the repository root on a Windows CUDA host:

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r experiments\E001-trajectory-sufficiency\requirements-windows-cu130.txt
.venv\Scripts\python.exe -m unittest discover -s experiments\E001-trajectory-sufficiency\tests -v
.venv\Scripts\python.exe experiments\E001-trajectory-sufficiency\src\run_pilot.py --config experiments\E001-trajectory-sufficiency\config\pilot.json --output experiments\E001-trajectory-sufficiency\results\pilot-replication.json --device auto
```

Use a new output name for every fixed run. Do not overwrite `pilot.json`.

### Smoke-run amendment

The original smoke test withheld longer composition depths. The teacher reached only 3.8% accuracy on that split, so it failed the required positive control. Before the fixed comparison, the split was changed to held-out ordered relation pairs at familiar depths. Both result artifacts were preserved as [the invalid first smoke](results/smoke.json) and [the amended smoke](results/smoke-v2.json). No fixed pilot condition was inspected before this amendment.

The original invalid-smoke configuration bytes were overwritten during the amendment before they were committed. The raw result retains their SHA-256, while [`smoke-invalid-depth-reconstructed.json`](config/smoke-invalid-depth-reconstructed.json) records the reconstructed effective settings and is explicitly not claimed to reproduce that byte hash. This provenance mistake motivated the workflow rule that amendments receive new configuration and output names.

## Question

Do basis-invariant relations across a teacher's activation trajectories provide transferable information that helps a smaller student solve unseen compositions, beyond labels, logits, pointwise activations, and extra examples?

## Directional prediction

At matched student architecture, training examples, optimizer steps, and auxiliary-target bytes, relational-trajectory distillation will outperform output-only distillation on held-out composition depth. It will beat shuffled-trajectory and random-projection controls and retain its benefit after an orthogonal change of teacher basis.

## Phase A: synthetic mechanism test

Use a generated symbolic language with entities, directed relations, and composable rules. Examples contain an entity followed by a relation sequence. The broader design withholds selected relation compositions and longer proof depths while reusing the primitive vocabulary.

This environment is intentionally small. Its purpose is to identify a mechanism with clean ground truth, not to establish general LLM compression.

### Models

- teacher: six-layer decoder Transformer, approximately 15–30 million parameters;
- student: three-layer decoder Transformer, approximately 3–8 million parameters;
- same tokenizer and hidden width where useful for the pointwise baseline;
- a width-mismatched student in the second replication.

These ranges describe the planned full experiment. P0 used the smaller concrete models recorded above so that every condition and seed could run locally in minutes.

### Conditions

1. ground-truth labels only;
2. teacher hard labels;
3. teacher logits;
4. logits plus pointwise hidden-state matching;
5. logits plus true relational trajectories;
6. logits plus trajectories shuffled across examples;
7. logits plus matched-dimensional random projections;
8. condition 5 after an independently sampled orthogonal basis change at each teacher layer;
9. additional ground-truth examples matched to condition 5's stored-target bytes.

### Relational target

The initial target is the sequence of centered Gram matrices among examples at selected layers, plus the change between successive matrices. This represents which examples the teacher separates or brings together without requiring identical neuron coordinates.

No trajectory feature will be selected using the test set.

P0 used the normalized centered Gram matrix at each selected layer. It did not add a separate layer-to-layer delta term.

## Data splits

- training primitives and observed compositions;
- development compositions for hyperparameters;
- held-out pairs of known primitives;
- longer composition depths;
- renamed-symbol split to detect lexical memorization;
- contradiction and distractor split.

The generator seed and split hashes will be committed before the first comparison run.

## Primary metric

Accuracy on unseen compositions at greater depth than the training distribution.

## Secondary metrics

- in-distribution accuracy;
- teacher imitation KL divergence;
- calibration;
- performance by composition depth;
- stored supervision bytes;
- training time and joules;
- inference latency and joules;
- student parameters and serialized bytes.

## Required comparisons

All conditions use the same student initialization per seed, training examples, optimizer-step ceiling, and early-stopping rule. Run at least three seeds. Report every run.

The auxiliary loss coefficient may be tuned on development data using the same search budget for pointwise, relational, shuffled, and random conditions.

## Interpretation table

- **Relational beats logits and controls on held-out depth:** evidence for transferable trajectory structure.
- **Relational beats shuffled but fails after basis change:** coordinate-dependent signal, not the desired geometric invariant.
- **Relational improves imitation only:** better sample compression, not evidence of transferred algorithmic generality.
- **Extra examples match relational:** trajectories are useful supervision but not uniquely efficient.
- **No condition beats logits:** retire this relational target before scaling model size.

## Stop rule

Phase A ends after:

- one architecture smoke-test round;
- the fixed comparison with three seeds;
- one width-mismatched replication if the primary effect is positive.

Do not tune on the held-out composition set. Do not move to pretrained billion-parameter models unless the preregistered relational condition beats all three negative controls.

After P0, scaling is additionally gated on an improvement at depths three and four. A depth-two-only advantage is insufficient evidence of an algorithm worth scaling.

## Phase B: optional language-model replication

If Phase A is positive, repeat the smallest informative subset with an open teacher around 1–2B parameters and a student around 0.5B parameters using parameter-efficient training. The exact models will be selected based on license, tokenizer compatibility, activation access, and measured fit on the 8 GB GPU and 16 GB M2.

Phase B is a replication of the mechanism, not permission to weaken the Phase A controls.
