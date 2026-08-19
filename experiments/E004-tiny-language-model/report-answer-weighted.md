# E004 answer-weighted capability diagnostic

Run: 2026-08-19
Frozen commit: `74c18976a6692e9952372535739849da683844ff`
Configuration SHA-256: `893488f25cc7ae24ae2f64044944a998e6f234618afaf2713391bc0927a5aa1a`
Result SHA-256: `b497d4a1909c76e3be7374f4969413141097ee91abfef1c8bcacdaae4d6a0851`

## Registered decision

After the ordinary capability smoke failed, this diagnostic changed exactly one
thing: controlled answer-token cross-entropy received weight `25`, approximately
the frozen mean record length. Every model, seed, token stream, split, gate,
optimizer, evaluation, and ladder rung stayed fixed. No KD or geometry condition
ran.

**Decision: answer weighting established strong shallow capability but neither
model cleared the registered positive controls. The controlled affine benchmark
is retired for H-010, and the fixed KD/geometry study must not run.**

## Final-rung comparison

Chance answer accuracy is 9.09%. Values are from the same frozen model seeds and
data schedules, but the two runs are not a multi-seed causal estimate.

### Teacher at 21,600 steps

| Endpoint | Ordinary CE | Answer-weighted CE | Difference |
| --- | ---: | ---: | ---: |
| Natural validation NLL | 2.1197 | 2.1288 | +0.0091 |
| ID overall | 16.80% | 43.36% | +26.56 pp |
| ID depth 2 | 29.24% | 90.06% | +60.82 pp |
| ID depth 3 | 11.70% | 26.32% | +14.62 pp |
| ID depth 4 | 9.41% | 13.53% | +4.12 pp |
| Held-out overall | 12.89% | 17.58% | +4.69 pp |
| Held-out depth 2 | 17.97% | 28.91% | +10.94 pp |
| Held-out depth 3 | 8.59% | 20.31% | +11.72 pp |
| Held-out depth 4 | 11.72% | 10.94% | -0.78 pp |
| Held-out depth 6 | 13.28% | 10.16% | -3.12 pp |

The weighted teacher's four held-out-pair accuracies were 14.84%, 10.16%,
21.88%, and 23.44%. It missed the 70% ID-overall gate, the 50%-per-depth ID
gate, the 55% held-out depth-two-through-four gate, and the 30%-per-pair gate.

### Student at 21,600 steps

| Endpoint | Ordinary CE | Answer-weighted CE | Difference |
| --- | ---: | ---: | ---: |
| Natural validation NLL | 2.2090 | 2.2517 | +0.0427 |
| ID overall | 6.84% | 48.44% | +41.60 pp |
| ID depth 2 | 4.09% | 88.30% | +84.21 pp |
| ID depth 3 | 5.85% | 46.20% | +40.35 pp |
| ID depth 4 | 10.59% | 10.59% | 0.00 pp |
| Held-out overall | 7.62% | 16.41% | +8.79 pp |
| Held-out depth 2 | 5.47% | 26.56% | +21.09 pp |
| Held-out depth 3 | 13.28% | 21.88% | +8.59 pp |
| Held-out depth 4 | 4.69% | 10.94% | +6.25 pp |
| Held-out depth 6 | 7.03% | 6.25% | -0.78 pp |

The student's primary held-out mean over depths two through four was 19.79%,
just below the registered 20% corridor floor. More importantly, its ID overall
and depth-four gates failed.

## Interpretation

The diagnostic supports one narrow claim: ordinary token-averaged CE severely
underweighted the rare answer position. Giving that position record-scale weight
created large depth-two ID gains in both models with almost no teacher natural-NLL
change. Objective dilution was therefore not merely a post hoc story.

It did not create a portable multi-step computation. Accuracy fell steeply with
depth, the teacher remained weak at depth three and near chance at depth four,
and held-out gains were much smaller than ID gains. This pattern is compatible
with learning shallow templates or local associations rather than an iterative
state-update rule. Architecture, optimization, data coverage, and the controlled
grammar itself remain confounded; the run does not identify a mechanism.

The weighted teacher recorded 1,321.33 training seconds versus 587.45 for the
ordinary run, but the GPU was heavily contended during its final rung. That
timing difference is invalid for efficiency inference. Peak PyTorch CUDA
allocation was effectively unchanged at 384,377,856 bytes. The driver exposed
no power telemetry, so no energy result is available.

## Consequence and next hypothesis

Per the registered stop rule, do not increase answer weight, mixture, model
size, or budget and do not run hidden-geometry conditions on this benchmark.
The next tiny-LM bridge should separate two questions:

1. On natural text alone, does learned hidden relation geometry improve held-out
   NLL beyond KD, target-token geometry, teacher-logit geometry, and an
   untrained-teacher geometry control?
2. On a future controlled task, can the teacher first demonstrate the complete
   iterative behavior under an ordinary, separately validated training setup?

The first is a smaller and cleaner language-model specificity test. It cannot
claim compositional transfer, but it does not make a failed synthetic positive
control the gatekeeper for whether hidden geometry affects language modeling at
all.
