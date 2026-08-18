# E001-P1 development report — capability gate failed

Date: 2026-08-18

Status: stopped before confirmation

## Decision

E001-P1 did not proceed to trajectory comparisons. Every registered student candidate failed the prospective 80% in-distribution learnability gate, while the teacher passed its positive control. The confirmatory relation-pair set and confirmatory seeds were never evaluated.

This prevents a floor-effect comparison from being presented as a test of transferred algorithms.

## Development protocol

The affine task used eight relation operators and excluded disjoint development and confirmatory ordered-pair sets from student training. Only the development-pair set was evaluated. The teacher had no exclusions.

The initial procedure tested a two-layer, width-256 student for 24, 48, and 72 epochs. After its depth-specific plateau, a prospective amendment allowed a four-layer width-128 student, followed by one four-layer width-256 candidate if necessary. All variants used three development seeds.

## Positive control

The 4,767,279-parameter teacher reached:

- 99.60% in-distribution accuracy;
- 99.57% development-pair accuracy;
- 99.21% accuracy at development depth four.

The task was therefore learnable by the model family, but not by the registered students.

## Student results

Values are mean plus or minus sample standard deviation across development seeds 201–203.

| Student | Parameters | Epochs | In-distribution | Development pairs |
| --- | ---: | ---: | ---: | ---: |
| 2 layers × 256 | 1,082,927 | 24 | 43.22% ± 0.19 | 4.50% ± 0.63 |
| 2 layers × 256 | 1,082,927 | 48 | 43.73% ± 0.32 | 4.59% ± 0.59 |
| 2 layers × 256 | 1,082,927 | 72 | 43.84% ± 0.05 | 5.11% ± 0.97 |
| 4 layers × 128 | 807,471 | 72 | 43.68% ± 0.31 | 3.27% ± 0.13 |
| 4 layers × 256 | 2,137,135 | 72 | 43.62% ± 0.13 | 3.98% ± 0.81 |

For the two-layer 72-epoch student, mean in-distribution performance was approximately 89% at depth two, 25% at depth three, and 5% at depth four. More epochs, twice the depth, and nearly twice the parameter count did not remove that pattern.

Raw results:

- [`p1-dev-budget.json`](results/p1-dev-budget.json)
- [`p1-dev-arch-4x128.json`](results/p1-dev-arch-4x128.json)
- [`p1-dev-arch-4x256.json`](results/p1-dev-arch-4x256.json)

## Interpretation

The generic encoder students appear to exploit shallow sequence correlations without acquiring a reusable sequential transition rule. That makes P0's relational advantage more specific: it improved a weak student's local held-out-pair behavior, but it was observed in a regime with almost no depth-four competence.

This does not show that the smaller parameter counts are intrinsically insufficient. The invariance of the plateau across depth, width, and training duration instead motivates an architectural baseline in which a context-selected transition operator is explicitly reused at each composition step.

## Consequence

H-002 remains unresolved and E001-P1 is closed as an invalid capability design. E002 will compare the generic Transformer against a compact, weight-tied finite-state executor. If the structured model generalizes with far fewer parameters and labels alone, architectural reuse is a stronger explanation of compositional efficiency than trajectory supervision on this task.
