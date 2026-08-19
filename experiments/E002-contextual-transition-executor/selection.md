# E002 development selection lock

Locked: 2026-08-18 before confirmatory execution

## Selected values

- transition-table budget: 5 epochs;
- relational auxiliary weight: 3.0;
- Transformer budget: 48 epochs;
- GRU budget: 80 epochs;
- student training examples: 20,000;
- teacher training examples: 80,000;
- evaluation examples per split: 10,000.

The five-epoch table executor achieved 100% development-pair accuracy at depths two, three, and four for all three selection seeds.

Mean Transformer development-pair accuracies were:

| Objective | Accuracy |
| --- | ---: |
| Logits | 12.89% |
| Relational, weight 0.1 | 17.33% |
| Relational, weight 0.3 | 17.75% |
| Relational, weight 1.0 | 24.29% |
| Relational, weight 3.0 | 26.21% |

Weight 3.0 was selected by the prospective rule. Its mean accuracy by depth was 91.76%, 17.68%, and 3.05% at depths two, three, and four.

## Frozen configurations

- `confirm-affine.json` SHA-256: `f683ea11d6c5d854dcc94c19cd13d03765e4ceaa3c8579265623484dfe42b201`
- `confirm-bitwise.json` SHA-256: `5bb451f0db20fe6f2a686d17e2afe6fa905284a67cdc705f9c34f02597a225ae`

Any subsequent change requires a new filename, new hash, and explicit amendment. Neither file had been executed when these hashes were recorded.
