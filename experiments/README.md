# Experiments

Each experiment gets an immutable ID and its own directory:

    experiments/
    └── E001-short-name/
        ├── README.md
        ├── config/
        ├── src/
        ├── tests/
        └── results/

Do not create all folders in advance. Add them when the experiment needs them.

## Registration template

### Identity

- Experiment:
- Hypothesis:
- Owner:
- Date registered:
- Status:

### Claim

State one directional prediction.

### Baselines and controls

- strongest simple baseline:
- matched budgets:
- negative control:
- positive control:
- ablations:

### Data

- source and license:
- train/development/test split:
- contamination risks:
- private or sensitive fields:

### Metrics

- primary metric:
- secondary metrics:
- uncertainty method:
- energy measurement method:
- exclusion rules:

### Stop rule

Define maximum runs, seeds, compute, and the condition that ends exploration.

### Result

Record environment, commit, complete run table, failures, and caveats. Link raw artifacts by content hash or release location rather than committing large files.

## First recommended experiment

Start with H-003 on a synthetic but adversarial stream of timestamped facts, reversals, and compositional procedures. It requires less interpretability machinery than hidden-state distillation and directly tests whether “not frozen” must mean “weights keep changing.”
