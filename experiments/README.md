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

## First experiment

[E001 — Trajectory sufficiency](E001-trajectory-sufficiency/README.md) tests whether basis-invariant relations across teacher activation trajectories transfer compositional behavior to a smaller student beyond what labels, logits, and pointwise activations provide.

Its first pilot is complete. Read the [E001-P0 report](E001-trajectory-sufficiency/report.md) and [raw result](E001-trajectory-sufficiency/results/pilot.json).

E001-P1 then stopped before confirmation because every registered student missed its learnability gate. The [development report](E001-trajectory-sufficiency/report-p1-development.md) records that failure without treating floor effects as a trajectory test.

[E002 — Context-selected transition execution](E002-contextual-transition-executor/README.md) follows the resulting architectural hypothesis: separate stored parameters from the context-selected operator applied at each composition step. Its [prospective protocol](E002-contextual-transition-executor/preregister.md) freezes the confirmatory comparison.

## Working loop

Use [WORKFLOW.md](WORKFLOW.md) for every experiment. The short version is:

1. preregister the decision and losing result;
2. commit or hash the configuration;
3. use smoke runs only to catch invalid designs or broken code;
4. run the frozen comparison;
5. preserve raw metrics without overwriting them;
6. report what changed, what did not, and what remains unsupported.
