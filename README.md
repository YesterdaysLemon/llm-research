# Wobbly Intelligence

An early-stage research notebook about learning systems that live, adapt, forget, consolidate, cooperate, and consume energy.

The working question is not simply whether brains and language models are alike. It is:

> Which functions of biological and collective intelligence survive translation into testable, useful algorithms?

This repository begins with intuitions spanning cognitive science, neurophysiology, computer science, machine learning, and philosophy of mind. Those intuitions are starting points, not findings.

## Current research threads

- continual learning and the stability-plasticity tradeoff
- replay and consolidation as precise versions of the sleep analogy
- contextual and time-varying representation geometry
- distillation of internal dynamics from larger to smaller models
- external memory versus learning in model weights
- stochasticity, robustness, and the possible computational value of noise
- energy per capability rather than vague brain-versus-datacenter comparisons
- collective intelligence, coordination gains, and process losses
- substrate independence and the limits of claims about consciousness

## Start here

- [Original seed ideas](notes/seed-ideas.md) preserves the initial note.
- [Landscape review](research/landscape.md) pressure-tests the note against research.
- [Hypothesis ledger](hypothesis.md) turns the strongest ideas and counterideas into falsifiable proposals.
- [Experiments](experiments/README.md) defines how tests should be registered and reported.

## Epistemic rules

Every substantial statement should be recognizable as one of:

1. **Observation** — a measured result with a source.
2. **Interpretation** — an explanation compatible with the evidence.
3. **Analogy** — a prompt for mechanism discovery, not evidence of shared mechanism.
4. **Hypothesis** — a claim with a test and a possible losing result.
5. **Result** — a completed experiment with code, environment, controls, and limitations.

The project will:

- preserve raw ideas before revising them;
- seek the strongest rival explanation, not only confirming citations;
- compare methods at matched data, compute, and evaluation budgets;
- report retention, adaptation, calibration, latency, and energy separately;
- use held-out and out-of-distribution evaluations where possible;
- keep intelligence, agency, and consciousness as distinct concepts;
- treat biological inspiration as a source of algorithms, not proof that biology must be copied.

## Repository layout

    .
    ├── README.md                 Project map and research norms
    ├── hypothesis.md             Falsifiable hypothesis ledger
    ├── notes/
    │   └── seed-ideas.md         Preserved raw thinking
    ├── research/
    │   └── landscape.md          Evidence, counterevidence, and reformulations
    └── experiments/
        └── README.md              Experiment registration and reporting template

As the project grows, each implemented experiment should receive a numbered directory such as experiments/E001-continual-memory. Large models, private data, and generated artifacts should not be committed.

## Project status

**Exploratory / pre-experimental.** No hypothesis in this repository is currently established by original experimental evidence.

## License

No license has been selected yet. Public visibility does not by itself grant permission to reuse the work.
