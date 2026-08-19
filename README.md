# Wobbly Intelligence

An early-stage research notebook about efficient learning systems that compress, adapt, forget, consolidate, cooperate, and consume energy.

The working question is not simply whether brains and language models are alike. It is:

> Which functions of biological and collective intelligence survive translation into testable, useful algorithms?

This repository begins with intuitions spanning cognitive science, neurophysiology, computer science, machine learning, and philosophy of mind. Those intuitions are starting points, not findings.

## Priority order

1. **Energy efficiency and capability per parameter**
2. **Continual learning and ecologies of models**
3. **Theory of mind as experimental motivation and an intuition pump**

The project does not presently treat machine consciousness as an engineering claim or near-term ethical finding.

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

- [Preprint](paper/preprint.md) reports the completed E001-E002 study, including the council-driven claim corrections, failed gates, and scope limits.
- [External council](research/council-2026-08-19.md) records the verified Opus 5 and Qwen 3.8 critique and its contribution to the project.
- [E004 design audit](research/council-2026-08-19-e004.md) records the verified pre-smoke blockers, prospective repairs, and the explicit Qwen timeout boundary.
- [E003 report](experiments/E003-label-geometry-specificity/report.md) records the completed teacher-specificity control.
- [E004 final diagnostic](experiments/E004-tiny-language-model/report-answer-weighted.md) records the failed tiny causal-LM capability gate, the shallow answer-weighting effect, and the benchmark retirement boundary.
- [E002 report](experiments/E002-contextual-transition-executor/report.md) is the compact experimental handoff.
- [NEXT](NEXT.md) is the current research-session entry point.
- [Original seed ideas](notes/seed-ideas.md) preserves the initial note.
- [Research priorities](notes/2026-08-18-priorities.md) records the first project decisions.
- [Landscape review](research/landscape.md) pressure-tests the note against research.
- [Compression frontier](research/compression-frontier.md) maps the strongest efficiency case and its failure modes.
- [Hypothesis ledger](hypothesis.md) turns the strongest ideas and counterideas into falsifiable proposals.
- [Experiments](experiments/README.md) defines how tests should be registered and reported.
- [Experiment workflow](experiments/WORKFLOW.md) defines the repeatable research loop.
- [E001-P0 report](experiments/E001-trajectory-sufficiency/report.md) is the exploratory pilot.
- [E001-P1 report](experiments/E001-trajectory-sufficiency/report-p1-development.md) records the failed capability gate.

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
    ├── NEXT.md                   Current decision and next bounded work
    ├── hypothesis.md             Falsifiable hypothesis ledger
    ├── notes/
    │   ├── seed-ideas.md         Preserved raw thinking
    │   └── 2026-08-18-priorities.md
    ├── research/
    │   ├── landscape.md          Broad evidence and counterevidence
    │   ├── compression-frontier.md
    │   └── council-2026-08-19.md Verified external-model audit
    ├── paper/
    │   ├── preprint.md            Canonical manuscript
    │   ├── analyze.py             Raw-result analysis and figures
    │   ├── build_pdf.py           Reproducible PDF build
    │   └── generated/             Derived statistics and figures
    ├── output/pdf/                Release-ready preprint PDF
    └── experiments/
        ├── README.md              Experiment registration template
        ├── WORKFLOW.md            Repeatable research-session loop
        ├── E001-trajectory-sufficiency/
            ├── README.md          Experiment design and amendments
            ├── report.md          Interpretation of the completed pilot
            ├── config/            Frozen machine-readable configurations
            ├── src/               Experiment implementation
            ├── tests/             Deterministic validity checks
            └── results/           Raw metrics and flat summaries
        ├── E002-contextual-transition-executor/
            ├── preregister.md     Frozen prospective protocol
            ├── report.md          Final compact handoff
            ├── src/ and tests/    Implementation and validity checks
            └── results/           Development and confirmatory metrics
        ├── E003-label-geometry-specificity/
            ├── preregister.md     Frozen target-label Gram control
            ├── config/            Single-condition confirmation
            ├── report.md          Registered decision and limitations
            └── results/           Raw metrics and generated analysis
        └── E004-tiny-language-model/
            ├── preregister.md     Two-stage causal-LM protocol
            ├── config/            Frozen smoke and later fixed configs
            ├── src/ and tests/    Dependency-light model, data, and checks
            └── results/           Preserved smoke and fixed artifacts

As the project grows, each implemented experiment should receive a numbered directory such as experiments/E004-tiny-language-model. Large models, private data, and intermediate generated artifacts should not be committed; compact paper figures and release PDFs are versioned for auditability.

## Project status

**Preprint / bounded synthetic evidence.** The affine confirmation supports a reproducible benefit from correspondence-preserving learned-teacher layer geometry. E003 shows that terminal-label equivalence does not explain that gain under the frozen objective, but the depth-four student remains below a credible capability floor and other causal accounts remain open. The 61.3-fold executor comparison proves a compact exact factorization exists on this task; it is not a general frontier or realized sparse-compute result. E004's first tiny causal-LM bridge stopped at its positive-control gate: answer weighting greatly improved shallow ID behavior, but no teacher/student gate passed and no geometry condition ran. No result yet establishes language-model distillation benefit, general compression, energy efficiency, continual learning, or consciousness claims.

## License

This repository is released under the [MIT License](LICENSE). Third-party
datasets and model artifacts retain their own licenses; each experiment records
those dependencies separately.
