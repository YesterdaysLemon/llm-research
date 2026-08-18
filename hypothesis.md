# Hypothesis ledger

Created: 2026-08-18

Status key: proposed, preregistered, running, supported, not supported, inconclusive, retired

These are testable descendants of the original seed ideas. A failed hypothesis is useful information and should remain in the ledger.

## Shared experimental rules

- Freeze the task suite, scoring code, and exclusion rules before training.
- Compare against simple baselines at matched data, parameter-update, and inference budgets.
- Use at least three random seeds for training claims unless the cost is prohibitive and explicitly reported.
- Keep development tasks separate from held-out and out-of-distribution tasks.
- Report absolute scores, uncertainty intervals, wall time, peak memory, and measured or estimated energy.
- Do not call a representation causal unless an intervention changes the predicted behavior.

## H-001 — Interference-aware consolidation

**Status:** proposed

**Origin:** sleep, replay, and continual learning

**Confidence:** medium

**Claim.** Under a fixed replay budget, selecting old examples or latent traces by measured interference with new learning will yield a better retention-adaptation Pareto frontier than uniform replay or naive sequential fine-tuning.

**Why it might win.** Replay reduces catastrophic forgetting, and hidden-representation replay has worked on continual-learning benchmarks ([van de Ven et al., 2020](https://www.nature.com/articles/s41467-020-17866-2)). Prioritizing items whose gradients or representations conflict with the new task should spend limited replay compute where forgetting risk is highest.

**Strong rival.** A cheap proxy for interference may be noisy or circular. Uniform replay may cover the old distribution better, and the selector's overhead may erase any gain.

**Minimum test.**

1. Choose one open 1B–4B model and two independently scored task families A and B.
2. Train the same LoRA setup on A, then B.
3. Compare no replay, uniform replay, loss-prioritized replay, and interference-prioritized replay.
4. Match replay examples, optimizer steps, and evaluation calls.
5. Repeat with A/B order reversed and with three seeds.

**Primary measure.** Area under the retention-versus-new-task-gain curve.

**Kill condition.** Retire the present selector if it does not beat uniform replay beyond uncertainty on both task orders, or if its total compute-adjusted gain is negative.

## H-002 — Relational trajectory distillation

**Status:** proposed

**Origin:** learning emergent behavior from activations

**Confidence:** medium-low

**Claim.** At fixed student size, data, and training compute, a mixed objective that transfers teacher outputs plus relational structure among selected hidden states will improve held-out compositional generalization more than output-only distillation.

**Why it might win.** Internal-representation and attention-relation distillation have improved compressed language models ([Aguilar et al., 2019](https://arxiv.org/abs/1910.03723); [Wang et al., 2020](https://arxiv.org/abs/2002.10957)). Relations among states may survive arbitrary basis differences better than pointwise activation matching.

**Strong rival.** Student and teacher can implement different valid algorithms. Hidden-state constraints may waste capacity, and closer teacher matching need not improve generalization ([Stanton et al., 2021](https://arxiv.org/abs/2106.05945)).

**Minimum test.**

1. Use a teacher at least twice the student's parameter count.
2. Build output-only, pointwise-hidden, relational-hidden, and mixed-loss students from the same initialization.
3. Match prompts, tokens, optimizer steps, and trainable parameters.
4. Evaluate both in-distribution imitation and held-out task compositions.
5. Intervene on the best-correlated hidden relation; do not rely on a probe alone.

**Primary measure.** Held-out accuracy per trainable parameter and training joule.

**Kill condition.** Reject the claim if the mixed objective fails to beat output-only distillation on held-out compositions across two task families, or if gains vanish under matched compute.

## H-003 — Hybrid memory beats weights that “never stop growing”

**Status:** proposed; adversarial to a favored seed

**Origin:** frozen snapshots and lifelong learning

**Confidence:** medium-high

**Claim.** For streams dominated by new facts and episodic experience, a frozen model with external memory plus occasional gated consolidation will dominate continuous weight updates on retention, editability, and compute.

**Why it might win.** External memory can add persistent knowledge without catastrophic parameter forgetting; fixed-base continual-memory results already support this possibility ([Li et al., 2024](https://arxiv.org/abs/2412.07393)). Facts that may change are easier to correct in an inspectable store than inside weights.

**Strong rival.** Retrieval can fail, add latency, and integrate knowledge only shallowly. Skills and abstractions may require parameter change.

**Minimum test.**

1. Stream timestamped facts, reversals, and a small set of transferable procedures.
2. Compare context only, retrieval memory, sequential LoRA, LoRA plus replay, and a hybrid that consolidates only repeated or procedural information.
3. Test immediate recall, delayed recall, contradiction handling, procedure transfer, and deletion.
4. Include distractor memories and paraphrased queries.

**Primary measure.** A Pareto frontier over accuracy, forgetting, update cost, deletion success, and latency.

**Kill condition.** Reject the broad hybrid claim if continuous weight updating wins on both factual and procedural streams at matched compute without worse deletion or retention.

## H-004 — Calibrated noise helps consolidation

**Status:** proposed

**Origin:** wet, wobbly, messy intelligence

**Confidence:** low

**Claim.** Small, calibrated noise during offline replay will improve robustness and reduce representational interference, while zero noise and high noise will perform worse.

**Why it might win.** Biological and artificial nonlinear systems can exhibit stochastic facilitation ([McDonnell & Ward, 2011](https://www.nature.com/articles/nrn3061)). Noise may encourage flatter solutions or prevent replay from copying brittle traces.

**Strong rival.** Familiar regularizers may explain the entire effect, or noise may simply corrupt scarce replay information.

**Minimum test.**

1. Hold the replay stream and optimizer fixed.
2. Sweep zero, low, medium, and high activation noise; run a separate gradient-noise sweep.
3. Include dropout and weight decay as ordinary regularization controls.
4. Test clean retention, corrupted inputs, calibration, and out-of-distribution transfer.

**Primary measure.** Predeclared composite of old-task retention and corrupted-input accuracy.

**Kill condition.** Reject the inverted-U claim if the optimum is consistently zero noise, if effects do not replicate across seeds, or if ordinary regularization explains them at lower cost.

## H-005 — Structured diversity creates collective gain

**Status:** proposed

**Origin:** intelligence in an ecology

**Confidence:** medium

**Claim.** Under the same total token and wall-clock budgets, independent proposals from epistemically diverse models followed by evidence-based adjudication will outperform both single-model self-consistency and unstructured multi-model conversation.

**Why it might win.** Human groups show measurable collective-intelligence differences ([Woolley et al., 2010](https://pubmed.ncbi.nlm.nih.gov/20929725/)). Independence can preserve diverse error modes, while adjudication can combine complementary evidence.

**Strong rival.** Communication and coordination create process loss. Model diversity may be cosmetic if all systems share training data and correlated blind spots.

**Minimum test.**

1. Freeze a mixed suite with verifiable answers and research questions scored against a hidden rubric.
2. Compare one model with self-consistency, homogeneous independent samples, heterogeneous independent samples, structured adjudication, and free-form group chat.
3. Match total generated tokens and cap tool calls.
4. Measure pairwise error correlation and communication overhead, not only final accuracy.

**Primary measure.** Accuracy and calibration per total token, with error-correlation reduction as a mediator.

**Kill condition.** Reject the diversity claim if structured heterogeneous groups do not beat the strongest matched-budget single-model baseline, or if any gain is fully explained by extra compute.

## H-006 — Conditional compute closes more of the energy gap than substrate mimicry

**Status:** proposed; counterweight to the wet-versus-silicon framing

**Origin:** brain efficiency

**Confidence:** medium

**Claim.** On a mixed-difficulty workload, routing easy items to a small model and escalating uncertain items will improve joules per accepted answer more than adding bio-inspired noise or continual weight plasticity to a single always-on model.

**Why it might win.** Workloads vary greatly in difficulty, while model size and output length strongly affect inference cost. Historical algorithmic gains have already reduced compute needed for fixed capabilities ([Hernandez & Brown, 2020](https://arxiv.org/abs/2005.04305)).

**Strong rival.** Reliable routing itself may require an expensive model, and small-model errors may be confidently wrong.

**Minimum test.**

1. Create an easy/medium/hard task mixture with hidden labels.
2. Compare small-only, large-only, confidence routing, learned routing, and a fixed random routing control.
3. Calibrate routers on a development split only.
4. Measure wall power where possible; otherwise report hardware-specific energy estimates separately from measured values.

**Primary measure.** Joules per answer meeting a fixed quality and calibration threshold.

**Kill condition.** Reject the routing claim if it cannot match large-model quality with lower total energy after router and retry costs.

## Priority order

1. **H-003** — cheapest conceptual discriminator: does durable learning need weight change?
2. **H-001** — converts the sleep analogy into a bounded algorithmic test.
3. **H-002** — closest to the original activation-distillation idea.
4. **H-006** — establishes honest energy measurement early.
5. **H-005** — tests the ecology claim after a reliable task harness exists.
6. **H-004** — most speculative; run only with strong controls.

## Questions that should change this ledger

- Is the project's main optimization target continual adaptation, energy, capability per parameter, or a theory of mind?
- Is “consciousness” meant as an engineering mechanism, a phenomenon to detect, or an ethical concern?
- What kinds of experience may the model learn from: raw conversation, explicit approval, environmental reward, or curated datasets?
- Which properties must remain stable while the system learns: factual reliability, personality, safety policy, skills, or all of them?
- What result would make us abandon the idea that evolving representation geometry is central?
- Are local experiments constrained to a particular GPU, RAM budget, operating system, or maximum run time?
