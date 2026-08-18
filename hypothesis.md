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

## Project objective

The primary optimization target is capability per parameter and per joule. Continual learning and model ecologies are secondary. Theory of mind supplies hypotheses and experimental distinctions; it is not evidence that current systems are conscious.

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

**Status:** proposed; E001-P0 pilot is suggestive but does not establish the claim

**Origin:** learning emergent behavior from activations

**Confidence:** medium-low

**Claim.** At fixed student size, data, and training compute, a mixed objective that transfers teacher outputs plus basis-invariant relational structure across hidden-state trajectories will improve held-out compositional generalization more than output-only or pointwise hidden-state distillation.

**Why it might win.** Internal-representation and attention-relation distillation have improved compressed language models ([Aguilar et al., 2019](https://arxiv.org/abs/1910.03723); [Wang et al., 2020](https://arxiv.org/abs/2002.10957)). Relations among states may survive arbitrary basis differences better than pointwise activation matching.

**Strong rival.** The trajectory may only be meaningful when interpreted by the teacher's weights. Student and teacher can implement different valid algorithms, hidden-state constraints may waste capacity, and closer teacher matching need not improve generalization ([Stanton et al., 2021](https://arxiv.org/abs/2106.05945)).

**Minimum test.**

1. Use a teacher at least twice the student's parameter count.
2. Build output-only, pointwise-hidden, relational-trajectory, and mixed-loss students from the same initialization.
3. Match prompts, tokens, optimizer steps, and trainable parameters.
4. Evaluate both in-distribution imitation and held-out task compositions.
5. Add shuffled-trajectory, random-projection, and orthogonal-basis controls.
6. Intervene on the best-correlated hidden relation; do not rely on a probe alone.

**Primary measure.** Held-out accuracy per trainable parameter and training joule.

**Kill condition.** Reject the claim if the mixed objective fails to beat output-only distillation on held-out compositions across two task families, or if gains vanish under matched compute.

**Coordinate-artifact condition.** Reject the geometric interpretation if the method requires the teacher's original coordinate basis, cannot distinguish real from example-shuffled trajectories, or provides no gain beyond an equally sized random auxiliary target.

**Pilot evidence.** In E001-P0, a relational Gram-matrix objective beat labels, logits, pointwise matching, and example-shuffled trajectories across three seeds on held-out ordered relation pairs. Mean held-out accuracy was 8.77% versus 4.23% for logits and 3.88% for shuffled trajectories. However, the advantage was concentrated in shallow compositions and depth-four performance remained near chance. This supports a weaker claim that relational trajectories can transfer some local structure; it does not yet support transfer of a general composition algorithm. See the [report](experiments/E001-trajectory-sufficiency/report.md).

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

## H-007 — Tail-preserving multiplicative compression

**Status:** proposed

**Origin:** capability per parameter and the suspicion of large avoidable redundancy

**Confidence:** medium for one order of storage reduction; low for multiple orders

**Claim.** A compression stack combining low-bit weights, learned structured sparsity, and trajectory-aware distillation can reduce stored model bytes by at least 8× and measured inference energy by at least 2× while preserving a predeclared broad capability floor better than ordinary output distillation at the same final size.

**Why it might win.** Quantization, pruning, and distillation attack different redundancies and can have complementary effects ([Movva et al., 2022](https://arxiv.org/abs/2208.09684)). Relational trajectory targets may protect rare distinctions that magnitude-based pruning or logits alone discard.

**Strong rival.** The easy benchmark mass is compressible but the difficult tail is capacity-bound. Unstructured sparsity may save bytes without saving energy, and the trajectory objective may only rearrange which capabilities are lost.

**Minimum test.**

1. Freeze a baseline teacher, broad evaluation suite, and difficult-tail subset.
2. Measure FP16, eight-bit, four-bit, structured-pruned, distilled, and combined variants.
3. Compare output-only and relational-trajectory distillation at the same student architecture and byte budget.
4. Measure model bytes, active parameters, wall time, peak memory, and joules separately.
5. Include calibration, rare compositions, corrupted inputs, and one continual-learning probe.

**Primary measure.** Hypervolume of the capability-versus-bytes-versus-joules Pareto frontier subject to a hard per-task capability floor.

**Kill condition.** Reject the 8×/2× target if no combined model meets the frozen capability floor on two model families, or if the energy gain disappears when measured on supported dense or structured-sparse kernels.

## Priority order

1. **H-002** — tests whether activation trajectories contain transferable structure rather than coordinates.
2. **H-007** — turns that mechanism into a capability-per-parameter and per-joule target.
3. **H-006** — establishes honest energy measurement and a strong systems baseline.
4. **H-003** — asks whether durable learning requires weight change.
5. **H-001** — converts the sleep analogy into a bounded algorithmic test.
6. **H-005** — tests the ecology claim after a reliable task harness exists.
7. **H-004** — most speculative; run only with strong controls.

## Open decisions

- Define the first frozen broad-capability suite and its difficult-tail subset.
- Decide how much average-score gain may compensate for a loss on one rare task; the default should be “none” until justified.
- Choose direct wall-power measurement or hardware telemetry for the M2 and RTX 4060 hosts.
- Decide whether the first large-model replication uses same-family teacher/student architectures or deliberately different ones.
