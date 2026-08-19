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

**Status:** affine learned-teacher advantage beyond logits and terminal-label equivalence supported; broad two-family, causal-mechanism, and algorithm-content claims inconclusive; strong algorithm-transfer criterion not supported

**Origin:** learning emergent behavior from activations

**Confidence:** medium-low

**Claim.** At fixed student size, data, and training compute, a mixed objective that transfers teacher outputs plus basis-invariant relational structure across hidden-state trajectories will improve held-out compositional generalization more than output-only or pointwise hidden-state distillation.

**Why it might win.** Internal-representation and attention-relation distillation have improved compressed language models ([Aguilar et al., 2019](https://arxiv.org/abs/1910.03723); [Wang et al., 2020](https://arxiv.org/abs/2002.10957)). Relations among states may survive arbitrary basis differences better than pointwise activation matching.

**Strong rival.** The final teacher layer may mainly encode answer-class similarity, making the relational objective a form of supervised metric learning that can be reproduced directly from labels. Student and teacher can implement different valid algorithms, hidden-state constraints may waste capacity, and closer teacher matching need not improve generalization ([Stanton et al., 2021](https://arxiv.org/abs/2106.05945)).

**Minimum test.**

1. Use a teacher at least twice the student's parameter count.
2. Build output-only, pointwise-hidden, relational-trajectory, and mixed-loss students from the same initialization.
3. Match prompts, tokens, optimizer steps, and trainable parameters.
4. Evaluate both in-distribution imitation and held-out task compositions.
5. Add shuffled-trajectory, random-projection, and orthogonal-basis controls.
6. Add a teacher-free target-label Gram before attributing a gain to teacher-specific internal structure.
7. Intervene on the best-correlated hidden relation; do not rely on a probe alone.

**Primary measure.** Held-out accuracy per trainable parameter and training joule.

**Kill condition.** Reject the claim if the mixed objective fails to beat output-only distillation on held-out compositions across two task families, or if gains vanish under matched compute.

**Coordinate-artifact condition.** Reject the geometric interpretation if the method requires the teacher's original coordinate basis, cannot distinguish real from example-shuffled trajectories, or provides no gain beyond an equally sized random auxiliary target.

**Evidence.** In E001-P0, a relational Gram-matrix objective beat labels, logits, pointwise matching, and example-shuffled representations across three exploratory seeds. E002 then prospectively confirmed an affine behavioral effect: relational accuracy was 23.31% versus 11.30% for logits across ten paired seeds. The difference was +12.01 percentage points, 95% CI [10.16, 13.86]. A post hoc depth analysis found that relational training retained more of its above-chance in-distribution capability on held-out pairs than logits did, but relational in-distribution depth-four accuracy was only 8.15%. The registered strong-algorithm floor failed, while the reason for deep failure remains inconclusive. A fixed bitwise cell was descriptively favorable but invalid because its teacher missed the positive-control gate. E003 then prospectively tested the council's target-label Gram rival. Learned-teacher geometry beat label geometry by +17.44 points [15.65, 19.23], while label geometry underperformed logits by -5.43 points [-6.66, -4.19]. Terminal-label equivalence therefore does not explain the affine effect under the frozen setup. See the [preprint](paper/preprint.md), [council audit](research/council-2026-08-19.md), and [E003 report](experiments/E003-label-geometry-specificity/report.md).

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

## H-008 — Context-selected parameter reuse beats trajectory supervision

**Status:** inconclusive for the registered cross-family claim; affine executor criteria passed; bitwise cell invalid

**Origin:** capability per parameter, context-aware parameter definitions, and the depth-specific E001 failure

**Confidence:** medium on finite-state composition; low beyond structured domains

**Claim.** On compositional transition tasks, a compact model that selects a relation-specific operator from context and reuses the same execution rule at every step will achieve higher held-out-composition accuracy per stored and active parameter than a generic Transformer student, even when the Transformer receives teacher logits or relational trajectories.

**Why it might win.** The architecture factors knowledge into transition operators and an iteration rule instead of asking a fixed-depth encoder to rediscover sequential execution. Only one operator is semantically selected per step, so stored parameters, selected structure, realized computation, and repeated computation become distinct quantities.

**Strong rival.** The executor may win only because its structure nearly specifies the synthetic task. It may fail on ambiguous natural language, learned state spaces, or tasks whose useful decomposition is unknown. A lookup transition table can also hide poor scaling in the number of states.

**Minimum test.** Compare labels, logits, and relational supervision for a generic Transformer and a weight-tied transition executor on affine and bitwise permutation families, withheld ordered pairs, multiple depths, data-size curves, and a no-single-step-supervision ablation. Match examples and report stored parameters, semantically selected parameters, measured operations or profiler traces, tensor bytes, wall time, and accuracy separately.

**Primary measure.** Held-out composition accuracy per stored parameter, with absolute accuracy and depth-four accuracy as hard floors.

**Kill condition.** Reject the general efficiency interpretation if the executor advantage disappears on the second task family, requires direct labels for every primitive transition, or is erased by a comparably sized generic recurrent baseline.

**Evidence.** On the valid affine confirmation, the transition-table executor reached 100% at every depth with 17,672 parameters, versus 23.31% for the 1,082,927-parameter relational Transformer. It eventually reached 100% without primitive-transition examples in an exploratory ten-epoch diagnostic, while the small GRU reached only 9.44%. This is a compact existence comparison, not a swept frontier or causal test of context selection. The reference implementation computes all table softmaxes before indexing, so semantic selection is not realized sparse compute. The fixed bitwise table also reached 100% descriptively, but the family cell was invalid under the registered teacher gate. Cross-family support therefore remains inconclusive.

## H-009 — Geometry-guided operator discovery

**Status:** proposed from the joint E001-E002 evidence

**Origin:** learned-teacher geometry transfer plus the task-aligned executor advantage

**Confidence:** low-medium on latent finite-state tasks; low beyond structured domains

**Claim.** A modular recurrent learner that jointly discovers latent state slots, a small context-selected operator library, and a reusable execution rule will generalize to unseen relation pairs and longer depths more efficiently than a generic Transformer or GRU. Relational teacher geometry will improve operator discovery compared with labels and logits alone, even though geometry by itself did not transfer the full algorithm.

**Why it might win.** E002 separates two observations: learned-teacher geometry improves a generic student's behavior, while a correctly factored executor supplies exact iteration. Combining a learned factorization with repeated execution could turn geometry into a routing or state-discovery signal instead of asking it to serve as a program.

**Strong rival.** The finite-state factorization may still be doing nearly all the work. A learned modular system could collapse to arbitrary slots, require hidden supervision, or lose its advantage when observations obscure entity identity. Relational geometry might reproduce the same shallow correlations seen in E002 without improving operator recovery or long-depth behavior.

**Minimum test.**

1. Generate at least two transition families with latent states observed through frozen random relabelings or noisy feature maps rather than direct entity IDs.
2. Compare a generic Transformer, a matched GRU, and a modular recurrent student with learned state slots and soft operator routing.
3. For the modular student, compare labels, teacher logits, and logits plus relational geometry from paired initializations.
4. Train through depth four; freeze evaluation at depths six and eight and on unseen ordered relation pairs.
5. Use at least three task-generation seeds and five training seeds, with a teacher positive-control gate and a modular labels-only capability gate.
6. Report absolute accuracy, stored and selected parameters, operator-use entropy, and factor-recovery metrics after permutation alignment. Treat recovery as secondary to behavior.

**Primary measure.** Held-out depth-six/eight accuracy per stored parameter, subject to an absolute accuracy floor fixed from development.

**Kill condition.** Reject the geometry-guided discovery claim if relational supervision does not improve the modular student over output-only supervision across both frozen families, if long-depth accuracy remains near chance, or if the method requires direct latent-state or operator labels.

## H-010 — Activation-specific distillation survives next-token controls

**Status:** E004 capability smoke preregistered; no language-model training result

**Origin:** applying H-002 to genuine causal language modeling as early as possible

**Confidence:** low-medium for a training-signal effect; low for an efficiency advantage

**Claim.** At fixed tiny causal-LM architecture, tokens, optimizer steps, and initialization, output distillation plus learned-teacher hidden-relation geometry will improve both ordinary held-out next-token loss and controlled compositional completion more than output distillation alone or output distillation plus target-token class geometry.

**Why it might win.** Hidden relations can expose similarities among contexts before those similarities collapse into the next-token distribution. Prior language-model distillation work has benefited from transferring hidden or attention relations, so the affine effect may survive the move from terminal classification to causal prediction.

**Strong rival.** The hidden Gram may mostly encode the same next-token classes already present in labels and logits. Any gain may come from a larger auxiliary tensor, more optimizer work, or a synthetic controlled stream rather than teacher-specific structure. A small student may also lack the capability needed to make out-of-distribution comparisons interpretable.

**Minimum test.**

1. Train or freeze one small causal teacher and one 1M–10M-parameter student with a compact tokenizer on a versioned corpus mixing a dominant simple-English stream with a smaller controlled compositional stream.
2. Compare ordinary next-token training, output distillation, output plus learned hidden relations, output plus next-token target geometry, and an ordinary extra-data or extra-compute baseline.
3. Pair at least five student seeds and match token order, optimizer steps, and initialization across conditions.
4. Freeze a natural-text validation loss, controlled in-distribution accuracy, unseen-composition accuracy by depth, model bytes, training wall time, and peak memory.
5. Require the teacher and labels-only student to clear predeclared capability gates before interpreting controlled extrapolation.
6. Treat a parameter-count gain as separate from measured latency or energy; use supported telemetry before making a joule claim.

**E004 registration.** The first implementation audit found process-dependent
held-out sampling, depth/pair confounding, train/evaluation position mismatch,
an underived smoke budget, and a commutative controlled algebra before any
training occurred. E004 now uses order-sensitive affine operators modulo 11,
balanced depth-by-pair evaluation, complete boundary-aligned records, and a
capability-only budget ladder. A successful ladder changes only the experiment's
validity state; H-010 remains untested until the frozen paired KD/geometry study
runs.

**Primary measure.** The joint result on natural-text validation NLL and capability-gated held-out controlled composition. The activation-specific claim requires learned hidden relations to beat both output distillation and target-token geometry under paired uncertainty intervals.

**Kill condition.** Reject activation-specific benefit under this design if learned hidden relations fail to beat target-token geometry, if any gain disappears against the extra-compute/data baseline, or if the student capability gate fails. A failed gate triggers redesign of the benchmark, not a positive or negative mechanism claim.

## Priority order

1. **H-010** — move immediately into a tiny causal language model while carrying E003's specificity and ordinary-compute controls.
2. **H-009** — test whether operator structure can be discovered rather than hand-specified.
3. **H-008 replication / H-007** — independently gate and then scale the efficiency claims.
4. **H-006** — establish honest energy measurement and a strong systems baseline.
5. **H-003** — ask whether durable learning requires weight change.
6. **H-001** — convert the sleep analogy into a bounded algorithmic test.
7. **H-005** — test the ecology claim after a reliable task harness exists.
8. **H-004** — most speculative; run only with strong controls.

## Open decisions

- Review the council-corrected E001-E003 preprint before treating the manuscript as submission-ready.
- Freeze the exact tiny-LM corpus mix, tokenizer, teacher/student sizes, and capability gates for H-010.
- Define the first frozen broad-capability suite and its difficult-tail subset before making a general compression claim.
- Choose direct wall-power measurement or hardware telemetry for the M2 and RTX 4060 hosts.
