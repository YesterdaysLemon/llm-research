# Self-play pretraining with zero data — critical reading

Date: 2026-09-26

## Scope and provenance

- **Paper.** Cowsik, Dolev, Li, De Luca, Cohen, Goodman, and Levine, *Self-Play Pretraining with Zero Data*, [arXiv:2609.30063v1](https://arxiv.org/abs/2609.30063), 24 Sep 2026. Read in full (32 pages).
- **Released artifacts.** The authors' [Solomonoff-Figures](https://github.com/nourya-aliz/Solomonoff-Figures) repository, pinned at commit `4b6d46725d9c4f42bd5fa78debbd2a2c960a6614`. It ships the scored data behind every figure and table, plus scoring code for checkpoints hosted on Hugging Face (`nourya-cohen/solomonoff-paper`). The self-play training code is not released.
- **What was done here.** Nothing was trained or re-scored. Every number below is either quoted from the paper (cited by section, table, or figure) or re-derived from the shipped scored data by [`self-play-pretraining/reanalyze.py`](self-play-pretraining/reanalyze.py). Its complete output is [`self-play-pretraining/reanalysis.json`](self-play-pretraining/reanalysis.json), and each derived number below names the JSON key it comes from.
- **Not accessed.** The Hugging Face checkpoints were unreachable under this session's network policy, so no released weight was evaluated.
- **Estimates.** Compute and runtime figures in the replication section are planning estimates, not measurements.

Labels follow the project's epistemic rules: **O** observation, **I** interpretation, **A** analogy, **H** hypothesis.

## What the paper does

A generator and a learner, both small Llama-style byte-level transformers from random initialization, co-evolve.

- **Generator.** Writes programs for a Brainf*ck-like universal machine: 8 primitives, 10 macro tokens (Table 4), and an end token. Each program runs with a random input tape and emits up to 4,096 bytes.
- **Learner.** Next-byte prediction on those outputs.
- **Generator training.** GRPO-style policy gradient with KL regularization toward the uniform program prior, plus reward-weighted SFT. Each round's pool mixes fresh samples, mutations drawn from a MAP-Elites archive, and replayed programs (§2.2, App. G).
- **Reward (Eq. 2).** `r = |⟨∇L(y; θ_e), P_e ⊙ (θ_⌊e/2⌋ − θ_e)⟩|`, where `P_e` is the AdamW step operator. To first order this is the absolute change in the learner's loss on `y` between the lookback checkpoint and now, i.e. absolute learning progress, evaluated on a fresh sample.
- **Evaluation.** Zero-shot bits per byte on 26 natural and synthetic corpora, compute-optimal frontiers over model size (99k–24.4M), round, and ensemble size K, in-context-learning probes, and a warm-start comparison.

## Observations

Values are bits per byte (bpb; 8.0 is a uniform byte guess) unless stated.

**O1 — Self-play beats i.i.d. programs from the same machine everywhere.** DCLM zero-shot: 4.39 self-play vs 6.91 fixed prior, best over all compute (`matched_compute_best_bpb.dclm`). Negating the reward drives every corpus above 7.6 bpb, including 10.6 on DCLM (Table 5).

**O2 — In-context learning is the strongest result.**
- 24M 4-seed ensemble at round 2816: first/last 1.00, reverse string 1.00, stack 0.98, associative recall 0.99, sum mod 256 0.88, max 0.69, min 0.66, mean 0.13.
- Fixed-prior learner of the same size: at most 0.03 on every task.
- Random-PCFG learner: associative recall 1.00, stack 0.32, all others at most 0.16 (`icl_best_accuracy`).
- Self-play's distinctive gains are therefore reverse string, stack, sum, and max/min, not associative recall.

**O3 — Random PCFGs beat self-play on text and code at matched learner compute.** The paper states this (§3.1).

| Corpus, C ≤ 1e18 | Self-play | PCFG | Fixed prior |
| --- | ---: | ---: | ---: |
| DCLM | 4.39 | 3.74 | 6.91 |
| GitHub Python | 2.92 | 2.52 | 6.60 |
| enwik9 text | 4.02 | 3.51 | 7.01 |
| Metamath | 2.87 | 2.73 | 6.80 |
| 8-bit audio | 2.19 | 5.02 | 3.40 |
| CIFAR-10 planar | 5.79 | 7.55 | 7.36 |

Source: `matched_compute_best_bpb`. The PCFG ladder stops at 6M parameters. The compute axis counts the learner only, excluding generator RL, JVP rewards, and program execution.

**O4 — Text frontiers end a decade before the runs do** (`frontier_end`).
- DCLM, Python, and Common Crawl: the last frontier point is the 24M model at round 768 with K=8, at C ≈ 9.4e17. Metamath's is at 1.3e18.
- Scored runs extend to 1e19.
- Audio, CIFAR, and melody frontiers continue to about 6–7e18.
- DCLM loss of the 24M model (`dclm_24M_trajectory`):
  - K=8 ensemble: 4.39 at round 768, then 4.67–4.85 over rounds 2048–8191.
  - Single model: 5.97 at round 768, rising to about 6.4 by round 1536, then falling to 5.75–5.82 late.
- The frontier's running minimum hides this non-monotonicity.

**O5 — The exponent depends on the fit form and the data version** (`exponents`). For DCLM on the shipped trajectory:

| Fit | Exponent |
| --- | ---: |
| Saturating (with fitted floor E = 2.30) | 0.119 |
| Pure power law | 0.074 |
| Last two decades | 0.066 |

- The literature range quoted for text in Table 2 is 0.048–0.099.
- The shipped fits differ from the paper's Table 2 on several corpora:
  - DNA: 0.264 shipped vs 0.435 in the paper.
  - Metamath: 0.164 vs 0.129.
  - C source: 0.158 vs 0.116.
  - Audio 8-bit: 0.212 vs 0.260.
- The mean over 25 corpora is 0.248 saturating vs 0.011 late slope.

**O6 — The ensemble carries much of the text transfer.** At 24M, round 768, DCLM is 5.97 for one model and 4.39 for the K=8 probability ensemble (`dclm_24M_trajectory`). The paper only says ensembling "is useful" (§3.1).

**O7 — The reward's measured contribution depends on when it is measured.** At the 1M rung with K=4, we report the fraction of the fixed-prior-to-canonical improvement that the shuffled-reward arm recovers.

| Round | DCLM | Metamath | Arithmetic | 8-bit audio | CIFAR |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1024 | 0.34 | 0.25 | 0.43 | 0.42 | 0.30 |
| 2048 | 0.49 | 0.61 | 0.86 | 0.52 | 0.36 |
| 4096 | 0.39 | 0.50 | 0.86 | 0.55 | 0.35 |
| 8191 (Table 5) | 0.74 | 0.75 | 0.93 | 0.61 | 0.37 |

Source: `shuffle_share_by_round_1M_K4` and `shuffle_share_final_1M`. The final row uses the paper's Table 5. The shipped summary CSV gives 0.72 for DCLM, and its canonical column differs from Table 5 by up to 0.1 bpb.

**O8 — Signed reward matches or beats the canonical absolute reward on text and code.** DCLM 5.06 vs 5.34; C source 4.10 vs 4.16 (Table 5). Appendix F nonetheless states that signed "does worse".

**O9 — Longer generator training does not keep improving text transfer** (Fig. 3; `curriculum_value_fig3`).
- 4-seed-subset DCLM loss for learners trained on corpora from generators at T = 1024, 2048, 4096: 5.20, 5.19, 5.28.
- Audio is flat after T = 2048. Only CIFAR keeps improving.
- Epiplexity rises from 445 to 1,705 over the same range.
- The paper says both measures "improve steadily" and validation loss "falls with T".

**O10 — The warm-start payoff is modest on text** (Fig. 6; `pre_pretraining_fig6`).

| Panel | Final bpb (scratch → warm) | Natural tokens to plateau |
| --- | --- | ---: |
| DCLM | 1.507 → 1.493 (seed ranges overlap) | 11% fewer |
| CIFAR-10 | 3.714 → 3.629 | 28% fewer |
| ESC-50 | 3.842 → 3.790 | 36% fewer |

The paper excludes self-play compute from this comparison by design, arguing amortization (App. A.2). It quotes token savings for CIFAR and ESC-50 but not DCLM.

**O11 — The discovered mathematical structure is mostly arithmetic** (`discovered_structure`).
- 20,045 detected hits: 97.6% arithmetic progressions, 378 geometric, 43 quadratic, 32 cubic, 29 Fibonacci.
- The macros shorten these programs. `L` = `[->+++<]`, so `+[.L>]` prints powers of three.
- A particular 7-token string has probability about 19⁻⁷ ≈ 1e-9 under the uniform 19-token prior. That is consistent with the reported zero hits in 1.64e8 uniform samples.

**O12 — Natural data enters design selection.**
- Hyperparameters are tuned on the average of DCLM and DNA validation loss, flagged as "limited leakage" (§3.1).
- The reward was chosen "after searching through several possibilities at small scale" (§2.2). Table 5 scores those variants on natural corpora.
- Frontier points are selected by evaluation loss.
- DNA, a tuning corpus, carries the paper's outlier exponent: 0.435 against 0.01–0.06 in the literature.

## Interpretation

**I1 — Established.** A learner-adaptive program distribution teaches far more transferable structure than i.i.d. programs from the same machine (O1). It induces in-context algorithms that neither baseline acquires (O2). This is a real contribution, and the release is unusually auditable.

**I2 — Not established: continued scaling on text.** "Predictable scaling in compute" holds as a fit to a running-minimum envelope over about 3.5 decades. For text, the last useful point comes a decade before the end of the runs (O4), and the headline number depends on an 8-seed ensemble (O6). "Self-play keeps improving text transfer" is not supported beyond about 1e18 learner FLOPs at these scales.

**I3 — The attribution to the adaptive reward is partly supported.**
- Mid-training, the informative reward accounts for most of the gain on text (O7, rounds 1024–4096).
- By the final round, a shuffled reward recovers about three quarters of it. That arm keeps the scaffolding: fresh proposals from a generator tied to the prior, a MAP-Elites archive whose niches are loop depth (0–8) and program length (App. G), mutation, and replay.
- Two readings fit: the reward mainly accelerates, or the scaffolding supplies most of the eventual structure.
- The missing control is the scaffolding with no RL. It decides between these readings better than any endpoint table.

**I4 — The theory section does not identify its conclusion.**
- In Eq. 7–13, the self-play exponent estimates γη, where η is how fast self-play supplies universal structure.
- A natural-data exponent estimates min(βν, γµ), where µ is how fast natural data supplies it.
- Numerical similarity says nothing about how much of natural scaling is universal unless η ≈ µ, which is assumed.
- The comparison is also unstable, because the exponent moves between 0.07 and 0.12 with the fit form (O5) and between the paper and its shipped data.

**I5 — The in-context tasks are near the machine's native operations**: copy, reverse, stack, and byte arithmetic mod 256. The fixed-prior learner shares the machine and fails, so the curriculum matters. "Held-out" should still be read as "held out from training, not from the machine's natural function class."

**I6 — The paper is more careful than its abstract.** The Discussion explicitly denies that universal pretraining can replace natural data: contingent facts must come from the world. The "compute rather than human knowledge" framing is a research direction, not a claim.

## Replication assessment

**Can the study be replicated?** Not in full with current resources. Parts of it can, and the most informative part is cheap.

| Tier | What | Needs | Status or estimate |
| --- | --- | --- | --- |
| R0 | Re-derive published numbers from shipped scored data | CPU only | **Done here** (`reanalysis.json`); it surfaced O5, O7, and O8 |
| R1 | Re-score released checkpoints; temperature-calibrated single model vs ensemble; cross-seed error correlation | CUDA GPU (the RTX 4060 host qualifies; the scoring stack needs `jvp_flash_attention`, sm_75+), Hugging Face access | Hours to about a day for DCLM plus a few corpora (estimate) |
| R2 | Independent small-scale reimplementation (99k–1M) of the mechanism claims | One GPU; unpublished details from the authors | Days (see below) |
| R3 | Full ladder: 6 sizes, 4–8+ seeds, per-scale tuning, 34.36B-token budget | Cluster-scale compute | Out of reach |

**Planning estimate for learner-side compute** (not a measurement):
- Per-token forward cost is F ≈ 2N + 2·L·T·d at T = 4,096.
- Training costs 3F. The JVP reward is assumed to cost about 2F. Total ≈ 5F per token over the paper's 34.36B-token budget.

| Rung | FLOPs per seed | Hours at 5–10 TFLOP/s sustained |
| --- | ---: | ---: |
| 99k | 1.1e17 | 3–6 |
| 1M | 1.1e18 | 29–59 |
| 24M | 1.4e19 | 390–780 (16–33 days) |

Generator updates are about 1% of learner tokens, and interpreter throughput is not included.

**Shortened budgets are defensible for mechanism tests.** Every rung except 99k reaches its best DCLM K=4 loss to within 0.05 bpb well before the end of its run (`dclm_K4_plateau_by_rung`):

| Rung | Round within 0.05 bpb | Last scored round |
| --- | ---: | ---: |
| 1M | 2,048 | 15,616 |
| 3.1M | 4,864 | 11,264 |
| 6.2M | 1,024 | 10,240 |
| 24M | 768 | 8,191 |
| 99k | 6,400 | 16,383 |

An R2 study at 1M over 2,048–4,096 rounds, with 4 arms and 3 seeds, is on the order of days on one GPU.

**R2 needs details the paper does not report:**
- learner and generator learning rates, and the KL coefficient β per scale;
- pool size and its fresh, mutation, and replay fractions (the figures README implies 1,536 programs per round);
- maximum program length;
- interpreter step and memory budgets;
- generator sampling temperature;
- warmup length.

Without the authors' code, R2 is a reimplementation, not a replication, and should be reported as such. The first step is to ask the corresponding authors for the training code or these values.

## Implications if the claims hold

These are interpretations and hypotheses for this project, whose primary target is capability per parameter and per joule.

1. **Separate sources of capability.** If transferable algorithmic structure can be generated from compute, a small model's scarce natural tokens can be reserved for contingent knowledge, while iteration, copying, and stack-like computation come from procedural data. This bears directly on the capability-gate failures in E001–E004: the tiny students and even the E004 teacher could not execute depth-four composition. The self-play learners acquire stack and reversal in context with no natural data. → **H-012.**
2. **The exchange rate is the metric.** The paper deliberately leaves self-play compute out of the warm-start comparison (O10). For a per-joule objective the question is natural tokens saved per synthetic FLOP, and how many downstream reuses reach break-even. This project can measure that honestly at tiny scale. → part of **H-012**.
3. **Learning progress as a general data value.** The reward is cheap: one JVP per candidate. It is not specific to programs. It can rank natural-data candidates for training or replay. Its absolute value responds to forgetting as well as learning, which makes it a concrete candidate for H-001's interference-aware replay selector. → **H-013.**
4. **Scaffolding versus the learning signal.** If most of the eventual gain comes from quality-diversity search, mutation, and replay rather than the learner-aware reward (I3), the cheaper recipe is structured search without RL. → **H-014**, adversarial to the paper's attribution.
5. **Diversity across seeds.** The large ensemble gain on unseen text (O6) implies diverse errors across seeds out of distribution. The released seeds are a free testbed for H-005's error-correlation measurements. This needs tier R1.
6. **Analogy.** A generator that proposes experience at the edge of competence, plus an archive and replay, resembles offline rehearsal of self-generated experience. This is a prompt for mechanism discovery, not evidence of shared mechanism with sleep or play.

## Proposed next work, in cost order

Full test specifications are in the [hypothesis ledger](../hypothesis.md).

1. **R1 checkpoint audit** (GPU hours). The decision it changes: whether the ensemble gain on text is mostly calibration. If one temperature-tuned model closes most of the K=8 gap, text transfer should be restated per model.
2. **H-013 learning-progress selection** on TinyStories with the existing tiny-LM harness. Selection FLOPs are counted. It needs no program machine, so it is the cheapest test of whether the paper's central signal is useful to this project.
3. **H-012 procedural pre-pretraining for tiny students**, with a compute-matched natural-data baseline and an initialization-statistics control. It uses a new capability-gated compositional probe, not E004's retired benchmark.
4. **H-014 scaffold attribution**, a small R2 reimplementation, after requesting the training code from the authors.

The active E005 decision in [NEXT](../NEXT.md) is unchanged. These items are candidates for prioritization, not replacements.

## Questions for the authors

1. What does the scaffolding-only arm (fixed prior, archive, mutation, replay, no RL) score on the Table 5 corpora and over rounds?
2. Why does the Table 5 canonical column differ from the shipped `reward_frontier_summary.csv`? Why do several Table 2 exponents differ from the shipped `scaling_exponents.csv`, most notably DNA (0.435 vs 0.264)?
3. What are single-model temperature-calibrated DCLM losses at the frontier points?
4. Can the self-play training code and the unreported hyperparameters be released?
