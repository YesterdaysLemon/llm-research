# E006 preregistration — learning-progress selection of natural text

Registered: 2026-09-26, before any E006 run on real data. Before registration, only these were run:
- CPU unit tests;
- synthetic-token end-to-end checks;
- a 40-step synthetic-token CPU timing benchmark.

No TinyStories text was loaded, and no selection effect on natural data has been observed.

## Exact decision

E006 tests [H-013](../../hypothesis.md#h-013--learning-progress-selection-of-natural-data). The directional claim is:

> Selecting each training batch from a larger candidate pool by the absolute preconditioned alignment between each candidate's gradient and the learner's recent parameter movement, `|⟨∇L(y), P ⊙ (θ_anchor − θ_t)⟩|`, lowers tiny-LM validation NLL at matched optimizer steps more than uniform, loss-prioritized, or gradient-norm-prioritized selection. It keeps an advantage when its scoring FLOPs are counted.

The run decides whether the generator reward of arXiv:2609.30063 (Eq. 2) is a useful selection rule for this project's tiny-LM line, and a candidate for H-001's replay selector. The alternatives are that it reduces to cheaper loss or gradient-norm prioritization, or that it loses once its scoring cost is counted.

This is a transfer of the paper's signal from program selection to natural-text window selection. It is not a replication of the paper.

## Why these arms

The absolute score factors exactly:

`|s_i| = ‖P^½ g_i‖ · ‖P^½ δ‖ · |cos_P(g_i, δ)|`

- `g_i` is the gradient of window `i`'s loss.
- `δ = θ_anchor − θ_t`.
- `‖P^½ δ‖` is shared across the pool at a step.

Ranking by `|s_i|` therefore equals ranking by gradient magnitude times direction. The `gradnorm` arm keeps only the magnitude; the `cosine` arm keeps only the direction. The `loss` arm is the ordinary difficulty baseline, and `signed` drops the absolute value.

The secondary noisy condition probes the paper's motivating failure mode: difficulty rewards chase injected randomness. It also tests App. F's claim that the absolute value helps, which the paper's own Table 5 contradicts on text: signed scores 5.06 vs 5.34 bpb on DCLM.

## Models and data

All reuse the completed E004 experiment, whose `data.py` and `model.py` are imported unchanged.

**Data.** `TinyStoriesV2-GPT4-valid.txt`:
- Hugging Face revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, CDLA-Sharing-1.0.
- 22,502,601 bytes; SHA-256 `6874bae9…c0585d`.
- The first 12,000 stories train and the next 2,000 validate, exactly as in E004.
- There is no controlled stream.

**Reproduction checks.** The runner refuses to start unless all three reproduce E004:
- vocabulary SHA-256: `bb0587253d44f87eaa5b2202daa1082f7437f8cdb07fa6133a5703048ee05588`;
- training tokens: 2,312,569;
- validation tokens: 381,261.

**Student.** E004's student:
- width 160, four layers, four heads, feed-forward width 640;
- tied embeddings, 64-token context, dropout 0;
- 1,575,360 parameters;
- initialized by E004's `model_from_config` under each seed.

**Attention.** The math SDPA backend is used on CPU and CUDA, for training, scoring and evaluation. E004 already used it on CUDA. The fused kernels lack forward-mode AD.

**Training, identical in every arm.**
- AdamW: learning rate 0.0004, weight decay 0.1, gradient clip 1.0.
- 20-step linear warmup, then a constant rate.
- Batches of 16 windows of 64 tokens.
- bfloat16 autocast on CUDA for the training step only.
- Scoring and evaluation run in float32 with full-precision matmuls (TF32 disabled).
- PyTorch deterministic algorithms are required and recorded.

## Candidate pool and selection

At step `t` (the number of completed updates), the pool is 64 random 64-token training windows, four times the batch.
- The windows come from a generator seeded with `seed + 100000`, consumed identically in every arm.
- Every arm sees the same pool at every step.
- An arm keeps 16 windows. Ties are broken by pool order.
- At `t = 0` there is no optimizer state, so every arm trains on the first 16 windows.

**Learning direction.**
- `P_t = lr_t / (sqrt(v̂_t) + 1e-8)` is AdamW's bias-corrected diagonal step operator.
- `θ_anchor` is the latest parameter snapshot at or before step `⌊t/2⌋`. Snapshots are stored every 50 steps, so the anchor lags the paper's exact `⌊t/2⌋` by fewer than 50 steps. It is the initialization for `t < 100`.
- The tangent is `v = P_t ⊙ (θ_anchor − θ_t)`.

| Arm | Selects the 16 windows with the largest | Per-step method cost |
| --- | --- | --- |
| `uniform` | (first 16 of the pool) | none |
| `loss` | mean CE at `θ_t` | one forward pass over the pool |
| `gradnorm` | `‖P^½ g_i‖` | per-sample gradients over the pool |
| `cosine` | `|⟨g_i, v⟩| / (‖P^½ g_i‖ ‖P^½ δ‖)` | per-sample gradients over the pool |
| `signed` | `s_i = ⟨g_i, v⟩` | one forward-mode JVP over the pool |
| `absolute` | `|s_i|` | one forward-mode JVP over the pool |

To first order, `s_i = L_i(θ_anchor) − L_i(θ_t)`, which is positive when the learner has improved on window `i`. A single JVP gives every `s_i`, because `v` is shared by all candidates.

## Conditions

- **`clean` (primary):** no noise.
- **`noisy` (secondary):**
  - Each pool window is independently replaced, with probability 0.25, by i.i.d. uniform tokens over the 2,044 non-special ids.
  - The noise uses a separate generator (`seed + 200000`), so the clean windows are unchanged.
  - Validation is always clean.

## Budgets and compute accounting

- Selection arms run 6,000 steps: 6,144,000 training tokens, about 2.7 epochs.
- `uniform` runs 22,000 steps = `⌈6,000 × 11/3⌉`. That is the nominal FLOP count of a JVP arm at 6,000 steps.
- The constant post-warmup learning rate makes uniform's first 6,000 steps exactly the matched-step uniform run.
- Evaluation steps: 250, 500, 1,000, 2,000, 3,000, 4,000, 5,000 and 6,000. Uniform is also evaluated at 8,000, 11,000, 14,000, 18,000 and 22,000.

**Nominal FLOP convention.** `F` is the forward FLOPs per token:

`F = 2·L·(4d² + 2·d·d_ff) + 2·d·V + 2·L·T·d`

| Operation | Cost per token |
| --- | --- |
| Training step | `3F` per trained token |
| Loss scoring | `1F` per pool token |
| JVP scoring | `2F` per pool token |
| Per-sample gradients | `3F` per pool token |

Diagnostics and evaluation are excluded from method cost.

**Measured cost on CPU.** This is an indicative timing (4 threads, synthetic tokens, 40 steps), not a result:

| Arm | Milliseconds per step |
| --- | ---: |
| Uniform | 55 |
| Loss | 127 |
| JVP arms | 317 |
| Per-sample arms | 495 |

The measured JVP-to-uniform ratio (5.8) exceeds the nominal 3.67. The FLOP-matched comparison is therefore lenient toward selection. Per-run scoring, training, diagnostic and evaluation seconds, plus peak CUDA memory, are recorded on the RTX 4060 host. No energy is reported, because E004 found no power telemetry on that host.

## Seeds

- Fixed seeds are 8601–8605. Each sets initialization; the pool seed is `seed + 100000` and the noise seed is `seed + 200000`.
- Every arm and condition shares these, so all contrasts are paired.
- The smoke uses seed 8600 only.

## Endpoints

**Primary endpoint:** validation NLL in nats per token, float32, over all non-overlapping 64-token windows of the 2,000 validation stories (5,957 windows expected), at step 6,000 in the clean condition.

**Primary contrasts.** Paired over the five seeds, with two-sided 95% Student-t intervals; negative means `absolute` is better.

| Contrast | Comparison |
| --- | --- |
| C1 | `absolute − uniform`, both at 6,000 steps |
| C2 | `absolute − loss`, both at 6,000 steps |
| C3 | `absolute − gradnorm`, both at 6,000 steps |
| C4 | `absolute` at 6,000 steps − `uniform` at 22,000 steps (matched nominal FLOPs) |

**Decision:**
- **Supported:** the upper bounds of C1, C2, C3 and C4 are all below zero.
- **Selection signal supported; compute efficiency not supported:** C1–C3 are below zero and C4 is not.
- **Not supported:** any of C1–C3 has an upper bound at or above zero. This is H-013's kill condition. Any interval lying entirely above zero is reported as `absolute` being reliably worse.
- **Invalid:** any validity gate fails. That is not a negative result.

The conjunction rule is an intersection-union test, so no multiplicity correction is applied.

**Secondary endpoints.** These are descriptive and change no H-013 status:
- `cosine − gradnorm`: does direction carry information beyond magnitude?
- `signed − absolute`.
- Each arm minus `uniform`.
- Mean area under the NLL curve from step 250 to 6,000.
- Mean Spearman correlations among the five scores at diagnostic steps. This tests the rival that `|s|` reduces to loss or gradient-norm ranking.
- Selected-noise shares.
- Measured method seconds.

**Registered predictions for the noisy condition** (mechanism, secondary):

| Prediction | Statement |
| --- | --- |
| P1 | `loss` selects noise in more than 50% of its windows (difficulty chases noise) |
| P2 | `absolute` selects noise above the 25% pool rate |
| P3 | `signed` selects noise below 25% |
| P4 | `signed − absolute` has a 95% upper bound below zero |

P2 is predicted because learning English raises the loss on uniform noise. Noise gradients are then anti-aligned with the learning trajectory, and the absolute value rewards anti-alignment.

Losing results:
- P2 false means the absolute value handles injected noise as the paper argues.
- P4 false means the sign does not matter for NLL, even if selection shares differ.

## Validity gates

- **G1:** in the clean condition, `uniform` at step 6,000 is at least 1.0 nat per token below the add-one-smoothed training-unigram NLL on the same validation windows, for every seed.
- **G2:** at every diagnostic step (every 250 steps, all arms, all five scores computed on the pool without affecting training), the JVP scores match per-sample-gradient dot products. The maximum absolute error must be at most `1e-3 × max|s|`.
- **G3:** every run is finite.

The data reproduction checks run before any training.

## Smoke phase

`config/smoke.json` differs from `config/fixed.json` only in the following. A test enforces that dataset, model, training, selection, arms, conditions and gates are identical.
- its identifier;
- seed 8600;
- 300 selection steps and 1,100 uniform steps;
- evaluation and analysis steps.

It runs every arm in both conditions once on the RTX 4060 host.

Its purpose:
- runtime and memory;
- compatibility of JVP and vmap with CUDA deterministic algorithms;
- the data reproduction checks;
- G2.

It may change the design only for broken code, impossible runtime or memory, an invalid control, or malformed data. Any amendment goes in a new configuration file and is documented before the fixed run. Smoke NLL is recorded but not interpreted: a different seed and 5% of the budget.

## Stop rule

1. Run the smoke once. Then run each of the 60 fixed runs once, whatever the sign of the early results.
2. A run that crashes from an infrastructure failure is rerun from scratch, which is deterministic, and the failure is recorded.
3. Do not tune, add arms, rerun completed runs, stop early, or replace paired uncertainty.
4. Preserve every per-run file in `results/fixed/`. The runner never overwrites an existing run file.

## Limits

- One model size, one corpus, one pool multiple, one lookback rule.
- The noise is synthetic.
- A positive result licenses a replay-selection test for H-001. It does not license a general data-curation claim.
- Stored parameters, token exposures, nominal FLOPs and wall time are separate quantities. No joule claim is possible without telemetry.
