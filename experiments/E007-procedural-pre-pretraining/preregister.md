# E007 preregistration — procedural pre-pretraining for a tiny student

Registered: 2026-09-27, before any E007 run on real data. Before registration, only these were run:
- CPU unit tests;
- synthetic-token end-to-end checks;
- generator timing and acceptance measurements.

No TinyStories text was loaded, and no warm-start effect has been observed.

## Exact decision

E007 tests the first clause of [H-012](../../hypothesis.md#h-012--procedural-pre-pretraining-supplies-tiny-model-computation):

> A tiny causal LM initialized by training on procedurally generated data reaches lower natural-text validation loss at a fixed natural-token budget than the same model from random initialization. The advantage survives two controls: (a) giving the synthetic-stage compute to the ordinary model as extra natural-text training; (b) a within-tensor shuffle of the warm-start weights that keeps their value statistics but destroys learned structure.

The run decides whether "pre-pretraining" on synthetic structure is worth pursuing for this project's tiny-LM line, where E001–E004 students repeatedly lacked capability. It does not test H-012's second clause, depth-general composition. That needs a capability-gated benchmark, and E004 showed that the positive control is the hard part. In-context probes are reported here as secondary evidence only.

## Sources of synthetic data

Both sources adapt arXiv:2609.30063. Rows are 65 tokens: a 64-token input and its shifted target. Symbols are ids of the natural vocabulary's 2,044 non-special tokens, so the model and tokenizer are unchanged.

### `pcfg`: random probabilistic context-free grammars (paper App. H)

Each row samples one fresh grammar:
- 2–16 terminal ids;
- 1–8 nonterminals, each with 1–4 productions;
- production weights `U(0,1) + 10⁻⁶`, normalized;
- right-hand sides of 1–4 symbols, each a terminal with probability 0.5.

Every nonterminal is repaired to have a terminal-only production. Words come from leftmost derivation, stopping at an empty stack, 64 symbols, or 10⁴ expansions. The row concatenates words until it is full.

The paper's PDF text reads "104 expansions". That is 10⁴ with a lost superscript, the same extraction loss as its "10−6".

Rows are generated fresh from a seeded stream.

### `programs`: the paper's universal prior, filtered to fill the context (paper App. E, Table 4)

**Language.** Tokens are drawn i.i.d. uniformly over the 19-symbol alphabet until the end token: 8 Brainf*ck primitives, 10 macros and `F`, with at most 256 tokens. Programs run on a 256-cell circular byte tape. `,` reads uniform random bytes, unmatched brackets are no-ops, and there is a 2,048-step budget.

**Verification.** The interpreter reproduces all five of the paper's Table 1 sequences (a unit test):
- arithmetic;
- geometric;
- Fibonacci;
- quadratic 9, 25, 59, 111, 181;
- cubic 0, 254, 236, 74, 152.

**Why filter.** Measured on 20,000 samples:
- only 26.6% of uniform-prior programs print two or more bytes;
- only 0.29% fill a 65-token row.

Unfiltered, the source would supervise under 3% of positions against 100% for `pcfg`, confounding content with signal quantity. E007 therefore keeps only programs that emit at least 65 bytes. This is a registered deviation from the paper's unfiltered prior.

**Bank.**
- 128,000 accepted outputs, which is exactly one epoch of the synthetic stage.
- Generated in 1,000-row chunks from seeds `8798 + chunk index`. The bank is identical for any number of build processes (a unit test), and it is cached.
- Each training seed visits every row once, in its own order.
- Bytes map to token ids through a fixed injection (seed 8799).

Accepted outputs are mostly constant or periodic streams. This source therefore mainly teaches "continue the pattern in context", which is plausibly relevant to the copy-like mechanisms language models use.

**Registered build.** The bank was built once in the authoring container before registration: CPU only, 4 processes, 1,484 s with other jobs competing.
- The cache file's SHA-256 is `aed512ad023d0a6fc8e7ac5c1161b5bcfcb4d367ee577da9b571e9138646e779`, recorded in both configurations.
- The runner records whether the GPU host builds the identical bank. A mismatch, such as from a PyTorch RNG change, is a same-distribution redraw. It is recorded, not fatal.
- Only 57,328 of the 128,000 rows (45%) are distinct, because many programs emit the same trivial stream. The source's effective diversity is therefore well below its row count. This is a property of the filtered prior, not a defect.

**Held-out sets.** Each source has 512 rows from seed 8791, disjoint from every training stream.

## Arms

| Arm | Initialization before natural training | Natural steps |
| --- | --- | ---: |
| `scratch` | E004 random initialization | 16,000 |
| `pcfg` | 8,000 steps on `pcfg` | 8,000 |
| `pcfg_shuffled` | `pcfg` weights, each tensor's entries randomly permuted | 8,000 |
| `programs` | 8,000 steps on `programs` | 8,000 |
| `programs_shuffled` | `programs` weights, each tensor's entries randomly permuted | 8,000 |

**What is held fixed across arms.**
- Every arm uses E004's student (1,575,360 parameters) and training settings:
  - AdamW, learning rate 0.0004, weight decay 0.1, gradient clip 1.0;
  - a 20-step warmup, then a constant rate;
  - batches of 16 × 64 tokens;
  - bfloat16 autocast on CUDA.
- These settings are reused for the synthetic stage without tuning.
- The natural stage starts a fresh optimizer with its own warmup.
- All weights transfer: blocks, position embeddings and the tied token embedding. The shuffle permutes each parameter tensor, tied weights once (seed `seed + 300000`).

**Pairing.**
- Within a seed, every arm starts from the same random initialization before any synthetic stage.
- Every arm draws the same natural batch at natural step `n` (stream seed `seed + 100000`).
- The synthetic stream seed is `seed + 400000`.

**Compute matching.** Every step has the same model, batch and sequence length, and therefore the same training FLOPs. A warm arm's total of 8,000 synthetic plus 8,000 natural steps equals `scratch` at natural step 16,000. With a constant rate, `scratch`'s first 8,000 steps are exactly the natural-token-matched comparison.

CPU time spent generating synthetic data and building the bank is excluded from this match and reported separately. By the CPU timings above, the `pcfg` generator adds about 13 ms per batch.

## Seeds, evaluation and endpoints

**Seeds.** Fixed: 8701–8705. Smoke: 8700 only.

**Evaluation.**
- Validation NLL (nats per token, float32) over all non-overlapping 64-token windows of E004's 2,000 validation stories.
- Measured at natural step 0 and every 250 natural steps.
- The runner refuses to start unless E004's vocabulary hash and token counts reproduce.

**Primary contrasts.** For each source `s`, paired over five seeds. The two-sided intervals are 97.5%, Bonferroni-adjusted over the two sources. Negative means the warm start is better.

| Contrast | Definition |
| --- | --- |
| D1 (head start) | NLL of `s` at natural 8,000 − `scratch` at natural 8,000 |
| D2 (compute-matched) | NLL of `s` at natural 8,000 − `scratch` at natural 16,000 |
| D3 (beyond weight statistics) | NLL of `s` at natural 8,000 − `s_shuffled` at natural 8,000 |

**Decision.** Only sources that pass the synthetic learning gate are eligible.

| Outcome | Condition |
| --- | --- |
| Supported | some source has upper bounds of D1, D2 and D3 all below zero |
| Head start supported, not compute-efficient | some source has D1 and D3 below zero, but none has D2 |
| Not supported as procedural structure | the sources with D1 below zero fail D3; initialization statistics explain the gain |
| Not supported | no source has D1 below zero; this is H-012's clause-1 kill condition |
| Invalid | a global gate fails, or no synthetic stage learns; not a negative result |

**Secondary endpoints.** These are descriptive and change no status:
- Natural-token savings to reach `scratch`'s own NLL at natural steps 2,000, 4,000 and 8,000, linearly interpolated per seed.
- D1 and D3 at natural steps 2,000 and 4,000, with 95% intervals.
- Zero-shot NLL at natural step 0.
- The synthetic held-out curves.
- Two in-context probes at natural steps 0 and 8,000 (and 16,000 for `scratch`), each over 512 fixed trials of word tokens from ids 100–1,099:
  - `copy`: repeat a random 20-word sequence;
  - `recall`: 8 key/value word pairs, then a key.

**Author's prior expectation.** This is not a decision rule.
- `pcfg` gives a head start (D1 < 0) that survives the shuffle (D3 < 0), but does not beat the compute-matched `scratch` (D2 ≥ 0). That matches the paper's small, narrowing text effect.
- `programs` is weaker on text but may raise the `copy` probe.

## Validity gates

- **G1:** `scratch` at natural step 8,000 is at least 1.0 nat below the add-one-smoothed training-unigram NLL on the same windows, for every seed.
- **G2:** for each source, held-out synthetic NLL falls by at least 1.0 nat from synthetic step 0 to 8,000, for every seed. A source that fails is excluded from inference.
- **G3:** all runs are finite.

## Smoke phase

`config/smoke.json` is identical in dataset, model, training, arms, synthetic sources, probes and gates (a test enforces this). It differs only in:
- seed 8700;
- 400 synthetic steps, 400 warm natural steps and 800 scratch natural steps;
- an evaluation interval of 100 and its analysis steps.

It builds and caches the 128,000-row program bank that the fixed run reuses. Its purpose:
- runtime and memory;
- CUDA determinism;
- the data reproduction checks;
- the bank build;
- the gates' plumbing.

Smoke may change the design only for broken code, impossible runtime or memory, an invalid control, or malformed data. Its NLL values are recorded but not interpreted.

## Stop rule

1. Run the smoke once. Then run each of the 25 fixed runs (5 seeds × 5 arms) once, whatever the early results.
2. A crashed run is rerun from scratch, which is deterministic. Synthetic checkpoints are reused only if their recorded parameter hash matches. The failure is recorded.
3. Do not tune, add sources or arms, change budgets, or rerun completed runs.
4. Preserve every per-run file in `results/fixed/`. Checkpoints and the bank cache are regenerable and not committed; their hashes are recorded in the result files.

## Limits

- One model size and a 64-token context, far shorter than the paper's 4,096.
- The synthetic budget is about 8.2M tokens, orders of magnitude below the paper's.
- No learner-adaptive (self-play) source. That needs the H-014 reimplementation.
- The synthetic stage reuses untuned natural-text hyperparameters.
- A negative result here does not refute larger-scale pre-pretraining.
- Stored parameters, token exposures, nominal FLOPs, CPU generation time and GPU wall time are separate quantities. No energy claim is made without telemetry.
