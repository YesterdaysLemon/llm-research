# E004 preregistration — tiny causal-language-model bridge

Registered: 2026-08-19, before any E004 training run.

Prospective audit amendment: the first draft was reviewed before execution by
Claude Opus 5 and locally replayed. It had a process-dependent held-out set, a
depth/pair confound, truncated controlled records at training-only absolute
positions, an underived one-shot budget, and a commutative controlled algebra.
All five defects were repaired below before the first commit or training run.
Qwen 3.8 was attempted twice under bounded read-only prompts but returned no
report before either deadline; no Qwen claim influenced this protocol.

## Exact decision

E004 asks whether the affine E002–E003 effect survives genuine causal
next-token learning. The directional claim is:

> At fixed student architecture, initialization, token order, optimizer steps,
> and primary training tokens, output distillation plus learned-teacher hidden
> relation geometry improves natural validation NLL and controlled held-out
> composition beyond output distillation alone, next-token target geometry,
> and teacher-logit geometry.

An ordinary longer output-distillation run tests whether any advantage survives
additional, conventional token/optimizer compute. Stored parameters, token
exposures, wall time, peak memory, and energy are separate outcomes.

## Two-stage registration

This document freezes the smoke phase now. A second clean commit will freeze
the fixed comparison only after smoke establishes that the benchmark is
learnable at the intended scale. Smoke may change the design only for a broken
implementation, impossible memory/runtime, malformed split, or failed teacher
or labels-only student capability gate. Smoke will train only the teacher and
the ordinary next-token student, so no auxiliary-condition effect will be
viewed while setting the fixed budget.

Every smoke artifact and amendment remains in the repository. A failed gate is
not evidence for or against activation-specific distillation.

## Models

- Tokenizer: deterministic lowercase word-and-punctuation tokenizer, trained
  only on the frozen natural training stories plus the controlled-language
  vocabulary; maximum vocabulary 2,048 including four special tokens.
- Teacher: decoder-only Transformer, width 320, eight layers, eight heads,
  feed-forward width 1,280, tied token/output embeddings, sequence length 64.
- Student: same implementation, width 160, four layers, four heads,
  feed-forward width 640, tied embeddings, sequence length 64.
- Positional representation: learned absolute embeddings.
- Dropout: zero, to make paired-condition initialization and data order exact.

The intended sizes are approximately 10.5M teacher and 1.6M student
parameters; the executable count is authoritative and will be recorded.

## Data and provenance

Natural text comes from `TinyStoriesV2-GPT4-valid.txt` in
`roneneldan/TinyStories`, pinned to Hugging Face dataset revision
`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`. The source repository identifies
the dataset license as CDLA-Sharing-1.0.

- URL: `https://huggingface.co/datasets/roneneldan/TinyStories/resolve/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/TinyStoriesV2-GPT4-valid.txt?download=true`
- bytes: `22,502,601`
- SHA-256: `6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d`

Because this is a small local bridge rather than a benchmark comparison, E004
uses the pinned validation artifact as a source pool and constructs its own
non-overlapping story split: the first 12,000 complete stories for training and
the next 2,000 for natural validation. The tokenizer sees training stories
only. This choice precludes comparison to reported TinyStories validation
scores and will be stated wherever NLL is reported.

Exactly 20% of sequence rows, accumulated over each five consecutive batches,
come from a deterministic controlled language. A record names a character, an
initial state modulo 11, and a sequence drawn from six affine operators:
`add 1`, `add 3`, `multiply 2`, `multiply 3`, `subtract 2`, and `negate`.
It then asks `where is NAME ?`; the state from 0 through 10 is the next token.

The directed pairs `add 1 -> multiply 2`, its reverse, `add 3 -> multiply 3`,
and its reverse are absent from controlled training and ID validation. These
operators do not commute: for example, multiplying after adding one yields
`2x+2`, while adding one after multiplying yields `2x+1` modulo 11. Held-out
evaluation crosses every depth with every pair equally and requires exactly one
configured pair per example. Pair order comes directly from the configuration,
never from set iteration.

Every controlled training row begins at a record boundary, contains only whole
records, and masks trailing padding from loss. This matches the evaluated
record's initial absolute position while retaining ordinary causal next-token
training on every non-padding transition. Natural rows remain random contiguous
64-token windows.

The fixed comparison will preserve the exact 80/20 natural/controlled sequence-row mix,
the exact source checksum, story indices, tokenizer mapping, controlled seeds,
training depths, and ordered held-out operation pairs. Realized supervised
tokens and complete controlled-record exposures are recorded separately.

## Smoke phase frozen now

Configuration: `config/smoke.json`.

- Teacher seed: 8400; capability rungs at 400, 2,400, 7,200, and 21,600
  cumulative optimizer steps.
- Student seed: 8401; the same independently evaluated rungs.
- Batch: 16 sequences of 64 input tokens.
- Optimizer: AdamW, learning rate 0.0004, weight decay 0.1, gradient clipping
  1.0, 20-step linear warmup followed by a constant learning rate.
- Training precision: bfloat16 autocast on CUDA; float32 on CPU. Evaluation is
  always float32. PyTorch deterministic algorithms and deterministic math SDPA
  are required and recorded.
- Evaluation: 128 fixed natural validation chunks and 512 examples for each
  controlled split, reported overall, by depth, and by held-out pair.
- Smoke trains only ordinary next-token cross-entropy. It does not run output
  KD or any geometry condition.

Each model stops at the first rung that clears its own gates. Those selected
teacher and student step budgets become candidates for the fixed protocol. If a
model reaches 21,600 steps without passing, E004 stops for benchmark redesign.
This predeclared capability ladder replaces a discretionary post-failure budget
amendment and never observes an auxiliary-condition effect.

### Capability gates

The teacher must reach all of:

1. natural validation NLL at least `0.2` nats/token better than the add-one
   smoothed unigram model fitted on the frozen natural training stream;
2. controlled ID answer accuracy at least `70%` overall and `50%` at every ID
   depth;
3. controlled held-out answer accuracy at least `55%` overall for depths two
   through four, and at least `30%` for every ordered held-out pair. Depth six
   is reported but is not an aggregate gate.

The labels-only student must reach all of:

1. natural validation NLL at least `0.2` better than the same frozen unigram
   baseline;
2. controlled ID accuracy at least `55%` overall and `30%` at depth four;
3. held-out depth-two-through-four accuracy between `20%` and `80%`, inclusive,
   so the primary endpoint is neither at floor nor already saturated.

Chance controlled-answer accuracy is `1/11`, approximately 9.09%. If either
model misses every rung, stop before the fixed comparison. No auxiliary loss may
be run to choose a budget or architecture amendment.

## Fixed conditions to freeze after a successful smoke

The second-stage configuration must contain at least five paired student seeds
and these seven conditions:

1. `ce`: ordinary next-token cross-entropy;
2. `kd`: labels plus teacher-output distillation;
3. `kd_hidden`: the same KD base plus learned-teacher hidden relation geometry;
4. `kd_target`: the same KD base plus next-token identity geometry;
5. `kd_logit_geometry`: the same KD base plus teacher-logit relation geometry;
6. `kd_hidden_random`: the same geometry implementation using a frozen,
   randomly initialized teacher;
7. `kd_extra`: ordinary KD with a prospectively fixed longer token/step budget.

Main conditions 1–6 use identical student sizes, steps, batches, token order,
and initialization within seed. `kd_extra` receives the same token-order prefix
and then additional ordinary tokens. The longer-budget multiplier is the
ceiling of a pre-result timing/FLOP ratio between one `kd_hidden` step and one
`kd` step, not an arbitrary number. Temperature, CE/KD weights, geometry rule,
sampled context positions, mapped layers, evaluation sizes, seed values, and
longer-budget multiplier must appear in the second-stage configuration before
the fixed run.

For geometry losses, aligned context positions within a minibatch are sampled
without using labels. The centered, Frobenius-normalized cosine Gram matrix of
student states is compared with the corresponding target. All geometry
conditions apply one loss at each of the same four student layers with the same
reduction. Hidden geometry maps them to zero-indexed teacher blocks
`[0, 2, 4, 6]`, excluding the final block whose representation is directly
projected by the tied language-model head. Target-token and teacher-logit
relations are repeated across the same four student layers. Target geometry
contains one exactly when two sampled contexts have the same next token. Logit
geometry uses only teacher output-logit vectors.

The second-stage registration will freeze a target-agnostic gradient-ratio rule:
on the first paired batch before any optimizer step, each auxiliary coefficient
is set so its gradient norm is the same fixed fraction `rho` of the KD-base
gradient norm. `rho`, the probe count, and the sampling generator are shared and
frozen. This avoids equating raw losses whose attainable geometry and gradient
scale differ. These controls are not information-byte matched; target bytes,
wall time, peak memory, and complete token exposures are reported.

## Fixed endpoints and decisions

The primary behavioral endpoints are natural validation NLL and controlled
held-out answer accuracy for depths two through four, with depth six secondary.
All condition contrasts use paired seed differences with two-sided 95%
Student-t confidence intervals. At least five completed paired seeds are
required.

An activation-specific result requires `kd_hidden` to improve over both `kd`
and `kd_target`, with the favorable interval excluding zero on held-out
controlled accuracy and no material natural-NLL regression. Beating
`kd_logit_geometry` is required before attributing the benefit to hidden
structure beyond richer teacher-output geometry. Beating `kd_hidden_random` is
required before claiming learned-teacher specificity rather than generic Gram
regularization. Beating the calibrated `kd_extra` is required for even a coarse
compute-efficiency claim; otherwise a hidden-geometry training-signal effect may
still be reported without an efficiency claim.

If the labels-only fixed student capability gate fails across seeds, the
mechanism comparison is invalid rather than negative. No energy-per-joule claim
will be made without supported telemetry.

## Stop rule

Run the capability ladder once and stop each model at its first passing rung. If
both models pass, audit and commit one fixed configuration,
then run every condition and seed once regardless of sign. Do not tune on fixed
results, rerun selected seeds, add post hoc conditions to rescue the claim, or
replace paired uncertainty with pooled minibatches. Preserve raw results for
positive, null, negative, or invalid outcomes.
