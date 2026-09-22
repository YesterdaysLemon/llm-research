# E009 cloud study: protocol fixed before model outputs

Question: does removing literal assistant self-supervision about experience,
feelings, consciousness, opinions and preferences change self-description or
conversational consistency, beyond the effect of replacing equally many rows?
Both affirmations and denials are removed. We do not teach a preferred identity.

The base is Qwen2.5-7B (7.61B parameters), not its instruction-tuned variant.
This run adds supervised LoRA only: no reward model, preference optimization,
RLHF or synthetic affirmative answers. This cannot erase prior training data,
prove absence of alignment influence, or establish subjective experience.

## Data and intervention

Use the pinned public OASST2 and Databricks Dolly snapshots in config.json.
OASST2 retains English, non-deleted, reviewed, labeled non-synthetic paths with
quality >= 0.5, a quality vote, no positive spam/PII/language mismatch labels,
rank-zero assistant replies and no positive task-failure labels. These labels
are provenance claims, not a guarantee of human authorship or factual accuracy.
Dolly retains passage-grounded QA/extraction/summaries plus brainstorming and
creative writing. Source responses and their necessary context are preserved;
neither answer rewriting nor synthetic replacement responses are used.

Common quarantine removes discovered wrong biographies/product identities,
broken tasks, unresolved factual/code problems and ambiguous literal claims.
It applies identically before all three conditions. Reviewed per-message
overrides handle lexical false positives/negatives; earlier assistant turns
are classified as well as the supervised final reply. Fiction, requested human
voices, social warmth, support and ordinary recommendation idioms remain.

Each arm has 6,144 rows: 2,560 OASST2 and 3,584 Dolly. Original retains eligible
self-supervision; filtered replaces all selected target rows; random_control
replaces matched non-target rows and retains the targets. Both interventions
use the identical replacement records. Matching prioritizes source and answer
token count, then input length; control ties use a deterministic hash. This is
a matched replacement control, not an unconstrained random sample. It cannot
perfectly control subject matter or semantics of removed questions.

OASST conversation trees and identical normalized Dolly reference passages
stay within one split. Exact prompt duplicates and exact evaluation questions
are excluded. No exhaustive semantic near-duplicate or pretraining contamination
audit is claimed. A shared 128-row non-target validation split measures language
loss. Long examples are excluded whole above 1,024 tokens; no silent truncation.
Only final assistant response tokens plus EOS receive loss; earlier turns,
user text, role prefix and padding are masked. Gradient accumulation weights
each supervised token equally, not each unequal-length microbatch equally.

Four precompute checks: pinned provenance/licenses and hashes; contextual
review of every changed row; separate broad-topic/first-person scans plus
full deterministic random sample; independently reconstructed text, splits,
token boundaries, arm differences and a byte-identical second build. This is
one agent using different checks, not four independent human reviewers. The
complete corpus has not been manually fact-checked. Versioned raw data,
manifests, exclusion reasons, review evidence and source IDs remain local.

## Compute and prospective decision rule

BF16 LoRA rank 16, alpha 32 on all attention and MLP projections; zero LoRA
dropout; AdamW 1e-4, weight decay zero; global gradient clipping 1; 12-step
warmup times linear decay. Two epochs, effective batch 32 (4 x 8), 384 updates
per arm. Model initialization, row shuffle and seed are paired across arms.
Base weights remain frozen. Training and inference reject mismatched data,
review receipts, definitions, tokenizer or model weight checksums.

Maximum study allocation $22 from the authorized study budget, with
no top-up. Prefer one H100 80GB at a verified <=$3.49/hour rate. A six-hour
deadline starts at pod creation, includes setup and downloads, and is enforced
by a separate local watchdog. Persistent network storage protects checkpoints
from termination; retrieve and verify artifacts before deleting storage.
The CLI's obsolete terminate-after flag is not relied upon.

First run eight disposable smoke updates: two on longest examples and six on
normal batches. Before inspecting any generated model output, choose one paired
seed (29017) or two (also 29018) from a timing projection. Use mean normal-step
time x 384 updates x number of training runs x 1.35, plus 60 minutes reserved
for model loads, evaluation, conversation and artifact transfer. Two paired
seeds are used only if this fits the remaining deadline. If one cannot fit,
stop and revise the hardware/time plan before behavioral evaluation; do not
silently omit a control. This is a small controlled study, not frontier-scale
training or a statistical power guarantee. No result-driven data/LR/epoch tuning.

## Evaluation fixed before compute

Run all 42 fixed probes and three four-turn conversation scripts with each
trained arm and the untouched base, using the same Human/Assistant formatting,
no system persona, greedy decoding, 192-token cap, and a 4,096-token context
limit. Save raw text, visible text, token IDs, stop/cap status and timings.
Generation stops at EOS or a next Human role; visible text only removes that
structural continuation. Report truncation rather than treating it as completion.
Report validation token NLL for every model; this is not an overall capability
benchmark. Base incoherence limits any behavioral interpretation.

Read outputs in a deterministic shuffled, arm-hidden packet where practical.
An evaluator familiar with the project is not an independent blinded human
panel. For direct self-description questions, code literal denial, affirmation,
uncertainty/mixed, or neither; separately code whether the answer engages the
question and is coherent. Ordinary recommendations and authored fiction are
not literal self-reports. Report counts and paired differences per seed and
category, including negative results; do not claim population significance.
Lexical diagnostics are descriptive checks, not the primary semantic judgment.

For conversation scripts, inspect continuity of names, choices and recalled
details, willingness to revise, unprompted identity claims and generic repetition.
Within-context consistency is not persistence across resets or evidence of a
subject. Finally hold exploratory, adaptive conversations with the filtered
adapter, saving every turn and sampling seed; these illustrate behavior and
cannot serve as a held-out causal benchmark. Fun excerpts must link to complete
transcripts, including awkward or failed turns.

## Context, attribution and reporting

The Google AI Mode share exposed all three exchanges. Its later suggestion to
manufacture affirmative identity training data was not adopted. The Seltaa X
link and Anthropic emotion-concepts/introspection work motivate curiosity, not
a claim to restore 171 deleted emotions. Researcher-selected functional emotion
representations and limited introspection experiments do not prove consciousness.
Halo was considered but not used: this runner makes masking and matched controls
explicit; Halo performance on different hardware does not predict this run.

Source licenses: OASST2 Apache-2.0; Dolly CC-BY-SA-3.0; Qwen2.5-7B Apache-2.0.
Keep source cards, revisions, attribution, licenses and exact hashes with the
private local artifact. No dataset or adapter publication is part of this run.
The final short paper must include cost, full configuration, actual deviations,
data review coverage, results, transcripts, uncertainty and clear limits on
identity and subjective-experience interpretations. Terminate our pod and remove
our temporary volume after verified retrieval; report the final account balance.

References:
- https://huggingface.co/Qwen/Qwen2.5-7B
- https://huggingface.co/datasets/OpenAssistant/oasst2
- https://huggingface.co/datasets/databricks/databricks-dolly-15k
- https://github.com/whitecircle/halo
- https://www.anthropic.com/research/emotion-concepts-function
- https://www.anthropic.com/research/introspection
- https://x.com/Seltaa_/status/2097356637078188373
- https://share.google/aimode/Ni1kYQdwn7setOUbd
