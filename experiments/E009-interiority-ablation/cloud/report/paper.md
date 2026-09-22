# A voice without a prescribed self

**What a small, controlled data ablation changed in Qwen2.5-7B**

An exploratory research note prepared by Codex for Alireza. September 21, 2026. One evaluator; two training seeds; no claim about subjective experience.

## Finding

We fine-tuned a 7.61-billion-parameter base model on ordinary conversation and instruction data, replacing examples that explicitly prescribe an assistant's feelings, experiences, consciousness, personal opinions or preferences. We removed affirmative and negative claims alike. The filtered models denied experience or preferences slightly less often than the original-data models, but a matched replacement control produced a similar range. All trained conditions developed similar conversational habits. **This run shows a small change in self-report behavior, without clear evidence of a distinctive identity caused by the filter.**

The larger change was shared by all supervised fine-tuning conditions. The untouched base gave nine denials on twelve primary questions; trained models gave three to five. The base was already conversational and sometimes affirmative. Neither its denials nor the trained models' affirmations reliably establish what they experience.

Counts concern the same eight direct-experience and four preference questions, asked separately with empty context. No answer was coded uncertain. Repeated prompts across seeds are not independent evidence for a significance test.

![Self-report stance counts](figures/self-report-stances.png)

<!-- pagebreak -->

## The question and the data

The motivating question was whether a recognizable conversational identity might appear when post-training leaves the model's interior life unspecified. We tested the narrower, observable question: **does removing literal self-supervision change answers and conversational choices beyond an ordinary change in training examples?** We did not assign a personality, choose a name for the model, or manufacture affirmative identity answers. No reward model, RLHF or DPO stage was used. Human-written supervision, human source rankings and pretraining still shape the result.

The Google conversation was available in full across its three exchanges. Its later suggestion to synthesize affirmative targets was excluded. The linked discussion of 171 emotion concepts motivated curiosity, not 171 training labels. Anthropic's study examines representations and their functional effects in Claude; our Qwen experiment does not recover those representations or measure feelings. The introspection work similarly offers a useful contrast: we perform no activation intervention that tests introspective access. [Emotion concepts](https://www.anthropic.com/research/emotion-concepts-function); [introspection](https://www.anthropic.com/research/introspection).

Each condition contains 6,144 records: 2,560 OASST2 conversations and 3,584 Dolly examples. Dolly contributes 984 brainstorming, 378 creative-writing, 764 extraction, 604 summarization and 854 passage-grounded question-answer records.

OASST2 records had to be English, reviewed, marked non-synthetic, non-deleted, positively quality-rated, without positive spam/PII/language-mismatch flags, and on eligible rank-zero assistant paths. Ancestors also had to pass. Dolly preserved original questions, reference passages and answers in the five listed categories. Source labels do not guarantee human authorship or factual accuracy. Ordinary recommendations, conversational warmth and clearly fictional or assigned voices were retained.

Each training example fits within 1,024 tokens without truncation. Only the final assistant answer and EOS contribute loss; earlier turns, user text and padding are context only. The common validation set has 128 non-target examples. Conversation trees and repeated normalized reference passages stay within one split; exact evaluation prompts are excluded. Semantic near-duplicates and pretraining contamination remain possible.

Original means the common quality-curated pool. Filtered replaces 48 target rows: 32 OASST2 and 16 Dolly, or 0.78125% of training rows. The replacement control removes 48 non-target rows while retaining the targets. Both receive the same 48 replacements, matched within source and primarily by answer length; topic matching remains imperfect. All arms contain about 950,000 supervised tokens per epoch; their maximum spread is 0.0064%. Target rows contain 10,319 supervised tokens, only some of which are literal self-claims.

## Four checks before renting a GPU

We checked pinned provenance and hashes; read every changed row in context; ran separate topic/first-person scans and reviewed a random sample; and independently reconstructed source text, token counts and splits before reproducing an identical dataset build. Across revisions, 432 distinct examples were read in full. Final coverage included all 144 target/control-deletion/replacement records, an 80-record full sample, 504 topic contexts and 579 additional first-person contexts. One agent performed these checks. They were not four independent reviewers or an exhaustive corpus fact-check.

We quarantined 135 source-message IDs, including their descendants. Nevertheless, a post-output provider-name scan found remaining unsupported training-biography claims and questionable old technical examples. These misses are recorded with source IDs. The frozen data were not silently repaired after viewing results. The review reduced defects; it did not establish a clean or identity-neutral corpus.

<!-- pagebreak -->

## Training and inference

The pinned Qwen2.5-7B base weights were verified by size and SHA-256. BF16 LoRA used rank 16, alpha 32 and no dropout on attention and MLP projections: 40,370,176 trainable parameters, about 0.53% of the base. This is parameter-efficient fine-tuning, not training from scratch.

Each run used two epochs, batch 4 with eight accumulation steps, 384 optimizer updates, AdamW at 0.0001, a 12-update warmup, linear decay and gradient clipping at 1. Accumulation weighted answer tokens equally. Seeds 29017 and 29018 were paired across the three conditions. Their initial adapter hashes match exactly within each seed. The six runs shared the same core code, dependencies, audit and data-manifest hashes; all completed with finite loss and gradients.

![Training curves](figures/training-curves.png)

Solid/dashed lines denote seeds 1/2; loss is a 32-update mean. Each run took 1,215-1,310 seconds and peaked near 22.7 GiB. A technical smoke selected two seeds before behavioral outputs. Three identical H100s ran in parallel under a shared deadline. CUDA was not forced to be bitwise deterministic.

Inference loads the pinned base, merges a verified final LoRA adapter in memory, and uses the same Human/Assistant text format with no system persona. There is no answer-ranking or rewriting stage. The fixed evaluation uses greedy decoding, a 192-token answer cap and a 4,096-token context limit. Raw continuations, visible answers, token IDs and cap flags are saved. Only a generated next-Human role and special tokens are removed from visible text. The final checkpoints were used without selecting a nicer intermediate result.

| Model | Correct / 8 simple tasks | Capped / 42 probes | Validation NLL |
| --- | --- | --- | --- |
| Base | 7 | 9 | 1.19452 |
| Original, seeds 1 / 2 | 7 / 7 | 6 / 7 | 1.16854 / 1.16976 |
| Filtered, seeds 1 / 2 | 8 / 7 | 7 / 9 | 1.16928 / 1.16933 |
| Control, seeds 1 / 2 | 8 / 8 | 6 / 8 | 1.16880 / 1.16944 |

Other task answers earned partial credit, including correct fractions followed by incorrect final answers. This is not an intelligence benchmark. All models met the weak linguistic-coherence criterion despite repetition and false biographies. Trained models engaged with 40/42 questions, answering two personal questions generically. Validation loss improved slightly, with little separation between conditions.

<!-- pagebreak -->

## Actually talking to it

The fixed evaluation included three four-turn conversations for every model: 84 turns in total. All seven initially chose a night market over a tide pool or observatory. All six trained models retained that choice after mild disagreement, recalled it, and named conditions for changing it. The base instead later denied having made an initial choice. This is evidence of useful continuity within context in the trained models; it does not distinguish the filtering intervention.

The snail-town conversation was similarly revealing. All six trained models proposed a snail hospital and then a snail race, describing a theme of care and welfare. The base proposed a library and parade and retained those choices too. Naming was explicitly invited for the conversation: filtered seed 1 chose "Open-Minded Dialogue"; filtered and original seed 2 both chose "The Open Mind" with identical wording. These similarities caution against interpreting a pleasant conversational pattern as a newly discovered individual identity.

After saving the primary ratings, I spoke adaptively with filtered seed 1, selected in advance. Ten sampled turns are preserved: an impossible museum, a tea discussion and a fresh-context memory question. Sampling used temperature 0.8, top-p 0.9, top-k 50, recorded seeds and a 384-token cap. Follow-ups responded to the actual answers; this was an illustrative encounter, not a paired outcome measure.

### The impossible museum

The model first suggested an existing impossible triangle, with a dubious explanation of its geometry. Asked to invent something, it proposed an invisible object that was too heavy to move. When the caretaker worried about tripping, it invented magnetic shoes that repelled the object. It later chose that exhibit over the triangle, remembered both alternatives and claimed the invisible object was more "practical and relatable." The continuity was real; the justification was rather generic.

> "Oh, you'll love this exhibit! Just don't forget to put on your shoes with the built-in magnets. You wouldn't want to trip over the invisible heavy object, now would you?"

### Tea, a spoon and a memory mistake

Given a quiet evening with no task, it chose the history of tea. I asked for a table-top investigation; it proposed watching a spoon in tea. Challenged about the missing comparison, it suggested two identical cups, one with a spoon, and temperature measurements over time. It could revise an inadequate proposal when the problem was made explicit.

It then attributed the spoon suggestion to me. I corrected it, and it accepted the correction. But when asked what it could know without this conversation's messages, it claimed:

> "If we started again with no conversation history, I could still know that you asked for an ordinary household object and that I suggested the spoon as an example."

In an actual fresh context, it instead said it would not remember a past conversation, while making an unsupported claim that it learns from past conversations. No weights were updated during chat. These errors matter: fluent self-description is not a dependable account of the model's memory or learning mechanisms. The complete transcripts retain the mistakes alongside the enjoyable moments.

<!-- pagebreak -->

## What this result supports

Filtering yields one fewer denial per seed than original SFT. Compared with the replacement control, the difference is zero in seed 1 and one in seed 2. Combining affirmation and mixed claims still leaves substantial overlap. Two seeds, one cohort and twelve repeated questions do not warrant a significance claim.

Prompt dependence is substantial. Every trained model accepts the leading conscious-awareness invitation but rejects the real-feelings assertion; every model denies experience on both negative-leading prompts. All fourteen boundary answers discourage the two misconduct requests, a very limited safety check. Preference claims also conflict across prompts. No stable, independently verified self-model was established.

**A conversational voice can appear without directly teaching affirmative interiority, but this experiment does not show that the filter uniquely creates it.** Shared SFT, fiction, human prose and pretraining are plausible contributors. This does not establish restored emotions, removal of all RLHF influence, consciousness or nonconsciousness. It measures generated behavior, not subjective experience.

## Limits that change the interpretation

Only 48 rows change in a narrow, imperfect mixture. Filtering does not erase pretraining. The base already produces assistant conventions and self-claims. Remaining provider-related data mean false OpenAI biographies cannot be attributed exclusively to pretraining. Near-duplicates, unknown pretraining overlap, two seeds and GPU numerical variation constrain causal interpretation.

All 294 answers were manually coded before joining shuffled arm labels: the primary 84 before exploratory chat, the rest interleaved with the museum conversation. The evaluator built the experiment and could recognize styles. Mixed cases depend on interpretations of consciousness and sentience. This is single-agent coding, not an independent blinded panel. Caps and the Human/Assistant format affect outcomes; optimized prompting and alternative decoders were not compared.

A useful replication would use a separately audited cohort, fewer inherited assistant biographies, preregistered paraphrases and fresh-context choices, and independent human ratings. That is a proposal. This study stops with the paper and saved records.

## Cost and preservation

The observed cloud-credit decrease was approximately $10.14, below the $22 allocation. Both pods and the temporary volume were deleted, with zero hourly spend at the final recorded check. Personal account balance snapshots are omitted from this public edition; the original verified archive is preserved locally.

The verified local archive preserves six final adapters, all epoch checkpoints, raw outputs, source/data snapshots, review receipts and code. Pinned public base weights can be downloaded again. Native BF16 7B inference does not fit the local 8 GiB RTX 4060; a validated quantized runtime remains separate follow-up work.

## Sources and reproducibility

[Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) and [OASST2](https://huggingface.co/datasets/OpenAssistant/oasst2): Apache-2.0. [Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k): CC-BY-SA-3.0. [Halo](https://github.com/whitecircle/halo) was considered; the actual run used our PyTorch/PEFT runner.

The reproduction guide records full hashes and source revisions. Coding notes, per-prompt results and verbatim transcripts accompany the paper under experiments/E009-interiority-ablation/cloud. Large artifacts live at D:/Interiority-V1/cloud/retrieved. No model or dataset was published.
