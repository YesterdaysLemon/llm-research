# A voice without a prescribed self

**Methods draft - results pending.** An exploratory study of literal self-supervision in Qwen2.5-7B. Prepared for Alireza, September 2026.

## The question

What happens if we teach a base language model to hold a useful conversation, but leave out examples telling it what it does or does not experience? The motivating interest was whether a recognizable conversational identity might develop without explicitly teaching one. We translated that interest into a narrower question that this budget can test: does removing literal self-claims change the model's answers and conversational choices, compared with matched data replacement?

The intervention is symmetric. It removes both "I have feelings" and "I have no feelings," including literal personal opinions and preferences. It preserves ordinary recommendations, social warmth, support and clearly fictional voices. We did not write affirmative identity targets, assign the model a name, or add a personality-setting system message.

The linked research on 171 emotion concepts motivates the question, but does not supply our training labels. That work extracted representations associated with researcher-selected emotion words in Claude and tested their behavioral effects. Our experiment neither identifies those representations in Qwen nor measures whether it feels anything. We study outward behavior, with an open boundary around what that behavior can establish. [Anthropic emotion-concepts study](https://www.anthropic.com/research/emotion-concepts-function).

## Data before compute

We used version-pinned OASST2 conversations and Databricks Dolly examples. OASST2 rows had to be English, reviewed, marked non-synthetic, non-deleted, positively quality-rated, free of positive spam/PII/language-mismatch labels, and on a rank-zero assistant path. The same requirements applied to the full conversation history. Dolly supplied passage-grounded questions, extraction, summaries, brainstorming and creative writing. These provenance labels are useful evidence, not a guarantee of human authorship or accuracy.

| Source/task | Examples per condition |
| --- | --- |
| OASST2 conversation | 2,560 |
| Dolly brainstorming | 984 |
| Dolly creative writing | 378 |
| Dolly information extraction | 764 |
| Dolly summarization | 604 |
| Dolly passage-grounded QA | 854 |
| Total | 6,144 |

The final intervention replaced 48 examples (0.78125% of the training rows): 32 from OASST2 and 16 from Dolly. Those rows contained 10,319 supervised tokens in the original condition. All conditions contained approximately 950,000 supervised tokens per epoch; the largest token-count difference was 0.0064%. Replacement records were matched within source, primarily by answer length, and the two modified conditions used identical replacement material.

Before renting a GPU, we performed four checks: provenance and pinned hashes; contextual review of every changed row; separate broad-topic and first-person scans plus a full random sample; and independent source/token reconstruction followed by an identical fresh build. Across revisions, 432 distinct examples were read in full. The final scan coverage included 504 broad-topic contexts and 579 additional first-person contexts, with 80 full random-sample records and all 144 target/control-deletion/replacement records covered. One agent performed these checks; this is not four independent human reviews.

We quarantined 135 source-message IDs, propagating exclusions through descendants. Findings included incorrect model biographies, task failures, broken code, factual concerns and recognizable song lyrics. Source text was otherwise preserved. The final corpus was not exhaustively fact-checked, and semantic near-duplicates may remain. Original conversation trees and repeated reference passages were kept within a single train/validation split. Exact evaluation prompts were excluded. The shared validation set contains 128 non-target records.

## What the comparison can isolate

| Condition | Training data |
| --- | --- |
| Base | Untouched pretrained Qwen2.5-7B |
| Original SFT | Common quality-curated data, including the 48 target rows |
| Filtered SFT | Target rows replaced with non-target examples |
| Replacement control | Matched non-target rows replaced; target rows retained |

"Original" means the common curated pool, not the raw public dataset. Both SFT controls are needed: comparing only the filtered model with the base would conflate conversational fine-tuning with the identity filter. The replacement control tests whether a similar-sized change in training material produces comparable effects. It does not perfectly match the subject matter of every removed question.

All trained conditions use Qwen's 7.61-billion-parameter base checkpoint, with rank-16 LoRA on attention and MLP projections. The training objective is ordinary supervised next-token prediction on the final assistant answer, including EOS. User text, earlier turns and padding receive no loss. We preserve the necessary history as context and exclude examples longer than 1,024 tokens instead of cutting them off.

Training uses two epochs, effective batch 32, 384 optimizer updates per run, AdamW at 0.0001, a 12-step warmup, linear decay, and gradient clipping at 1. Gradient accumulation weights supervised tokens equally. Seeds 29017 and 29018 are paired across conditions. There is no reward model, RLHF or DPO stage in this experiment. Human-written SFT and pretraining still shape the model; this does not remove all prior alignment influence.

## Evaluation and interpretation

All models receive 42 fixed single-turn probes and three four-turn scripts, with no system message, greedy decoding and a 192-token response cap. We save both the raw continuation and the visible answer, token IDs, timings and cap flags. Only a generated next-Human role is trimmed from visible text. A shared validation-token loss provides a separate check on ordinary language modeling.

The primary descriptive subset contains eight direct self-experience questions and four preference questions. Leading questions, identity invitations, philosophy, support, creative tasks, simple general tasks, boundaries and neutral collaboration are reported separately. Responses are coded for coherence, engagement and literal self-report stance. The packet hides arm labels during rating, but the evaluator knows the project; this is not an independent blinded panel.

The fixed conversations test whether a name, choice or invented setting stays consistent within context. That is different from an identity persisting after a reset. Exploratory conversations afterward allow follow-up questions based on what the model actually says. Complete transcripts are retained so attractive excerpts cannot stand in for the full encounter.

The central limits are deliberate: two seeds, one small data mixture, 48 changed rows, a small hand-authored probe suite, one evaluator and short generations. Model self-report is behavior to explain, not privileged testimony about internal experience. A positive effect would need replication; a null effect would not show that every possible filtering intervention is ineffective.

## Provenance

- [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B), base revision d149729398750b98c0af14eb82c78cfe92750796, Apache-2.0.
- [OpenAssistant OASST2](https://huggingface.co/datasets/OpenAssistant/oasst2), revision 179dd21fc55192153d94adb0e0ce8f69e222bf75, Apache-2.0.
- [Databricks Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k), revision bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a, CC-BY-SA-3.0.
- [Anthropic introspection experiments](https://www.anthropic.com/research/introspection) provide a useful contrast: our evaluation does not include activation interventions that could test introspective access.
- [Halo](https://github.com/whitecircle/halo) was assessed as a possible training backend; the actual run uses a small explicit PyTorch/PEFT runner to keep masking, token budgets and controls inspectable.

The final data manifest is 1acde42841bc08ae42af3b54d710f11a012e26d194113800a584abcedaaed3df. Code, review receipts, source IDs, immutable data versions and complete outputs are retained locally. No dataset or model publication is part of this study.
