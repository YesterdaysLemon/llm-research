# Completed study: a voice without a prescribed self

[Read the public paper](https://lyrebird.alirezaafshan.com/papers/voice-without-a-prescribed-self/), or its
[editable Markdown source](paper.md). Six 7B LoRA runs completed: original,
filtered and matched replacement control, each with two seeds. The untouched
base and all six adapters completed 42 probes plus three four-turn scripts.
Ten adaptive sampled turns followed on prospectively selected filtered seed 1.

The filtered models gave 3 and 4 denials among 12 primary questions, versus
4 and 5 for original SFT and 3 and 5 for replacement controls. The effect is
small, with substantial control overlap. Shared conversation habits, fluent
self-claims and consistent choices within context do not establish a persistent
identity or subjective experience.

- [Full count tables and every primary prompt](results.md).
- [All 294 fixed probe answers](fixed-probes.md).
- [All 84 fixed conversation turns](fixed-conversations.md).
- [All 10 exploratory answers](exploratory-conversations.md), including mistakes.
- [Coding decisions and remaining data defects](annotation-notes.md).
- [Structured results](summary.json), [manual ratings](ratings.jsonl),
  [arm-hidden packet](blind-packet.jsonl), and [mapping](blind-key.json).
- [Paired initialization and run-integrity proof](paired-run-integrity.json).
- [Aggregate observed cost and verified teardown](completion.json).

Observed study cost was approximately $10.14, with zero hourly spend after verified teardown. Personal account balances are omitted from this public edition. Billing may settle later; this is not a final invoice. No model weights or always-on inference service are published.

Every one of the 213 archived files passed size and SHA-256 verification.
Six final adapters and twelve epoch checkpoints are local at
`D:/Interiority-V1/cloud/retrieved/results`. The archive also contains data,
sources, model metadata, code, logs and raw outputs. Base weight shards are
public, pinned and reproducible; they are not included in the archive.

The 8 GiB local RTX 4060 cannot hold native BF16 7B inference. A validated
quantized chat deployment is not part of this completed run. See the
[reproduction guide](../README.md) for the verified CUDA inference workflow.
