# E010: Learned retention under cache compression

Date: 2026-09-22. Status: prospective local pilot, before outcomes.

## Decision

Does exposing a lightweight retention policy to altered KV states improve held-out retrieval after repeated eviction and INT8 cache storage, compared with the same policy trained on intact traces?

This is a scaled, independently implemented KVP-style ranking-objective experiment. It is not a reproduction of Apple's full benchmark or implementation. The base model stays frozen. There is no weight quantization in this first pilot. Cache quantization and weight quantization must not be conflated.

## Fixed design

- Local, existing Qwen2.5-1.5B-Instruct weights; BF16 inference on the RTX 4060. Record file hashes and software versions.
- Deterministic synthetic lookup and correction tasks, with disjoint train (seed 11000), smoke (21000), and test (31000) generators. All examples and prompts retained. No executable generated code.
- 32 training traces; 24 held-out test tasks, half requiring a later correction to override an earlier value. One base model, one task family, three policy seeds: 41, 42, 43.
- A separate two-layer width-32 MLP for each KV head and layer, input post-RoPE keys, values, absolute position features. Plackett-Luce rankings, sampled with Gumbel sorting, trained by leave-one-out REINFORCE on future attention cost. Labels use future tokens only during training.
- Native and compression-exposed training use the same examples, token positions, teacher utility labels, initial policy parameters, and optimizer updates. Training contexts retain their target fact and correction, using eight distractor records and an irrelevant preamble. Exposed features are obtained after evicting preamble positions 4 through 63 before processing the last 64 context tokens, with real INT8 storage. Features are aligned to identical retained positions in both arms. This is a fixed exposure intervention, not on-policy training; it does not train the policy on its own eviction decisions.
- Baselines: full cache, sink-plus-recency, and seeded random selection with the same protected prefix/recent window. Learned policies use the same protection. The random baseline is a negative control, not a competitive claim.
- Resident working-memory budgets: 4 and 8 MiB for packed KV tensors, per-token scales and positions, and policy parameters. Baselines receive the bytes not occupied by a policy. FP16/BF16 and INT8 storage each tested. Both cache tensors are physically shortened; masks alone do not count as eviction.
- Chunked prompt processing (64 tokens), then up to 12 greedy generated tokens. Compress after every prompt chunk and generation step. Record transient CUDA allocated/reserved peaks separately: the resident budget is not a hard cap on transient working memory or total GPU use.
- Primary endpoint: exact target-value retrieval (normalized first answer token). Report correction and ordinary cases separately. Secondary: teacher-relative answer token NLL, measured resident bytes, CUDA peak allocated/reserved bytes, prompt/generation timing. Seed differences are not independent task samples; retain per-task paired records.
- Fixed 160 optimizer updates per training arm and seed. No best-checkpoint selection. Train only controllers. Test outputs cannot tune prompts, policies, or hyperparameters.

## Gates and interpretation

Before the fixed run, the full-cache model must answer at least 6/8 separate smoke tasks correctly; otherwise preserve the failed gate and stop this task design. Code correctness gates include full-cache parity against stock Transformers, explicit chunk causal masking, stable absolute positions after eviction, storage-byte accounting, and disjoint generator seeds.

Every fixed condition runs even after an unfavorable outcome. A controller improvement is exploratory unless replicated on new data/models. No claim of fitting a larger model, speedup, general reasoning improvement, independent review, or novelty follows from this pilot. At these short contexts controller overhead may erase cache savings; report that plainly. Contemporary comparators such as SnapKV, ForesightKV, LKV, and AgentKV are required before a competitive-method claim.

## Prior art

- KVP: https://arxiv.org/abs/2602.10238
- ForesightKV: https://arxiv.org/abs/2602.03203
- LKV: https://arxiv.org/abs/2605.06676
- DistillCache: https://arxiv.org/abs/2608.08878
- AgentKV: https://arxiv.org/abs/2609.14872
- HqeKV: https://aclanthology.org/2026.findings-acl.201/

The proposed contribution is a reproducible robustness and resource-accounting pilot, not a new claim to learned eviction.
