# Compression frontier

Last reviewed: 2026-08-18

## The strongest version of the idea

Language models generalize, so they already compress regularities in their training distribution. The open question is whether their present parameterization, precision, architecture, training data, and inference policy contain enough avoidable redundancy to yield another large improvement in capability per parameter and per joule.

“Without loss of generality” needs an operational definition. No finite benchmark establishes preservation on every possible input. This project will instead ask whether a method preserves:

- average capability on a frozen broad suite;
- difficult and rare-tail tasks;
- calibration and robustness;
- out-of-distribution composition;
- editability and continual-learning capacity; and
- measured inference efficiency on the target hardware.

## Why substantial compression is plausible

### Numerical precision is redundant

GPTQ reduced large GPT-family weights to three or four bits with negligible degradation on its reported evaluations and achieved hardware-dependent inference speedups ([Frantar et al., 2022](https://arxiv.org/abs/2210.17323)). Moving from FP16 to four-bit weights is already approximately a fourfold storage reduction before architectural changes.

### Many parameters can be skipped on common metrics

SparseGPT reported at least 50% one-shot sparsity with minimal accuracy loss in large GPT-family models ([Frantar & Alistarh, 2023](https://arxiv.org/abs/2301.00774)). Sparsity only becomes an energy or latency win when the hardware and kernels actually skip the work.

### Compression methods can multiply

Pruning, quantization, and distillation produced complementary size reductions across BERT variants and GLUE tasks ([Movva et al., 2022](https://arxiv.org/abs/2208.09684)). Earlier vision models achieved 35–49× storage compression without loss on their reported accuracy metric, with smaller but still material energy gains ([Han et al., 2015](https://arxiv.org/abs/1510.00149)). Sparse, quantized BERT work reported 40× encoder compression with less than one percent loss on SQuAD ([Zafrir et al., 2021](https://arxiv.org/abs/2111.05754)).

These results justify searching for an order of magnitude. They do not establish a hundredfold reduction for a general-purpose autoregressive model.

### Parameter count is not equal to information content

Controlled factual-learning experiments estimated about two bits of stored knowledge per parameter under their setup ([Allen-Zhu & Li, 2024](https://arxiv.org/abs/2404.05405)). A separate memorization study estimated roughly 3.6 bits per parameter when generalization was removed ([Morris et al., 2025](https://arxiv.org/abs/2505.24832)). These are not universal constants, but they make it unreasonable to equate a 16-bit stored weight with 16 independent bits of useful knowledge.

### Representation geometry correlates with compression

The intrinsic dimension of linguistic representations has been reported to correlate with information-theoretic coding length and ease of adaptation ([Cheng et al., 2023](https://openreview.net/forum?id=ESgkAKGUJP)). This connects the geometric intuition to a measurable compression question, although correlation does not identify a transferable algorithm.

## Why literal lossless generality is doubtful

### Average benchmarks hide the tail

Small-magnitude weights that appear removable on common metrics can matter increasingly on difficult downstream tasks, and subsequent training may not recover the loss ([Yin et al., 2024](https://arxiv.org/abs/2310.02277)). The model may use redundant routes for ordinary prompts and reserve weak-looking components for rare compositions, calibration, or robustness.

### Distillation has a capacity gap

A stronger teacher can produce a worse student when the student's hypothesis space or optimization process cannot absorb the teacher distribution. Controlled scaling experiments found that distillation is useful only in particular data and compute regimes and does not beat sufficiently resourced supervised learning in the limit ([Busbridge et al., 2025](https://proceedings.mlr.press/v267/busbridge25a.html)).

### Scaling laws imply real capacity tradeoffs

Language-model loss has followed predictable power laws in parameter count, data, and training compute across wide ranges ([Kaplan et al., 2020](https://arxiv.org/abs/2001.08361)). Existing systems are not perfectly optimized, but the smooth benefit from scale argues against all extra parameters being disposable copies.

### Compression target and hardware matter

Four different quantities are often conflated:

1. **file size** — bytes used to store weights;
2. **active parameters** — weights touched for a token;
3. **operations** — arithmetic actually executed;
4. **energy** — hardware-specific joules including memory movement and overhead.

A Huffman-coded or unstructured sparse model may be much smaller on disk without being faster. A smaller dense student can be faster while losing rare capabilities. Every result must name its denominator.

## What activation trajectories can contain

For fixed weights, an activation trajectory is a deterministic function of the input and prior sampled tokens. It therefore contains information about how that particular model transformed that input. But its coordinates only have meaning relative to the surrounding weights.

The interesting residual question is:

> After accounting for labels, logits, examples, and arbitrary coordinate choices, do relations across trajectories provide a compact sufficient statistic that improves a smaller student's behavior on unseen compositions?

Possible transferable objects include:

- pairwise similarities among examples at each layer;
- changes in similarity from early to late layers;
- low-rank subspaces associated with task distinctions;
- token-to-token attention relations;
- trajectory velocity, curvature, or convergence;
- causal effects of perturbing a subspace; and
- which examples the teacher separates but the student collapses together.

Raw pointwise activations are a weak target because teachers and students can use different bases and widths. Gram matrices, centered-kernel alignment, distance ranks, and learned optimal-transport alignments are closer to basis-invariant hypotheses. Relational contextual distillation has already improved compressed language representations in some settings ([Park et al., 2021](https://aclanthology.org/2021.emnlp-main.30/)), and multi-granularity relational objectives have also reported gains ([Liu et al., 2022](https://aclanthology.org/2022.acl-long.71/)).

## The decisive controls

An activation method is interesting only if it beats:

- ground-truth training alone;
- additional training examples at the same byte and compute budget;
- teacher hard labels;
- teacher logits;
- pointwise hidden-state matching;
- trajectories shuffled between examples;
- a random projection with the same dimension;
- teacher trajectories after an orthogonal basis change; and
- output distillation plus an equally sized auxiliary loss.

If real relational trajectories beat shuffled trajectories but survive orthogonal basis changes, they contain task-linked structure rather than merely stable coordinates. If they improve imitation but not held-out composition, they are a compression code for the observed sample rather than a transferred algorithm.

## Practical thesis

The most defensible current bet is:

> Present LLMs contain enough numerical, structural, training, and inference redundancy for another order-of-magnitude storage improvement in selected regimes, but preserving the full difficult tail will require learned compression objectives rather than magnitude pruning alone.

The more speculative bet is:

> Basis-invariant relations across activation trajectories expose teacher computations in a form that lets smaller students preserve more of that difficult tail per parameter.

The first experiment targets the speculative bet because ordinary quantization and pruning already have strong baselines.
