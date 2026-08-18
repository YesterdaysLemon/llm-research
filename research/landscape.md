# Landscape review

Last reviewed: 2026-08-18

This is a first adversarial reading of the seed note. It distinguishes a useful intuition from the stronger claim that the available evidence can actually support.

## 1. Brain and model energy

### Seed

The brain uses less than 100 watts while LLMs require gigawatts for the same calculations.

### What survives

The adult human brain operates on roughly 20 watts of metabolic power, and signaling is a major part of its energy budget ([Balasubramanian, 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8364152/)). Machine-learning energy is large enough to deserve direct measurement: one eight-H100 node drew up to about 8.4 kW in a training study ([Latif et al., 2024](https://arxiv.org/abs/2412.08602)), and global data centers consumed an estimated 415 TWh in 2024 ([IEA, 2025](https://www.iea.org/reports/energy-and-ai/executive-summary)).

### What does not survive

There is no established equivalence between what a brain computes and what an LLM training run or inference service computes. Watts measure instantaneous power; watt-hours measure energy. A person, a single prompt, a training run, a deployed service, and the global data-center fleet are different denominators. “Gigawatts for the same calculations” is therefore not a defensible present claim.

### Better research form

Measure joules per accepted answer at a fixed quality threshold, along with latency and hardware. Algorithmic improvement can be as important as hardware: the compute required to reach AlexNet-level ImageNet performance fell 44-fold from 2012 to 2019 ([Hernandez & Brown, 2020](https://arxiv.org/abs/2005.04305)).

## 2. Frozen snapshots and continual learning

### Seed

A deployed LLM is frozen while a person keeps learning throughout life.

### Supporting evidence

Ordinary inference does not update model parameters. Sequential fine-tuning can also damage previously learned knowledge, formatting behavior, and reliability ([Li & Lee, 2024](https://arxiv.org/abs/2401.03129); [Luo et al., 2023](https://arxiv.org/abs/2308.08747)). This is the stability-plasticity problem in a concrete form.

Replay is a plausible mechanism. Generating and replaying internal representations reduced catastrophic forgetting on continual-learning benchmarks ([van de Ven et al., 2020](https://www.nature.com/articles/s41467-020-17866-2)). A sleep-like unsupervised phase also recovered old tasks in several image benchmarks, though it remained below stronger rehearsal methods in some settings ([Tadros et al., 2022](https://www.nature.com/articles/s41467-022-34938-7)).

### Counterevidence and qualification

Frozen weights do not imply frozen behavior. Context, tools, retrieval, and external memory can change what a model can use without parameter updates. A memory-compression method reported continual knowledge gains while keeping the base LLM fixed ([Li et al., 2024](https://arxiv.org/abs/2412.07393)). In-context adaptation is not the same as durable weight learning, but it can satisfy part of the practical requirement.

“Sleep” is too broad to be a mechanism. Replay, synaptic down-selection, regularization, and offline evaluation are different algorithms and should be tested separately.

### Better research form

Compare three memory timescales under the same budget: context memory, external persistent memory, and parameter updates. Ask which information belongs in each and when consolidation from one timescale to another pays for itself.

## 3. The geometry of meaning

### Seed

Words occupy a fixed semantic landscape, while nearby context selects a local path through it.

### Supporting evidence

Representations do contain geometric structure. Low-dimensional subspaces can encode linguistic features, and interventions in some of those subspaces can causally alter BERT outputs ([Hernandez & Andreas, 2021](https://aclanthology.org/2021.conll-1.7/)). Languages also change over historical time; diachronic embeddings recover measurable patterns of semantic change ([Hamilton et al., 2016](https://aclanthology.org/P16-1141/)).

### Counterevidence and qualification

The input embedding table may be fixed at inference, but a token's hidden representation is reconstructed at every layer and depends strongly on context. In BERT, ELMo, and GPT-2, a static word embedding explained less than five percent of the variance in contextualized representations on average ([Ethayarajh, 2019](https://aclanthology.org/D19-1006/)). The live landscape is therefore not simply static.

Decodability is not causality. A flexible probe can learn a task even when the source representation does not robustly encode the claimed structure; control tasks exposed this problem in popular probing setups ([Hewitt & Liang, 2019](https://arxiv.org/abs/1909.03368)).

### Better research form

Separate four objects: the token lookup table, context-conditioned hidden states, model parameters, and an analyst's low-dimensional projection. “Geometry changed” should specify which object changed, under which intervention, and whether that change caused behavior.

## 4. Distilling “thought patterns”

### Seed

A small model might become much smarter by learning the activation patterns of a much larger model rather than only its token outputs.

### Supporting evidence

This is close to internal-representation and attention distillation. Matching internal BERT representations improved over soft-label-only distillation on GLUE in one study ([Aguilar et al., 2019](https://arxiv.org/abs/1910.03723)). MiniLM compressed Transformers by distilling self-attention relations and reported strong task-agnostic transfer ([Wang et al., 2020](https://arxiv.org/abs/2002.10957)).

### Counterevidence and qualification

Teacher and student hidden spaces can differ in width, symmetry, layer role, and learned basis. A pointwise activation match may punish a student for finding a different valid computation. More generally, students often fail to match teacher predictive distributions, and closer teacher matching does not always produce better generalization ([Stanton et al., 2021](https://arxiv.org/abs/2106.05945)).

An activation is also not automatically a “thought.” To justify that language, the representation should predict behavior under held-out conditions and survive a causal intervention test.

### Better research form

Compare output-only distillation with relational hidden-state objectives, trajectory summaries, and mixed losses. Evaluate on tasks and compositions absent from the distillation set. The target is transfer of capability, not visual similarity between activation plots.

## 5. Wetness, noise, and reliability

### Seed

Biological intelligence is wet, noisy, and forgiving; silicon is discrete and reliable.

### Supporting evidence

Neural activity contains variability at many levels. Noise can sometimes improve weak-signal detection or exploration through stochastic facilitation, including stochastic resonance ([McDonnell & Ward, 2011](https://www.nature.com/articles/nrn3061)).

### Counterevidence and qualification

Noise is not generally beneficial. Its effect depends on amplitude, task, nonlinearity, and where it enters the system; excessive noise destroys information. Digital neural networks already use stochastic optimization, dropout, sampling, quantization, and noisy regularization. “Wet versus digital” is not the same axis as “stochastic versus deterministic.”

### Better research form

Look for an inverted-U relationship between a specified noise process and a specified outcome. Compare activation noise, gradient noise, data corruption, and sampling temperature separately.

## 6. Computability and physical substrate

### Seed

Silicon only handles computable problems, while it is unknown whether the brain's important problems are computable.

### Supporting evidence

There is a real philosophical distinction between the mathematical Church-Turing thesis, which concerns effective procedures, and stronger physical claims about simulating every physical system. The original thesis does not by itself prove that a digital machine can perfectly simulate a brain ([Copeland, 2023](https://plato.stanford.edu/entries/church-turing/)).

### Counterevidence and qualification

No reproducible cognitive behavior is currently known to require hypercomputation. Analog or noisy dynamics may be expensive to simulate precisely without computing a non-Turing-computable function. Practical intelligence can also be limited by complexity, data, embodiment, or interaction while remaining computable.

### Better research form

Do not use unknown computability as the default explanation for present capability gaps. First test ordinary explanations: architecture, learning rule, memory, data, sensors, action loops, and resource constraints.

## 7. Intelligence, consciousness, and substrate independence

### Seed

LLM success is evidence that intelligence and consciousness are not limited to biological media.

### What the evidence supports

LLMs are evidence that some capabilities associated with language and reasoning can be implemented in non-biological systems. That weakens claims that those capabilities require carbon-based neurons specifically.

### What the evidence does not support

Behavioral success alone does not establish consciousness. A theory-derived indicator review found no strong current AI candidate for consciousness while also finding no obvious technical barrier to systems satisfying proposed indicators ([Butlin et al., 2023](https://arxiv.org/abs/2308.08708)). A later framework emphasizes that consciousness science remains substantially uncertain ([Butlin et al., 2025](https://doi.org/10.1016/j.tics.2025.10.011)).

### Better research form

Keep three ledgers: capability, agency, and possible consciousness indicators. Evidence may move one ledger without moving the others. This repository is currently equipped to test capability and adaptation, not phenomenal experience.

## 8. Intelligence in an ecology

### Seed

Intelligence becomes more powerful in groups because coordination and cooperation create more than the sum of individuals.

### Supporting evidence

Across two studies with 699 people, group performance showed a general collective-intelligence factor across varied tasks ([Woolley et al., 2010](https://pubmed.ncbi.nlm.nih.gov/20929725/)). The broad intuition that group organization matters is empirically credible.

### Counterevidence and qualification

Groups do not automatically create superadditivity. Communication, coordination, duplicated work, conformity, and social loafing create process loss. Larger teams can reduce individual performance through relational and coordination losses ([Mueller, 2012](https://doi.org/10.1016/j.obhdp.2011.08.004)).

### Better research form

Compare groups at equal total inference and communication budgets. Manipulate member diversity, independence before discussion, aggregation rule, and adjudication. A larger transcript is not necessarily more intelligence.

## Provisional synthesis

The most promising bridge from the original philosophy to experimental work is a **multi-timescale adaptive system**:

1. context provides fast, volatile adaptation;
2. external memory provides persistent but revisable knowledge;
3. selective parameter updates provide slow skill acquisition;
4. offline replay tests and consolidates changes;
5. structured groups search independently before sharing;
6. energy and communication costs constrain every layer.

That is a research program, not yet a result. The [hypothesis ledger](../hypothesis.md) records tests that could make it lose.
