# E004 capability diagnostic — answer-weighted causal loss

Registered: 2026-08-19, after recording the failed initial smoke and before
executing this configuration.

## Motivation and exact decision

The initial E004 capability smoke failed every controlled teacher and student
gate while both models learned natural text. At the final teacher rung, one
answer target occurred for each of 150,088 complete controlled records, but the
mixed objective contained 21,253,379 supervised positions. Answer positions
were therefore only 0.706% of the causal loss sites.

This diagnostic asks one question: **does making the controlled answer position
material to the same causal objective establish the positive controls needed by
the fixed study?** It does not test distillation or hidden geometry.

## Single amendment

Configuration: `config/smoke-answer-weighted.json`.

For controlled records only, multiply cross-entropy at the token immediately
after `?` by `25.0`. The mean frozen controlled-record length is
`495,395 / 20,000 = 24.77` tokens, so weight 25 makes the answer contribute
approximately as much controlled loss as the rest of its record. Natural tokens
and all other controlled tokens retain weight one. The loss is normalized by
the sum of weights, not by the unweighted token count.

The answer mask is derived only from the controlled source row and the `?`
input token. It is never applied to natural-text questions. Complete controlled
records contain one `?`, and the executable asserts that the number of answer
positions equals the number of packed controlled records.

Everything else is byte-for-byte or semantically identical to the initial
smoke: corpus, tokenizer, split, models, seeds, 80/20 sequence-row mixture,
optimizer, learning rate, batch, deterministic settings, gates, evaluation,
and the 400/2,400/7,200/21,600 ladder. The initial result SHA-256 is frozen in
the new configuration.

## Decisions and stop rule

Teacher and student independently stop at the first rung clearing the original
gates. If both pass, the selected budgets may be used in a separately committed
fixed KD/geometry protocol; answer weighting must then be shared by every fixed
condition.

If the teacher reaches 21,600 steps without passing, retire this controlled
benchmark for H-010. Do not increase the answer weight, controlled fraction,
model size, or budget, and do not run a KD/geometry condition. If the teacher
passes but the student misses its corridor at every rung, the fixed study also
remains blocked; a future benchmark or pretrained-teacher design receives a new
experiment ID.

This diagnostic can establish benchmark learnability, not algorithmic transfer,
activation-specific information, compression, or efficiency.

