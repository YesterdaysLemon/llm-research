# E004 capability-smoke report

Run: 2026-08-19  
Frozen commit: `bc4ca00fdc4b3edbb06d387525220b940e1447c0`  
Configuration SHA-256: `7bff46ca8ac07e92312ba167ada2012f21f20b0178ba8816793f16c76637d0c5`  
Result SHA-256: `2acf8356169932dff5df04e47df160537aa6c92149bbd9d56a4ac94f862a15fd`

## Registered decision

The capability-only smoke asked whether the 10,540,160-parameter teacher and
1,575,360-parameter labels-only student could learn the mixed natural and
controlled causal stream well enough to license a fixed KD/geometry study. No
distillation or geometry condition ran. Each model was evaluated at 400, 2,400,
7,200, and 21,600 cumulative steps and would stop at its first passing rung.

**Decision: neither model passed any rung. The fixed H-010 comparison is not
licensed.** This is a benchmark/capability failure, not evidence for or against
activation-specific distillation.

## Absolute results

The add-one-smoothed training-unigram baseline was `5.3530` nats per natural
validation token. Chance controlled-answer accuracy was `1/11 = 9.09%`.

### Teacher

| Step | Natural NLL | ID overall | ID d2 | ID d3 | ID d4 | Held-out overall | Held d2 | Held d3 | Held d4 | Held d6 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 400 | 3.4565 | 10.35% | 13.45% | 6.43% | 11.18% | 7.42% | 5.47% | 10.16% | 7.81% | 6.25% |
| 2,400 | 2.6326 | 10.74% | 14.04% | 6.43% | 11.76% | 8.79% | 5.47% | 10.94% | 7.81% | 10.94% |
| 7,200 | 2.2786 | 8.59% | 8.19% | 9.94% | 7.65% | 8.01% | 6.25% | 11.72% | 6.25% | 7.81% |
| 21,600 | 2.1197 | 16.80% | 29.24% | 11.70% | 9.41% | 12.89% | 17.97% | 8.59% | 11.72% | 13.28% |

At the final rung, accuracy by ordered held-out pair was 10.16%, 9.38%,
16.41%, and 15.63%. The teacher learned some shallow behavior but missed the
70% ID, per-depth, held-out, and per-pair gates.

Teacher training took 587.45 cumulative seconds excluding evaluation and
recorded 384,376,832 peak CUDA bytes through PyTorch. It saw 150,088 complete
controlled records, 3,558,659 controlled supervised tokens, and 17,694,720
natural supervised tokens.

### Student

| Step | Natural NLL | ID overall | ID d2 | ID d3 | ID d4 | Held-out overall | Held d2 | Held d3 | Held d4 | Held d6 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 400 | 3.6791 | 8.59% | 6.43% | 6.43% | 12.94% | 7.81% | 6.25% | 7.81% | 7.03% | 10.16% |
| 2,400 | 2.8190 | 8.98% | 13.45% | 7.02% | 6.47% | 10.16% | 7.81% | 14.84% | 11.72% | 6.25% |
| 7,200 | 2.4394 | 9.77% | 13.45% | 8.77% | 7.06% | 8.40% | 7.03% | 11.72% | 7.03% | 7.81% |
| 21,600 | 2.2090 | 6.84% | 4.09% | 5.85% | 10.59% | 7.62% | 5.47% | 13.28% | 4.69% | 7.03% |

Student training took 268.26 cumulative seconds excluding evaluation and
recorded 241,271,808 peak CUDA bytes. It saw 150,053 complete controlled
records. It missed the ID gates and the 20% held-out floor at every rung.

These peak figures are framework-allocated CUDA bytes, not whole-system memory
or energy. The local driver did not expose power draw, so no joule result is
reported.

## What was learned

The natural side of the mixed LM objective worked: both models beat the frozen
unigram baseline at the first rung and continued improving. The controlled
answer side did not. Teacher depth-two ID and held-out accuracy rose above chance
at the final rung, but depth three/four and most pair cells remained near chance.
The student never developed credible controlled capability.

The cleanest live explanation is objective dilution. Each complete controlled
record contributes exactly one answer target. At the final teacher rung, answer
positions were therefore 150,088 out of 21,253,379 supervised positions, or
0.706%. Ordinary causal CE can reduce loss substantially by learning natural
text and the repetitive controlled grammar without learning the modular
arithmetic answer. This is an interpretation, not a demonstrated cause.

Other live explanations include inadequate architecture for state tracking,
interference between natural and controlled streams, absolute-position effects,
or an optimization failure unrelated to target frequency. The present run does
not distinguish them.

## Answer-weighted diagnostic outcome

The separately registered diagnostic multiplied loss at the controlled
answer position by `25`, approximately the mean controlled-record length
(`495,395 / 20,000 = 24.77` tokens). This makes one answer contribute about as
much controlled loss as the rest of its record while leaving every token,
architecture, seed, split, gate, and ladder rung unchanged.

It produced large depth-two ID gains but the teacher and student still failed
every positive-control decision at the final rung. The benchmark is therefore
retired for H-010 and no KD or geometry condition may run. See the
[diagnostic report](report-answer-weighted.md).
