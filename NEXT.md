# NEXT

Last updated: 2026-08-18

## Current evidence

E001-P0 produced a consistent but limited signal: basis-invariant relational trajectory distillation improved a small student's accuracy on held-out ordered relation pairs compared with labels, logits, pointwise matching, and shuffled trajectories.

The gain mostly appeared at composition depth two and weakened sharply with depth. This does not yet show transfer of a general composition algorithm.

Read the [pilot report](experiments/E001-trajectory-sufficiency/report.md) before interpreting or extending the result.

## Active decision

E001-P1 stopped before confirmation because every registered student missed its capability gate. Read the [development report](experiments/E001-trajectory-sufficiency/report-p1-development.md).

The next bounded experiment is E002: compare the generic Transformer with a compact context-selected transition executor. The question is whether architectural parameter reuse produces compositional capability more efficiently than adding teacher trajectories to an unsuitable student.

## Start-of-session checklist

1. Run git status and preserve unrelated work.
2. Read the active hypothesis, most recent report, and this file.
3. State one decision the session can change.
4. Freeze the configuration or analysis rule before looking at the relevant result.
5. End by updating the report or this file, including null and negative outcomes.
