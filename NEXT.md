# NEXT

Last updated: 2026-08-18

## Current evidence

E001-P0 produced a consistent but limited signal: basis-invariant relational trajectory distillation improved a small student's accuracy on held-out ordered relation pairs compared with labels, logits, pointwise matching, and shuffled trajectories.

The gain mostly appeared at composition depth two and weakened sharply with depth. This does not yet show transfer of a general composition algorithm.

Read the [pilot report](experiments/E001-trajectory-sufficiency/report.md) before interpreting or extending the result.

## Next decision

Run E001-P1 only after preregistering controls that distinguish teacher information from a generic relational regularizer:

1. matched-dimensional random targets;
2. relations derived from an untrained teacher;
3. student self-relations without teacher information;
4. additional labeled examples matched to the trajectory target's stored bytes;
5. a development-only auxiliary-weight sweep fixed before the comparison.

Do not move to billion-parameter models until a relational condition improves depth-three and depth-four held-out composition and beats these controls.

## Start-of-session checklist

1. Run git status and preserve unrelated work.
2. Read the active hypothesis, most recent report, and this file.
3. State one decision the session can change.
4. Freeze the configuration or analysis rule before looking at the relevant result.
5. End by updating the report or this file, including null and negative outcomes.
