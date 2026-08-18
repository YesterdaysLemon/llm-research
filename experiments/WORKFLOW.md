# Experiment workflow

This is the default loop for research time in this repository.

## Evidence states

An idea moves through:

1. **seed** — an intuition worth articulating;
2. **hypothesis** — a directional claim with a rival and kill condition;
3. **design** — frozen data, controls, metrics, and stop rule;
4. **smoke** — code and validity check only;
5. **pilot** — bounded fixed comparison;
6. **replication** — new seeds, task family, architecture, or hardware;
7. **supported, not supported, inconclusive, or retired**.

A pilot cannot skip directly to “established.”

## Before a run

1. Read the linked hypothesis and latest report.
2. Write the exact decision the run can change.
3. Freeze the configuration, seeds, primary metric, comparison conditions, exclusions, and stop rule.
4. Run deterministic unit tests.
5. Record the configuration SHA-256 and Git commit.
6. Prefer a clean worktree. If a run must use a dirty tree, record the complete status and configuration hash.

## Smoke runs

A smoke run may change the design only when it reveals:

- broken code;
- impossible memory or runtime requirements;
- an invalid control;
- a teacher or positive control that cannot perform the task;
- data leakage or a malformed split.

It may not be used to select the result that looks most favorable. Preserve smoke metrics and document any amendment before the fixed run.

Never edit a configuration in place after it has produced an artifact. Copy it to a new filename, make the amendment there, and link both configurations from the report.

## Fixed runs

- Run every registered condition and seed.
- Do not stop because early conditions look positive or negative.
- Write raw results to a new file; never overwrite a completed fixed run.
- Record per-seed and per-subgroup metrics, not only means.
- Keep stored bytes, active computation, wall time, and energy as separate quantities.
- If power telemetry is unavailable, say so. Do not relabel a power-limit bound as measured energy.

## Report

Every report answers:

1. What was predicted?
2. What was compared?
3. What happened in absolute terms?
4. Did the negative controls separate?
5. Where did the effect disappear?
6. What alternative explanation remains?
7. Which hypothesis status, if any, changes?
8. What is the smallest decisive next test?

Include nulls, failures, amendments, and unavailable measurements.

## End of a research session

1. Run tests and repository validation.
2. Update the experiment report.
3. Update [NEXT](../NEXT.md) with one bounded next decision.
4. Inspect the complete diff and secret scan.
5. Commit intentionally and publish only the validated research state.

The objective is not a stream of positive results. It is a durable chain from intuition to evidence.
