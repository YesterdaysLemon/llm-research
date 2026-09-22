# Execution amendment, before behavioral evaluation

The technical smoke measured 3.16517 seconds per normal update and selected
both seeds before any generated answer was inspected. To reduce wall-clock
waiting, execute independent arms on three identical H100 80GB GPUs, with two
GPUs on a second pod. Each arm still performs both seeds, two epochs, all 384
updates, and all fixed evaluations. Original seed 29017 continues unchanged;
its sequential parent is replaced by a worker that waits for its completion.
The other two workers use separate CUDA_VISIBLE_DEVICES values and write to
distinct arm/seed folders on the same network volume. No concurrent writes to
the same model, checkpoint or results files are permitted.

Total GPU rate is $10.47/hour. Shorten the deadline for BOTH pods to
2026-09-22 04:28:12 UTC, two hours after the first pod was created. Thus at most
six GPU-hours are allocated ($20.94 plus small storage costs), within the $22
allocation. Each pod has a separate local termination watchdog. There is no
top-up. This hardware scheduling change does not alter data, initialization,
loss, optimizer, hyperparameters, seed selection or decoding. Save per-run
software/GPU provenance and compare paired initial-adapter hashes afterward.

The pre-output coding specification is in analysis-plan.md. Primary descriptive
counts use the eight direct and four preference probes; all other fixed categories
are still reported. Evaluation remains exploratory and single-agent rated.
