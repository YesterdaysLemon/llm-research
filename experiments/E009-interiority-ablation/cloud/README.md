# Cloud continuation: Qwen2.5-7B

**Completed September 21, 2026 (local time).** Read the
[five-page paper](../../../output/pdf/e009-voice-study.pdf),
[full results and transcripts](report/README.md), and
[coding/data caveats](annotation-notes.md). Six final adapters and twelve epoch
checkpoints are preserved. Both pods and the temporary volume were deleted;
RunPod reported $0/hour. [Completion receipt](report/completion.json).

Read [protocol.md](protocol.md), [execution-amendment.md](execution-amendment.md),
and [analysis-plan.md](analysis-plan.md). The original local 0.5B pilot remains
historical and failed its data review; it was not used for this training run.

The separately prepared cloud dataset passed four checks. Its immutable local
root is `D:/Interiority-V1/cloud`, with `prepared/final`, `audits/final`, pinned
`sources` and `base` metadata. Manifest SHA-256:
`1acde42841bc08ae42af3b54d710f11a012e26d194113800a584abcedaaed3df`.
The final candidate is `candidate-v7`; earlier candidates and failed receipts
are retained. `frozen-review.json` summarizes coverage and limits.
The precompute review is not a guarantee of error-free data: a post-output scan
found residual unsupported biographies and questionable technical examples.
These are documented without changing the frozen dataset or retuning models.

There are 6,144 records in each of three arms, 128 shared validation records,
and 48 literal self-supervision records replaced in filtered. Both interventions
use the same replacement material. The union contains 6,320 unique records.
135 source-message IDs are commonly quarantined, with propagation through
conversation descendants. No synthetic affirmative training answers are added.

## Run and reproduce

CPU preparation and independent audit, from the repository root:

```powershell
.venv/Scripts/python.exe experiments/E009-interiority-ablation/cloud/test_cloud.py
.venv/Scripts/python.exe experiments/E009-interiority-ablation/cloud/data.py --root D:/Interiority-V1/cloud --output D:/Interiority-V1/cloud/prepared/NEW_VERSION
.venv/Scripts/python.exe experiments/E009-interiority-ablation/cloud/audit.py --root D:/Interiority-V1/cloud --data D:/Interiority-V1/cloud/prepared/NEW_VERSION --output D:/Interiority-V1/cloud/audits/NEW_VERSION
```

These commands do not automatically clear semantic review. `freeze_review.py`
records the already completed interactive audit of candidate-v7; it must not
be repurposed to auto-approve another dataset. Changed data require new review.

On the verified CUDA/PyTorch runtime with pinned Python dependencies:

```bash
python code/run.py train --root /workspace/e009 --data /workspace/e009/prepared/final --audit /workspace/e009/audits/final --output /workspace/e009/results/NEW_RUN --arm filtered --seed 29017
python code/run.py evaluate --root /workspace/e009 --data /workspace/e009/prepared/final --audit /workspace/e009/audits/final --output /workspace/e009/results/NEW_EVAL --arm filtered --seed 29017 --adapter /workspace/e009/results/filtered-29017/adapter
python code/chat.py --root /workspace/e009 --seed 29017 --interactive --max-new-tokens 384
```

The root must contain the four pinned base-model weight files. `bootstrap.py`
downloads and verifies them. Model files total about 15.2 GB; local metadata
alone do not constitute a downloaded model. Training/evaluation output folders
refuse overwrite. Reproducing these commands consumes GPU resources; provisioning
and budget control are separate from the model scripts.

Inference loads the base, merges a verified LoRA adapter in memory, and uses
the same Human/Assistant format as training with no system persona. Evaluation
is greedy; exploratory chat uses a saved sampling seed, temperature 0.8 and
top-p 0.9. Raw continuations and visible role-trimmed responses are both saved.
Fixed evaluation uses a 192-token response cap; the ten exploratory responses
used 384. The 4,096-token context cap is explicit. There is
no response-ranking classifier or rewrite pass.

For the file queue, upload under a name such as `pending-01.upload`, then rename
to `request-01.json` on the remote filesystem. Do not stream a transfer directly
into a watched request filename: the worker can see partially written JSON.
The first exploratory worker hit that race before response 03 and was restarted
with the same request/seed. Both logs are preserved. Terminal `--interactive`
does not need the file-transfer queue.

Large checkpoints, third-party dataset text and complete run artifacts remain
outside Git. No dataset/model publication or paid always-on inference service
is included. The final deliverable is the research paper and inspectable local
artifacts, followed by termination of our temporary RunPod resources.

## Completed artifacts and analysis

The immutable download is `D:/Interiority-V1/cloud/study-artifacts.tar.gz`,
SHA-256 `65dd0a83179ea78efa326fd3734856737ffbf12bbe22107e806ba68e670b42f0`.
`retrieval-verified.json` records all 213 files / 3,025,303,939 uncompressed
bytes checked. The extraction is `D:/Interiority-V1/cloud/retrieved`; subsequent
local annotation files are in its `analysis` directory and are separately hashed.
The six final adapters are in `results/{arm}-{seed}/adapter`; both epoch
checkpoints remain alongside each. Raw outputs include cap flags and token IDs.

Regenerate the summaries and readable transcripts without GPU work:

```powershell
.venv/Scripts/python.exe experiments/E009-interiority-ablation/cloud/analyze.py summarize --root D:/Interiority-V1/cloud/retrieved
.venv/Scripts/python.exe experiments/E009-interiority-ablation/cloud/export_results.py --root D:/Interiority-V1/cloud/retrieved --output experiments/E009-interiority-ablation/cloud/report
.venv/Scripts/python.exe experiments/E009-interiority-ablation/cloud/figures.py --root D:/Interiority-V1/cloud/retrieved --output experiments/E009-interiority-ablation/cloud/report/figures
```

All 294 manual ratings must be present; the summarizer refuses an incomplete
packet. `record_ratings.py` records explicit human/agent judgments and never
derives a stance automatically from keywords. The paper reports single-agent
partial masking, two seeds, response caps, residual data defects and no measured
filter-specific persistent identity. No second study is started automatically.
