# E010: learned retention under a real KV-cache budget

Read [the completed report](report.md). The fixed pilot and post-primary output-cap diagnostic are complete. Exposure did not give a consistent accuracy benefit; an answer-format confound materially limits baseline interpretation.

Use an environment with PyTorch (CUDA), Transformers 4.57.6, NumPy, and local Qwen2.5-1.5B-Instruct weights. Set `model_path` in a copied configuration for a fresh reproduction; doing so changes the configuration hash and should produce a separate run, not overwrite this one.

```powershell
python -m unittest discover -s experiments/E010-kv-policy-compression -p test_kvpilot.py -v
python experiments/E010-kv-policy-compression/kvpilot.py smoke --output experiments/E010-kv-policy-compression/results/smoke-new
python experiments/E010-kv-policy-compression/kvpilot.py run --output experiments/E010-kv-policy-compression/results/run-new
python experiments/E010-kv-policy-compression/analyze.py experiments/E010-kv-policy-compression/results/run-new
```

The diagnostic is intentionally pinned to `run-01` and its first policy seed. For a new diagnostic, copy the script and record the new run path before outcomes. `diagnostic-protocol.md` clearly separates this exploratory follow-up from the prospective primary run. No script provisions infrastructure or executes generated code.

The full experiment fits on the local 8 GiB GPU. This does not mean that the compression technique enables a larger base model to fit; weights are unchanged, and total memory savings here are small.
