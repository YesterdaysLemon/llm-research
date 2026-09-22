#!/usr/bin/env bash
set -euo pipefail
cd /workspace/e009
python -m pip install --break-system-packages --disable-pip-version-check -r code/requirements.txt
python -m pip list --format=json > /workspace/e009/runtime-packages-parallel.json
setsid env CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false python -u code/worker.py --root /workspace/e009 --arm filtered --deadline 1790051292 > worker-filtered.log 2>&1 < /dev/null &
setsid env CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false python -u code/worker.py --root /workspace/e009 --arm random_control --deadline 1790051292 > worker-random_control.log 2>&1 < /dev/null &
printf 'workers started\n'
