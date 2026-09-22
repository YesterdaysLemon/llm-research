#!/usr/bin/env bash
set -euo pipefail
cd /workspace/e009
trap 'printf "%s\n" "$?" > /workspace/e009/launcher-exit-code.txt' EXIT
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
python -m pip install --break-system-packages --disable-pip-version-check -r code/requirements.txt
python -u code/bootstrap.py --root /workspace/e009
python -u code/orchestrate.py --root /workspace/e009 --deadline "$1"
