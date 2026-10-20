#!/usr/bin/env bash
# Run from the project root in a Python 3.12 Linux environment.
set -euo pipefail
project_python="${PROJECT_PYTHON:-python3.12}"
project_venv="${PROJECT_VENV:-.venv-autodl}"
torch_channel="${TORCH_CHANNEL:-cu128}"
case "$torch_channel" in cu126|cu128|cu130) ;; *) echo 'TORCH_CHANNEL must be cu126, cu128, or cu130' >&2; exit 2;; esac
test -f requirements-lock.txt
"$project_python" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 required"'
if test -e "$project_venv"; then
  echo "Environment already exists: $project_venv. Reuse it or choose a new PROJECT_VENV." >&2
  exit 2
fi
"$project_python" -m venv "$project_venv"
"$project_venv/bin/python" -m pip install 'torch==2.11.0' --index-url "https://download.pytorch.org/whl/$torch_channel"
"$project_venv/bin/python" -m pip install -r requirements-lock.txt
"$project_venv/bin/python" -m pip check
"$project_venv/bin/python" -m experiments.check_environment --device cuda --output results/pretrain/autodl_environment.json
"$project_venv/bin/python" -m pip freeze > results/pretrain/autodl_freeze.txt
echo "Setup checked. No training has started. Activate with: source $project_venv/bin/activate"
