#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

conda create -n python3.12-jax-gpu python=3.12 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate python3.12-jax-gpu

conda install -c nvidia cuda-toolkit -y

pip install -r "$REPO_ROOT/requirements.txt"

# CPU-only alternative:
# pip install --upgrade jax
