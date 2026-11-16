#!/bin/bash
# Create the SYS843 virtual environment on Rorqual (Digital Research Alliance of Canada).
#
# Run once on a LOGIN node (compute nodes have no internet), from the clone:
#     bash scripts/rorqual/setup_env.sh
#
# Alliance convention: load a Python module, create a virtualenv with --no-download, and
# install the Alliance's prebuilt wheels with --no-index (the cluster's torch build targets its
# CUDA and the H100s). Login nodes reach PyPI, which is used only for what the wheelhouse lacks:
# timm must be exactly 0.9.12 (timm 1.x breaks this code), and dotwiz may be missing.
# The venv lives outside the repository (default ~/venvs/sys843-vit-peft, see env.sh).
# Check the Python module with `module avail python` and override it if needed:
#     PYTHON_MODULE=python/3.12 bash scripts/rorqual/setup_env.sh
set -euo pipefail

PYTHON_MODULE="${PYTHON_MODULE:-python/3.11}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/sys843-vit-peft}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TIMM_VERSION=0.9.12

set +u  # Lmod reads unset variables
module load StdEnv/2023 "$PYTHON_MODULE"
set -u

if [ ! -d "$VENV_DIR" ]; then
    mkdir -p "$(dirname "$VENV_DIR")"
    virtualenv --no-download "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

pip install --no-index --upgrade pip setuptools wheel
# Everything requirements.txt needs except timm and dotwiz, from the Alliance wheelhouse.
pip install --no-index torch torchvision numpy pillow pyyaml scikit-learn pandas tqdm \
    tabulate termcolor iopath pytest
# timm's own dependencies from the wheelhouse first, so PyPI only supplies timm itself.
pip install --no-index huggingface_hub safetensors || echo "note: huggingface_hub/safetensors not in the wheelhouse"

# timm 0.9.12: wheelhouse if it has that exact version (`avail_wheels timm --all-versions`),
# else PyPI with --no-deps (its dependencies are installed above).
if ! pip install --no-index "timm==$TIMM_VERSION"; then
    echo "timm==$TIMM_VERSION is not in the wheelhouse: installing it from PyPI"
    pip install --no-deps "timm==$TIMM_VERSION"
fi
# dotwiz (config objects): wheelhouse, else PyPI.
pip install --no-index dotwiz || pip install dotwiz

# Verify the imports the processed VTAB-1k path needs, and the timm pin.
(
    cd "$REPO_DIR"
    python - <<PY
import torch, torchvision, timm, dotwiz, sklearn, yaml, numpy, PIL
assert timm.__version__ == "$TIMM_VERSION", f"timm {timm.__version__}, need $TIMM_VERSION"
import main_tune, main  # the code imports without TensorFlow or CLIP
print("torch", torch.__version__, "| torchvision", torchvision.__version__, "| timm", timm.__version__,
      "| CUDA build", torch.version.cuda)
PY
)
echo "Virtualenv ready: $VENV_DIR"
echo "Next: source scripts/rorqual/env.sh, prepare data and weights (scripts/data), then"
echo "      sbatch \"\$SYS843_REPO/scripts/rorqual/check_env.sbatch\""
