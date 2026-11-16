#!/bin/bash
# Create the SYS843 virtual environment on Rorqual (Digital Research Alliance of Canada).
#
# Run on a LOGIN node (compute nodes have no internet), from the clone:
#     bash scripts/rorqual/setup_env.sh
# Safe to rerun: an existing venv is reused and installed packages are left as they are.
#
# Alliance convention: load a Python module, create a virtualenv with --no-download, and
# install the Alliance's prebuilt wheels with --no-index (the cluster's torch build targets its
# CUDA and the H100s). Login nodes reach PyPI, which is used only for what the wheelhouse lacks,
# and only for prebuilt wheels (--only-binary=:all:): login nodes have no Rust toolchain and a
# source build would fail or pull in the wrong torch.
#   Wheelhouse: torch torchvision numpy pillow pyyaml scikit-learn pandas tqdm tabulate
#               termcolor iopath (with portalocker) pytest, and normally huggingface_hub and
#               safetensors.
#   PyPI:       timm==0.9.12 (the wheelhouse has 0.9.2/0.9.7/0.9.16/1.x, and timm 1.x breaks
#               this code), as a pure-Python wheel with --no-deps; huggingface_hub/safetensors
#               only if the wheelhouse lacks them.
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

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    mkdir -p "$(dirname "$VENV_DIR")"
    virtualenv --no-download "$VENV_DIR"
else
    echo "Reusing the existing virtualenv $VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Wheelhouse first; else a prebuilt wheel from PyPI (never a source build).
wheelhouse_or_pypi() {
    if ! pip install --no-index "$@"; then
        echo "$* not in the wheelhouse: installing a prebuilt wheel from PyPI"
        pip install --only-binary=:all: "$@"
    fi
}

pip install --no-index --upgrade pip setuptools wheel
# Everything requirements.txt needs except timm, from the Alliance wheelhouse.
pip install --no-index torch torchvision numpy pillow pyyaml scikit-learn pandas tqdm \
    tabulate termcolor iopath pytest
# timm's own dependencies, so PyPI only supplies timm itself.
wheelhouse_or_pypi huggingface_hub
wheelhouse_or_pypi safetensors

# timm 0.9.12: wheelhouse if it has that exact version (`avail_wheels timm --all-versions`),
# else PyPI with --no-deps (its dependencies are installed above). A rerun finds it installed.
if ! pip install --no-index "timm==$TIMM_VERSION"; then
    echo "timm==$TIMM_VERSION is not in the wheelhouse: installing it from PyPI"
    pip install --no-deps --only-binary=:all: "timm==$TIMM_VERSION"
fi

# Verify the imports the processed VTAB-1k path needs, and the timm pin.
(
    cd "$REPO_DIR"
    python - <<PY
import torch, torchvision, timm, sklearn, yaml, numpy, PIL, iopath
assert timm.__version__ == "$TIMM_VERSION", f"timm {timm.__version__}, need $TIMM_VERSION"
import main_tune, main  # the code imports without TensorFlow or CLIP
print("torch", torch.__version__, "| torchvision", torchvision.__version__, "| timm", timm.__version__,
      "| CUDA build", torch.version.cuda)
PY
)
echo "Virtualenv ready: $VENV_DIR"
echo "Next: source scripts/rorqual/env.sh, prepare data and weights (scripts/data), then"
echo "      sbatch \"\$SYS843_REPO/scripts/rorqual/check_env.sbatch\""
