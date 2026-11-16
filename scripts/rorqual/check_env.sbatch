#!/bin/bash
# Environment check on one full Rorqual H100: GPU, library versions, data and weights in place,
# then the repository's test suite (the CPU smoke tests, here running on the GPU).
#
# Submit after setup_env.sh and the data/weights preparation (the account comes from env.sh):
#     source ~/sys843-vit-peft/scripts/rorqual/env.sh
#     sbatch "$SYS843_REPO/scripts/rorqual/check_env.sbatch"
# Time: the suite takes about 3 min on a laptop CPU; 30 min covers staging, a slow start and
# the GPU initialisation with a wide margin. 16 cores is one H100's share of a Rorqual node.
#SBATCH --job-name=sys843-check-env
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
set -euo pipefail

# shellcheck disable=SC1091
source "${SYS843_REPO:-$SLURM_SUBMIT_DIR}/scripts/rorqual/env.sh"
sys843_activate

cd "$SYS843_REPO"
nvidia-smi
python - <<'PY'
import platform, torch, torchvision, timm
print("python", platform.python_version(), "| torch", torch.__version__, "| torchvision", torchvision.__version__,
      "| timm", timm.__version__, "| CUDA", torch.version.cuda, "| cuDNN", torch.backends.cudnn.version())
assert torch.cuda.is_available(), "no CUDA device visible"
print("GPU:", torch.cuda.get_device_name(0), "|", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), "GiB")
assert timm.__version__ == "0.9.12", "timm must be 0.9.12"
PY
git log -1 --oneline

echo "weights: $SYS843_WEIGHTS"
( cd "$SYS843_WEIGHTS" && sha256sum --check ViT-B_16_in21k.npz.sha256 )
echo "data tarballs: $SYS843_TARS"
( cd "$SYS843_TARS" && sha256sum --check --quiet SHA256SUMS && echo "SHA256SUMS ok: $(wc -l < SHA256SUMS) tarballs" )
sys843_stage_task eurosat  # one task end to end: verify, extract, check the splits

PETL_TEST_WEIGHTS="$SYS843_WEIGHTS/ViT-B_16_in21k.npz" python -m pytest -q tests
echo "RESULT ok"
