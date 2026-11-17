#!/bin/bash
# Hyperparameter search for ONE (SYS843 method config, task): every grid point of the config's
# lr x wd grid (20 points for every config) on train800, selected on the final-epoch val200
# accuracy (near-ties broken by the rule in experiment/selection.py). No final run here: that is final.sbatch, on a full H100.
# Output: $SYS843_RUNS/sys843_vtab/processed_vtab/<task>/<method>/tune_summary.csv.
#
# Submit from a login node (the account comes from env.sh), one job per (method, task):
#     source ~/sys843-vit-peft/scripts/rorqual/env.sh
#     sbatch --job-name=sys843-tune-lora_r8-eurosat --export=ALL,SYS843_METHOD=lora_r8,SYS843_TASK=eurosat \
#         "$SYS843_REPO/scripts/rorqual/tune.sbatch"
#
# GPU size: a 3g.40gb MIG slice by default, NOT the 1g.10gb slice. The code trains in fp32 at
# batch 64; the VPT paper (arXiv 2203.12119, Table 12, A100, batch 64) measures 9.8-10.3 GB
# peak training memory for adapters and VPT with p=1, 11.7 GB for full fine-tuning and 25.8 GB
# for VPT with p=200, and a 1g.10gb slice has 10 GB. 3g.40gb fits every config. Search timings
# are not reported (all time and memory numbers in the report come from full-H100 final runs),
# so the slice size only has to keep the job within its limit. After the pilot, a config whose
# tune_summary.csv shows peak_memory_reserved_mib well under 9,000 can go on the small slice:
#     sbatch --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1 --cpus-per-task=2 ... tune.sbatch
# 6 CPU cores is about 3/7 of an H100's 16 cores per GPU on a Rorqual node (5 loader workers).
#
# Time: a grid point is 100 epochs x 768 images plus one val200 pass, about 4 PFLOP for
# ViT-B/16 at 197 tokens, roughly 5 min on 3/7 of an H100 in fp32; VPT with p=200 doubles the
# tokens (about 2.2x the time). Estimates: 20 points ~2 h, the p=200 VPT tasks ~4.5 h. The 12 h limit is about 2.5x the slowest estimate; the pilot replaces
# these guesses (train_time_s in tune_summary.csv).
# Resumable: Slurm sends USR1 5 minutes before the limit; the job stops main_tune.py and
# resubmits itself to start after this job ends (up to SYS843_MAX_CHUNKS jobs). Finished grid
# points are skipped; the interrupted one starts over. Resubmitting by hand is also safe, but
# never run two jobs on the same (method, task) at once: both would append to tune_summary.csv.
#SBATCH --job-name=sys843-tune
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=%x-%j.out
set -euo pipefail

# shellcheck disable=SC1091
source "${SYS843_REPO:-$SLURM_SUBMIT_DIR}/scripts/rorqual/env.sh"
sys843_check_method
if [ -z "${SYS843_TASK:-}" ]; then
    echo "error: set SYS843_TASK (sbatch --export=ALL,SYS843_METHOD=...,SYS843_TASK=eurosat ...)" >&2
    exit 2
fi
sys843_activate

cd "$SYS843_REPO"
nvidia-smi
echo "method: $SYS843_METHOD, task: $SYS843_TASK, runs root: $SYS843_RUNS, chunk ${SYS843_CHUNK:-1}"
sys843_run_resumable "$SYS843_REPO/scripts/rorqual/tune.sbatch" \
    bash "$SYS843_REPO/scripts/rorqual/run_tasks.sh" tune "$SYS843_METHOD" "$SYS843_TASK"
