#!/bin/bash
# Final runs for ONE SYS843 method config over one or more tasks: for each task, the best grid
# point of its finished search is retrained on train800val200 with seeds 0, 1 and 2, and the
# test set is evaluated once per seed after the last epoch. Fails for a task whose search is
# incomplete. Output: $SYS843_RUNS/sys843_vtab/processed_vtab/<task>/<method>/final_result.json
# and final_summary.csv (per-seed rows, mean and std).
#
# Submit from a login node once the tuning jobs are done (the account comes from env.sh);
# SYS843_TASKS is a space-separated list (quote it), or SYS843_TASK for one task:
#     source ~/sys843-vit-peft/scripts/rorqual/env.sh
#     sbatch --job-name=sys843-final-lora_r8 \
#         --export=ALL,SYS843_METHOD=lora_r8,SYS843_TASKS="eurosat dtd kitti caltech101" \
#         "$SYS843_REPO/scripts/rorqual/final.sbatch"
#
# GPU size: a 3g.40gb MIG slice, the same slice type as tuning, for every final job, so the
# reported training time, per-epoch time and peak memory (recorded per seed in final_result.json)
# are comparable across methods: every method runs on identical hardware, and the report gives
# slice times. These runs are the only source of timing numbers for the report. Full-H100 jobs
# waited days on Rorqual (ReqNodeNotAvail) while 3g.40gb slices start within minutes. The
# resources are #SBATCH lines, not sbatch flags, so a self-resubmitted chunk stays on the slice.
# 6 CPU cores is about 3/7 of an H100's 16 cores per GPU on a Rorqual node (5 loader workers).
# Do not override the GPU for some methods only (e.g. --gpus=h100:1 --cpus-per-task=16
# --mem=64G for a full H100): the timings would no longer be comparable.
#
# Time: per seed, 100 epochs x 15 steps (train800val200, drop_last), about 1.25x a tuning
# epoch on the same slice, i.e. roughly 7.5-15 min per seed, plus one test pass (up to 73,728
# images for dSprites). The per-seed time is not measured yet. The 3 h limit covers the 3 pilot
# tasks of the slowest config (VPT p=200, up to about 44 min per task for 3 seeds) with margin.
# Tasks per job and --time for the sweep are set once the pilot final timings are in.
# Resumable: on Slurm's USR1 warning the job resubmits itself (same tasks); tasks with a
# final_result.json are skipped, and an interrupted task reruns its 3 seeds.
#SBATCH --job-name=sys843-final
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=%x-%j.out
set -euo pipefail

# shellcheck disable=SC1091
source "${SYS843_REPO:-$SLURM_SUBMIT_DIR}/scripts/rorqual/env.sh"
sys843_check_method
tasks="${SYS843_TASKS:-${SYS843_TASK:-}}"
if [ -z "$tasks" ]; then
    echo "error: set SYS843_TASKS=\"task1 task2 ...\" or SYS843_TASK (sbatch --export=ALL,...)" >&2
    exit 2
fi
sys843_activate

cd "$SYS843_REPO"
nvidia-smi
echo "method: $SYS843_METHOD, tasks: $tasks, runs root: $SYS843_RUNS, chunk ${SYS843_CHUNK:-1}"
# shellcheck disable=SC2086  # $tasks is a list of task names
sys843_run_resumable "$SYS843_REPO/scripts/rorqual/final.sbatch" \
    bash "$SYS843_REPO/scripts/rorqual/run_tasks.sh" final "$SYS843_METHOD" $tasks
