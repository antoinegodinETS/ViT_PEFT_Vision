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
# GPU size: a full H100 for every final job, so the reported training time, per-epoch time and
# peak memory (recorded per seed in final_result.json) are comparable across methods. These
# runs are the only source of timing numbers for the report. 16 CPU cores is an H100's share of
# a Rorqual node (15 loader workers, enough to decode the largest test sets quickly).
#
# Time: per seed, 100 epochs x 960 images plus one test pass (up to 73,728 images for dSprites).
# The per-seed time on a full H100 is not measured yet (pilot final timings pending); the tuning
# pilot measured VPT-Shallow p=200 at 1.82x LoRA per epoch on a 3g.40gb slice. The Alliance asks
# for jobs of about 1 h or more, so give each job 4-6 tasks; the 4 h limit and the tasks per job
# are to be revisited with the pilot final timings.
# Resumable: on Slurm's USR1 warning the job resubmits itself (same tasks); tasks with a
# final_result.json are skipped, and an interrupted task reruns its 3 seeds.
#SBATCH --job-name=sys843-final
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
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
