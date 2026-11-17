#!/bin/bash
# Run one stage of main_tune.py for one SYS843 method config over a list of tasks, inside a job:
# stage each task's data to $SLURM_TMPDIR, then tune (--stage tune) or run the final seeds
# (--stage final). Results go to $SYS843_RUNS. Called by tune.sbatch and final.sbatch.
#     bash scripts/rorqual/run_tasks.sh <tune|final> <method> <task> [<task> ...]
# A final run whose final_result.json exists is skipped without staging its data; main_tune.py
# itself skips tuned grid points. On SIGTERM (sent by sys843_run_resumable at Slurm's USR1
# warning) the running main_tune.py is stopped and the script exits.
# A sanity-check config (setting `check` in methods.yml, e.g. vpt_deep_p32_sgd) runs only on the
# pilot tasks ($SYS843_PILOT_TASKS); other tasks are skipped unless SYS843_CHECK_ANY_TASK=1.
set -euo pipefail

stage="$1"
method="$2"
shift 2
# shellcheck disable=SC1091
source "$SYS843_REPO/scripts/rorqual/env.sh"
cd "$SYS843_REPO"

child=""
trap 'echo "SIGTERM: stopping main_tune.py"; [ -z "$child" ] || kill -TERM "$child" 2>/dev/null; exit 143' TERM

workers=$(( ${SLURM_CPUS_PER_TASK:-2} - 1 ))
failed=0
for task in "$@"; do
    if sys843_is_check_method "$method" && [ "${SYS843_CHECK_ANY_TASK:-0}" != 1 ] \
        && ! [[ " $SYS843_PILOT_TASKS " == *" $task "* ]]; then
        echo "== $method / $task: check config, not a pilot task ($SYS843_PILOT_TASKS), skipping" \
            "(set SYS843_CHECK_ANY_TASK=1 to run it anyway)"
        continue
    fi
    out="$SYS843_RUNS/sys843_vtab/processed_vtab/$task/$method"
    if [ "$stage" = final ] && [ -f "$out/final_result.json" ]; then
        echo "== $method / $task: final_result.json exists, skipping"
        continue
    fi
    echo "== $method / $task: stage $stage, $(date -Is)"
    sys843_stage_task "$task"
    python main_tune.py --sys843 "$method" --data "processed_vtab-$task" --stage "$stage" \
        --data_path "$SYS843_DATA_PATH" --pretrained_dir "$SYS843_WEIGHTS" --tune_root "$SYS843_RUNS" \
        --num_workers "$workers" &
    child=$!
    status=0
    wait "$child" || status=$?
    child=""
    if [ "$status" -ne 0 ]; then
        echo "== $method / $task: main_tune.py failed with status $status" >&2
        failed=1
    fi
    rm -rf "${SLURM_TMPDIR:?}/vtab-1k/$task"  # free the node-local disk for the next task
done
exit "$failed"
