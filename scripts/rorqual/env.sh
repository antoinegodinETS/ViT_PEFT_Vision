# Shared Rorqual settings for the SYS843 job scripts (pattern of the SYS819 project's scripts).
#
# This file is the ONE place the allocation account is set (SBATCH_ACCOUNT, read by sbatch).
# Source it on a login node before submitting, then use plain sbatch:
#     source ~/sys843-vit-peft/scripts/rorqual/env.sh
#     sbatch "$SYS843_REPO/scripts/rorqual/check_env.sbatch"
# Jobs inherit these variables (sbatch exports the environment) and source this file again
# for the helper functions. Every value can be overridden by exporting it first.

export SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-def-chdesa}"
export SALLOC_ACCOUNT="${SALLOC_ACCOUNT:-$SBATCH_ACCOUNT}"

if [ -z "${SYS843_REPO:-}" ]; then
    SYS843_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
export SYS843_REPO
export PYTHON_MODULE="${PYTHON_MODULE:-python/3.11}"            # must match setup_env.sh
export VENV_DIR="${VENV_DIR:-$HOME/venvs/sys843-vit-peft}"
# Project space persists; scratch (~/links/scratch) is purged after 60 days.
export SYS843_PROJECT="${SYS843_PROJECT:-$HOME/links/projects/$SBATCH_ACCOUNT/$USER}"
export SYS843_STORE="${SYS843_STORE:-$SYS843_PROJECT/sys843}"
export SYS843_DATA_STORE="${SYS843_DATA_STORE:-$SYS843_STORE/data}"        # vtab-1k.zip (+ .sha256)
export SYS843_TARS="${SYS843_TARS:-$SYS843_DATA_STORE/vtab-1k-tars}"      # <task>.tar, SHA256SUMS
export SYS843_WEIGHTS="${SYS843_WEIGHTS:-$SYS843_STORE/weights}"          # ViT-B_16_in21k.npz
export SYS843_RUNS="${SYS843_RUNS:-$SYS843_STORE/runs}"                   # tuning + final results
# Read by the code (utils/global_var.py) when --pretrained_dir / --tune_root are not given.
export PETL_PRETRAINED_DIR="$SYS843_WEIGHTS"
export PETL_TUNE_DIR="$SYS843_RUNS"

# Load the Python module and activate the project virtualenv.
sys843_activate() {
    set +u  # Lmod reads unset variables
    module load StdEnv/2023 "$PYTHON_MODULE"
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    set -u
}

# Extract one task's tarball to the node-local disk and set SYS843_DATA_PATH (the --data_path).
#     sys843_stage_task eurosat
sys843_stage_task() {
    local dest="${SLURM_TMPDIR:?sys843_stage_task runs inside a Slurm job}/vtab-1k"
    SYS843_DATA_PATH="$(PYTHON=python "$SYS843_REPO/scripts/data/stage_task.sh" "$SYS843_TARS" "$1" "$dest")"
    export SYS843_DATA_PATH
    echo "task $1 staged in $SYS843_DATA_PATH: $(du -sh "$SYS843_DATA_PATH/$1" | cut -f1)"
}

# Run a command; on Slurm's USR1 warning (#SBATCH --signal=B:USR1@<s>), stop it and resubmit
# this job script to start after this job ends, so the next chunk resumes where this one stopped
# (main_tune.py skips finished grid points and finished final runs; the interrupted grid point
# or final run starts over).
#     sys843_run_resumable <this job script> <command> [args...]
# SYS843_MAX_CHUNKS (default 5) caps the resubmissions; SYS843_CHUNK counts them.
sys843_run_resumable() {
    local script="$1"
    shift
    local chunk="${SYS843_CHUNK:-1}" status pid interrupted=0
    "$@" &
    pid=$!
    trap 'echo "USR1 from Slurm: stopping"; interrupted=1; kill -TERM "$pid" 2>/dev/null' USR1
    set +e
    wait "$pid"
    status=$?
    while kill -0 "$pid" 2>/dev/null; do  # wait returns early when the trap fires
        wait "$pid"
        status=$?
    done
    set -e
    trap - USR1
    if [ "$interrupted" = 1 ]; then
        if [ "$chunk" -lt "${SYS843_MAX_CHUNKS:-5}" ]; then
            echo "time limit near: resubmitting $script as chunk $((chunk + 1)), after job $SLURM_JOB_ID"
            sbatch --dependency="afterany:$SLURM_JOB_ID" --job-name="$SLURM_JOB_NAME" \
                --export="ALL,SYS843_CHUNK=$((chunk + 1))" "$script"
        else
            echo "time limit near and SYS843_MAX_CHUNKS reached: resubmit $script by hand"
        fi
        return 3
    fi
    return "$status"
}

# Check SYS843_METHOD against experiment/config/sys843/methods.yml.
sys843_check_method() {
    if [ -z "${SYS843_METHOD:-}" ] || ! grep -q "^${SYS843_METHOD}:" "$SYS843_REPO/experiment/config/sys843/methods.yml"; then
        echo "error: set SYS843_METHOD to one of:" \
            "$(grep -oE '^[a-z0-9_]+:' "$SYS843_REPO/experiment/config/sys843/methods.yml" | tr -d : | tr '\n' ' ')" >&2
        return 2
    fi
}

# The 3 pilot tasks: the only tasks a sanity-check config (setting `check` in methods.yml,
# e.g. vpt_deep_p32_sgd) runs on.
export SYS843_PILOT_TASKS="oxford_flowers102 eurosat clevr_count"

# Names of the configs of the 19-task sweep: every methods.yml entry whose setting is not `check`.
#     METHODS="$(sys843_sweep_methods)"
sys843_sweep_methods() {
    grep -E '^[a-z0-9_]+:' "$SYS843_REPO/experiment/config/sys843/methods.yml" \
        | grep -vE 'setting: *\[[^]]*\bcheck\b' | cut -d: -f1 | tr '\n' ' ' | sed 's/ $//'
}

# Success if config $1 is a sanity check (setting `check` in methods.yml).
sys843_is_check_method() {
    grep -qE "^$1:.*setting: *\[[^]]*\bcheck\b" "$SYS843_REPO/experiment/config/sys843/methods.yml"
}
