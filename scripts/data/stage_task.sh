#!/usr/bin/env bash
# Job-side helper: extract one task's tarball into a local directory (e.g. $SLURM_TMPDIR)
# and print the resulting --data_path on stdout (everything else goes to stderr).
#
#   DATA_PATH=$(scripts/data/stage_task.sh <tar_dir> <task> <dest_dir>)
#   python main_tune.py --data processed_vtab-<task> --data_path "$DATA_PATH" ...
#
# <tar_dir> holds <task>.tar and SHA256SUMS from repack_vtab1k.py. The tarball is
# checked against SHA256SUMS (skip with STAGE_VERIFY=0), extracted to <dest_dir>/<task>/,
# and the split files are checked with check_vtab1k.py (skip with STAGE_CHECK=0; extra
# arguments for it in STAGE_CHECK_ARGS, e.g. "--counts train800=8 ..."). Python is
# $PYTHON, else python3. If <dest_dir>/<task>/.staged exists, the task is already there.
set -euo pipefail

if [ $# -ne 3 ]; then
    echo "usage: $0 <tar_dir> <task> <dest_dir>" >&2
    exit 2
fi
tar_dir=$1
task=$2
dest=$3
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python=${PYTHON:-python3}
tarball="$tar_dir/$task.tar"

if [ ! -f "$tarball" ]; then
    echo "ERROR: $tarball not found" >&2
    exit 1
fi
mkdir -p "$dest"
dest=$(cd "$dest" && pwd)

if [ -f "$dest/$task/.staged" ]; then
    echo "$task already staged in $dest" >&2
else
    if [ "${STAGE_VERIFY:-1}" != 0 ]; then
        echo "verifying $tarball against $tar_dir/SHA256SUMS ..." >&2
        line=$(grep -E "  $task\.tar\$" "$tar_dir/SHA256SUMS" || true)
        if [ -z "$line" ]; then
            echo "ERROR: $task.tar is not listed in $tar_dir/SHA256SUMS" >&2
            exit 1
        fi
        ( cd "$tar_dir" && printf '%s\n' "$line" | sha256sum --check --quiet - ) >&2
    fi
    echo "extracting $tarball -> $dest/$task" >&2
    rm -rf "${dest:?}/$task"
    tar -xf "$tarball" -C "$dest" --no-same-owner
    if [ ! -d "$dest/$task" ]; then
        echo "ERROR: $tarball did not contain $task/" >&2
        exit 1
    fi
    if [ "${STAGE_CHECK:-1}" != 0 ]; then
        # shellcheck disable=SC2086  # STAGE_CHECK_ARGS is a list of words
        "$python" "$here/check_vtab1k.py" "$dest" --tasks "$task" ${STAGE_CHECK_ARGS:-} >&2
    fi
    touch "$dest/$task/.staged"
fi
printf '%s\n' "$dest"
