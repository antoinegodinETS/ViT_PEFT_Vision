"""The fork must not ship result files that main_tune.py's skip-if-exists logic could reuse."""
import os
import subprocess

from conftest import REPO
from utils import global_var

# Files whose presence makes main_tune.py / main_collect_prediction.py / main_evaluate.py skip work.
RESULT_NAMES = ('tune_summary.csv', 'final_result.json', 'final_summary.csv', 'logits.npy', 'gt.npy', 'model.pt')


def test_no_result_files_tracked():
    tracked = subprocess.run(['git', 'ls-files'], cwd=REPO, capture_output=True, text=True, check=True).stdout.split()
    bundled = [p for p in tracked
               if os.path.basename(p) in RESULT_NAMES or os.path.basename(p).endswith('_result.json')
               or p.startswith('tune_output/') or p.startswith('runs/')]
    assert bundled == []


def test_default_tune_roots_are_fresh_and_ignored():
    for root in (global_var.TUNE_DIR, global_var.TUNE_DIR_TEST):
        if root.startswith(('./runs/', 'runs/')):  # skip when overridden by PETL_TUNE_DIR*
            ignored = subprocess.run(['git', 'check-ignore', '-q', os.path.join(root, 'x', 'tune_summary.csv')],
                                     cwd=REPO)
            assert ignored.returncode == 0, f'{root} must be gitignored'
    assert 'tune_output' not in os.path.normpath(global_var.TUNE_DIR).split(os.sep)
