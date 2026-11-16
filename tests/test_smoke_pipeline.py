"""CPU smoke test of the tune -> final-run pipeline on a tiny random VTAB-1k-style dataset.

Needs the real IN-21k ViT-B/16 weights. The test looks for them in
$PETL_TEST_WEIGHTS (path to ViT-B_16.npz or ViT-B_16_in21k.npz), then in
pretrained_weights/ViT-B_16_in21k.npz, and is skipped if neither exists.
Each run is 1 epoch with batch size 4; the accuracies are meaningless.
"""
import csv
import json
import os
import subprocess
import sys

import pytest
import yaml

from conftest import REPO, EXPECTED_TRAINABLE_NO_HEAD
from fake_vtab import make_fake_vtab

TASK = 'eurosat'


def find_weights():
    candidates = [os.environ.get('PETL_TEST_WEIGHTS'),
                  os.path.join(REPO, 'pretrained_weights', 'ViT-B_16_in21k.npz')]
    for path in candidates:
        if path and os.path.isfile(path):
            return os.path.abspath(path)
    return None


WEIGHTS = find_weights()
pytestmark = pytest.mark.skipif(WEIGHTS is None, reason='IN-21k ViT-B_16 npz not found (set PETL_TEST_WEIGHTS)')

METHOD_YAML = {
    'lora_r8': {'method_name': 'lora', 'lora_bottleneck': 8},
    'adaptformer_d16': {'method_name': 'adaptformer', 'ft_mlp_module': 'adapter', 'ft_mlp_mode': 'parallel',
                        'ft_mlp_ln': 'before', 'adapter_init': 'lora_kaiming', 'adapter_bottleneck': 16,
                        'adapter_scaler': 0.1},
    'vpt_deep_p32': {'method_name': 'vpt_deep', 'vpt_mode': 'deep', 'vpt_num': 32},
    'vpt_shallow_p10': {'method_name': 'vpt_shallow', 'vpt_mode': 'shallow', 'vpt_num': 10},
    'full': {'method_name': 'full', 'full': True},
}

RUN_STAT_FIELDS = ['train_time_s', 'epoch_times_s', 'epoch_time_mean_s', 'peak_memory_allocated_mib',
                   'peak_memory_reserved_mib', 'device', 'gpu_name', 'torch_version', 'timm_version',
                   'trainable_params', 'trainable_params_no_head']


@pytest.fixture(scope='module')
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp('smoke')
    data = root / 'data'
    make_fake_vtab(str(data), TASK)
    weights_dir = root / 'weights'
    weights_dir.mkdir()
    os.symlink(WEIGHTS, weights_dir / 'ViT-B_16_in21k.npz')
    with open(os.path.join(REPO, 'experiment/config/default_vtab_processed.yml')) as f:
        default = yaml.safe_load(f)
    default.update(epoch=1, warmup_epoch=0, batch_size=4, test_batch_size=4, data_path=str(data),
                   experiment_name='smoke')
    default_yml = root / 'default.yml'
    default_yml.write_text(yaml.safe_dump(default))
    return root, default_yml, weights_dir


def run_tune(workspace, name, method, lrs, seeds):
    root, default_yml, weights_dir = workspace
    method_yml = root / f'{name}_method.yml'
    method_yml.write_text(yaml.safe_dump(method))
    lrwd_yml = root / f'{name}_lrwd.yml'
    lrwd_yml.write_text(yaml.safe_dump({'lr': lrs, 'wd': [0.0001]}))
    tune_root = root / f'tune_{name}'
    cmd = [sys.executable, 'main_tune.py', '--data', f'processed_vtab-{TASK}', '--default', str(default_yml),
           '--tune', str(method_yml), '--lrwd', str(lrwd_yml), '--tune_root', str(tune_root),
           '--pretrained_dir', str(weights_dir), '--seeds', *[str(s) for s in seeds]]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    out = tune_root / 'smoke' / 'processed_vtab' / TASK / method['method_name']
    return out


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize('name', list(METHOD_YAML))
def test_one_epoch_per_method(workspace, name):
    out = run_tune(workspace, name, METHOD_YAML[name], lrs=[0.001], seeds=[0])
    result = json.loads((out / 'final_result.json').read_text())
    assert result['trainable_params_no_head'] == EXPECTED_TRAINABLE_NO_HEAD[name]
    assert result['inserted_parameters'] == EXPECTED_TRAINABLE_NO_HEAD[name]
    assert result['trainable_params'] == EXPECTED_TRAINABLE_NO_HEAD[name] + 768 * 10 + 10
    assert result['device'] == 'cpu' and result['gpu_name'] is None
    assert result['avg_peak_memory_allocated_mib'] is None  # no CUDA on CPU
    assert result['timm_version'] == '0.9.12'
    assert result['normalized'] is False
    assert result['seeds'] == [0] and len(result['per_seed']) == 1
    seed_row = result['per_seed'][0]
    for field in RUN_STAT_FIELDS + ['test_top1', 'test_eval_time_s']:
        assert field in seed_row, field
    assert len(seed_row['epoch_times_s']) == 1 and seed_row['train_time_s'] > 0

    tune_rows = read_csv(out / 'tune_summary.csv')
    assert len(tune_rows) == 1
    for field in RUN_STAT_FIELDS + ['val_top1', 'val_best_top1', 'random_seed', 'normalized']:
        assert field in tune_rows[0], field


def test_tune_then_final_with_two_seeds(workspace):
    out = run_tune(workspace, 'pipeline', METHOD_YAML['lora_r8'], lrs=[0.001, 0.01], seeds=[0, 1])
    tune_rows = read_csv(out / 'tune_summary.csv')
    assert len(tune_rows) == 2

    result = json.loads((out / 'final_result.json').read_text())
    assert result['seeds'] == [0, 1]
    assert [row['seed'] for row in result['per_seed']] == [0, 1]
    assert len(result['accs']) == 2
    assert result['avg_acc'] == pytest.approx(sum(result['accs']) / 2)
    assert result['std_acc'] is not None and result['std_train_time_s'] is not None
    assert result['best_tune']['lr'] in (0.001, 0.01)
    assert 'avg_best_acc' not in result  # no best-over-epochs test number any more

    summary = read_csv(out / 'final_summary.csv')
    assert [row['run_idx'] for row in summary] == ['0', '1', 'mean', 'std']
    assert float(summary[2]['test_top1']) == pytest.approx(result['avg_acc'])
    assert summary[3]['train_time_s'] != ''


def test_main_py_single_run_writes_stats(workspace):
    root, default_yml, weights_dir = workspace
    out_root = root / 'main_out'
    cmd = [sys.executable, 'main.py', '--data', f'processed_vtab-{TASK}', '--data_path', str(root / 'data'),
           '--epoch', '1', '--warmup_epoch', '0', '--batch_size', '4', '--test_batch_size', '4',
           '--lora_bottleneck', '8', '--output_root', str(out_root), '--pretrained_dir', str(weights_dir)]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    results = list(out_root.rglob('run_result.json'))
    assert len(results) == 1
    result = json.loads(results[0].read_text())
    assert result['trainable_params_no_head'] == EXPECTED_TRAINABLE_NO_HEAD['lora_r8']
    assert result['normalized'] is False
    for field in RUN_STAT_FIELDS:
        assert field in result, field


METRIC_FIELDS = ['train_loss', 'train_top1', 'train_top5', 'val_loss', 'val_top1', 'val_top5',
                 'val_best_loss', 'val_best_top1', 'val_best_top5']


def test_tuning_result_does_not_depend_on_grid_order(workspace):
    """Each grid point is re-seeded with the tuning seed, so lr=0.001 gives the same
    numbers whether it runs first or second in the grid."""
    first = run_tune(workspace, 'order_first', METHOD_YAML['lora_r8'], lrs=[0.001, 0.01], seeds=[0])
    second = run_tune(workspace, 'order_second', METHOD_YAML['lora_r8'], lrs=[0.01, 0.001], seeds=[0])
    rows_first = read_csv(first / 'tune_summary.csv')
    rows_second = read_csv(second / 'tune_summary.csv')
    assert [eval(r['param_set'])['lr'] for r in rows_first] == [0.001, 0.01]
    assert [eval(r['param_set'])['lr'] for r in rows_second] == [0.01, 0.001]
    run_first, run_second = rows_first[0], rows_second[1]
    for field in METRIC_FIELDS:
        assert run_first[field] == run_second[field], field
    for row in rows_first + rows_second:
        assert row['random_seed'] == '42'  # the tuning seed from default_vtab_processed.yml


def test_sys843_per_task_config_end_to_end(workspace):
    """--sys843 picks the SYS843 method config and resolves its per-task prompt length
    (VPT-Shallow on EuroSAT: p=50 in the VPT paper); default and grid are shrunk for speed."""
    root, _, weights_dir = workspace
    with open(os.path.join(REPO, 'experiment/config/sys843/default_vtab_processed.yml')) as f:
        default = yaml.safe_load(f)
    default.update(epoch=1, warmup_epoch=0, batch_size=4, test_batch_size=4, data_path=str(root / 'data'))
    default_yml = root / 'sys843_default.yml'
    default_yml.write_text(yaml.safe_dump(default))
    lrwd_yml = root / 'sys843_lrwd.yml'
    lrwd_yml.write_text(yaml.safe_dump({'lr': [0.001], 'wd': [0.0]}))
    tune_root = root / 'tune_sys843'
    cmd = [sys.executable, 'main_tune.py', '--sys843', 'vpt_shallow_paper', '--data', f'processed_vtab-{TASK}',
           '--default', str(default_yml), '--lrwd', str(lrwd_yml), '--tune_root', str(tune_root),
           '--pretrained_dir', str(weights_dir), '--seeds', '0']
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    out = tune_root / 'sys843_vtab' / 'processed_vtab' / TASK / 'vpt_shallow_paper'
    result = json.loads((out / 'final_result.json').read_text())
    assert result['trainable_params_no_head'] == 50 * 768
    assert result['best_tune'] == {'lr': 0.001, 'wd': 0.0}
