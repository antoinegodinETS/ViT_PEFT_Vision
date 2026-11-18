"""The SYS843 configs (experiment/config/sys843) load through main_tune.py and give the
expected trainable-parameter counts and grid sizes. Random init, no weights needed."""
import os
import subprocess

import pytest
from sklearn.model_selection import ParameterGrid

import main_tune
from conftest import REPO
from experiment import build_model
from model.vision_transformer import VisionTransformerPETL
from utils.global_var import VTAB_DATASETS

# Per-task prompt lengths copied by hand from the VPT paper (arXiv 2203.12119), Table 13,
# to cross-check the YAML files.
VPT_PAPER_P = {
    #                    shallow, deep
    'cifar':                (100, 10),
    'caltech101':           (5, 10),
    'dtd':                  (1, 10),
    'oxford_flowers102':    (200, 1),
    'oxford_iiit_pet':      (50, 1),
    'svhn':                 (200, 50),
    'sun397':               (1, 5),
    'patch_camelyon':       (5, 100),
    'eurosat':              (50, 100),
    'resisc45':             (50, 10),
    'diabetic_retinopathy': (10, 1),
    'clevr_count':          (100, 50),
    'clevr_dist':           (200, 200),
    'dmlab':                (100, 100),
    'kitti':                (100, 50),
    'dsprites_loc':         (100, 10),
    'dsprites_ori':         (100, 50),
    'smallnorb_azi':        (200, 200),
    'smallnorb_ele':        (200, 200),
}


def expected_no_head(name, task):
    fixed = {'lora_r8': 294_912, 'adaptformer_d16': 304_320, 'adaptformer_paper': 1_189_632,
             'vpt_deep_p32': 294_912, 'vpt_deep_p32_sgd': 294_912, 'full': 85_798_656}
    if name in fixed:
        return fixed[name]
    shallow, deep = VPT_PAPER_P[task]
    return {'vpt_shallow_paper': shallow * 768, 'vpt_deep_paper': 12 * deep * 768}[name]


EXPECTED_GRID_POINTS = {'lora_r8': 20, 'adaptformer_d16': 20, 'adaptformer_paper': 20, 'vpt_deep_p32': 20,
                        'vpt_deep_paper': 20, 'vpt_shallow_paper': 20, 'full': 20, 'vpt_deep_p32_sgd': 20}
METHODS = list(EXPECTED_GRID_POINTS)
SWEEP_METHODS = ['lora_r8', 'adaptformer_d16', 'adaptformer_paper', 'vpt_deep_p32', 'vpt_deep_paper',
                 'vpt_shallow_paper', 'full']
CHECK_METHODS = ['vpt_deep_p32_sgd']
PILOT_TASKS = ['oxford_flowers102', 'eurosat', 'clevr_count']
PEFT_LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
GRID_LRS = {'full': [3e-6, 1e-5, 3e-5, 1e-4, 3e-4], 'vpt_shallow_paper': [3e-4, 1e-3, 3e-3, 1e-2, 3e-2], 'vpt_deep_p32_sgd': [25.0, 5.0, 1.0, 0.25, 0.05]}
GRID_WDS = {'vpt_deep_p32_sgd': [0.01, 0.001, 0.0001, 0.0]}


@pytest.fixture(autouse=True)
def in_repo_without_weights(monkeypatch):
    monkeypatch.chdir(REPO)  # config paths are relative to the repository, as for main_tune.py
    monkeypatch.setattr(VisionTransformerPETL, 'load_pretrained', lambda self, *a, **k: None)


def load(name, task):
    args = main_tune.setup_parser().parse_args(['--sys843', name, '--data', f'processed_vtab-{task}'])
    return main_tune.build_params(args)


def count_no_head(params, grid_point, task):
    params.update(grid_point)
    params.class_num = VTAB_DATASETS[task]
    _, _, no_head = build_model.get_model(params)
    return no_head


def test_index_lists_every_method():
    assert sorted(main_tune.sys843_methods()) == sorted(METHODS)


@pytest.mark.parametrize('name', METHODS)
def test_config_loads_with_expected_grid_and_protocol(name):
    params, grid = load(name, 'eurosat')
    assert len(ParameterGrid(grid)) == EXPECTED_GRID_POINTS[name]
    assert set(grid) == {'lr', 'wd'}  # no method hyperparameter is searched
    assert grid['lr'] == GRID_LRS.get(name, PEFT_LRS)
    assert grid['wd'] == GRID_WDS.get(name, [0.0, 1e-4, 1e-3, 1e-2])
    assert params.method_name == name
    assert params.experiment_name == 'sys843_vtab'
    assert params.pretrained_weights == 'vit_base_patch16_224_in21k'
    assert params.optimizer == ('sgd' if name == 'vpt_deep_p32_sgd' else 'adamw')
    assert params.epoch == 100 and params.warmup_epoch == 10
    assert params.tie_tolerance == 0.5
    assert params.random_seed == 42 and 42 not in list(params.final_seeds)  # tuning and final seeds disjoint
    assert params.normalized is False and params.final_acc_hp is True
    assert params.eval_freq > params.epoch  # validation only after the last epoch
    assert list(params.final_seeds) == [0, 1, 2]
    assert 'per_task' not in params


@pytest.mark.parametrize('name', [m for m in METHODS if not m.endswith('_paper') or m == 'adaptformer_paper'])
def test_fixed_configs_parameter_count(name):
    params, grid = load(name, 'eurosat')
    for point in ParameterGrid({k: v for k, v in grid.items() if k not in ('lr', 'wd')}):
        assert count_no_head(params, point, 'eurosat') == expected_no_head(name, 'eurosat')


@pytest.mark.parametrize('name', ['vpt_shallow_paper', 'vpt_deep_paper'])
def test_vpt_paper_per_task_prompt_lengths(name):
    counted = {}
    for task in VTAB_DATASETS:
        params, grid = load(name, task)
        p = VPT_PAPER_P[task][0 if 'shallow' in name else 1]
        assert params.vpt_num == p, task
        if p not in counted:
            counted[p] = count_no_head(params, {}, task)
        assert counted[p] == expected_no_head(name, task), task


def test_explicit_file_arguments_override_sys843():
    args = main_tune.setup_parser().parse_args(['--sys843', 'lora_r8', '--lrwd', 'experiment/config/sys843/lr_wd_full.yml'])
    _, grid = main_tune.build_params(args)
    assert grid['lr'] == GRID_LRS['full']


def test_mai_defaults_unchanged_without_sys843():
    args = main_tune.resolve_config_paths(main_tune.setup_parser().parse_args([]))
    assert (args.default, args.tune, args.lrwd) == (main_tune.DEFAULT_CONFIG, main_tune.DEFAULT_TUNE,
                                                    main_tune.DEFAULT_LRWD)


def test_unknown_task_in_per_task_config_fails():
    with pytest.raises(SystemExit):
        load('vpt_deep_paper', 'not_a_task')


def test_adaptformer_scale_is_fixed_at_paper_default():
    for name in ('adaptformer_d16', 'adaptformer_paper'):
        params, grid = load(name, 'eurosat')
        assert params.adapter_scaler == 0.1 and 'adapter_scaler' not in grid


def test_vpt_sgd_check_uses_vpt_paper_recipe():
    params, _ = load('vpt_deep_p32_sgd', 'eurosat')
    assert params.vpt_mode == 'deep' and params.vpt_num == 32
    assert params.optimizer == 'sgd' and params.momentum == 0.9 and params.lr_base_batch == 256
    assert params.batch_size == 64  # peak lr = base lr x 64 / 256


def test_sweep_and_check_methods():
    assert main_tune.sys843_sweep_methods() == SWEEP_METHODS
    assert main_tune.sys843_check_methods() == CHECK_METHODS


def grid_size(name):
    return len(ParameterGrid(load(name, 'eurosat')[1]))


def test_run_counts():
    """The counts stated in experiment/config/sys843/README.md, from the configs themselves."""
    per_task = sum(grid_size(m) for m in main_tune.sys843_sweep_methods())
    assert per_task == 140
    assert per_task * len(VTAB_DATASETS) == 2_660
    assert len(main_tune.sys843_sweep_methods()) * len(VTAB_DATASETS) * 3 == 399
    check = main_tune.sys843_check_methods()
    assert sum(grid_size(m) for m in check) * len(PILOT_TASKS) == 60
    assert len(check) * len(PILOT_TASKS) * 3 == 9


def bash(script, env=None):
    full_env = dict(os.environ, SYS843_REPO=REPO, **(env or {}))
    return subprocess.run(['bash', '-c', f'source scripts/rorqual/env.sh; {script}'], cwd=REPO,
                          capture_output=True, text=True, env=full_env)


def test_env_sh_lists_the_same_sweep_methods():
    proc = bash('sys843_sweep_methods; echo; echo "$SYS843_PILOT_TASKS"')
    assert proc.returncode == 0, proc.stderr
    sweep, pilot = proc.stdout.splitlines()
    assert sweep.split() == SWEEP_METHODS and pilot.split() == PILOT_TASKS


def test_run_tasks_skips_check_config_outside_the_pilot(tmp_path):
    """run_tasks.sh skips a check config on a non-pilot task before staging any data."""
    proc = bash('bash scripts/rorqual/run_tasks.sh tune vpt_deep_p32_sgd cifar dtd',
                env={'SYS843_RUNS': str(tmp_path)})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.count('not a pilot task') == 2


def test_tune_time_per_config_from_methods_yml():
    """sys843_tune_time (scripts/rorqual/env.sh) reads every config's tune_time in methods.yml."""
    expected = {'lora_r8': '03:30:00', 'adaptformer_d16': '03:30:00', 'adaptformer_paper': '03:30:00',
                'vpt_deep_p32': '03:30:00', 'vpt_deep_p32_sgd': '03:30:00', 'full': '04:30:00',
                'vpt_deep_paper': '06:00:00', 'vpt_shallow_paper': '06:00:00'}
    assert set(expected) == set(main_tune.sys843_methods())
    for name, limit in expected.items():
        assert main_tune.sys843_methods()[name]['tune_time'] == limit
        out = subprocess.run(['bash', '-c', f'source scripts/rorqual/env.sh && sys843_tune_time {name}'],
                             cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
        assert out == limit
    bad = subprocess.run(['bash', '-c', 'source scripts/rorqual/env.sh && sys843_tune_time nope'],
                         cwd=REPO, capture_output=True, text=True)
    assert bad.returncode != 0
