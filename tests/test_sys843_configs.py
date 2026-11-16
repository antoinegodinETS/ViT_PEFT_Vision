"""The SYS843 configs (experiment/config/sys843) load through main_tune.py and give the
expected trainable-parameter counts and grid sizes. Random init, no weights needed."""
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
             'vpt_deep_p32': 294_912, 'full': 85_798_656}
    if name in fixed:
        return fixed[name]
    shallow, deep = VPT_PAPER_P[task]
    return {'vpt_shallow_paper': shallow * 768, 'vpt_deep_paper': 12 * deep * 768}[name]


EXPECTED_GRID_POINTS = {'lora_r8': 20, 'adaptformer_d16': 40, 'adaptformer_paper': 20, 'vpt_deep_p32': 20,
                        'vpt_deep_paper': 20, 'vpt_shallow_paper': 20, 'full': 16}
METHODS = list(EXPECTED_GRID_POINTS)


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
    lrs = {'full': [1e-5, 3e-5, 1e-4, 3e-4]}.get(name, [1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    assert grid['lr'] == lrs and grid['wd'] == [0.0, 1e-4, 1e-3, 1e-2]
    assert params.method_name == name
    assert params.experiment_name == 'sys843_vtab'
    assert params.pretrained_weights == 'vit_base_patch16_224_in21k'
    assert params.optimizer == 'adamw' and params.epoch == 100 and params.warmup_epoch == 10
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
    assert grid['lr'] == [1e-5, 3e-5, 1e-4, 3e-4]


def test_mai_defaults_unchanged_without_sys843():
    args = main_tune.resolve_config_paths(main_tune.setup_parser().parse_args([]))
    assert (args.default, args.tune, args.lrwd) == (main_tune.DEFAULT_CONFIG, main_tune.DEFAULT_TUNE,
                                                    main_tune.DEFAULT_LRWD)


def test_unknown_task_in_per_task_config_fails():
    with pytest.raises(SystemExit):
        load('vpt_deep_paper', 'not_a_task')
