"""Grid-point selection with the tie-break rule (experiment/selection.py), the tie tolerance
setting, and the optimizer's linear lr scaling (VPT SGD check). No weights needed."""
import pytest
import torch

import main_tune
from conftest import REPO
from engine.optimizer import effective_lr, make_optimizer
from experiment.run import tie_tolerance
from experiment.selection import select_best_tune, select_grid_point
from utils.dotdict import DotDict


def rows(points):
    """tune_summary.csv rows (strings) for [(param_set, val_top1)], val_best_top1 = val_top1 + 1."""
    return [[repr(p), '1.0', '90.0', '99.0', '0.5', str(acc), '100.0', '0.4', str(acc + 1)]
            for p, acc in points]


def test_single_best_point_wins():
    best, tied = select_grid_point([({'lr': 0.01, 'wd': 0.0}, 80.0), ({'lr': 0.001, 'wd': 0.0}, 79.0)])
    assert best == {'lr': 0.01, 'wd': 0.0} and len(tied) == 1


def test_within_tolerance_lowest_lr_then_lowest_wd():
    scored = [({'lr': 0.01, 'wd': 0.0}, 80.0),
              ({'lr': 0.001, 'wd': 0.01}, 79.5),   # tied (0.5 pp = one val200 image)
              ({'lr': 0.001, 'wd': 0.001}, 79.5),  # tied, same lr, lower wd: wins
              ({'lr': 0.0001, 'wd': 0.0}, 79.0)]   # 1.0 pp below: not tied
    best, tied = select_grid_point(scored, tolerance=0.5)
    assert best == {'lr': 0.001, 'wd': 0.001}
    assert len(tied) == 3


def test_zero_tolerance_keeps_only_exact_ties():
    scored = [({'lr': 0.01, 'wd': 0.0}, 80.0), ({'lr': 0.001, 'wd': 0.0}, 79.5), ({'lr': 0.003, 'wd': 0.0}, 80.0)]
    best, tied = select_grid_point(scored, tolerance=0.0)
    assert best == {'lr': 0.003, 'wd': 0.0} and len(tied) == 2


def test_larger_tolerance_ties_more_points():
    scored = [({'lr': 0.01, 'wd': 0.0}, 80.0), ({'lr': 0.0001, 'wd': 0.0}, 79.0)]
    assert select_grid_point(scored, tolerance=1.0)[0] == {'lr': 0.0001, 'wd': 0.0}


def test_selection_does_not_depend_on_row_order():
    scored = [({'lr': lr, 'wd': wd}, 80.0) for lr in (0.01, 0.001) for wd in (0.01, 0.0)]
    assert select_grid_point(scored)[0] == select_grid_point(scored[::-1])[0] == {'lr': 0.001, 'wd': 0.0}


def test_invalid_input():
    with pytest.raises(ValueError):
        select_grid_point([])
    with pytest.raises(ValueError):
        select_grid_point([({'lr': 0.1}, 1.0)], tolerance=-1)


def test_select_best_tune_records_the_choice():
    tune_rows = rows([({'lr': 0.01, 'wd': 0.0}, 80.0), ({'lr': 0.001, 'wd': 0.0}, 79.5),
                      ({'lr': 0.0001, 'wd': 0.0}, 70.0)])
    best, selection = select_best_tune(tune_rows, final_acc_hp=True, tolerance=0.5)
    assert best == {'lr': 0.001, 'wd': 0.0}
    assert selection['metric'] == 'val_top1' and selection['tolerance_pp'] == 0.5
    assert selection['n_grid_points'] == 3 and selection['n_tied'] == 2
    assert selection['best_val_acc'] == 80.0 and selection['selected_val_acc'] == 79.5
    assert selection['selected'] == best
    assert [t['param_set'] for t in selection['tied']] == [{'lr': 0.001, 'wd': 0.0}, {'lr': 0.01, 'wd': 0.0}]


def test_select_best_tune_on_best_along_training():
    tune_rows = rows([({'lr': 0.01, 'wd': 0.0}, 80.0)])
    tune_rows[0][8] = '95.0'
    _, selection = select_best_tune(tune_rows, final_acc_hp=False)
    assert selection['metric'] == 'val_best_top1' and selection['best_val_acc'] == 95.0


def test_tie_tolerance_from_config_cli_and_default(monkeypatch):
    monkeypatch.chdir(REPO)
    parse = main_tune.setup_parser().parse_args
    params, _ = main_tune.build_params(parse(['--sys843', 'lora_r8', '--data', 'processed_vtab-eurosat']))
    assert tie_tolerance(params) == 0.5
    params, _ = main_tune.build_params(parse(['--sys843', 'lora_r8', '--data', 'processed_vtab-eurosat',
                                              '--tie_tolerance', '1.5']))
    assert tie_tolerance(params) == 1.5
    assert tie_tolerance(DotDict({})) == 0.5  # configs without the key (Mai's)


def opt_params(**kw):
    base = dict(lr=1.0, wd=0.001, batch_size=64, momentum=0.9, optimizer='sgd')
    base.update(kw)
    return DotDict(base)


def test_effective_lr_linear_scaling():
    assert effective_lr(opt_params(lr=25.0, lr_base_batch=256)) == pytest.approx(6.25)
    assert effective_lr(opt_params(lr=0.05, lr_base_batch=256, batch_size=128)) == pytest.approx(0.025)
    assert effective_lr(opt_params(lr=0.001)) == 0.001  # no lr_base_batch: lr used as is


def test_make_optimizer_sgd_with_scaled_lr():
    w = torch.nn.Parameter(torch.zeros(2))
    opt = make_optimizer([w], opt_params(lr=1.0, lr_base_batch=256))
    assert isinstance(opt, torch.optim.SGD)
    group = opt.param_groups[0]
    assert group['lr'] == pytest.approx(0.25) and group['momentum'] == 0.9 and group['weight_decay'] == 0.001
    adamw = make_optimizer([w], opt_params(optimizer='adamw', lr=0.001))
    assert isinstance(adamw, torch.optim.AdamW) and adamw.param_groups[0]['lr'] == 0.001
    with pytest.raises(ValueError):
        make_optimizer([w], opt_params(optimizer='rmsprop'))
