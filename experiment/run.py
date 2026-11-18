import os
from experiment.build_model import get_model
from experiment.build_loader import get_loader
from utils.global_var import OUTPUT_DIR, TUNE_DIR, TUNE_DIR_TEST, DEFAULT_FINAL_SEEDS
from engine.trainer import Trainer
from timm.utils import get_outdir
from utils.log_utils import logging_env_setup
from utils.misc import method_name, get_param
from utils.dotdict import config_dict
from datetime import datetime
import yaml
import torch
from utils.misc import set_seed
from collections import OrderedDict
from statistics import mean, stdev
import json
import time
import csv
import numpy as np
from utils.setup_logging import get_logger
from experiment.selection import select_best_tune, DEFAULT_TIE_TOLERANCE

logger = get_logger("PETL_vision")


def train(params, train_loader, val_loader, test_loader):
    """Build a model and train it; trainer.run_stats holds its time/memory measurements."""
    model, tune_parameters, model_grad_params_no_head = get_model(params)
    trainer = Trainer(model, tune_parameters, params)
    train_metrics, best_eval_metrics, eval_metrics = trainer.train_classifier(train_loader, val_loader, test_loader)
    return train_metrics, best_eval_metrics, eval_metrics, model_grad_params_no_head, trainer

def basic_run(params):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    data_name = params.data.split("-")[-1]
    dataset_name = params.data.split("-")[0]
    method = method_name(params)
    start_time = datetime.now().strftime("%Y-%m-%d-%H:%M")
    output_root = get_param(params, 'output_root', None) or OUTPUT_DIR
    output_dir = os.path.join(output_root, params.pretrained_weights, dataset_name, method, data_name, start_time)
    params.output_dir = get_outdir(output_dir)
    params_text = yaml.safe_dump(config_dict(params), default_flow_style=False)
    with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
        f.write(params_text)
    logging_env_setup(params)
    logger.info(f'Start loading {data_name}')
    train_loader, val_loader, test_loader = get_loader(params, logger)

    train_metrics, best_eval_metrics, eval_metrics, _, trainer = train(params, train_loader, val_loader, test_loader)
    result = OrderedDict([('train', train_metrics), ('eval', eval_metrics), ('best_eval', best_eval_metrics),
                          ('normalized', bool(get_param(params, 'normalized', False))),
                          ('seed', params.random_seed)])
    result.update(trainer.run_stats)
    with open(os.path.join(output_dir, 'run_result.json'), 'w') as f:
        json.dump(result, f, indent=2)


def update_output_dir(default_params, test):
    logger.info(f'start running {default_params.method_name}')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    data_name = default_params.data.split("-")[-1]
    dataset_name = default_params.data.split("-")[0]
    method = default_params.method_name
    if test:
        output_dir = os.path.join(TUNE_DIR_TEST, default_params.experiment_name, dataset_name, data_name, method)
    else:
        tune_root = get_param(default_params, 'tune_root', None) or TUNE_DIR
        output_dir = os.path.join(tune_root, default_params.experiment_name, dataset_name, data_name, method)
    default_params.output_dir = output_dir

    logging_env_setup(default_params)
    return output_dir, data_name




def collect_prediction(default_params):
    output_dir, data_name = update_output_dir(default_params, False)

    if os.path.isfile(os.path.join(default_params.output_dir, 'logits.npy')) and os.path.isfile(os.path.join(default_params.output_dir, 'gt.npy')):
        logger.info('finished, next')
        return

    logger.info(f'-----------------------{default_params.method_name}: {data_name}------------------------------')
    with open(os.path.join(default_params.output_dir, 'tune_summary.csv'), newline='') as f:
        tune_summary = list(csv.reader(f))[1:]

    best_tune, _ = select_best_tune(tune_summary, True, tie_tolerance(default_params))

    logger.info(f'get best tune {best_tune}')
    final_params = default_params
    final_params.update(best_tune)
    final_params.early_patience = 100
    final_params.eval_freq = 100
    final_params.final_run = True
    logger.info(final_params)
    train_loader, _, test_loader = get_loader(default_params, logger)
    logger.info(f'----------- Start final runs-----------')
    train_start = time.time()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    set_seed(final_params.random_seed)
    model, tune_parameters, model_grad_params_no_head = get_model(final_params)
    trainer = Trainer(model, tune_parameters, final_params)
    train_metrics, best_eval_metrics, eval_metrics = trainer.train_classifier(train_loader, None, test_loader)
    logger.info(
        f'train acc {train_metrics["top1"]}, best acc {best_eval_metrics["top1"]}, last acc {eval_metrics["top1"]}')

    logits, gt = trainer.collect_logits(test_loader)
    with open(os.path.join(final_params.output_dir, 'logits.npy'), 'wb') as f:
        np.save(f, logits)
    with open(os.path.join(final_params.output_dir, 'gt.npy'), 'wb') as f:
        np.save(f, gt)

    train_end = time.time()
    logger.info(f'----------- Total train time : {(train_end - train_start) / 60} mins-----------')




def final_seeds(params):
    """Seeds of the final runs: --seeds / final_seeds in the config, else DEFAULT_FINAL_SEEDS (0, 1, 2)."""
    seeds = get_param(params, 'final_seeds', None)
    if seeds is None:
        return list(DEFAULT_FINAL_SEEDS)
    if isinstance(seeds, (int, str)):
        seeds = [seeds]
    return [int(seed) for seed in seeds]


def tie_tolerance(params):
    """Tie tolerance of the grid-point selection, in percentage points (--tie_tolerance /
    tie_tolerance in the config, default 0.5 = one image of val200)."""
    tol = get_param(params, 'tie_tolerance', None)
    return DEFAULT_TIE_TOLERANCE if tol is None else float(tol)


def summary_name_for(result_name):
    """final_result.json -> final_summary.csv; any other result name X.json -> X_summary.csv."""
    if result_name == 'final_result.json':
        return 'final_summary.csv'
    return os.path.splitext(result_name)[0] + '_summary.csv'


def _mean_std(values):
    """Mean and sample standard deviation (n - 1); std is None with fewer than two values."""
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    return mean(values), (stdev(values) if len(values) > 1 else None)


SEED_STAT_KEYS = ['test_top1', 'test_top5', 'test_loss', 'train_top1', 'train_top5', 'train_loss', 'train_time_s',
                  'epoch_time_mean_s', 'test_eval_time_s', 'peak_memory_allocated_mib', 'peak_memory_reserved_mib']


def final_run(default_params, sum_name=None, result_name='final_result.json', grid=None):
    """Retrain the best tuned config on train800val200 for each seed and test once at the end.

    The test set is evaluated a single time per seed, after the last epoch; that
    last-epoch accuracy is the reported metric (no per-epoch test evaluation and
    no best-over-epochs number). Per-seed results plus mean and sample std go to
    the summary CSV and the result JSON. grid: the current grid points; tuned rows of other
    points are ignored by the selection (see experiment.selection.select_best_tune).
    """
    if sum_name is None:
        sum_name = summary_name_for(result_name)
    if os.path.isfile(os.path.join(default_params.output_dir, result_name)):
        logger.info(f'{result_name} exist, next')
        return
    with open(os.path.join(default_params.output_dir, 'tune_summary.csv'), newline='') as f:
        tune_summary = list(csv.reader(f))[1:]
    # final_acc_hp: select on the final-epoch val accuracy, else on the best one along training;
    # near-ties are broken by the rule in experiment/selection.py.
    best_tune, selection = select_best_tune(tune_summary, bool(default_params.final_acc_hp),
                                            tie_tolerance(default_params), grid=grid)
    with open(os.path.join(default_params.output_dir, 'tune_selection.json'), 'w') as f:
        json.dump(selection, f, indent=2)
    logger.info(f'selected {best_tune}: {selection["n_tied"]} grid point(s) within '
                f'{selection["tolerance_pp"]} pp of the best val accuracy {selection["best_val_acc"]}')
    final_params = default_params
    final_params.update(best_tune)
    final_params.early_patience = 100
    logger.info(final_params)

    final_params.final_run = True
    train_loader, _, test_loader = get_loader(default_params, logger)
    random_seeds = final_seeds(final_params)
    logger.info(f'----------- Start final runs: {len(random_seeds)} run(s) with seeds {random_seeds}-----------')
    rows = []
    train_start = time.time()
    for idx, seed in enumerate(random_seeds):
        train_this_start = time.time()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        final_params.random_seed = seed
        set_seed(seed)
        # Train without looking at the test set, then evaluate it once.
        train_metrics, _, _, model_grad_params_no_head, trainer = train(final_params, train_loader, None, None)
        trainer.sync()
        eval_start = time.perf_counter()
        test_metrics = trainer.eval_classifier(test_loader, 'test')
        trainer.sync()
        test_eval_time = time.perf_counter() - eval_start

        row = OrderedDict([('run_idx', idx), ('seed', seed)])
        row.update([('train_' + k, v) for k, v in train_metrics.items()])
        row.update([('test_' + k, v) for k, v in test_metrics.items()])
        row['test_eval_time_s'] = round(test_eval_time, 4)
        row.update(trainer.run_stats)
        rows.append(row)
        del trainer
        train_this_end = time.time()
        logger.info(f'----------- seed {seed}: test top1 {test_metrics["top1"]}, '
                    f'train this run : {(train_this_end - train_this_start) / 60} mins-----------')

    stats = OrderedDict()
    for key in SEED_STAT_KEYS:
        stats[key] = _mean_std([row[key] for row in rows])
    write_final_summary(rows, stats, os.path.join(final_params.output_dir, sum_name))

    accs = [row['test_top1'] for row in rows]
    last = rows[-1]
    result = OrderedDict([
        ('metric', 'test top-1 accuracy after the last epoch, test set evaluated once per seed'),
        ('avg_acc', stats['test_top1'][0]),
        ('std_acc', stats['test_top1'][1]),
        ('accs', accs),
        ('seeds', random_seeds),
    ])
    for key in SEED_STAT_KEYS[1:]:
        result['avg_' + key], result['std_' + key] = stats[key]
    result.update([
        ('inserted_parameters', model_grad_params_no_head),
        ('trainable_params', last['trainable_params']),
        ('trainable_params_no_head', last['trainable_params_no_head']),
        ('device', last['device']),
        ('gpu_name', last['gpu_name']),
        ('torch_version', last['torch_version']),
        ('timm_version', last['timm_version']),
        ('normalized', bool(get_param(final_params, 'normalized', False))),
        ('epochs', final_params.epoch),
        ('best_tune', best_tune),
        ('selection', selection),
        ('per_seed', rows),
    ])
    with open(os.path.join(final_params.output_dir, result_name), 'w') as f:
        json.dump(result, f, indent=2)
    train_end = time.time()
    logger.info(f'----------- final test top1 {result["avg_acc"]} +- {result["std_acc"]} over seeds {random_seeds}-----------')
    logger.info(f'----------- Total train time : {(train_end - train_start) / 60} mins-----------')


def write_final_summary(rows, stats, filename):
    """Per-seed rows, then a 'mean' and a 'std' row (sample std) over the seeds."""
    fieldnames = list(rows[0].keys())
    with open(filename, mode='w', newline='') as cf:
        dw = csv.DictWriter(cf, fieldnames=fieldnames)
        dw.writeheader()
        for row in rows:
            dw.writerow(_csv_row(row))
        for label, pos in (('mean', 0), ('std', 1)):
            agg = OrderedDict((k, '') for k in fieldnames)
            agg['run_idx'] = label
            for key, values in stats.items():
                if key in agg and values[pos] is not None:
                    agg[key] = values[pos]
            dw.writerow(agg)


def _csv_row(row):
    """Lists (e.g. per-epoch times) go into a CSV cell as JSON."""
    return OrderedDict((k, json.dumps(v) if isinstance(v, (list, tuple, dict)) else v) for k, v in row.items())




def evaluate(default_params):
    _, _, test_loader = get_loader(default_params, logger)
    if 'eval' in default_params.test_data:
        result_name = f'{default_params.test_data.split("_")[1]}_result.json'
    else:
        result_name = f'{default_params.test_data}_result.json'
    if not os.path.isfile(os.path.join(default_params.output_dir, result_name)):
        if not os.path.isfile(os.path.join(default_params.output_dir, 'final_result.json')):
            logger.info('no final_result.json, the model is not fine-tuned, show model zero shot performance')
            best_tune = ()
            result_name = 'zero_shot_' + result_name
        else:
            result = json.load(open(os.path.join(default_params.output_dir, 'final_result.json')))
            best_tune = result['best_tune']
            default_params.update(best_tune)

        model, tune_parameters, model_grad_params_no_head = get_model(default_params)
        trainer = Trainer(model, tune_parameters, default_params)
        if not os.path.isfile(os.path.join(default_params.output_dir, 'model.pt')):
            assert not os.path.isfile(os.path.join(default_params.output_dir, 'final_result.json'))
            logger.info('no model.pt, shows zero shot performance')
        else:
            trainer.load_weight()
        eval_metrics = trainer.eval_classifier(test_loader, 'test')
        json.dump(
            {"avg_acc": eval_metrics['top1'], "inserted_parameters": model_grad_params_no_head,
             'best_tune': best_tune},
            open(os.path.join(default_params.output_dir, result_name), 'w'))
    else:
        logger.info(f'finish {result_name} for {default_params.method_name}')
    return


def result_tracker(first_col, train_metrics, eval_metrics, best_eval_metrics, filename, write_header=False, first_col_name='param_set',
                   eval_name='val_', extra=None):
    """Append one row. The first nine columns keep their original order (final_run selects
    on columns 5 and 8); `extra` (e.g. time and memory stats) is appended after them.
    The header is written when the file is missing or empty (write_header is ignored)."""
    rowd = OrderedDict([(first_col_name, first_col)])
    rowd.update([('train_' + k, v) for k, v in train_metrics.items()])
    rowd.update([(eval_name + k, v) for k, v in eval_metrics.items()])
    rowd.update([(eval_name + "best_" + k, v) for k, v in best_eval_metrics.items()])
    if extra:
        rowd.update(extra)
    write_header = not os.path.isfile(filename) or os.path.getsize(filename) == 0
    with open(filename, mode='a', newline='') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header:
            dw.writeheader()
        dw.writerow(_csv_row(rowd))
