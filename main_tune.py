import argparse
import os
from utils.setup_logging import get_logger
from utils.misc import set_seed, str2bool
import time
from utils.misc import load_yaml
from experiment.tune import hyper_tune_final_run, STAGES

logger = get_logger("PETL_vision")

# Mai et al.'s defaults, used when neither the file argument nor --sys843 is given.
DEFAULT_CONFIG = 'experiment/config/default_vtab_processed.yml'
DEFAULT_TUNE = 'experiment/config/method/lora.yml'
DEFAULT_LRWD = 'experiment/config/lr_wd_vtab_processed.yml'
SYS843_DIR = 'experiment/config/sys843'


def main():
    args = setup_parser().parse_args()
    default_params, hyper_tune_params = build_params(args)
    set_seed(default_params.random_seed)
    start = time.time()
    hyper_tune_final_run(default_params, hyper_tune_params, test=args.test, stage=args.stage)
    end = time.time()
    logger.info(f'----------- Total Run time : {(end - start) / 60} mins-----------')


def sys843_methods(sys843_dir=SYS843_DIR):
    """The SYS843 method configs: {name: {tune, lrwd, setting}} from <sys843_dir>/methods.yml."""
    return load_yaml(os.path.join(sys843_dir, 'methods.yml'))


# Setting of the sanity-check configs (e.g. vpt_deep_p32_sgd): run on the pilot tasks only,
# never part of the 19-task sweep.
CHECK_SETTING = 'check'


def sys843_sweep_methods(sys843_dir=SYS843_DIR):
    """Names of the configs of the 19-task sweep: every config whose setting is not `check`."""
    return [name for name, entry in sys843_methods(sys843_dir).items()
            if CHECK_SETTING not in entry['setting']]


def sys843_check_methods(sys843_dir=SYS843_DIR):
    """Names of the sanity-check configs (setting `check`), run on the pilot tasks only."""
    return [name for name, entry in sys843_methods(sys843_dir).items()
            if CHECK_SETTING in entry['setting']]


def resolve_config_paths(args):
    """Fill --default/--tune/--lrwd from --sys843 <name> (SYS843 configs) or Mai's defaults.

    An explicit --default/--tune/--lrwd always wins over --sys843.
    """
    if args.sys843 is not None:
        methods = sys843_methods()
        if args.sys843 not in methods:
            raise SystemExit(f'--sys843 {args.sys843}: unknown, choose from {sorted(methods)}')
        entry = methods[args.sys843]
        args.default = args.default or os.path.join(SYS843_DIR, 'default_vtab_processed.yml')
        args.tune = args.tune or os.path.join(SYS843_DIR, entry['tune'])
        args.lrwd = args.lrwd or os.path.join(SYS843_DIR, entry['lrwd'])
    args.default = args.default or DEFAULT_CONFIG
    args.tune = args.tune or DEFAULT_TUNE
    args.lrwd = args.lrwd or DEFAULT_LRWD
    return args


def task_name(data):
    """processed_vtab-cifar -> cifar."""
    return data.split('-', 1)[-1]


def apply_per_task(method_params, task):
    """Resolve a `per_task: {param: {task: value}}` block of a method config for one task.

    Used for settings that the original papers give per task (e.g. VPT prompt lengths).
    """
    per_task = method_params.pop('per_task', None)
    if not per_task:
        return method_params
    for param, by_task in per_task.items():
        if task not in by_task:
            raise SystemExit(f'per_task.{param} has no value for task {task!r}')
        method_params[param] = by_task[task]
    return method_params


def build_params(args):
    """Default config + method config + lr/wd grid -> (default_params, hyper_tune_params).

    Lists in the method and lr/wd configs are tuned (every combination), scalars are fixed.
    """
    resolve_config_paths(args)
    default_params = load_yaml(args.default)
    default_params.data = args.data
    method_params = apply_per_task(load_yaml(args.tune), task_name(args.data))
    lr_wd = load_yaml(args.lrwd)
    hyper_tune_params = {}
    method_static = {}
    for k, v in method_params.items():
        if isinstance(v, list):
            hyper_tune_params[k] = v
        else:
            method_static[k] = v
    for k, v in lr_wd.items():
        if isinstance(v, list):
            hyper_tune_params[k] = v
        else:
            method_static[k] = v
    default_params.update(method_static)
    default_params.final_output_name = args.final_output_name
    # Settings below apply to both the tuning runs and the final runs.
    if args.normalized is not None:
        default_params.normalized = args.normalized
    else:
        default_params.normalized = bool(default_params.get('normalized', False))
    if args.seeds is not None:
        default_params.final_seeds = args.seeds
    if args.tune_seed is not None:
        default_params.random_seed = args.tune_seed
    if args.tie_tolerance is not None:
        default_params.tie_tolerance = args.tie_tolerance
    if args.tune_root is not None:
        default_params.tune_root = args.tune_root
    if args.pretrained_dir is not None:
        default_params.pretrained_dir = args.pretrained_dir
    if args.data_path is not None:
        default_params.data_path = args.data_path
    if args.num_workers is not None:
        default_params.num_workers = args.num_workers
    if args.test_bs is not None:
        default_params.test_batch_size = args.test_bs
        if args.bs is not None:
            default_params.batch_size = args.bs
    return default_params, hyper_tune_params


def setup_parser():
    parser = argparse.ArgumentParser(description='PETL_Vision_tune')
    parser.add_argument('--data', default='processed_vtab-caltech101')
    parser.add_argument('--sys843', default=None, metavar='NAME',
                        help='Use the SYS843 configs for method NAME (see experiment/config/sys843/methods.yml): '
                             'sets --default, --tune and --lrwd unless they are given')
    parser.add_argument('--default', default=None, help=f'Default config. Default: {DEFAULT_CONFIG}')
    parser.add_argument('--tune', default=None, help=f'Method config. Default: {DEFAULT_TUNE}')
    parser.add_argument('--lrwd', default=None, help=f'lr/wd grid config. Default: {DEFAULT_LRWD}')
    parser.add_argument('--test', action='store_true', help="put results in test folder")
    parser.add_argument('--stage', choices=STAGES, default='all',
                        help='all: tune the missing grid points, then the final runs; tune: stop after tuning; '
                             'final: final runs only, error if tuning is incomplete')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='DataLoader workers. Default: num_workers in --default, else 4 (1 with debug)')
    parser.add_argument('--final_output_name', default='final_result.json')
    parser.add_argument('--test_bs', type=int, default=None)
    parser.add_argument('--bs', type=int, default=None)
    parser.add_argument('--normalized', type=str2bool, default=None, metavar='{true,false}',
                        help='Normalize images with the ImageNet mean and std, for tuning and final runs. '
                             'Default: the value in --default (false in default_vtab_processed.yml)')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Seeds of the final runs. Default: final_seeds in --default, else 0 1 2')
    parser.add_argument('--tune_seed', type=int, default=None,
                        help='Seed set before every tuning configuration (recorded as random_seed in '
                             'tune_summary.csv). Default: random_seed in --default (42)')
    parser.add_argument('--tie_tolerance', type=float, default=None, metavar='PP',
                        help='Grid points within PP percentage points of the best validation accuracy are '
                             'tied; the lowest lr, then the lowest wd, wins (experiment/selection.py). '
                             'Default: tie_tolerance in --default, else 0.5 (one image of val200)')
    parser.add_argument('--tune_root', default=None,
                        help='Root of the tuning outputs. Default: $PETL_TUNE_DIR or ./runs/tune')
    parser.add_argument('--pretrained_dir', default=None,
                        help='Directory holding the pretrained weights. Default: $PETL_PRETRAINED_DIR or ./pretrained_weights')
    parser.add_argument('--data_path', default=None,
                        help='Dataset root. Default: data_path in --default')
    return parser


if __name__ == '__main__':
    main()
