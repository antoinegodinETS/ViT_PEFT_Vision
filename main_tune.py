import argparse
from utils.setup_logging import get_logger
from utils.misc import set_seed, str2bool
import time
from utils.misc import load_yaml
from experiment.tune import hyper_tune_final_run

logger = get_logger("PETL_vision")

def main():
    args = setup_parser().parse_args()
    default_params = load_yaml(args.default)
    default_params.data = args.data
    method_params = load_yaml(args.tune)
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
    if args.tune_root is not None:
        default_params.tune_root = args.tune_root
    if args.pretrained_dir is not None:
        default_params.pretrained_dir = args.pretrained_dir
    if args.data_path is not None:
        default_params.data_path = args.data_path
    set_seed(default_params.random_seed)
    if args.test_bs is not None:
        default_params.test_batch_size = args.test_bs
        if args.bs is not None:
            default_params.batch_size = args.bs
    start = time.time()
    hyper_tune_final_run(default_params, hyper_tune_params, test=args.test)
    end = time.time()
    logger.info(f'----------- Total Run time : {(end - start) / 60} mins-----------')


def setup_parser():
    parser = argparse.ArgumentParser(description='PETL_Vision_tune')
    parser.add_argument('--data', default='processed_vtab-caltech101')
    parser.add_argument('--default', default='experiment/config/default_vtab_processed.yml')
    parser.add_argument('--tune',  default='experiment/config/method/lora.yml')
    parser.add_argument('--lrwd', default='experiment/config/lr_wd_vtab_processed.yml')
    parser.add_argument('--test', action='store_true', help="put results in test folder")
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
    parser.add_argument('--tune_root', default=None,
                        help='Root of the tuning outputs. Default: $PETL_TUNE_DIR or ./runs/tune')
    parser.add_argument('--pretrained_dir', default=None,
                        help='Directory holding the pretrained weights. Default: $PETL_PRETRAINED_DIR or ./pretrained_weights')
    parser.add_argument('--data_path', default=None,
                        help='Dataset root. Default: data_path in --default')
    return parser


if __name__ == '__main__':
    main()
