"""
In VPT repo, they handle bias term very carefully when weight decay > 0 for different optimizers.
They also used a special implementation of Adamw from huggingface with weight decay fix. Check their repo's optimizer.py
for additional information.
We skip them for now as AdaptFormer and ConvPass skip this as either

"""

import torch.optim as optim
from typing import Any, Callable, Iterable, List, Tuple, Optional

from utils.setup_logging import get_logger
from utils.misc import get_param

logger = get_logger("PETL_vision")


def effective_lr(params):
    """The optimizer's peak learning rate.

    With `lr_base_batch` set (e.g. 256, the VPT paper's rule, Jia et al. 2022, App. A.1),
    `lr` is a base learning rate scaled linearly with the batch size:
    lr x batch_size / lr_base_batch. Without it (every other config), `lr` is used as is.
    """
    base_batch = get_param(params, 'lr_base_batch', None)
    if base_batch is None:
        return params.lr
    return params.lr * params.batch_size / base_batch


def make_optimizer(tune_parameters, params):
    lr = effective_lr(params)
    if lr != params.lr:
        logger.info(f'lr {params.lr} x batch {params.batch_size} / {params.lr_base_batch} = {lr}')
    if params.optimizer == 'adam':
        optimizer = optim.Adam(
            tune_parameters,
            lr=lr,
            weight_decay=params.wd,
        )

    elif params.optimizer == 'adamw':
        optimizer = optim.AdamW(
            tune_parameters,
            lr=lr,
            weight_decay=params.wd,
        )
    elif params.optimizer == 'sgd':
        optimizer = optim.SGD(
            tune_parameters,
            lr=lr,
            weight_decay=params.wd,
            momentum=params.momentum,
        )
    else:
        raise ValueError(f'unknown optimizer {params.optimizer!r}: choose sgd, adam or adamw')
    return optimizer
