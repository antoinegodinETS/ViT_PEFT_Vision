"""Choice of the grid point that goes to the final runs, from the rows of tune_summary.csv.

Rule (SYS843): the best grid point is the one with the highest validation accuracy (val_top1,
the final-epoch accuracy, with final_acc_hp; val_best_top1 otherwise). Every grid point whose
accuracy is within `tolerance` percentage points of the best is tied with it (0.5 by default:
one image of val200). Among the tied points the lowest lr wins, then the lowest wd, i.e. the
most conservative setting; any remaining tie goes to the first in sorted parameter order.
"""
from collections import OrderedDict

DEFAULT_TIE_TOLERANCE = 0.5
# Columns of tune_summary.csv (see experiment.run.result_tracker).
VAL_TOP1_COL = 5
VAL_BEST_TOP1_COL = 8
# Slack for the rounding of the stored accuracies (2 decimals).
_EPS = 1e-6

RULE = ('highest validation accuracy; grid points within tolerance_pp of it are tied; '
        'among tied points the lowest lr, then the lowest wd')


def _conservative_key(param_set):
    lr = param_set.get('lr', 0.0)
    wd = param_set.get('wd', 0.0)
    rest = sorted((k, repr(v)) for k, v in param_set.items() if k not in ('lr', 'wd'))
    return (lr, wd, rest)


def select_grid_point(scored, tolerance=DEFAULT_TIE_TOLERANCE):
    """scored: list of (param_set dict, accuracy in %). Returns (param_set, tied list)."""
    if not scored:
        raise ValueError('no tuned grid point to select from')
    if tolerance is None or tolerance < 0:
        raise ValueError(f'tie tolerance must be >= 0, got {tolerance!r}')
    best_acc = max(acc for _, acc in scored)
    tied = [(p, acc) for p, acc in scored if acc >= best_acc - tolerance - _EPS]
    tied.sort(key=lambda pa: _conservative_key(pa[0]))
    return tied[0][0], tied


def select_best_tune(tune_rows, final_acc_hp=True, tolerance=DEFAULT_TIE_TOLERANCE):
    """tune_rows: tune_summary.csv rows without the header (lists of strings).

    Returns (best_tune, selection), where selection records the rule, the tolerance and
    every tied grid point, for tune_selection.json and final_result.json.
    """
    col = VAL_TOP1_COL if final_acc_hp else VAL_BEST_TOP1_COL
    scored = [(eval(row[0]), float(row[col])) for row in tune_rows]
    best_tune, tied = select_grid_point(scored, tolerance)
    best_acc = max(acc for _, acc in scored)
    selection = OrderedDict([
        ('rule', RULE),
        ('metric', 'val_top1' if final_acc_hp else 'val_best_top1'),
        ('tolerance_pp', tolerance),
        ('n_grid_points', len(scored)),
        ('best_val_acc', best_acc),
        ('n_tied', len(tied)),
        ('selected', best_tune),
        ('selected_val_acc', tied[0][1]),
        ('tied', [OrderedDict([('param_set', p), ('val_acc', acc)]) for p, acc in tied]),
    ])
    return best_tune, selection
