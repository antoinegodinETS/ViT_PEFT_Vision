"""Check extracted VTAB-1k task directories: split sizes, labels and image files.

    python scripts/data/check_vtab1k.py <data_path> [--tasks eurosat kitti ...]

<data_path> is the directory given to main_tune.py --data_path; it holds <task>/.
Per task:
  - train800.txt, val200.txt, train800val200.txt and test.txt exist, with
    800 / 200 / 1000 lines and the VPT paper's test size (Table 7);
  - train800val200 is exactly train800 + val200 (as a multiset of lines);
  - test shares no image with train800val200;
  - every line is "<relative path> <label>" with 0 <= label < number of classes,
    and every listed image exists (skip with --no-files).
Exits 1 if any task fails. --counts overrides the expected sizes (for tests).
Without --tasks, every task directory present under <data_path> is checked.

Standard library only.
"""
import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vtab1k_tasks import TASKS, SPLIT_SIZES  # noqa: E402


def read_split(path, num_classes, errors):
    entries = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(' ', 1)
            if len(parts) != 2 or not parts[1].lstrip('-').isdigit():
                errors.append(f'{os.path.basename(path)}:{n}: bad line {line!r}')
                continue
            rel, label = parts[0], int(parts[1])
            if not 0 <= label < num_classes:
                errors.append(f'{os.path.basename(path)}:{n}: label {label} outside [0, {num_classes})')
            entries.append((rel, label))
    return entries


def check_task(data_path, task, expected, check_files=True):
    """Return (list of errors, {split file: count})."""
    num_classes = TASKS[task][0]
    task_dir = os.path.join(data_path, task)
    errors, counts, splits = [], {}, {}
    if not os.path.isdir(task_dir):
        return [f'missing directory {task_dir}'], counts
    for name, want in expected.items():
        path = os.path.join(task_dir, name)
        if not os.path.isfile(path):
            errors.append(f'missing {name}')
            continue
        splits[name] = read_split(path, num_classes, errors)
        counts[name] = len(splits[name])
        if counts[name] != want:
            errors.append(f'{name}: {counts[name]} lines, expected {want}')
    if all(n in splits for n in ('train800.txt', 'val200.txt', 'train800val200.txt')):
        if Counter(splits['train800.txt'] + splits['val200.txt']) != Counter(splits['train800val200.txt']):
            errors.append('train800val200.txt is not train800.txt + val200.txt')
    if 'test.txt' in splits and 'train800val200.txt' in splits:
        overlap = {r for r, _ in splits['test.txt']} & {r for r, _ in splits['train800val200.txt']}
        if overlap:
            errors.append(f'{len(overlap)} test image(s) also in train800val200, e.g. {sorted(overlap)[0]}')
    if check_files:
        seen = set()
        missing = []
        for entries in splits.values():
            for rel, _ in entries:
                if rel not in seen:
                    seen.add(rel)
                    if not os.path.isfile(os.path.join(task_dir, rel)):
                        missing.append(rel)
        if missing:
            errors.append(f'{len(missing)} listed image(s) missing, e.g. {missing[0]}')
    return errors, counts


def parse_counts(items):
    """['train800=8', 'val200=4', 'test=4'] -> {'train800.txt': 8, ...}."""
    out = {}
    for item in items or []:
        key, _, value = item.partition('=')
        out[key if key.endswith('.txt') else key + '.txt'] = int(value)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('data_path')
    parser.add_argument('--tasks', nargs='+', choices=list(TASKS), metavar='TASK')
    parser.add_argument('--no-files', action='store_true', help='Do not check that the images exist')
    parser.add_argument('--counts', nargs='+', metavar='SPLIT=N',
                        help='Override expected sizes for all tasks, e.g. train800=8 val200=4 '
                             'train800val200=12 test=4 (for synthetic data)')
    args = parser.parse_args()
    tasks = args.tasks or [t for t in TASKS if os.path.isdir(os.path.join(args.data_path, t))]
    if not tasks:
        print(f'ERROR: no VTAB-1k task directory under {args.data_path}', file=sys.stderr)
        sys.exit(1)
    overrides = parse_counts(args.counts)
    failed = 0
    for task in tasks:
        expected = dict(SPLIT_SIZES, **{'test.txt': TASKS[task][1]})
        expected.update(overrides)
        errors, counts = check_task(args.data_path, task, expected, check_files=not args.no_files)
        summary = ' '.join(f'{k[:-4]}={v}' for k, v in counts.items())
        if errors:
            failed += 1
            print(f'FAIL {task}: {summary}')
            for err in errors:
                print(f'    {err}')
        else:
            print(f'ok   {task}: {summary}')
    print(f'{len(tasks) - failed}/{len(tasks)} task(s) ok')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
