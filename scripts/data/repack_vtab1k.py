"""Repack vtab-1k.zip into one uncompressed tarball per task.

    python scripts/data/repack_vtab1k.py <vtab-1k.zip> <out_dir> [--tasks eurosat kitti ...]

The zip holds vtab-1k/<task>/... for the 19 tasks. Each tarball <out_dir>/<task>.tar
holds <task>/... (split files and images), so extracting it into a directory D gives
D/<task>/ and D is the --data_path. The zip is read once, in its own order; every
member goes to its task's tarball. <out_dir>/SHA256SUMS (sha256sum -c format) and
<out_dir>/MANIFEST.tsv (task, files, bytes) are written at the end. The JPEGs are
already compressed, so the tarballs are not.

Standard library only; run on a login node or anywhere with the zip.
"""
import argparse
import hashlib
import os
import sys
import tarfile
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vtab1k_tasks import TASKS, ZIP_ROOT  # noqa: E402


def task_of(name):
    """vtab-1k/<task>/rest -> (task, '<task>/rest'); None for the root and unsafe names."""
    parts = name.split('/')
    if len(parts) < 3 or parts[0] != ZIP_ROOT or parts[1] not in TASKS:
        return None
    if name.startswith('/') or any(p in ('', '.', '..') for p in parts[1:-1]) or parts[-1] in ('.', '..'):
        raise ValueError(f'unsafe member name in zip: {name!r}')
    return parts[1], '/'.join(parts[1:])


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def repack(zip_path, out_dir, tasks):
    os.makedirs(out_dir, exist_ok=True)
    tars, counts, sizes = {}, {t: 0 for t in tasks}, {t: 0 for t in tasks}
    unknown_top = set()
    start = time.time()
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for i, info in enumerate(zf.infolist()):
                parts = info.filename.split('/')
                if len(parts) >= 2 and parts[0] == ZIP_ROOT and parts[1] and parts[1] not in TASKS:
                    unknown_top.add(parts[1])
                mapped = task_of(info.filename)
                if mapped is None or info.is_dir():
                    continue
                task, arcname = mapped
                if task not in tasks:
                    continue
                if task not in tars:
                    tars[task] = tarfile.open(os.path.join(out_dir, task + '.tar.part'), 'w', format=tarfile.PAX_FORMAT)
                tinfo = tarfile.TarInfo(arcname)
                tinfo.size = info.file_size
                tinfo.mtime = time.mktime(info.date_time + (0, 0, -1))
                tinfo.mode = 0o644
                with zf.open(info) as src:
                    tars[task].addfile(tinfo, src)
                counts[task] += 1
                sizes[task] += info.file_size
                if (i + 1) % 20000 == 0:
                    print(f'{i + 1} members read, {time.time() - start:.0f}s', file=sys.stderr)
    finally:
        for tar in tars.values():
            tar.close()
    missing = [t for t in tasks if counts[t] == 0]
    if missing:
        raise SystemExit(f'ERROR: no files for task(s) {missing} under {ZIP_ROOT}/ in {zip_path}')
    if unknown_top:
        print(f'warning: ignored unknown directories under {ZIP_ROOT}/: {sorted(unknown_top)}', file=sys.stderr)
    for task in tasks:
        os.replace(os.path.join(out_dir, task + '.tar.part'), os.path.join(out_dir, task + '.tar'))
    write_sums(out_dir)
    with open(os.path.join(out_dir, 'MANIFEST.tsv'), 'w') as f:
        f.write('task\tfiles\tbytes\n')
        for task in tasks:
            f.write(f'{task}\t{counts[task]}\t{sizes[task]}\n')
    print(f'wrote {len(tasks)} tarball(s) to {out_dir} in {time.time() - start:.0f}s', file=sys.stderr)


def write_sums(out_dir):
    """SHA256SUMS over every <task>.tar in out_dir (covers tarballs from earlier --tasks runs)."""
    lines = []
    for name in sorted(os.listdir(out_dir)):
        if name.endswith('.tar') and name[:-4] in TASKS:
            lines.append(f'{sha256_of(os.path.join(out_dir, name))}  {name}\n')
    with open(os.path.join(out_dir, 'SHA256SUMS'), 'w') as f:
        f.writelines(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('zip')
    parser.add_argument('out_dir')
    parser.add_argument('--tasks', nargs='+', default=list(TASKS), choices=list(TASKS), metavar='TASK',
                        help='Tasks to repack (default: all 19)')
    args = parser.parse_args()
    repack(args.zip, args.out_dir, args.tasks)


if __name__ == '__main__':
    main()
