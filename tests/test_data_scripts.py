"""scripts/data on a tiny synthetic vtab-1k.zip in the real layout (vtab-1k/<task>/...):
download verification (file:// URL), repack into per-task tarballs, staging one task
the way a job does, and the split check. No network, no weights."""
import hashlib
import os
import subprocess
import sys
import zipfile

import pytest

from conftest import REPO
from fake_vtab import make_fake_vtab
from utils.global_var import VTAB_DATASETS

SCRIPTS = os.path.join(REPO, 'scripts', 'data')
sys.path.insert(0, SCRIPTS)
from vtab1k_tasks import TASKS  # noqa: E402

FAKE_TASKS = ['eurosat', 'kitti']
SMALL_COUNTS = ['train800=8', 'val200=4', 'train800val200=12', 'test=4']


def run(cmd, env=None, check=True):
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=dict(os.environ, **(env or {})))
    if check:
        assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return proc


@pytest.fixture(scope='module')
def fake_zip(tmp_path_factory):
    root = tmp_path_factory.mktemp('vtabzip')
    src = root / 'src' / 'vtab-1k'
    for task in FAKE_TASKS:
        make_fake_vtab(str(src), task, num_classes=TASKS[task][0])
    zip_path = root / 'vtab-1k.zip'
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.write(src, 'vtab-1k/')  # directory entries, as in the real zip
        for dirpath, dirnames, filenames in sorted(os.walk(src)):
            dirnames.sort()
            rel_dir = os.path.relpath(dirpath, src.parent)
            if rel_dir != 'vtab-1k':
                zf.write(dirpath, rel_dir + '/')
            for name in sorted(filenames):
                zf.write(os.path.join(dirpath, name), os.path.join(rel_dir, name))
    return zip_path


@pytest.fixture(scope='module')
def tarballs(fake_zip, tmp_path_factory):
    out = tmp_path_factory.mktemp('tars')
    run([sys.executable, os.path.join(SCRIPTS, 'repack_vtab1k.py'), str(fake_zip), str(out), '--tasks', *FAKE_TASKS])
    return out


def test_task_table_matches_code():
    assert {t: c for t, (c, _) in TASKS.items()} == VTAB_DATASETS


def test_repack_writes_one_tarball_per_task(tarballs):
    assert sorted(p.name for p in tarballs.glob('*.tar')) == [f'{t}.tar' for t in FAKE_TASKS]
    assert not list(tarballs.glob('*.part'))
    sums = (tarballs / 'SHA256SUMS').read_text().splitlines()
    assert len(sums) == 2
    for line in sums:
        digest, name = line.split('  ')
        assert hashlib.sha256((tarballs / name).read_bytes()).hexdigest() == digest
    manifest = [line.split('\t')[:2] for line in (tarballs / 'MANIFEST.tsv').read_text().splitlines()[1:]]
    # 16 images (8 train + 4 val + 4 test) + 4 split files per task
    assert manifest == [['eurosat', '20'], ['kitti', '20']]


def test_repack_fails_for_a_task_missing_from_the_zip(fake_zip, tmp_path):
    proc = run([sys.executable, os.path.join(SCRIPTS, 'repack_vtab1k.py'), str(fake_zip), str(tmp_path),
                '--tasks', 'eurosat', 'dtd'], check=False)
    assert proc.returncode != 0 and 'dtd' in proc.stderr


def test_stage_task_extracts_checks_and_prints_data_path(tarballs, tmp_path):
    dest = tmp_path / 'slurm_tmpdir' / 'vtab-1k'
    env = {'PYTHON': sys.executable, 'STAGE_CHECK_ARGS': ' '.join(['--counts'] + SMALL_COUNTS)}
    proc = run(['bash', os.path.join(SCRIPTS, 'stage_task.sh'), str(tarballs), 'eurosat', str(dest)], env=env)
    assert proc.stdout.strip() == str(dest)
    assert sorted(os.listdir(dest)) == ['eurosat']
    for name in ('train800.txt', 'val200.txt', 'train800val200.txt', 'test.txt'):
        assert (dest / 'eurosat' / name).is_file()
    assert len(list((dest / 'eurosat' / 'images').iterdir())) == 16
    assert 'ok   eurosat' in proc.stderr
    # A second call finds the task already staged.
    again = run(['bash', os.path.join(SCRIPTS, 'stage_task.sh'), str(tarballs), 'eurosat', str(dest)], env=env)
    assert again.stdout.strip() == str(dest) and 'already staged' in again.stderr


def test_stage_task_rejects_a_corrupted_tarball(tarballs, tmp_path):
    bad = tmp_path / 'tars'
    bad.mkdir()
    for name in ('kitti.tar', 'SHA256SUMS'):
        (bad / name).write_bytes((tarballs / name).read_bytes())
    with open(bad / 'kitti.tar', 'r+b') as f:
        f.seek(1000)
        f.write(b'X')
    proc = run(['bash', os.path.join(SCRIPTS, 'stage_task.sh'), str(bad), 'kitti', str(tmp_path / 'dest')],
               env={'PYTHON': sys.executable}, check=False)
    assert proc.returncode != 0 and proc.stdout == ''


def test_check_with_real_counts_fails_on_tiny_data(tarballs, tmp_path):
    run(['tar', '-xf', str(tarballs / 'kitti.tar'), '-C', str(tmp_path)])
    proc = run([sys.executable, os.path.join(SCRIPTS, 'check_vtab1k.py'), str(tmp_path)], check=False)
    assert proc.returncode == 1
    assert 'train800.txt: 8 lines, expected 800' in proc.stdout and 'test.txt: 4 lines, expected 711' in proc.stdout


def test_check_catches_split_errors(tarballs, tmp_path):
    run(['tar', '-xf', str(tarballs / 'eurosat.tar'), '-C', str(tmp_path)])
    check = [sys.executable, os.path.join(SCRIPTS, 'check_vtab1k.py'), str(tmp_path), '--counts', *SMALL_COUNTS]
    assert run(check).stdout.splitlines()[-1] == '1/1 task(s) ok'
    task = tmp_path / 'eurosat'
    # train800val200 no longer equal to train800 + val200, a label out of range, a missing image
    tv = (task / 'train800val200.txt').read_text().splitlines()
    tv[0] = tv[0].rsplit(' ', 1)[0] + ' 99'
    (task / 'train800val200.txt').write_text('\n'.join(tv) + '\n')
    os.remove(task / (task / 'test.txt').read_text().split()[0])
    out = run(check, check=False)
    assert out.returncode == 1
    assert 'not train800.txt + val200.txt' in out.stdout
    assert 'label 99 outside [0, 10)' in out.stdout
    assert '1 listed image(s) missing' in out.stdout


def test_download_verifies_size_and_checksum(fake_zip, tmp_path):
    script = os.path.join(SCRIPTS, 'download_vtab1k.sh')
    data = fake_zip.read_bytes()
    good = {'VTAB_URL': fake_zip.as_uri(), 'VTAB_SIZE': str(len(data)),
            'VTAB_SHA256': hashlib.sha256(data).hexdigest()}
    run(['bash', script, str(tmp_path / 'ok')], env=good)
    recorded = (tmp_path / 'ok' / 'vtab-1k.zip.sha256').read_text()
    assert recorded == f'{good["VTAB_SHA256"]}  vtab-1k.zip\n'
    proc = subprocess.run(['sha256sum', '--check', 'vtab-1k.zip.sha256'], cwd=tmp_path / 'ok', capture_output=True)
    assert proc.returncode == 0
    bad_sha = dict(good, VTAB_SHA256='0' * 64)
    assert run(['bash', script, str(tmp_path / 'bad_sha')], env=bad_sha, check=False).returncode == 1
    bad_size = dict(good, VTAB_SIZE=str(len(data) + 1))
    assert run(['bash', script, str(tmp_path / 'bad_size')], env=bad_size, check=False).returncode == 1


def test_weights_download_writes_the_name_the_code_loads(tmp_path):
    src = tmp_path / 'ViT-B_16.npz'
    src.write_bytes(b'not really weights')
    env = {'WEIGHTS_URL': src.as_uri(), 'WEIGHTS_SIZE': str(src.stat().st_size), 'WEIGHTS_SHA256': ''}
    run(['bash', os.path.join(SCRIPTS, 'download_weights.sh'), str(tmp_path / 'w')], env=env)
    assert (tmp_path / 'w' / 'ViT-B_16_in21k.npz').read_bytes() == b'not really weights'
    assert (tmp_path / 'w' / 'ViT-B_16_in21k.npz.sha256').is_file()
