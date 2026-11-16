"""Create a tiny random dataset in the processed VTAB-1k layout.

    <root>/<task>/images/{train800,val200,train800val200,test}/*.jpg
    <root>/<task>/{train800.txt, val200.txt, train800val200.txt, test.txt}

Each split file has one "images/<split>/NNNNNN.jpg label" line per image, as in
the real V-PETL VTAB-1k folders: every split has its own image folder, and the
train800val200 images are byte copies of the train800 then val200 images. Accuracies on this data are meaningless; it only exercises
the code paths.

Usage: python tests/fake_vtab.py <root> [--task eurosat]
"""
import argparse
import io
import os

import numpy as np
from PIL import Image

# Image counts per split (train800 / val200 / test); train800val200 is train + val.
DEFAULT_SIZES = {'train': 8, 'val': 4, 'test': 4}


def make_fake_vtab(root, task='eurosat', num_classes=10, sizes=None, image_size=32, seed=0):
    sizes = dict(DEFAULT_SIZES, **(sizes or {}))
    rng = np.random.default_rng(seed)
    task_dir = os.path.join(root, task)
    folders = {'train': 'train800', 'val': 'val200', 'trainval': 'train800val200', 'test': 'test'}
    images = {}  # split -> [(jpeg bytes, label)]
    for split, count in sizes.items():
        images[split] = []
        for i in range(count):
            pixels = rng.integers(0, 256, size=(image_size, image_size, 3), dtype=np.uint8)
            buf = io.BytesIO()
            Image.fromarray(pixels).save(buf, format='JPEG')
            images[split].append((buf.getvalue(), i % num_classes))
    images['trainval'] = images['train'] + images['val']
    lines = {}
    for split, items in images.items():
        os.makedirs(os.path.join(task_dir, 'images', folders[split]), exist_ok=True)
        lines[split] = []
        for i, (data, label) in enumerate(items):
            rel = f'images/{folders[split]}/{i:06d}.jpg'
            with open(os.path.join(task_dir, rel), 'wb') as f:
                f.write(data)
            lines[split].append(f'{rel} {label}')
    files = {
        'train800.txt': lines['train'],
        'val200.txt': lines['val'],
        'train800val200.txt': lines['trainval'],
        'test.txt': lines['test'],
    }
    for name, content in files.items():
        with open(os.path.join(task_dir, name), 'w') as f:
            f.write('\n'.join(content) + '\n')
    return task_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('root')
    parser.add_argument('--task', default='eurosat')
    args = parser.parse_args()
    print(make_fake_vtab(args.root, args.task))
