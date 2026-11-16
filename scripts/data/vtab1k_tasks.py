"""The 19 VTAB-1k tasks of the processed (NOAH / V-PETL) layout: directory name ->
(number of classes, test-set size). Standard library only, so the data scripts run
on a login node without the training environment.

Classes match utils/global_var.py VTAB_DATASETS; test sizes are those of the VPT
paper (arXiv 2203.12119), Appendix, Table 7. Every task has train800 / val200 /
train800val200 = 800 / 200 / 1000 images.
"""
TASKS = {
    'caltech101': (102, 6084),
    'cifar': (100, 10000),
    'dtd': (47, 1880),
    'oxford_flowers102': (102, 6149),
    'oxford_iiit_pet': (37, 3669),
    'svhn': (10, 26032),
    'sun397': (397, 21750),
    'patch_camelyon': (2, 32768),
    'eurosat': (10, 5400),
    'resisc45': (45, 6300),
    'diabetic_retinopathy': (5, 42670),
    'clevr_count': (8, 15000),
    'clevr_dist': (6, 15000),
    'dmlab': (6, 22735),
    'kitti': (4, 711),
    'dsprites_loc': (16, 73728),
    'dsprites_ori': (16, 73728),
    'smallnorb_azi': (18, 12150),
    'smallnorb_ele': (9, 12150),
}
SPLIT_SIZES = {'train800.txt': 800, 'val200.txt': 200, 'train800val200.txt': 1000}
ZIP_ROOT = 'vtab-1k'
