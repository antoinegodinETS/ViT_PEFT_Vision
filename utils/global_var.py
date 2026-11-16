import os
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, \
    OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

TFDS_DATASETS = {
    'caltech101': 102,
    'cifar(num_classes=100)': 100,
    'dtd': 47,
    'oxford_flowers102': 102,
    'oxford_iiit_pet': 37,
    'patch_camelyon': 2,
    'sun397': 397,
    'svhn': 10,
    'resisc45': 45,
    'eurosat': 10,
    'dmlab': 6,
    'kitti(task="closest_vehicle_distance")': 4,
    'smallnorb(predicted_attribute="label_azimuth")': 18,
    'smallnorb(predicted_attribute="label_elevation")': 9,
    'dsprites(predicted_attribute="label_x_position",num_classes=16)': 16,
    'dsprites(predicted_attribute="label_orientation",num_classes=16)': 16,
    'clevr(task="closest_object_distance")': 6,
    'clevr(task="count_all")': 8,
    'diabetic_retinopathy(config="btgraham-300")': 5
}

VTAB_DATASETS = {'caltech101': 102, 'clevr_count': 8, 'dmlab': 6, 'dsprites_ori': 16, 'eurosat': 10, 'oxford_flowers102': 102,
                 'patch_camelyon': 2,
                 'smallnorb_azi': 18, 'svhn': 10, 'cifar': 100, 'clevr_dist': 6, 'dsprites_loc': 16, 'dtd': 47,
                 'kitti': 4, 'oxford_iiit_pet': 37, 'resisc45': 45,
                 'smallnorb_ele': 9, 'sun397': 397, 'diabetic_retinopathy': 5}

MEAN_STD_DICT = {
    'vit_base_patch16_224_in21k': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    'vit_base_patch14_dinov2': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    'vit_base_mae': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    'vit_base_patch16_clip_224': (OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)
}

CIFAR_VAL_SPLIT = {
    'cifar100-500': 0.2,
    'cifar100-1k': 0.2,
    'cifar100-5k': 0.1,
    'cifar100-10k': 0.1,
    'cifar100-full': 0.1
}

RETINOPATHY_VAL_SPLIT = {
    'retinopathy-500': 0.2,
    'retinopathy-5k': 0.1,
    'retinopathy-10k': 0.1,
    'retinopathy-full': 0.1
}

RESISC_VAL_SPLIT = {
    'resisc-225': 0.2,
    'resisc-450': 0.2,
    'resisc-900': 0.1,
    'resisc-2250': 0.1,
    'resisc-4500': 0.1,
    'resisc-9000': 0.1,
    'resisc-full': 0.1

}


# Output roots and the pretrained-weights directory. Precedence: CLI argument
# (--output_root / --tune_root / --pretrained_dir), then these environment
# variables, then the defaults below. The tuning roots live under ./runs
# (gitignored, starts empty): main_tune.py skips any run whose tune_summary.csv /
# final_result.json already exist, so it must never point at published results
# (upstream's ./tune_output, removed from this fork).
OUTPUT_DIR = os.environ.get("PETL_OUTPUT_DIR", "./output")
TUNE_DIR = os.environ.get("PETL_TUNE_DIR", "./runs/tune")
TUNE_DIR_TEST = os.environ.get("PETL_TUNE_DIR_TEST", "./runs/tune_test")
PRETRAINED_DIR = os.environ.get("PETL_PRETRAINED_DIR", "pretrained_weights")

# Seeds of the final runs (retrain on train800val200, test once), unless
# overridden by --seeds or a final_seeds key in the YAML config.
DEFAULT_FINAL_SEEDS = [0, 1, 2]
