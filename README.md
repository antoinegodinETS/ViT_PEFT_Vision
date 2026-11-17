## Changes in this fork (SYS843)

This fork of [OSU-MLB/ViT_PEFT_Vision](https://github.com/OSU-MLB/ViT_PEFT_Vision) (Mai et al., CVPR 2025) is used to compare LoRA, AdaptFormer, VPT-Shallow, VPT-Deep and full fine-tuning on VTAB-1k (SYS843 course project, ETS). The original README follows below this section. Changes:

- **LoRA on W_q and W_v only**, with separate A/B matrices per projection (upstream added a delta to q, k and v from a shared A). `--lora_bottleneck r` is still the rank (r=8 gives 294,912 trainable parameters). New `--lora_alpha` (scale alpha/r, default alpha=r, i.e. scale 1 as before) and `--lora_dropout` (default 0.1 as before).
- **VPT-Shallow prompts go through all 12 blocks** and are removed after the last one (upstream removed them after block 0). VPT-Deep is unchanged.
- **No `torch.cuda.empty_cache()` inside the blocks** (it ran up to twice per block per forward pass and distorted timings).
- **Final run** (`main_tune.py` after tuning): each seed trains on `train800val200` and evaluates the test set **once, after the last epoch**; that last-epoch accuracy is the reported metric. There is no per-epoch test evaluation and no best-over-epochs number any more.
- **Seeds**: the final run loops over seeds (`--seeds`, default `0 1 2`). `final_summary.csv` has one row per seed plus `mean` and `std` (sample std) rows; `final_result.json` has `avg_acc`, `std_acc`, `accs`, `per_seed` and mean/std of the time and memory figures.
- **Tuning is re-seeded per configuration**: before each grid point, `main_tune.py` re-seeds Python, NumPy and torch with the tuning seed (`random_seed` in the default config, 42, or `--tune_seed`), so model init, data order and dropout are the same whatever the grid order or which points a resumed run skips (upstream seeded once, so only the first point started from the seed). The seed is in the `random_seed` column of `tune_summary.csv`.
- **Time and memory logging** in `tune_summary.csv` (columns appended after the original ones), `final_summary.csv`, `final_result.json` and `main.py`'s `run_result.json`: total and per-epoch training time (with `torch.cuda.synchronize()` on GPU), peak `max_memory_allocated` / `max_memory_reserved` in MiB (reset at training start; null on CPU), device, GPU name, torch/timm versions, trainable parameters with and without the head.
- **Configurable paths**: `--output_root` (main.py), `--tune_root`, `--data_path` (main_tune.py) and `--pretrained_dir` (both), or the env vars `PETL_OUTPUT_DIR`, `PETL_TUNE_DIR`, `PETL_TUNE_DIR_TEST`, `PETL_PRETRAINED_DIR`. Defaults: `./output`, `./runs/tune` (see below), `./pretrained_weights`.
- **`normalized` is explicit**: default `false` in both `main.py` and `main_tune.py` (as in the published `default_vtab_processed.yml`: resize to 224 and scale to [0, 1], no ImageNet mean/std). Set it with `--normalized {true,false}`; it is recorded in the result files.
- **Lighter dependencies**: TensorFlow/TFDS and CLIP are imported only on the `tfds_vtab-*`, `clevr*` and CLIP paths, the unused `tkinter` import is gone, and the code falls back to the CPU when no GPU is present. `requirements.txt` lists what the processed VTAB-1k path needs and pins `timm==0.9.12` (timm 1.x breaks `forward_features`); torch is not pinned.
- **Upstream's published results are removed.** The `tune_output/` folder (Mai et al.'s `tune_summary.csv` / `final_result.json` files) is deleted from the fork (git history keeps it). `main_tune.py` skips any configuration already listed in `tune_summary.csv` and any final run whose `final_result.json` exists, so those files could silently stand in for our runs. The default tuning root is now `./runs/tune` (test runs: `./runs/tune_test`); `runs/` and `tune_output/` are gitignored. `main_tune.py` still resumes from **our own** earlier results in the tuning root, so start from an empty `--tune_root` for a fresh run.
- **SYS843 configs** in [`experiment/config/sys843/`](experiment/config/sys843/README.md) (Mai's configs are untouched). They hold the protocol default, the lr x wd grids (PEFT 5 x 4, full fine-tuning 5 x 4: 20 points for every config), budget configs at ~0.3M trainable parameters without the head (LoRA r=8, AdaptFormer d=16 s=0.1, VPT-Deep p=32) and original-paper configs with citations (AdaptFormer's default image setting d=64 s=0.1, VPT-Shallow and VPT-Deep with the VPT paper's per-task prompt lengths, full fine-tuning), plus a pilot-only check of VPT-Deep under the VPT paper's SGD recipe (`vpt_deep_p32_sgd`). The grid point is selected with a fixed tie-break rule (within 0.5 pp of the best validation accuracy, lowest lr then lowest wd; `--tie_tolerance`), recorded in `tune_selection.json` and `final_result.json`. SGD runs can scale the lr with the batch size (`lr_base_batch`). Run one with `python main_tune.py --sys843 <name> --data processed_vtab-<task> ...`. A method config may give per-task values in a `per_task:` block. The linked README lists every config, its parameter count and its grid size.
- **Stages**: `main_tune.py --stage tune` stops after the search, and `--stage final` runs only the final seeds (it fails if a grid point is missing). The default `all` does both, as before. Tuning and final runs can therefore use different GPUs. `--num_workers` sets the DataLoader workers (default 4, as before).
- **Data and weights scripts** in [`scripts/data/`](scripts/data/README.md). They download V-PETL's `vtab-1k.zip` from Hugging Face and check its size and sha256, repack it into 19 per-task tarballs, download `ViT-B_16.npz` under the name the code loads, stage one task into a job's local disk, and check the split files (800 / 200 / 1000 / VPT test sizes).
- **Rorqual job scripts** in `scripts/rorqual/`; see [Running on Rorqual](#running-on-rorqual) below.

To run one seed per job, pass `--seeds <s> --final_output_name final_result_seed<s>.json` (the summary becomes `final_result_seed<s>_summary.csv`).

### CPU smoke test

```bash
pip install -r requirements.txt   # CPU torch is enough
# Real IN-21k weights (413 MB), kept out of git:
mkdir -p pretrained_weights
curl -L -o pretrained_weights/ViT-B_16_in21k.npz https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz
python -m pytest tests -q        # about 3 min on a laptop CPU
```

`tests/test_model_units.py` needs no weights (random init). It checks the trainable parameter counts (LoRA r=8: 294,912; AdaptFormer d=16: 304,320; VPT-Deep p=32: 294,912; VPT-Shallow p=10: 7,680; full: 85,798,656), that LoRA leaves W_k untouched, and that VPT-Shallow prompts reach the last block. `tests/test_sys843_configs.py` loads every SYS843 config through `main_tune.py`. It checks the parameter counts (VPT paper settings for all 19 tasks), the grid sizes, the run counts, the protocol settings and that `run_tasks.sh` keeps the check config to the pilot tasks. `tests/test_selection.py` checks the tie-break rule and the SGD lr scaling. `tests/test_data_scripts.py` runs the data scripts on a tiny synthetic `vtab-1k.zip`. `tests/test_no_bundled_results.py` checks that no result files are tracked.

`tests/test_smoke_pipeline.py` builds a tiny random dataset in the processed VTAB-1k layout (`tests/fake_vtab.py`). It runs one epoch of each method through `main_tune.py`, then a tune->final run with two configs and two seeds, and checks the result files. It also checks:
- that a grid point gives the same numbers whether it runs first or second;
- that `--stage tune` and `--stage final` work in turn;
- that `--sys843` works end to end with a per-task config and with the SGD check config.

It uses `pretrained_weights/ViT-B_16_in21k.npz` or the file named by `PETL_TEST_WEIGHTS`, and is skipped if neither exists. On a GPU (`check_env.sbatch`) the same tests expect CUDA in the result files.

### Running on Rorqual

The scripts in `scripts/rorqual/` follow the pattern of the SYS819 project's Rorqual scripts. Rorqual has H100 80 GB nodes with 16 CPU cores per GPU. Compute nodes have no internet, so every download happens on a login node. Persistent files go under `~/links/projects/def-chdesa/$USER/sys843/`; scratch is purged after 60 days.

| File | What it does |
|---|---|
| `env.sh` | The one place the account is set (`SBATCH_ACCOUNT=def-chdesa`). It also sets the paths: `$SYS843_DATA_STORE` (zip), `$SYS843_TARS` (per-task tarballs), `$SYS843_WEIGHTS` and `$SYS843_RUNS` (results, outside the repository). Helpers: `sys843_activate` (module and venv), `sys843_stage_task` (extract one task to `$SLURM_TMPDIR`) and `sys843_run_resumable` (on Slurm's USR1 warning, stop and resubmit after this job), `sys843_sweep_methods` (the sweep configs, without the `check` ones) and `$SYS843_PILOT_TASKS`. Source it before every `sbatch`. |
| `setup_env.sh` | Login node. Creates `~/venvs/sys843-vit-peft` with `module load StdEnv/2023 python/3.11`, `virtualenv --no-download` and wheelhouse installs (`pip --no-index`). `timm==0.9.12` is not in the wheelhouse and comes from PyPI as a prebuilt wheel (`--only-binary=:all:`, so nothing is compiled on the login node). It then verifies the imports. Safe to rerun on an existing venv. |
| `check_env.sbatch` | One full H100, 30 min. Runs `nvidia-smi`, prints the versions, checks the weights and tarball checksums, stages EuroSAT, and runs the test suite on the GPU. |
| `tune.sbatch` | One (method config, task) search with `--stage tune`, 12 h limit, resumable. Runs on a **3g.40gb** MIG slice by default (see below). |
| `final.sbatch` | The 3 seeds of one method config on one or more tasks (`SYS843_TASKS`) with `--stage final`, on a full H100, 4 h limit, resumable. The only source of the time and memory numbers in the report. |
| `run_tasks.sh` | Shared by both job scripts: for each task, stage the data, run `main_tune.py`, then free the local disk. A `check` config runs on the pilot tasks only (`SYS843_CHECK_ANY_TASK=1` overrides). |

**Why 3g.40gb and not 1g.10gb for the search.** The code trains in fp32 at batch 64. The VPT paper (Table 12, A100, batch 64) measures 9.8 to 10.3 GB peak training memory for adapters and VPT with p=1, 11.7 GB for full fine-tuning and 25.8 GB for VPT with p=200. A 1g.10gb slice has 10 GB. After the pilot, `peak_memory_reserved_mib` in `tune_summary.csv` shows which configs fit the small slice; pass `--gpus=nvidia_h100_80gb_hbm3_1g.10gb:1 --cpus-per-task=2` to `sbatch` for those.

Workflow, from a login node in the clone (`~/sys843-vit-peft`):

```bash
# 1. Environment and data (once)
bash scripts/rorqual/setup_env.sh
source scripts/rorqual/env.sh
scripts/data/download_vtab1k.sh "$SYS843_DATA_STORE"
sys843_activate && python scripts/data/repack_vtab1k.py "$SYS843_DATA_STORE/vtab-1k.zip" "$SYS843_TARS"
scripts/data/download_weights.sh "$SYS843_WEIGHTS"
sbatch "$SYS843_REPO/scripts/rorqual/check_env.sbatch"        # must end with RESULT ok

# 2. Pilot: every method config on Flowers-102, EuroSAT and CLEVR-count
METHODS="$(sys843_sweep_methods)"   # the 7 sweep configs; the check config vpt_deep_p32_sgd is left out
for m in $METHODS; do for t in $SYS843_PILOT_TASKS; do
    sbatch --job-name=sys843-tune-$m-$t --export=ALL,SYS843_METHOD=$m,SYS843_TASK=$t "$SYS843_REPO/scripts/rorqual/tune.sbatch"
done; done
# when they are done:
for m in $METHODS; do
    sbatch --job-name=sys843-final-$m --export=ALL,SYS843_METHOD=$m,SYS843_TASKS="$SYS843_PILOT_TASKS" \
        "$SYS843_REPO/scripts/rorqual/final.sbatch"
done
# Pilot only: the VPT SGD check (vpt_deep_p32_sgd), see experiment/config/sys843/README.md.
# Check train_time_s and peak_memory_reserved_mib in the tune_summary.csv / final_result.json
# files under $SYS843_RUNS/sys843_vtab/, then adjust --time or the MIG slice before the sweep.

# 3. Sweep: the other 16 tasks, same two steps (finished pilot tasks are skipped anyway)
TASKS="caltech101 cifar dtd oxford_iiit_pet svhn sun397 patch_camelyon resisc45 diabetic_retinopathy \
       clevr_dist dmlab kitti dsprites_loc dsprites_ori smallnorb_azi smallnorb_ele"
for m in $METHODS; do for t in $TASKS; do
    sbatch --job-name=sys843-tune-$m-$t --export=ALL,SYS843_METHOD=$m,SYS843_TASK=$t "$SYS843_REPO/scripts/rorqual/tune.sbatch"
done; done
# then final.sbatch per method with 4-6 tasks per job (jobs should last about 1 h or more)
```

Each job writes `<job-name>-<jobid>.out` in the submit directory. A job that reaches its time limit resubmits itself to run after it, up to `SYS843_MAX_CHUNKS` (default 5) jobs. Finished grid points and finished final runs are skipped. Never run two jobs on the same (method, task) at once: both would append to its `tune_summary.csv`. These scripts have only been checked with shellcheck; they have not run on Rorqual yet.

----------------------------------------------------



<h1 style="font-size: 100 pt;" align=center><strong>Lessons and Insights from a Unifying Study of Parameter-Efficient Fine-Tuning (PEFT) in Visual Recognition</strong></h1>
<h2 align=center style="color:blue;font-size: 120 pt"> CVPR'25 (Highlight) </h2>
<div align="center">

:book: [Paper](https://arxiv.org/abs/2409.16434) | :clipboard: [Project page](https://zheda-mai.github.io/PEFT_Vision_CVPR25/)

</div>

----------------------------------------------------
 
Parameter-efficient fine tuning (PEFT) has attracted significant attention lately due to the increasing size of pre-trained models and the need to fine-tune them for superior downstream performance. This community-wide enthusiasm has
sparked a plethora of approaches. We provide a systematic and comprehensive code base implementing 16 PEFT methods, which serves as a valuable resource for researchers and enables **consistent and reproducible evaluations** of PEFT methods. We conduct a unifying empirical study of these representative PEFT approaches in **various scenarios**, including low shots, many shots, different domain gaps, and robustness between in-distribution and OOD. Our findings offer several insightful directions for future research, including **leveraging prediction differences** in other learning paradigms such as semi-supervised learning and domain adaptation, **robust fine-tuning with PEFT**, and  providing **empirical evidence for PEFT mechanism understanding**.
More details can be found in [our paper](https://arxiv.org/pdf/2409.16434). 


<h2 style="font-size: 80 pt;" align=center>Highlights of Insights</h2>
<div align="center">
<img src="main.png" width="100%" height="100%">
</div>


----------------------------------------------------

This code base contains the following features:
1.  [Evaluate a PEFT method on one dataset with selected hyper-parameters](#evaluate-a-peft-method-on-one-dataset-in-vtab-1k)
2. [Run hyper-parameter tuning for PEFT methods](#run-hyper-parameter-tuning-for-vtab-1k)
3. [Evaluate PEFT methods' robustness to domain shift ](#evaluate-robustness-of-peft-methods-to-domain-shift)
4. [Evaluate PEFT methods on Many-shots (full) Datasets](#evaluate-peft-methods-on-many-shots-datasets)

You can extend this code base to include:
1. [New datasets](#to-add-a-new-dataset)
2. [New backbones](#to-add-a-new-backbone) 
3. [New methods](#to-add-a-new-method) 

 

# Environment Setup  
```bash  
source env_setup.sh
```  
  
# Data Preparation
You can put all the data in a folder and pass the path to `--data_path` argument.  

## VTAB-1k  
We provide two ways to prepare VTAB-1k dataset  
 
1. Processed Version  
   - Download the processed version from [here](https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/mai_145_buckeyemail_osu_edu/ERnzBDVaA6BBpTCoBV0RbVIBF7JEZlIwlGyPdg4AhS_y4g?e=Kjc5uo).  
   - When you pass the data name for `--data` argument, add `process_vtab`, e.g. `process_vtab-cifar`  
 - You can find all the dataset names of the processed version VTAB1-k from `VTAB_DATASETS` in `utils/global_var.py`  
  2. TFDS  
   - Following [VPT](https://github.com/KMnP/vpt/blob/main/VTAB_SETUP.md)'s instruction to download the data through TFDS  
   - When you pass the data name for `--data` argument, add `tfds_vtab` before each VTAB dataset,  e.g. `tfds_vtab-cifar(num_classes=100)`  
   - You can find all the dataset names of the TFDS version VTAB1-k from `TFDS_DATASETS` in `utils/global_var.py`  
  
Both ways apply the same transforms to the dataset: Resize, ToTensor and optionally Normalize with mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225].  
  
## Robustness to Domain Shift  
 - Download ImageNet from the [official website](https://www.image-net.org/download.php)  
 - When you pass the data name for `--data` argument, use `imagenet-fs`  
 - Download ImageNet-Sketch from https://github.com/HaohanWang/ImageNet-Sketch  
 - Download ImageNet-A from https://github.com/hendrycks/natural-adv-examples  
 - Download ImageNet-R from https://github.com/hendrycks/imagenet-r  
 - Download ImageNetV2 from https://github.com/modestyachts/ImageNetV2.  
  
## CIFAR100  
Download when you run the code  

## To add a new dataset
1. add a new dataset file in `/data/dataset`.
2. modify [build_loader.py](experiment/build_loader.py) to include the new dataset.

# Pretrain weights
Download the pretrained weights from the following links and put them in the `pretrained_weights` folder.  
1. [ViT-B-Sup-21k](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz) rename it as `ViT-B_16_in21k.npz`  
2. [ViT-B-CLIP](https://huggingface.co/timm/vit_base_patch16_clip_224.openai/tree/main) rename it as `ViT-B_16_clip.bin`  
3. [ViT-B-DINOV2](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth) rename it as `ViT-B_14_dinov2.pth`  
  
## To add a new backbone
- modify `get_base_model()` in [build_model.py](experiment/build_model.py).
  
# Quick Start
  
## Evaluate a PEFT method on one dataset in VTAB-1K
This section shows example commands to run 21 PEFT methods and baselines on the DTD dataset in VTAB-1K. More argument options can be found in main.py. 

**SSF**

    CUDA_VISIBLE_DEVICES=0  python main.py --ssf --data processed_vtab-dtd 

**VPT-Deep**

    CUDA_VISIBLE_DEVICES=0  python main.py --vpt_mode deep --vpt_num 10 --data processed_vtab-dtd --data_path data_folder/vtab_processed  

**VPT-Shallow**

    CUDA_VISIBLE_DEVICES=0  python main.py --vpt_mode shallow --vpt_num 10 --data processed_vtab-dtd --data_path data_folder/vtab_processed  

**AdaptFormer**  

    CUDA_VISIBLE_DEVICES=0  python main.py --ft_mlp_module adapter --ft_mlp_mode parallel --ft_mlp_ln before --adapter_bottleneck 64 --adapter_init lora_kaiming --adapter_scaler 0.1 --data processed_vtab-dtd  

**Convpass**  

     CUDA_VISIBLE_DEVICES=0  python main.py --ft_attn_module convpass --ft_attn_mode parallel --ft_attn_ln after --ft_mlp_module convpass --ft_mlp_mode parallel --ft_mlp_ln after --convpass_scaler 0.1 --data processed_vtab-dtd  --debug --optimizer adamw --data_path data_folder/vtab_processed  

  
**Adapter - VPT Version (Zero-init)**

    CUDA_VISIBLE_DEVICES=0  python main.py --ft_mlp_module adapter --ft_mlp_mode sequential_after --adapter_bottleneck 8 --adapter_init zero --adapter_scaler 1 --data processed_vtab-dtd  --debug --data_path data_folder/vtab_processed  

  
**Pfeiffer Adapter - LoRA init** 

    CUDA_VISIBLE_DEVICES=0  python main.py --ft_mlp_module adapter --ft_mlp_mode sequential_after --adapter_bottleneck 8 --adapter_init lora_xavier --adapter_scaler 1 --data processed_vtab-dtd  --debug --data_path data_folder/vtab_processed  

  
**Pfeiffer Adapter  - Random init** 

    CUDA_VISIBLE_DEVICES=0  python main.py --ft_mlp_module adapter --ft_mlp_mode sequential_after --adapter_bottleneck 8 --adapter_init xavier --adapter_scaler 1 --data processed_vtab-dtd  --debug --data_path data_folder/vtab_processed  

  
**Houlsby Adapter  - Random init** 

    CUDA_VISIBLE_DEVICES=0  python main.py --ft_attn_module adapter --ft_attn_mode sequential_after --ft_mlp_module adapter --ft_mlp_mode sequential_after --adapter_bottleneck 8 --adapter_init xavier --adapter_scaler 1 --data processed_vtab-dtd  --debug --data_path data_folder/vtab_processed  

  
**LoRA**  

    CUDA_VISIBLE_DEVICES=0  python main.py --lora_bottleneck 8 --data processed_vtab-dtd  --debug --data_path data_folder/vtab_processed  

  
**FacT-TK**  

    CUDA_VISIBLE_DEVICES=0  python main.py --data processed_vtab-dtd  --optimizer adamw --fact_type tk --drop_path_rate 0.1 --debug --fact_dim 32 --fact_scaler 0.1 --data_path data_folder/vtab_processed  

  
**Fact-TT**  

    CUDA_VISIBLE_DEVICES=0  python main.py --data processed_vtab-dtd  --optimizer adamw --fact_type tt --drop_path_rate 0.1 --data_path data_folder/vtab_processed  

  
**ReAdapter**  

    CUDA_VISIBLE_DEVICES=0  python main.py --ft_attn_module repadapter --ft_attn_mode sequential_before --ft_mlp_module repadapter --ft_mlp_mode sequential_before --repadapter_bottleneck 8 --repadapter_scaler 10 --data processed_vtab-smallnorb_azi  --optimizer adamw --debug  

  
**BitFit**  

    CUDA_VISIBLE_DEVICES=0  python main.py --bitfit --data processed_vtab-clevr_count  --debug --data_path data_folder/vtab_processed  

  
**VQT**  

    CUDA_VISIBLE_DEVICES=0  python main.py --vqt_num 10 --data_path data_folder/vtab_processed  --data processed_vtab-dtd 

  Run
**LayerNorm  Tuning**

    CUDA_VISIBLE_DEVICES=0  python main.py --ln --data processed_vtab-dtd   --debug --data_path data_folder/vtab_processed  

  
**DiffFit**  

    CUDA_VISIBLE_DEVICES=0  python main.py --difffit --data processed_vtab-dtd   --debug --data_path data_folder/vtab_processed  

  
**Selective Attention  Tuning**

    CUDA_VISIBLE_DEVICES=0  python main.py --attention_index 9 10 11 --attention_type qkv --data_path data_folder/vtab_processed  --data processed_vtab-dtd 

**Selective MLP Tuning**  

    CUDA_VISIBLE_DEVICES=0  python main.py --mlp_index 9 10 11 --mlp_type full --data_path data_folder/vtab_processed  --data processed_vtab-dtd 

  
**Selective Block Tuning**  

    CUDA_VISIBLE_DEVICES=0  python main.py --block_index 9 10 11 --data_path data_folder/vtab_processed  --data processed_vtab-dtd 

  
**Full Tuning**  

    CUDA_VISIBLE_DEVICES=0  python main.py --full --eval_freq 2 --data_path data_folder/vtab_processed  --data processed_vtab-dtd 


## To add a new method  
1. add a new module file in `./model/`.   
2. add the module accordingly in [block.py](model/block.py), [mlp.py](model/mlp.py), [patch_embed.py](model/patch_embed.py), [vision_transformer.py](model/vision_transformer.py), [attention.py](model/attention.py).
3. add the name of the added module in TUNE_MODULES and modify `get_model()` in `./experiment/build_model.py` accordingly.
4. add required arguments in `main.py`  

## Run Hyper-parameter Tuning for VTAB-1K

This an example command to run hyper-parameter tuning for Caltech101 in VTAB-1K. 
```bash  
CUDA_VISIBLE_DEVICES=0 python main_tune.py --data processed_vtab-caltech101 --default experiment/config/default_vtab_processed.yml --tune experiment/config/method/lora.yml --lrwd experiment/config/lr_wd_vtab_processed.yml  
```  
There are three crucial arguments: 
- The `default` config file is used to set the default hyperparameters for the training.  
- The `tune` config file is used to set the tunable hyperparameters for the method.  
- The `lrwd` config file is used to set the learning rate and weight decay for the training.  

All the config files can be found in the `experiment/config` folder.   

Upstream provides its tuning results `tune_summary.csv` and final run results `final_result.json` in a `tune_output` folder; **this fork removed it** (see the SYS843 section above), and new runs go to `./runs/tune`. 

If you want to rerun the final run using the best tuning parameters, you can specify a new output name for the final run using this option `--final_output_name`. 


To tune all methods for one VTAB-1K dataset, here is an example command:  
```bash
dataset='processed_vtab-dsprites_loc'
for METHOD in lora_p_adapter repadapter rand_h_adapter adaptformer convpass fact_tk fact_tt lora difffit full linear ssf bitfit ln vpt_shallow vpt_deep
  do
    CUDA_VISIBLE_DEVICES=0 python main_tune.py --data ${dataset}  --default experiment/config/default_vtab_processed.yml --tune experiment/config/method/$METHOD.yml --lrwd experiment/config/lr_wd_vtab_processed.yml
  done
```


## Evaluate robustness of PEFT methods to domain shift  
We use the CLIP ViT-B/16 model and add an FC layer as the prediction head with zero-initialized bias and initialize weights using the class label text embedded by the text encoder. Subsequently, we discard the text encoder and apply PEFT methods to the visual encoder, fine-tuning only the PEFT modules and the head.

The code to generate the prediction head for CLIP can be found at [build_clip_zs_classifier.py](experiment/build_clip_zs_classifier.py).  

**Fine-tuning CLIP on downstream data.** 

This is an example command to run a PEFT method (LoRA with dimension 32) for the 100-shot ImageNet dataset:
```bash
CUDA_VISIBLE_DEVICES=0  python main.py --data fs-imagenet --data_path data_folder/imagenet/images --warmup_lr_init 1.0e-7 --lr 0.00003 --wd 0.005 --eval_freq 1 --store_ckp --lora_bottleneck 32  --batch_size 256 --final_acc_hp --early_patience 10
```
This is an example command to tune a few PEFT methods (Adaptre with dimension 64, LoRA with dimension 16, SSF) for the 100-shot ImageNet dataset:
```bash
for METHOD in rand_h_adapter_64 lora_16 ssf
do
  for dataset in fs-imagenet
    do
      CUDA_VISIBLE_DEVICES=0 python main_tune.py --data ${dataset} --test_bs 2048 --bs 256  --default experiment/config/clip_fs_imagenet.yml --tune experiment/config/method-imagenet/$METHOD.yml --lrwd experiment/config/lr_wd_clip_imagenet.yml
    done
done
```
**Evaluation fine-tuned CLIP on OOD data.**

To evaluate the performance of fine-tuned models on OOD data, here is an example command:
```bash  
for METHOD in rand_h_adapter_64 lora_16 ssf
do
  for dataset in fs-imagenet eval_imagenet-v2 eval_imagenet-r eval_imagenet-a eval_imagenet-s
    do
      CUDA_VISIBLE_DEVICES=0 python main_evaluate.py --test_data ${dataset} --bs 2048 --default experiment/config/clip_fs_imagenet.yml --tune experiment/config/method-imagenet/$METHOD.yml --data_path /research/nfs_chao_209/zheda
    done
done
```  

**Weight-Space Ensembles (WiSE) for PEFT.**

When it comes to applying WiSE to PEFT methods, there are two types of PEFT methods. 
- Methods that insert additional parameters to the model, such as Adapter.
-  Methods that directly fine-tune existing parameters, such as BitFit. 

For the former, we use `merge_petl.py` and use `merge_model.py` for the latter. 

Example commands for each type:
```bash
CUDA_VISIBLE_DEVICES=0 python merge_model.py --bs 1024  --default experiment/config/clip_fs_imagenet.yml --tune experiment/config/method-imagenet/ln.yml
CUDA_VISIBLE_DEVICES=0 python merge_petl.py --bs 1024 --default experiment/config/clip_fs_imagenet.yml --tune experiment/config/method-imagenet/fact_tk_64.yml
```

To get the WiSE curve plots, you can use `WiSE_PETL.ipynb`.

##  Evaluate PEFT methods on Many-shots Datasets
We use three datasets for mann-shot experiments: CIFAR100, Clevr-distance and RESISC.

Here is an example command to run the PEFT methods for the Clevr-distance dataset. Config files for CIFAR and RESISC can be found in the `experiment/config` folder.
```bash
for METHOD in rand_h_adapter_8 lora_8 fact_tk_8 
do
  for dataset in clevr
    do
      CUDA_VISIBLE_DEVICES=0 python main_tune.py --data ${dataset}  --default experiment/config/default_clevr.yml --tune experiment/config/method_clevr/$METHOD.yml --lrwd experiment/config/lr_wd_clevr.yml
    done
done
```
## Results
Upstream stores its tuning results (`tune_summary.csv`) and final run results (`final_result.json`) in a `tune_output` folder; this fork removed that folder, and its own runs write to `./runs/tune` (or `--tune_root`).


## To run new hyperparameter tuning
1. add a general config file and a learning rate and weight decay config file in `./experiment/config/`.
2. add new method config files. You can create a new folder in `./experiment/config/` for your experiment.

## To collect logits and ground truth for PEFT methods
Use [main_collect_prediction.py](main_collect_prediction.py). 

## Citation 

If you use this paper/code in your research, please consider citing us:

```
@article{mai2024lessons,
  title={Lessons Learned from a Unifying Empirical Study of Parameter-Efficient Transfer Learning (PETL) in Visual Recognition},
  author={Mai, Zheda and Zhang, Ping and Tu, Cheng-Hao and Chen, Hong-You and Zhang, Li and Chao, Wei-Lun},
  journal={arXiv preprint arXiv:2409.16434},
  year={2024}
}
```


## References:
- [SSF](https://github.com/dongzelian/SSF)  
- [AdaptFormer](https://github.com/ShoufaChen/AdaptFormer)  
- [ConvPass](https://github.com/JieShibo/PETL-ViT)  
- [LoRA](https://github.com/ZhangYuanhan-AI/NOAH)  
- [Adapter (GitHub)](https://github.com/ZhangYuanhan-AI/NOAH), [Adapter (Pfeiffer et al., 2020)](https://arxiv.org/pdf/2007.07779.pdf), [Adapter (Houlsby et al., 2019)](https://arxiv.org/pdf/1902.00751.pdf)  
- [VPT](https://github.com/KMnP/vpt)  
- [FacT](https://github.com/JieShibo/PETL-ViT)  
- [RepAdapter](https://github.com/luogen1996/RepAdapter)  
- [VQT](https://openaccess.thecvf.com/content/CVPR2023/papers/Tu_Visual_Query_Tuning_Towards_Effective_Usage_of_Intermediate_Representations_for_CVPR_2023_paper.pdf)  
- [BitFit](https://arxiv.org/pdf/2106.10199.pdf)  
- [DiffFit](https://arxiv.org/pdf/2304.06648.pdf)  


