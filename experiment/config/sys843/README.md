# SYS843 configurations

These configs set the SYS843 protocol on processed VTAB-1k. Mai et al.'s configs in `experiment/config/` are unchanged. Run one (method config, task) grid with:

```bash
python main_tune.py --sys843 <name> --data processed_vtab-<task> --data_path <dir>/vtab-1k \
    --pretrained_dir <weights dir> --tune_root <runs root>
```

`--sys843 <name>` takes the `tune` and `lrwd` files listed for `<name>` in `methods.yml`, plus `default_vtab_processed.yml`. An explicit `--default`, `--tune` or `--lrwd` overrides the matching file. The run tunes every grid point on train800/val200 and keeps the best final-epoch validation accuracy. It then retrains that point on train800val200 with seeds 0, 1 and 2, and tests each seed once after the last epoch.

## Protocol (`default_vtab_processed.yml`)

ViT-B/16 supervised on ImageNet-21k. AdamW with a cosine schedule: 100 epochs, 10 warmup epochs, warmup lr 1e-6, minimum lr 1e-5. Batch 64 and drop-path 0.1. All of these are Mai's defaults. `normalized: false`. Validation runs only after the last epoch (`eval_freq` 200 > 100 epochs), and `final_acc_hp: true` selects on that final-epoch accuracy. Every grid point is seeded with the tuning seed (`random_seed: 42`). The final runs use seeds `[0, 1, 2]`.

## Learning-rate x weight-decay grids

| File | lr | wd | Points |
|---|---|---|---:|
| `lr_wd_peft.yml` | 1e-4, 3e-4, 1e-3, 3e-3, 1e-2 | 0, 1e-4, 1e-3, 1e-2 | 20 |
| `lr_wd_full.yml` | 1e-5, 3e-5, 1e-4, 3e-4 | 0, 1e-4, 1e-3, 1e-2 | 16 |

## Method configs

Parameter counts are trainable parameters without the classification head. The head adds 768 x C + C for C classes. `tests/test_sys843_configs.py` checks every count through the code's own count path (`experiment/build_model.get_model`).

| Name (`--sys843`) | Setting | Method config | Trainable params (no head) | Grid points per task |
|---|---|---|---:|---:|
| `lora_r8` | budget and paper | LoRA r=8 on W_q, W_v | 294,912 | 20 |
| `adaptformer_d16` | budget | AdaptFormer bottleneck 16, scale tuned in {0.1, 0.2} | 304,320 | 40 |
| `vpt_deep_p32` | budget | VPT-Deep, 32 prompts per block | 294,912 | 20 |
| `adaptformer_paper` | paper | AdaptFormer bottleneck 64, scale 0.1 | 1,189,632 | 20 |
| `vpt_shallow_paper` | paper | VPT-Shallow, per-task prompt length (table below) | 768 x p | 20 |
| `vpt_deep_paper` | paper | VPT-Deep, per-task prompt length (table below) | 12 x 768 x p | 20 |
| `full` | paper | Full fine-tuning | 85,798,656 | 16 |

Over the 7 configs, each task has 156 grid points to tune, or 2,964 tuning runs over the 19 tasks. Each (config, task) pair then has 3 final runs, or 399 in all.

Sources of the paper settings (each config file carries the citation):

- **LoRA**: Hu et al., arXiv 2106.09685. Sec. 4.2 applies LoRA to W_q and W_v only, and Appendix D, Table 9 uses r_q = r_v = 8 with alpha = 8 for RoBERTa-base, i.e. scale 1, this fork's default. The LoRA paper has no VTAB-1k experiment, and its setting coincides with the budget one, so `lora_r8` serves as both.
- **AdaptFormer**: Chen et al., arXiv 2205.13535, Sec. 4.5, Table 3a/3c. The paper chooses middle dimension 64 and s = 0.1 as its defaults, and AdaptFormer-64 is the setting of Table 6 (supervised IN-21k ViT-B/16). **The paper has no VTAB-1k experiment**, so this is its default image setting, not a VTAB-1k-specific one.
- **VPT**: Jia et al., arXiv 2203.12119, Appendix, Table 13 lists the per-task prompt lengths for VTAB-1k with ViT-B/16, under "Prompt length (p)". The VPT paper used SGD with its own lr grid. SYS843 keeps the common AdamW protocol and grid above for every method.
- **Full fine-tuning**: no method hyperparameter. It uses the lower lr grid.

VPT-Shallow cannot reach the ~0.3M budget (its largest paper setting, p=200, has 153,600 parameters), so it is reported at the paper setting only.

### VPT per-task prompt lengths (VPT paper, Table 13)

| Task | VPT-Shallow p | params | VPT-Deep p | params |
|---|---:|---:|---:|---:|
| cifar | 100 | 76,800 | 10 | 92,160 |
| caltech101 | 5 | 3,840 | 10 | 92,160 |
| dtd | 1 | 768 | 10 | 92,160 |
| oxford_flowers102 | 200 | 153,600 | 1 | 9,216 |
| oxford_iiit_pet | 50 | 38,400 | 1 | 9,216 |
| svhn | 200 | 153,600 | 50 | 460,800 |
| sun397 | 1 | 768 | 5 | 46,080 |
| patch_camelyon | 5 | 3,840 | 100 | 921,600 |
| eurosat | 50 | 38,400 | 100 | 921,600 |
| resisc45 | 50 | 38,400 | 10 | 92,160 |
| diabetic_retinopathy | 10 | 7,680 | 1 | 9,216 |
| clevr_count | 100 | 76,800 | 50 | 460,800 |
| clevr_dist | 200 | 153,600 | 200 | 1,843,200 |
| dmlab | 100 | 76,800 | 100 | 921,600 |
| kitti | 100 | 76,800 | 50 | 460,800 |
| dsprites_loc | 100 | 76,800 | 10 | 92,160 |
| dsprites_ori | 100 | 76,800 | 50 | 460,800 |
| smallnorb_azi | 200 | 153,600 | 200 | 1,843,200 |
| smallnorb_ele | 200 | 153,600 | 200 | 1,843,200 |

Note: the VPT paper prints 79.4 as the mean shallow prompt length of the Natural group, but the seven Natural values above average 79.6. The other five group means match the paper.

## Per-task settings in a method config

A method config may hold a `per_task:` block, `{param: {task: value}}`. `main_tune.py` replaces that block with the value for the `--data` task, and stops with an error if the task is missing.
