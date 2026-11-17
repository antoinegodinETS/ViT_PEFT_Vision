# SYS843 configurations

These configs set the SYS843 protocol on processed VTAB-1k. Mai et al.'s configs in `experiment/config/` are unchanged. Run one (method config, task) grid with:

```bash
python main_tune.py --sys843 <name> --data processed_vtab-<task> --data_path <dir>/vtab-1k \
    --pretrained_dir <weights dir> --tune_root <runs root>
```

`--sys843 <name>` takes the `tune` and `lrwd` files listed for `<name>` in `methods.yml`, plus `default_vtab_processed.yml`. An explicit `--default`, `--tune` or `--lrwd` overrides the matching file. The run tunes every grid point on train800/val200 and selects one by its final-epoch validation accuracy, with the tie-break rule below. It then retrains that point on train800val200 with seeds 0, 1 and 2, and tests each seed once after the last epoch.

## Protocol (`default_vtab_processed.yml`)

ViT-B/16 supervised on ImageNet-21k. AdamW with a cosine schedule: 100 epochs, 10 warmup epochs, warmup lr 1e-6, minimum lr 1e-5. Batch 64 and drop-path 0.1. All of these are Mai's defaults. `normalized: false`. Validation runs only after the last epoch (`eval_freq` 200 > 100 epochs), and `final_acc_hp: true` selects on that final-epoch accuracy. Every grid point is seeded with the tuning seed (`random_seed: 42`). The final runs use seeds `[0, 1, 2]`. The tuning seed and the final seeds are disjoint, so no final run repeats a tuning run's random state.

### Grid-point selection and tie-break rule

Validation is val200, so one image is 0.5 percentage points and two grid points a few images apart are within noise. The rule (`experiment/selection.py`), fixed before any run:

1. The best grid point has the highest final-epoch validation accuracy (`val_top1` in `tune_summary.csv`).
2. Every grid point within the tie tolerance of the best is tied with it. The tolerance is 0.5 percentage points (one val200 image), `tie_tolerance` in `default_vtab_processed.yml` or `--tie_tolerance` on the command line.
3. Among the tied points, the lowest lr wins, then the lowest wd: the most conservative setting.

The choice is written to `tune_selection.json` next to `tune_summary.csv` and copied under `selection` in `final_result.json`: the rule, the tolerance, the best validation accuracy, the number of grid points, the number tied (`n_tied`), the tied points with their accuracies, and the selected point (also in `best_tune`).

## Learning-rate x weight-decay grids

| File | lr | wd | Points |
|---|---|---|---:|
| `lr_wd_peft.yml` | 1e-4, 3e-4, 1e-3, 3e-3, 1e-2 | 0, 1e-4, 1e-3, 1e-2 | 20 |
| `lr_wd_full.yml` | 3e-6, 1e-5, 3e-5, 1e-4, 3e-4 | 0, 1e-4, 1e-3, 1e-2 | 20 |
| `lr_wd_vpt_sgd.yml` (SGD check only) | base lr 25, 5, 1, 0.25, 0.05 | 0.01, 1e-3, 1e-4, 0 | 20 |

Every config therefore has 20 grid points per task, and no method hyperparameter is searched.

## Method configs

Parameter counts are trainable parameters without the classification head. The head adds 768 x C + C for C classes. `tests/test_sys843_configs.py` checks every count through the code's own count path (`experiment/build_model.get_model`).

| Name (`--sys843`) | Setting | Method config | Trainable params (no head) | Grid points per task |
|---|---|---|---:|---:|
| `lora_r8` | budget and paper | LoRA r=8 on W_q, W_v | 294,912 | 20 |
| `adaptformer_d16` | budget | AdaptFormer bottleneck 16, scale 0.1 (paper default) | 304,320 | 20 |
| `vpt_deep_p32` | budget | VPT-Deep, 32 prompts per block | 294,912 | 20 |
| `adaptformer_paper` | paper | AdaptFormer's default image-classification setting: bottleneck 64, scale 0.1 | 1,189,632 | 20 |
| `vpt_shallow_paper` | paper | VPT-Shallow, per-task prompt length (table below) | 768 x p | 20 |
| `vpt_deep_paper` | paper | VPT-Deep, per-task prompt length (table below) | 12 x 768 x p | 20 |
| `full` | paper | Full fine-tuning | 85,798,656 | 20 |
| `vpt_deep_p32_sgd` | check | VPT-Deep, 32 prompts per block, VPT paper's SGD recipe | 294,912 | 20 |

The 19-task sweep is the 7 configs without the `check` setting: 7 x 20 = 140 grid points to tune per task, or 2,660 tuning runs over the 19 tasks. Each (config, task) pair then has 3 final runs, or 399 in all. The SGD check adds 20 x 3 = 60 tuning runs and 9 final runs on the 3 pilot tasks. `tests/test_sys843_configs.py` (`test_run_counts`) recomputes these numbers from the configs.

Sources of the paper settings (each config file carries the citation):

- **LoRA**: Hu et al., arXiv 2106.09685. Sec. 4.2 applies LoRA to W_q and W_v only, and Appendix D, Table 9 uses r_q = r_v = 8 with alpha = 8 for RoBERTa-base, i.e. scale 1, this fork's default. The LoRA paper has no VTAB-1k experiment, and its setting coincides with the budget one, so `lora_r8` serves as both.
- **AdaptFormer**: Chen et al., arXiv 2205.13535, Sec. 4.5, Table 3a/3c. The paper chooses middle dimension 64 and s = 0.1 as its defaults, and AdaptFormer-64 is the setting of Table 6 (supervised IN-21k ViT-B/16). **The AdaptFormer paper has no VTAB-1k experiment**, so `adaptformer_paper` is the paper's default image-classification setting, not a VTAB-1k-specific original configuration. The budget config `adaptformer_d16` uses the same default scale s = 0.1, so it has the same 20-point grid as the other PEFT configs.
- **VPT**: Jia et al., arXiv 2203.12119, Appendix, Table 13 lists the per-task prompt lengths for VTAB-1k with ViT-B/16, under "Prompt length (p)". The VPT paper used SGD with its own lr grid. SYS843 keeps the common AdamW protocol and grid above for every method, and checks on the pilot tasks that this does not disadvantage VPT (below).
- **Full fine-tuning**: no method hyperparameter. It uses the lower lr grid, with 20 points like the PEFT configs.

VPT-Shallow cannot reach the ~0.3M budget (its largest paper setting, p=200, has 153,600 parameters), so it is reported at the paper setting only.

## VPT SGD check (`vpt_deep_p32_sgd`, pilot tasks only)

A sanity check that running VPT under the shared AdamW recipe does not disadvantage it. `vpt_deep_p32_sgd` is `vpt_deep_p32` (VPT-Deep, 32 prompts per block) tuned with the VPT paper's recipe (Jia et al., App. A.1 and Table 6): SGD with momentum 0.9, cosine schedule, 10 warmup epochs, 100 epochs, and lr = base_lr x batch_size / 256 (`lr_base_batch: 256`; `engine/optimizer.effective_lr`). The grid is base lr {25, 5, 1, 0.25, 0.05} x wd {0.01, 0.001, 0.0001, 0} (`lr_wd_vpt_sgd.yml`), 20 points; at batch 64 the peak lr is base_lr / 4. Everything else (batch, warmup start, minimum lr, drop-path, data, seeds, selection rule) is the shared protocol. Compare its pilot results with `vpt_deep_p32`.

It runs on the 3 pilot tasks (oxford_flowers102, eurosat, clevr_count) only: its `setting` in `methods.yml` is `check`, so `main_tune.sys843_sweep_methods()` and `sys843_sweep_methods` in `scripts/rorqual/env.sh` leave it out of the sweep, and `scripts/rorqual/run_tasks.sh` skips it on any other task unless `SYS843_CHECK_ANY_TASK=1`. On Rorqual, after `source scripts/rorqual/env.sh`:

```bash
for t in $SYS843_PILOT_TASKS; do
    sbatch --job-name=sys843-tune-vpt_deep_p32_sgd-$t --export=ALL,SYS843_METHOD=vpt_deep_p32_sgd,SYS843_TASK=$t \
        "$SYS843_REPO/scripts/rorqual/tune.sbatch"
done
# when they are done:
sbatch --job-name=sys843-final-vpt_deep_p32_sgd \
    --export=ALL,SYS843_METHOD=vpt_deep_p32_sgd,SYS843_TASKS="$SYS843_PILOT_TASKS" "$SYS843_REPO/scripts/rorqual/final.sbatch"
```

Elsewhere: `python main_tune.py --sys843 vpt_deep_p32_sgd --data processed_vtab-<pilot task> ...` as above.

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
