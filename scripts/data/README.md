# Data and weights preparation

Run the downloads once on a machine with internet, such as a cluster login node. Keep the results on persistent project storage. Compute jobs then only extract one task's tarball to node-local disk. The Python scripts need only the standard library. The shell scripts need `bash`, `curl`, `sha256sum` and `tar`.

| Script | Where | What it does |
|---|---|---|
| `download_vtab1k.sh <dir>` | login node | Downloads `vtab-1k.zip` (V-PETL's processed VTAB-1k on Hugging Face, `datasets/XiN0919/VTAB-1k`) into `<dir>`. Resumes a partial download. Checks the size (4,659,851,071 bytes) and the sha256 that Hugging Face publishes for the file (`87edd008…7726`). Writes `vtab-1k.zip.sha256`. |
| `repack_vtab1k.py <zip> <tar_dir>` | login node / any CPU | Reads the zip once and writes 19 uncompressed tarballs `<tar_dir>/<task>.tar`, each holding `<task>/…`. Also writes `SHA256SUMS` and `MANIFEST.tsv` (files and bytes per task). `--tasks` repacks a subset. |
| `check_vtab1k.py <data_path>` | anywhere | For each task under `<data_path>`, checks that train800 / val200 / train800val200 have 800 / 200 / 1000 lines and that test has the VPT paper's size (Table 7). Also checks that train800val200 = train800 + val200, that test does not overlap trainval, that labels are in range, and that every listed image exists. Exits 1 on any failure. |
| `download_weights.sh [<dir>]` | login node | Downloads the IN-21k `ViT-B_16.npz` (413 MB) to `<dir>/ViT-B_16_in21k.npz`, the name the code loads. `<dir>` defaults to `$PETL_PRETRAINED_DIR`, else `./pretrained_weights`; pass the same directory as `--pretrained_dir`. Checks the size and sha256 and writes `ViT-B_16_in21k.npz.sha256`. |
| `stage_task.sh <tar_dir> <task> <dest>` | inside a job | Checks `<task>.tar` against `SHA256SUMS`, extracts it to `<dest>/<task>/` and runs `check_vtab1k.py` on that task. Prints `<dest>` on stdout: that is the `--data_path`. A second call for the same task does nothing. |

Typical sequence (paths are examples):

```bash
STORE=~/links/projects/def-chdesa/$USER/sys843
scripts/data/download_vtab1k.sh  $STORE/data                    # ~4.7 GB
python scripts/data/repack_vtab1k.py $STORE/data/vtab-1k.zip $STORE/data/vtab-1k-tars
scripts/data/download_weights.sh $STORE/weights

# Optional full check of every task (extracts all 426k files, so use local disk):
mkdir -p /tmp/$USER/vtab && for t in $STORE/data/vtab-1k-tars/*.tar; do tar -xf $t -C /tmp/$USER/vtab; done
python scripts/data/check_vtab1k.py /tmp/$USER/vtab && rm -rf /tmp/$USER/vtab

# In a job:
DATA_PATH=$(scripts/data/stage_task.sh $STORE/data/vtab-1k-tars eurosat "$SLURM_TMPDIR/vtab-1k")
python main_tune.py --sys843 lora_r8 --data processed_vtab-eurosat --data_path "$DATA_PATH" \
    --pretrained_dir $STORE/weights --tune_root $STORE/runs
```

`tests/test_data_scripts.py` runs the repack, stage and check path on a tiny synthetic `vtab-1k.zip` in the same layout. It also runs the download checks against a `file://` URL. It never downloads the real files.
