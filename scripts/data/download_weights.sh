#!/usr/bin/env bash
# Download the supervised ImageNet-21k ViT-B/16 weights (Google's ViT-B_16.npz) and
# save them under the name the code loads: <weights_dir>/ViT-B_16_in21k.npz.
# Run on a login node (needs internet).
#
#   scripts/data/download_weights.sh [<weights_dir>]
#
# <weights_dir> defaults to $PETL_PRETRAINED_DIR, else ./pretrained_weights; pass the
# same directory to main.py / main_tune.py with --pretrained_dir (or set
# PETL_PRETRAINED_DIR). Checks the size (412,815,506 bytes) and the sha256 of the copy
# used for the SYS843 CPU tests; writes ViT-B_16_in21k.npz.sha256 next to it.
# Override with WEIGHTS_URL / WEIGHTS_SIZE / WEIGHTS_SHA256 (empty: record only).
set -euo pipefail

WEIGHTS_URL=${WEIGHTS_URL:-https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz}
WEIGHTS_SIZE=${WEIGHTS_SIZE:-412815506}
WEIGHTS_SHA256=${WEIGHTS_SHA256-ac7478131e1b77e5c2f0225849730bd19ee89cfffe600c2dfd2272b34a50a43d}

if [ $# -gt 1 ]; then
    echo "usage: $0 [<weights_dir>]" >&2
    exit 2
fi
dir=${1:-${PETL_PRETRAINED_DIR:-pretrained_weights}}
mkdir -p "$dir"
out="$dir/ViT-B_16_in21k.npz"

size_of() { stat -c %s "$1" 2>/dev/null || stat -f %z "$1"; }

if [ -f "$out" ] && [ "$(size_of "$out")" = "$WEIGHTS_SIZE" ]; then
    echo "$out already has the expected size, not downloading again" >&2
else
    echo "downloading $WEIGHTS_URL -> $out" >&2
    curl --fail --location --retry 5 --retry-delay 10 --output "$out.part" "$WEIGHTS_URL"
    mv "$out.part" "$out"
fi

actual_size=$(size_of "$out")
if [ "$actual_size" != "$WEIGHTS_SIZE" ]; then
    echo "ERROR: $out is $actual_size bytes, expected $WEIGHTS_SIZE" >&2
    exit 1
fi
actual_sha=$(sha256sum "$out" | cut -d' ' -f1)
if [ -n "$WEIGHTS_SHA256" ] && [ "$actual_sha" != "$WEIGHTS_SHA256" ]; then
    echo "ERROR: sha256 $actual_sha, expected $WEIGHTS_SHA256" >&2
    exit 1
fi
( cd "$dir" && printf '%s  %s\n' "$actual_sha" ViT-B_16_in21k.npz > ViT-B_16_in21k.npz.sha256 )
echo "OK: $out ($actual_size bytes, sha256 $actual_sha)" >&2
