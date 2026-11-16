#!/usr/bin/env bash
# Download the processed VTAB-1k zip (V-PETL's copy on Hugging Face) and verify it.
# Run on a login node (needs internet). Resumes a partial download.
#
#   scripts/data/download_vtab1k.sh <dest_dir>
#
# Writes <dest_dir>/vtab-1k.zip and <dest_dir>/vtab-1k.zip.sha256 (sha256sum format).
# Checks the size (4,659,851,071 bytes) and the sha256 published by Hugging Face
# (the X-Linked-ETag of the LFS file). Override with VTAB_URL / VTAB_SIZE / VTAB_SHA256
# (an empty VTAB_SHA256 only records the checksum).
set -euo pipefail

VTAB_URL=${VTAB_URL:-https://huggingface.co/datasets/XiN0919/VTAB-1k/resolve/main/vtab-1k.zip}
VTAB_SIZE=${VTAB_SIZE:-4659851071}
VTAB_SHA256=${VTAB_SHA256-87edd008ebd4a2921609eea4a1629fb3c5ddaece46938d7e1fc1224723667726}

if [ $# -ne 1 ]; then
    echo "usage: $0 <dest_dir>" >&2
    exit 2
fi
dest=$1
mkdir -p "$dest"
zip="$dest/vtab-1k.zip"

size_of() { stat -c %s "$1" 2>/dev/null || stat -f %z "$1"; }

if [ -f "$zip" ] && [ "$(size_of "$zip")" = "$VTAB_SIZE" ]; then
    echo "$zip already has the expected size, not downloading again" >&2
else
    echo "downloading $VTAB_URL -> $zip" >&2
    curl --fail --location --retry 5 --retry-delay 10 --continue-at - --output "$zip" "$VTAB_URL"
fi

actual_size=$(size_of "$zip")
if [ "$actual_size" != "$VTAB_SIZE" ]; then
    echo "ERROR: $zip is $actual_size bytes, expected $VTAB_SIZE" >&2
    exit 1
fi

echo "computing sha256 of $zip ..." >&2
actual_sha=$(sha256sum "$zip" | cut -d' ' -f1)
if [ -n "$VTAB_SHA256" ] && [ "$actual_sha" != "$VTAB_SHA256" ]; then
    echo "ERROR: sha256 $actual_sha, expected $VTAB_SHA256" >&2
    exit 1
fi
( cd "$dest" && printf '%s  %s\n' "$actual_sha" vtab-1k.zip > vtab-1k.zip.sha256 )
echo "OK: $zip ($actual_size bytes, sha256 $actual_sha)" >&2
echo "recorded in $dest/vtab-1k.zip.sha256" >&2
