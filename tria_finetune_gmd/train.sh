#!/usr/bin/env bash
# Fine-tunes TRIA's small_musdb_moises_fsl_2b on GMD (README.md, prepare.py): from the released weights, batch 8, AdamW
# lr 1e-5, at most 80,000 iterations, validation every 2,000 (TRIA's cadence); train_early_stop.py ends the run after 10
# checks without a new best validation token cross-entropy (TRIA's own criterion for best/). An interrupted run resumes
# from runs/gmd_long/latest. The published run went all 80,000 iterations (every check a new best; runs/gmd_long/done.json).
#
#   GMD_ROOT=<groove/> bash tria_finetune_gmd/train.sh cuda:0
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE=${1:-cuda:0}
PY="$HERE/.venv/bin/python"
log() { echo "[$(date '+%F %T')] $*"; }

"$PY" "$HERE/prepare.py" --name gmd_long
log "train ($DEVICE)"
cd "$HERE/upstream"  # TRIA's paths (tokenizer weights) are relative to its checkout
CUDA_VISIBLE_DEVICES=${DEVICE#cuda:} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRIA_PATIENCE=10 \
  "$PY" "$HERE/train_early_stop.py" --args.load "$HERE/gmd_long.yml"
log "done: $(tr -d '\n ' < "$HERE/runs/gmd_long/done.json" | cut -c1-200)"
