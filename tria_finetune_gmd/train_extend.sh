#!/usr/bin/env bash
# Continues the published fine-tune (runs/gmd_long, 80,000 iterations, train.sh) to 160,000 iterations in
# runs/gmd_long_160k, to test whether a longer fine-tune changes the comparison. The run starts from gmd_long's final
# checkpoint with its optimizer, learning-rate schedule and validation history (TRIA's own resume), so iterations
# 80,000-159,999 continue the same run with the same data, batch and decaying learning rate. It keeps a checkpoint at
# 120,000 and, as train.sh, stops after 10 validation checks without a new best. runs/gmd_long is left unchanged.
#
#   GMD_ROOT=<groove/> bash tria_finetune_gmd/train_extend.sh cuda:0
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE=${1:-cuda:0}
PY="$HERE/.venv/bin/python"
RUN="$HERE/runs/gmd_long_160k"
log() { echo "[$(date '+%F %T')] $*"; }

if [[ ! -e "$RUN/latest/extras.pt" ]]; then  # first start; a restart resumes from its own latest/
  mkdir -p "$RUN/latest"
  cp "$HERE/runs/gmd_long/best/model.pt" "$HERE/runs/gmd_long/best/extras.pt" "$RUN/latest/"
fi
"$PY" "$HERE/prepare.py" --name gmd_long_160k --num-iters 160000
sed -i 's/^save_iters: \[\]$/save_iters: [120000]/' "$HERE/gmd_long_160k.yml"
log "train ($DEVICE)"
cd "$HERE/upstream"  # TRIA's paths (tokenizer weights) are relative to its checkout
CUDA_VISIBLE_DEVICES=${DEVICE#cuda:} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRIA_PATIENCE=10 \
  "$PY" "$HERE/train_early_stop.py" --args.load "$HERE/gmd_long_160k.yml"
log "done: $(tr -d '\n ' < "$RUN/done.json" | cut -c1-200)"
