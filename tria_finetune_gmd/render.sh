#!/usr/bin/env bash
# Renders the GMD test split (1,733 bars) with TRIA, released and fine-tuned (runs/gmd_long), each with both prompt
# settings (export_audio.py): the bar itself as both prompts, and the bar's notes rendered with a General MIDI soundfont
# plus 2 s of the same recording outside the bar. Resumes: clips already in predictions/<name>/wavs are kept.
#
#   GMD_ROOT=<groove/> SOUNDFONT=<FluidR3_GM.sf2> bash tria_finetune_gmd/render.sh cuda:0
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
DEVICE=${1:-cuda:0}
: "${GMD_ROOT:?set GMD_ROOT to the groove folder of GMD}" "${SOUNDFONT:?set SOUNDFONT to FluidR3_GM.sf2}"
FINETUNED=runs/gmd_long/best/model.pt
log() { echo "[$(date '+%F %T')] $*"; }
mkdir -p logs predictions
for spec in "finetuned_midi $FINETUNED midi" "finetuned_bar $FINETUNED bar" "released_midi released midi" "released_bar released bar"; do
  set -- $spec
  log "render $1"
  .venv/bin/python export_audio.py --checkpoint "$2" --prompts "$3" --out-dir "predictions/$1" --device "$DEVICE" \
    > "logs/render_$1.log" 2>&1 || { log "failed: logs/render_$1.log"; exit 1; }
done
log "done"
