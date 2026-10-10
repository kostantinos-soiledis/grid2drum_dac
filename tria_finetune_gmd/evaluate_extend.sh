#!/usr/bin/env bash
# Renders and evaluates the extended fine-tune (train_extend.sh, runs/gmd_long_160k: up to 160,000 iterations, best
# validation checkpoint as for the published run) the way render.sh and evaluate.sh treat the published systems: both
# prompt settings over the GMD test split, scored next to the 80,000-iteration model and our 250 Hz model against the
# real GMD bars, with the same metrics and recording-clustered statistics. The pairs ask whether the longer fine-tune
# changes anything (160k - 80k, per prompt setting) and where it leaves TRIA against our model (160k - ours).
# Results go to results/finetune_160k/; results/ of the published evaluation stays unchanged.
#
#   GMD_ROOT=<groove/> SOUNDFONT=<FluidR3_GM.sf2> PYTHON_BIN=<eval python> bash tria_finetune_gmd/evaluate_extend.sh cuda:0
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
DEVICE=${1:-cuda:0}
SHARDS=4                             # render shards side by side on $DEVICE
PY=${PYTHON_BIN:-python}             # the repo's evaluation environment (torch, fadtk, pyloudnorm)
TRIA_PY="$HERE/.venv/bin/python"     # TRIA's environment (README.md)
KAD_PY="$HERE/.venv_kad/bin/python"  # kadtk (README.md)
GMD=${GMD_ROOT:?set GMD_ROOT to GMD groove/ (info.csv, audio, MIDI)}
: "${SOUNDFONT:?set SOUNDFONT to FluidR3_GM.sf2}"
CHECKPOINT="$HERE/runs/gmd_long_160k/best/model.pt"
OURS="$ROOT/runs/final/exports/grid_250hz"
RESULTS="$HERE/results/finetune_160k"
EVAL="$RESULTS/evaluation"
log() { echo "[$(date '+%F %T')] $*"; }
[[ -f "$HERE/runs/gmd_long_160k/done.json" && -f "$CHECKPOINT" ]] || { echo "train_extend.sh has not finished"; exit 1; }

mkdir -p "$HERE/logs"
for prompts in bar midi; do
  out="$HERE/predictions/finetuned160k_$prompts"
  log "render $out"
  pids=()
  for ((k = 0; k < SHARDS; k++)); do
    "$TRIA_PY" "$HERE/export_audio.py" --checkpoint "$CHECKPOINT" --prompts "$prompts" --out-dir "$out" \
      --gmd-root "$GMD" --soundfont "$SOUNDFONT" --device "$DEVICE" --shard "$k/$SHARDS" \
      > "$HERE/logs/render_finetuned160k_${prompts}_shard$k.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  # one pass without --shard writes the manifest and summary (every clip exists, none is re-rendered)
  "$TRIA_PY" "$HERE/export_audio.py" --checkpoint "$CHECKPOINT" --prompts "$prompts" --out-dir "$out" \
    --gmd-root "$GMD" --soundfont "$SOUNDFONT" --device "$DEVICE" > "$HERE/logs/render_finetuned160k_$prompts.log" 2>&1
done

names=(grid_250hz tria_finetuned_bar tria_finetuned_midi tria_finetuned160k_bar tria_finetuned160k_midi)
dirs=("$OURS" "$HERE/predictions/finetuned_bar" "$HERE/predictions/finetuned_midi"
      "$HERE/predictions/finetuned160k_bar" "$HERE/predictions/finetuned160k_midi")
for d in "${dirs[@]}"; do [[ -f "$d/manifest.jsonl" ]] || { echo "missing $d/manifest.jsonl"; exit 1; }; done
pairs=(
  tria_finetuned160k_bar:tria_finetuned_bar tria_finetuned160k_midi:tria_finetuned_midi  # 160k - 80k iterations
  tria_finetuned160k_bar:grid_250hz tria_finetuned160k_midi:grid_250hz                   # 160k TRIA against our model
)
mkdir -p "$RESULTS/stats"

systems=()
for i in "${!names[@]}"; do systems+=(--system "${names[$i]}=${dirs[$i]}"); done
log "metrics against the real bars"
"$PY" "$HERE/evaluate_metrics.py" "${systems[@]}" --out "$EVAL" --gmd-root "$GMD" --fad-model clap-laion-music \
  --fad-repeats 8 --device "$DEVICE"

log "clustered paired statistics"
"$PY" "$ROOT/code/experiment/scripts/clustered_paired_stats.py" --per-clip "$EVAL/acoustic_eval/per_clip_metrics.csv" \
  --out "$RESULTS/stats/acoustic.json" --pairs "${pairs[@]}" --metrics mel_mae_db onset_flux_cosine --reps 5000 --seed 1234
"$PY" "$ROOT/code/experiment/scripts/clustered_paired_stats.py" --per-clip "$EVAL/direct_per_clip_metrics.csv" \
  --out "$RESULTS/stats/direct_audio.json" --pairs "${pairs[@]}" --metrics audio_l1 mrstft_logmag_l1 onset_f1_30ms onset_f1_50ms \
  onset_precision_30ms onset_recall_30ms onset_precision_50ms onset_recall_50ms --reps 5000 --seed 1234

log "KAD"
staged=()
for n in "${names[@]}"; do staged+=(--system "$n=$EVAL/systems/$n"); done
rm -rf "$EVAL/kad"  # kadtk reuses cached embeddings: none may predate this run's staged clips
PATH="$(dirname "$(readlink -f "$KAD_PY")"):$PATH" "$KAD_PY" "$HERE/kad_eval.py" \
  --reference "$EVAL/acoustic_eval/fad_assets/reference_audio" "${staged[@]}" --pairs "${pairs[@]}" \
  --reps 5000 --seed 1234 --out "$RESULTS/stats/kad.json" --work "$EVAL/kad" --device "$DEVICE"

log "summary"
"$PY" "$HERE/summarize.py" --evaluation-dir "$EVAL" --names "${names[@]}" --out-dir "$RESULTS"
"$PY" "$ROOT/code/experiment/scripts/relativize_artifact_paths.py" --repo-root "$ROOT" "$RESULTS"
log "complete: $RESULTS/summary.md"
