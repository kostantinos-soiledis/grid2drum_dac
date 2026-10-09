#!/usr/bin/env bash
# Final evaluation of TRIA on the GMD test split (README.md) with the ablations' metrics (run_final_results.sh), our
# 250 Hz model's test exports alongside, every metric against the real GMD bars:
#   1. evaluate_metrics.py: the real bars as reference; every system's clips gain-matched to their real bar's loudness;
#      the evaluator's per-clip acoustic metrics (log-mel MAE, onset-flux cosine, ...) and FAD-inf (CLAP-LAION-Music,
#      8 repeats); waveform L1 and MR-STFT on peak-normalized audio; onset F1 at +-30 and +-50 ms
#   2. code/experiment/scripts/clustered_paired_stats.py: paired tests clustered by recording (bootstrap CIs and
#      sign-flip p-values, 5000 resamples)
#   3. kad_eval.py: KAD through kadtk with PANNs-WGLM against the real bars (one kernel for all systems), with
#      recording-clustered bootstrap CIs and recording-level permutation tests for the pairs
#   4. summarize.py: results/summary.md and the CSV tables (Holm-adjusted p-values)
# Large artifacts (staged audio, embeddings, per-clip tables) stay in results/evaluation/ (not in git).
#
#   GMD_ROOT=<groove/> PYTHON_BIN=<eval python> bash tria_finetune_gmd/evaluate.sh cuda:0 [--max-items N]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
DEVICE=${1:-cuda:0}
MAX_ITEMS=0
[[ "${2:-}" == "--max-items" ]] && MAX_ITEMS=${3:?}
PY=${PYTHON_BIN:-python}             # the repo's evaluation environment (torch, fadtk, pyloudnorm)
GMD=${GMD_ROOT:?set GMD_ROOT to GMD groove/ (info.csv, audio, MIDI)}
KAD_PY="$HERE/.venv_kad/bin/python"  # kadtk (README.md)
OURS="$ROOT/runs/final/exports/grid_250hz"
RESULTS="$HERE/results"
EVAL="$RESULTS/evaluation"
log() { echo "[$(date '+%F %T')] $*"; }

names=(grid_250hz tria_released_bar tria_released_midi tria_finetuned_bar tria_finetuned_midi)
dirs=("$OURS" "$HERE/predictions/released_bar" "$HERE/predictions/released_midi" "$HERE/predictions/finetuned_bar"
      "$HERE/predictions/finetuned_midi")
for d in "${dirs[@]}"; do [[ -f "$d/manifest.jsonl" ]] || { echo "missing $d/manifest.jsonl (render.sh)"; exit 1; }; done
pairs=(
  tria_finetuned_bar:tria_released_bar tria_finetuned_midi:tria_released_midi        # fine-tuning on GMD
  tria_released_midi:tria_released_bar tria_finetuned_midi:tria_finetuned_bar        # prompts: MIDI + reference vs the bar
  tria_released_bar:grid_250hz tria_released_midi:grid_250hz                         # TRIA against our model
  tria_finetuned_bar:grid_250hz tria_finetuned_midi:grid_250hz
)
mkdir -p "$RESULTS/stats"

systems=()
for i in "${!names[@]}"; do systems+=(--system "${names[$i]}=${dirs[$i]}"); done
log "metrics against the real bars"
"$PY" "$HERE/evaluate_metrics.py" "${systems[@]}" --out "$EVAL" --gmd-root "$GMD" --fad-model clap-laion-music \
  --fad-repeats 8 --max-items "$MAX_ITEMS" --device "$DEVICE"

log "clustered paired statistics"
"$PY" "$ROOT/code/experiment/scripts/clustered_paired_stats.py" --per-clip "$EVAL/acoustic_eval/per_clip_metrics.csv" \
  --out "$RESULTS/stats/acoustic.json" --pairs "${pairs[@]}" --metrics mel_mae_db onset_flux_cosine --reps 5000 --seed 1234
"$PY" "$ROOT/code/experiment/scripts/clustered_paired_stats.py" --per-clip "$EVAL/direct_per_clip_metrics.csv" \
  --out "$RESULTS/stats/direct_audio.json" --pairs "${pairs[@]}" --metrics audio_l1 mrstft_logmag_l1 onset_f1_30ms onset_f1_50ms --reps 5000 \
  --seed 1234

log "KAD"
staged=()
for n in "${names[@]}"; do staged+=(--system "$n=$EVAL/systems/$n"); done
rm -rf "$EVAL/kad"  # kadtk reuses cached embeddings: none may predate this run's staged clips
# kadtk resamples with sox/ffmpeg, which live next to the evaluation interpreter
PATH="$(dirname "$(readlink -f "$KAD_PY")"):$PATH" "$KAD_PY" "$HERE/kad_eval.py" \
  --reference "$EVAL/acoustic_eval/fad_assets/reference_audio" "${staged[@]}" --pairs "${pairs[@]}" \
  --max-items "$MAX_ITEMS" --reps 5000 --seed 1234 --out "$RESULTS/stats/kad.json" --work "$EVAL/kad" --device "$DEVICE"

log "summary"
"$PY" "$HERE/summarize.py" --evaluation-dir "$EVAL" --names "${names[@]}" --out-dir "$RESULTS"
"$PY" "$ROOT/code/experiment/scripts/relativize_artifact_paths.py" --repo-root "$ROOT" "$RESULTS"
log "complete: $RESULTS/summary.md"
