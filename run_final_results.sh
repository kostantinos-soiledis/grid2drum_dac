#!/usr/bin/env bash
# Reproduce the final result path serially on one GPU.
#
# Main comparisons:
#   1. representation: pre-snap DAC projected latents vs post-snap PCA
#   2. regression:     post-snap PCA diffusion vs direct regression
#   3. capacity:       6-layer vs 8-layer direct regression
# Secondary checks:
#   4. grid rate:      90 / 120 / 250 / 500 Hz
#   5. RVQ supervision: post-snap PCA diffusion with vs without RVQ-CE (250 Hz arm)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$ROOT/code/experiment"
RUNS="$ROOT/runs/final"
PRED="$RUNS/exports"
RESULTS="$ROOT/results/final"

PYTHON_BIN="${PYTHON_BIN:-python}"
POSTSNAP_CACHE="${POSTSNAP_CACHE:-$ROOT/../pca_diffusion/cache_4beats_dac44q9_pca72_native_bpmgeom_duration_v1}"
PRESNAP_CACHE="${PRESNAP_CACHE:-$ROOT/../pca_diffusion/cache_4beats_dac44q9_presnap72_bpmgeom_duration_v1}"
GRID_500_OVERLAY="${GRID_500_OVERLAY:-$ROOT/../pca_diffusion/cache_4beats_dac44q9_pca72_grid500_overlay_v1}"
SOURCE_CACHE_ROOT="${SOURCE_CACHE_ROOT:-}"
DATASET_ROOT="${DATASET_ROOT:-}"
DEVICE="${DEVICE:-cuda:0}"
SCORE_DEVICE="${SCORE_DEVICE:-}"
FAD_MODEL="${FAD_MODEL:-clap-laion-music}"
MAX_ITEMS=0
FORCE=0
SKIP_TRAINING=0
SKIP_GRID=0
DRY_RUN=0

usage() {
  sed -n '2,10p' "$0"
  printf '\nUsage: %s [options]\n' "$0"
  cat <<'EOF'

Options:
  --python PATH            Python executable (or set PYTHON_BIN)
  --device DEVICE          training/export device (default: cuda:0)
  --score-device DEVICE    acoustic/FAD device (default: --device)
  --postsnap-cache PATH    post-snap PCA cache
  --presnap-cache PATH     native pre-snap latent cache
  --grid-500-overlay PATH  MIDI-rerendered 500 Hz symbolic-grid overlay
  --source-cache-root PATH source cache override for cache/overlay construction
  --dataset-root PATH      Groove dataset, needed for missing pre-snap/500 Hz caches
  --max-items N            evaluation smoke limit; 0 means the full test set
  --skip-training          require and reuse all canonical checkpoints
  --skip-grid              omit the secondary grid-Hz check
  --force                  rebuild exports and evaluation
  --dry-run                print commands without executing them
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --score-device) SCORE_DEVICE="$2"; shift 2 ;;
    --postsnap-cache) POSTSNAP_CACHE="$2"; shift 2 ;;
    --presnap-cache) PRESNAP_CACHE="$2"; shift 2 ;;
    --grid-500-overlay) GRID_500_OVERLAY="$2"; shift 2 ;;
    --source-cache-root) SOURCE_CACHE_ROOT="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --max-items) MAX_ITEMS="$2"; shift 2 ;;
    --skip-training) SKIP_TRAINING=1; shift ;;
    --skip-grid) SKIP_GRID=1; shift ;;
    --force) FORCE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
SCORE_DEVICE="${SCORE_DEVICE:-$DEVICE}"

run() {
  printf '  %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

require_file() {
  if [[ ! -f "$1" && "$DRY_RUN" -eq 0 ]]; then
    echo "missing required file: $1" >&2
    exit 1
  fi
}

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

run_reached_epoch() {
  local history="$1" expected_last_epoch="$2"
  [[ -f "$history" ]] || return 1
  awk -F, -v expected="$expected_last_epoch" '
    NR == 1 {
      for (i = 1; i <= NF; i++) {
        if ($i == "epoch") epoch_column = i
      }
      next
    }
    epoch_column { last_epoch = $epoch_column }
    END {
      gsub(/\r/, "", last_epoch)
      exit !(epoch_column && last_epoch ~ /^[0-9]+$/ && last_epoch + 0 >= expected)
    }
  ' "$history"
}

mkdir -p "$RUNS/grid_hz" "$PRED" "$RESULTS/logs" "$RESULTS/stats"
if [[ "$DRY_RUN" -eq 0 ]]; then
  exec > >(tee -a "$RESULTS/logs/final_results.log") 2>&1
fi
require_file "$POSTSNAP_CACHE/config.json"

if [[ ! -f "$PRESNAP_CACHE/config.json" ]]; then
  [[ -n "$DATASET_ROOT" || "$DRY_RUN" -eq 1 ]] || {
    echo "pre-snap cache is missing; pass --dataset-root to build it" >&2
    exit 1
  }
  log "build native pre-snap latent cache (no PCA)"
  build_args=(
    "$PYTHON_BIN" "$EXP/scripts/build_presnap_cache.py"
    --src-cache "$POSTSNAP_CACHE" --out-root "$PRESNAP_CACHE"
    --dataset-root "$DATASET_ROOT" --device "$DEVICE"
  )
  [[ -n "$SOURCE_CACHE_ROOT" ]] && build_args+=(--source-cache-root "$SOURCE_CACHE_ROOT")
  run "${build_args[@]}"
  run "$PYTHON_BIN" "$EXP/scripts/build_presnap_cache.py" \
    --src-cache "$POSTSNAP_CACHE" --out-root "$PRESNAP_CACHE" --finalize
fi

if [[ "$SKIP_GRID" -eq 0 && ! -f "$GRID_500_OVERLAY/config.json" ]]; then
  [[ -n "$DATASET_ROOT" || "$DRY_RUN" -eq 1 ]] || {
    echo "500 Hz overlay is missing; pass --dataset-root to re-render it from MIDI" >&2
    exit 1
  }
  log "build true 500 Hz frontend grid overlay from MIDI"
  overlay_args=(
    "$PYTHON_BIN" "$EXP/scripts/build_grid_rate_overlay.py"
    --base-cache "$POSTSNAP_CACHE" --out-root "$GRID_500_OVERLAY"
    --dataset-root "$DATASET_ROOT" --rate-hz 500
  )
  [[ -n "$SOURCE_CACHE_ROOT" ]] && overlay_args+=(--source-cache-root "$SOURCE_CACHE_ROOT")
  run "${overlay_args[@]}"
fi

train_diffusion() {
  local name="$1" cache="$2" run_dir="$3" rvq_ce="$4" grid_hz="$5" radii="$6" primary="$7" overlay="${8:-}"
  local batch_size="${9:-4}" grad_accum_steps="${10:-1}"
  if [[ -f "$run_dir/best_diffusion.pt" ]] && run_reached_epoch "$run_dir/history.csv" 149; then
    echo "reuse checkpoint: $name"
    return
  fi
  [[ "$SKIP_TRAINING" -eq 0 ]] || { echo "missing checkpoint: $run_dir/best_diffusion.pt" >&2; exit 1; }
  local args=(
    "$PYTHON_BIN" "$EXP/train_cli.py"
    --cache-root "$cache" --out-dir "$run_dir" --device "$DEVICE"
    --epochs 150 --batch-size "$batch_size" --eval-batch-size "$batch_size"
    --grad-accum-steps "$grad_accum_steps" --num-workers 4
    --seed 1234 --lr 1e-4 --weight-decay 1e-4
    --num-steps 25 --d-model 768 --num-layers 6 --num-heads 8
    --frontend-radii "$radii" --frontend-primary-radius "$primary"
    --grid-rate-hz "$grid_hz" --rvq-ce-weight "$rvq_ce"
    --use-bpm-training-geometry
  )
  [[ -n "$overlay" ]] && args+=(--grid-overlay-root "$overlay")
  if [[ -f "$run_dir/last.pt" ]]; then args+=(--resume); else args+=(--overwrite); fi
  log "train $name"
  run "${args[@]}"
}

train_direct() {
  local name="$1" layers="$2" run_dir="$3"
  if [[ -f "$run_dir/best_direct.pt" ]] && run_reached_epoch "$run_dir/history.csv" 149; then
    echo "reuse checkpoint: $name"
    return
  fi
  [[ "$SKIP_TRAINING" -eq 0 ]] || { echo "missing checkpoint: $run_dir/best_direct.pt" >&2; exit 1; }
  local args=(
    "$PYTHON_BIN" "$EXP/standalone_direct_pca_regressor.py"
    --cache-root "$POSTSNAP_CACHE" --out-dir "$run_dir" --device "$DEVICE"
    --epochs 150 --batch-size 4 --eval-batch-size 4 --num-workers 4
    --seed 1234 --lr 1e-4 --weight-decay 1e-4
    --d-model 1024 --num-layers "$layers" --num-heads 8
    --loss huber --huber-beta 0.25 --export-val-predictions 0
  )
  if [[ -f "$run_dir/last.pt" ]]; then args+=(--resume); else args+=(--overwrite); fi
  log "train $name"
  run "${args[@]}"
}

log "canonical training arms (strictly serial)"
train_diffusion postsnap_pca "$POSTSNAP_CACHE" "$RUNS/postsnap_pca" 0.0 0 "0,22,41,55" 22 ""
train_diffusion presnap_latent "$PRESNAP_CACHE" "$RUNS/presnap_latent" 0.0 0 "0,22,41,55" 22 ""
train_direct regression_6layer 6 "$RUNS/regression_6layer"
train_direct regression_8layer_capacity 8 "$RUNS/regression_8layer_capacity"
if [[ "$SKIP_GRID" -eq 0 ]]; then
  train_diffusion grid_90hz "$POSTSNAP_CACHE" "$RUNS/grid_hz/90hz" 0.1 90 "0,8,15,20" 8 ""
  train_diffusion grid_120hz "$POSTSNAP_CACHE" "$RUNS/grid_hz/120hz" 0.1 120 "0,11,20,26" 11 ""
  train_diffusion grid_250hz "$POSTSNAP_CACHE" "$RUNS/grid_hz/250hz" 0.1 250 "0,22,41,55" 22 ""
  # The 500 Hz frontend OOMs at batch 4 on a 10 GiB GPU. Keep the effective
  # training batch at 4 while reducing each forward/backward pass to one clip.
  train_diffusion grid_500hz "$POSTSNAP_CACHE" "$RUNS/grid_hz/500hz" 0.1 500 "0,44,82,110" 44 "$GRID_500_OVERLAY" 1 4
fi

prediction_dirs=()
prediction_names=()

export_diffusion() {
  local name="$1" run_dir="$2" cache="$3" overlay="${4:-}" out="$PRED/$1"
  require_file "$run_dir/best_diffusion.pt"
  if [[ "$FORCE" -eq 1 || ! -f "$out/manifest.jsonl" || "$run_dir/best_diffusion.pt" -nt "$out/manifest.jsonl" ]]; then
    log "export $name"
    local export_args=(
      "$PYTHON_BIN" "$EXP/scripts/export_best_diffusion_predictions.py"
      --train-dir "$run_dir" --cache-root "$cache" --split test --out-dir "$out"
      --num-steps 25 --x0-clip-norm 6 --num-beats 4 --beat-crossfade-ms 10
      --use-bpm-inference-geometry --sample-seed 1234 --max-items "$MAX_ITEMS"
      --batch-size 4 --num-workers 4 --device "$DEVICE" --overwrite
    )
    [[ -n "$overlay" ]] && export_args+=(--grid-overlay-root "$overlay")
    run "${export_args[@]}"
  fi
  prediction_names+=("$name")
  prediction_dirs+=("$out")
}

export_direct() {
  local name="$1" run_dir="$2" out="$PRED/$1"
  require_file "$run_dir/best_direct.pt"
  if [[ "$FORCE" -eq 1 || ! -f "$out/manifest.jsonl" || "$run_dir/best_direct.pt" -nt "$out/manifest.jsonl" ]]; then
    log "export $name"
    run "$PYTHON_BIN" "$EXP/scripts/export_direct_pca_predictions.py" \
      --run-dir "$run_dir" --cache-root "$POSTSNAP_CACHE" --split test --out-dir "$out" \
      --max-items "$MAX_ITEMS" --batch-size 8 --num-workers 2 --device "$DEVICE" --overwrite
  fi
  prediction_names+=("$name")
  prediction_dirs+=("$out")
}

export_diffusion postsnap_pca "$RUNS/postsnap_pca" "$POSTSNAP_CACHE" ""
export_diffusion presnap_latent "$RUNS/presnap_latent" "$PRESNAP_CACHE" ""
export_direct regression_6layer "$RUNS/regression_6layer"
export_direct regression_8layer_capacity "$RUNS/regression_8layer_capacity"
if [[ "$SKIP_GRID" -eq 0 ]]; then
  export_diffusion grid_90hz "$RUNS/grid_hz/90hz" "$POSTSNAP_CACHE" ""
  export_diffusion grid_120hz "$RUNS/grid_hz/120hz" "$POSTSNAP_CACHE" ""
  export_diffusion grid_250hz "$RUNS/grid_hz/250hz" "$POSTSNAP_CACHE" ""
  export_diffusion grid_500hz "$RUNS/grid_hz/500hz" "$POSTSNAP_CACHE" "$GRID_500_OVERLAY"
fi

EVAL="$RESULTS/evaluation"
NEED_EVAL="$FORCE"
if [[ ! -f "$EVAL/acoustic_eval/overall_summary.csv" ]]; then
  NEED_EVAL=1
else
  for prediction_dir in "${prediction_dirs[@]}"; do
    if [[ "$prediction_dir/manifest.jsonl" -nt "$EVAL/acoustic_eval/overall_summary.csv" ]]; then
      NEED_EVAL=1
      break
    fi
  done
fi
if [[ "$NEED_EVAL" -eq 1 ]]; then
  log "joint acoustic evaluation"
  run "$PYTHON_BIN" "$EXP/scripts/run_diffusion_acoustic_eval.py" --skip-export \
    --prediction-dirs "${prediction_dirs[@]}" --prediction-names "${prediction_names[@]}" \
    --cache-root "$POSTSNAP_CACHE" --out-dir "$EVAL" \
    --fad-model "$FAD_MODEL" --fad-workers 1 --fad-inf-workers 1 --fad-repeats 8 \
    --max-items "$MAX_ITEMS" --batch-size 8 --num-workers 4 \
    --device "$SCORE_DEVICE" --no-plots --overwrite
else
  echo "reuse evaluation: $EVAL"
fi

log "build readable comparison tables"
run "$PYTHON_BIN" "$EXP/scripts/summarize_final_results.py" \
  --evaluation-dir "$EVAL" --out-dir "$RESULTS"

if [[ "$DRY_RUN" -eq 0 ]]; then
  acoustic_pairs=(
    presnap_latent:postsnap_pca
    regression_6layer:postsnap_pca
    regression_8layer_capacity:regression_6layer
  )
  if [[ "$SKIP_GRID" -eq 0 ]]; then
    # grid_250hz is post-snap PCA diffusion plus the RVQ-CE term, so it doubles
    # as the RVQ-supervision arm against postsnap_pca.
    acoustic_pairs+=(grid_90hz:grid_250hz grid_120hz:grid_250hz grid_500hz:grid_250hz grid_250hz:postsnap_pca)
  fi
  run "$PYTHON_BIN" "$EXP/scripts/clustered_paired_stats.py" \
    --per-clip "$EVAL/acoustic_eval/per_clip_metrics.csv" \
    --out "$RESULTS/stats/acoustic.json" --pairs "${acoustic_pairs[@]}" \
    --metrics mel_mae_db onset_flux_cosine --reps 5000 --seed 1234
  run "$PYTHON_BIN" "$EXP/scripts/clustered_paired_stats.py" \
    --per-clip "$RESULTS/direct_per_clip_metrics.csv" \
    --out "$RESULTS/stats/direct_audio.json" --pairs "${acoustic_pairs[@]}" \
    --metrics audio_l1 mrstft_logmag_l1 --reps 5000 --seed 1234
  run "$PYTHON_BIN" "$EXP/scripts/relativize_artifact_paths.py" \
    --repo-root "$ROOT" "$RUNS" "$RESULTS"
fi

log "complete: $RESULTS/summary.md"
