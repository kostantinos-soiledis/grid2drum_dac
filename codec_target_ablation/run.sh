#!/usr/bin/env bash
# Codec-target ablation: the parallel token control of the paper (250 Hz EDG, no RVQ-CE).
#
#   tokens_9x1024: the regression arm's MGE + Transformer (6 layers, width 1024) predicting DAC's
#   nine RVQ codes with cross-entropy; the frozen DAC decodes the argmax codes (train_tokens.py)
#
# Stages, in order: train, export, eval, stats. Finished steps are skipped and an interrupted training
# resumes, so rerunning the same command continues where it stopped. The evaluation scores against the
# same decoded targets as results/final, and the stats pair the arm with postsnap_pca and
# regression_6layer from results/final.
#
#   bash codec_target_ablation/run.sh --device cuda:0
#   bash codec_target_ablation/run.sh --stages eval,stats
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
EXP="$ROOT/code/experiment"
PYTHON_BIN="${PYTHON_BIN:-python}"
POSTSNAP_CACHE="$ROOT/caches/cache_4beats_dac44q9_pca72_native_bpmgeom_duration_v1"
FINAL_EVAL="$ROOT/results/final/evaluation"
RUNS="$HERE/runs"
EXPORTS="$HERE/exports"
RESULTS="$HERE/results"
DEVICE="cuda:0"
STAGES="train,export,eval,stats"
EPOCHS=150
DRY_RUN=0
NAME="tokens_9x1024"

usage() { sed -n '2,13p' "$0"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --stages) STAGES="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

has() { [[ ",$1," == *",$2,"* ]]; }
log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }
run() {
  printf '  %q' "$@"; printf '\n'
  if [[ "$DRY_RUN" -eq 0 ]]; then "$@"; fi
}

# True once history.csv records the last epoch (same check as run_final_results.sh).
run_reached_epoch() {
  local history="$1" expected_last_epoch="$2"
  [[ -f "$history" ]] || return 1
  awk -F, -v expected="$expected_last_epoch" '
    NR == 1 { for (i = 1; i <= NF; i++) if ($i == "epoch") epoch_column = i; next }
    epoch_column { last_epoch = $epoch_column }
    END { gsub(/\r/, "", last_epoch); exit !(epoch_column && last_epoch ~ /^[0-9]+$/ && last_epoch + 0 >= expected) }
  ' "$history"
}

resume_flag() { if [[ -f "$1/last.pt" ]]; then echo --resume; else echo --overwrite; fi; }

if has "$STAGES" train; then
  run_dir="$RUNS/$NAME"
  if run_reached_epoch "$run_dir/history.csv" $((EPOCHS - 1)); then
    echo "reuse checkpoint: $run_dir"
  else
    log "train $NAME on $DEVICE"
    run "$PYTHON_BIN" "$HERE/train_tokens.py" \
      --cache-root "$POSTSNAP_CACHE" --out-dir "$run_dir" --device "$DEVICE" \
      --epochs "$EPOCHS" --batch-size 4 --eval-batch-size 4 --num-workers 4 \
      --seed 1234 --lr 1e-4 --weight-decay 1e-4 \
      --d-model 1024 --num-layers 6 --num-heads 8 \
      --frontend-radii 0,22,41,55 --frontend-primary-radius 22 \
      "$(resume_flag "$run_dir")"
  fi
fi

if has "$STAGES" export; then
  log "export $NAME"
  run "$PYTHON_BIN" "$HERE/export_tokens.py" \
    --run-dir "$RUNS/$NAME" --cache-root "$POSTSNAP_CACHE" --split test \
    --out-dir "$EXPORTS/$NAME" --device "$DEVICE" --overwrite
fi

if has "$STAGES" eval; then
  log "score $NAME against the decoded targets of results/final"
  run "$PYTHON_BIN" "$EXP/scripts/run_diffusion_acoustic_eval.py" --skip-export \
    --prediction-dirs "$EXPORTS/$NAME" --prediction-names "$NAME" \
    --cache-root "$POSTSNAP_CACHE" --out-dir "$RESULTS/evaluation" \
    --fad-model clap-laion-music --fad-workers 1 --fad-inf-workers 1 --fad-repeats 8 \
    --max-items 0 --batch-size 8 --num-workers 4 \
    --device "$DEVICE" --no-plots --overwrite
fi

if has "$STAGES" stats; then
  log "paired tests against the final arms"
  run "$PYTHON_BIN" "$HERE/combine_per_clip.py" \
    --evaluation-dir "$RESULTS/evaluation" --final-evaluation-dir "$FINAL_EVAL" \
    --final-models postsnap_pca regression_6layer --out-dir "$RESULTS"
  pairs=("$NAME:postsnap_pca" "$NAME:regression_6layer")
  run "$PYTHON_BIN" "$EXP/scripts/clustered_paired_stats.py" \
    --per-clip "$RESULTS/acoustic_per_clip.csv" --out "$RESULTS/stats/acoustic.json" \
    --pairs "${pairs[@]}" --metrics mel_mae_db onset_flux_cosine --reps 5000 --seed 1234
  run "$PYTHON_BIN" "$EXP/scripts/clustered_paired_stats.py" \
    --per-clip "$RESULTS/direct_per_clip.csv" --out "$RESULTS/stats/direct_audio.json" \
    --pairs "${pairs[@]}" --metrics audio_l1 mrstft_logmag_l1 --reps 5000 --seed 1234
  log "tables: $RESULTS/evaluation/acoustic_eval/overall_summary.csv, direct_audio_eval/*/summary.json; tests: $RESULTS/stats/"
fi
