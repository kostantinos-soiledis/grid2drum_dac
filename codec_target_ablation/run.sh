#!/usr/bin/env bash
# Codec-target ablation: two arms for the "latent and objective" group of Table 1 (250 Hz EDG, no RVQ-CE).
#
#   tokens    tokens_9x1024: the regression arm's MGE + Transformer (6 layers, width 1024) predicting DAC's
#             nine RVQ codes with cross-entropy; the frozen DAC decodes the argmax codes (train_tokens.py)
#   full1024  postsnap_full1024: postsnap_pca's diffusion recipe on the raw 1024-D post-snap latent z_q
#             instead of its 72-D PCA coordinates; its cache comes from the unchanged
#             build_diffusion_cache.py with --pca-k 0, and sampling clips x0 at +-6 like the final
#             diffusion arms (run_final_results.sh, the paper's m). Its test set is exported twice
#             with the same samples: as generated, and as postsnap_full1024_projected, with each latent
#             projected onto DAC's 72-D post-snap subspace before decoding (export_projected.py)
#
# Stages, in order: cache (full1024 only), train, export, eval, stats. Finished steps are skipped and an
# interrupted training resumes, so rerunning the same command continues where it stopped. The
# evaluation scores against the same decoded targets as results/final, and the stats pair each arm with
# postsnap_pca and regression_6layer from results/final.
#
#   bash codec_target_ablation/run.sh --device cuda:0                          # both arms, serially
#   bash codec_target_ablation/run.sh --arms tokens --stages train,export      # one arm per GPU ...
#   bash codec_target_ablation/run.sh --arms full1024 --stages cache,train,export --device cuda:2
#   bash codec_target_ablation/run.sh --stages eval,stats                      # ... then score both
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
EXP="$ROOT/code/experiment"
PYTHON_BIN="${PYTHON_BIN:-python}"
POSTSNAP_CACHE="$ROOT/caches/cache_4beats_dac44q9_pca72_native_bpmgeom_duration_v1"
SOURCE_CACHE="$ROOT/caches/4beats_dac44q9_native_v1"
FULL_CACHE="$ROOT/caches/cache_4beats_dac44q9_full1024_bpmgeom_duration_v1"
FINAL_EVAL="$ROOT/results/final/evaluation"
RUNS="$HERE/runs"
EXPORTS="$HERE/exports"
RESULTS="$HERE/results"
DEVICE="cuda:0"
ARMS="tokens,full1024"
STAGES="cache,train,export,eval,stats"
EPOCHS=150
DRY_RUN=0

usage() { sed -n '2,21p' "$0"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --arms) ARMS="$2"; shift 2 ;;
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

if has "$STAGES" cache && has "$ARMS" full1024; then
  if [[ -f "$FULL_CACHE/manifests/test.jsonl" ]]; then
    echo "reuse cache: $FULL_CACHE"
  else
    log "build the 1024-D post-snap cache (train first; it writes config.json)"
    for split in train validation test; do
      run "$PYTHON_BIN" "$EXP/scripts/build_diffusion_cache.py" \
        --source-cache-root "$SOURCE_CACHE" --out-root "$FULL_CACHE" --split "$split" --pca-k 0 \
        --geometry-mode bpm --bpm-source duration --target-token-rate-hz 86.1328125 \
        --device "$DEVICE" --overwrite
    done
  fi
fi

if has "$STAGES" train && has "$ARMS" tokens; then
  run_dir="$RUNS/tokens_9x1024"
  if run_reached_epoch "$run_dir/history.csv" $((EPOCHS - 1)); then
    echo "reuse checkpoint: $run_dir"
  else
    log "train tokens_9x1024 on $DEVICE"
    run "$PYTHON_BIN" "$HERE/train_tokens.py" \
      --cache-root "$POSTSNAP_CACHE" --out-dir "$run_dir" --device "$DEVICE" \
      --epochs "$EPOCHS" --batch-size 4 --eval-batch-size 4 --num-workers 4 \
      --seed 1234 --lr 1e-4 --weight-decay 1e-4 \
      --d-model 1024 --num-layers 6 --num-heads 8 \
      --frontend-radii 0,22,41,55 --frontend-primary-radius 22 \
      "$(resume_flag "$run_dir")"
  fi
fi

if has "$STAGES" train && has "$ARMS" full1024; then
  run_dir="$RUNS/postsnap_full1024"
  if run_reached_epoch "$run_dir/history.csv" $((EPOCHS - 1)); then
    echo "reuse checkpoint: $run_dir"
  else
    log "train postsnap_full1024 on $DEVICE"
    run "$PYTHON_BIN" "$EXP/train_cli.py" \
      --cache-root "$FULL_CACHE" --out-dir "$run_dir" --device "$DEVICE" \
      --epochs "$EPOCHS" --batch-size 4 --eval-batch-size 4 --grad-accum-steps 1 --num-workers 4 \
      --seed 1234 --lr 1e-4 --weight-decay 1e-4 \
      --num-steps 25 --d-model 768 --num-layers 6 --num-heads 8 \
      --frontend-radii 0,22,41,55 --frontend-primary-radius 22 \
      --grid-rate-hz 0 --rvq-ce-weight 0.0 --use-bpm-training-geometry \
      "$(resume_flag "$run_dir")"
  fi
fi

if has "$STAGES" export && has "$ARMS" tokens; then
  log "export tokens_9x1024"
  run "$PYTHON_BIN" "$HERE/export_tokens.py" \
    --run-dir "$RUNS/tokens_9x1024" --cache-root "$POSTSNAP_CACHE" --split test \
    --out-dir "$EXPORTS/tokens_9x1024" --device "$DEVICE" --overwrite
fi

if has "$STAGES" export && has "$ARMS" full1024; then
  export_args=(
    --train-dir "$RUNS/postsnap_full1024" --cache-root "$FULL_CACHE" --split test
    --num-steps 25 --x0-clip-bound 6 --num-beats 4 --beat-crossfade-ms 10
    --use-bpm-inference-geometry --sample-seed 1234 --max-items 0
    --batch-size 4 --num-workers 4 --device "$DEVICE" --overwrite
  )
  log "export postsnap_full1024"
  run "$PYTHON_BIN" "$EXP/scripts/export_best_diffusion_predictions.py" \
    "${export_args[@]}" --out-dir "$EXPORTS/postsnap_full1024"
  log "export postsnap_full1024_projected"
  run "$PYTHON_BIN" "$HERE/export_projected.py" --pca-basis "$POSTSNAP_CACHE/pca_basis.pt" \
    "${export_args[@]}" --out-dir "$EXPORTS/postsnap_full1024_projected"
fi

names=()
has "$ARMS" tokens && names+=(tokens_9x1024)
has "$ARMS" full1024 && names+=(postsnap_full1024 postsnap_full1024_projected)

if has "$STAGES" eval; then
  log "score ${names[*]} against the decoded targets of results/final"
  dirs=()
  for name in "${names[@]}"; do dirs+=("$EXPORTS/$name"); done
  run "$PYTHON_BIN" "$EXP/scripts/run_diffusion_acoustic_eval.py" --skip-export \
    --prediction-dirs "${dirs[@]}" --prediction-names "${names[@]}" \
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
  pairs=()
  has "$ARMS" tokens && pairs+=(tokens_9x1024:postsnap_pca tokens_9x1024:regression_6layer)
  has "$ARMS" full1024 && pairs+=(
    postsnap_full1024:postsnap_pca
    postsnap_full1024_projected:postsnap_pca
    postsnap_full1024_projected:postsnap_full1024
  )
  run "$PYTHON_BIN" "$EXP/scripts/clustered_paired_stats.py" \
    --per-clip "$RESULTS/acoustic_per_clip.csv" --out "$RESULTS/stats/acoustic.json" \
    --pairs "${pairs[@]}" --metrics mel_mae_db onset_flux_cosine --reps 5000 --seed 1234
  run "$PYTHON_BIN" "$EXP/scripts/clustered_paired_stats.py" \
    --per-clip "$RESULTS/direct_per_clip.csv" --out "$RESULTS/stats/direct_audio.json" \
    --pairs "${pairs[@]}" --metrics audio_l1 mrstft_logmag_l1 --reps 5000 --seed 1234
  log "tables: $RESULTS/evaluation/acoustic_eval/overall_summary.csv, direct_audio_eval/*/summary.json; tests: $RESULTS/stats/"
fi
