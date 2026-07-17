#!/usr/bin/env bash
# =====================================================================================
# Overnight "solid foundation" experiments for the DAC-subspace diffusion paper.
#
# Runs only the experiments that are actually NECESSARY (per the readiness review),
# all through the existing single evaluation pipeline so every number is comparable:
#
#   Stage 1  Sampling-seed stability (NO training).
#            Re-export the two best diffusion models (plain 25-step, RVQ-CE 12-step)
#            with N explicit sampling seeds, score each export through the unified
#            acoustic evaluator (mel / onset-flux / band-balance / FAD-inf), so the
#            paper can report mean +/- std over sampling seeds instead of one sample.
#
#   Stage 2  Event-level control faithfulness across systems (NO training).
#            Run the existing per-family onset diagnostic with ONE fixed config over
#            best-plain (5 seeds), best-RVQ-CE (5 seeds), direct regressor, grid
#            render, and symbolic-NN retrieval. Reuses Stage 1's seed exports, so no
#            extra diffusion decoding is needed.
#
#   Stage 3  Parameter-matched direct baseline (OPTIONAL; the only training job).
#            Train one ~100M direct PCA regressor (d_model 1024, 8 layers) that is at
#            least as large as the 91.85M diffusion model, export, and score through
#            the same pipeline + control diagnostic. Gives one appendix row that lets
#            the diffusion-vs-direct claim stand without a parameter-count loophole.
#            Set RUN_STAGE3=0 to skip.
#
# The script is idempotent (existing outputs are skipped unless FORCE=1) and never
# aborts the whole night on a single failure: each stage logs a WARN and continues,
# and a per-stage status summary is printed at the end. Run it detached, e.g.:
#
#   DEVICE=cuda:2 nohup bash scripts/run_overnight_foundation.sh > overnight.out 2>&1 &
#
# Then rebuild the paper tables with:  python scripts/build_paper_results.py --out-dir paper_results --strict
# and aggregate the new artifacts with: python scripts/aggregate_overnight.py (run automatically at the end).
# =====================================================================================
set -uo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || { echo "cannot cd to repo root: $REPO" >&2; exit 1; }

# ---- configuration (override via environment) ---------------------------------------
PY="${PY:-python}"
CACHE="${DAC_CACHE:?set DAC_CACHE to the diffusion cache root}"
RUNS_ROOT="${RUNS_ROOT:-.}"
DEVICE="${DEVICE:-auto}"
SEEDS="${SEEDS:-1234 11 22 33 44}"      # sampling seeds for Stage 1/2
MAX_ITEMS="${MAX_ITEMS:-0}"             # 0 = full test set; FAD-inf needs >=500 clips
EXPORT_BS="${EXPORT_BS:-16}"
FAD_MODEL="${FAD_MODEL:-clap-laion-music}"
RUN_STAGE3="${RUN_STAGE3:-1}"          # 1 = train parameter-matched direct baseline
FORCE="${FORCE:-0}"                    # 1 = recompute even if outputs exist

# ---- fixed paths --------------------------------------------------------------------
OVERNIGHT="${OVERNIGHT_ROOT:-paper_results/overnight}"
SEED_OUT="$OVERNIGHT/seed_stability"
CTRL_OUT="$OVERNIGHT/control_faithfulness"
S3_DIR="$RUNS_ROOT/runs_direct/direct_pca_d1024_l8_seed1234"
S3_EVAL="$OVERNIGHT/matched_baseline_eval"
mkdir -p "$OVERNIGHT" "$SEED_OUT" "$CTRL_OUT"
LOG="$OVERNIGHT/run_$(date +%Y%m%d_%H%M%S).log"

# ---- systems --------------------------------------------------------------------------
# name|train_dir|num_steps  (the two best models from run_metrics.csv)
SWEEP_MODELS=(
  "diffusion_pca_25steps|$RUNS_ROOT/runs_dac/dac_25steps|25"
  "diffusion_pca_rvq_ce_12steps|$RUNS_ROOT/runs_dac_ce/dac_12steps|12"
)
# deterministic systems already exported on disk, scored for the control diagnostic
DET_CONTROL=(
  "direct_pca_d1024_l6_seed1234|$RUNS_ROOT/runs_direct/direct_pca_d1024_l6_seed1234/test_set_predictions"
  "grid_render|$RUNS_ROOT/runs_baselines/dac_test_v1/grid_render"
  "symbolic_nn_train|$RUNS_ROOT/runs_baselines/dac_test_v1/symbolic_nn_train"
)

declare -A STATUS

log() { printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }
banner() { log "============================================================"; log "$*"; log "============================================================"; }

if [ "$MAX_ITEMS" != "0" ] && [ "$MAX_ITEMS" -lt 500 ]; then
  log "WARN: MAX_ITEMS=$MAX_ITEMS < 500; FAD-inf is unreliable below ~500 clips."
fi

log "repo=$REPO"
log "PY=$PY  DEVICE=$DEVICE  CACHE=$CACHE"
log "SEEDS='$SEEDS'  MAX_ITEMS=$MAX_ITEMS  FAD_MODEL=$FAD_MODEL  RUN_STAGE3=$RUN_STAGE3  FORCE=$FORCE"
log "log file: $LOG"

# =====================================================================================
# Stage 1 - sampling-seed stability
# =====================================================================================
banner "STAGE 1 - sampling-seed stability"
seed_dirs=(); seed_names=()
stage1_ok=1
for spec in "${SWEEP_MODELS[@]}"; do
  IFS='|' read -r mname tdir steps <<< "$spec"
  if [ ! -f "$tdir/best_diffusion.pt" ]; then
    log "WARN: checkpoint missing for $mname ($tdir/best_diffusion.pt); skipping its seeds"
    stage1_ok=0
    continue
  fi
  for s in $SEEDS; do
    od="$tdir/seed_sweep/seed_${s}"
    if [ "$FORCE" != "1" ] && [ -f "$od/manifest.jsonl" ]; then
      log "skip export (exists): $od"
    else
      log "export $mname seed=$s steps=$steps -> $od"
      "$PY" scripts/export_best_diffusion_predictions.py \
        --train-dir "$tdir" --split test --out-dir "$od" \
        --num-steps "$steps" --guidance-scale 1.0 --x0-clip-norm 6.0 \
        --num-beats 4 --beat-crossfade-ms 10 --use-bpm-inference-geometry \
        --sample-seed "$s" --cache-root "$CACHE" --max-items "$MAX_ITEMS" \
        --batch-size "$EXPORT_BS" --num-workers 4 --device "$DEVICE" --overwrite \
        >>"$LOG" 2>&1 || { log "WARN: export failed for $mname seed=$s"; stage1_ok=0; }
    fi
    if [ -f "$od/manifest.jsonl" ]; then
      seed_dirs+=("$od"); seed_names+=("${mname}_seed${s}")
    fi
  done
done

if [ "${#seed_dirs[@]}" -gt 0 ]; then
  if [ "$FORCE" != "1" ] && [ -f "$SEED_OUT/acoustic_eval/overall_summary.csv" ]; then
    log "skip seed acoustic eval (exists): $SEED_OUT/acoustic_eval/overall_summary.csv"
  else
    log "scoring ${#seed_dirs[@]} seed exports through the unified acoustic pipeline"
    "$PY" scripts/run_diffusion_acoustic_eval.py --skip-export \
      --prediction-dirs "${seed_dirs[@]}" \
      --prediction-names "${seed_names[@]}" \
      --cache-root "$CACHE" --out-dir "$SEED_OUT" \
      --fad-model "$FAD_MODEL" --fad-workers 1 --fad-inf-workers 1 \
      --batch-size "$EXPORT_BS" --num-workers 4 --device "$DEVICE" \
      --no-plots --overwrite \
      >>"$LOG" 2>&1 || { log "WARN: seed acoustic eval failed"; stage1_ok=0; }
  fi
else
  log "WARN: no seed exports available to score"
  stage1_ok=0
fi
STATUS[stage1]=$([ "$stage1_ok" = "1" ] && echo OK || echo PARTIAL)

# =====================================================================================
# Stage 2 - event-level control faithfulness (one fixed config for every system)
# =====================================================================================
banner "STAGE 2 - control faithfulness"
stage2_ok=1
run_control() {  # $1=name $2=predictions_dir
  local name="$1" dir="$2"
  if [ ! -f "$dir/manifest.jsonl" ]; then
    log "WARN: control skipped, no manifest: $dir"; stage2_ok=0; return
  fi
  if [ "$FORCE" != "1" ] && [ -f "$CTRL_OUT/$name/summary.json" ]; then
    log "skip control (exists): $CTRL_OUT/$name"; return
  fi
  log "control-faithfulness: $name"
  "$PY" scripts/evaluate_control_faithfulness.py \
    --predictions-dir "$dir" --cache-root "$CACHE" --split test \
    --out-dir "$CTRL_OUT/$name" --overwrite \
    >>"$LOG" 2>&1 || { log "WARN: control eval failed: $name"; stage2_ok=0; }
}
# diffusion systems: reuse Stage 1 seed exports (mean +/- std over seeds)
for spec in "${SWEEP_MODELS[@]}"; do
  IFS='|' read -r mname tdir steps <<< "$spec"
  for s in $SEEDS; do
    run_control "${mname}_seed${s}" "$tdir/seed_sweep/seed_${s}"
  done
done
# deterministic systems
for spec in "${DET_CONTROL[@]}"; do
  IFS='|' read -r dname ddir <<< "$spec"
  run_control "$dname" "$ddir"
done
STATUS[stage2]=$([ "$stage2_ok" = "1" ] && echo OK || echo PARTIAL)

# =====================================================================================
# Stage 3 - parameter-matched direct baseline (optional; only training job)
# =====================================================================================
if [ "$RUN_STAGE3" = "1" ]; then
  banner "STAGE 3 - parameter-matched direct baseline (d_model 1024, 8 layers ~= 100M)"
  stage3_ok=1
  if [ "$FORCE" != "1" ] && [ -f "$S3_DIR/best.pt" ]; then
    log "skip training (exists): $S3_DIR/best.pt"
  else
    log "training parameter-matched direct regressor -> $S3_DIR"
    "$PY" standalone_direct_pca_regressor.py \
      --cache-root "$CACHE" --out-dir "$S3_DIR" \
      --epochs 150 --batch-size 4 --eval-batch-size 4 --num-workers 4 \
      --d-model 1024 --num-layers 8 --num-heads 8 \
      --loss huber --huber-beta 0.25 --export-val-predictions 0 \
      --seed 1234 --device "$DEVICE" --overwrite \
      >>"$LOG" 2>&1 || { log "WARN: matched-baseline training failed"; stage3_ok=0; }
  fi
  if [ "$stage3_ok" = "1" ]; then
    if [ "$FORCE" != "1" ] && [ -f "$S3_DIR/test_set_predictions/manifest.jsonl" ]; then
      log "skip matched-baseline export (exists)"
    else
      "$PY" scripts/export_direct_pca_predictions.py \
        --run-dir "$S3_DIR" --cache-root "$CACHE" --split test \
        --out-dir "$S3_DIR/test_set_predictions" \
        --batch-size 8 --num-workers 2 --device "$DEVICE" --overwrite \
        >>"$LOG" 2>&1 || { log "WARN: matched-baseline export failed"; stage3_ok=0; }
    fi
  fi
  if [ "$stage3_ok" = "1" ]; then
    "$PY" scripts/run_diffusion_acoustic_eval.py --skip-export \
      --prediction-dirs "$S3_DIR/test_set_predictions" \
      --prediction-names direct_pca_d1024_l8_seed1234 \
      --cache-root "$CACHE" --out-dir "$S3_EVAL" \
      --fad-model "$FAD_MODEL" --fad-workers 1 --fad-inf-workers 1 \
      --batch-size 16 --num-workers 4 --device "$DEVICE" --no-plots --overwrite \
      >>"$LOG" 2>&1 || { log "WARN: matched-baseline acoustic eval failed"; stage3_ok=0; }
    run_control "direct_pca_d1024_l8_seed1234" "$S3_DIR/test_set_predictions"
  fi
  STATUS[stage3]=$([ "$stage3_ok" = "1" ] && echo OK || echo PARTIAL)
else
  banner "STAGE 3 - skipped (RUN_STAGE3=0)"
  STATUS[stage3]=SKIPPED
fi

# =====================================================================================
# Aggregate machine-readable summaries for the paper edits
# =====================================================================================
banner "AGGREGATE"
"$PY" scripts/aggregate_overnight.py \
  --overnight-root "$OVERNIGHT" --seeds "$SEEDS" \
  >>"$LOG" 2>&1 && STATUS[aggregate]=OK || { log "WARN: aggregation failed"; STATUS[aggregate]=PARTIAL; }

# =====================================================================================
banner "DONE - per-stage status"
for k in stage1 stage2 stage3 aggregate; do
  log "  ${k}: ${STATUS[$k]:-?}"
done
log "artifacts under: $OVERNIGHT"
log "next: python scripts/build_paper_results.py --out-dir paper_results --strict"
