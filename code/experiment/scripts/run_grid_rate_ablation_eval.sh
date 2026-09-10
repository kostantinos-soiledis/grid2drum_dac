#!/usr/bin/env bash
# Acoustic evaluation for the grid-resolution ablation (90 / 120 / 250 Hz).
#
# Each arm is exported on the grid rate it was trained on -- the exporter reads
# grid_rate_hz from the run config, so the 250 Hz baseline decimates nothing --
# and all arms are then scored in a single acoustic-eval pass so FAD and the
# bootstrap CIs share one reference set.
#
# Usage:
#   run_grid_rate_ablation_eval.sh [--device cuda:2] [--max-items 0] [--skip-export]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP_DIR="${REPO_ROOT}/code/experiment"

CACHE_ROOT="${GRID_ABLATION_CACHE_ROOT:-${REPO_ROOT}/../pca_diffusion/cache_4beats_dac44q9_pca72_native_bpmgeom_duration_v1}"
ABLATION_ROOT="${GRID_ABLATION_OUT_ROOT:-${REPO_ROOT}/runs/runs_dac_ce/grid_rate_ablation}"
BASELINE_DIR="${GRID_ABLATION_BASELINE_DIR:-${REPO_ROOT}/runs/runs_dac_ce/dac_25steps}"
OUT_DIR="${GRID_ABLATION_EVAL_OUT:-${REPO_ROOT}/results/eval/grid_rate_ablation}"
# Exports live OUTSIDE OUT_DIR: run_diffusion_acoustic_eval.py --overwrite does an
# rmtree on its --out-dir, so anything staged inside it is deleted before scoring.
EXPORT_ROOT="${GRID_ABLATION_EXPORT_ROOT:-${REPO_ROOT}/results/eval/grid_rate_ablation_exports}"
DEVICE="cuda:2"
# Scoring runs on its own card: fadtk loads CLAP onto the eval device, and sharing
# one 10 GB card with the eval model made the CLAP load fail (laion_clap masks the
# real error behind a bare `except:` reading "Import Model for base not found").
SCORE_DEVICE=""
MAX_ITEMS=0
BATCH_SIZE=4
SAMPLE_SEED=1234
NUM_STEPS=25
FAD_MODEL="${FAD_MODEL:-clap-laion-music}"
SKIP_EXPORT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)       DEVICE="$2"; shift 2 ;;
    --score-device) SCORE_DEVICE="$2"; shift 2 ;;
    --max-items)   MAX_ITEMS="$2"; shift 2 ;;
    --cache-root)  CACHE_ROOT="$2"; shift 2 ;;
    --out-dir)     OUT_DIR="$2"; shift 2 ;;
    --skip-export) SKIP_EXPORT=1; shift ;;
    -h|--help)     sed -n '2,10p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# name|train_dir  -- the 250 Hz arm is the untouched existing run.
ARMS=(
  "grid90hz_25steps|${ABLATION_ROOT}/grid90hz_25steps"
  "grid120hz_25steps|${ABLATION_ROOT}/grid120hz_25steps"
  "grid250hz_25steps|${BASELINE_DIR}"
)

mkdir -p "${OUT_DIR}" "${EXPORT_ROOT}"
# The log also lives outside OUT_DIR so the scoring rmtree cannot truncate it.
LOG="${EXPORT_ROOT}/eval.log"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${LOG}"; }

SCORE_DEVICE="${SCORE_DEVICE:-${DEVICE}}"

log "grid-rate ablation eval | export device=${DEVICE} score device=${SCORE_DEVICE}"
log "cache=${CACHE_ROOT}"

pred_dirs=()
pred_names=()

for spec in "${ARMS[@]}"; do
  IFS='|' read -r name train_dir <<< "${spec}"
  if [[ ! -f "${train_dir}/best_diffusion.pt" ]]; then
    log "WARN: no checkpoint for ${name} (${train_dir}); skipping"
    continue
  fi
  export_dir="${EXPORT_ROOT}/${name}"

  if [[ "${SKIP_EXPORT}" -eq 0 || ! -f "${export_dir}/manifest.jsonl" ]]; then
    log "export ${name} -> ${export_dir}"
    ( cd "${EXP_DIR}" && python -u scripts/export_best_diffusion_predictions.py \
        --train-dir "${train_dir}" \
        --split test \
        --out-dir "${export_dir}" \
        --num-steps "${NUM_STEPS}" \
        --x0-clip-norm 6.0 \
        --num-beats 4 \
        --beat-crossfade-ms 10 \
        --use-bpm-inference-geometry \
        --sample-seed "${SAMPLE_SEED}" \
        --cache-root "${CACHE_ROOT}" \
        --max-items "${MAX_ITEMS}" \
        --batch-size "${BATCH_SIZE}" \
        --num-workers 4 \
        --device "${DEVICE}" \
        --overwrite ) >>"${LOG}" 2>&1 \
      || { log "WARN: export failed for ${name}"; continue; }
  else
    log "skip export (exists): ${export_dir}"
  fi

  if [[ -f "${export_dir}/manifest.jsonl" ]]; then
    pred_dirs+=("${export_dir}")
    pred_names+=("${name}")
  fi
done

if [[ "${#pred_dirs[@]}" -eq 0 ]]; then
  log "ERROR: nothing exported; aborting"
  exit 1
fi

log "scoring ${#pred_dirs[@]} arms: ${pred_names[*]}"
( cd "${EXP_DIR}" && python -u scripts/run_diffusion_acoustic_eval.py --skip-export \
    --prediction-dirs "${pred_dirs[@]}" \
    --prediction-names "${pred_names[@]}" \
    --cache-root "${CACHE_ROOT}" \
    --out-dir "${OUT_DIR}" \
    --fad-model "${FAD_MODEL}" \
    --fad-workers 1 --fad-inf-workers 1 \
    --batch-size "${BATCH_SIZE}" --num-workers 4 \
    --device "${SCORE_DEVICE}" \
    --no-plots --overwrite ) >>"${LOG}" 2>&1 \
  || { log "ERROR: acoustic eval failed"; exit 1; }

log "done -> ${OUT_DIR}/acoustic_eval/overall_summary.csv"
