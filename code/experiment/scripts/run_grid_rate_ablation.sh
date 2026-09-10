#!/usr/bin/env bash
# Grid-resolution ablation: 90 / 120 Hz vs the existing 250 Hz run.
#
# The cache is rendered once at 250 Hz; coarser rates are decimated at load time
# via --grid-rate-hz, so no extra caches are built and the audio/target side is
# identical across arms. The 250 Hz arm is the existing
# our_hybrid_multiscale_25steps run and is NOT retrained here.
#
# Radii are scaled to hold the receptive fields fixed in seconds (~88/164/220 ms):
#   250 Hz -> 0,22,41,55   (existing run)
#   120 Hz -> 0,11,20,26
#    90 Hz -> 0,8,15,20
#
# Usage:
#   run_grid_rate_ablation.sh [--device cuda:2] [--rates "90 120"] [--smoke] [--dry-run]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP_DIR="${REPO_ROOT}/code/experiment"

CACHE_ROOT="${GRID_ABLATION_CACHE_ROOT:-${REPO_ROOT}/../pca_diffusion/cache_4beats_dac44q9_pca72_native_bpmgeom_duration_v1}"
OUT_ROOT="${GRID_ABLATION_OUT_ROOT:-${REPO_ROOT}/runs/runs_dac_ce/grid_rate_ablation}"
DEVICE="cuda:2"
RATES="90 120"
SMOKE=0
DRY_RUN=0

# Held fixed to match the existing 250 Hz run (its run_config.json).
EPOCHS=75
BATCH_SIZE=4
EVAL_BATCH_SIZE=4
SEED=1234
LR=1e-4
NUM_STEPS=25
D_MODEL=768
NUM_LAYERS=6
NUM_HEADS=8
NUM_WORKERS=4
# Matches the existing 250 Hz run; without it that arm carries an extra loss
# term and the comparison has two variables in it.
RVQ_CE_WEIGHT=0.1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)     DEVICE="$2"; shift 2 ;;
    --rates)      RATES="$2"; shift 2 ;;
    --cache-root) CACHE_ROOT="$2"; shift 2 ;;
    --out-root)   OUT_ROOT="$2"; shift 2 ;;
    --epochs)     EPOCHS="$2"; shift 2 ;;
    --smoke)      SMOKE=1; shift ;;
    --dry-run)    DRY_RUN=1; shift ;;
    -h|--help)    sed -n '2,16p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

radii_for_rate() {
  case "$1" in
    90)  echo "0,8,15,20" ;;
    120) echo "0,11,20,26" ;;
    250) echo "0,22,41,55" ;;
    *)   echo "no radii defined for rate $1 Hz" >&2; return 1 ;;
  esac
}

primary_for_rate() {
  case "$1" in
    90) echo 8 ;; 120) echo 11 ;; 250) echo 22 ;;
    *)  echo "no primary radius for rate $1 Hz" >&2; return 1 ;;
  esac
}

if [[ ! -d "${CACHE_ROOT}" ]]; then
  echo "cache root not found: ${CACHE_ROOT}" >&2
  exit 1
fi

# A smoke run trains a couple of epochs on a slice, to prove the pipeline end to
# end without holding the GPU for hours.
if [[ "${SMOKE}" -eq 1 ]]; then
  EPOCHS=2
  MAX_TRAIN_ITEMS=64
  MAX_VAL_ITEMS=32
  OUT_ROOT="${OUT_ROOT}_smoke"
else
  MAX_TRAIN_ITEMS=0
  MAX_VAL_ITEMS=0
fi

echo "=============================================="
echo " grid-rate ablation"
echo "   device      : ${DEVICE}"
echo "   rates       : ${RATES} Hz  (250 Hz reuses the existing run)"
echo "   cache       : ${CACHE_ROOT}"
echo "   out root    : ${OUT_ROOT}"
echo "   epochs      : ${EPOCHS}   seed: ${SEED}   rvq_ce: ${RVQ_CE_WEIGHT}"
[[ "${SMOKE}" -eq 1 ]] && echo "   MODE        : SMOKE (${MAX_TRAIN_ITEMS} train items)"
echo "=============================================="

mkdir -p "${OUT_ROOT}"

for RATE in ${RATES}; do
  RADII="$(radii_for_rate "${RATE}")"
  PRIMARY="$(primary_for_rate "${RATE}")"
  RUN_DIR="${OUT_ROOT}/grid${RATE}hz_25steps"

  echo
  echo "---- ${RATE} Hz  (radii ${RADII}, primary ${PRIMARY}) -> ${RUN_DIR}"

  CMD=(python -u "${EXP_DIR}/train_cli.py"
    --cache-root "${CACHE_ROOT}"
    --out-dir "${RUN_DIR}"
    --device "${DEVICE}"
    --grid-rate-hz "${RATE}"
    --frontend-radii "${RADII}"
    --frontend-primary-radius "${PRIMARY}"
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --eval-batch-size "${EVAL_BATCH_SIZE}"
    --seed "${SEED}"
    --lr "${LR}"
    --num-steps "${NUM_STEPS}"
    --d-model "${D_MODEL}"
    --num-layers "${NUM_LAYERS}"
    --num-heads "${NUM_HEADS}"
    --num-workers "${NUM_WORKERS}"
    --rvq-ce-weight "${RVQ_CE_WEIGHT}"
    --use-bpm-training-geometry
  )
  [[ "${MAX_TRAIN_ITEMS}" -gt 0 ]] && CMD+=(--max-train-items "${MAX_TRAIN_ITEMS}")
  [[ "${MAX_VAL_ITEMS}"   -gt 0 ]] && CMD+=(--max-val-items   "${MAX_VAL_ITEMS}")

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '  %q' "${CMD[@]}"; echo
    continue
  fi

  mkdir -p "${RUN_DIR}"
  # Runs are sequential on purpose: one run peaks near 9.3 GB, so two will not
  # share a 10 GB card.
  ( cd "${EXP_DIR}" && "${CMD[@]}" ) 2>&1 | tee "${RUN_DIR}/train.log"
  echo "  done: ${RUN_DIR}"
done

echo
echo "all requested rates finished."
