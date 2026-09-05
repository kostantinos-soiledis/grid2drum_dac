#!/usr/bin/env bash
# Refresh the paper artifacts that are both claim-relevant and reproducible from
# completed runs. This script does not train models or rerun the already-complete
# main acoustic sweep. Set FORCE=1 to recompute the transferred capacity control.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$CODE_ROOT/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PCA_DIFFUSION_ROOT="${PCA_DIFFUSION_ROOT:-$WORKSPACE_ROOT/../pca_diffusion}"
EXPERIMENT_RUNS_ROOT="${EXPERIMENT_RUNS_ROOT:-$WORKSPACE_ROOT/runs}"
PAPER_RESULTS_ROOT="${PAPER_RESULTS_ROOT:-$WORKSPACE_ROOT/results/paper_results}"
PAPER_FIGURE_ROOT="${PAPER_FIGURE_ROOT:-$WORKSPACE_ROOT/paper/figures}"
DEVICE="${DEVICE:-auto}"
FORCE="${FORCE:-0}"
FAD_MODEL="${FAD_MODEL:-clap-laion-music}"

SIBLING_CACHE="$PCA_DIFFUSION_ROOT/cache_4beats_dac44q9_pca72_native_bpmgeom_duration_v1"
DAC_CACHE_ROOT="${DAC_CACHE_ROOT:-$SIBLING_CACHE}"
PAPER_METADATA_CACHE_ROOT="${PAPER_METADATA_CACHE_ROOT:-$DAC_CACHE_ROOT}"
CAPACITY_NAME="direct_pca_d1024_l8_seed1234"
CAPACITY_PREDICTIONS="${CAPACITY_PREDICTIONS:-$PCA_DIFFUSION_ROOT/runs_direct/$CAPACITY_NAME/test_set_predictions}"
CAPACITY_OUTPUT="$PAPER_RESULTS_ROOT/overnight/matched_baseline_eval"
FRONTEND_INFERENCE_ROOT="${FRONTEND_INFERENCE_ROOT:-$PCA_DIFFUSION_ROOT/eval/frontend_ablation}"
FRONTEND_TRAINING_ROOT="${FRONTEND_TRAINING_ROOT:-$PCA_DIFFUSION_ROOT/eval/frontend_training_ablation}"

require_file() {
  if [ ! -f "$1" ]; then
    echo "missing required file: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [ ! -d "$1" ]; then
    echo "missing required directory: $1" >&2
    exit 1
  fi
}

require_file "$PYTHON_BIN"
require_dir "$EXPERIMENT_RUNS_ROOT"
mkdir -p "$PAPER_RESULTS_ROOT/overnight" "$PAPER_FIGURE_ROOT/ablation"
cd "$CODE_ROOT"

echo "[1/5] Parameter-matched direct-model capacity control"
if [ "$FORCE" = "1" ] || [ ! -f "$CAPACITY_OUTPUT/acoustic_eval/overall_summary.csv" ]; then
  require_dir "$DAC_CACHE_ROOT/examples/test"
  require_file "$CAPACITY_PREDICTIONS/manifest.jsonl"
  "$PYTHON_BIN" scripts/run_diffusion_acoustic_eval.py \
    --skip-export \
    --prediction-dirs "$CAPACITY_PREDICTIONS" \
    --prediction-names "$CAPACITY_NAME" \
    --cache-root "$DAC_CACHE_ROOT" \
    --out-dir "$CAPACITY_OUTPUT" \
    --fad-model "$FAD_MODEL" \
    --fad-workers 1 --fad-inf-workers 1 --fad-repeats 8 \
    --batch-size 16 --num-workers 4 --device "$DEVICE" \
    --no-plots --overwrite
else
  echo "  retained completed evaluation: $CAPACITY_OUTPUT"
fi

echo "[2/5] PCA-versus-native subspace paired statistics"
require_file "$PAPER_RESULTS_ROOT/native_subspace_eval/acoustic_eval/per_clip_metrics.csv"
require_file "$PAPER_RESULTS_ROOT/full_acoustic_eval/acoustic_eval/per_clip_metrics.csv"
require_file "$PAPER_RESULTS_ROOT/native_subspace_eval_direct/direct_audio_eval/dac_native_25steps/per_clip_metrics.csv"
require_file "$PAPER_RESULTS_ROOT/full_acoustic_eval/direct_audio_eval/diffusion_pca_25steps/per_clip_metrics.csv"
"$PYTHON_BIN" scripts/native_subspace_paired_test.py \
  --native-acoustic "$PAPER_RESULTS_ROOT/native_subspace_eval/acoustic_eval/per_clip_metrics.csv" \
  --pca-acoustic "$PAPER_RESULTS_ROOT/full_acoustic_eval/acoustic_eval/per_clip_metrics.csv" \
  --native-direct "$PAPER_RESULTS_ROOT/native_subspace_eval_direct/direct_audio_eval/dac_native_25steps/per_clip_metrics.csv" \
  --pca-direct "$PAPER_RESULTS_ROOT/full_acoustic_eval/direct_audio_eval/diffusion_pca_25steps/per_clip_metrics.csv" \
  --native-model dac_native_25steps --pca-model diffusion_pca_25steps \
  --seed 1234 --n-boot 2000 --n-perm 2000 \
  | tee "$PAPER_RESULTS_ROOT/native_subspace_paired_stats.txt"

echo "[3/5] Conditioning-ablation tables, paired tests, and figure"
if [ -f "$FRONTEND_INFERENCE_ROOT/suite_summary.json" ] && [ -f "$FRONTEND_TRAINING_ROOT/suite_summary.json" ]; then
  "$PYTHON_BIN" scripts/build_frontend_ablation_paper_assets.py \
    --inference-root "$FRONTEND_INFERENCE_ROOT" \
    --training-root "$FRONTEND_TRAINING_ROOT" \
    --out-dir "$PAPER_RESULTS_ROOT" \
    --figure-dir "$PAPER_FIGURE_ROOT/ablation" \
    --bootstrap-samples 2000 --permutation-samples 2000 --seed 1234
else
  echo "  raw frontend suites unavailable; retained the existing derived artifacts"
fi

echo "[4/5] Sampling-seed and event-control summaries"
"$PYTHON_BIN" scripts/aggregate_overnight.py \
  --overnight-root "$PAPER_RESULTS_ROOT/overnight"

echo "[5/5] Canonical paper tables and manifests"
require_file "$PAPER_METADATA_CACHE_ROOT/config.json"
require_file "$PAPER_METADATA_CACHE_ROOT/summaries/train.json"
require_file "$PAPER_METADATA_CACHE_ROOT/summaries/validation.json"
require_file "$PAPER_METADATA_CACHE_ROOT/summaries/test.json"
"$PYTHON_BIN" scripts/build_paper_results.py \
  --repo-root "$EXPERIMENT_RUNS_ROOT" \
  --cache-root "$PAPER_METADATA_CACHE_ROOT" \
  --out-dir "$PAPER_RESULTS_ROOT" \
  --strict

echo "Paper results refreshed under: $PAPER_RESULTS_ROOT"
