# Code

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

## Cache Creation

Build the aligned source cache from local dataset files:

```bash
cd code/experiment
python scripts/build_source_cache.py --source-root /path/to/e-gmd --out-root ../../runs/source_cache
```

Build the framewise diffusion cache from a source cache:

```bash
python scripts/build_diffusion_cache.py \
  --source-cache-root ../../runs/source_cache \
  --out-root ../../runs/diffusion_cache \
  --split train
```

The packaged repo includes only `runs/mini_cache` for demo/smoke inference.
Full cache creation requires the external dataset files.

## Training

Train the diffusion model:

```bash
cd code/experiment
python train_cli.py --cache-root ../../runs/diffusion_cache --out-dir ../../runs/model_train
```

Train the sketch expander:

```bash
python train_sketch_expander_cli.py --cache-root ../../runs/diffusion_cache --out-dir ../../runs/sketch_expander_train
```

Train the direct PCA baseline:

```bash
python standalone_direct_pca_regressor.py --cache-root ../../runs/diffusion_cache --out-dir ../../runs/runs_direct/direct_pca_regressor
```

## Evaluation

Export diffusion predictions:

```bash
python scripts/export_best_diffusion_predictions.py \
  --train-dir ../../runs/runs_dac/dac_25steps \
  --cache-root ../../runs/mini_cache \
  --out-dir ../../results/smoke_predictions \
  --max-items 1 \
  --overwrite
```

Evaluate exported WAVs:

```bash
python scripts/evaluate_diffusion_predictions.py \
  --cache-root ../../runs/mini_cache \
  --predictions-dir ../../results/smoke_predictions \
  --max-items 1 \
  --overwrite
```

Aggregate existing paper result artifacts:

```bash
python scripts/build_paper_results.py --repo-root ../../runs --out-dir ../../results/paper_results
```

## Demo

Run a deterministic CLI smoke test:

```bash
cd code/demo
python scripts/sketch_diffusion_infer.py \
  --sketch-json smoke_sketch.json \
  --sketch-checkpoint ../../runs/sketch_expander_dac44_native_v5/best_sketch_expander.pt \
  --diffusion-train-dir ../../runs/runs_dac/dac_25steps \
  --cache-root ../../runs/mini_cache \
  --device cpu \
  --out-dir /tmp/drumtogrid_smoke_audio \
  --overwrite
```

Run the verified CLI demo smoke:

```bash
python scripts/sketch_diffusion_infer.py \
  --sketch-json smoke_sketch.json \
  --sketch-checkpoint ../../runs/sketch_expander_dac44_native_v5/best_sketch_expander.pt \
  --diffusion-train-dir ../../runs/runs_dac/dac_25steps \
  --cache-root ../../runs/mini_cache \
  --device cpu \
  --out-dir /tmp/drumtogrid_smoke_audio \
  --overwrite
```
