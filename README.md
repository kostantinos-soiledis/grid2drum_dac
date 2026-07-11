# Grid2Drum-DAC

Drum-grid–conditioned audio generation via latent diffusion in a PCA subspace of the DAC codec.

**▶ Listen first: [live demo page](https://kostantinos-soiledis.github.io/grid2drum_dac/)** — side-by-side generated, regressor-baseline, and ground-truth drum audio for held-out examples.

## How it works

![Model overview: training (top) and inference (bottom)](paper/figures/representation/semantic_pca_dac_diffusion_story_cropped.png)

**Training (top):** target audio is encoded by a frozen [DAC](https://github.com/descriptinc/descript-audio-codec) codec (9 RVQ codebooks); the summed codebook embeddings are projected to a normalized 72-dim PCA latent sequence. A trainable multiscale frontend turns the drum grid into a conditioning sequence, and a shared DiT denoiser is trained with noise-prediction MSE (optionally with RVQ-codebook regularization).

**Inference (bottom):** the user-requested drum grid goes through the frontend, reverse diffusion sampling starts from noise guided by that conditioning, and the predicted PCA latent is de-normalized, inverse-projected back to 1024 dims, and decoded to 44.1 kHz audio by the frozen DAC decoder.

Qualitative comparison against the direct PCA-regressor baseline:

![Qualitative spectrogram comparison](paper/figures/qualitative/spectrogram_comparison.png)

## Try the demo

Model weights (~2 GB) ship via Git LFS:

```bash
git lfs install
git clone https://github.com/kostantinos-soiledis/grid2drum_dac.git
# code-only clone, no weights: GIT_LFS_SKIP_SMUDGE=1 git clone <url>
```

The shipped checkpoints are described in [runs/README.md](runs/README.md) and
[runs/weights/manifest.json](runs/weights/manifest.json).

### Smoke test (CPU)

```bash
cd code/demo
pip install -r ../../requirements.txt
python scripts/sketch_diffusion_infer.py \
  --sketch-json smoke_sketch.json \
  --sketch-checkpoint ../../runs/sketch_expander_dac44_native_v5/best_sketch_expander.pt \
  --diffusion-train-dir ../../runs/runs_dac/dac_25steps \
  --cache-root ../../runs/mini_cache \
  --device cpu \
  --out-dir /tmp/drumtogrid_smoke_audio \
  --overwrite
```

This writes `/tmp/drumtogrid_smoke_audio/output.wav`.

### Interactive listener

```bash
python code/demo/app.py
```

Launches the Gradio UI; it expects the runtime files under `runs/` (present
after a full LFS clone).

## Using the repo

The full pipeline is cache → train → evaluate. Every script accepts `--help`
for the complete option list; the commands below show the main entry points.

### 1. Build caches

```bash
# Beat-level source cache from the Groove MIDI Dataset
# (audio -> codec tokens + aligned drum grids)
python code/experiment/scripts/build_source_cache.py \
  --source-root /path/to/gmd \
  --out-root runs/source_cache \
  --codec-family dac \
  --split train \
  --device cuda

# Framewise diffusion cache (PCA targets + seconds-grid conditioning)
python code/experiment/scripts/build_diffusion_cache.py \
  --source-cache-root runs/source_cache \
  --out-root runs/diffusion_cache \
  --split train
```

### 2. Train

```bash
# Latent diffusion model (DiT denoiser + frontend)
python code/experiment/train_cli.py \
  --cache-root runs/diffusion_cache \
  --out-dir runs/my_diffusion \
  --device cuda

# Sketch expander (drum-grid sketch -> conditioning)
python code/experiment/train_sketch_expander_cli.py \
  --cache-root runs/diffusion_cache \
  --out-dir runs/my_sketch_expander
```

### 3. Evaluate

```bash
python code/experiment/scripts/run_diffusion_acoustic_eval.py \
  --checkpoint runs/runs_dac/dac_25steps/best_diffusion.pt \
  --cache-root runs/diffusion_cache \
  --split test
```

Exports predictions and runs the acoustic evaluation (metrics, FAD, plots).
The paper's aggregated metrics and full evaluation outputs live under
[results/paper_results/](results/paper_results/), and
`code/experiment/scripts/build_paper_results.py` reassembles them.
