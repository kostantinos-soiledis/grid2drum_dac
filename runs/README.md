# Runs

## Final experiment: `runs/final/`

Every arm was trained for 150 epochs with seed 1234; the checkpoint with the
lowest validation loss is the one evaluated.

| Path | Purpose |
| --- | --- |
| `postsnap_pca/` | 25-step diffusion on 72-D PCA of the post-snap DAC latent |
| `presnap_latent/` | 25-step diffusion on native 72-D pre-snap DAC projected latents; no PCA |
| `regression_6layer/` | deterministic direct-regression baseline |
| `regression_8layer_capacity/` | parameter-capacity control for the regression baseline |
| `grid_hz/{90hz,120hz,250hz,500hz}/` | secondary conditioning-rate check; 500 Hz uses a MIDI-rerendered overlay |
| `exports/` | decoded test-set predictions reused by the evaluation (manifests and summaries only) |

The pre-snap targets are native DAC projected latents (layout
`dac_projected_latents_presnap`): nine 8-D quantizer projections before
nearest-codebook snapping, concatenated. They are not PCA coordinates.

Run [`run_final_results.sh`](../run_final_results.sh) from the repository root.
It resumes incomplete runs and skips only runs whose history reached epoch 149.
The 500 Hz run uses microbatch 1 with four-step gradient accumulation to keep
the effective training batch at 4 on 10 GiB GPUs.

Only configs, histories and export summaries are versioned here; the
checkpoints themselves are not.

## Demo runtime files

[`code/demo`](../code/demo) loads its models from the original paths below,
committed through Git LFS and listed in [`weights/manifest.json`](weights/manifest.json).
The three checkpoints are byte-identical to final arms:

| Demo path | Final arm |
| --- | --- |
| `runs_dac/dac_25steps/` | `final/postsnap_pca/` |
| `runs_dac_ce/dac_25steps/` | `final/grid_hz/250hz/` |
| `runs_direct/direct_pca_d1024_l6_seed1234/` | `final/regression_6layer/` |

`mini_cache/`, `sketch_expander_dac44_native_v5/` and `third_party/dac_44khz/`
hold the demo's cache statistics, sketch expander and DAC codec weights.
