# Codec target ablation

These controls use the same GMD splits, 250 Hz expressive drum grid, and
frozen DAC as the paper's plain 72-D post-snap diffusion model.

## Original 1024-D results are invalid as a representation comparison

The original `postsnap_full1024` run used `x_dim=1024`, `d_model=768`,
and a linear 768-to-1024 epsilon head. Isotropic Gaussian noise spans all
1024 coordinates even though clean DAC outputs occupy a 72-D affine
subspace. At least 256 noise directions cannot pass through that head:
the expected per-coordinate noise MSE is bounded below by 256/1024 = 0.25.
The selected checkpoint's validation loss is 0.2612356. Output projection
does not fix the denoiser or the preceding diffusion trajectory.

These artifacts remain as diagnostics. They do **not** establish that
diffusion in the 1024-D post-snap space performs poorly when implemented
without this architectural bottleneck. The manuscript excludes these
results until a corrected rerun is available.

| Export | Target and decoding | Parameters | Selected epoch (zero based) |
| --- | --- | ---: | ---: |
| `tokens_9x1024` | Nine parallel 1024-way classifiers, cross-entropy training, argmax codes | 85,871,104 | 7 |
| `postsnap_full1024` | Invalid width-768 full-space configuration | 93,316,608 | 147 |
| `postsnap_full1024_projected` | Same invalid configuration with final output projection | 93,316,608 | 147 |

The token history records 28 epochs; validation cross-entropy is lowest at
epoch 7 and increases afterwards. The 1024-D history records all 150
epochs.
The token arm tests parallel classification, rather than autoregressive
or iterative token generation.

`results/evaluation/acoustic_eval/overall_summary.csv` contains the mel,
flux, and FAD∞ scores; `direct_audio_eval/*/summary.json` contains waveform
L1 and MR-STFT. `results/stats/` contains paired tests and confidence
intervals clustered by the 71 test recordings. References are the same DAC
reconstructions used by `results/final`.

## Corrected rerun

`run.sh` now defaults to `full1024,pca72_matched`: both diffusion models
use width 1280, six blocks, 25 steps, and batch size 1 with four gradient
accumulation steps (effective batch size 4). This width also leaves room
for the final LayerNorm, whose normalized features have one affine
constraint. The 72-D control has the same width to avoid confounding
representation and backbone capacity. Clipping is explicitly 6 in both
training previews and exports. Training rejects `d_model <= x_dim` before
an output directory can be overwritten.
The full-space and matched 72-D models have 256.1 M and 253.6 M parameters,
respectively; this is a larger backbone than the original 768-wide model.

```bash
python codec_target_ablation/audit_full1024.py --forward --decode
bash codec_target_ablation/run.sh --device cuda:2
```

The corrected models are `postsnap_full1024_d1280` and
`postsnap_pca72_d1280`; the projection export is
`postsnap_full1024_d1280_projected`. Metrics and paired tests go to
`results/rerun_d1280/`. These paths preserve the original failed runs and
prevent their checkpoints or metrics from being silently reused.
`--diffusion-width` changes both models together and must exceed 1024
and be divisible by eight. Token experiments remain available with
`--arms tokens`.

The audit writes `results/audit_full1024.json`, checking all split
identities, sampled grid/code/latent tensors, normalization, PCA
reconstruction, frozen-DAC decoding, and the trained noise head.

Checkpoints, rendered audio, decoded caches, and FAD assets are ignored.
Scripts, configs, histories, CSV metrics, and statistics remain visible to
Git. The full pipeline can regenerate local artifacts with `run.sh`.

The paper's listening page is maintained on the `anonymous` branch.
Extend its existing twelve examples from the evaluated exports with:

```bash
python codec_target_ablation/update_demo.py --docs-root /path/to/anonymous-checkout/docs
```

This requires NumPy, SoundFile, and pyloudnorm. After the rerun, it adds the matched 72-D
model and all three controls, matching each file by recording ID and beat
index. Audio follows the existing page's BS.1770 loudness matching with a
0.999 peak limit. `docs/codec_target_examples.json` records source exports
and gains for every new audio file. Repeating the command updates the same
players without duplicating them.

Until then, mark the existing clips as an invalid configuration using
`--mark-legacy-invalid`. That action changes labels and provenance without
generating or replacing audio.
