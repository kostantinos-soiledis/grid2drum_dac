# Codec target ablation

This control uses the same GMD splits, 250 Hz expressive drum grid, and
frozen DAC as the paper's plain 72-D post-snap diffusion model.

| Export | Target and decoding | Parameters | Selected epoch (zero based) |
| --- | --- | ---: | ---: |
| `tokens_9x1024` | Nine parallel 1024-way classifiers, cross-entropy training, argmax codes | 85,871,104 | 7 |

The token history records 28 epochs; validation cross-entropy is lowest at
epoch 7 and increases afterwards. `run.sh` configures a maximum of 150
epochs. The arm tests parallel classification, rather than autoregressive
or iterative token generation.

`results/evaluation/acoustic_eval/overall_summary.csv` contains the mel,
flux, and FAD∞ scores; `direct_audio_eval/*/summary.json` contains waveform
L1 and MR-STFT. `results/stats/` contains paired tests and confidence
intervals clustered by the 71 test recordings. References are the same DAC
reconstructions used by `results/final`.

```bash
bash codec_target_ablation/run.sh --stages eval,stats
```

Checkpoints, rendered audio, decoded caches, and FAD assets are ignored.
Scripts, configs, histories, CSV metrics, and statistics remain visible to
Git. The full pipeline can regenerate local artifacts with `run.sh`.

## Why not diffuse in the 1024-D post-snap latent

The paper diffuses in the 72 PCA coordinates of $z_q$ rather than in its
1024-D space (Sec. 3.6). `AUDIT.md` gives the measurements behind that
paragraph; these two scripts produce them without training anything:

```bash
python codec_target_ablation/audit_full1024.py --checkpoint <1024-D run>/best_diffusion.pt --forward --decode
python codec_target_ablation/diagnose_full1024.py --checkpoint <1024-D run>/best_diffusion.pt \
  --checkpoint runs/final/postsnap_pca/best_diffusion.pt
```

Both read the 1024-D post-snap cache, which the unchanged cache builder
writes with `--pca-k 0`:

```bash
for split in train validation test; do
  python code/experiment/scripts/build_diffusion_cache.py \
    --source-cache-root caches/4beats_dac44q9_native_v1 \
    --out-root caches/cache_4beats_dac44q9_full1024_bpmgeom_duration_v1 --split "$split" --pca-k 0 \
    --geometry-mode bpm --bpm-source duration --target-token-rate-hz 86.1328125 --overwrite
done
```

## Listening page

The paper's listening page is maintained on the `anonymous` branch.
Extend its existing twelve examples from the evaluated exports with:

```bash
python codec_target_ablation/update_demo.py --docs-root /path/to/anonymous-checkout/docs
```

This requires NumPy, SoundFile, and pyloudnorm. It adds the plain 72-D
model and the token control, matching each file by recording ID and beat
index. Audio follows the existing page's BS.1770 loudness matching with a
0.999 peak limit. `docs/codec_target_examples.json` records source exports
and gains for every new audio file. Repeating the command updates the same
players without duplicating them.
