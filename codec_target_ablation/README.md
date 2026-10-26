# Codec target ablation

These controls use the same GMD splits, 250 Hz expressive drum grid, and
frozen DAC as the paper's plain 72-D post-snap diffusion model.

| Export | Target and decoding | Parameters | Selected epoch (zero based) |
| --- | --- | ---: | ---: |
| `tokens_9x1024` | Nine parallel 1024-way classifiers, cross-entropy training, argmax codes | 85,871,104 | 7 |
| `postsnap_full1024` | Diffusion in standardized 1024-D post-snap latents | 93,316,608 | 147 |
| `postsnap_full1024_projected` | Same checkpoint and sampling seeds, with output projected onto the exact 72-D affine subspace before DAC decoding | 93,316,608 | 147 |

The token history records 28 epochs; validation cross-entropy is lowest at
epoch 7 and increases afterwards. The 1024-D history records all 150
epochs. `run.sh` configures a maximum of 150 epochs for both models.
The token arm tests parallel classification, rather than autoregressive
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

The paper's listening page is maintained on the `anonymous` branch.
Extend its existing twelve examples from the evaluated exports with:

```bash
python codec_target_ablation/update_demo.py --docs-root /path/to/anonymous-checkout/docs
```

This requires NumPy, SoundFile, and pyloudnorm. It adds the plain 72-D
model and all three controls, matching each file by recording ID and beat
index. Audio follows the existing page's BS.1770 loudness matching with a
0.999 peak limit. `docs/codec_target_examples.json` records source exports
and gains for every new audio file. Repeating the command updates the same
players without duplicating them.
