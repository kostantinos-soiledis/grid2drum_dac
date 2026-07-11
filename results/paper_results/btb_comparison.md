# Break-the-Beat comparison

Lower is better for FAD and RMS error; higher is better for onset F1,
CMLt, and AMLt.

| System | Evaluation set | n | FAD-VGGish ↓ | FAD-CLAP ↓ | Onset F1 ↑ | RMS error ↓ | CMLt ↑ | AMLt ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Break-the-Beat proposed | E-GMD, Beat + Fill | not reported separately | 0.18 | 0.072 | 60.91% | 13.32 dBFS | 0.43 | 0.62 |
| Break-the-Beat proposed | E-GMD + StemGMD | 791 | 0.09 | 0.061 | 70.08% | 10.53 dBFS | 0.42 | 0.51 |
| Ours v4 proxy | E-GMD held-out kits | 109 | 0.947127 | 0.066156 | 92.94% | 4.14 dBFS | 0.674 | 0.748 |

## Interpretation

Against the E-GMD-only Break-the-Beat row, ours is close on CLAP FAD
(0.066 versus 0.072), stronger on onset alignment, RMS dynamics, and beat
continuity, and substantially weaker on VGGish FAD (0.947 versus 0.18).

This is a protocol-matched proxy, not a controlled head-to-head result. Our
evaluation uses 109 four-beat, mono, DAC-decoded held-out-kit transfers. The
Break-the-Beat paper uses 791 two-bar stereo E-GMD/StemGMD pairs, a different
kit split, and native waveform references. FAD is sample-count dependent.
Their exact test manifest and evaluation implementation are not released.

The CLAP score uses the paper's named LAION music checkpoint,
`music_audioset_epoch_15_esc_90.14`. VGGish clips produce one embedding frame
for many short examples, which exposes a NaN bug in fadtk's per-file online
covariance path. The reported VGGish score pools all 202 frames before
estimating the covariance; this is the ordinary FAD estimator.

Generated metrics:

- Checkpoint: `runs_dac_native/egmd43_refenc_cnn_v4/best_task.pt`
- Reference encoder:
  `runs_dac_native/egmd43_refenc_cnn_v4/best_task_encoder.pt`
- Generator: `scripts/btb_compare_gen.py`
- Alignment/continuity scorer: `scripts/btb_score.py`
- Robust FAD scorer: `scripts/btb_fad_score.py`
