# Audit of the original 1024-D post-snap experiment

The original full-space experiment has an architectural error and must
not be used to conclude that the 1024-D representation fails. Both its
plain and projected exports use the same affected checkpoint.

## Confirmed cause

`run.sh` reused the 72-D model's width of 768 while changing its input
and epsilon output to 1024 dimensions. The checkpoint confirms:

| Projection | Weight shape |
| --- | --- |
| `x_proj.weight` | 768 × 1024 |
| `out_proj.weight` | 1024 × 768 |

For output `W h + b`, all predictions lie in a fixed affine subspace of
dimension at most 768. The target epsilon is isotropic 1024-D Gaussian
noise. Its expected squared error outside that subspace is at least 256,
giving a per-coordinate MSE floor of **0.25**, regardless of training
length. The input projection also discards 256 directions. The final
LayerNorm can impose a further affine constraint at a fixed timestep.

The selected epoch-147 checkpoint has validation noise MSE **0.2612356**.
Monte Carlo projection of Gaussian noise onto its actual output span
leaves MSE **0.2507440**. A forward check on one validation excerpt shows:

| Diffusion step | Noise MSE | MSE outside output span | Clean-estimate MSE before clipping | Clipped at ±6 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.263864 | 0.251331 | 0.001441 | 0% |
| 12 | 0.261496 | 0.250636 | 0.303648 | 0% |
| 24 | 0.260004 | 0.248973 | 66,994.7 | 98.08% |

At the terminal step, alpha-bar is 3.88095e-6. The conversion from epsilon
to the clean estimate amplifies the irreducible error by roughly 500 in
amplitude. Final output projection cannot repair the denoiser or replay
its corrupted reverse trajectory.

## Other implementation paths checked

- All 11,523 train, 1,534 validation, and 1,733 test manifest identities,
  durations, and frame counts match between the 72-D and 1024-D caches.
- Three tensor examples per split have identical raw targets, codec IDs,
  grids, articulation IDs, and grid/codec timestamps.
- Checkpoint normalization exactly matches the 1024-D cache; its smallest
  standard deviation is 2.63529. Normalize/denormalize round trips pass.
- The sampled PCA reconstruction error is at most 3.54e-5 in relative L2,
  consistent with finite-precision cached codec outputs.
- On a 32-frame target, frozen-DAC audio decoded from raw latents versus
  reconstructed PCA coordinates differs by only **1.23e-6 relative L2**.
  The actual output-projection helper gives the same decoder agreement.

The full machine-readable audit is in `results/audit_full1024.json`.
Reproduce it with:

```bash
python codec_target_ablation/audit_full1024.py --forward --decode
```

## Rerun correction

Both the 1024-D model and a matched 72-D control use width **1280**, six
blocks, the original epsilon objective and 25-step schedule, explicit
clipping at 6, and effective batch size four (batch one, accumulation
four). Width exceeds the noise dimension and allows the final normalized
features to span the output. New run, export, and evaluation directories
prevent reuse or overwriting of the original artifacts.

The training CLI now rejects `d_model <= x_dim` before device setup or
output-directory modification. Historical checkpoints remain loadable
for diagnosis. Projection remains after denormalization and before DAC
decoding; exports now record that operation in their summary.

The corrected full-space model has 256,067,072 parameters; the matched
72-D model has 253,629,000. A CPU forward/backward/AdamW step using the
full six-block model passed finite-loss, gradient, and parameter checks.
After that step, its output matrix has rank 1024, with smallest
singular value 0.0622. This implementation smoke check is recorded in
`results/rerun_smoke.json`; it is not an audio-quality evaluation. Five
regression tests cover early rejection before overwrite, the width
requirement, affine projection/padding, fresh matched rerun commands, and
safe, idempotent correction labels on the listening page.

```bash
bash codec_target_ablation/run.sh --device cuda:2
```

This rerun is required before reporting a valid 72-D versus 1024-D result.
The code fixes do not establish the corrected model's audio quality.
