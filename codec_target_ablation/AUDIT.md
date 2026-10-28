# Why the 1024-D post-snap latent is not a diffusion target

The paper (Sec. 4.2, Ablations) diffuses in the 72 PCA coordinates of $z_q$ rather than
in its 1024-D space. These are the measurements behind that paragraph.
`audit_full1024.py` wrote `results/audit_full1024.json`; the remaining
values come from the computation in `diagnose_full1024.py`. The 1024-D
checkpoints it was applied to are not kept; the variance and
`postsnap_pca` values are reproduced by the committed script.

## Same information

All 11,523 train, 1,534 validation, and 1,733 test identities, durations,
and frame counts match between the 72-D and 1024-D caches. The PCA change
of coordinates reconstructs sampled frames to at most 3.54e-5 relative L2,
and frozen-DAC audio decoded from raw latents and from reconstructed PCA
coordinates differs by 1.23e-6 relative L2.

## Empty dimensions need a wider noise head

Isotropic noise spans all 1024 coordinates, while $z_q$ spans 72. A linear
epsilon head of width $d < 1024$ cannot represent $1024-d$ noise
directions: at $d = 768$ the noise MSE is bounded below by 0.25. The
width-768 run reached 0.2612 validation noise MSE, and Gaussian noise
projected outside its output span leaves 0.2507. The trained head kept
only **4.3%** of the 72-D data subspace in its span; a random 72-D
subspace keeps 75.1%. Its principal-angle cosines have median 0.036 and
maximum 0.687. Training kept the noise-only directions, which $x_t$
determines exactly, and dropped those that carry the drums.

## Per-coordinate standardization raises the signal-to-noise ratio

Standardizing each of the 1024 coordinates of 45,824 training frames gives
total variance 1021 on 72 non-zero eigenvalues: mean 14.2 (2.6 to 71.5),
the rest below 1e-5. Each standardized PCA coordinate has variance 1. At
the same step, $x_t$ in the 1024-D space shows $x_0$ at about 14 times the
signal-to-noise ratio, $14.2\,\bar\alpha_t/(1-\bar\alpha_t)$: with the
25-step cosine schedule it exceeds 1 at 20 steps, against 12 for unit
variance. Multiplying every std by $\sqrt{1024/72}$ gives the
72 directions mean variance 1.00 (0.18 to 5.0), but does not restore use of
the grid (below).

## The denoiser ignored the grid

Conditioning features of the training preview excerpt (std over frames
divided by mean magnitude) and the change in the epsilon prediction when
the grid is zeroed, relative to its squared error (summed per frame,
t = 6):

| Checkpoint | Frontend time variation | Grid effect / error |
| --- | ---: | ---: |
| `postsnap_pca` (72-D, epoch 148) | 0.93 | 3.6 / 34.3 |
| 1024-D, width 768, epoch 147 | 0.004 | 1.4e-6 / 266 |
| 1024-D, width 1280, epoch 0 | 0.013 | 2.2e-4 / 84 |
| 1024-D, width 1280, rescaled stds, epoch 0 | 0.021 | 3.0e-4 / 66 |
| 1024-D, width 1280, rescaled stds, epoch 1 | 0.030 | 5.4e-4 / 60 |
| Random initialization (seed 1234) | 0.62 | – |

All parameters of the 1024-D runs were trained; none was frozen. The
frontend output became nearly constant over time within one epoch, and
the width-1280 heads span the whole data subspace. In the healthy
72-D run the frontend output already varies with the beats at epoch 0.
