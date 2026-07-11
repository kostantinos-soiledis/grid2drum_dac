# Overnight foundation results

## Sampling-seed stability (unified acoustic pipeline)

| model | n_seeds | mel_mae_db (mean+/-std) | onset_flux_cosine (mean+/-std) | band_balance_l1 (mean+/-std) | fad_inf (mean+/-std) |
| --- | --- | --- | --- | --- | --- |
| diffusion_pca_25steps | 5 | 5.6563 +/- 0.0231 | 0.8499 +/- 0.0009 | 0.0338 +/- 0.0005 | 0.0199 +/- 0.0002 |
| diffusion_pca_rvq_ce_12steps | 5 | 5.4011 +/- 0.0202 | 0.8640 +/- 0.0006 | 0.0336 +/- 0.0005 | 0.0205 +/- 0.0003 |

## Control faithfulness (fixed onset diagnostic)

| model | n | macro F1 | macro timing (ms) | kick F1 | snare F1 | tom F1 | hihat F1 | cymbal F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| diffusion_pca_25steps | 5 | 0.377 | 8.5 | 0.374 | 0.444 | 0.137 | 0.687 | 0.244 |
| diffusion_pca_rvq_ce_12steps | 5 | 0.370 | 8.1 | 0.364 | 0.436 | 0.131 | 0.682 | 0.237 |
| direct_pca_d1024_l6 | 1 | 0.436 | 7.2 | 0.548 | 0.470 | 0.180 | 0.722 | 0.262 |
| direct_pca_d1024_l8 | 1 | 0.438 | 7.4 | 0.555 | 0.479 | 0.180 | 0.718 | 0.258 |
| grid_render | 1 | 0.433 | 5.1 | 0.504 | 0.557 | 0.182 | 0.690 | 0.231 |
| symbolic_nn_train | 1 | 0.258 | 20.5 | 0.281 | 0.312 | 0.096 | 0.422 | 0.180 |
