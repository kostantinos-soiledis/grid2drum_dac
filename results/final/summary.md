# Final results

All tables come from one joint evaluation, so every arm uses the same test clips, decoded targets, metric implementation, and FAD configuration.

## Representation

Does diffusion work better on native pre-snap DAC latents or post-snap PCA latents?

| Arm | Mel MAE ↓ | Onset cosine ↑ | FAD∞ ↓ | Audio L1 ↓ | MR-STFT ↓ | Params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pre-snap native latent | 6.442 | 0.8204 | 0.04925 | 0.05287 | 0.11 | 91.85M |
| post-snap PCA | 5.673 | 0.8517 | 0.01955 | 0.05343 | 0.106 | 91.85M |

## Regression

Does diffusion improve over deterministic direct regression?

| Arm | Mel MAE ↓ | Onset cosine ↑ | FAD∞ ↓ | Audio L1 ↓ | MR-STFT ↓ | Params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| diffusion | 5.673 | 0.8517 | 0.01955 | 0.05343 | 0.106 | 91.85M |
| direct regression | 13.03 | 0.8355 | 0.3544 | 0.0451 | 0.1359 | 76.50M |

## Capacity

Is the direct-regression result explained by model capacity?

| Arm | Mel MAE ↓ | Onset cosine ↑ | FAD∞ ↓ | Audio L1 ↓ | MR-STFT ↓ | Params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 6-layer regression | 13.03 | 0.8355 | 0.3544 | 0.0451 | 0.1359 | 76.50M |
| 8-layer capacity control | 13.51 | 0.8375 | 0.3532 | 0.04474 | 0.1353 | 101.69M |

## Grid Hz

How sensitive is the system to the conditioning-grid rate?

| Arm | Mel MAE ↓ | Onset cosine ↑ | FAD∞ ↓ | Audio L1 ↓ | MR-STFT ↓ | Params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 90 Hz | 5.309 | 0.8416 | 0.02046 | 0.05404 | 0.106 | 91.85M |
| 120 Hz | 5.199 | 0.8471 | 0.01904 | 0.05262 | 0.1031 | 91.85M |
| 250 Hz | 5.471 | 0.8606 | 0.02041 | 0.0517 | 0.1039 | 91.85M |
| 500 Hz (MIDI re-render) | 5.611 | 0.8705 | 0.0188 | 0.04994 | 0.1046 | 91.85M |

## RVQ supervision

Does the training-only RVQ cross-entropy term help post-snap PCA diffusion?

| Arm | Mel MAE ↓ | Onset cosine ↑ | FAD∞ ↓ | Audio L1 ↓ | MR-STFT ↓ | Params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| plain diffusion | 5.673 | 0.8517 | 0.01955 | 0.05343 | 0.106 | 91.85M |
| diffusion + RVQ-CE | 5.471 | 0.8606 | 0.02041 | 0.0517 | 0.1039 | 91.85M |

## Statistical tests

Source-recording-clustered bootstrap intervals and sign-flip tests are written to `stats/acoustic.json` and `stats/direct_audio.json` by `run_final_results.sh`.
