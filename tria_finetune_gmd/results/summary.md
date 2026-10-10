# TRIA on the GMD test split

1733 clips of 71 test recordings, every metric against the real GMD bars. Each system's clips are gain-matched to their real bar's loudness (ITU-R BS.1770) first, which only FAD and KAD see: the per-clip metrics are means over clips of peak-normalized audio. Onset F1 matches the clip's onsets to the real bar's within ±30 or ±50 ms (librosa's onset detector on both, mir_eval), which a few milliseconds of constant output delay do not change, unlike the onset cosine. FAD∞ is CLAP-LAION-Music over 8 repeats; KAD (kadtk, ×100) uses PANNs-WGLM embeddings with one kernel for all systems (bandwidth from the real bars), 95% CI from a bootstrap over recordings.

| System | Mel MAE ↓ | Onset cosine ↑ | Onset F1 ±30 ms ↑ | Onset F1 ±50 ms ↑ | FAD∞ ↓ | Audio L1 ↓ | MR-STFT ↓ | Params | KAD ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ours (250 Hz grid, diffusion + RVQ-CE) | 5.972 | 0.8612 | 0.9565 | 0.9579 | 0.02889 | 0.05416 | 0.1161 | 91.85M | 1.338 [1.111, 1.759] |
| TRIA released, bar as prompts | 8.844 | 0.695 | 0.8796 | 0.8971 | 0.08184 | 0.0817 | 0.1854 | 43.05M | 8.497 [7.023, 10.37] |
| TRIA released, MIDI render + reference | 10.8 | 0.632 | 0.7925 | 0.8219 | 0.09507 | 0.08356 | 0.2177 | 43.05M | 9.369 [7.545, 11.56] |
| TRIA fine-tuned, bar as prompts | 7.83 | 0.7439 | 0.8938 | 0.9073 | 0.02922 | 0.08362 | 0.1749 | 43.05M | 1.579 [1.195, 2.212] |
| TRIA fine-tuned, MIDI render + reference | 9.649 | 0.6765 | 0.814 | 0.8409 | 0.03248 | 0.08787 | 0.2034 | 43.05M | 1.345 [0.9951, 2.076] |

## Onsets

Precision: the share of the clip's onsets that match one of the real bar's; recall: the share of the real bar's onsets that the clip reproduces (one-to-one matching within the window). Onsets per clip: the real bars average 8.74.

| System | Precision ±30 ms | Recall ±30 ms | F1 ±30 ms | Precision ±50 ms | Recall ±50 ms | F1 ±50 ms | Onsets per clip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ours (250 Hz grid, diffusion + RVQ-CE) | 0.966 | 0.953 | 0.957 | 0.968 | 0.954 | 0.958 | 8.57 |
| TRIA released, bar as prompts | 0.892 | 0.879 | 0.88 | 0.91 | 0.897 | 0.897 | 8.6 |
| TRIA released, MIDI render + reference | 0.828 | 0.784 | 0.792 | 0.858 | 0.813 | 0.822 | 8.29 |
| TRIA fine-tuned, bar as prompts | 0.911 | 0.888 | 0.894 | 0.925 | 0.901 | 0.907 | 8.48 |
| TRIA fine-tuned, MIDI render + reference | 0.842 | 0.808 | 0.814 | 0.87 | 0.835 | 0.841 | 8.37 |

## Paired comparisons (A − B)

Recording-clustered: 95% CI from a bootstrap over the 71 recordings; p from a recording-level sign-flip test (per-clip metrics) or a recording-level swap of the two systems' clips (KAD), 5000 resamples each; Holm: adjusted over the pairs of each metric.

| A − B | Metric | Difference | 95% CI | p | p (Holm) |
| --- | --- | ---: | ---: | ---: | ---: |
| tria_finetuned_bar − tria_released_bar | mel_mae_db | -1.014 | [-1.259, -0.8229] | 0.0002 | 0.0016 |
| tria_finetuned_bar − tria_released_bar | onset_flux_cosine | 0.04896 | [0.03928, 0.05955] | 0.0002 | 0.0016 |
| tria_finetuned_bar − tria_released_bar | audio_l1 | 0.001915 | [0.0002211, 0.003517] | 0.048 | 0.048 |
| tria_finetuned_bar − tria_released_bar | mrstft_logmag_l1 | -0.01057 | [-0.01391, -0.007166] | 0.0002 | 0.0016 |
| tria_finetuned_bar − tria_released_bar | onset_f1_30ms | 0.01427 | [0.005458, 0.02459] | 0.0028 | 0.0028 |
| tria_finetuned_bar − tria_released_bar | onset_f1_50ms | 0.0102 | [0.002356, 0.01946] | 0.022 | 0.022 |
| tria_finetuned_bar − tria_released_bar | onset_precision_30ms | 0.01904 | [0.009224, 0.02941] | 0.0006 | 0.0016 |
| tria_finetuned_bar − tria_released_bar | onset_recall_30ms | 0.008957 | [-0.001259, 0.02034] | 0.11 | 0.11 |
| tria_finetuned_bar − tria_released_bar | onset_precision_50ms | 0.01507 | [0.005892, 0.02471] | 0.0018 | 0.0036 |
| tria_finetuned_bar − tria_released_bar | onset_recall_50ms | 0.004699 | [-0.004735, 0.01514] | 0.39 | 0.39 |
| tria_finetuned_bar − tria_released_bar | kad | -6.918 | [-8.316, -5.692] | 0.0002 | 0.0016 |
| tria_finetuned_midi − tria_released_midi | mel_mae_db | -1.148 | [-1.329, -0.9828] | 0.0002 | 0.0016 |
| tria_finetuned_midi − tria_released_midi | onset_flux_cosine | 0.04459 | [0.03556, 0.05373] | 0.0002 | 0.0016 |
| tria_finetuned_midi − tria_released_midi | audio_l1 | 0.004311 | [0.001727, 0.006751] | 0.0036 | 0.011 |
| tria_finetuned_midi − tria_released_midi | mrstft_logmag_l1 | -0.01424 | [-0.01801, -0.01063] | 0.0002 | 0.0016 |
| tria_finetuned_midi − tria_released_midi | onset_f1_30ms | 0.02154 | [0.01213, 0.03169] | 0.0002 | 0.0016 |
| tria_finetuned_midi − tria_released_midi | onset_f1_50ms | 0.01897 | [0.01099, 0.02749] | 0.0002 | 0.0016 |
| tria_finetuned_midi − tria_released_midi | onset_precision_30ms | 0.01392 | [0.00327, 0.02431] | 0.015 | 0.015 |
| tria_finetuned_midi − tria_released_midi | onset_recall_30ms | 0.02421 | [0.01439, 0.03564] | 0.0002 | 0.0016 |
| tria_finetuned_midi − tria_released_midi | onset_precision_50ms | 0.01121 | [0.00212, 0.02044] | 0.016 | 0.016 |
| tria_finetuned_midi − tria_released_midi | onset_recall_50ms | 0.02147 | [0.01215, 0.03178] | 0.0002 | 0.0016 |
| tria_finetuned_midi − tria_released_midi | kad | -8.025 | [-9.627, -6.463] | 0.0002 | 0.0016 |
| tria_released_midi − tria_released_bar | mel_mae_db | 1.953 | [1.682, 2.235] | 0.0002 | 0.0016 |
| tria_released_midi − tria_released_bar | onset_flux_cosine | -0.06304 | [-0.08243, -0.04651] | 0.0002 | 0.0016 |
| tria_released_midi − tria_released_bar | audio_l1 | 0.001859 | [0.0005182, 0.003088] | 0.014 | 0.029 |
| tria_released_midi − tria_released_bar | mrstft_logmag_l1 | 0.03223 | [0.0262, 0.03729] | 0.0002 | 0.0016 |
| tria_released_midi − tria_released_bar | onset_f1_30ms | -0.08709 | [-0.1077, -0.07106] | 0.0002 | 0.0016 |
| tria_released_midi − tria_released_bar | onset_f1_50ms | -0.07518 | [-0.09188, -0.06186] | 0.0002 | 0.0016 |
| tria_released_midi − tria_released_bar | onset_precision_30ms | -0.06452 | [-0.0814, -0.05092] | 0.0002 | 0.0016 |
| tria_released_midi − tria_released_bar | onset_recall_30ms | -0.09504 | [-0.1177, -0.07625] | 0.0002 | 0.0016 |
| tria_released_midi − tria_released_bar | onset_precision_50ms | -0.05146 | [-0.06505, -0.04012] | 0.0002 | 0.0016 |
| tria_released_midi − tria_released_bar | onset_recall_50ms | -0.08345 | [-0.103, -0.06727] | 0.0002 | 0.0016 |
| tria_released_midi − tria_released_bar | kad | 0.8719 | [-0.03268, 1.936] | 0.083 | 0.33 |
| tria_finetuned_midi − tria_finetuned_bar | mel_mae_db | 1.819 | [1.575, 2.07] | 0.0002 | 0.0016 |
| tria_finetuned_midi − tria_finetuned_bar | onset_flux_cosine | -0.0674 | [-0.08812, -0.04982] | 0.0002 | 0.0016 |
| tria_finetuned_midi − tria_finetuned_bar | audio_l1 | 0.004255 | [0.002006, 0.006342] | 0.0012 | 0.0048 |
| tria_finetuned_midi − tria_finetuned_bar | mrstft_logmag_l1 | 0.02855 | [0.0232, 0.03363] | 0.0002 | 0.0016 |
| tria_finetuned_midi − tria_finetuned_bar | onset_f1_30ms | -0.07981 | [-0.09993, -0.06298] | 0.0002 | 0.0016 |
| tria_finetuned_midi − tria_finetuned_bar | onset_f1_50ms | -0.06641 | [-0.08237, -0.05283] | 0.0002 | 0.0016 |
| tria_finetuned_midi − tria_finetuned_bar | onset_precision_30ms | -0.06964 | [-0.08898, -0.05356] | 0.0002 | 0.0016 |
| tria_finetuned_midi − tria_finetuned_bar | onset_recall_30ms | -0.07978 | [-0.1025, -0.06049] | 0.0002 | 0.0016 |
| tria_finetuned_midi − tria_finetuned_bar | onset_precision_50ms | -0.05532 | [-0.06998, -0.04218] | 0.0002 | 0.0016 |
| tria_finetuned_midi − tria_finetuned_bar | onset_recall_50ms | -0.06668 | [-0.08632, -0.04915] | 0.0002 | 0.0016 |
| tria_finetuned_midi − tria_finetuned_bar | kad | -0.2347 | [-0.5273, 0.1194] | 0.14 | 0.33 |
| tria_released_bar − grid_250hz | mel_mae_db | 2.873 | [2.388, 3.334] | 0.0002 | 0.0016 |
| tria_released_bar − grid_250hz | onset_flux_cosine | -0.1663 | [-0.2061, -0.1275] | 0.0002 | 0.0016 |
| tria_released_bar − grid_250hz | audio_l1 | 0.02754 | [0.02473, 0.03027] | 0.0002 | 0.0016 |
| tria_released_bar − grid_250hz | mrstft_logmag_l1 | 0.06936 | [0.05949, 0.07963] | 0.0002 | 0.0016 |
| tria_released_bar − grid_250hz | onset_f1_30ms | -0.07696 | [-0.09853, -0.0564] | 0.0002 | 0.0016 |
| tria_released_bar − grid_250hz | onset_f1_50ms | -0.06079 | [-0.07908, -0.04349] | 0.0002 | 0.0016 |
| tria_released_bar − grid_250hz | onset_precision_30ms | -0.07417 | [-0.09373, -0.0564] | 0.0002 | 0.0016 |
| tria_released_bar − grid_250hz | onset_recall_30ms | -0.07356 | [-0.09836, -0.04972] | 0.0002 | 0.0016 |
| tria_released_bar − grid_250hz | onset_precision_50ms | -0.05787 | [-0.07424, -0.04295] | 0.0002 | 0.0016 |
| tria_released_bar − grid_250hz | onset_recall_50ms | -0.05735 | [-0.07994, -0.03644] | 0.0002 | 0.0016 |
| tria_released_bar − grid_250hz | kad | 7.159 | [5.619, 8.947] | 0.0002 | 0.0016 |
| tria_released_midi − grid_250hz | mel_mae_db | 4.825 | [4.375, 5.302] | 0.0002 | 0.0016 |
| tria_released_midi − grid_250hz | onset_flux_cosine | -0.2293 | [-0.2617, -0.1958] | 0.0002 | 0.0016 |
| tria_released_midi − grid_250hz | audio_l1 | 0.0294 | [0.02639, 0.03238] | 0.0002 | 0.0016 |
| tria_released_midi − grid_250hz | mrstft_logmag_l1 | 0.1016 | [0.09041, 0.1139] | 0.0002 | 0.0016 |
| tria_released_midi − grid_250hz | onset_f1_30ms | -0.164 | [-0.1919, -0.1384] | 0.0002 | 0.0016 |
| tria_released_midi − grid_250hz | onset_f1_50ms | -0.136 | [-0.16, -0.114] | 0.0002 | 0.0016 |
| tria_released_midi − grid_250hz | onset_precision_30ms | -0.1387 | [-0.1628, -0.1173] | 0.0002 | 0.0016 |
| tria_released_midi − grid_250hz | onset_recall_30ms | -0.1686 | [-0.2013, -0.1389] | 0.0002 | 0.0016 |
| tria_released_midi − grid_250hz | onset_precision_50ms | -0.1093 | [-0.1282, -0.09146] | 0.0002 | 0.0016 |
| tria_released_midi − grid_250hz | onset_recall_50ms | -0.1408 | [-0.1695, -0.1146] | 0.0002 | 0.0016 |
| tria_released_midi − grid_250hz | kad | 8.031 | [6.114, 10.07] | 0.0002 | 0.0016 |
| tria_finetuned_bar − grid_250hz | mel_mae_db | 1.858 | [1.421, 2.295] | 0.0002 | 0.0016 |
| tria_finetuned_bar − grid_250hz | onset_flux_cosine | -0.1173 | [-0.1496, -0.08363] | 0.0002 | 0.0016 |
| tria_finetuned_bar − grid_250hz | audio_l1 | 0.02945 | [0.02588, 0.03282] | 0.0002 | 0.0016 |
| tria_finetuned_bar − grid_250hz | mrstft_logmag_l1 | 0.0588 | [0.05173, 0.06647] | 0.0002 | 0.0016 |
| tria_finetuned_bar − grid_250hz | onset_f1_30ms | -0.06269 | [-0.08066, -0.04627] | 0.0002 | 0.0016 |
| tria_finetuned_bar − grid_250hz | onset_f1_50ms | -0.05059 | [-0.06451, -0.03664] | 0.0002 | 0.0016 |
| tria_finetuned_bar − grid_250hz | onset_precision_30ms | -0.05514 | [-0.06977, -0.04013] | 0.0002 | 0.0016 |
| tria_finetuned_bar − grid_250hz | onset_recall_30ms | -0.0646 | [-0.08525, -0.04573] | 0.0002 | 0.0016 |
| tria_finetuned_bar − grid_250hz | onset_precision_50ms | -0.04279 | [-0.05438, -0.03124] | 0.0002 | 0.0016 |
| tria_finetuned_bar − grid_250hz | onset_recall_50ms | -0.05265 | [-0.06954, -0.0366] | 0.0002 | 0.0016 |
| tria_finetuned_bar − grid_250hz | kad | 0.2415 | [-0.2954, 0.8287] | 0.09 | 0.33 |
| tria_finetuned_midi − grid_250hz | mel_mae_db | 3.677 | [3.256, 4.104] | 0.0002 | 0.0016 |
| tria_finetuned_midi − grid_250hz | onset_flux_cosine | -0.1847 | [-0.2112, -0.1577] | 0.0002 | 0.0016 |
| tria_finetuned_midi − grid_250hz | audio_l1 | 0.03371 | [0.02904, 0.03834] | 0.0002 | 0.0016 |
| tria_finetuned_midi − grid_250hz | mrstft_logmag_l1 | 0.08735 | [0.07809, 0.09683] | 0.0002 | 0.0016 |
| tria_finetuned_midi − grid_250hz | onset_f1_30ms | -0.1425 | [-0.1691, -0.1182] | 0.0002 | 0.0016 |
| tria_finetuned_midi − grid_250hz | onset_f1_50ms | -0.117 | [-0.1395, -0.09722] | 0.0002 | 0.0016 |
| tria_finetuned_midi − grid_250hz | onset_precision_30ms | -0.1248 | [-0.1493, -0.1031] | 0.0002 | 0.0016 |
| tria_finetuned_midi − grid_250hz | onset_recall_30ms | -0.1444 | [-0.1743, -0.1174] | 0.0002 | 0.0016 |
| tria_finetuned_midi − grid_250hz | onset_precision_50ms | -0.09811 | [-0.1172, -0.07987] | 0.0002 | 0.0016 |
| tria_finetuned_midi − grid_250hz | onset_recall_50ms | -0.1193 | [-0.1449, -0.09648] | 0.0002 | 0.0016 |
| tria_finetuned_midi − grid_250hz | kad | 0.006819 | [-0.5114, 0.6883] | 0.96 | 0.96 |
