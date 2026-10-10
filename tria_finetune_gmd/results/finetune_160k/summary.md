# TRIA on the GMD test split

1733 clips of 71 test recordings, every metric against the real GMD bars. Each system's clips are gain-matched to their real bar's loudness (ITU-R BS.1770) first, which only FAD and KAD see: the per-clip metrics are means over clips of peak-normalized audio. Onset F1 matches the clip's onsets to the real bar's within ±30 or ±50 ms (librosa's onset detector on both, mir_eval), which a few milliseconds of constant output delay do not change, unlike the onset cosine. FAD∞ is CLAP-LAION-Music over 8 repeats; KAD (kadtk, ×100) uses PANNs-WGLM embeddings with one kernel for all systems (bandwidth from the real bars), 95% CI from a bootstrap over recordings.

| System | Mel MAE ↓ | Onset cosine ↑ | Onset F1 ±30 ms ↑ | Onset F1 ±50 ms ↑ | FAD∞ ↓ | Audio L1 ↓ | MR-STFT ↓ | Params | KAD ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ours (250 Hz grid, diffusion + RVQ-CE) | 5.972 | 0.8612 | 0.9565 | 0.9579 | 0.02889 | 0.05416 | 0.1161 | 91.85M | 1.338 [1.111, 1.759] |
| TRIA fine-tuned, bar as prompts | 7.83 | 0.7439 | 0.8938 | 0.9073 | 0.02922 | 0.08362 | 0.1749 | 43.05M | 1.579 [1.195, 2.212] |
| TRIA fine-tuned, MIDI render + reference | 9.649 | 0.6765 | 0.814 | 0.8409 | 0.03248 | 0.08787 | 0.2034 | 43.05M | 1.345 [0.9951, 2.076] |
| TRIA fine-tuned 160k, bar as prompts | 8.105 | 0.7267 | 0.8838 | 0.8994 | 0.02696 | 0.0867 | 0.1753 | 43.05M | 1.345 [1.032, 1.942] |
| TRIA fine-tuned 160k, MIDI render + reference | 9.794 | 0.6687 | 0.8069 | 0.8336 | 0.02992 | 0.08881 | 0.2 | 43.05M | 1.205 [0.8865, 1.918] |

## Onsets

Precision: the share of the clip's onsets that match one of the real bar's; recall: the share of the real bar's onsets that the clip reproduces (one-to-one matching within the window). Onsets per clip: the real bars average 8.74.

| System | Precision ±30 ms | Recall ±30 ms | F1 ±30 ms | Precision ±50 ms | Recall ±50 ms | F1 ±50 ms | Onsets per clip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ours (250 Hz grid, diffusion + RVQ-CE) | 0.966 | 0.953 | 0.957 | 0.968 | 0.954 | 0.958 | 8.57 |
| TRIA fine-tuned, bar as prompts | 0.911 | 0.888 | 0.894 | 0.925 | 0.901 | 0.907 | 8.48 |
| TRIA fine-tuned, MIDI render + reference | 0.842 | 0.808 | 0.814 | 0.87 | 0.835 | 0.841 | 8.37 |
| TRIA fine-tuned 160k, bar as prompts | 0.905 | 0.875 | 0.884 | 0.921 | 0.891 | 0.899 | 8.41 |
| TRIA fine-tuned 160k, MIDI render + reference | 0.836 | 0.8 | 0.807 | 0.864 | 0.827 | 0.834 | 8.37 |

## Paired comparisons (A − B)

Recording-clustered: 95% CI from a bootstrap over the 71 recordings; p from a recording-level sign-flip test (per-clip metrics) or a recording-level swap of the two systems' clips (KAD), 5000 resamples each; Holm: adjusted over the pairs of each metric.

| A − B | Metric | Difference | 95% CI | p | p (Holm) |
| --- | --- | ---: | ---: | ---: | ---: |
| tria_finetuned160k_bar − tria_finetuned_bar | mel_mae_db | 0.2749 | [0.1711, 0.3889] | 0.0002 | 0.0008 |
| tria_finetuned160k_bar − tria_finetuned_bar | onset_flux_cosine | -0.0172 | [-0.02061, -0.01362] | 0.0002 | 0.0008 |
| tria_finetuned160k_bar − tria_finetuned_bar | audio_l1 | 0.003088 | [0.002015, 0.004246] | 0.0002 | 0.0008 |
| tria_finetuned160k_bar − tria_finetuned_bar | mrstft_logmag_l1 | 0.0003862 | [-0.00172, 0.002327] | 0.73 | 0.73 |
| tria_finetuned160k_bar − tria_finetuned_bar | onset_f1_30ms | -0.01002 | [-0.01496, -0.005394] | 0.0004 | 0.0008 |
| tria_finetuned160k_bar − tria_finetuned_bar | onset_f1_50ms | -0.007896 | [-0.01223, -0.003923] | 0.0006 | 0.0012 |
| tria_finetuned160k_bar − tria_finetuned_bar | onset_precision_30ms | -0.006191 | [-0.01207, -0.0004842] | 0.042 | 0.085 |
| tria_finetuned160k_bar − tria_finetuned_bar | onset_recall_30ms | -0.01256 | [-0.01779, -0.007872] | 0.0002 | 0.0008 |
| tria_finetuned160k_bar − tria_finetuned_bar | onset_precision_50ms | -0.003949 | [-0.009022, 0.0009033] | 0.13 | 0.21 |
| tria_finetuned160k_bar − tria_finetuned_bar | onset_recall_50ms | -0.01049 | [-0.01546, -0.005947] | 0.0002 | 0.0008 |
| tria_finetuned160k_bar − tria_finetuned_bar | kad | -0.2343 | [-0.6132, 0.1031] | 0.087 | 0.35 |
| tria_finetuned160k_midi − tria_finetuned_midi | mel_mae_db | 0.1454 | [0.03092, 0.2521] | 0.029 | 0.029 |
| tria_finetuned160k_midi − tria_finetuned_midi | onset_flux_cosine | -0.00787 | [-0.01131, -0.00478] | 0.0002 | 0.0008 |
| tria_finetuned160k_midi − tria_finetuned_midi | audio_l1 | 0.0009355 | [-0.0009464, 0.002695] | 0.32 | 0.32 |
| tria_finetuned160k_midi − tria_finetuned_midi | mrstft_logmag_l1 | -0.003465 | [-0.006707, -0.0003178] | 0.025 | 0.05 |
| tria_finetuned160k_midi − tria_finetuned_midi | onset_f1_30ms | -0.007087 | [-0.01286, -0.002479] | 0.0058 | 0.0058 |
| tria_finetuned160k_midi − tria_finetuned_midi | onset_f1_50ms | -0.007325 | [-0.01288, -0.002626] | 0.0028 | 0.0028 |
| tria_finetuned160k_midi − tria_finetuned_midi | onset_precision_30ms | -0.005379 | [-0.01246, 0.001102] | 0.12 | 0.12 |
| tria_finetuned160k_midi − tria_finetuned_midi | onset_recall_30ms | -0.008059 | [-0.01489, -0.002361] | 0.0066 | 0.0066 |
| tria_finetuned160k_midi − tria_finetuned_midi | onset_precision_50ms | -0.006029 | [-0.01359, 0.001131] | 0.11 | 0.21 |
| tria_finetuned160k_midi − tria_finetuned_midi | onset_recall_50ms | -0.008003 | [-0.01444, -0.002953] | 0.002 | 0.002 |
| tria_finetuned160k_midi − tria_finetuned_midi | kad | -0.1396 | [-0.5472, 0.2061] | 0.27 | 0.82 |
| tria_finetuned160k_bar − grid_250hz | mel_mae_db | 2.133 | [1.694, 2.585] | 0.0002 | 0.0008 |
| tria_finetuned160k_bar − grid_250hz | onset_flux_cosine | -0.1345 | [-0.1668, -0.1014] | 0.0002 | 0.0008 |
| tria_finetuned160k_bar − grid_250hz | audio_l1 | 0.03254 | [0.02933, 0.0358] | 0.0002 | 0.0008 |
| tria_finetuned160k_bar − grid_250hz | mrstft_logmag_l1 | 0.05918 | [0.05184, 0.06692] | 0.0002 | 0.0008 |
| tria_finetuned160k_bar − grid_250hz | onset_f1_30ms | -0.07271 | [-0.09238, -0.05508] | 0.0002 | 0.0008 |
| tria_finetuned160k_bar − grid_250hz | onset_f1_50ms | -0.05849 | [-0.07476, -0.04401] | 0.0002 | 0.0008 |
| tria_finetuned160k_bar − grid_250hz | onset_precision_30ms | -0.06133 | [-0.07918, -0.04424] | 0.0002 | 0.0008 |
| tria_finetuned160k_bar − grid_250hz | onset_recall_30ms | -0.07716 | [-0.09835, -0.05796] | 0.0002 | 0.0008 |
| tria_finetuned160k_bar − grid_250hz | onset_precision_50ms | -0.04674 | [-0.06015, -0.03325] | 0.0002 | 0.0008 |
| tria_finetuned160k_bar − grid_250hz | onset_recall_50ms | -0.06313 | [-0.08148, -0.04627] | 0.0002 | 0.0008 |
| tria_finetuned160k_bar − grid_250hz | kad | 0.007224 | [-0.321, 0.3784] | 0.96 | 0.96 |
| tria_finetuned160k_midi − grid_250hz | mel_mae_db | 3.822 | [3.394, 4.246] | 0.0002 | 0.0008 |
| tria_finetuned160k_midi − grid_250hz | onset_flux_cosine | -0.1926 | [-0.2204, -0.164] | 0.0002 | 0.0008 |
| tria_finetuned160k_midi − grid_250hz | audio_l1 | 0.03464 | [0.03101, 0.03815] | 0.0002 | 0.0008 |
| tria_finetuned160k_midi − grid_250hz | mrstft_logmag_l1 | 0.08388 | [0.07518, 0.09242] | 0.0002 | 0.0008 |
| tria_finetuned160k_midi − grid_250hz | onset_f1_30ms | -0.1496 | [-0.1795, -0.1236] | 0.0002 | 0.0008 |
| tria_finetuned160k_midi − grid_250hz | onset_f1_50ms | -0.1243 | [-0.1485, -0.1038] | 0.0002 | 0.0008 |
| tria_finetuned160k_midi − grid_250hz | onset_precision_30ms | -0.1302 | [-0.1582, -0.1061] | 0.0002 | 0.0008 |
| tria_finetuned160k_midi − grid_250hz | onset_recall_30ms | -0.1524 | [-0.1846, -0.1248] | 0.0002 | 0.0008 |
| tria_finetuned160k_midi − grid_250hz | onset_precision_50ms | -0.1041 | [-0.1263, -0.08366] | 0.0002 | 0.0008 |
| tria_finetuned160k_midi − grid_250hz | onset_recall_50ms | -0.1273 | [-0.1546, -0.1036] | 0.0002 | 0.0008 |
| tria_finetuned160k_midi − grid_250hz | kad | -0.1327 | [-0.527, 0.4147] | 0.31 | 0.82 |
