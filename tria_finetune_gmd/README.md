# TRIA on GMD: released and fine-tuned baseline

[TRIA](https://github.com/interactiveaudiolab/tria) ("The Rhythm In Anything", O'Reilly et al., ISMIR 2025) generates
drums from two audio prompts: a **rhythm** prompt, which it reduces to a two-band transient envelope (onset times,
loudness and band), and a **timbre** prompt, of which it keeps at most 2 s to learn the kit. This folder compares it with
our model on the GMD test split: the released checkpoint, the same checkpoint fine-tuned on GMD's training recordings,
and two ways of prompting each.

## Setup

```bash
git clone https://github.com/interactiveaudiolab/tria.git tria_finetune_gmd/upstream
cd tria_finetune_gmd/upstream && git checkout 332a82c3cce9f49d8004a201100325491c1ca8db && git lfs pull && cd ../..
python -m venv tria_finetune_gmd/.venv          # then install torch/torchaudio for your CUDA (verified: 2.2.1, Python 3.12)
tria_finetune_gmd/.venv/bin/pip install -r tria_finetune_gmd/requirements.txt
export GMD_ROOT=<groove-v1.0.0/groove>            # GMD: info.csv, audio and MIDI
export SOUNDFONT=<FluidR3_GM.sf2>                 # General MIDI soundfont; FluidSynth on PATH
```

KAD needs [kadtk](https://github.com/YoonjinXD/kadtk) in a second venv over the repo's evaluation environment
(`requirements_kad.txt`). The test bars come from the repo's GMD cache
(`caches/cache_4beats_dac44q9_pca72_native_bpmgeom_duration_v1`) and our model's clips from its final test-set export
(`runs/final/exports/grid_250hz`, `run_final_results.sh`).

## Fine-tuning on GMD (`prepare.py`, `train_early_stop.py`, `train.sh`)

TRIA's own training script fine-tunes when it resumes from a checkpoint without its optimizer state, so `prepare.py`
copies only the released `small_musdb_moises_fsl_2b` weights (GMD is in neither released checkpoint's data) and writes
the config from the checkpoint's own (`upstream/conf/small_musdb_moises_fsl_2b.yml`). Everything not listed is that
config: model, rhythm features, masking, augmentations, sampler, AMP.

| Setting | Value | Why |
| --- | --- | --- |
| data | GMD train (846 recordings, 8.45 h); validation 120 recordings; test never used | our models' splits |
| train manifests | beats (7.97 h) and fills (0.40 h), weighted by hours | TRIA draws recordings uniformly, and 64% of GMD's training recordings, mostly fills, are shorter than its 6 s excerpt |
| loudness cutoff | −40 LUFS (released: −24) | GMD sits near −28 LUFS, so −24 would always fail and give random excerpts |
| batch | 8 × 6 s | 12 runs out of memory on a 10 GB GPU (band-pass augmentation), 24 at once |
| AdamW lr | 1e-5 (released config: 1e-4) | a fine-tune |
| length | at most 80,000 iterations (640,000 excerpts) | |
| validation | every 2,000 iterations (TRIA's cadence) | |
| early stopping | after 10 checks without a new best validation token cross-entropy (TRIA's own criterion for `best/`) | `train_early_stop.py`: TRIA's trainer unchanged plus that one check |

The run went all 80,000 iterations: every one of the 41 checks was a new best, though the last 20,000 iterations gained
little. Validation token cross-entropy fell from 0.1632 (released weights) to 0.1482 (`runs/gmd_long/done.json`).
The fine-tuned weights are `runs/gmd_long/best/model.pt` (172 MB, not in git).

```bash
bash tria_finetune_gmd/train.sh cuda:0   # prepare, then train into runs/gmd_long (resumes from runs/gmd_long/latest)
```

## Rendering the test split (`export_audio.py`, `midi_prompt.py`, `render.sh`)

Every model renders the same 1,733 4-beat bars of the 71 GMD test recordings, with TRIA's sampler settings from the
checkpoint's training config (top-p 0.85, no restriction to the timbre prompt's codes, guidance 2, 8 iterations per
codebook, mono), seed 1234 + clip index. TRIA normalizes its output to −20 LUFS; the evaluation matches every system's
clips to the real bars' loudness first.

| Prompts | Rhythm prompt | Timbre prompt | TRIA hears the target |
| --- | --- | --- | --- |
| `bar` | the bar itself | the bar itself | yes |
| `midi` | the bar's notes rendered with FluidSynth and a General MIDI soundfont | the 2 s of the same recording before the bar (after it at a recording's start), never overlapping it | no |

The `midi` notes are those our model is conditioned on: the bar's onsets in its 250 Hz grid (time, instrument,
articulation, velocity), which are the bar's GMD MIDI notes, plus the notes that start up to 30 ms before the bar. The
bar start comes from a beat tracker on the audio and often falls just after the downbeat (in 22% of test bars a note
starts within 20 ms before it); the grid then holds that note only as instrument state carried into the bar, our model
plays it and the target has it, so the prompt plays it at the bar start. Each grid instrument plays one General MIDI
drum (rims and edges are the same drum). The render is read 5 ms late: its onsets lag the real bars' by that much
(FluidSynth starts its drum samples 1–3 ms after the note-on, and their attacks are softer than GMD's kit), measured on
300 validation bars as the peak of the summed onset-envelope cross-correlation (5.3 ms). On test bars the compensated
render's median onset lag is 0 ms, as our model's; uncompensated, TRIA's `midi` clips came out ~10 ms late against
~4 ms in `bar` mode (TRIA's own delay).

```bash
bash tria_finetune_gmd/render.sh cuda:0   # predictions/{released,finetuned}_{bar,midi}/ (wavs, manifest.jsonl, summary.json)
```

## Evaluation (`evaluate.sh`, `evaluate_metrics.py`, `kad_eval.py`, `summarize.py`)

The ablations' metrics (`run_final_results.sh`) on the five systems together, our model included, with two changes
that a comparison across codecs needs:

- **Reference: the real bars.** The repo's evaluator scores against the decoded target, the bar's cached 9-codebook
  DAC latent through DAC's decoder: the codec ceiling, the right reference among our ablations, which all generate
  those latents (`results/final/summary.md`). TRIA generates with the same codec (DAC 44.1 kHz, 9 codebooks), yet the
  decoded target is not neutral between the two: it is 3 dB of log-mel MAE away from the real bar, and on 60 test bars
  our model's clips scored 0.6 dB better against it than against the real bar, TRIA's 0.6 dB worse.
  `evaluate_metrics.py` keeps the evaluator's metric code and swaps the reference for the real bar, the GMD recording
  from the bar's start for its duration (the audio the `bar` prompts are cut from). Our model's numbers here therefore
  differ from the ablation tables.
- **Loudness.** Each system's clips are first gain-matched to their real bar's integrated loudness (ITU-R BS.1770;
  the real bars sit near −31 LUFS, our model's clips near −24, TRIA's at −20), and every metric reads these clips. The
  per-clip metrics are computed on peak-normalized audio and do not change; FAD and KAD then compare what the drums
  sound like rather than output levels.

Metrics:

- **Per-clip** (`code/experiment/evaluate_ablations_4beat_acoustic.py`): log-mel MAE and onset-flux cosine; waveform
  L1 and MR-STFT log-magnitude L1 (`code/experiment/scripts/evaluate_diffusion_predictions.py`'s definitions).
- **Onset F1** at ±30 and ±50 ms: librosa's onset detector on the clip and on the real bar (same settings), matched
  with `mir_eval`, as rhythm adherence is scored in TRIA's, DARC's and Break-the-Beat's papers. TRIA plays about 5 ms
  late even with the bar itself as its rhythm prompt (its own inference, unchanged), which the onset-flux cosine, with
  its 5.8 ms frames, penalizes heavily (advancing TRIA's clips by 6 ms raises it from 0.68 to 0.79); onset F1 does not
  move with it.
- **FAD∞** with CLAP-LAION-Music (8 repeats).
- **KAD**, one number per system: kadtk (the implementation TRIA's paper uses) with PANNs-WGLM embeddings
  (Wavegram-LogMel CNN14, which the KAD paper finds most correlated with human judgments; TRIA's paper calls it "PANN"),
  against the 1,733 real bars. All systems share one Gaussian kernel, its bandwidth the median pairwise distance of the
  real bars' embeddings (the KAD paper's definition; kadtk's code would take it from each evaluated set); unbiased
  MMD², ×100 (kadtk's scale; the KAD paper's α is 1000). Not comparable in value with TRIA's Table 3, whose reference is random MoisesDB drums and whose rhythm
  prompts are beatboxing and tapping.
- **Statistics clustered by recording** (71): paired per-clip differences with bootstrap 95% CIs and sign-flip
  p-values (`code/experiment/scripts/clustered_paired_stats.py`). For KAD, a bootstrap over recordings gives each
  system's 95% CI and the pairs' CIs, and a permutation test that swaps the two systems' clips of whole recordings gives
  the pairs' p-values. 5,000 resamples each; p-values also Holm-adjusted over the pairs of each metric.
- **Pairs:** fine-tuned − released (per prompt setting), `midi` − `bar` (per checkpoint), and every TRIA system −
  our model.

```bash
GMD_ROOT=<groove/> PYTHON_BIN=<repo evaluation python> bash tria_finetune_gmd/evaluate.sh cuda:0
```

The evaluation environment also needs `pyloudnorm` (0.1.1).

Results: `results/summary.md`, with `results/overall.csv`, `results/pairs.csv` and `results/stats/`.

## Files

```
prepare.py            fine-tune setup: manifests, config, start weights
train_early_stop.py   TRIA's train.py with early stopping on its validation cross-entropy
train.sh              prepare and train (runs/gmd_long; done.json has the validation history)
export_audio.py       render the test split with a checkpoint and a prompt setting
midi_prompt.py        the midi rhythm prompt: the bar's notes from our model's grid, rendered with a GM soundfont
render.sh             the four renders: {released, fine-tuned} x {bar, midi}
evaluate.sh           per-clip metrics, FAD, KAD and clustered statistics
evaluate_metrics.py   real-bar reference, loudness matching, the evaluator's per-clip metrics and FAD
kad_eval.py           KAD (kadtk) with recording-clustered bootstrap and permutation tests
summarize.py          tables: results/summary.md, overall.csv, pairs.csv
requirements*.txt     TRIA's venv; the KAD venv
```
