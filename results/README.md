# Final results

`results/final/` holds the outcome of
[`run_final_results.sh`](../run_final_results.sh). Read
[`final/summary.md`](final/summary.md) first. It has one table per question:

1. pre-snap native latent vs post-snap PCA;
2. diffusion vs direct regression;
3. 6-layer vs 8-layer regression capacity control;
4. a compact 90/120/250/500 Hz grid-rate check;
5. post-snap PCA diffusion with vs without the RVQ cross-entropy term.

- `final/comparisons/`: the same tables as CSV;
- `final/evaluation/`: the joint evaluation that scores every system together
  (per-clip metrics, FAD, summaries);
- `final/direct_per_clip_metrics.csv`: per-clip waveform L1 and MR-STFT for all
  systems;
- `final/stats/`: source-recording-clustered paired bootstrap intervals and
  sign-flip tests for each comparison.

Regenerable audio and target/FAD caches are excluded from version control.
