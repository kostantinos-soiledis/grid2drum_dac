# Runs

This is a compact run bundle. Each main paper checkpoint directory keeps:

- `best_diffusion.pt` or `best_direct.pt`
- `config.json` where available
- `run_config.json`
- `history.csv`

The plain diffusion checkpoints that were only available with optimizer state
were stripped into inference-only files. The noisy originals remain in the
parent experiment workspace.

To keep the bundle lean, only the most useful checkpoints ship: two diffusion
models plus one baseline. The full step-count sweep (6/12/25/50 steps, plain and
RVQ-CE) and all baselines are still reported under
`results/paper_results/` (`run_metrics.csv`, `full_acoustic_eval/`); only these
weights are included.

Included checkpoint families:

- `runs_dac/dac_25steps` (plain PCA diffusion, 25 steps)
- `runs_dac_ce/dac_25steps` (RVQ-CE PCA diffusion, 25 steps)
- `runs_direct/direct_pca_d1024_l6_seed1234` (direct PCA regressor baseline)
- `sketch_expander_dac44_native_v5`
- `mini_cache` and `third_party/dac_44khz` for local demo decoding

`frontend_ablation_metadata/` keeps only configs and histories for ablation
runs; the full ablation checkpoints and prediction caches were intentionally
left out.
