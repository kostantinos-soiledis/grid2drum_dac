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

## Metadata-only run records

Every run the paper cites now has its provenance in this bundle even when its
weights do not ship. These directories carry `run_config.json`, `config.json`
where available, `history.csv`, and the export `test_set_predictions/summary.json`
(parameter count, selected epoch, best validation loss, RTF, device) — but no
checkpoints and no prediction WAVs:

- `runs_dac/dac_6steps`, `dac_12steps`, `dac_50steps` (plain diffusion sweep)
- `runs_dac_ce/dac_6steps` (RVQ-CE sweep)
- `runs_dac_native/dac_25steps` (native-basis ablation; metrics in
  `results/paper_results/native_subspace_eval*/`)
- `runs_direct/direct_pca_d1024_l8_seed1234` (101.69M capacity control)
- `runs_baselines/dac_test_v1/` (reconstruction ceilings, procedural render,
  source-code decode, symbolic nearest-neighbour retrieval)

The capacity control is the one run whose numbers are not yet in
`results/paper_results/`. Its 1,733 test clips are exported in the parent
workspace but were never put through `run_diffusion_acoustic_eval.py`, so it has
no row in `core_complete_run_metrics.csv`. Nothing needs retraining; the
evaluation must run in the parent workspace because it needs the GMD-derived
target cache, which is not redistributable.

Absolute paths in every copied record are rewritten to the `<DRUMTOGRID_ROOT>`
placeholder, matching the rest of the bundle.

Archived configs predate the removal of conditioning dropout and classifier-free
guidance from the code. They may still contain `cond_dropout_prob` and
`guidance_scale` keys; both were always at their no-op values (`0.0` and `1.0`),
so the records describe exactly the behaviour the current code produces. They
are kept verbatim as provenance rather than rewritten.
