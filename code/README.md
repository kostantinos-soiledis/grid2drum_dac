# Experiment code

The active experiment surface is intentionally small. From the repository root,
run:

```bash
./run_final_results.sh --python /path/to/your/torch-python --device cuda:0
```

The runner is serial, resumable, and owns the canonical run/result names. Use
`--dry-run` to inspect every command and `--help` for cache overrides.

Active entry points:

- `experiment/train_cli.py`: diffusion training;
- `experiment/standalone_direct_pca_regressor.py`: direct regression;
- `experiment/scripts/build_source_cache.py` and `build_diffusion_cache.py`:
  base cache construction;
- `experiment/scripts/build_presnap_cache.py`: native pre-snap DAC targets,
  with no PCA transform;
- `experiment/scripts/build_grid_rate_overlay.py`: true higher-rate frontend
  grids re-rendered from MIDI while reusing the canonical audio targets;
- `experiment/scripts/export_*_predictions.py`: deterministic exports;
- `experiment/scripts/run_diffusion_acoustic_eval.py`: shared acoustic scoring;
- `experiment/scripts/clustered_paired_stats.py`: source-clustered inference;
- `experiment/scripts/summarize_final_results.py`: readable result tables;
- `experiment/scripts/relativize_artifact_paths.py`: rewrites absolute
  workstation paths in run/result artifacts as repo-relative paths.

`demo/` is the local listener app (grid editor, generation, and comparison
against the regression baseline); see [`demo/README.md`](demo/README.md).

Earlier frontend, native-subspace, overnight, sketch-expander, and paper
aggregation scripts were retired with the results reset and remain in the git
history.
