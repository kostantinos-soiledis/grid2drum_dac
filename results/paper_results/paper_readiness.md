# Paper Readiness

This report is generated from completed artifacts only. It separates claims that are currently supported from claims that must be fixed with more evaluation or rerouted to limitations/future work.

## Supported With Current Artifacts

- Direct PCA regression versus PCA diffusion can be discussed with paired automatic metrics.
- RVQ-CE as an auxiliary loss can be discussed as a completed ablation.
- Denoising-step quality/runtime trade-offs can be discussed for completed diffusion runs.
- Symbolic grid rendering can be used as a paired-metric baseline, except for missing FAD.

## Must Fix Before Strong Submission

- Rows with missing metrics: none.
- Reconstruction ceilings need paired acoustic metrics so the paper can quantify the PCA bottleneck rather than only model error.
- Direct PCA regressor needs FAD-infinity if the paper compares distributional quality against diffusion.
- Source-code decode and symbolic nearest-neighbor retrieval need the same metric set as learned models if they remain in the main table.

## Reroute If Not Fixed

- Token-space versus token-embedding-space results should be written as motivation or future work unless discrete-token and full-embedding runs are produced.
- The UI should be presented as an analysis/demo interface, not as user-study evidence.
- The conclusion should remain provisional until `missing_metrics.csv` is empty or all missing rows are removed from the main claim set.

## Artifact Summary

- Git commit: `75b0d9ae6f49ddc24ab7b152b7036376a814462b`
- Aggregated rows: 14
- Rows with missing metrics: 0
- Cache: `<DAC_CACHE_ROOT>`
