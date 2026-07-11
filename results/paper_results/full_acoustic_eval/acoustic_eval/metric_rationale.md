# Metric Rationale

This evaluation compares decoded EnCodec 32 kHz predictions against cached 32 kHz target clips.

Design choices:

- `fadtk` FAD-inf is the distribution-level metric. This follows recent FAD work that recommends extrapolation toward infinite sample size to reduce bias.
- Recent music-generation evaluation work suggests no single automatic metric is sufficient, so the script combines distributional, spectral, temporal, and dynamic views.
- The paired metrics are an inference from the literature: transient timing is represented with broadband and band-limited spectral flux, timbral balance with low/mid/high spectral ratios plus centroid, and punch/level with RMS and crest-factor deltas.
- Clip-level inference uses paired bootstrap confidence intervals and paired sign-flip permutation tests over shared dataset indices.
- FAD∞ uncertainty here reflects deterministic multi-seed Monte Carlo variability of the extrapolation, not a dataset-level significance test.

References:

- Kilgour et al., 2019. Fréchet Audio Distance: A Reference-Free Metric for Evaluating Music Enhancement Algorithms. https://www.isca-archive.org/interspeech_2019/kilgour19_interspeech.html
- Gui et al., 2024. Adapting Fréchet Audio Distance for Generative Music Evaluation. https://www.microsoft.com/en-us/research/publication/adapting-frechet-audio-distance-for-generative-music-evaluation/
- Grötschla et al., 2025. Benchmarking Music Generation Models and Metrics via Human Preference Studies. https://openreview.net/forum?id=105yqGIpVW
- Saitis and Wallmark, 2024. Timbral brightness perception investigated through multimodal interference. https://pubmed.ncbi.nlm.nih.gov/39090510/
- Auditory and vibrotactile interactions in perception of timbre acoustic features, 2025. https://pubmed.ncbi.nlm.nih.gov/41168236/
