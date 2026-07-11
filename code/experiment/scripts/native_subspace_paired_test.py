#!/usr/bin/env python
"""Paired bootstrap CI + sign-flip permutation test: native vs PCA subspace.

Reproduces the PCA-vs-native-subspace ablation statistics reported in the paper
(\\Cref{tab:native_subspace}) on the shared test clips, using the same 2000
percentile-bootstrap / 2000 two-sided sign-flip protocol as the main results.
Metrics: mel MAE (acoustic eval) and MRSTFT log-magnitude L1 + waveform L1
(direct audio eval). Clips are paired by dataset_index.
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

DEF = {
    "native_acoustic": "paper_results/native_subspace_eval/acoustic_eval/per_clip_metrics.csv",
    "pca_acoustic": "paper_results/full_acoustic_eval/acoustic_eval/per_clip_metrics.csv",
    "native_direct": "paper_results/native_subspace_eval_direct/direct_audio_eval/dac_native_25steps/per_clip_metrics.csv",
    "pca_direct": "paper_results/full_acoustic_eval/direct_audio_eval/diffusion_pca_25steps/per_clip_metrics.csv",
    "native_model": "dac_native_25steps",
    "pca_model": "diffusion_pca_25steps",
}


def load(path: str, val: str, model: str | None = None, key: str = "dataset_index") -> dict[int, float]:
    out: dict[int, float] = {}
    for r in csv.DictReader(open(path)):
        if model is not None and r.get("model") != model:
            continue
        try:
            out[int(r[key])] = float(r[val])
        except (KeyError, ValueError):
            pass
    return out


def paired(nat: dict[int, float], pca: dict[int, float], name: str, rng, n_boot: int, n_perm: int) -> None:
    ks = sorted(set(nat) & set(pca))
    a = np.array([nat[k] for k in ks])
    b = np.array([pca[k] for k in ks])
    d = a - b  # native - pca
    n = len(d)
    md = d.mean()
    boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    obs = abs(md)
    cnt = sum(abs(((rng.integers(0, 2, n) * 2 - 1) * d).mean()) >= obs for _ in range(n_perm))
    p = (cnt + 1) / (n_perm + 1)
    print(f"{name:9s} n={n}  native={a.mean():.4f}  pca={b.mean():.4f}  "
          f"diff(native-pca)={md:+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]  sign-flip p={p:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    for k, v in DEF.items():
        ap.add_argument(f"--{k.replace('_', '-')}", default=v)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=2000)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    paired(load(a.native_acoustic, "mel_mae_db", a.native_model),
           load(a.pca_acoustic, "mel_mae_db", a.pca_model), "mel", rng, a.n_boot, a.n_perm)
    paired(load(a.native_direct, "mrstft_logmag_l1"),
           load(a.pca_direct, "mrstft_logmag_l1"), "mrstft", rng, a.n_boot, a.n_perm)
    paired(load(a.native_direct, "audio_l1"),
           load(a.pca_direct, "audio_l1"), "audioL1", rng, a.n_boot, a.n_perm)


if __name__ == "__main__":
    main()
