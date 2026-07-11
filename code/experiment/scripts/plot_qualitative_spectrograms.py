#!/usr/bin/env python
"""Render a qualitative mel-spectrogram comparison panel for the paper.

For one held-out test clip, plot log-mel spectrograms of the target (DAC
reconstruction) alongside the symbolic renderer, the direct PCA regressor, and
the RVQ-CE diffusion model. The clip is chosen automatically as the most
onset-dense among a set of candidate indices so the qualitative differences
(smeared vs. crisp transients, high-band energy) are visible.

Outputs figures/qualitative/spectrogram_comparison.{pdf,png}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio

REPO = Path(__file__).resolve().parent.parent

# (column title, prediction directory) in display order.
SYSTEMS = [
    ("Target (DAC)", "runs_baselines/dac_test_v1/target_dac_recon"),
    ("Symbolic render", "runs_baselines/dac_test_v1/grid_render"),
    ("Direct regressor", "runs_direct/direct_pca_d1024_l6_seed1234/test_set_predictions"),
    ("Diffusion + RVQ-CE (25)", "eval/frontend_ablation/predictions/dac_ce/dac_25steps/none"),
]

SR = 44100
N_FFT = 1024
HOP = 256
N_MELS = 128


def load_manifest(pred_dir: Path) -> dict[int, Path]:
    idx_to_wav: dict[int, Path] = {}
    for line in (pred_dir / "manifest.jsonl").read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        idx_to_wav[int(row["dataset_index"])] = pred_dir / row["wav"]
    return idx_to_wav


def load_mono(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav.mean(0)  # mono


def onset_density(wav: torch.Tensor, mel_db: np.ndarray) -> float:
    # crude spectral-flux peak count as an "interestingness" proxy
    flux = np.clip(np.diff(mel_db, axis=1), 0, None).sum(0)
    if flux.size == 0:
        return 0.0
    thr = flux.mean() + flux.std()
    return float((flux > thr).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=-1, help="dataset_index; -1 = auto-pick onset-dense clip")
    ap.add_argument("--out-dir", type=str, default="figures/qualitative")
    args = ap.parse_args()

    manifests = {title: load_manifest(REPO / d) for title, d in SYSTEMS}
    target_man = manifests["Target (DAC)"]

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=2.0
    )
    to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)

    def mel_db(wav: torch.Tensor) -> np.ndarray:
        return to_db(mel(wav.unsqueeze(0))).squeeze(0).numpy()

    if args.index >= 0:
        chosen = args.index
    else:
        candidates = sorted(target_man)[::10]
        best, best_score = candidates[0], -1.0
        for idx in candidates:
            m = mel_db(load_mono(target_man[idx]))
            s = onset_density(None, m)
            if s > best_score:
                best, best_score = idx, s
        chosen = best
    print(f"chosen dataset_index = {chosen}")

    # render
    specs = []
    for title, _ in SYSTEMS:
        wav = load_mono(manifests[title][chosen])
        specs.append((title, mel_db(wav)))
    vmax = max(s.max() for _, s in specs)
    vmin = vmax - 80.0
    minT = min(s.shape[1] for _, s in specs)

    fig, axes = plt.subplots(1, len(specs), figsize=(11.5, 2.6), sharey=True)
    for ax, (title, s) in zip(axes, specs):
        s = s[:, :minT]
        extent = [0, minT * HOP / SR, 0, SR / 2 / 1000.0]
        im = ax.imshow(s, origin="lower", aspect="auto", extent=extent,
                       vmin=vmin, vmax=vmax, cmap="magma")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("Freq (kHz)", fontsize=8)
    cbar = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01)
    cbar.set_label("dB", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"spectrogram_comparison.{ext}", bbox_inches="tight", dpi=200)
    print(f"wrote {out_dir}/spectrogram_comparison.pdf")


if __name__ == "__main__":
    main()
