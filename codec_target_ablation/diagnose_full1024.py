#!/usr/bin/env python3
"""Why we diffuse in the 72 PCA coordinates rather than the 1024-D post-snap latent (paper, Sec. 3.6).

Three CPU measurements; nothing is trained or rewritten.

  variance   Eigenvalues of the per-coordinate standardized 1024-D training targets. z_q spans a
             72-D affine subspace, so standardization puts its total variance of 1024 on 72
             directions (about 14 each) against unit diffusion noise; the PCA arm has 1 per direction.
  head span  Fraction of that 72-D data subspace inside a checkpoint's epsilon output span. The
             952 empty dimensions still carry noise, so a head of width <= 1024 must drop directions.
  grid use   Variation of the conditioning features over time (std over frames / mean magnitude) and
             how far zeroing the grid moves the epsilon prediction, relative to its error. Each
             checkpoint is also measured at its random initialization (seed 1234).

  python codec_target_ablation/diagnose_full1024.py --checkpoint <run>/best_diffusion.pt \
      --checkpoint runs/final/postsnap_pca/best_diffusion.pt
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "experiment"))

from data.diffusion_dataset import build_diffusion_dataloader  # noqa: E402
from model import (  # noqa: E402
    ConditionalDiffusionTransformer,
    DiffusionTransformerConfig,
    GaussianDiffusion1D,
    _prepare_batch_tensors,
    apply_bpm_training_geometry_to_prepared_batch,
    apply_seq_mask,
    normalize_latent,
)

STEPS = (6, 12, 18, 22, 24)


def target_variance(full_cache: Path, every: int) -> dict:
    stats = torch.load(full_cache / "target_stats.pt", map_location="cpu", weights_only=False)
    rows = [json.loads(line) for line in (full_cache / "manifests/train.jsonl").read_text().splitlines() if line.strip()]
    frames = torch.cat([
        torch.load(full_cache / row["out_pt"], map_location="cpu", weights_only=False)["target_sum_td"].float()
        for row in rows[::every]
    ])
    x = (frames - torch.as_tensor(stats["target_mean"]).float()) / torch.as_tensor(stats["target_std"]).float()
    eig = torch.linalg.eigvalsh(torch.cov(x.T)).flip(0)
    return {
        "frames": int(frames.shape[0]),
        "total_variance": float(eig.sum()),
        "top72_mean": float(eig[:72].mean()),
        "top72_max": float(eig[0]),
        "top72_min": float(eig[71]),
        "remaining_max": float(eig[72:].abs().max()),
    }


def load_model(path: Path, *, trained: bool):
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    known = {field.name for field in dataclasses.fields(DiffusionTransformerConfig)}
    cfg = DiffusionTransformerConfig(**{k: v for k, v in payload["config"].items() if k in known})
    torch.manual_seed(1234)
    model = ConditionalDiffusionTransformer(cfg).eval()
    if trained:
        model.load_state_dict(payload["model_state_dict"])
    mean = torch.as_tensor(payload["target_mean"]).float()
    std = torch.as_tensor(payload["target_std"]).float().clamp_min(1.0e-6)
    return model, mean, std, payload


def head_span(model, std: torch.Tensor, components: torch.Tensor) -> dict:
    span = torch.linalg.qr(model.out_proj.weight.detach().float(), mode="reduced")[0]
    data = torch.linalg.qr((components / std).T, mode="reduced")[0]  # data subspace, standardized
    generator = torch.Generator().manual_seed(1234)
    random = torch.linalg.qr(torch.randn(data.shape, generator=generator), mode="reduced")[0]
    k = int(data.shape[1])
    return {
        "head_rank": int(span.shape[1]),
        "data_fraction_in_span": float((span.T @ data).pow(2).sum() / k),
        "random_fraction_in_span": float((span.T @ random).pow(2).sum() / k),
    }


@torch.no_grad()
def grid_use(model, mean, std, cache: Path) -> dict:
    diffusion = GaussianDiffusion1D(num_steps=25)
    batch = next(iter(build_diffusion_dataloader(cache, split="validation", batch_size=4, shuffle=False, num_workers=0)))
    prep = apply_bpm_training_geometry_to_prepared_batch(_prepare_batch_tensors(batch, torch.device("cpu")), num_beats=4)
    mask = prep["target_valid_mask_bt"]
    common = dict(grid_ids=prep["grid_ids"], grid_times_sec=prep["grid_times_sec"],
                  token_times_sec=prep["token_times_sec"], grid_valid_mask_bt=prep["grid_valid_mask"])
    cond, _ = model.encode_conditioning(grid=prep["grid"], target_valid_mask_bt=mask, **common)
    # Example 3 of the first validation batch is the training preview's excerpt.
    features = cond[3, : int(mask[3].sum())]
    out = {"frontend_time_variation": float(features.std(0).mean() / features.abs().mean()), "grid_effect": {}}
    x0 = normalize_latent(prep["target_btd"], mean, std)
    generator = torch.Generator().manual_seed(0)
    for step in STEPS:
        t = torch.full((x0.shape[0],), step, dtype=torch.long)
        noise = apply_seq_mask(torch.randn(x0.shape, generator=generator), mask)
        x_t = apply_seq_mask(diffusion.q_sample(x0, t, noise), mask)
        eps = model(x_t=x_t, t=t, target_valid_mask_bt=mask, grid=prep["grid"], **common)
        eps_zero = model(x_t=x_t, t=t, target_valid_mask_bt=mask, grid=torch.zeros_like(prep["grid"]), **common)
        moved = ((eps - eps_zero) ** 2).sum(-1)[mask].mean()
        error = ((eps - noise) ** 2).sum(-1)[mask].mean()
        out["grid_effect"][str(step)] = {"moved": float(moved), "error": float(error), "ratio": float(moved / error)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full-cache", type=Path, default=ROOT / "caches/cache_4beats_dac44q9_full1024_bpmgeom_duration_v1")
    parser.add_argument("--pca-cache", type=Path, default=ROOT / "caches/cache_4beats_dac44q9_pca72_native_bpmgeom_duration_v1")
    parser.add_argument("--checkpoint", type=Path, action="append", default=[], help="diffusion checkpoint; repeatable")
    parser.add_argument("--every", type=int, default=40, help="use every Nth training excerpt for the variance")
    parser.add_argument("--output", type=Path, default=ROOT / "codec_target_ablation/results/diagnose_full1024.json")
    args = parser.parse_args()
    torch.set_num_threads(8)
    basis = torch.load(args.pca_cache / "pca_basis.pt", map_location="cpu", weights_only=False)
    components = torch.as_tensor(basis["components"]).float()

    report = {"variance": target_variance(args.full_cache, args.every), "checkpoints": {}}
    print(json.dumps(report["variance"]))
    for path in args.checkpoint:
        entry = {}
        for trained in (True, False):
            model, mean, std, payload = load_model(path, trained=trained)
            full = int(model.cfg.x_dim) == int(components.shape[1])
            result = grid_use(model, mean, std, args.full_cache if full else args.pca_cache)
            if trained and full:
                result.update(head_span(model, std, components))
            entry["trained" if trained else "init"] = result
        entry.update(epoch=int(payload.get("epoch", -1)), x_dim=int(model.cfg.x_dim), d_model=int(model.cfg.d_model))
        report["checkpoints"][str(path)] = entry
        print(path, json.dumps(entry))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
