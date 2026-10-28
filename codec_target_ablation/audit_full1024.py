#!/usr/bin/env python3
"""Audit a 1024-D post-snap diffusion run without training or changing its artifacts.

Checks all split identities, sampled cache tensors, normalization, exact PCA
reconstruction, and the checkpoint's epsilon output rank. Optional forward
and decoder checks run on CPU. No checkpoint or cache is rewritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "experiment"))

from data.audio_codec_utils import load_audio_codec_model  # noqa: E402
from data.diffusion_dataset import build_diffusion_dataloader  # noqa: E402
from model import (  # noqa: E402
    ConditionalDiffusionTransformer,
    DiffusionTransformerConfig,
    GaussianDiffusion1D,
    decode_latent_to_audio,
    denormalize_latent,
    normalize_latent,
)


def project_postsnap_latents(latent: torch.Tensor, mean: torch.Tensor, components: torch.Tensor) -> torch.Tensor:
    """Project unstandardized codec latents onto the 72-D post-snap subspace, keeping zero padding."""
    mu, u = mean.to(latent), components.to(latent)
    projected = mu + ((latent - mu) @ u.T) @ u
    return torch.where(latent.ne(0).any(dim=-1, keepdim=True), projected, latent)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="a 1024-D diffusion checkpoint")
    parser.add_argument("--full-cache", type=Path, default=ROOT / "caches/cache_4beats_dac44q9_full1024_bpmgeom_duration_v1")
    parser.add_argument("--pca-cache", type=Path, default=ROOT / "caches/cache_4beats_dac44q9_pca72_native_bpmgeom_duration_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "codec_target_ablation/results/audit_full1024.json")
    parser.add_argument("--forward", action="store_true")
    parser.add_argument("--decode", action="store_true")
    args = parser.parse_args()
    for name in ("checkpoint", "full_cache", "pca_cache", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    torch.set_num_threads(4)
    torch.manual_seed(1234)

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    basis = torch.load(args.pca_cache / "pca_basis.pt", map_location="cpu", weights_only=False)
    mean = torch.as_tensor(basis["mean"], dtype=torch.float32)
    u = torch.as_tensor(basis["components"], dtype=torch.float32)
    target_mean, target_std = payload["target_mean"].float(), payload["target_std"].float()
    stats = torch.load(args.full_cache / "target_stats.pt", map_location="cpu", weights_only=False)
    report = {
        "checkpoint": str(args.checkpoint.relative_to(ROOT)) if args.checkpoint.is_relative_to(ROOT) else str(args.checkpoint),
        "seed": 1234,
        "checkpoint_epoch": payload["epoch"],
        "recorded_validation_noise_mse": payload["val_diffusion_loss"],
        "normalization": {
            "checkpoint_cache_mean_max_difference": (target_mean - stats["target_mean"]).abs().max().item(),
            "checkpoint_cache_std_max_difference": (target_std - stats["target_std"]).abs().max().item(),
            "minimum_std": target_std.min().item(),
        },
        "pca_orthonormality_max_error": (u @ u.T - torch.eye(u.shape[0])).abs().max().item(),
        "splits": {},
        "sampled_cache_checks": [],
    }
    assert report["normalization"]["checkpoint_cache_mean_max_difference"] == 0
    assert report["normalization"]["checkpoint_cache_std_max_difference"] == 0
    assert target_std.min() > 0
    assert report["pca_orthonormality_max_error"] < 1e-5
    tensor_keys = ("target_sum_td", "source_codes_ct", "grid_ft", "grid_ids_ft", "grid_times_sec_t", "token_times_sec_t")
    identity_keys = ("source_id", "beat_index", "source_manifest_index", "duration_sec", "target_num_frames", "grid_num_frames")
    decoder_target = None
    for split in ("train", "validation", "test"):
        full_rows = read_rows(args.full_cache / "manifests" / f"{split}.jsonl")
        pca_rows = read_rows(args.pca_cache / "manifests" / f"{split}.jsonl")
        assert len(full_rows) == len(pca_rows)
        for a, b in zip(full_rows, pca_rows):
            assert all(a[k] == b[k] for k in identity_keys), (split, a, b)
        report["splits"][split] = {"num_examples": len(full_rows), "all_identities_and_frame_counts_match": True}
        for index in (0, len(full_rows) // 2, len(full_rows) - 1):
            full = torch.load(args.full_cache / full_rows[index]["out_pt"], map_location="cpu", weights_only=False)
            pca = torch.load(args.pca_cache / pca_rows[index]["out_pt"], map_location="cpu", weights_only=False)
            for key in tensor_keys:
                assert torch.equal(full[key], pca[key]), (split, index, key)
            z = full["target_sum_td"].float()
            reconstructed = mean + pca["target_pc_tk"].float() @ u
            projected = project_postsnap_latents(z, mean, u)
            normalized = normalize_latent(z[None], target_mean, target_std)
            roundtrip = denormalize_latent(normalized, target_mean, target_std)[0]
            relative_error = (z - reconstructed).norm() / z.norm()
            projection_error = (z - projected).norm() / z.norm()
            # Cached FP32 quantizer outputs and an analytical affine basis
            # agree numerically rather than bit for bit (about 3e-5 here).
            assert relative_error < 1e-3
            assert projection_error < 1e-3
            assert (z - roundtrip).abs().max() < 1e-5
            report["sampled_cache_checks"].append({
                "split": split, "index": index, "raw_latents_codes_grids_and_times_identical": True,
                "pca_reconstruction_relative_l2": relative_error.item(),
                "projection_relative_l2": projection_error.item(),
                "normalization_roundtrip_max_error": (z - roundtrip).abs().max().item(),
            })
            if split == "test" and index == 0:
                decoder_target = z[:32][None], pca["target_pc_tk"][:32][None].float()
        print(f"Checked {split}: {len(full_rows)} aligned excerpts, three tensor samples", flush=True)

    state = payload["model_state_dict"]
    w = state["out_proj.weight"].float()
    q, _ = torch.linalg.qr(w, mode="reduced")
    eps = torch.randn(4096, w.shape[0])
    residual = eps - (eps @ q) @ q.T
    diffusion = GaussianDiffusion1D(num_steps=int(payload["num_steps"]))
    alpha_bar = diffusion.alpha_bars[-1].item()
    floor = max(0.0, 1.0 - w.shape[1] / w.shape[0])
    report["noise_output_bottleneck"] = {
        "input_projection_shape": list(state["x_proj.weight"].shape),
        "output_projection_shape": list(w.shape),
        "minimum_unrepresentable_noise_dimensions": max(0, w.shape[0] - w.shape[1]),
        "expected_noise_mse_lower_bound": floor,
        "monte_carlo_noise_mse_outside_output_span": residual.square().mean().item(),
        "terminal_alpha_bar": alpha_bar,
        "terminal_x0_mse_lower_bound_before_clipping": floor * (1 - alpha_bar) / alpha_bar,
    }
    print(json.dumps(report["noise_output_bottleneck"], indent=2), flush=True)

    if args.forward:
        cfg = DiffusionTransformerConfig(**payload["config"])
        model = ConditionalDiffusionTransformer(cfg).eval()
        model.load_state_dict(state)
        loader = build_diffusion_dataloader(args.full_cache, split="validation", batch_size=1, shuffle=False, num_workers=0, max_items=1)
        batch = next(iter(loader))
        mask = batch["target_valid_mask_bt"]
        cond, cond_mask = model.encode_conditioning(
            grid=batch["grid"], grid_ids=batch["grid_ids"], grid_times_sec=batch["grid_times_sec"],
            token_times_sec=batch["token_times_sec"], target_valid_mask_bt=mask, grid_valid_mask_bt=batch["grid_valid_mask"],
        )
        clean = normalize_latent(batch["target_btd"], target_mean, target_std)
        report["validation_forward_checks"] = []
        for step in (0, 12, 24):
            noise = torch.randn_like(clean)
            t = torch.tensor([step])
            noisy = diffusion.q_sample(clean, t, noise)
            predicted = model(noisy, t, target_valid_mask_bt=mask, token_times_sec=batch["token_times_sec"], cond_btd=cond, cond_valid_mask_bt=cond_mask)
            error = predicted - noise
            perpendicular = error - (error @ q) @ q.T
            x0 = diffusion.predict_x0_from_eps(noisy, t, predicted)
            report["validation_forward_checks"].append({
                "step": step, "noise_mse": error[mask].square().mean().item(),
                "noise_mse_outside_output_span": perpendicular[mask].square().mean().item(),
                "clean_mse_before_clipping": (x0 - clean)[mask].square().mean().item(),
                "clean_fraction_clipped_at_6": x0[mask].abs().gt(6).float().mean().item(),
            })
        del model

    if args.decode:
        codec, _, _ = load_audio_codec_model(codec_family="dac", codec_model_id=str(ROOT / "runs/third_party/dac_44khz"), device="cpu", dac_num_quantizers=9)
        z, p = decoder_target
        projected = project_postsnap_latents(z, mean, u)
        a = decode_latent_to_audio(z, codec, target_layout="framewise_sum")
        b = decode_latent_to_audio(p, codec, target_layout="framewise_pca", target_pca_basis=basis)
        c = decode_latent_to_audio(projected, codec, target_layout="framewise_sum")
        report["decoder_target_checks"] = {
            "frames": z.shape[1], "sample_rate": 44100,
            "pca_vs_raw_relative_l2": ((a - b).norm() / a.norm()).item(),
            "projected_vs_raw_relative_l2": ((a - c).norm() / a.norm()).item(),
        }
        assert report["decoder_target_checks"]["pca_vs_raw_relative_l2"] < 1e-3
        assert report["decoder_target_checks"]["projected_vs_raw_relative_l2"] < 1e-3

    if int(payload["config"]["d_model"]) <= int(payload["config"]["x_dim"]):
        report["conclusion"] = "The epsilon head is narrower than the 1024-D noise it must predict."
    else:
        report["conclusion"] = "The epsilon head is wide enough for 1024-D noise."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Audit written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
