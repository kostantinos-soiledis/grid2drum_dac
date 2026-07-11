#!/usr/bin/env python3
"""Export standalone direct-PCA regressor checkpoints as decoded WAV predictions."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT.parent.parent
RUNS_ROOT = PACKAGE_ROOT / "runs"
RESULTS_ROOT = PACKAGE_ROOT / "results"


def _preload_stdlib_inspect() -> None:
    original_path = list(sys.path)
    repo = str(REPO_ROOT)
    sys.path = [path for path in sys.path if path not in {"", repo}]
    try:
        import inspect  # noqa: F401
        import dataclasses  # noqa: F401
    finally:
        sys.path = original_path


_preload_stdlib_inspect()

import torch

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.encodec_utils import (
    load_audio_codec_model,
    load_target_pca_basis,
    resolve_codec_metadata_from_cache_config,
    resolve_device,
    resolve_target_pca_basis_path_from_cache_config,
)
from io_utils import save_audio, write_json, write_jsonl
from model import decode_latent_to_audio
from scripts.conditioning_ablation import (
    VALID_CONDITIONING_ABLATIONS,
    apply_conditioning_ablation,
    conditioning_ablation_help,
    normalize_conditioning_ablation,
)
from scripts.dac_export_utils import clip_file_name, device_name, samples_per_latent_frame
from standalone_direct_pca_regressor import (
    DirectPCASequenceRegressor,
    DirectRegressorConfig,
    batch_to_device,
    build_loader,
    denormalize_latent,
)


def _progress(iterable: Any, *, desc: str) -> Any:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, leave=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=str, default=str(RUNS_ROOT / "runs_direct" / "direct_pca_regressor"))
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--cache-root", type=str, default=str(RUNS_ROOT / "mini_cache"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument(
        "--conditioning-ablation",
        type=str,
        default="none",
        choices=VALID_CONDITIONING_ABLATIONS,
        help=conditioning_ablation_help(),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve_checkpoint(run_dir: Path, explicit: str) -> Path:
    if str(explicit).strip():
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        return path
    for name in ("best_direct.pt", "best.pt", "last.pt"):
        path = run_dir / name
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"no direct PCA checkpoint found under {run_dir}")


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve()
    split = str(args.split).strip().lower()
    conditioning_ablation = normalize_conditioning_ablation(str(args.conditioning_ablation))
    checkpoint_path = _resolve_checkpoint(run_dir, str(args.checkpoint))
    default_name = f"{split}_set_predictions" if conditioning_ablation == "none" else f"{split}_set_predictions_{conditioning_ablation}"
    out_dir = Path(args.out_dir).expanduser().resolve() if str(args.out_dir).strip() else (run_dir / default_name).resolve()
    if out_dir.exists():
        if bool(args.overwrite):
            shutil.rmtree(out_dir)
        elif any(out_dir.iterdir()):
            raise FileExistsError(f"output directory already exists and is not empty: {out_dir}")
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    resolved_device = resolve_device(str(args.device))
    device = torch.device(resolved_device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    payload = dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False))
    cfg = DirectRegressorConfig(**dict(payload["config"]))
    model = DirectPCASequenceRegressor(cfg).to(device).eval()
    model.load_state_dict(dict(payload["model_state_dict"]))
    target_mean = torch.as_tensor(payload["target_mean"], dtype=torch.float32, device=device).view(-1)
    target_std = torch.as_tensor(payload["target_std"], dtype=torch.float32, device=device).view(-1).clamp_min(1.0e-6)

    codec_metadata = resolve_codec_metadata_from_cache_config(cache_root)
    codec_model, _codec_device, codec_metadata = load_audio_codec_model(device=resolved_device, metadata=codec_metadata)
    pca_basis_path = resolve_target_pca_basis_path_from_cache_config(cache_root)
    if pca_basis_path is None:
        raise FileNotFoundError(f"cache has no PCA basis: {cache_root}")
    pca_basis = load_target_pca_basis(pca_basis_path, device=device)
    sample_rate = int(codec_metadata.codec_sample_rate)

    loader = build_loader(
        cache_root,
        split=split,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        max_items=int(args.max_items),
        pin_memory=False,
    )

    manifest_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    forward_sec = 0.0
    decode_sec = 0.0
    total_audio_sec = 0.0
    dataset_index = 0
    for batch_index, batch_cpu in enumerate(_progress(loader, desc=f"export-direct[{split}]")):
        batch_ablation_cpu = apply_conditioning_ablation(
            batch_cpu,
            conditioning_ablation,
            batch_index=int(batch_index),
        )
        batch = batch_to_device(batch_ablation_cpu, device)
        mask = torch.as_tensor(batch["target_valid_mask_bt"], dtype=torch.bool, device=device)
        t0 = time.perf_counter()
        pred_norm = model(
            grid=batch["grid"],
            grid_ids=batch["grid_ids"],
            grid_times_sec=batch["grid_times_sec"],
            token_times_sec=batch["token_times_sec"],
            target_valid_mask_bt=mask,
            grid_valid_mask_bt=batch["grid_valid_mask"],
        )
        pred_pca = denormalize_latent(pred_norm, target_mean, target_std)
        pred_pca = pred_pca * mask.unsqueeze(-1).to(dtype=pred_pca.dtype)
        forward_sec += float(time.perf_counter() - t0)
        t1 = time.perf_counter()
        decoded = decode_latent_to_audio(pred_pca, codec_model, target_pca_basis=pca_basis)
        decode_sec += float(time.perf_counter() - t1)
        spf = samples_per_latent_frame(int(decoded.shape[-1]), int(mask.shape[1]))
        for batch_idx in range(int(decoded.shape[0])):
            source_id = str(batch["source_id"][batch_idx])
            beat_index = int(torch.as_tensor(batch["beat_index_b"][batch_idx]).item())
            source_manifest_index = int(torch.as_tensor(batch["source_manifest_index_b"][batch_idx]).item())
            target_frames = int(torch.as_tensor(batch["target_num_frames_b"][batch_idx]).item())
            num_samples = min(int(decoded.shape[-1]), int(target_frames) * int(spf))
            wav_rel = Path("wavs") / clip_file_name(dataset_index, source_id, beat_index)
            save_audio(out_dir / wav_rel, decoded[int(batch_idx) : int(batch_idx) + 1, :, :num_samples].detach().cpu(), sample_rate=sample_rate)
            duration_sec = float(num_samples) / float(sample_rate)
            total_audio_sec += duration_sec
            manifest_rows.append({
                "dataset_index": int(dataset_index),
                "source_id": source_id,
                "source_manifest_index": int(source_manifest_index),
                "beat_index": int(beat_index),
                "split": split,
                "conditioning_ablation": str(conditioning_ablation),
                "sample_rate": int(sample_rate),
                "num_samples": int(num_samples),
                "duration_sec": float(duration_sec),
                "target_num_frames": int(target_frames),
                "wav": str(wav_rel),
            })
            dataset_index += 1

    wall_sec = float(time.perf_counter() - started)
    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "out_dir": str(out_dir),
        "split": str(split),
        "conditioning_ablation": str(conditioning_ablation),
        "num_examples": int(len(manifest_rows)),
        "sample_rate": int(sample_rate),
        "resolved_device": str(resolved_device),
        "device_name": device_name(device),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "best_val_loss": float(payload.get("best_val_loss", float("nan"))),
        "num_parameters": int(sum(int(param.numel()) for param in model.parameters())),
        "export_wall_sec_total": float(wall_sec),
        "model_forward_sec_total": float(forward_sec),
        "codec_decode_sec_total": float(decode_sec),
        "total_audio_sec_generated": float(total_audio_sec),
        "clips_per_sec": float(len(manifest_rows)) / max(float(wall_sec), 1.0e-8),
        "audio_sec_per_sec": float(total_audio_sec) / max(float(wall_sec), 1.0e-8),
        "rtf_end_to_end": float(wall_sec) / float(total_audio_sec) if float(total_audio_sec) > 0.0 else None,
        "rtf_model_only": float(forward_sec) / float(total_audio_sec) if float(total_audio_sec) > 0.0 else None,
    }
    if device.type == "cuda":
        summary["peak_gpu_mem_allocated_mb"] = float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2))
        summary["peak_gpu_mem_reserved_mb"] = float(torch.cuda.max_memory_reserved(device) / (1024.0 ** 2))
    write_jsonl(out_dir / "manifest.jsonl", manifest_rows)
    write_json(out_dir / "summary.json", summary)
    print(f"exported {len(manifest_rows)} direct PCA predictions to {out_dir}")


if __name__ == "__main__":
    main()
