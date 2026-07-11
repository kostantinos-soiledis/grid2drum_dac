#!/usr/bin/env python3
"""Export WAV predictions for the current best diffusion checkpoint."""

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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

from data.diffusion_dataset import build_diffusion_dataloader
from data.encodec_utils import (
    load_audio_codec_model,
    load_target_pca_basis,
    resolve_codec_metadata_from_cache_config,
    resolve_codec_metadata_from_payload,
    resolve_device,
    resolve_target_layout_from_cache_config,
    resolve_target_pca_basis_path_from_cache_config,
)
from io_utils import save_audio, write_json, write_jsonl
from model import (
    DEFAULT_BEAT_CROSSFADE_MS,
    DEFAULT_INFERENCE_NUM_BEATS,
    DEFAULT_SAMPLE_X0_CLIP_NORM,
    _prepare_batch_tensors,
    apply_beat_crossfade,
    ConditionalDiffusionTransformer,
    DiffusionTransformerConfig,
    GaussianDiffusion1D,
    decode_latent_to_audio,
    denormalize_latent,
    resolve_inference_geometry,
    resolve_target_token_rate_hz,
    sample_ddpm,
)
from scripts.conditioning_ablation import (
    VALID_CONDITIONING_ABLATIONS,
    apply_conditioning_ablation,
    conditioning_ablation_help,
    normalize_conditioning_ablation,
)
try:
    from refiner import (
        DEFAULT_DAC_REFINER_STRENGTH,
        apply_dac_refiner_to_latent,
        load_dac_refiner_checkpoint,
    )
except ModuleNotFoundError as exc:
    if str(getattr(exc, "name", "")) != "refiner":
        raise
    DEFAULT_DAC_REFINER_STRENGTH = 1.0

    def load_dac_refiner_checkpoint(*_args: Any, **_kwargs: Any) -> tuple[Any, dict[str, Any]]:
        raise ModuleNotFoundError(
            "DAC refiner support is unavailable because refiner.py is missing from this checkout"
        ) from exc

    def apply_dac_refiner_to_latent(*_args: Any, **_kwargs: Any) -> Any:
        raise ModuleNotFoundError(
            "DAC refiner support is unavailable because refiner.py is missing from this checkout"
        ) from exc


def _progress(iterable: Any, *, desc: str) -> Any:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, leave=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the current best diffusion checkpoint and export split predictions under model_train.",
    )
    parser.add_argument("--train-dir", type=str, default=str(RUNS_ROOT / "model_train"))
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument(
        "--cache-root",
        type=str,
        default=str(RUNS_ROOT / "mini_cache"),
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=-1,
        help="Optional base seed for DDPM sampling; batch index is added so exported batches use distinct noise.",
    )
    parser.add_argument("--x0-clip-norm", type=float, default=DEFAULT_SAMPLE_X0_CLIP_NORM)
    parser.add_argument("--num-steps", type=int, default=400)
    parser.add_argument("--num-beats", type=int, default=DEFAULT_INFERENCE_NUM_BEATS)
    parser.add_argument(
        "--target-token-rate-hz",
        type=float,
        default=0.0,
        help="Token rate used only with --use-bpm-inference-geometry. Defaults to codec frame rate.",
    )
    parser.add_argument("--beat-crossfade-ms", type=float, default=DEFAULT_BEAT_CROSSFADE_MS)
    parser.add_argument("--use-bpm-inference-geometry", action="store_true")
    parser.add_argument("--refiner-checkpoint", type=str, default="")
    parser.add_argument("--refiner-strength", type=float, default=DEFAULT_DAC_REFINER_STRENGTH)
    parser.add_argument("--disable-refiner", action="store_true")
    parser.add_argument(
        "--conditioning-ablation",
        type=str,
        default="none",
        choices=VALID_CONDITIONING_ABLATIONS,
        help=conditioning_ablation_help(),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sanitize_name(text: str) -> str:
    clean = "".join(char if char.isalnum() else "_" for char in str(text))
    while "__" in clean:
        clean = clean.replace("__", "_")
    return clean.strip("_") or "sample"


def _clip_file_name(dataset_index: int, source_id: str, beat_index: int) -> str:
    return f"{int(dataset_index):06d}__{_sanitize_name(source_id)}__beat_{int(beat_index):04d}.wav"


def _resolve_checkpoint_path(train_dir: Path, explicit_checkpoint: str) -> Path:
    if str(explicit_checkpoint).strip():
        checkpoint_path = Path(explicit_checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    candidates = [
        train_dir / "best_diffusion.pt",
        train_dir / "best.pt",
        train_dir / "last.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"could not find a diffusion checkpoint under {train_dir}; looked for {[str(x.name) for x in candidates]}"
    )


def _resolve_out_dir(train_dir: Path, split: str, explicit_out_dir: str, conditioning_ablation: str = "none") -> Path:
    if str(explicit_out_dir).strip():
        return Path(explicit_out_dir).expanduser().resolve()
    split_name = str(split).strip().lower()
    suffix = "" if str(conditioning_ablation) == "none" else f"_{str(conditioning_ablation)}"
    return (train_dir / f"{split_name}_set_predictions{suffix}").resolve()


def _samples_per_latent_frame(decoded_num_samples: int, latent_num_frames: int) -> int:
    decoded = int(decoded_num_samples)
    frames = int(latent_num_frames)
    if decoded <= 0:
        raise ValueError(f"decoded_num_samples must be positive, got {decoded_num_samples}")
    if frames <= 0:
        raise ValueError(f"latent_num_frames must be positive, got {latent_num_frames}")
    if int(decoded) % int(frames) != 0:
        raise ValueError(
            f"decoded_num_samples={decoded} is not divisible by latent_num_frames={frames}"
        )
    return int(decoded // frames)


def _available_splits(cache_root: Path) -> list[str]:
    manifests_dir = cache_root / "manifests"
    if not manifests_dir.is_dir():
        return []
    return sorted(path.stem for path in manifests_dir.glob("*.jsonl"))


def _device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return str(torch.cuda.get_device_name(device))
    return str(device)


def _load_inference_state(
    checkpoint_path: Path,
    *,
    device: torch.device,
    fallback_num_steps: int,
) -> tuple[ConditionalDiffusionTransformer, GaussianDiffusion1D, torch.Tensor, torch.Tensor, dict[str, Any]]:
    payload = dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False))
    config_payload = dict(payload.get("config") or {})
    if not config_payload:
        raise KeyError(f"checkpoint is missing config: {checkpoint_path}")
    config_payload.setdefault("positional_encoding", "index")
    config_payload.setdefault("positional_rate_hz", 50.0)
    model_state = payload.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise KeyError(f"checkpoint is missing model_state_dict: {checkpoint_path}")

    cfg = DiffusionTransformerConfig(**config_payload)
    model = ConditionalDiffusionTransformer(cfg).to(device).eval()
    model.load_state_dict(model_state)

    num_steps = int(payload.get("num_steps") or int(fallback_num_steps))
    diffusion = GaussianDiffusion1D(num_steps=num_steps).to(device)
    target_mean = torch.as_tensor(payload["target_mean"], dtype=torch.float32, device=device).view(-1)
    target_std = torch.as_tensor(payload["target_std"], dtype=torch.float32, device=device).view(-1).clamp_min(1.0e-6)
    payload["resolved_num_steps"] = int(num_steps)
    return model, diffusion, target_mean.contiguous(), target_std.contiguous(), payload


def main() -> None:
    args = _parse_args()

    train_dir = Path(args.train_dir).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve()
    split = str(args.split).strip().lower()
    conditioning_ablation = normalize_conditioning_ablation(str(args.conditioning_ablation))
    checkpoint_path = _resolve_checkpoint_path(train_dir, str(args.checkpoint))
    out_dir = _resolve_out_dir(train_dir, split, str(args.out_dir), conditioning_ablation)

    available_splits = _available_splits(cache_root)
    split_manifest = cache_root / "manifests" / f"{split}.jsonl"
    if not split_manifest.is_file():
        raise FileNotFoundError(
            f"split={split!r} not found under {cache_root}; available_splits={available_splits}"
        )

    if out_dir.exists():
        if bool(args.overwrite):
            shutil.rmtree(out_dir)
        elif any(out_dir.iterdir()):
            raise FileExistsError(f"output directory already exists and is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    resolved_device = resolve_device(str(args.device))
    device = torch.device(resolved_device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    pin_memory = resolved_device.startswith("cuda")

    model, diffusion, target_mean, target_std, checkpoint_payload = _load_inference_state(
        checkpoint_path,
        device=device,
        fallback_num_steps=int(args.num_steps),
    )
    refiner_model = None
    refiner_payload: dict[str, Any] = {}
    if str(args.refiner_checkpoint).strip() and not bool(args.disable_refiner):
        refiner_model, refiner_payload = load_dac_refiner_checkpoint(args.refiner_checkpoint, device=device)
    codec_metadata = resolve_codec_metadata_from_payload(
        checkpoint_payload,
        fallback=resolve_codec_metadata_from_cache_config(cache_root),
    )
    target_layout = str(
        checkpoint_payload.get("target_layout")
        or resolve_target_layout_from_cache_config(cache_root)
    ).strip().lower()
    target_pca_basis = None
    if checkpoint_payload.get("target_pca_basis") is not None:
        target_pca_basis = load_target_pca_basis(
            checkpoint_payload["target_pca_basis"],
            device=device,
        )
    else:
        basis_path = resolve_target_pca_basis_path_from_cache_config(cache_root)
        if basis_path is not None:
            target_pca_basis = load_target_pca_basis(basis_path, device=device)
    if target_layout == "framewise_pca" and target_pca_basis is None:
        raise FileNotFoundError(
            f"checkpoint/cache use target_layout=framewise_pca but no PCA basis could be resolved: {checkpoint_path}"
        )
    encodec_model, _resolved_codec_device, codec_metadata = load_audio_codec_model(
        device=resolved_device,
        metadata=codec_metadata,
    )
    sample_rate = int(checkpoint_payload.get("sample_rate") or codec_metadata.codec_sample_rate)
    target_token_rate_hz = (
        float(args.target_token_rate_hz)
        if float(args.target_token_rate_hz) > 0.0
        else resolve_target_token_rate_hz(codec_metadata)
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    dataloader = build_diffusion_dataloader(
        cache_root,
        split=split,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        max_items=int(args.max_items),
        pin_memory=pin_memory,
    )

    manifest_rows: list[dict[str, Any]] = []
    export_started_at = time.perf_counter()
    model_forward_sec_total = 0.0
    codec_decode_sec_total = 0.0
    total_audio_sec_generated = 0.0
    dataset_index = 0

    for batch_index, batch in enumerate(_progress(dataloader, desc=f"export[{split}]")):
        ablated_batch = apply_conditioning_ablation(
            batch,
            conditioning_ablation,
            batch_index=int(batch_index),
        )
        prepared = _prepare_batch_tensors(
            ablated_batch,
            device,
            require_target=not bool(args.use_bpm_inference_geometry),
            require_timing=not bool(args.use_bpm_inference_geometry),
        )
        inference_geometry = resolve_inference_geometry(
            prepared,
            use_bpm_inference_geometry=bool(args.use_bpm_inference_geometry),
            inference_num_beats=int(args.num_beats),
            target_token_rate_hz=float(target_token_rate_hz),
        )
        forward_started_at = time.perf_counter()
        pred_latent_norm = sample_ddpm(
            model=model,
            diffusion=diffusion,
            batch=ablated_batch,
            device=device,
            guidance_scale=float(args.guidance_scale),
            x0_clip_norm=float(args.x0_clip_norm) if args.x0_clip_norm is not None else None,
            sample_seed=(int(args.sample_seed) + int(batch_index) if int(args.sample_seed) >= 0 else None),
            use_bpm_inference_geometry=bool(args.use_bpm_inference_geometry),
            inference_num_beats=int(args.num_beats),
            target_token_rate_hz=float(target_token_rate_hz),
            inference_geometry=inference_geometry,
        )
        target_valid_mask_bt = inference_geometry["target_valid_mask_bt"]
        pred_latent = denormalize_latent(pred_latent_norm, target_mean, target_std)
        pred_latent = pred_latent * target_valid_mask_bt.unsqueeze(-1)
        if refiner_model is not None:
            pred_latent = apply_dac_refiner_to_latent(
                refiner_model,
                pred_latent,
                prepared,
                inference_geometry,
                strength=float(args.refiner_strength),
            )
            pred_latent = pred_latent * target_valid_mask_bt.unsqueeze(-1)
        model_forward_sec_total += float(time.perf_counter() - forward_started_at)

        decode_started_at = time.perf_counter()
        decoded_audio = decode_latent_to_audio(
            pred_latent,
            encodec_model,
            target_pca_basis=target_pca_basis,
        )
        codec_decode_sec_total += float(time.perf_counter() - decode_started_at)

        if int(decoded_audio.dim()) == 2:
            decoded_audio = decoded_audio.unsqueeze(1)
        if int(decoded_audio.dim()) != 3:
            raise RuntimeError(f"unexpected decoded audio shape: {tuple(decoded_audio.shape)}")
        samples_per_frame = _samples_per_latent_frame(
            decoded_num_samples=int(decoded_audio.shape[-1]),
            latent_num_frames=int(target_valid_mask_bt.shape[1]),
        )

        batch_size = int(decoded_audio.shape[0])
        for batch_idx in range(batch_size):
            source_id = str(batch["source_id"][batch_idx])
            beat_index = int(torch.as_tensor(batch["beat_index_b"][batch_idx]).item())
            source_manifest_index = int(torch.as_tensor(batch["source_manifest_index_b"][batch_idx]).item())
            split_name = str(batch["split"][batch_idx])
            wav_name = _clip_file_name(dataset_index, source_id, beat_index)
            wav_rel = Path("wavs") / wav_name
            target_num_frames = int(inference_geometry["target_num_frames_b"][int(batch_idx)].item())
            num_samples = int(target_num_frames) * int(samples_per_frame)
            audio_i = decoded_audio[int(batch_idx), :, : int(num_samples)].detach()
            if float(args.beat_crossfade_ms) > 0.0:
                beat_boundaries_valid_mask = inference_geometry["beat_boundaries_valid_mask"][int(batch_idx)]
                beat_boundaries_sec = inference_geometry["beat_boundaries_sec"][int(batch_idx)][beat_boundaries_valid_mask]
                audio_i = apply_beat_crossfade(
                    audio_i,
                    beat_boundaries_sec,
                    sample_rate=int(sample_rate),
                    beat_crossfade_ms=float(args.beat_crossfade_ms),
                )
            audio_i = audio_i.unsqueeze(0).cpu()
            num_samples = int(audio_i.shape[-1])
            duration_sec = float(num_samples) / float(sample_rate)
            total_audio_sec_generated += float(duration_sec)
            save_audio(out_dir / wav_rel, audio_i, sample_rate=sample_rate)
            manifest_rows.append(
                {
                    "dataset_index": int(dataset_index),
                    "source_id": source_id,
                    "source_manifest_index": int(source_manifest_index),
                    "beat_index": int(beat_index),
                    "split": split_name,
                    "conditioning_ablation": str(conditioning_ablation),
                    "sample_rate": int(sample_rate),
                    "num_samples": int(num_samples),
                    "duration_sec": float(duration_sec),
                    "target_num_frames": int(target_num_frames),
                    "wav": str(wav_rel),
                }
            )
            dataset_index += 1

    export_wall_sec_total = float(time.perf_counter() - export_started_at)
    num_examples = int(len(manifest_rows))
    clips_per_sec = float(num_examples) / max(export_wall_sec_total, 1.0e-8)
    audio_sec_per_sec = float(total_audio_sec_generated) / max(export_wall_sec_total, 1.0e-8)
    summary_payload = {
        "train_dir": str(train_dir),
        "checkpoint": str(checkpoint_path),
        "out_dir": str(out_dir),
        "split": split,
        "conditioning_ablation": str(conditioning_ablation),
        "num_examples": int(num_examples),
        "batch_size": int(args.batch_size),
        "guidance_scale": float(args.guidance_scale),
        "sample_seed": (int(args.sample_seed) if int(args.sample_seed) >= 0 else None),
        "samples_per_conditioning_input": 1,
        "x0_clip_norm": float(args.x0_clip_norm) if args.x0_clip_norm is not None else None,
        "use_bpm_inference_geometry": bool(args.use_bpm_inference_geometry),
        "inference_num_beats": int(args.num_beats),
        "beat_crossfade_ms": float(args.beat_crossfade_ms),
        "target_token_rate_hz": float(target_token_rate_hz),
        "refiner_checkpoint": str(refiner_payload.get("checkpoint_path") or args.refiner_checkpoint or ""),
        "refiner_enabled": bool(refiner_model is not None),
        "refiner_strength": float(args.refiner_strength),
        "resolved_device": resolved_device,
        "device_name": _device_name(device),
        "sample_rate": int(sample_rate),
        "num_steps": int(checkpoint_payload["resolved_num_steps"]),
        "checkpoint_epoch": int(checkpoint_payload.get("epoch", -1)),
        "best_val_loss": (
            float(checkpoint_payload["best_val_loss"])
            if checkpoint_payload.get("best_val_loss") is not None
            else None
        ),
        "best_checkpoint_metric_name": (
            str(checkpoint_payload["best_checkpoint_metric_name"])
            if checkpoint_payload.get("best_checkpoint_metric_name") is not None
            else None
        ),
        "best_checkpoint_metric_value": (
            float(checkpoint_payload["best_checkpoint_metric_value"])
            if checkpoint_payload.get("best_checkpoint_metric_value") is not None
            else None
        ),
        "best_checkpoint_epoch": (
            int(checkpoint_payload["best_checkpoint_epoch"])
            if checkpoint_payload.get("best_checkpoint_epoch") is not None
            else None
        ),
        "num_parameters": int(sum(int(param.numel()) for param in model.parameters())),
        "export_wall_sec_total": float(export_wall_sec_total),
        "model_forward_sec_total": float(model_forward_sec_total),
        "codec_decode_sec_total": float(codec_decode_sec_total),
        "total_audio_sec_generated": float(total_audio_sec_generated),
        "clips_per_sec": float(clips_per_sec),
        "audio_sec_per_sec": float(audio_sec_per_sec),
        "rtf_end_to_end": (
            float(export_wall_sec_total) / float(total_audio_sec_generated)
            if float(total_audio_sec_generated) > 0.0
            else None
        ),
        "rtf_model_only": (
            float(model_forward_sec_total) / float(total_audio_sec_generated)
            if float(total_audio_sec_generated) > 0.0
            else None
        ),
    }
    if device.type == "cuda":
        summary_payload["peak_gpu_mem_allocated_mb"] = float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2))
        summary_payload["peak_gpu_mem_reserved_mb"] = float(torch.cuda.max_memory_reserved(device) / (1024.0 ** 2))

    write_jsonl(out_dir / "manifest.jsonl", manifest_rows)
    write_json(out_dir / "summary.json", summary_payload)
    print(f"exported {num_examples} predictions to {out_dir}")


if __name__ == "__main__":
    main()
