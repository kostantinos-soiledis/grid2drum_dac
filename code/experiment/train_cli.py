#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import faulthandler
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = REPO_ROOT.parent.parent
RUNS_ROOT = PACKAGE_ROOT / "runs"
RESULTS_ROOT = PACKAGE_ROOT / "results"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")


def _preload_stdlib_inspect() -> None:
    """Avoid the repo-local inspect.py shadowing Python's stdlib inspect."""
    original_path = list(sys.path)

    def _keep_path(path: str) -> bool:
        if path == "":
            try:
                return Path.cwd().resolve() != REPO_ROOT
            except OSError:
                return True
        try:
            return Path(path).resolve() != REPO_ROOT
        except OSError:
            return True

    sys.path = [path for path in sys.path if _keep_path(str(path))]
    try:
        import inspect  # noqa: F401
        import dataclasses  # noqa: F401
    finally:
        sys.path = original_path


_preload_stdlib_inspect()

import torch
from dataclasses import asdict

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

from data.diffusion_dataset import build_diffusion_dataloader
from data.encodec_utils import (
    extract_codebook_embeddings,
    load_audio_codec_model,
    load_target_pca_basis,
    resolve_codec_metadata_from_cache_config,
    resolve_codec_metadata_from_payload,
    resolve_device,
    resolve_target_layout_from_cache_config,
    resolve_target_pca_basis_path_from_cache_config,
)
from io_utils import append_jsonl, write_json, write_jsonl
from model import (
    DEFAULT_INFERENCE_NUM_BEATS,
    DEFAULT_AUDIO_MRSTFT_RESOLUTIONS,
    DEFAULT_AUDIO_MRSTFT_WEIGHT,
    DEFAULT_AUDIO_WAVE_L1_WEIGHT,
    DEFAULT_FRONTEND_CHUNK_SIZE,
    DEFAULT_FRONTEND_CLASS_LOCAL_DIM,
    DEFAULT_FRONTEND_EMBED_DIM,
    DEFAULT_FRONTEND_OUTPUT_KIND,
    DEFAULT_FRONTEND_PADDING_MODE,
    DEFAULT_FRONTEND_PRIMARY_RADIUS,
    DEFAULT_FRONTEND_RADII,
    DEFAULT_FRONTEND_STEP_SECONDS,
    DEFAULT_FRONTEND_VARIANT,
    DEFAULT_SAMPLE_X0_CLIP_NORM,
    ConditionalDiffusionTransformer,
    DiffusionTransformerConfig,
    GaussianDiffusion1D,
    build_frontend_cfg_from_batch,
    diffusion_train_step,
    load_or_compute_target_normalization,
    resolve_encodec_sample_rate,
    resolve_target_token_rate_hz,
    save_eval_plot_multi_t,
    save_inference_wav,
)
from scripts.conditioning_ablation import (
    VALID_CONDITIONING_ABLATIONS,
    apply_conditioning_ablation,
    conditioning_ablation_help,
    normalize_conditioning_ablation,
)

DEFAULT_EVAL_PLOT_STEPS = "auto"


def _progress(iterable: Any, *, desc: str, total: int | None = None) -> Any:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, leave=False)


def _parse_int_tuple(text: str) -> tuple[int, ...]:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if not parts:
        raise ValueError("expected at least one integer value")
    return tuple(int(part) for part in parts)


def _parse_optional_int_tuple(text: str) -> tuple[int, ...]:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if not parts:
        return tuple()
    return tuple(int(part) for part in parts)


def _format_mrstft_resolutions(resolutions: tuple[tuple[int, int], ...]) -> str:
    return ",".join(f"{int(n_fft)}:{int(hop)}" for n_fft, hop in tuple(resolutions))


def _parse_mrstft_resolutions(text: str) -> tuple[tuple[int, int], ...]:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if not parts:
        raise ValueError("expected at least one n_fft:hop resolution")
    resolved: list[tuple[int, int]] = []
    for part in parts:
        if ":" not in part:
            raise ValueError(
                f"invalid MRSTFT resolution {part!r}; expected comma-separated n_fft:hop pairs"
            )
        n_fft_text, hop_text = part.split(":", maxsplit=1)
        n_fft = int(n_fft_text)
        hop = int(hop_text)
        if n_fft <= 0 or hop <= 0:
            raise ValueError(f"MRSTFT resolution values must be positive, got {part!r}")
        resolved.append((int(n_fft), int(hop)))
    return tuple(resolved)


def _default_eval_plot_steps_for_num_steps(num_steps: int) -> tuple[int, ...]:
    max_step = int(num_steps) - 1
    if max_step < 0:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    if max_step == 0:
        return (0,)
    fractions = (0.25, 0.5, 0.85)
    steps = sorted({min(max_step, max(0, int(round(max_step * frac)))) for frac in fractions})
    return tuple(steps)


def _resolve_eval_plot_steps(text: str, *, num_steps: int) -> tuple[int, ...]:
    normalized_text = str(text).strip().lower()
    max_step = int(num_steps) - 1
    if max_step < 0:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    if normalized_text == DEFAULT_EVAL_PLOT_STEPS:
        return _default_eval_plot_steps_for_num_steps(int(num_steps))
    plot_steps = _parse_int_tuple(str(text))
    invalid = [int(step) for step in plot_steps if int(step) < 0 or int(step) > max_step]
    if not invalid:
        return plot_steps
    raise ValueError(
        f"--eval-plot-steps must be 'auto' or within [0, {max_step}] for num_steps={num_steps}; got {text!r}"
    )


def _build_fixed_preview_noises(
    *,
    seed: int,
    target_shape: tuple[int, ...],
    preview_shape: tuple[int, ...],
    dtype: torch.dtype,
    plot_steps: tuple[int, ...],
    num_steps: int,
) -> tuple[int, dict[int, torch.Tensor], torch.Tensor, dict[int, torch.Tensor]]:
    preview_seed = int(seed) + 1_000_003
    preview_generator = torch.Generator(device="cpu")
    preview_generator.manual_seed(int(preview_seed))

    fixed_noises = {
        int(step): torch.randn(target_shape, dtype=dtype, generator=preview_generator)
        for step in tuple(plot_steps)
    }
    fixed_start_noise = torch.randn(preview_shape, dtype=dtype, generator=preview_generator)
    fixed_step_noises = {
        int(step): torch.randn(preview_shape, dtype=dtype, generator=preview_generator)
        for step in range(1, int(num_steps))
    }
    return int(preview_seed), fixed_noises, fixed_start_noise, fixed_step_noises


def _should_retry_without_pin_memory(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "pin memory" in message
        or ("out of memory" in message and "cuda" in message)
        or "acceleratorerror" in exc.__class__.__name__.lower()
    )


def _resolve_checkpoint_metric_name(
    requested_metric: str,
    *,
    audio_wave_l1_weight: float,
    audio_mrstft_weight: float,
) -> str:
    requested = str(requested_metric).strip().lower()
    if requested != "auto":
        return requested
    if float(audio_mrstft_weight) > 0.0:
        return "val_audio_mrstft"
    if float(audio_wave_l1_weight) > 0.0:
        return "val_audio_wave_l1"
    return "val_loss"


def _resolve_bpm_preview_target_frames(
    batch: dict[str, Any],
    sample_idx: int,
    *,
    num_beats: int,
    target_token_rate_hz: float,
) -> int:
    bpm = torch.as_tensor(batch.get("bpm"), dtype=torch.float32).view(-1)
    duration = torch.as_tensor(batch.get("duration_sec"), dtype=torch.float32).view(-1)
    if int(bpm.numel()) <= int(sample_idx):
        raise IndexError(f"sample_idx={sample_idx} out of range for preview BPM tensor")
    if int(duration.numel()) <= int(sample_idx):
        raise IndexError(f"sample_idx={sample_idx} out of range for preview duration tensor")
    bpm_value = float(bpm[int(sample_idx)].item())
    if bpm_value > 1.0e-6:
        duration_sec = (float(max(1, int(num_beats))) * 60.0) / bpm_value
    else:
        duration_sec = float(duration[int(sample_idx)].item())
    if not duration_sec > 0.0:
        raise ValueError(f"preview sample has invalid BPM/duration: bpm={bpm_value} duration={duration_sec}")
    return int(max(1, round(float(duration_sec) * float(target_token_rate_hz))))


def _write_history_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if str(key) not in fieldnames:
                fieldnames.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_history_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = str(line).strip()
            if not text:
                continue
            rows.append(dict(json.loads(text)))
    return rows


def _optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device=device)


def _load_init_checkpoint_payload(
    init_checkpoint: str,
    *,
    expected_cfg: DiffusionTransformerConfig,
) -> dict[str, Any]:
    checkpoint_path = Path(init_checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"init checkpoint not found: {checkpoint_path}")
    payload = dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False))
    config_payload = dict(payload.get("config") or {})
    if not config_payload:
        raise KeyError(f"init checkpoint missing config: {checkpoint_path}")
    config_payload.setdefault("positional_encoding", "index")
    config_payload.setdefault("positional_rate_hz", 50.0)
    loaded_cfg = DiffusionTransformerConfig(**config_payload)
    payload["_allow_partial_model_state_dict"] = False
    if asdict(loaded_cfg) != asdict(expected_cfg):
        timbre_keys = {
            "timbre_conditioning",
            "timbre_bank_dim",
            "timbre_num_families",
            "timbre_max_classes",
            "timbre_velocity_bins",
            "timbre_dropout_prob",
            "timbre_class_dropout_prob",
        }
        loaded_base = {key: value for key, value in asdict(loaded_cfg).items() if key not in timbre_keys}
        expected_base = {key: value for key, value in asdict(expected_cfg).items() if key not in timbre_keys}
        if loaded_base != expected_base:
            raise ValueError(
                "init checkpoint config does not match the requested model/frontend configuration. "
                f"checkpoint={checkpoint_path}"
            )
        payload["_allow_partial_model_state_dict"] = True
    payload["checkpoint_path"] = str(checkpoint_path)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the diffusion model on the cached seconds frontend inputs.")
    parser.add_argument(
        "--cache-root",
        type=str,
        default=str(RUNS_ROOT / "mini_cache"),
    )
    parser.add_argument("--out-dir", type=str, default=str(RUNS_ROOT / "model_train"))
    parser.add_argument("--init-checkpoint", type=str, default="")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from --out-dir/last.pt, preserving history and optimizer state.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default="",
        help="Resume from an explicit checkpoint instead of --out-dir/last.pt.",
    )
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="validation")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--no-pin-memory",
        action="store_true",
        help="Disable DataLoader pinned memory for CUDA runs; useful for isolating native CUDA/NVML crashes.",
    )
    parser.add_argument(
        "--dataloader-multiprocessing-context",
        type=str,
        default="auto",
        choices=("auto", "fork", "spawn", "forkserver"),
        help="Multiprocessing start method for DataLoader workers; auto uses spawn for CUDA runs.",
    )
    parser.add_argument(
        "--no-persistent-workers",
        action="store_true",
        help="Disable persistent DataLoader workers when --num-workers > 0.",
    )
    parser.add_argument("--max-train-items", type=int, default=0)
    parser.add_argument("--max-val-items", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sample-idx", type=int, default=3)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument(
        "--num-steps",
        type=int,
        default=400,
        help="Number of diffusion timesteps in the noise schedule; training progress bars report epoch batches.",
    )
    parser.add_argument(
        "--eval-plot-steps",
        type=str,
        default=DEFAULT_EVAL_PLOT_STEPS,
        help="Comma-separated diffusion steps for eval plots, or 'auto' to scale with --num-steps.",
    )
    parser.add_argument("--x0-clip-norm", type=float, default=DEFAULT_SAMPLE_X0_CLIP_NORM)
    parser.add_argument("--audio-wave-l1-weight", type=float, default=DEFAULT_AUDIO_WAVE_L1_WEIGHT)
    parser.add_argument("--audio-mrstft-weight", type=float, default=DEFAULT_AUDIO_MRSTFT_WEIGHT)
    parser.add_argument(
        "--audio-mrstft-resolutions",
        type=str,
        default=_format_mrstft_resolutions(DEFAULT_AUDIO_MRSTFT_RESOLUTIONS),
    )
    parser.add_argument(
        "--checkpoint-metric",
        type=str,
        default="auto",
        choices=(
            "auto",
            "val_loss",
            "val_diffusion_loss",
            "val_audio_wave_l1",
            "val_audio_mrstft",
            "val_x0",
            "val_x0_loss",
            "val_timbre_proj_mse",
            "val_quant_embed_mse",
            "val_rvq_ce",
            "val_onset_weighted_x0",
        ),
    )
    parser.add_argument("--timbre-probe-path", type=str, default="")
    parser.add_argument(
        "--timbre-bank-path",
        type=str,
        default="",
        help="Optional support_bank.pt exported by timbre_transfer for reference-kit conditioning.",
    )
    parser.add_argument("--timbre-dropout-prob", type=float, default=0.0)
    parser.add_argument("--timbre-class-dropout-prob", type=float, default=0.0)
    parser.add_argument("--x0-mse-weight", type=float, default=0.0)
    parser.add_argument("--timbre-proj-mse-weight", type=float, default=0.0)
    parser.add_argument("--quant-embed-mse-weight", type=float, default=0.0)
    parser.add_argument("--rvq-ce-weight", type=float, default=0.0)
    parser.add_argument("--onset-loss-weighting", action="store_true")
    parser.add_argument("--onset-token-radius", type=int, default=1)
    parser.add_argument(
        "--use-bpm-training-geometry",
        action="store_true",
        help=(
            "Retiming-only current-cache mode: derive train/val geometry from BPM while preserving "
            "cached grid and target frame counts."
        ),
    )
    parser.add_argument(
        "--bpm-geometry-num-beats",
        type=int,
        default=DEFAULT_INFERENCE_NUM_BEATS,
        help="Beat count used for BPM-derived train/preview geometry.",
    )
    parser.add_argument(
        "--fixed-sample-epochs",
        type=str,
        default="",
        help="Comma-separated epochs to export fixed preview WAVs/plots regardless of checkpoint improvement.",
    )
    parser.add_argument("--frontend-variant", type=str, default=DEFAULT_FRONTEND_VARIANT)
    parser.add_argument("--frontend-embed-dim", type=int, default=DEFAULT_FRONTEND_EMBED_DIM)
    parser.add_argument("--frontend-output-kind", type=str, default=DEFAULT_FRONTEND_OUTPUT_KIND)
    parser.add_argument("--frontend-radii", type=str, default=",".join(str(x) for x in DEFAULT_FRONTEND_RADII))
    parser.add_argument("--frontend-primary-radius", type=int, default=DEFAULT_FRONTEND_PRIMARY_RADIUS)
    parser.add_argument("--frontend-padding-mode", type=str, default=DEFAULT_FRONTEND_PADDING_MODE)
    parser.add_argument("--frontend-step-seconds", type=float, default=DEFAULT_FRONTEND_STEP_SECONDS)
    parser.add_argument("--frontend-chunk-size", type=int, default=DEFAULT_FRONTEND_CHUNK_SIZE)
    parser.add_argument("--frontend-class-local-fusion", action="store_true")
    parser.add_argument("--frontend-class-local-dim", type=int, default=DEFAULT_FRONTEND_CLASS_LOCAL_DIM)
    parser.add_argument("--no-concat-multiscale-frontend", action="store_true")
    parser.add_argument(
        "--conditioning-ablation",
        type=str,
        default="none",
        choices=VALID_CONDITIONING_ABLATIONS,
        help="Training/validation symbolic conditioning mode. " + conditioning_ablation_help(),
    )
    parser.add_argument("--positional-encoding", type=str, default="seconds", choices=("seconds", "index"))
    parser.add_argument(
        "--positional-rate-hz",
        type=float,
        default=0.0,
        help="Seconds positional scale. Defaults to the cache codec frame rate.",
    )
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cond-dropout-prob", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_timbre_bank_payload(path_text: str) -> dict[str, Any] | None:
    path = Path(str(path_text).strip()).expanduser().resolve()
    if not str(path_text).strip():
        return None
    if not path.is_file():
        raise FileNotFoundError(f"timbre bank not found: {path}")
    payload = dict(torch.load(path, map_location="cpu", weights_only=False))
    required = (
        "timbre_bank_latents",
        "timbre_bank_family_ids",
        "timbre_bank_class_ids",
        "timbre_bank_velocity",
        "timbre_bank_mask",
        "timbre_family_default_indices",
        "timbre_class_token_indices",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"timbre bank is missing required tensors: {missing}")
    payload["_path"] = str(path)
    return payload


def _attach_timbre_bank_to_batch(batch: dict[str, Any], bank: dict[str, Any] | None) -> dict[str, Any]:
    if bank is None:
        return batch
    out = dict(batch)
    batch_size = int(torch.as_tensor(batch["grid"]).shape[0])

    def _expand(key: str) -> torch.Tensor:
        value = torch.as_tensor(bank[key]).contiguous()
        if int(value.dim()) >= 1 and int(value.shape[0]) == int(batch_size):
            return value
        return value.unsqueeze(0).expand(int(batch_size), *tuple(value.shape)).contiguous()

    for key in (
        "timbre_bank_latents",
        "timbre_bank_family_ids",
        "timbre_bank_class_ids",
        "timbre_bank_velocity",
        "timbre_bank_mask",
        "timbre_family_default_indices",
        "timbre_class_token_indices",
    ):
        out[key] = _expand(key)
    return out


def _save_diffusion_checkpoint(
    path: str | Path,
    *,
    model: ConditionalDiffusionTransformer,
    optimizer: torch.optim.Optimizer | None,
    cfg: DiffusionTransformerConfig,
    frontend_cfg: dict[str, Any],
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    sample_rate: int,
    codec_metadata: dict[str, Any],
    num_steps: int,
    epoch: int,
    best_val_loss: float,
    extra_payload: dict[str, Any] | None = None,
) -> None:
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "best_val_loss": float(best_val_loss),
        "target_mean": torch.as_tensor(target_mean, dtype=torch.float32).detach().cpu(),
        "target_std": torch.as_tensor(target_std, dtype=torch.float32).detach().cpu(),
        "sample_rate": int(sample_rate),
        "codec_metadata": dict(codec_metadata),
        "num_steps": int(num_steps),
        "frontend_cfg": dict(frontend_cfg),
        "config": asdict(cfg),
    }
    if extra_payload:
        payload.update(extra_payload)
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path_obj)


def main() -> None:
    faulthandler.enable(all_threads=True)
    args = _parse_args()
    conditioning_ablation = normalize_conditioning_ablation(str(args.conditioning_ablation))
    if int(args.grad_accum_steps) <= 0:
        raise ValueError(f"--grad-accum-steps must be >= 1, got {args.grad_accum_steps}")
    if int(args.bpm_geometry_num_beats) <= 0:
        raise ValueError(f"--bpm-geometry-num-beats must be >= 1, got {args.bpm_geometry_num_beats}")
    frontend_radii = _parse_int_tuple(str(args.frontend_radii))
    audio_mrstft_resolutions = _parse_mrstft_resolutions(str(args.audio_mrstft_resolutions))
    plot_steps = _resolve_eval_plot_steps(str(args.eval_plot_steps), num_steps=int(args.num_steps))
    fixed_sample_epochs = _parse_optional_int_tuple(str(args.fixed_sample_epochs))
    cache_root_path = Path(args.cache_root).expanduser().resolve()
    invalid_fixed_epochs = [int(epoch) for epoch in fixed_sample_epochs if int(epoch) < 0]
    if invalid_fixed_epochs:
        raise ValueError(f"--fixed-sample-epochs must be non-negative, got {invalid_fixed_epochs}")

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    resolved_device = resolve_device(str(args.device))
    device = torch.device(resolved_device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    pin_memory = resolved_device.startswith("cuda") and not bool(args.no_pin_memory)
    dataloader_mp_context: str | None = None
    requested_mp_context = str(args.dataloader_multiprocessing_context).strip().lower()
    if int(args.num_workers) > 0:
        if requested_mp_context == "auto":
            dataloader_mp_context = "spawn" if device.type == "cuda" else None
        else:
            dataloader_mp_context = requested_mp_context
    persistent_workers = int(args.num_workers) > 0 and not bool(args.no_persistent_workers)
    timbre_projection: torch.Tensor | None = None
    timbre_probe_metadata: dict[str, Any] = {}
    timbre_probe_path = str(args.timbre_probe_path).strip()
    if float(args.timbre_proj_mse_weight) > 0.0 and not timbre_probe_path:
        raise ValueError("--timbre-probe-path is required when --timbre-proj-mse-weight > 0")
    if timbre_probe_path:
        probe_path = Path(timbre_probe_path).expanduser().resolve()
        if not probe_path.is_file():
            raise FileNotFoundError(f"timbre probe not found: {probe_path}")
        probe_payload = dict(torch.load(probe_path, map_location="cpu", weights_only=False))
        projection_payload = probe_payload.get("projection_matrix")
        if projection_payload is None:
            raise KeyError(f"timbre probe missing projection_matrix: {probe_path}")
        timbre_projection = torch.as_tensor(projection_payload, dtype=torch.float32, device=device).contiguous()
        if int(timbre_projection.dim()) != 2:
            raise ValueError(
                f"expected timbre projection [K,D], got {tuple(timbre_projection.shape)} from {probe_path}"
            )
        timbre_probe_metadata = dict(probe_payload.get("metadata") or {})

    out_dir = Path(args.out_dir).resolve()
    resume_checkpoint_arg = str(args.resume_checkpoint or "").strip()
    resume_requested = bool(args.resume) or bool(resume_checkpoint_arg)
    if bool(args.overwrite) and bool(resume_requested):
        raise ValueError("--overwrite cannot be combined with --resume/--resume-checkpoint")
    if bool(resume_requested) and str(args.init_checkpoint).strip():
        raise ValueError("--resume/--resume-checkpoint cannot be combined with --init-checkpoint")
    resume_checkpoint_path = (
        Path(resume_checkpoint_arg).expanduser().resolve()
        if resume_checkpoint_arg
        else (out_dir / "last.pt").resolve()
    )

    if bool(args.overwrite) and out_dir.exists():
        shutil.rmtree(out_dir)
    existing_run_artifacts = [
        path
        for path in (
            out_dir / "last.pt",
            out_dir / "best_diffusion.pt",
            out_dir / "history.jsonl",
            out_dir / "history.csv",
            out_dir / "run_config.json",
        )
        if path.exists()
    ]
    if bool(resume_requested):
        if not resume_checkpoint_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_checkpoint_path}")
    elif existing_run_artifacts:
        artifacts = ", ".join(str(path.name) for path in existing_run_artifacts)
        raise RuntimeError(
            f"out-dir already contains training artifacts ({artifacts}). "
            "Use --resume to continue it, --overwrite to replace it, or choose a new --out-dir."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "history.jsonl"
    history_csv_path = out_dir / "history.csv"
    if not bool(resume_requested):
        history_path.write_text("", encoding="utf-8")

    def _make_train_loader(*, pin_memory_enabled: bool):
        return build_diffusion_dataloader(
            args.cache_root,
            split=str(args.train_split),
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=int(args.num_workers),
            max_items=int(args.max_train_items),
            pin_memory=pin_memory_enabled,
            persistent_workers=bool(persistent_workers),
            multiprocessing_context=dataloader_mp_context,
        )

    def _make_val_loader(*, pin_memory_enabled: bool):
        return build_diffusion_dataloader(
            args.cache_root,
            split=str(args.val_split),
            batch_size=int(args.eval_batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            max_items=int(args.max_val_items),
            pin_memory=pin_memory_enabled,
            persistent_workers=bool(persistent_workers),
            multiprocessing_context=dataloader_mp_context,
        )

    train_loader = _make_train_loader(pin_memory_enabled=pin_memory)
    val_loader = _make_val_loader(pin_memory_enabled=pin_memory)
    try:
        sample_batch = next(iter(train_loader))
        fixed_val_batch = next(iter(val_loader))
    except Exception as exc:
        if pin_memory and _should_retry_without_pin_memory(exc):
            print(
                "dataloader warmup failed with pin_memory=True; retrying with pin_memory=False "
                f"({exc.__class__.__name__}: {exc})"
            )
            pin_memory = False
            train_loader = _make_train_loader(pin_memory_enabled=False)
            val_loader = _make_val_loader(pin_memory_enabled=False)
            sample_batch = next(iter(train_loader))
            fixed_val_batch = next(iter(val_loader))
        else:
            raise
    train_batches_per_epoch = int(len(train_loader))
    val_batches_per_epoch = int(len(val_loader))

    frontend_cfg = build_frontend_cfg_from_batch(
        sample_batch,
        variant=str(args.frontend_variant),
        embed_dim=int(args.frontend_embed_dim),
        output_kind=str(args.frontend_output_kind),
        radii=frontend_radii,
        primary_radius=int(args.frontend_primary_radius),
        padding_mode=str(args.frontend_padding_mode),
        step_seconds=float(args.frontend_step_seconds),
        chunk_size=int(args.frontend_chunk_size),
        class_local_fusion=bool(args.frontend_class_local_fusion),
        class_local_dim=int(args.frontend_class_local_dim),
    )
    codec_metadata = resolve_codec_metadata_from_cache_config(args.cache_root)
    target_layout = resolve_target_layout_from_cache_config(args.cache_root)
    target_pca_basis_path = resolve_target_pca_basis_path_from_cache_config(args.cache_root)
    target_dim = int(sample_batch["target_btd"].shape[-1])
    target_full_dim = int(sample_batch.get("target_full_dim", target_dim))
    timbre_bank_payload = _load_timbre_bank_payload(str(args.timbre_bank_path))
    if timbre_bank_payload is not None:
        bank_dim = int(torch.as_tensor(timbre_bank_payload["timbre_bank_latents"]).shape[-1])
        sample_batch = _attach_timbre_bank_to_batch(sample_batch, timbre_bank_payload)
        fixed_val_batch = _attach_timbre_bank_to_batch(fixed_val_batch, timbre_bank_payload)
    else:
        bank_dim = 0
    fixed_val_batch = apply_conditioning_ablation(
        fixed_val_batch,
        conditioning_ablation,
        batch_index=0,
    )
    if timbre_projection is not None and int(timbre_projection.shape[1]) != int(target_dim):
        probe_target_dim = timbre_probe_metadata.get("target_dim")
        probe_cache_root = timbre_probe_metadata.get("cache_root", "")
        raise ValueError(
            "timbre projection dimension does not match the training target dimension: "
            f"projection={tuple(timbre_projection.shape)} target_dim={int(target_dim)} "
            f"probe_target_dim={probe_target_dim!r} probe_cache_root={probe_cache_root!r}. "
            "Export a probe from the same cache root/target layout as this training run."
        )
    target_token_rate_hz = (
        float(args.positional_rate_hz)
        if float(args.positional_rate_hz) > 0.0
        else resolve_target_token_rate_hz(codec_metadata)
    )
    cfg = DiffusionTransformerConfig(
        x_dim=int(target_dim),
        frontend_cfg=frontend_cfg,
        concat_multiscale_frontend=not bool(args.no_concat_multiscale_frontend),
        positional_encoding=str(args.positional_encoding),
        positional_rate_hz=float(target_token_rate_hz),
        d_model=int(args.d_model),
        num_layers=int(args.num_layers),
        num_heads=int(args.num_heads),
        mlp_ratio=float(args.mlp_ratio),
        dropout=float(args.dropout),
        cond_dropout_prob=float(args.cond_dropout_prob),
        timbre_conditioning=timbre_bank_payload is not None,
        timbre_bank_dim=int(bank_dim),
        timbre_dropout_prob=float(args.timbre_dropout_prob),
        timbre_class_dropout_prob=float(args.timbre_class_dropout_prob),
    )
    model = ConditionalDiffusionTransformer(cfg).to(device)
    diffusion = GaussianDiffusion1D(num_steps=int(args.num_steps)).to(device)
    init_payload: dict[str, Any] | None = None
    checkpoint_to_load = str(resume_checkpoint_path) if bool(resume_requested) else str(args.init_checkpoint).strip()
    if checkpoint_to_load:
        init_payload = _load_init_checkpoint_payload(
            checkpoint_to_load,
            expected_cfg=cfg,
        )
        codec_metadata = resolve_codec_metadata_from_payload(init_payload, fallback=codec_metadata).to_dict()
        strict_load = not bool(init_payload.get("_allow_partial_model_state_dict", False))
        load_result = model.load_state_dict(dict(init_payload["model_state_dict"]), strict=bool(strict_load))
        if not bool(strict_load):
            print(
                "loaded base checkpoint with new timbre modules: "
                f"missing={list(load_result.missing_keys)} unexpected={list(load_result.unexpected_keys)}"
            )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    if bool(resume_requested):
        if init_payload is None or init_payload.get("optimizer_state_dict") is None:
            raise KeyError(f"resume checkpoint missing optimizer_state_dict: {checkpoint_to_load}")
        optimizer.load_state_dict(dict(init_payload["optimizer_state_dict"]))
        _optimizer_state_to_device(optimizer, device)

    target_pca_basis: dict[str, Any] | None = None
    if init_payload is not None and init_payload.get("target_pca_basis") is not None:
        target_pca_basis = load_target_pca_basis(
            init_payload["target_pca_basis"],
            device=device,
        )
    elif target_pca_basis_path is not None:
        target_pca_basis = load_target_pca_basis(target_pca_basis_path, device=device)
    if str(target_layout) == "framewise_pca" and target_pca_basis is None:
        raise FileNotFoundError(
            f"cache {cache_root_path} uses target_layout=framewise_pca but no PCA basis could be resolved"
        )

    audio_codec_model, _resolved_codec_device, codec_meta_obj = load_audio_codec_model(
        device=resolved_device,
        metadata=codec_metadata,
    )
    codec_metadata = codec_meta_obj.to_dict()
    quant_codebook_embed_ckd: torch.Tensor | None = None
    if float(args.quant_embed_mse_weight) > 0.0 or float(args.rvq_ce_weight) > 0.0:
        quant_codebook_embed_ckd = extract_codebook_embeddings(
            audio_codec_model,
            device=device,
        ).detach().contiguous()
    if init_payload is not None and init_payload.get("target_mean") is not None and init_payload.get("target_std") is not None:
        target_mean = torch.as_tensor(init_payload["target_mean"], dtype=torch.float32, device=device).view(-1).contiguous()
        target_std = (
            torch.as_tensor(init_payload["target_std"], dtype=torch.float32, device=device)
            .view(-1)
            .clamp_min(1.0e-6)
            .contiguous()
        )
    else:
        target_mean, target_std = load_or_compute_target_normalization(
            args.cache_root,
            train_loader,
            device=device,
            x_dim=int(cfg.x_dim),
        )
    sample_rate = int(
        (init_payload or {}).get("sample_rate")
        or resolve_encodec_sample_rate(audio_codec_model)
    )
    fixed_batch_size = int(fixed_val_batch["target_btd"].shape[0])
    if int(fixed_batch_size) <= 0:
        raise RuntimeError("validation preview batch is empty")
    preview_sample_idx = int(args.sample_idx)
    if not (0 <= int(preview_sample_idx) < int(fixed_batch_size)):
        clamped_sample_idx = min(max(int(preview_sample_idx), 0), int(fixed_batch_size) - 1)
        print(
            f"sample_idx={preview_sample_idx} out of range for validation batch size={fixed_batch_size}; "
            f"using sample_idx={clamped_sample_idx} for previews"
        )
        preview_sample_idx = int(clamped_sample_idx)

    checkpoint_metric_name = _resolve_checkpoint_metric_name(
        str(args.checkpoint_metric),
        audio_wave_l1_weight=float(args.audio_wave_l1_weight),
        audio_mrstft_weight=float(args.audio_mrstft_weight),
    )
    resume_start_epoch = 0
    if bool(resume_requested):
        if init_payload is None:
            raise RuntimeError("internal error: resume requested without a loaded checkpoint payload")
        if "epoch" not in init_payload:
            raise KeyError(f"resume checkpoint missing epoch: {checkpoint_to_load}")
        resume_start_epoch = int(init_payload["epoch"]) + 1
    history_rows: list[dict[str, Any]] = []
    preview_seed = int(args.seed) + 1_000_003
    preview_use_bpm_inference_geometry = bool(args.use_bpm_training_geometry)
    training_geometry_payload = {
        "use_bpm_training_geometry": bool(args.use_bpm_training_geometry),
        "bpm_geometry_num_beats": int(args.bpm_geometry_num_beats),
        "bpm_training_geometry_preserve_cached_frame_counts": bool(args.use_bpm_training_geometry),
        "preview_use_bpm_inference_geometry": bool(preview_use_bpm_inference_geometry),
    }

    run_config = {
        "cache_root": str(cache_root_path),
        "out_dir": str(out_dir),
        "init_checkpoint": "" if bool(resume_requested) else str((init_payload or {}).get("checkpoint_path") or ""),
        "resume": bool(resume_requested),
        "resume_checkpoint": str((init_payload or {}).get("checkpoint_path") or "") if bool(resume_requested) else "",
        "resume_start_epoch": int(resume_start_epoch),
        "train_split": str(args.train_split),
        "val_split": str(args.val_split),
        "resolved_device": resolved_device,
        "frontend_cfg": frontend_cfg,
        "model_cfg": asdict(cfg),
        "num_steps": int(args.num_steps),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "grad_accum_steps": int(args.grad_accum_steps),
        "train_batches_per_epoch": int(train_batches_per_epoch),
        "val_batches_per_epoch": int(val_batches_per_epoch),
        "num_workers": int(args.num_workers),
        "pin_memory": bool(pin_memory),
        "dataloader_multiprocessing_context": "" if dataloader_mp_context is None else str(dataloader_mp_context),
        "persistent_workers": bool(persistent_workers),
        "max_train_items": int(args.max_train_items),
        "max_val_items": int(args.max_val_items),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "conditioning_ablation": str(conditioning_ablation),
        "sample_idx": int(args.sample_idx),
        "preview_sample_idx": int(preview_sample_idx),
        "guidance_scale": float(args.guidance_scale),
        **training_geometry_payload,
        "eval_plot_steps": [int(x) for x in plot_steps],
        "x0_clip_norm": float(args.x0_clip_norm),
        "audio_wave_l1_weight": float(args.audio_wave_l1_weight),
        "audio_mrstft_weight": float(args.audio_mrstft_weight),
        "audio_mrstft_resolutions": [[int(n_fft), int(hop)] for n_fft, hop in audio_mrstft_resolutions],
        "timbre_probe_path": str(Path(timbre_probe_path).expanduser().resolve()) if timbre_probe_path else "",
        "timbre_bank_path": "" if timbre_bank_payload is None else str(timbre_bank_payload.get("_path", "")),
        "timbre_bank_summary": {} if timbre_bank_payload is None else dict(timbre_bank_payload.get("summary") or {}),
        "timbre_probe_metadata": timbre_probe_metadata,
        "timbre_projection_shape": (
            [int(x) for x in timbre_projection.shape] if timbre_projection is not None else []
        ),
        "x0_mse_weight": float(args.x0_mse_weight),
        "timbre_proj_mse_weight": float(args.timbre_proj_mse_weight),
        "quant_embed_mse_weight": float(args.quant_embed_mse_weight),
        "rvq_ce_weight": float(args.rvq_ce_weight),
        "quant_codebook_embedding_shape": (
            [int(x) for x in quant_codebook_embed_ckd.shape] if quant_codebook_embed_ckd is not None else []
        ),
        "onset_loss_weighting": bool(args.onset_loss_weighting),
        "onset_token_radius": int(args.onset_token_radius),
        "fixed_sample_epochs": [int(epoch) for epoch in fixed_sample_epochs],
        "checkpoint_metric_name": str(checkpoint_metric_name),
        "sample_rate": int(sample_rate),
        "codec_metadata": dict(codec_metadata),
        "target_layout": str(target_layout),
        "target_full_dim": int(target_full_dim),
        "target_token_rate_hz": float(target_token_rate_hz),
        "target_pca_basis_path": (
            str(target_pca_basis_path)
            if target_pca_basis_path is not None
            else ""
        ),
        "num_parameters": int(sum(int(param.numel()) for param in model.parameters())),
        "preview_seed": int(preview_seed),
    }
    write_json(out_dir / "run_config.json", run_config)
    write_json(out_dir / "config.json", run_config)
    print(f"resolved eval plot steps: {list(plot_steps)}")
    print(
        "training plan: "
        f"diffusion_num_steps={int(args.num_steps)} "
        f"start_epoch={int(resume_start_epoch)} "
        f"target_epochs={int(args.epochs)} "
        f"use_bpm_training_geometry={bool(args.use_bpm_training_geometry)} "
        f"train_batches_per_epoch={int(train_batches_per_epoch)} "
        f"val_batches_per_epoch={int(val_batches_per_epoch)}"
    )

    target0 = fixed_val_batch["target_btd"][int(preview_sample_idx) : int(preview_sample_idx) + 1]
    preview_target_frames = (
        _resolve_bpm_preview_target_frames(
            fixed_val_batch,
            int(preview_sample_idx),
            num_beats=int(args.bpm_geometry_num_beats),
            target_token_rate_hz=float(target_token_rate_hz),
        )
        if bool(preview_use_bpm_inference_geometry)
        else int(target0.shape[1])
    )
    preview_seed, fixed_noises, fixed_start_noise, fixed_step_noises = _build_fixed_preview_noises(
        seed=int(args.seed),
        target_shape=tuple(int(x) for x in target0.shape),
        preview_shape=(1, int(preview_target_frames), int(target0.shape[-1])),
        dtype=target0.dtype,
        plot_steps=plot_steps,
        num_steps=int(args.num_steps),
    )

    def _export_fixed_preview(epoch_value: int, *, samples_dir: Path, plots_dir: Path) -> Path:
        wav_path_local = save_inference_wav(
            model=model,
            diffusion=diffusion,
            encodec_model=audio_codec_model,
            batch=fixed_val_batch,
            device=device,
            epoch=int(epoch_value),
            target_mean=target_mean,
            target_std=target_std,
            sample_rate=sample_rate,
            out_dir=samples_dir,
            sample_idx=int(preview_sample_idx),
            guidance_scale=float(args.guidance_scale),
            start_noise=fixed_start_noise,
            step_noises=fixed_step_noises,
            x0_clip_norm=float(args.x0_clip_norm),
            target_pca_basis=target_pca_basis,
            use_bpm_inference_geometry=bool(preview_use_bpm_inference_geometry),
            inference_num_beats=int(args.bpm_geometry_num_beats),
            target_token_rate_hz=float(target_token_rate_hz),
        )
        save_eval_plot_multi_t(
            model=model,
            diffusion=diffusion,
            batch=fixed_val_batch,
            device=device,
            epoch=int(epoch_value),
            out_dir=plots_dir,
            sample_idx=int(preview_sample_idx),
            t_values=plot_steps,
            fixed_noises=fixed_noises,
            target_mean=target_mean,
            target_std=target_std,
            x0_clip_norm=float(args.x0_clip_norm),
            use_bpm_training_geometry=bool(args.use_bpm_training_geometry),
            bpm_geometry_num_beats=int(args.bpm_geometry_num_beats),
        )
        return Path(wav_path_local)

    best_val_loss = float("inf")
    best_checkpoint_metric = float("inf")
    best_checkpoint_epoch = -1
    global_step = 0
    if bool(resume_requested):
        assert init_payload is not None
        history_rows = [
            row
            for row in _read_history_jsonl(history_path)
            if int(row.get("epoch", -1)) < int(resume_start_epoch)
        ]
        write_jsonl(history_path, history_rows)
        _write_history_csv(history_csv_path, history_rows)
        best_val_loss = float(init_payload.get("best_val_loss", best_val_loss))
        best_checkpoint_metric = float(
            init_payload.get(
                "best_checkpoint_metric_value",
                init_payload.get("best_checkpoint_metric", best_checkpoint_metric),
            )
        )
        if not torch.isfinite(torch.tensor(best_checkpoint_metric)):
            best_checkpoint_metric = float(best_val_loss)
        best_checkpoint_epoch = int(
            init_payload.get(
                "best_checkpoint_epoch",
                init_payload.get("epoch", -1),
            )
        )
        if "global_step" in init_payload:
            global_step = int(init_payload["global_step"])
        elif history_rows:
            global_step = int(sum(int(row.get("train_steps", train_batches_per_epoch)) for row in history_rows))
        else:
            global_step = int(resume_start_epoch) * int(train_batches_per_epoch)
        print(
            "resumed checkpoint: "
            f"path={checkpoint_to_load} "
            f"next_epoch={int(resume_start_epoch)} "
            f"global_step={int(global_step)} "
            f"best_{checkpoint_metric_name}={float(best_checkpoint_metric):.6f}"
        )
    if int(resume_start_epoch) >= int(args.epochs):
        print(
            f"checkpoint already reached epoch {int(resume_start_epoch) - 1}; "
            f"--epochs={int(args.epochs)} leaves nothing to train"
        )
        return
    for epoch in range(int(resume_start_epoch), int(args.epochs)):
        model.train()
        train_running_loss = 0.0
        train_running_diffusion_loss = 0.0
        train_running_audio_wave_l1 = 0.0
        train_running_audio_mrstft = 0.0
        train_running_x0 = 0.0
        train_running_x0_loss = 0.0
        train_running_timbre_proj_mse = 0.0
        train_running_quant_embed_mse = 0.0
        train_running_rvq_ce = 0.0
        train_running_onset_weighted_x0 = 0.0
        train_n_batches = 0
        train_valid_tokens_seen = 0
        train_audio_seconds_seen = 0.0
        train_started_at = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(
            _progress(train_loader, desc=f"train {epoch:03d} batches", total=int(train_batches_per_epoch))
        ):
            batch = _attach_timbre_bank_to_batch(batch, timbre_bank_payload)
            batch = apply_conditioning_ablation(
                batch,
                conditioning_ablation,
                batch_index=int(batch_index),
            )
            out = diffusion_train_step(
                model,
                diffusion,
                batch,
                device,
                target_mean=target_mean,
                target_std=target_std,
                encodec_model=audio_codec_model,
                audio_sample_rate=sample_rate,
                audio_wave_l1_weight=float(args.audio_wave_l1_weight),
                audio_mrstft_weight=float(args.audio_mrstft_weight),
                audio_mrstft_resolutions=audio_mrstft_resolutions,
                x0_clip_norm=float(args.x0_clip_norm) if args.x0_clip_norm is not None else None,
                timbre_projection=timbre_projection,
                x0_mse_weight=float(args.x0_mse_weight),
                timbre_proj_mse_weight=float(args.timbre_proj_mse_weight),
                quant_embed_mse_weight=float(args.quant_embed_mse_weight),
                rvq_ce_weight=float(args.rvq_ce_weight),
                quant_codebook_embed_ckd=quant_codebook_embed_ckd,
                onset_loss_weighting=bool(args.onset_loss_weighting),
                onset_token_radius=int(args.onset_token_radius),
                target_pca_basis=target_pca_basis,
                use_bpm_training_geometry=bool(args.use_bpm_training_geometry),
                bpm_geometry_num_beats=int(args.bpm_geometry_num_beats),
            )
            loss = torch.as_tensor(out["loss"])
            (loss / float(args.grad_accum_steps)).backward()

            train_running_loss += float(loss.item())
            train_running_diffusion_loss += float(torch.as_tensor(out["diffusion_loss"]).item())
            train_running_audio_wave_l1 += float(torch.as_tensor(out["audio_wave_l1"]).item())
            train_running_audio_mrstft += float(torch.as_tensor(out["audio_mrstft"]).item())
            train_running_x0 += float(torch.as_tensor(out["x0_mse_median"]).item())
            train_running_x0_loss += float(torch.as_tensor(out["x0_loss"]).item())
            train_running_timbre_proj_mse += float(torch.as_tensor(out["timbre_proj_mse"]).item())
            train_running_quant_embed_mse += float(torch.as_tensor(out["quant_embed_mse"]).item())
            train_running_rvq_ce += float(torch.as_tensor(out["rvq_ce"]).item())
            train_running_onset_weighted_x0 += float(torch.as_tensor(out["onset_weighted_x0"]).item())
            train_n_batches += 1
            train_valid_tokens_seen += int(torch.as_tensor(batch["target_valid_mask_bt"], dtype=torch.bool).sum().item())
            train_audio_seconds_seen += float(torch.as_tensor(batch["duration_sec"], dtype=torch.float32).sum().item())
            should_step = (int(train_n_batches) % int(args.grad_accum_steps) == 0) or (
                int(train_n_batches) == int(train_batches_per_epoch)
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        if train_n_batches <= 0:
            raise RuntimeError("train loader produced no batches")

        train_elapsed_sec = float(time.perf_counter() - train_started_at)
        train_loss = float(train_running_loss / float(train_n_batches))
        train_diffusion_loss = float(train_running_diffusion_loss / float(train_n_batches))
        train_audio_wave_l1 = float(train_running_audio_wave_l1 / float(train_n_batches))
        train_audio_mrstft = float(train_running_audio_mrstft / float(train_n_batches))
        train_x0 = float(train_running_x0 / float(train_n_batches))
        train_x0_loss = float(train_running_x0_loss / float(train_n_batches))
        train_timbre_proj_mse = float(train_running_timbre_proj_mse / float(train_n_batches))
        train_quant_embed_mse = float(train_running_quant_embed_mse / float(train_n_batches))
        train_rvq_ce = float(train_running_rvq_ce / float(train_n_batches))
        train_onset_weighted_x0 = float(train_running_onset_weighted_x0 / float(train_n_batches))
        train_steps_per_sec = float(train_n_batches) / max(float(train_elapsed_sec), 1.0e-8)
        train_tokens_per_sec = float(train_valid_tokens_seen) / max(float(train_elapsed_sec), 1.0e-8)
        train_audio_seconds_per_sec = float(train_audio_seconds_seen) / max(float(train_elapsed_sec), 1.0e-8)
        train_peak_gpu_mem_allocated_mb = (
            float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2))
            if device.type == "cuda"
            else None
        )
        train_peak_gpu_mem_reserved_mb = (
            float(torch.cuda.max_memory_reserved(device) / (1024.0 ** 2))
            if device.type == "cuda"
            else None
        )

        model.eval()
        val_running_loss = 0.0
        val_running_diffusion_loss = 0.0
        val_running_audio_wave_l1 = 0.0
        val_running_audio_mrstft = 0.0
        val_running_x0 = 0.0
        val_running_x0_loss = 0.0
        val_running_timbre_proj_mse = 0.0
        val_running_quant_embed_mse = 0.0
        val_running_rvq_ce = 0.0
        val_running_onset_weighted_x0 = 0.0
        val_n_batches = 0
        val_valid_tokens_seen = 0
        val_audio_seconds_seen = 0.0
        val_started_at = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        with torch.no_grad():
            for batch_index, batch in enumerate(_progress(val_loader, desc=f"val   {epoch:03d} batches")):
                batch = _attach_timbre_bank_to_batch(batch, timbre_bank_payload)
                batch = apply_conditioning_ablation(
                    batch,
                    conditioning_ablation,
                    batch_index=int(batch_index),
                )
                out = diffusion_train_step(
                    model,
                    diffusion,
                    batch,
                    device,
                    target_mean=target_mean,
                    target_std=target_std,
                    encodec_model=audio_codec_model,
                    audio_sample_rate=sample_rate,
                    audio_wave_l1_weight=float(args.audio_wave_l1_weight),
                    audio_mrstft_weight=float(args.audio_mrstft_weight),
                    audio_mrstft_resolutions=audio_mrstft_resolutions,
                    x0_clip_norm=float(args.x0_clip_norm) if args.x0_clip_norm is not None else None,
                    timbre_projection=timbre_projection,
                    x0_mse_weight=float(args.x0_mse_weight),
                    timbre_proj_mse_weight=float(args.timbre_proj_mse_weight),
                    quant_embed_mse_weight=float(args.quant_embed_mse_weight),
                    rvq_ce_weight=float(args.rvq_ce_weight),
                    quant_codebook_embed_ckd=quant_codebook_embed_ckd,
                    onset_loss_weighting=bool(args.onset_loss_weighting),
                    onset_token_radius=int(args.onset_token_radius),
                    target_pca_basis=target_pca_basis,
                    use_bpm_training_geometry=bool(args.use_bpm_training_geometry),
                    bpm_geometry_num_beats=int(args.bpm_geometry_num_beats),
                )
                val_running_loss += float(torch.as_tensor(out["loss"]).item())
                val_running_diffusion_loss += float(torch.as_tensor(out["diffusion_loss"]).item())
                val_running_audio_wave_l1 += float(torch.as_tensor(out["audio_wave_l1"]).item())
                val_running_audio_mrstft += float(torch.as_tensor(out["audio_mrstft"]).item())
                val_running_x0 += float(torch.as_tensor(out["x0_mse_median"]).item())
                val_running_x0_loss += float(torch.as_tensor(out["x0_loss"]).item())
                val_running_timbre_proj_mse += float(torch.as_tensor(out["timbre_proj_mse"]).item())
                val_running_quant_embed_mse += float(torch.as_tensor(out["quant_embed_mse"]).item())
                val_running_rvq_ce += float(torch.as_tensor(out["rvq_ce"]).item())
                val_running_onset_weighted_x0 += float(torch.as_tensor(out["onset_weighted_x0"]).item())
                val_n_batches += 1
                val_valid_tokens_seen += int(torch.as_tensor(batch["target_valid_mask_bt"], dtype=torch.bool).sum().item())
                val_audio_seconds_seen += float(torch.as_tensor(batch["duration_sec"], dtype=torch.float32).sum().item())

        if val_n_batches <= 0:
            raise RuntimeError("validation loader produced no batches")

        val_elapsed_sec = float(time.perf_counter() - val_started_at)
        val_loss = float(val_running_loss / float(val_n_batches))
        val_diffusion_loss = float(val_running_diffusion_loss / float(val_n_batches))
        val_audio_wave_l1 = float(val_running_audio_wave_l1 / float(val_n_batches))
        val_audio_mrstft = float(val_running_audio_mrstft / float(val_n_batches))
        val_x0 = float(val_running_x0 / float(val_n_batches))
        val_x0_loss = float(val_running_x0_loss / float(val_n_batches))
        val_timbre_proj_mse = float(val_running_timbre_proj_mse / float(val_n_batches))
        val_quant_embed_mse = float(val_running_quant_embed_mse / float(val_n_batches))
        val_rvq_ce = float(val_running_rvq_ce / float(val_n_batches))
        val_onset_weighted_x0 = float(val_running_onset_weighted_x0 / float(val_n_batches))
        val_steps_per_sec = float(val_n_batches) / max(float(val_elapsed_sec), 1.0e-8)
        val_tokens_per_sec = float(val_valid_tokens_seen) / max(float(val_elapsed_sec), 1.0e-8)
        val_audio_seconds_per_sec = float(val_audio_seconds_seen) / max(float(val_elapsed_sec), 1.0e-8)
        val_peak_gpu_mem_allocated_mb = (
            float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2))
            if device.type == "cuda"
            else None
        )
        val_peak_gpu_mem_reserved_mb = (
            float(torch.cuda.max_memory_reserved(device) / (1024.0 ** 2))
            if device.type == "cuda"
            else None
        )
        best_val_loss = min(float(best_val_loss), float(val_loss))

        metric_values = {
            "val_loss": float(val_loss),
            "val_diffusion_loss": float(val_diffusion_loss),
            "val_audio_wave_l1": float(val_audio_wave_l1),
            "val_audio_mrstft": float(val_audio_mrstft),
            "val_x0": float(val_x0),
            "val_x0_loss": float(val_x0_loss),
            "val_timbre_proj_mse": float(val_timbre_proj_mse),
            "val_quant_embed_mse": float(val_quant_embed_mse),
            "val_rvq_ce": float(val_rvq_ce),
            "val_onset_weighted_x0": float(val_onset_weighted_x0),
        }
        checkpoint_metric_value = float(metric_values[str(checkpoint_metric_name)])
        checkpoint_improved = False
        checkpoint_improved = float(checkpoint_metric_value) < float(best_checkpoint_metric)
        if checkpoint_improved:
            best_checkpoint_metric = float(checkpoint_metric_value)
            best_checkpoint_epoch = int(epoch)

        epoch_row = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "train_diffusion_loss": float(train_diffusion_loss),
            "train_audio_wave_l1": float(train_audio_wave_l1),
            "train_audio_mrstft": float(train_audio_mrstft),
            "train_x0": float(train_x0),
            "train_x0_loss": float(train_x0_loss),
            "train_timbre_proj_mse": float(train_timbre_proj_mse),
            "train_quant_embed_mse": float(train_quant_embed_mse),
            "train_rvq_ce": float(train_rvq_ce),
            "train_onset_weighted_x0": float(train_onset_weighted_x0),
            "train_steps": int(train_n_batches),
            "train_batches_per_epoch": int(train_batches_per_epoch),
            "train_valid_tokens_seen": int(train_valid_tokens_seen),
            "train_audio_seconds_seen": float(train_audio_seconds_seen),
            "train_elapsed_sec": float(train_elapsed_sec),
            "train_steps_per_sec": float(train_steps_per_sec),
            "train_tokens_per_sec": float(train_tokens_per_sec),
            "train_audio_seconds_per_sec": float(train_audio_seconds_per_sec),
            "train_peak_gpu_mem_allocated_mb": train_peak_gpu_mem_allocated_mb,
            "train_peak_gpu_mem_reserved_mb": train_peak_gpu_mem_reserved_mb,
            "val_loss": float(val_loss),
            "val_diffusion_loss": float(val_diffusion_loss),
            "val_audio_wave_l1": float(val_audio_wave_l1),
            "val_audio_mrstft": float(val_audio_mrstft),
            "val_x0": float(val_x0),
            "val_x0_loss": float(val_x0_loss),
            "val_timbre_proj_mse": float(val_timbre_proj_mse),
            "val_quant_embed_mse": float(val_quant_embed_mse),
            "val_rvq_ce": float(val_rvq_ce),
            "val_onset_weighted_x0": float(val_onset_weighted_x0),
            "val_steps": int(val_n_batches),
            "val_valid_tokens_seen": int(val_valid_tokens_seen),
            "val_audio_seconds_seen": float(val_audio_seconds_seen),
            "val_elapsed_sec": float(val_elapsed_sec),
            "val_steps_per_sec": float(val_steps_per_sec),
            "val_tokens_per_sec": float(val_tokens_per_sec),
            "val_audio_seconds_per_sec": float(val_audio_seconds_per_sec),
            "val_peak_gpu_mem_allocated_mb": val_peak_gpu_mem_allocated_mb,
            "val_peak_gpu_mem_reserved_mb": val_peak_gpu_mem_reserved_mb,
            "best_val_loss": float(best_val_loss),
            "checkpoint_metric_name": str(checkpoint_metric_name),
            "checkpoint_metric_value": float(checkpoint_metric_value),
            "best_checkpoint_metric_value": float(best_checkpoint_metric),
            "best_checkpoint_epoch": int(best_checkpoint_epoch),
            "global_step": int(global_step),
            "checkpoint_improved": bool(checkpoint_improved),
        }
        history_rows.append(epoch_row)
        append_jsonl(history_path, epoch_row)
        _write_history_csv(history_csv_path, history_rows)

        _save_diffusion_checkpoint(
            out_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            cfg=cfg,
            frontend_cfg=frontend_cfg,
            target_mean=target_mean,
            target_std=target_std,
            sample_rate=sample_rate,
            codec_metadata=codec_metadata,
            num_steps=int(args.num_steps),
            epoch=epoch,
            best_val_loss=float(best_val_loss),
            extra_payload={
                **epoch_row,
                **training_geometry_payload,
                "conditioning_ablation": str(conditioning_ablation),
                "best_checkpoint_metric_name": str(checkpoint_metric_name),
                "best_checkpoint_metric_value": float(best_checkpoint_metric),
                "best_checkpoint_epoch": int(best_checkpoint_epoch),
                "global_step": int(global_step),
                "init_checkpoint": "" if bool(resume_requested) else str((init_payload or {}).get("checkpoint_path") or ""),
                "resume_checkpoint": str(checkpoint_to_load) if bool(resume_requested) else "",
                "target_layout": str(target_layout),
                "target_full_dim": int(target_full_dim),
                "target_pca_basis_path": (
                    str(target_pca_basis_path)
                    if target_pca_basis_path is not None
                    else ""
                ),
                "target_pca_basis": None
                if target_pca_basis is None
                else {
                    **target_pca_basis,
                    "mean": torch.as_tensor(target_pca_basis["mean"], dtype=torch.float32).detach().cpu(),
                    "components": torch.as_tensor(target_pca_basis["components"], dtype=torch.float32).detach().cpu(),
                },
            },
        )

        if checkpoint_improved:
            _save_diffusion_checkpoint(
                out_dir / "best_diffusion.pt",
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                frontend_cfg=frontend_cfg,
                target_mean=target_mean,
                target_std=target_std,
                sample_rate=sample_rate,
                codec_metadata=codec_metadata,
                num_steps=int(args.num_steps),
                epoch=epoch,
                best_val_loss=float(best_val_loss),
                extra_payload={
                    **epoch_row,
                    **training_geometry_payload,
                    "conditioning_ablation": str(conditioning_ablation),
                    "best_checkpoint_metric_name": str(checkpoint_metric_name),
                    "best_checkpoint_metric_value": float(best_checkpoint_metric),
                    "best_checkpoint_epoch": int(best_checkpoint_epoch),
                    "global_step": int(global_step),
                    "init_checkpoint": "" if bool(resume_requested) else str((init_payload or {}).get("checkpoint_path") or ""),
                    "resume_checkpoint": str(checkpoint_to_load) if bool(resume_requested) else "",
                    "target_layout": str(target_layout),
                    "target_full_dim": int(target_full_dim),
                    "target_pca_basis_path": (
                        str(target_pca_basis_path)
                        if target_pca_basis_path is not None
                        else ""
                    ),
                    "target_pca_basis": None
                    if target_pca_basis is None
                    else {
                        **target_pca_basis,
                        "mean": torch.as_tensor(target_pca_basis["mean"], dtype=torch.float32).detach().cpu(),
                        "components": torch.as_tensor(target_pca_basis["components"], dtype=torch.float32).detach().cpu(),
                    },
                },
            )
            wav_path = _export_fixed_preview(
                int(epoch),
                samples_dir=out_dir / "best_samples",
                plots_dir=out_dir / "eval_plots",
            )
            print(f"saved best checkpoint and wav: {wav_path}")

        if int(epoch) in set(int(x) for x in fixed_sample_epochs):
            fixed_wav_path = _export_fixed_preview(
                int(epoch),
                samples_dir=out_dir / "fixed_samples",
                plots_dir=out_dir / "fixed_eval_plots",
            )
            print(f"saved fixed epoch preview: {fixed_wav_path}")

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.6f} train_diff={train_diffusion_loss:.6f} "
            f"train_audio_l1={train_audio_wave_l1:.6f} train_mrstft={train_audio_mrstft:.6f} "
            f"train_x0={train_x0:.6f} train_x0_loss={train_x0_loss:.6f} "
            f"train_timbre={train_timbre_proj_mse:.6f} train_quant={train_quant_embed_mse:.6f} "
            f"train_rvq_ce={train_rvq_ce:.6f} "
            f"train_onset_x0={train_onset_weighted_x0:.6f} "
            f"val_loss={val_loss:.6f} val_diff={val_diffusion_loss:.6f} "
            f"val_audio_l1={val_audio_wave_l1:.6f} val_mrstft={val_audio_mrstft:.6f} "
            f"val_x0={val_x0:.6f} val_x0_loss={val_x0_loss:.6f} "
            f"val_timbre={val_timbre_proj_mse:.6f} val_quant={val_quant_embed_mse:.6f} "
            f"val_rvq_ce={val_rvq_ce:.6f} "
            f"val_onset_x0={val_onset_weighted_x0:.6f} "
            f"{checkpoint_metric_name}={checkpoint_metric_value:.6f} "
            f"best_{checkpoint_metric_name}={best_checkpoint_metric:.6f} "
            f"best_val_loss={best_val_loss:.6f}"
        )


if __name__ == "__main__":
    main()
