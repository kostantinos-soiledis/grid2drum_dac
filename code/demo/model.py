import math
import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from data.encodec_utils import (
    decode_codes_to_audio_b1t,
    decode_quantized_latent_to_audio,
    load_target_pca_basis,
    reconstruct_latent_from_pca,
    requantize_latent_to_codes_bct,
    resolve_audio_codec_sample_rate,
    rvq_sum_latents,
    token_ids_to_codebook_embeddings,
)
from data.diffusion_dataset import estimate_target_normalization
from data.seconds_frontend import build_seconds_frontend_from_cfg


# Derived from scripts/analyze_frontend_radii.py on 60k train frames of 4beats_v9.
DEFAULT_FRONTEND_RADII: tuple[int, ...] = (0, 22, 41, 55)
DEFAULT_FRONTEND_PRIMARY_RADIUS = 22
DEFAULT_FRONTEND_VARIANT = "hybrid"
DEFAULT_FRONTEND_EMBED_DIM = 64
DEFAULT_FRONTEND_OUTPUT_KIND = "feat"
DEFAULT_FRONTEND_PADDING_MODE = "reflect"
DEFAULT_FRONTEND_STEP_SECONDS = 0.0
DEFAULT_FRONTEND_CHUNK_SIZE = 0
DEFAULT_FRONTEND_CLASS_LOCAL_DIM = 8
DEFAULT_FRONTEND_CONCAT_MULTISCALE = True
DEFAULT_SAMPLE_X0_CLIP_NORM = 6.0
DEFAULT_AUDIO_WAVE_L1_WEIGHT = 0.0
DEFAULT_AUDIO_MRSTFT_WEIGHT = 0.0
DEFAULT_AUDIO_MRSTFT_RESOLUTIONS: tuple[tuple[int, int], ...] = (
    (512, 128),
    (1024, 256),
    (2048, 512),
)
DEFAULT_INFERENCE_NUM_BEATS = 4
DEFAULT_BEAT_CROSSFADE_MS = 10.0
DEFAULT_TARGET_TOKEN_RATE_HZ = 50.0
DEFAULT_POSITIONAL_ENCODING = "seconds"



def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim=None, eps: float = 1e-8):
    mask_f = mask.float()
    if dim is None:
        return (x * mask_f).sum() / mask_f.sum().clamp_min(eps)
    return (x * mask_f).sum(dim=dim) / mask_f.sum(dim=dim).clamp_min(eps)


def apply_seq_mask(x: torch.Tensor, valid_mask_bt: torch.Tensor) -> torch.Tensor:
    return x * valid_mask_bt.unsqueeze(-1).to(x.dtype)


def _normalize_stats_vector(
    value,
    *,
    x_dim: int,
    device: torch.device,
    default_fill: float,
    name: str,
):
    if value is None:
        return torch.full((x_dim,), float(default_fill), dtype=torch.float32, device=device)
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device).view(-1)
    if int(tensor.numel()) != int(x_dim):
        raise ValueError(f"{name} must have {x_dim} values, got {tuple(tensor.shape)}")
    return tensor.contiguous()


def normalize_latent(x, mean, std):
    resolved_mean = _normalize_stats_vector(
        mean,
        x_dim=int(x.shape[-1]),
        device=x.device,
        default_fill=0.0,
        name="target_mean",
    )
    resolved_std = _normalize_stats_vector(
        std,
        x_dim=int(x.shape[-1]),
        device=x.device,
        default_fill=1.0,
        name="target_std",
    ).clamp_min(1.0e-8)
    return (x - resolved_mean.view(1, 1, -1)) / resolved_std.view(1, 1, -1)


def denormalize_latent(x, mean, std):
    resolved_mean = _normalize_stats_vector(
        mean,
        x_dim=int(x.shape[-1]),
        device=x.device,
        default_fill=0.0,
        name="target_mean",
    )
    resolved_std = _normalize_stats_vector(
        std,
        x_dim=int(x.shape[-1]),
        device=x.device,
        default_fill=1.0,
        name="target_std",
    ).clamp_min(1.0e-8)
    return x * resolved_std.view(1, 1, -1) + resolved_mean.view(1, 1, -1)


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def sinusoidal_positions(length: int, dim: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    half = dim // 2
    div_term = torch.exp(
        torch.arange(0, half, device=device, dtype=torch.float32) * (-math.log(10000.0) / half)
    )
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0:half] = torch.sin(position * div_term)
    pe[:, half:2 * half] = torch.cos(position * div_term)
    if dim % 2:
        pe[:, -1] = 0
    return pe.unsqueeze(0)


def sinusoidal_time_positions(
    times_sec_bt: torch.Tensor,
    dim: int,
    *,
    rate_hz: float,
) -> torch.Tensor:
    times = torch.as_tensor(times_sec_bt, dtype=torch.float32)
    if int(times.dim()) != 2:
        raise ValueError(f"times_sec_bt must be [B,T], got {tuple(times.shape)}")
    position = times.unsqueeze(-1) * float(max(1.0e-6, float(rate_hz)))
    half = dim // 2
    div_term = torch.exp(
        torch.arange(0, half, device=times.device, dtype=torch.float32) * (-math.log(10000.0) / half)
    ).view(1, 1, -1)
    pe = torch.zeros(
        int(times.shape[0]),
        int(times.shape[1]),
        int(dim),
        device=times.device,
        dtype=torch.float32,
    )
    pe[:, :, 0:half] = torch.sin(position * div_term)
    pe[:, :, half:2 * half] = torch.cos(position * div_term)
    if dim % 2:
        pe[:, :, -1] = 0
    return pe


def build_frontend_cfg_from_batch(
    batch: Mapping[str, Any],
    *,
    variant: str = DEFAULT_FRONTEND_VARIANT,
    embed_dim: int = DEFAULT_FRONTEND_EMBED_DIM,
    output_kind: str = DEFAULT_FRONTEND_OUTPUT_KIND,
    radii: Sequence[int] = DEFAULT_FRONTEND_RADII,
    primary_radius: int = DEFAULT_FRONTEND_PRIMARY_RADIUS,
    padding_mode: str = DEFAULT_FRONTEND_PADDING_MODE,
    step_seconds: float = DEFAULT_FRONTEND_STEP_SECONDS,
    chunk_size: int = DEFAULT_FRONTEND_CHUNK_SIZE,
    class_local_fusion: bool = False,
    class_local_dim: int = DEFAULT_FRONTEND_CLASS_LOCAL_DIM,
) -> dict[str, Any]:
    radii_eff = [int(x) for x in list(radii or ()) if int(x) >= 0]
    if not radii_eff:
        radii_eff = [int(primary_radius)]
    grid = torch.as_tensor(batch["grid"])
    return {
        "input_dim_source": int(grid.shape[1]),
        "class_id_vocab_sizes": [int(x) for x in list(batch.get("class_id_vocab_sizes") or [])],
        "source_feature_names": [str(x) for x in list(batch.get("feature_row_names") or [])],
        "class_names": [str(x) for x in list(batch.get("class_names") or [])],
        "variant": str(variant),
        "embed_dim": int(embed_dim),
        "output_kind": str(output_kind),
        "multiscale_enabled": bool(len(radii_eff) > 1),
        "multiscale_radii": [int(x) for x in list(radii_eff)],
        "primary_radius": int(primary_radius),
        "window_radius": int(primary_radius),
        "padding_mode": str(padding_mode),
        "step_seconds": float(step_seconds),
        "chunk_size": int(chunk_size),
        "class_local_fusion": bool(class_local_fusion),
        "class_local_dim": int(class_local_dim),
    }


def _prepare_batch_tensors(
    batch: Mapping[str, Any],
    device: torch.device,
    *,
    require_target: bool = True,
    require_timing: bool = True,
) -> dict[str, torch.Tensor | None]:
    def _tensor(key: str, dtype: torch.dtype, *, required: bool = True) -> torch.Tensor | None:
        value = batch.get(key)
        if value is None:
            if bool(required):
                raise KeyError(f"batch is missing required key: {key}")
            return None
        return torch.as_tensor(value, device=device, dtype=dtype).contiguous()

    return {
        "grid": _tensor("grid", torch.float32),
        "grid_ids": _tensor("grid_ids", torch.long, required=False),
        "family_onsets_bft": _tensor("family_onsets_bft", torch.bool, required=False),
        "grid_valid_mask": _tensor("grid_valid_mask", torch.bool),
        "grid_times_sec": _tensor("grid_times_sec", torch.float32, required=require_timing),
        "token_times_sec": _tensor("token_times_sec", torch.float32, required=require_timing),
        "beat_boundaries_sec": _tensor("beat_boundaries_sec", torch.float32, required=False),
        "beat_boundaries_valid_mask": _tensor("beat_boundaries_valid_mask", torch.bool, required=False),
        "bpm": _tensor("bpm", torch.float32, required=False),
        "duration_sec": _tensor("duration_sec", torch.float32, required=False),
        "target_btd": _tensor("target_btd", torch.float32, required=require_target),
        "target_sum_btd": _tensor("target_sum_btd", torch.float32, required=False),
        "target_valid_mask_bt": _tensor("target_valid_mask_bt", torch.bool, required=require_target),
        "source_codes_bct": _tensor("source_codes_bct", torch.long, required=False),
        "x0_prior_btd": _tensor("x0_prior_btd", torch.float32, required=False),
    }


def _slice_prepared_batch(
    prepared: Mapping[str, torch.Tensor | None],
    sample_idx: int,
) -> dict[str, torch.Tensor | None]:
    return {
        key: (
            None
            if value is None
            else value[int(sample_idx) : int(sample_idx) + 1].contiguous()
        )
        for key, value in prepared.items()
    }


def _prepare_geometry_tensors(
    geometry: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        str(key): torch.as_tensor(value, device=device).contiguous()
        for key, value in geometry.items()
    }


def _slice_inference_geometry(
    geometry: Mapping[str, torch.Tensor],
    sample_idx: int,
) -> dict[str, torch.Tensor]:
    sliced: dict[str, torch.Tensor] = {}
    for key, value in geometry.items():
        tensor = torch.as_tensor(value)
        if int(tensor.dim()) > 0:
            if int(tensor.shape[0]) <= int(sample_idx):
                raise IndexError(f"sample_idx={sample_idx} out of range for inference geometry key {key!r}")
            sliced[str(key)] = tensor[int(sample_idx) : int(sample_idx) + 1].contiguous()
        else:
            sliced[str(key)] = tensor.contiguous()
    return sliced


def lengths_to_mask(lengths_b: torch.Tensor, *, max_len: int | None = None) -> torch.Tensor:
    lengths = torch.as_tensor(lengths_b, dtype=torch.long).view(-1)
    if int(lengths.numel()) <= 0:
        resolved_max_len = int(max_len or 0)
        return torch.zeros((0, max(0, resolved_max_len)), dtype=torch.bool, device=lengths.device)
    resolved_max_len = int(max_len) if max_len is not None else int(lengths.max().item())
    if int(resolved_max_len) <= 0:
        return torch.zeros((int(lengths.shape[0]), 0), dtype=torch.bool, device=lengths.device)
    steps = torch.arange(int(resolved_max_len), device=lengths.device, dtype=torch.long).view(1, -1)
    return (steps < lengths.view(-1, 1)).contiguous()


def uniform_frame_times_from_durations(
    frame_counts_b: torch.Tensor,
    duration_sec_b: torch.Tensor,
    *,
    max_num_frames: int | None = None,
) -> torch.Tensor:
    frame_counts = torch.as_tensor(frame_counts_b, dtype=torch.long).view(-1)
    duration_sec = torch.as_tensor(duration_sec_b, dtype=torch.float32, device=frame_counts.device).view(-1)
    if tuple(frame_counts.shape) != tuple(duration_sec.shape):
        raise ValueError(
            f"frame_counts_b and duration_sec_b must match, got {tuple(frame_counts.shape)} / {tuple(duration_sec.shape)}"
        )
    resolved_max_frames = int(max_num_frames) if max_num_frames is not None else int(frame_counts.max().item())
    if int(resolved_max_frames) <= 0:
        return torch.zeros((int(frame_counts.shape[0]), 0), dtype=torch.float32, device=frame_counts.device)
    frame_counts_safe = frame_counts.clamp_min(1).to(dtype=torch.float32).view(-1, 1)
    frame_steps = torch.arange(int(resolved_max_frames), device=frame_counts.device, dtype=torch.float32).view(1, -1)
    centers = ((frame_steps + 0.5) / frame_counts_safe) * duration_sec.view(-1, 1)
    valid_mask_bt = lengths_to_mask(frame_counts, max_len=int(resolved_max_frames))
    return (centers * valid_mask_bt.to(dtype=centers.dtype)).contiguous()


def _metadata_get(codec_metadata: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
    if isinstance(codec_metadata, Mapping):
        return codec_metadata.get(key, default)
    return getattr(codec_metadata, key, default)


def _metadata_positive_float(codec_metadata: Mapping[str, Any] | Any, key: str) -> float | None:
    try:
        value = float(_metadata_get(codec_metadata, key))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value <= 0.0:
        return None
    return float(value)


def _metadata_positive_int(codec_metadata: Mapping[str, Any] | Any, key: str) -> int | None:
    try:
        value = int(_metadata_get(codec_metadata, key))
    except (TypeError, ValueError):
        return None
    if int(value) <= 0:
        return None
    return int(value)


def _legacy_dac_hop_length(codec_metadata: Mapping[str, Any] | Any) -> int | None:
    codec_family = str(_metadata_get(codec_metadata, "codec_family", "") or "").strip().lower()
    codec_model_id = str(_metadata_get(codec_metadata, "codec_model_id", "") or "").strip().lower()
    sample_rate = _metadata_positive_int(codec_metadata, "codec_sample_rate")
    if (
        codec_model_id == "descript/dac_44khz"
        and codec_family in {"", "dac"}
        and int(sample_rate or 0) == 44100
    ):
        return 512
    return None


def resolve_codec_hop_length(codec_metadata: Mapping[str, Any] | Any | None) -> int | None:
    if codec_metadata is None:
        return None
    hop_length = _metadata_positive_int(codec_metadata, "codec_hop_length")
    if hop_length is not None:
        return int(hop_length)
    hop_length = _metadata_positive_int(codec_metadata, "hop_length")
    if hop_length is not None:
        return int(hop_length)
    return _legacy_dac_hop_length(codec_metadata)


def resolve_target_token_rate_hz(
    codec_metadata: Mapping[str, Any] | Any | None,
    *,
    fallback: float = DEFAULT_TARGET_TOKEN_RATE_HZ,
) -> float:
    if codec_metadata is None:
        return float(fallback)
    sample_rate = _metadata_positive_float(codec_metadata, "codec_sample_rate")
    hop_length = resolve_codec_hop_length(codec_metadata)
    if sample_rate is not None and hop_length is not None:
        return float(sample_rate) / float(hop_length)
    rate = _metadata_positive_float(codec_metadata, "codec_frame_rate")
    if rate is None:
        rate = _metadata_positive_float(codec_metadata, "frame_rate")
    if rate is None:
        return float(fallback)
    return float(rate)


def uniform_beat_boundaries_from_durations(
    duration_sec_b: torch.Tensor,
    *,
    num_beats: int,
) -> torch.Tensor:
    num_beats_eff = int(max(1, int(num_beats)))
    duration_sec = torch.as_tensor(duration_sec_b, dtype=torch.float32).view(-1)
    fractions = torch.linspace(
        0.0,
        1.0,
        steps=int(num_beats_eff) + 1,
        device=duration_sec.device,
        dtype=duration_sec.dtype,
    ).view(1, -1)
    return (duration_sec.view(-1, 1) * fractions).contiguous()


def _resolve_duration_from_bpm(
    bpm_b: torch.Tensor | None,
    *,
    num_beats: int,
    fallback_duration_sec_b: torch.Tensor | None = None,
) -> torch.Tensor:
    if bpm_b is None:
        if fallback_duration_sec_b is None:
            raise ValueError("bpm is required when fallback_duration_sec_b is not provided")
        fallback = torch.as_tensor(fallback_duration_sec_b, dtype=torch.float32).view(-1)
        if not bool(torch.all(fallback > 0.0)):
            raise ValueError("fallback_duration_sec_b must be positive")
        return fallback.contiguous()

    bpm = torch.as_tensor(bpm_b, dtype=torch.float32).view(-1)
    duration_sec = torch.full_like(bpm, 0.0)
    valid_bpm_mask = bpm > 1.0e-6
    if bool(valid_bpm_mask.any()):
        duration_sec[valid_bpm_mask] = (float(max(1, int(num_beats))) * 60.0) / bpm[valid_bpm_mask]
    if bool((~valid_bpm_mask).any()):
        if fallback_duration_sec_b is None:
            raise ValueError("bpm must be positive for every example when fallback_duration_sec_b is not provided")
        fallback = torch.as_tensor(
            fallback_duration_sec_b,
            dtype=torch.float32,
            device=bpm.device,
        ).view(-1)
        if tuple(fallback.shape) != tuple(bpm.shape):
            raise ValueError(
                f"fallback_duration_sec_b must match bpm shape, got {tuple(fallback.shape)} / {tuple(bpm.shape)}"
            )
        fallback_invalid = fallback[~valid_bpm_mask]
        if not bool(torch.all(fallback_invalid > 0.0)):
            raise ValueError("fallback_duration_sec_b must be positive for examples with invalid bpm")
        duration_sec[~valid_bpm_mask] = fallback_invalid
    return duration_sec.contiguous()


def resolve_inference_geometry(
    prepared: Mapping[str, torch.Tensor | None],
    *,
    use_bpm_inference_geometry: bool = False,
    inference_num_beats: int = DEFAULT_INFERENCE_NUM_BEATS,
    target_token_rate_hz: float = DEFAULT_TARGET_TOKEN_RATE_HZ,
) -> dict[str, torch.Tensor]:
    grid = prepared.get("grid")
    grid_valid_mask = prepared.get("grid_valid_mask")
    if grid is None or grid_valid_mask is None:
        raise ValueError("prepared batch must include grid and grid_valid_mask")
    if not bool(use_bpm_inference_geometry):
        token_times_sec = prepared.get("token_times_sec")
        target_valid_mask_bt = prepared.get("target_valid_mask_bt")
        beat_boundaries_sec = prepared.get("beat_boundaries_sec")
        beat_boundaries_valid_mask = prepared.get("beat_boundaries_valid_mask")
        grid_times_sec = prepared.get("grid_times_sec")
        duration_sec = prepared.get("duration_sec")
        if (
            token_times_sec is None
            or target_valid_mask_bt is None
            or beat_boundaries_sec is None
            or beat_boundaries_valid_mask is None
            or grid_times_sec is None
            or duration_sec is None
        ):
            raise ValueError(
                "non-derived inference geometry requires grid_times_sec, token_times_sec, "
                "target_valid_mask_bt, beat_boundaries_sec, beat_boundaries_valid_mask, and duration_sec"
            )
        return {
            "grid_times_sec": grid_times_sec.contiguous(),
            "token_times_sec": token_times_sec.contiguous(),
            "target_valid_mask_bt": target_valid_mask_bt.to(dtype=torch.bool).contiguous(),
            "beat_boundaries_sec": beat_boundaries_sec.contiguous(),
            "beat_boundaries_valid_mask": beat_boundaries_valid_mask.to(dtype=torch.bool).contiguous(),
            "duration_sec": duration_sec.contiguous(),
            "target_num_frames_b": target_valid_mask_bt.to(dtype=torch.long).sum(dim=1).contiguous(),
        }

    duration_sec = _resolve_duration_from_bpm(
        prepared.get("bpm"),
        num_beats=int(inference_num_beats),
        fallback_duration_sec_b=prepared.get("duration_sec"),
    )
    grid_num_frames_b = grid_valid_mask.to(dtype=torch.long).sum(dim=1)
    target_num_frames_b = torch.round(duration_sec * float(max(1.0e-6, float(target_token_rate_hz)))).to(dtype=torch.long)
    target_num_frames_b = target_num_frames_b.clamp_min(1)
    max_target_len = int(target_num_frames_b.max().item())
    beat_boundaries_sec = uniform_beat_boundaries_from_durations(
        duration_sec,
        num_beats=int(inference_num_beats),
    )
    return {
        "grid_times_sec": uniform_frame_times_from_durations(
            grid_num_frames_b,
            duration_sec,
            max_num_frames=int(grid_valid_mask.shape[1]),
        ),
        "token_times_sec": uniform_frame_times_from_durations(
            target_num_frames_b,
            duration_sec,
            max_num_frames=int(max_target_len),
        ),
        "target_valid_mask_bt": lengths_to_mask(target_num_frames_b, max_len=int(max_target_len)),
        "beat_boundaries_sec": beat_boundaries_sec,
        "beat_boundaries_valid_mask": torch.ones_like(beat_boundaries_sec, dtype=torch.bool),
        "duration_sec": duration_sec.contiguous(),
        "target_num_frames_b": target_num_frames_b.contiguous(),
    }


def apply_bpm_training_geometry_to_prepared_batch(
    prepared: Mapping[str, torch.Tensor | None],
    *,
    num_beats: int = DEFAULT_INFERENCE_NUM_BEATS,
) -> dict[str, torch.Tensor | None]:
    """Retimes cached training tensors to BPM-derived durations without resizing targets."""
    grid_valid_mask = prepared.get("grid_valid_mask")
    target_valid_mask = prepared.get("target_valid_mask_bt")
    if grid_valid_mask is None or target_valid_mask is None:
        raise ValueError("BPM training geometry requires grid_valid_mask and target_valid_mask_bt")

    duration_sec = _resolve_duration_from_bpm(
        prepared.get("bpm"),
        num_beats=int(num_beats),
        fallback_duration_sec_b=prepared.get("duration_sec"),
    )
    grid_num_frames_b = torch.as_tensor(grid_valid_mask, dtype=torch.bool).to(dtype=torch.long).sum(dim=1)
    target_num_frames_b = torch.as_tensor(target_valid_mask, dtype=torch.bool).to(dtype=torch.long).sum(dim=1)
    beat_boundaries_sec = uniform_beat_boundaries_from_durations(
        duration_sec,
        num_beats=int(num_beats),
    )
    retimed = dict(prepared)
    retimed.update(
        {
            "grid_times_sec": uniform_frame_times_from_durations(
                grid_num_frames_b,
                duration_sec,
                max_num_frames=int(grid_valid_mask.shape[1]),
            ),
            "token_times_sec": uniform_frame_times_from_durations(
                target_num_frames_b,
                duration_sec,
                max_num_frames=int(target_valid_mask.shape[1]),
            ),
            "beat_boundaries_sec": beat_boundaries_sec,
            "beat_boundaries_valid_mask": torch.ones_like(beat_boundaries_sec, dtype=torch.bool),
            "duration_sec": duration_sec.contiguous(),
            "target_valid_mask_bt": torch.as_tensor(target_valid_mask, dtype=torch.bool).contiguous(),
        }
    )
    return retimed


class TimestepMLP(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        return self.net(t_emb)


class AdaLNModulation(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, d_model * 9),
        )

    def forward(self, t_ctx: torch.Tensor):
        return self.net(t_ctx).chunk(9, dim=-1)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiffusionTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp = FeedForward(d_model, mlp_ratio=mlp_ratio, dropout=dropout)
        self.mod = AdaLNModulation(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        t_ctx: torch.Tensor,
        target_valid_mask_bt: torch.Tensor,
        cond_valid_mask_bt: torch.Tensor,
    ) -> torch.Tensor:
        target_pad_bt = ~target_valid_mask_bt
        cond_pad_bt = ~cond_valid_mask_bt

        x = apply_seq_mask(x, target_valid_mask_bt)
        cond = apply_seq_mask(cond, cond_valid_mask_bt)

        (
            shift_sa, scale_sa, gate_sa,
            shift_ca, scale_ca, gate_ca,
            shift_ff, scale_ff, gate_ff,
        ) = self.mod(t_ctx)

        h = self.norm1(x)
        h = modulate(h, shift_sa, scale_sa)
        h, _ = self.self_attn(
            query=h,
            key=h,
            value=h,
            key_padding_mask=target_pad_bt,
            need_weights=False,
        )
        x = x + gate_sa.unsqueeze(1) * self.drop(h)
        x = apply_seq_mask(x, target_valid_mask_bt)

        h = self.norm2(x)
        h = modulate(h, shift_ca, scale_ca)
        h, _ = self.cross_attn(
            query=h,
            key=cond,
            value=cond,
            key_padding_mask=cond_pad_bt,
            need_weights=False,
        )
        x = x + gate_ca.unsqueeze(1) * self.drop(h)
        x = apply_seq_mask(x, target_valid_mask_bt)

        h = self.norm3(x)
        h = modulate(h, shift_ff, scale_ff)
        h = self.mlp(h)
        x = x + gate_ff.unsqueeze(1) * self.drop(h)
        x = apply_seq_mask(x, target_valid_mask_bt)
        return x


@dataclass
class DiffusionTransformerConfig:
    x_dim: int = 128
    frontend_cfg: Optional[dict[str, Any]] = None
    concat_multiscale_frontend: bool = DEFAULT_FRONTEND_CONCAT_MULTISCALE
    positional_encoding: str = DEFAULT_POSITIONAL_ENCODING
    positional_rate_hz: float = DEFAULT_TARGET_TOKEN_RATE_HZ
    d_model: int = 256
    num_layers: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    x0_prior_conditioning: bool = False
    x0_prior_dim: int = 72


class ConditionalDiffusionTransformer(nn.Module):
    def __init__(self, cfg: DiffusionTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.summary_frontend = build_seconds_frontend_from_cfg(cfg.frontend_cfg)
        if self.summary_frontend is None:
            raise ValueError("frontend_cfg is required for seconds-grid conditioning")
        if hasattr(self.summary_frontend, "window_radii"):
            self.frontend_scale_radii = tuple(sorted(int(x) for x in list(getattr(self.summary_frontend, "window_radii"))))
            self.frontend_primary_radius = int(getattr(self.summary_frontend, "primary_radius"))
        else:
            self.frontend_primary_radius = int(getattr(self.summary_frontend, "window_radius", 0))
            self.frontend_scale_radii = (int(self.frontend_primary_radius),)
        self.concat_multiscale_frontend = bool(cfg.concat_multiscale_frontend and hasattr(self.summary_frontend, "forward_multiscale"))
        frontend_output_dim = int(getattr(self.summary_frontend, "output_dim"))
        cond_dim = int(frontend_output_dim) * (int(len(self.frontend_scale_radii)) if bool(self.concat_multiscale_frontend) else 1)
        self.positional_encoding = str(getattr(cfg, "positional_encoding", DEFAULT_POSITIONAL_ENCODING)).strip().lower()
        if self.positional_encoding not in {"index", "seconds"}:
            raise ValueError(f"unsupported positional_encoding={self.positional_encoding!r}")
        self.positional_rate_hz = float(
            max(1.0e-6, float(getattr(cfg, "positional_rate_hz", DEFAULT_TARGET_TOKEN_RATE_HZ)))
        )

        self.x_proj = nn.Linear(cfg.x_dim, cfg.d_model)
        self.cond_proj = nn.Linear(cond_dim, cfg.d_model)
        self.x0_prior_conditioning = bool(getattr(cfg, "x0_prior_conditioning", False))
        self.x0_prior_proj: nn.Linear | None = None
        self.x0_prior_norm: nn.LayerNorm | None = None
        self.x0_prior_to_cond: nn.Linear | None = None
        if bool(self.x0_prior_conditioning):
            x0_prior_dim = int(getattr(cfg, "x0_prior_dim", 0) or cfg.x_dim)
            self.x0_prior_proj = nn.Linear(int(x0_prior_dim), int(cfg.d_model))
            self.x0_prior_norm = nn.LayerNorm(int(cfg.d_model))
            self.x0_prior_to_cond = nn.Linear(int(cfg.d_model), int(cond_dim))
            nn.init.zeros_(self.x0_prior_to_cond.weight)
            nn.init.zeros_(self.x0_prior_to_cond.bias)
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.blocks = nn.ModuleList(
            [
                DiffusionTransformerBlock(
                    d_model=cfg.d_model,
                    num_heads=cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.final_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model * 2),
        )
        self.out_proj = nn.Linear(cfg.d_model, cfg.x_dim)

    def encode_conditioning(
        self,
        *,
        grid: torch.Tensor,
        grid_ids: Optional[torch.Tensor],
        grid_times_sec: torch.Tensor,
        token_times_sec: torch.Tensor,
        target_valid_mask_bt: torch.Tensor,
        grid_valid_mask_bt: Optional[torch.Tensor] = None,
        x0_prior_btd: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frontend_kwargs = {
            "grid_ids_bct": grid_ids,
            "grid_times_sec_bt": grid_times_sec,
            "token_times_sec_bt": token_times_sec,
            "grid_valid_mask_bt": grid_valid_mask_bt,
            "valid_mask_bt": target_valid_mask_bt,
        }
        if bool(self.concat_multiscale_frontend):
            scale_features = {
                int(scale_radius): scale_feat
                for scale_radius, scale_feat in dict(
                    self.summary_frontend.forward_multiscale(grid, **frontend_kwargs)
                ).items()
            }
            cond_btd = torch.cat(
                [
                    scale_features[int(scale_radius)]
                    for scale_radius in list(self.frontend_scale_radii)
                ],
                dim=-1,
            ).contiguous()
        else:
            cond_btd = self.summary_frontend(grid, **frontend_kwargs)

        cond_valid_mask_bt = target_valid_mask_bt.to(dtype=torch.bool)
        cond_btd = apply_seq_mask(cond_btd, cond_valid_mask_bt)
        target_len = int(target_valid_mask_bt.shape[1])
        batch_size = int(grid.shape[0])
        if (
            bool(self.x0_prior_conditioning)
            and x0_prior_btd is not None
            and self.x0_prior_proj is not None
            and self.x0_prior_norm is not None
            and self.x0_prior_to_cond is not None
        ):
            prior = torch.as_tensor(x0_prior_btd, dtype=torch.float32, device=grid.device)
            if int(prior.dim()) != 3:
                raise ValueError(f"x0_prior_btd must be [B,T,D], got {tuple(prior.shape)}")
            if int(prior.shape[0]) != int(batch_size):
                raise ValueError(
                    f"x0_prior_btd batch must be {batch_size}, got {tuple(prior.shape)}"
                )
            expected_dim = int(self.x0_prior_proj.in_features)
            if int(prior.shape[-1]) != expected_dim:
                raise ValueError(
                    f"x0_prior_btd last dim must be {expected_dim}, got {tuple(prior.shape)}"
                )
            if int(prior.shape[1]) != int(target_len):
                prior = F.interpolate(
                    prior.transpose(1, 2),
                    size=int(target_len),
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2).contiguous()
            prior = apply_seq_mask(prior, cond_valid_mask_bt)
            cond_btd = cond_btd + self.x0_prior_to_cond(
                self.x0_prior_norm(self.x0_prior_proj(prior))
            )
        return cond_btd.contiguous(), cond_valid_mask_bt.contiguous()

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        *,
        target_valid_mask_bt: torch.Tensor,
        grid: Optional[torch.Tensor] = None,
        grid_ids: Optional[torch.Tensor] = None,
        grid_times_sec: Optional[torch.Tensor] = None,
        token_times_sec: Optional[torch.Tensor] = None,
        grid_valid_mask_bt: Optional[torch.Tensor] = None,
        beat_boundaries_sec: Optional[torch.Tensor] = None,
        beat_boundaries_valid_mask: Optional[torch.Tensor] = None,
        bpm: Optional[torch.Tensor] = None,
        duration_sec: Optional[torch.Tensor] = None,
        cond_btd: Optional[torch.Tensor] = None,
        cond_valid_mask_bt: Optional[torch.Tensor] = None,
        x0_prior_btd: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del beat_boundaries_sec, beat_boundaries_valid_mask, bpm, duration_sec
        bsz, target_len, _ = x_t.shape
        device = x_t.device

        if cond_btd is None or cond_valid_mask_bt is None:
            missing = [
                name
                for name, value in (
                    ("grid", grid),
                    ("grid_times_sec", grid_times_sec),
                    ("token_times_sec", token_times_sec),
                )
                if value is None
            ]
            if missing:
                raise ValueError(f"missing conditioning inputs: {missing}")
            cond_btd, cond_valid_mask_bt = self.encode_conditioning(
                grid=grid,
                grid_ids=grid_ids,
                grid_times_sec=grid_times_sec,
                token_times_sec=token_times_sec,
                target_valid_mask_bt=target_valid_mask_bt,
                grid_valid_mask_bt=grid_valid_mask_bt,
                x0_prior_btd=x0_prior_btd,
            )

        if int(cond_btd.shape[0]) != int(target_valid_mask_bt.shape[0]) or int(cond_valid_mask_bt.shape[0]) != int(target_valid_mask_bt.shape[0]):
            raise ValueError(
                f"conditioning batch must align with target_valid_mask_bt, got {tuple(cond_btd.shape)} / {tuple(cond_valid_mask_bt.shape)} / {tuple(target_valid_mask_bt.shape)}"
            )

        if self.positional_encoding == "seconds" and token_times_sec is not None:
            token_pos = sinusoidal_time_positions(
                torch.as_tensor(token_times_sec, dtype=torch.float32, device=device),
                self.cfg.d_model,
                rate_hz=float(self.positional_rate_hz),
            )
            if tuple(token_pos.shape[:2]) != tuple(target_valid_mask_bt.shape):
                raise ValueError(
                    "token_times_sec must align with target_valid_mask_bt for seconds positional encoding, got "
                    f"{tuple(token_pos.shape[:2])} / {tuple(target_valid_mask_bt.shape)}"
                )
            x_pos = token_pos
            if int(cond_btd.shape[1]) == int(target_len):
                c_pos = token_pos
            else:
                extra = cond_btd.new_zeros((bsz, int(cond_btd.shape[1]) - int(target_len), int(self.cfg.d_model)))
                c_pos = torch.cat([token_pos, extra], dim=1).contiguous()
        else:
            x_pos = sinusoidal_positions(target_len, self.cfg.d_model, device)
            c_pos = sinusoidal_positions(int(cond_btd.shape[1]), self.cfg.d_model, device)

        x = self.x_proj(x_t) + x_pos
        c = self.cond_proj(cond_btd) + c_pos
        t_emb = timestep_embedding(t, self.cfg.d_model)
        t_ctx = self.time_mlp(t_emb)

        x = apply_seq_mask(x, target_valid_mask_bt)
        c = apply_seq_mask(c, cond_valid_mask_bt)

        for block in self.blocks:
            x = block(
                x=x,
                cond=c,
                t_ctx=t_ctx,
                target_valid_mask_bt=target_valid_mask_bt,
                cond_valid_mask_bt=cond_valid_mask_bt,
            )

        shift, scale = self.final_mod(t_ctx).chunk(2, dim=-1)
        x = self.final_norm(x)
        x = modulate(x, shift, scale)
        x = self.out_proj(x)
        x = apply_seq_mask(x, target_valid_mask_bt)
        return x


def cosine_beta_schedule(num_steps: int, s: float = 0.008) -> torch.Tensor:
    steps = num_steps + 1
    x = torch.linspace(0, num_steps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-5, 0.999).float()


class GaussianDiffusion1D(nn.Module):
    def __init__(self, num_steps: int = 1000):
        super().__init__()
        betas = cosine_beta_schedule(num_steps=num_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.num_steps = num_steps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        alpha_bars_prev = torch.cat([torch.ones(1, device=betas.device), alpha_bars[:-1]], dim=0)
        posterior_var = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        self.register_buffer("posterior_variance", posterior_var.clamp_min(1e-20))
        posterior_mean_coef1 = betas * torch.sqrt(alpha_bars_prev) / (1.0 - alpha_bars)
        posterior_mean_coef2 = (1.0 - alpha_bars_prev) * torch.sqrt(alphas) / (1.0 - alpha_bars)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        b = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)
        return a * x0 + b * noise

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        a = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        b = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)
        return (x_t - b * eps) / a.clamp_min(1e-8)

    def posterior_mean_from_x0(self, x_t: torch.Tensor, t: torch.Tensor, x0_hat: torch.Tensor) -> torch.Tensor:
        coef1 = self.posterior_mean_coef1[t].view(-1, 1, 1)
        coef2 = self.posterior_mean_coef2[t].view(-1, 1, 1)
        return coef1 * x0_hat + coef2 * x_t


def resolve_valid_audio_num_samples(
    duration_sec_b: torch.Tensor,
    *,
    sample_rate: int,
    max_num_samples: int,
) -> torch.Tensor:
    duration_sec = torch.as_tensor(duration_sec_b, dtype=torch.float32).view(-1)
    valid_num_samples_b = torch.round(duration_sec * float(sample_rate)).to(dtype=torch.long)
    return valid_num_samples_b.clamp(min=1, max=max(1, int(max_num_samples))).contiguous()


def audio_valid_mask(
    valid_num_samples_b: torch.Tensor,
    *,
    max_num_samples: int,
) -> torch.Tensor:
    valid_num_samples = torch.as_tensor(valid_num_samples_b, dtype=torch.long).view(-1)
    return lengths_to_mask(valid_num_samples, max_len=int(max_num_samples))


def masked_audio_l1_per_example(
    pred_audio_bct: torch.Tensor,
    target_audio_bct: torch.Tensor,
    valid_num_samples_b: torch.Tensor,
) -> torch.Tensor:
    pred_audio = torch.as_tensor(pred_audio_bct, dtype=torch.float32)
    target_audio = torch.as_tensor(target_audio_bct, dtype=torch.float32, device=pred_audio.device)
    if tuple(pred_audio.shape) != tuple(target_audio.shape):
        raise ValueError(
            f"pred_audio_bct and target_audio_bct must match, got {tuple(pred_audio.shape)} / {tuple(target_audio.shape)}"
        )
    if int(pred_audio.dim()) != 3:
        raise ValueError(f"expected [B,C,T] audio tensors, got {tuple(pred_audio.shape)}")
    weights_b1t = audio_valid_mask(
        valid_num_samples_b,
        max_num_samples=int(pred_audio.shape[-1]),
    ).to(device=pred_audio.device, dtype=pred_audio.dtype)[:, None, :]
    denom_b = weights_b1t.sum(dim=(1, 2)).clamp_min(1.0)
    return (((pred_audio - target_audio).abs()) * weights_b1t).sum(dim=(1, 2)) / denom_b


def _safe_stft(audio_bt: torch.Tensor, *, n_fft: int, hop: int) -> torch.Tensor:
    audio = torch.as_tensor(audio_bt, dtype=torch.float32)
    if int(audio.dim()) != 2:
        raise ValueError(f"audio_bt must be [B,T], got {tuple(audio.shape)}")
    n_fft_eff = int(max(16, int(n_fft)))
    hop_eff = int(max(1, int(hop)))
    if int(audio.shape[-1]) < int(n_fft_eff):
        audio = F.pad(audio, (0, int(n_fft_eff) - int(audio.shape[-1])))
    window = torch.hann_window(int(n_fft_eff), device=audio.device, dtype=audio.dtype)
    return torch.stft(
        audio,
        n_fft=int(n_fft_eff),
        hop_length=int(hop_eff),
        win_length=int(n_fft_eff),
        window=window,
        center=True,
        return_complex=True,
    )


def mrstft_logmag_l1_per_example(
    pred_audio_bct: torch.Tensor,
    target_audio_bct: torch.Tensor,
    valid_num_samples_b: torch.Tensor,
    *,
    resolutions: Sequence[tuple[int, int]] = DEFAULT_AUDIO_MRSTFT_RESOLUTIONS,
) -> torch.Tensor:
    pred_audio = torch.as_tensor(pred_audio_bct, dtype=torch.float32)
    target_audio = torch.as_tensor(target_audio_bct, dtype=torch.float32, device=pred_audio.device)
    if tuple(pred_audio.shape) != tuple(target_audio.shape):
        raise ValueError(
            f"pred_audio_bct and target_audio_bct must match, got {tuple(pred_audio.shape)} / {tuple(target_audio.shape)}"
        )
    if int(pred_audio.dim()) != 3:
        raise ValueError(f"expected [B,C,T] audio tensors, got {tuple(pred_audio.shape)}")
    pred_audio_bt = pred_audio.mean(dim=1)
    target_audio_bt = target_audio.mean(dim=1)
    valid_mask_bt = audio_valid_mask(
        valid_num_samples_b,
        max_num_samples=int(pred_audio_bt.shape[-1]),
    ).to(device=pred_audio.device)
    pred_audio_bt = pred_audio_bt * valid_mask_bt.to(dtype=pred_audio_bt.dtype)
    target_audio_bt = target_audio_bt * valid_mask_bt.to(dtype=target_audio_bt.dtype)
    total_b = pred_audio_bt.new_zeros((int(pred_audio_bt.shape[0]),), dtype=torch.float32)
    resolutions_eff = tuple((int(n_fft), int(hop)) for n_fft, hop in tuple(resolutions))
    if not resolutions_eff:
        raise ValueError("expected at least one MRSTFT resolution")
    for n_fft, hop in resolutions_eff:
        pred_spec = _safe_stft(pred_audio_bt, n_fft=int(n_fft), hop=int(hop))
        target_spec = _safe_stft(target_audio_bt, n_fft=int(n_fft), hop=int(hop))
        pred_logmag = torch.log1p(pred_spec.abs())
        target_logmag = torch.log1p(target_spec.abs())
        valid_frames_b = 1 + torch.div(valid_mask_bt.sum(dim=1).to(torch.long), int(max(1, hop)), rounding_mode="floor")
        frame_mask_bt = audio_valid_mask(
            valid_frames_b,
            max_num_samples=int(pred_logmag.shape[-1]),
        ).to(device=pred_audio.device, dtype=pred_logmag.dtype)
        weights_bft = frame_mask_bt[:, None, :]
        denom_b = (weights_bft.sum(dim=(1, 2)) * float(pred_logmag.shape[1])).clamp_min(1.0)
        total_b = total_b + (((pred_logmag - target_logmag).abs()) * weights_bft).sum(dim=(1, 2)) / denom_b
    return total_b / float(len(resolutions_eff))


def _onset_boost_for_class_name(
    name: str,
    *,
    kick_snare_boost: float = 3.0,
    hihat_boost: float = 1.0,
) -> float:
    normalized = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    if "kick" in normalized or normalized in {"bd", "bass_drum"}:
        return float(kick_snare_boost)
    if "snare" in normalized or normalized in {"sd"}:
        return float(kick_snare_boost)
    if (
        "hihat" in normalized
        or "hi_hat" in normalized
        or normalized.endswith("_hh")
        or normalized.startswith("hh_")
        or normalized == "hh"
    ):
        return float(hihat_boost)
    return 0.0


def _build_onset_token_weights(
    prepared: Mapping[str, torch.Tensor | None],
    batch: Mapping[str, Any],
    *,
    base_weight: float = 1.0,
    kick_snare_boost: float = 3.0,
    hihat_boost: float = 1.0,
    token_radius: int = 1,
) -> torch.Tensor:
    target_mask = torch.as_tensor(prepared["target_valid_mask_bt"], dtype=torch.bool)
    weights = torch.full_like(target_mask, float(base_weight), dtype=torch.float32)
    weights = weights * target_mask.to(dtype=weights.dtype)

    family_onsets = prepared.get("family_onsets_bft")
    grid_times = prepared.get("grid_times_sec")
    token_times = prepared.get("token_times_sec")
    if family_onsets is None or grid_times is None or token_times is None:
        return weights.contiguous()

    class_names = [str(name) for name in list(batch.get("class_names") or [])]
    if not class_names:
        return weights.contiguous()

    boosts = [
        _onset_boost_for_class_name(
            name,
            kick_snare_boost=float(kick_snare_boost),
            hihat_boost=float(hihat_boost),
        )
        for name in class_names
    ]
    if not any(float(boost) > 0.0 for boost in boosts):
        return weights.contiguous()

    grid_valid = prepared.get("grid_valid_mask")
    radius = max(0, int(token_radius))
    batch_size = int(target_mask.shape[0])
    num_families = int(family_onsets.shape[1])
    token_count = int(target_mask.shape[1])
    inf = torch.tensor(float("inf"), dtype=token_times.dtype, device=token_times.device)

    for batch_idx in range(batch_size):
        target_valid_b = target_mask[batch_idx]
        if not bool(target_valid_b.any()):
            continue
        token_times_b = token_times[batch_idx]
        valid_token_count = int(target_valid_b.shape[0])
        if valid_token_count <= 0:
            continue
        grid_valid_b = (
            grid_valid[batch_idx]
            if grid_valid is not None
            else torch.ones_like(grid_times[batch_idx], dtype=torch.bool)
        )
        for family_idx in range(min(len(boosts), num_families)):
            boost = float(boosts[family_idx])
            if boost <= 0.0:
                continue
            onset_mask = family_onsets[batch_idx, family_idx] & grid_valid_b
            if not bool(onset_mask.any()):
                continue
            onset_times = grid_times[batch_idx][onset_mask]
            for onset_time in onset_times:
                distances = (token_times_b - onset_time).abs().masked_fill(~target_valid_b, inf)
                center_idx = int(distances.argmin().item())
                start = max(0, center_idx - radius)
                stop = min(token_count, center_idx + radius + 1)
                if stop <= start:
                    continue
                weights[batch_idx, start:stop] = weights[batch_idx, start:stop] + (
                    boost * target_valid_b[start:stop].to(dtype=weights.dtype)
                )
    return weights.contiguous()


def _masked_token_mean(
    per_token_bt: torch.Tensor,
    valid_mask_bt: torch.Tensor,
    token_weights_bt: torch.Tensor | None = None,
) -> torch.Tensor:
    per_token = torch.as_tensor(per_token_bt, dtype=torch.float32)
    mask = torch.as_tensor(valid_mask_bt, dtype=torch.bool, device=per_token.device)
    weights = mask.to(dtype=per_token.dtype)
    if token_weights_bt is not None:
        weights = weights * torch.as_tensor(token_weights_bt, dtype=per_token.dtype, device=per_token.device)
    return (per_token * weights).sum() / weights.sum().clamp_min(1.0e-8)


def _resolve_codebook_embeddings(
    quant_codebook_embed_ckd: torch.Tensor | None,
    *,
    x_dim: int,
    device: torch.device,
) -> torch.Tensor:
    if quant_codebook_embed_ckd is None:
        raise ValueError("quant_codebook_embed_ckd is required when codebook auxiliary losses are enabled")
    codebook_embed = torch.as_tensor(
        quant_codebook_embed_ckd,
        dtype=torch.float32,
        device=device,
    ).detach()
    if int(codebook_embed.dim()) != 3 or int(codebook_embed.shape[-1]) != int(x_dim):
        raise ValueError(
            "quant_codebook_embed_ckd must have shape [C,K,D] with "
            f"D={int(x_dim)}, got {tuple(codebook_embed.shape)}"
        )
    if int(codebook_embed.shape[0]) <= 0 or int(codebook_embed.shape[1]) <= 0:
        raise ValueError(f"quant_codebook_embed_ckd must be non-empty, got {tuple(codebook_embed.shape)}")
    return codebook_embed.contiguous()


def _resolve_rvq_target_codes_bct(
    *,
    prepared: Mapping[str, torch.Tensor | None],
    encodec_model: Any | None,
    target_codec_raw: torch.Tensor,
    target_mask: torch.Tensor,
    device: torch.device,
    target_pca_basis: Mapping[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    source_codes = prepared.get("source_codes_bct")
    if source_codes is not None:
        target_codes = torch.as_tensor(source_codes, dtype=torch.long, device=device)
        if int(target_codes.dim()) != 3:
            raise ValueError(f"source_codes_bct must be [B,C,T], got {tuple(target_codes.shape)}")
    else:
        if encodec_model is None:
            raise ValueError("encodec_model is required for RVQ CE when source_codes_bct is absent")
        target_codes = requantize_latent_to_codes_bct(
            encodec_model,
            apply_seq_mask(target_codec_raw, target_mask),
            device=device,
            target_pca_basis=target_pca_basis,
        )

    compared_frames = int(min(int(target_codes.shape[-1]), int(target_mask.shape[-1]), int(target_codec_raw.shape[1])))
    target_codes = target_codes[:, :, :compared_frames].contiguous()
    valid_mask = target_mask[:, :compared_frames].to(dtype=torch.bool).contiguous()
    valid_mask = valid_mask & target_codes.ge(0).all(dim=1)
    target_codes = target_codes.clamp_min(0).contiguous()
    return target_codes, valid_mask


def _resolve_target_pca_basis(
    target_pca_basis: Mapping[str, Any] | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any] | None:
    if target_pca_basis is None:
        return None
    return load_target_pca_basis(target_pca_basis, device=device, dtype=dtype)


def _rvq_cross_entropy_loss(
    pred_latent_btd: torch.Tensor,
    target_codes_bct: torch.Tensor,
    codebook_embed_ckd: torch.Tensor,
    valid_mask_bt: torch.Tensor,
    token_weights_bt: torch.Tensor | None = None,
) -> torch.Tensor:
    if int(pred_latent_btd.dim()) != 3:
        raise ValueError(f"pred_latent_btd must be [B,T,D], got {tuple(pred_latent_btd.shape)}")
    if int(target_codes_bct.dim()) != 3:
        raise ValueError(f"target_codes_bct must be [B,C,T], got {tuple(target_codes_bct.shape)}")
    if int(codebook_embed_ckd.dim()) != 3:
        raise ValueError(f"codebook_embed_ckd must be [C,K,D], got {tuple(codebook_embed_ckd.shape)}")
    batch_size, num_frames, x_dim = [int(x) for x in list(pred_latent_btd.shape)]
    if int(target_codes_bct.shape[0]) != int(batch_size):
        raise ValueError(
            f"target_codes_bct batch must match pred_latent_btd, got {tuple(target_codes_bct.shape)} "
            f"vs {tuple(pred_latent_btd.shape)}"
        )
    if int(codebook_embed_ckd.shape[0]) != int(target_codes_bct.shape[1]):
        raise ValueError(
            f"codebook count mismatch: codes={tuple(target_codes_bct.shape)} "
            f"embeddings={tuple(codebook_embed_ckd.shape)}"
        )
    if int(codebook_embed_ckd.shape[-1]) != int(x_dim):
        raise ValueError(
            f"codebook embedding dimension must be {int(x_dim)}, got {tuple(codebook_embed_ckd.shape)}"
        )

    compared_frames = int(min(int(num_frames), int(target_codes_bct.shape[-1]), int(valid_mask_bt.shape[-1])))
    pred_latent = pred_latent_btd[:, :compared_frames, :].to(dtype=torch.float32)
    target_codes = target_codes_bct[:, :, :compared_frames].to(device=pred_latent.device, dtype=torch.long)
    valid_mask = valid_mask_bt[:, :compared_frames].to(device=pred_latent.device, dtype=torch.bool)
    valid_mask = valid_mask & target_codes.ge(0).all(dim=1)
    target_codes = target_codes.clamp_min(0).contiguous()
    if not bool(valid_mask.any()):
        return pred_latent.sum() * 0.0

    token_weights = None
    if token_weights_bt is not None:
        token_weights = torch.as_tensor(
            token_weights_bt[:, :compared_frames],
            dtype=pred_latent.dtype,
            device=pred_latent.device,
        )
        token_weights = token_weights * valid_mask.to(dtype=token_weights.dtype)

    target_codebook_latents = token_ids_to_codebook_embeddings(
        target_codes,
        codebook_embed_ckd.to(device=pred_latent.device, dtype=torch.float32),
        valid_bt=valid_mask,
    ).detach()

    valid_flat = valid_mask.reshape(-1)
    weights_flat = None if token_weights is None else token_weights.reshape(-1)
    prev_sum = torch.zeros_like(pred_latent)
    total = pred_latent.new_zeros(())
    denom = pred_latent.new_zeros(())
    for codebook_idx in range(int(target_codes.shape[1])):
        residual_flat = (pred_latent - prev_sum).reshape(-1, int(x_dim))[valid_flat]
        labels_flat = target_codes[:, int(codebook_idx), :].reshape(-1)[valid_flat]
        embed_kd = codebook_embed_ckd[int(codebook_idx)].to(device=pred_latent.device, dtype=torch.float32)
        dist_sq = (
            residual_flat.square().sum(dim=-1, keepdim=True)
            + embed_kd.square().sum(dim=-1).view(1, -1)
            - (2.0 * residual_flat.matmul(embed_kd.transpose(0, 1)))
        )
        logits = -torch.sqrt(dist_sq.clamp_min(1.0e-12))
        ce = F.cross_entropy(logits, labels_flat, reduction="none")
        if weights_flat is None:
            total = total + ce.sum()
            denom = denom + torch.as_tensor(float(int(ce.numel())), dtype=denom.dtype, device=denom.device)
        else:
            weights_valid = weights_flat[valid_flat]
            total = total + (ce * weights_valid).sum()
            denom = denom + weights_valid.sum()
        prev_sum = prev_sum + target_codebook_latents[:, int(codebook_idx), :, :]
    return total / denom.clamp_min(1.0e-8)


def diffusion_train_step(
    model: ConditionalDiffusionTransformer,
    diffusion: GaussianDiffusion1D,
    batch: Mapping[str, Any],
    device: torch.device,
    *,
    target_mean=None,
    target_std=None,
    encodec_model: Any | None = None,
    audio_sample_rate: int | None = None,
    audio_wave_l1_weight: float = DEFAULT_AUDIO_WAVE_L1_WEIGHT,
    audio_mrstft_weight: float = DEFAULT_AUDIO_MRSTFT_WEIGHT,
    audio_mrstft_resolutions: Sequence[tuple[int, int]] = DEFAULT_AUDIO_MRSTFT_RESOLUTIONS,
    x0_clip_norm: float | None = DEFAULT_SAMPLE_X0_CLIP_NORM,
    x0_mse_weight: float = 0.0,
    quant_embed_mse_weight: float = 0.0,
    rvq_ce_weight: float = 0.0,
    quant_codebook_embed_ckd: torch.Tensor | None = None,
    onset_loss_weighting: bool = False,
    onset_token_radius: int = 1,
    target_pca_basis: Mapping[str, Any] | None = None,
    use_bpm_training_geometry: bool = False,
    bpm_geometry_num_beats: int = DEFAULT_INFERENCE_NUM_BEATS,
):
    prepared = _prepare_batch_tensors(batch, device)
    if bool(use_bpm_training_geometry):
        prepared = apply_bpm_training_geometry_to_prepared_batch(
            prepared,
            num_beats=int(bpm_geometry_num_beats),
        )
    target_raw = prepared["target_btd"]
    target = normalize_latent(target_raw, target_mean, target_std)
    target_mask = prepared["target_valid_mask_bt"]
    resolved_target_pca_basis = _resolve_target_pca_basis(
        target_pca_basis,
        device=device,
        dtype=target_raw.dtype,
    )
    target_codec_raw: torch.Tensor | None = None

    def _target_codec_latent() -> torch.Tensor:
        nonlocal target_codec_raw
        if target_codec_raw is not None:
            return target_codec_raw
        target_sum = prepared.get("target_sum_btd")
        if target_sum is not None:
            target_codec_raw = apply_seq_mask(target_sum, target_mask)
        else:
            target_codec_raw = apply_seq_mask(
                reconstruct_latent_from_pca(target_raw, resolved_target_pca_basis),
                target_mask,
            )
        return target_codec_raw

    noise = torch.randn_like(target)
    noise = apply_seq_mask(noise, target_mask)
    x0_prior = prepared.get("x0_prior_btd")
    if x0_prior is not None:
        x0_prior = normalize_latent(x0_prior, target_mean, target_std)
        x0_prior = apply_seq_mask(x0_prior, target_mask)

    batch_size = int(target.shape[0])
    t = torch.randint(0, diffusion.num_steps, (batch_size,), device=device)

    x_t = diffusion.q_sample(target, t, noise)
    x_t = apply_seq_mask(x_t, target_mask)

    pred_eps = model(
        x_t=x_t,
        t=t,
        target_valid_mask_bt=target_mask,
        grid=prepared["grid"],
        grid_ids=prepared["grid_ids"],
        grid_times_sec=prepared["grid_times_sec"],
        token_times_sec=prepared["token_times_sec"],
        grid_valid_mask_bt=prepared["grid_valid_mask"],
        beat_boundaries_sec=prepared["beat_boundaries_sec"],
        beat_boundaries_valid_mask=prepared["beat_boundaries_valid_mask"],
        bpm=prepared["bpm"],
        duration_sec=prepared["duration_sec"],
        x0_prior_btd=x0_prior,
    )

    loss_per_bt = ((pred_eps - noise) ** 2).mean(dim=-1)
    diffusion_loss = loss_per_bt[target_mask].mean()
    x0_hat = diffusion.predict_x0_from_eps(x_t, t, pred_eps)
    if x0_clip_norm is not None:
        x0_hat = x0_hat.clamp(min=-float(x0_clip_norm), max=float(x0_clip_norm))
    x0_hat = apply_seq_mask(x0_hat, target_mask)

    loss = diffusion_loss
    x0_loss = x0_hat.new_zeros(())
    quant_embed_mse = x0_hat.new_zeros(())
    rvq_ce = x0_hat.new_zeros(())
    onset_weighted_x0 = x0_hat.new_zeros(())
    per_tok_x0 = ((x0_hat - target) ** 2).mean(dim=-1)
    use_x0_loss = float(x0_mse_weight) > 0.0
    use_quant_embed_loss = float(quant_embed_mse_weight) > 0.0
    use_rvq_ce_loss = float(rvq_ce_weight) > 0.0
    token_weights = None
    if bool(onset_loss_weighting) and (
        use_x0_loss or use_quant_embed_loss or use_rvq_ce_loss
    ):
        token_weights = _build_onset_token_weights(
            prepared,
            batch,
            token_radius=int(onset_token_radius),
        ).to(device=x0_hat.device, dtype=x0_hat.dtype)
        onset_weighted_x0 = _masked_token_mean(per_tok_x0, target_mask, token_weights)

    if use_x0_loss:
        x0_loss = _masked_token_mean(per_tok_x0, target_mask, None)
        x0_objective = onset_weighted_x0 if token_weights is not None else x0_loss
        loss = loss + (float(x0_mse_weight) * x0_objective)

    pred_latent_raw: torch.Tensor | None = None
    pred_codec_latent_raw: torch.Tensor | None = None
    codebook_embed: torch.Tensor | None = None
    if use_quant_embed_loss or use_rvq_ce_loss:
        codebook_embed = _resolve_codebook_embeddings(
            quant_codebook_embed_ckd,
            x_dim=int(_target_codec_latent().shape[-1]),
            device=device,
        )

    if use_quant_embed_loss:
        if encodec_model is None:
            raise ValueError("encodec_model is required when quant_embed_mse_weight > 0")
        if codebook_embed is None:
            raise AssertionError("codebook embeddings should have been resolved")
        pred_latent_raw = denormalize_latent(x0_hat, target_mean, target_std)
        pred_latent_raw = apply_seq_mask(pred_latent_raw, target_mask)
        pred_codec_latent_raw = reconstruct_latent_from_pca(
            pred_latent_raw,
            resolved_target_pca_basis,
        )
        pred_codec_latent_raw = apply_seq_mask(pred_codec_latent_raw, target_mask)
        with torch.no_grad():
            target_requant_codes = requantize_latent_to_codes_bct(
                encodec_model,
                _target_codec_latent(),
                device=device,
            )
            compared_frames = int(min(int(target_requant_codes.shape[-1]), int(target_mask.shape[-1])))
            target_requant_codes = target_requant_codes[:, :, :compared_frames]
            quant_valid_mask = target_mask[:, :compared_frames]
            target_codebook_latents = token_ids_to_codebook_embeddings(
                target_requant_codes,
                codebook_embed,
                valid_bt=quant_valid_mask,
            )
            target_requant_sum = rvq_sum_latents(
                target_codebook_latents,
                valid_bt=quant_valid_mask,
            )
        pred_quant_aligned = pred_codec_latent_raw[:, :compared_frames, :]
        per_tok_quant = ((pred_quant_aligned - target_requant_sum) ** 2).mean(dim=-1)
        token_weights_quant = None if token_weights is None else token_weights[:, :compared_frames]
        quant_embed_mse = _masked_token_mean(per_tok_quant, quant_valid_mask, token_weights_quant)
        loss = loss + (float(quant_embed_mse_weight) * quant_embed_mse)

    if use_rvq_ce_loss:
        if codebook_embed is None:
            raise AssertionError("codebook embeddings should have been resolved")
        if pred_latent_raw is None:
            pred_latent_raw = denormalize_latent(x0_hat, target_mean, target_std)
            pred_latent_raw = apply_seq_mask(pred_latent_raw, target_mask)
        if pred_codec_latent_raw is None:
            pred_codec_latent_raw = reconstruct_latent_from_pca(
                pred_latent_raw,
                resolved_target_pca_basis,
            )
            pred_codec_latent_raw = apply_seq_mask(pred_codec_latent_raw, target_mask)
        target_rvq_codes, rvq_valid_mask = _resolve_rvq_target_codes_bct(
            prepared=prepared,
            encodec_model=encodec_model,
            target_codec_raw=_target_codec_latent(),
            target_mask=target_mask,
            device=device,
            target_pca_basis=resolved_target_pca_basis,
        )
        token_weights_rvq = None if token_weights is None else token_weights[:, : int(rvq_valid_mask.shape[-1])]
        rvq_ce = _rvq_cross_entropy_loss(
            pred_codec_latent_raw[:, : int(rvq_valid_mask.shape[-1]), :],
            target_rvq_codes,
            codebook_embed,
            rvq_valid_mask,
            token_weights_rvq,
        )
        loss = loss + (float(rvq_ce_weight) * rvq_ce)

    audio_wave_l1 = x0_hat.new_zeros(())
    audio_mrstft = x0_hat.new_zeros(())
    if encodec_model is not None and (
        float(audio_wave_l1_weight) > 0.0 or float(audio_mrstft_weight) > 0.0
    ):
        if pred_latent_raw is None:
            pred_latent_raw = denormalize_latent(x0_hat, target_mean, target_std)
            pred_latent_raw = apply_seq_mask(pred_latent_raw, target_mask)
        if pred_codec_latent_raw is None:
            pred_codec_latent_raw = reconstruct_latent_from_pca(
                pred_latent_raw,
                resolved_target_pca_basis,
            )
            pred_codec_latent_raw = apply_seq_mask(pred_codec_latent_raw, target_mask)
        # EnCodec's decoder contains recurrent layers. When the frozen model stays in eval mode,
        # CuDNN RNN backward can fail on CUDA, so route these loss decodes through the non-CuDNN path.
        with torch.backends.cudnn.flags(enabled=False):
            pred_audio_bct = decode_latent_to_audio(
                pred_latent_raw,
                encodec_model,
                target_pca_basis=resolved_target_pca_basis,
            )
            with torch.no_grad():
                target_audio_bct = decode_latent_to_audio(
                    _target_codec_latent(),
                    encodec_model,
                )
        max_num_samples = int(min(pred_audio_bct.shape[-1], target_audio_bct.shape[-1]))
        valid_num_samples_b = resolve_valid_audio_num_samples(
            prepared["duration_sec"],
            sample_rate=int(audio_sample_rate or resolve_encodec_sample_rate(encodec_model)),
            max_num_samples=int(max_num_samples),
        )
        pred_audio_eff = pred_audio_bct[..., : int(max_num_samples)]
        target_audio_eff = target_audio_bct[..., : int(max_num_samples)]
        audio_wave_l1 = masked_audio_l1_per_example(
            pred_audio_eff,
            target_audio_eff,
            valid_num_samples_b,
        ).mean()
        audio_mrstft = mrstft_logmag_l1_per_example(
            pred_audio_eff,
            target_audio_eff,
            valid_num_samples_b,
            resolutions=audio_mrstft_resolutions,
        ).mean()
        loss = loss + (float(audio_wave_l1_weight) * audio_wave_l1) + (float(audio_mrstft_weight) * audio_mrstft)

    with torch.no_grad():
        per_tok = ((x0_hat - target) ** 2).mean(dim=-1)
        per_ex = []
        for idx in range(per_tok.shape[0]):
            per_ex.append(per_tok[idx][target_mask[idx]].mean())
        x0_mse_median = torch.stack(per_ex).median()

    return {
        "loss": loss,
        "diffusion_loss": diffusion_loss,
        "audio_wave_l1": audio_wave_l1,
        "audio_mrstft": audio_mrstft,
        "x0_loss": x0_loss,
        "quant_embed_mse": quant_embed_mse,
        "rvq_ce": rvq_ce,
        "onset_weighted_x0": onset_weighted_x0,
        "x0_mse_median": x0_mse_median,
        "t": t,
    }


@torch.no_grad()
def sample_ddpm(
    model: ConditionalDiffusionTransformer,
    diffusion: GaussianDiffusion1D,
    batch: Mapping[str, Any],
    device: torch.device,
    x0_clip_norm: float | None = DEFAULT_SAMPLE_X0_CLIP_NORM,
    sample_idx: int | None = None,
    start_noise: torch.Tensor | None = None,
    step_noises: Mapping[int, torch.Tensor] | None = None,
    sample_seed: int | None = None,
    use_bpm_inference_geometry: bool = False,
    inference_num_beats: int = DEFAULT_INFERENCE_NUM_BEATS,
    target_token_rate_hz: float = DEFAULT_TARGET_TOKEN_RATE_HZ,
    inference_geometry: Mapping[str, Any] | None = None,
):
    prepared = _prepare_batch_tensors(
        batch,
        device,
        require_target=not bool(use_bpm_inference_geometry),
        require_timing=not bool(use_bpm_inference_geometry),
    )
    if sample_idx is not None:
        grid = prepared["grid"]
        if grid is None:
            raise ValueError("prepared batch is missing grid")
        batch_size = int(grid.shape[0])
        if not (0 <= int(sample_idx) < int(batch_size)):
            raise IndexError(f"sample_idx={sample_idx} out of range for batch size={int(batch_size)}")
        prepared = _slice_prepared_batch(prepared, int(sample_idx))
    if inference_geometry is None:
        geometry = resolve_inference_geometry(
            prepared,
            use_bpm_inference_geometry=bool(use_bpm_inference_geometry),
            inference_num_beats=int(inference_num_beats),
            target_token_rate_hz=float(target_token_rate_hz),
        )
    else:
        geometry = _prepare_geometry_tensors(inference_geometry, device=device)
        if sample_idx is not None:
            geometry = _slice_inference_geometry(geometry, int(sample_idx))
    target_mask = geometry["target_valid_mask_bt"]
    grid = prepared["grid"]
    grid_ids = prepared["grid_ids"]
    grid_valid_mask = prepared["grid_valid_mask"]
    if grid is None or grid_valid_mask is None:
        raise ValueError("prepared batch is missing grid or grid_valid_mask")
    cond_btd, cond_valid_mask_bt = model.encode_conditioning(
        grid=grid,
        grid_ids=grid_ids,
        grid_times_sec=geometry["grid_times_sec"],
        token_times_sec=geometry["token_times_sec"],
        target_valid_mask_bt=target_mask,
        grid_valid_mask_bt=grid_valid_mask,
    )

    batch_size = int(target_mask.shape[0])
    target_len = int(target_mask.shape[1])
    latent_dim = int(model.cfg.x_dim)
    noise_generator = None
    if sample_seed is not None:
        noise_generator = torch.Generator(device=device)
        noise_generator.manual_seed(int(sample_seed))
    if start_noise is None:
        x = torch.randn(
            batch_size,
            target_len,
            latent_dim,
            device=device,
            generator=noise_generator,
        )
    else:
        x = torch.as_tensor(start_noise, dtype=torch.float32, device=device).clone()
        expected_shape = (batch_size, target_len, latent_dim)
        if tuple(x.shape) != expected_shape:
            raise ValueError(f"start_noise must have shape {expected_shape}, got {tuple(x.shape)}")
    x = apply_seq_mask(x, target_mask)

    for step in reversed(range(diffusion.num_steps)):
        t = torch.full((batch_size,), step, device=device, dtype=torch.long)
        eps = model(
            x_t=x,
            t=t,
            target_valid_mask_bt=target_mask,
            token_times_sec=geometry["token_times_sec"],
            cond_btd=cond_btd,
            cond_valid_mask_bt=cond_valid_mask_bt,
        )
        x0_hat = diffusion.predict_x0_from_eps(x, t, eps)
        if x0_clip_norm is not None:
            x0_hat = x0_hat.clamp(min=-float(x0_clip_norm), max=float(x0_clip_norm))
        x0_hat = apply_seq_mask(x0_hat, target_mask)
        mean = diffusion.posterior_mean_from_x0(x, t, x0_hat)

        if step > 0:
            if step_noises is None or int(step) not in step_noises:
                z = torch.randn(
                    tuple(x.shape),
                    dtype=x.dtype,
                    device=x.device,
                    generator=noise_generator,
                )
            else:
                z = torch.as_tensor(step_noises[int(step)], dtype=torch.float32, device=device).clone()
                if tuple(z.shape) != tuple(x.shape):
                    raise ValueError(f"step_noises[{step}] must have shape {tuple(x.shape)}, got {tuple(z.shape)}")
            var = diffusion.posterior_variance[t].view(-1, 1, 1)
            x = mean + torch.sqrt(var) * z
        else:
            x = mean
        x = apply_seq_mask(x, target_mask)

    return x


def _plot_matrix(
    ax: Any,
    matrix_td: torch.Tensor,
    *,
    title: str,
    token_times_sec_t: torch.Tensor | None = None,
    vabs: float | None = None,
    ylabel: str = "latent dim",
    cmap: str = "coolwarm",
    vmin: float | None = None,
    vmax: float | None = None,
    transpose: bool = True,
) -> None:
    matrix = torch.as_tensor(matrix_td, dtype=torch.float32).detach().cpu()
    image_data = matrix.T.numpy() if bool(transpose) else matrix.numpy()
    extent = None
    if token_times_sec_t is not None:
        times = torch.as_tensor(token_times_sec_t, dtype=torch.float32).detach().cpu().view(-1).numpy()
        if int(times.shape[0]) == int(image_data.shape[1]) and int(times.shape[0]) > 0:
            lo = float(times[0])
            hi = float(times[-1]) if int(times.shape[0]) > 1 else float(times[0] + 1.0)
            extent = (lo, hi, -0.5, float(image_data.shape[0]) - 0.5)
    image = ax.imshow(
        image_data,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap=str(cmap),
        vmin=float(vmin) if vmin is not None else (-float(vabs) if vabs is not None else None),
        vmax=float(vmax) if vmax is not None else (float(vabs) if vabs is not None else None),
        extent=extent,
    )
    ax.set_title(title)
    ax.set_xlabel("time (seconds)" if extent is not None else "time")
    ax.set_ylabel(str(ylabel))
    plt.colorbar(image, ax=ax, fraction=0.02, pad=0.01)


def save_prediction_diagnostic_plot(
    out_path: str | os.PathLike[str],
    *,
    input_features_td: torch.Tensor | None,
    pred_latent_td: torch.Tensor,
    target_latent_td: torch.Tensor,
    target_codes_ct: torch.Tensor | None = None,
    pred_codes_ct: torch.Tensor | None = None,
    pred_target_token_acc: float | None = None,
    token_times_sec_t: torch.Tensor | None = None,
) -> None:
    input_features = None if input_features_td is None else torch.as_tensor(input_features_td, dtype=torch.float32)
    pred = torch.as_tensor(pred_latent_td, dtype=torch.float32)
    target = torch.as_tensor(target_latent_td, dtype=torch.float32)
    target_codes = None if target_codes_ct is None else torch.as_tensor(target_codes_ct, dtype=torch.float32)
    pred_codes = None if pred_codes_ct is None else torch.as_tensor(pred_codes_ct, dtype=torch.float32)
    diff = pred - target

    target_vabs = max(float(target.abs().max().item()) if int(target.numel()) > 0 else 0.0, 1.0e-6)
    pred_vabs = max(float(pred.abs().max().item()) if int(pred.numel()) > 0 else 0.0, 1.0e-6)
    diff_vabs = max(float(diff.abs().max().item()) if int(diff.numel()) > 0 else 0.0, 1.0e-6)
    has_codes = target_codes is not None and pred_codes is not None
    num_latent_rows = 4 if input_features is not None else 3
    total_rows = num_latent_rows + (1 if has_codes else 0)
    fig = plt.figure(figsize=(18, 3.5 * total_rows), constrained_layout=True)
    grid = fig.add_gridspec(total_rows, 2)

    row_idx = 0
    if input_features is not None:
        input_vabs = max(float(input_features.abs().max().item()), 1.0e-6)
        _plot_matrix(
            fig.add_subplot(grid[row_idx, :]),
            input_features,
            title=f"Conditioning input [T,{int(input_features.shape[-1])}]",
            token_times_sec_t=token_times_sec_t,
            vabs=input_vabs,
            ylabel="feature dim",
        )
        row_idx += 1

    _plot_matrix(
        fig.add_subplot(grid[row_idx, :]),
        target,
        title=f"Target latent [T,128] | absmax={target_vabs:.2f}",
        token_times_sec_t=token_times_sec_t,
        vabs=target_vabs,
    )
    _plot_matrix(
        fig.add_subplot(grid[row_idx + 1, :]),
        pred,
        title=f"Predicted latent [T,128] | absmax={pred_vabs:.2f}",
        token_times_sec_t=token_times_sec_t,
        vabs=pred_vabs,
    )
    _plot_matrix(
        fig.add_subplot(grid[row_idx + 2, :]),
        diff,
        title=f"Prediction error [T,128] | absmax={diff_vabs:.2f}",
        token_times_sec_t=token_times_sec_t,
        vabs=diff_vabs,
    )
    row_idx += 3

    if has_codes:
        target_ax = fig.add_subplot(grid[row_idx, 0])
        pred_ax = fig.add_subplot(grid[row_idx, 1])
        codebook_max = max(
            int(target_codes.max().item()) if int(target_codes.numel()) > 0 else 0,
            int(pred_codes.max().item()) if int(pred_codes.numel()) > 0 else 0,
        )
        _plot_matrix(
            target_ax,
            target_codes,
            title="Target requantized codes [C,T]",
            token_times_sec_t=token_times_sec_t,
            ylabel="codebook",
            cmap="tab20",
            vmin=-1.0,
            vmax=float(max(1, codebook_max)),
            transpose=False,
        )
        token_acc_text = "n/a" if pred_target_token_acc is None else f"{float(pred_target_token_acc):.4f}"
        _plot_matrix(
            pred_ax,
            pred_codes,
            title=f"Predicted requantized codes [C,T] | target_acc={token_acc_text}",
            token_times_sec_t=token_times_sec_t,
            ylabel="codebook",
            cmap="tab20",
            vmin=-1.0,
            vmax=float(max(1, codebook_max)),
            transpose=False,
        )

    out_path_str = os.fspath(out_path)
    out_dir = os.path.dirname(out_path_str)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path_str, dpi=160)
    plt.close(fig)


@torch.no_grad()
def save_eval_plot_multi_t(
    model,
    diffusion,
    batch,
    device,
    epoch,
    out_dir="eval_plots",
    sample_idx=0,
    t_values=(100, 300, 600),
    fixed_noises=None,
    target_mean=None,
    target_std=None,
    x0_clip_norm: float | None = DEFAULT_SAMPLE_X0_CLIP_NORM,
    use_bpm_training_geometry: bool = False,
    bpm_geometry_num_beats: int = DEFAULT_INFERENCE_NUM_BEATS,
):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    prepared = _prepare_batch_tensors(batch, device)
    if bool(use_bpm_training_geometry):
        prepared = apply_bpm_training_geometry_to_prepared_batch(
            prepared,
            num_beats=int(bpm_geometry_num_beats),
        )
    batch_size = int(prepared["target_btd"].shape[0])
    if not (0 <= int(sample_idx) < int(batch_size)):
        raise IndexError(f"sample_idx={sample_idx} out of range for batch size={int(batch_size)}")

    single = _slice_prepared_batch(prepared, sample_idx)
    target_i = single["target_btd"]
    target_mask_i = single["target_valid_mask_bt"]
    target_norm_i = normalize_latent(target_i, target_mean, target_std)
    cond_i, cond_mask_i = model.encode_conditioning(
        grid=single["grid"],
        grid_ids=single["grid_ids"],
        grid_times_sec=single["grid_times_sec"],
        token_times_sec=single["token_times_sec"],
        target_valid_mask_bt=target_mask_i,
        grid_valid_mask_bt=single["grid_valid_mask"],
    )

    valid_len = int(target_mask_i[0].sum().item())
    target_np = target_i[0, :valid_len].detach().cpu().T.numpy()

    nrows = 1 + len(t_values)
    fig, axes = plt.subplots(nrows, 1, figsize=(14, 3 * nrows), squeeze=False)
    ax = axes[0, 0]
    im = ax.imshow(target_np, aspect="auto", origin="lower")
    ax.set_title(f"Epoch {epoch} - Target")
    ax.set_ylabel("latent dim")
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)

    for row, t_eval in enumerate(t_values, start=1):
        if not (0 <= int(t_eval) < int(diffusion.num_steps)):
            raise ValueError(
                f"t_eval={t_eval} out of range for diffusion.num_steps={int(diffusion.num_steps)}; "
                f"expected values in [0, {int(diffusion.num_steps) - 1}]"
            )
        if fixed_noises is not None and t_eval in fixed_noises:
            noise = fixed_noises[t_eval].to(device)
        else:
            noise = torch.randn_like(target_i)
        if noise.shape != target_i.shape:
            raise ValueError(f"Noise shape {noise.shape} != target shape {target_i.shape}")

        noise = noise * target_mask_i.unsqueeze(-1)
        t = torch.full((1,), t_eval, device=device, dtype=torch.long)
        x_t = diffusion.q_sample(target_norm_i, t, noise)
        x_t = x_t * target_mask_i.unsqueeze(-1)

        pred_eps = model(
            x_t=x_t,
            t=t,
            target_valid_mask_bt=target_mask_i,
            token_times_sec=single["token_times_sec"],
            cond_btd=cond_i,
            cond_valid_mask_bt=cond_mask_i,
        )
        x0_hat_norm = diffusion.predict_x0_from_eps(x_t, t, pred_eps)
        if x0_clip_norm is not None:
            x0_hat_norm = x0_hat_norm.clamp(min=-float(x0_clip_norm), max=float(x0_clip_norm))
        x0_hat_norm = x0_hat_norm * target_mask_i.unsqueeze(-1)
        x0_hat = denormalize_latent(x0_hat_norm, target_mean, target_std)
        x0_hat = x0_hat * target_mask_i.unsqueeze(-1)

        pred_np = x0_hat[0, :valid_len].detach().cpu().T.numpy()
        mse_val = ((x0_hat - target_i)[0, :valid_len].pow(2).mean()).item()

        ax = axes[row, 0]
        im = ax.imshow(pred_np, aspect="auto", origin="lower")
        ax.set_title(f"Predicted x0_hat at t={t_eval}  |  mse={mse_val:.4f}")
        ax.set_ylabel("latent dim")
        if row == nrows - 1:
            ax.set_xlabel("time frame")
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)

    plt.tight_layout()
    save_path = os.path.join(out_dir, f"epoch_{epoch:03d}_multi_t.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def decode_latent_to_audio(
    pred_latent_btd,
    encodec_model,
    *,
    target_pca_basis: Mapping[str, Any] | None = None,
):
    latent = reconstruct_latent_from_pca(
        torch.as_tensor(pred_latent_btd, dtype=torch.float32),
        target_pca_basis,
    )
    return decode_quantized_latent_to_audio(encodec_model, latent)


def resolve_encodec_sample_rate(encodec_model, default: int = 32000) -> int:
    return resolve_audio_codec_sample_rate(encodec_model, default=default)


def stitch_audio_segments_with_crossfade(
    audio_segments_ct: Sequence[torch.Tensor],
    *,
    crossfade_num_samples: int,
) -> torch.Tensor:
    if not audio_segments_ct:
        raise ValueError("expected at least one audio segment")
    segments = [torch.as_tensor(segment, dtype=torch.float32).contiguous() for segment in list(audio_segments_ct)]
    out = segments[0]
    if int(out.dim()) != 2:
        raise ValueError(f"expected audio segments with shape [C,T], got {tuple(out.shape)}")
    crossfade = int(max(0, int(crossfade_num_samples)))
    for segment in list(segments[1:]):
        if tuple(segment.shape[:-1]) != tuple(out.shape[:-1]):
            raise ValueError(
                f"audio segment channel dimensions must match, got {tuple(out.shape)} / {tuple(segment.shape)}"
            )
        overlap = int(min(crossfade, int(out.shape[-1]), int(segment.shape[-1])))
        if int(overlap) <= 0:
            out = torch.cat((out, segment), dim=-1).contiguous()
            continue
        fade_t = torch.linspace(0.0, 1.0, steps=int(overlap), device=out.device, dtype=out.dtype)
        fade_out = torch.cos(0.5 * math.pi * fade_t).view(1, -1)
        fade_in = torch.sin(0.5 * math.pi * fade_t).view(1, -1)
        blended = (out[..., -int(overlap) :] * fade_out) + (segment[..., : int(overlap)] * fade_in)
        out = torch.cat((out[..., :-int(overlap)], blended, segment[..., int(overlap) :]), dim=-1).contiguous()
    return out.contiguous()


def apply_beat_crossfade(
    audio_ct: torch.Tensor,
    beat_boundaries_sec_t: torch.Tensor,
    *,
    sample_rate: int,
    beat_crossfade_ms: float = DEFAULT_BEAT_CROSSFADE_MS,
) -> torch.Tensor:
    audio = torch.as_tensor(audio_ct, dtype=torch.float32).contiguous()
    if int(audio.dim()) != 2:
        raise ValueError(f"audio_ct must be [C,T], got {tuple(audio.shape)}")
    crossfade_num_samples = int(round((max(0.0, float(beat_crossfade_ms)) / 1000.0) * float(sample_rate)))
    if int(crossfade_num_samples) <= 0:
        return audio.contiguous()

    boundaries_sec = torch.as_tensor(
        beat_boundaries_sec_t,
        dtype=torch.float32,
        device=audio.device,
    ).view(-1)
    if int(boundaries_sec.numel()) < 2:
        return audio.contiguous()
    total_num_samples = int(audio.shape[-1])
    boundaries = torch.round(boundaries_sec * float(sample_rate)).to(dtype=torch.long)
    boundaries = boundaries.clamp(min=0, max=max(0, total_num_samples))
    boundaries[0] = 0
    boundaries[-1] = int(total_num_samples)
    if int(boundaries.numel()) == 2:
        return audio.contiguous()

    left_context = int(crossfade_num_samples // 2)
    right_context = int(crossfade_num_samples - left_context)
    segments: list[torch.Tensor] = []
    last_beat_idx = int(boundaries.numel()) - 2
    for beat_idx in range(int(boundaries.numel()) - 1):
        nominal_lo = int(boundaries[int(beat_idx)].item())
        nominal_hi = int(boundaries[int(beat_idx) + 1].item())
        seg_lo = int(nominal_lo if int(beat_idx) == 0 else max(0, nominal_lo - left_context))
        seg_hi = int(nominal_hi if int(beat_idx) == int(last_beat_idx) else min(total_num_samples, nominal_hi + right_context))
        if int(seg_hi) <= int(seg_lo):
            continue
        segments.append(audio[..., int(seg_lo) : int(seg_hi)].contiguous())
    if not segments:
        return audio.contiguous()

    smoothed = stitch_audio_segments_with_crossfade(
        segments,
        crossfade_num_samples=int(crossfade_num_samples),
    )
    if int(smoothed.shape[-1]) > int(total_num_samples):
        smoothed = smoothed[..., : int(total_num_samples)]
    elif int(smoothed.shape[-1]) < int(total_num_samples):
        smoothed = F.pad(smoothed, (0, int(total_num_samples) - int(smoothed.shape[-1])))
    return smoothed.contiguous()


def _code_accuracy_stats(pred_codes_bct: torch.Tensor, ref_codes_bct: torch.Tensor) -> dict[str, Any]:
    pred_codes = torch.as_tensor(pred_codes_bct, dtype=torch.long)
    ref_codes = torch.as_tensor(ref_codes_bct, dtype=torch.long)
    if int(pred_codes.dim()) != 3 or int(ref_codes.dim()) != 3:
        raise ValueError(f"expected [B,C,T] code tensors, got {tuple(pred_codes.shape)} / {tuple(ref_codes.shape)}")
    if tuple(pred_codes.shape[:2]) != tuple(ref_codes.shape[:2]):
        raise ValueError(
            f"pred_codes and ref_codes must match on batch/codebook dims, got {tuple(pred_codes.shape)} / {tuple(ref_codes.shape)}"
        )
    compared_num_frames = int(min(int(pred_codes.shape[-1]), int(ref_codes.shape[-1])))
    if int(compared_num_frames) <= 0:
        raise ValueError(
            f"pred_codes and ref_codes must have at least one overlapping frame, got {tuple(pred_codes.shape)} / {tuple(ref_codes.shape)}"
        )
    pred_codes = pred_codes[:, :, : int(compared_num_frames)]
    ref_codes = ref_codes[:, :, : int(compared_num_frames)]
    equal = pred_codes.eq(ref_codes)
    return {
        "token_acc": float(equal.float().mean().item()),
        "per_codebook_token_acc": [
            float(equal[:, int(codebook_idx), :].float().mean().item())
            for codebook_idx in range(int(equal.shape[1]))
        ],
        "exact_match": bool(equal.all().item()),
        "compared_num_frames": int(compared_num_frames),
        "pred_num_frames": int(pred_codes_bct.shape[-1]),
        "ref_num_frames": int(ref_codes_bct.shape[-1]),
        "shape_match": bool(tuple(pred_codes_bct.shape) == tuple(ref_codes_bct.shape)),
    }


@torch.no_grad()
def save_inference_wav(
    model,
    diffusion,
    encodec_model,
    batch,
    device,
    epoch,
    target_mean,
    target_std,
    sample_rate=None,
    out_dir="best_samples",
    sample_idx=0,
    start_noise=None,
    step_noises: Mapping[int, torch.Tensor] | None = None,
    x0_clip_norm: float | None = DEFAULT_SAMPLE_X0_CLIP_NORM,
    use_bpm_inference_geometry: bool = True,
    inference_num_beats: int = DEFAULT_INFERENCE_NUM_BEATS,
    target_token_rate_hz: float = DEFAULT_TARGET_TOKEN_RATE_HZ,
    beat_crossfade_ms: float = DEFAULT_BEAT_CROSSFADE_MS,
    target_pca_basis: Mapping[str, Any] | None = None,
):
    os.makedirs(out_dir, exist_ok=True)

    pred_latent = sample_ddpm(
        model=model,
        diffusion=diffusion,
        batch=batch,
        device=device,
        sample_idx=sample_idx,
        start_noise=start_noise,
        step_noises=step_noises,
        x0_clip_norm=x0_clip_norm,
        use_bpm_inference_geometry=bool(use_bpm_inference_geometry),
        inference_num_beats=int(inference_num_beats),
        target_token_rate_hz=float(target_token_rate_hz),
    )
    prepared = _prepare_batch_tensors(
        batch,
        device,
        require_target=False,
        require_timing=not bool(use_bpm_inference_geometry),
    )
    single = _slice_prepared_batch(prepared, int(sample_idx))
    geometry = resolve_inference_geometry(
        single,
        use_bpm_inference_geometry=bool(use_bpm_inference_geometry),
        inference_num_beats=int(inference_num_beats),
        target_token_rate_hz=float(target_token_rate_hz),
    )
    target_mask_i = geometry["target_valid_mask_bt"]
    pred_latent = denormalize_latent(pred_latent, target_mean, target_std)
    pred_latent = pred_latent * target_mask_i.unsqueeze(-1)
    resolved_target_pca_basis = _resolve_target_pca_basis(
        target_pca_basis,
        device=device,
        dtype=pred_latent.dtype,
    )

    audio = decode_latent_to_audio(
        pred_latent,
        encodec_model,
        target_pca_basis=resolved_target_pca_basis,
    )
    if audio.dim() == 3:
        wav = audio[0]
    elif audio.dim() == 2:
        wav = audio[0].unsqueeze(0)
    else:
        raise ValueError(f"Unexpected audio shape: {audio.shape}")

    write_sample_rate = int(sample_rate) if sample_rate is not None else resolve_encodec_sample_rate(encodec_model)
    if float(beat_crossfade_ms) > 0.0:
        wav = apply_beat_crossfade(
            wav,
            geometry["beat_boundaries_sec"][0],
            sample_rate=int(write_sample_rate),
            beat_crossfade_ms=float(beat_crossfade_ms),
        )
    wav = wav.detach().cpu()
    peak = wav.abs().max().clamp_min(1e-8)
    wav = 0.95 * wav / peak

    save_path = os.path.join(out_dir, f"best_epoch_{epoch:03d}.wav")
    torchaudio.save(save_path, wav, sample_rate=write_sample_rate)

    if "source_codes_bct" in batch and "target_btd" in batch:
        target_mask_single = geometry["target_valid_mask_bt"]
        cond_i, _ = model.encode_conditioning(
            grid=single["grid"],
            grid_ids=single["grid_ids"],
            grid_times_sec=geometry["grid_times_sec"],
            token_times_sec=geometry["token_times_sec"],
            target_valid_mask_bt=target_mask_single,
            grid_valid_mask_bt=single["grid_valid_mask"],
        )

        source_codes = torch.as_tensor(batch["source_codes_bct"][int(sample_idx) : int(sample_idx) + 1], dtype=torch.long, device=device)
        target_plot_ref = torch.as_tensor(batch["target_btd"][int(sample_idx) : int(sample_idx) + 1], dtype=torch.float32, device=device)
        target_ref = torch.as_tensor(
            batch.get("target_sum_btd", batch["target_btd"])[int(sample_idx) : int(sample_idx) + 1],
            dtype=torch.float32,
            device=device,
        )
        source_audio = decode_codes_to_audio_b1t(encodec_model, source_codes, device=device)
        target_direct_audio = decode_latent_to_audio(target_ref, encodec_model)
        target_requant_codes = requantize_latent_to_codes_bct(encodec_model, target_ref, device=device)
        pred_requant_codes = requantize_latent_to_codes_bct(
            encodec_model,
            pred_latent,
            device=device,
            target_pca_basis=resolved_target_pca_basis,
        )
        target_requant_audio = decode_codes_to_audio_b1t(encodec_model, target_requant_codes, device=device)
        pred_requant_audio = decode_codes_to_audio_b1t(encodec_model, pred_requant_codes, device=device)
        valid_len = int(
            min(
                int(target_mask_single[0].sum().item()),
                int(target_plot_ref.shape[1]),
                int(target_requant_codes.shape[-1]),
                int(pred_requant_codes.shape[-1]),
            )
        )

        save_prediction_diagnostic_plot(
            os.path.join(out_dir, f"best_epoch_{epoch:03d}_pred_latent.png"),
            input_features_td=cond_i[0, :valid_len],
            pred_latent_td=pred_latent[0, :valid_len],
            target_latent_td=target_plot_ref[0, :valid_len],
            target_codes_ct=target_requant_codes[0, :, :valid_len],
            pred_codes_ct=pred_requant_codes[0, :, :valid_len],
            pred_target_token_acc=float(
                pred_requant_codes[:, :, :valid_len].eq(target_requant_codes[:, :, :valid_len]).float().mean().item()
            ),
            token_times_sec_t=geometry["token_times_sec"][0, :valid_len],
        )

        for suffix, audio_tensor in (
            ("source_codes", source_audio),
            ("target_direct", target_direct_audio),
            ("target_requant", target_requant_audio),
            ("pred_requant", pred_requant_audio),
        ):
            wav_i = audio_tensor[0].detach().cpu()
            peak_i = wav_i.abs().max().clamp_min(1e-8)
            wav_i = 0.95 * wav_i / peak_i
            torchaudio.save(
                os.path.join(out_dir, f"best_epoch_{epoch:03d}_{suffix}.wav"),
                wav_i,
                sample_rate=write_sample_rate,
            )

        debug_payload = {
            "sample_idx": int(sample_idx),
            "write_sample_rate": int(write_sample_rate),
            "use_bpm_inference_geometry": bool(use_bpm_inference_geometry),
            "inference_num_beats": int(inference_num_beats),
            "target_token_rate_hz": float(target_token_rate_hz),
            "beat_crossfade_ms": float(beat_crossfade_ms),
            "pred_requant_vs_source": _code_accuracy_stats(pred_requant_codes, source_codes),
            "pred_requant_vs_target_requant": _code_accuracy_stats(pred_requant_codes, target_requant_codes),
            "target_requant_vs_source": _code_accuracy_stats(target_requant_codes, source_codes),
            "pred_direct_vs_pred_requant_audio_l1": float(
                (
                    decode_latent_to_audio(
                        pred_latent,
                        encodec_model,
                        target_pca_basis=resolved_target_pca_basis,
                    )
                    - pred_requant_audio
                ).abs().mean().item()
            ),
            "target_direct_vs_source_audio_l1": float((target_direct_audio - source_audio).abs().mean().item()),
            "target_requant_vs_source_audio_l1": float((target_requant_audio - source_audio).abs().mean().item()),
            "pred_requant_codes_shape": list(pred_requant_codes.shape),
            "target_requant_codes_shape": list(target_requant_codes.shape),
            "source_codes_shape": list(source_codes.shape),
        }
        import json

        with open(os.path.join(out_dir, f"best_epoch_{epoch:03d}_decode_debug.json"), "w", encoding="utf-8") as handle:
            json.dump(debug_payload, handle, indent=2, sort_keys=True)

    return save_path


def load_or_compute_target_normalization(cache_root: str, train_loader, *, device: torch.device, x_dim: int):
    stats_path = os.path.join(cache_root, "target_stats.pt")
    if os.path.exists(stats_path):
        payload = torch.load(stats_path, map_location="cpu", weights_only=False)
        mean = torch.as_tensor(payload["target_mean"], dtype=torch.float32, device=device).view(-1)
        std = torch.as_tensor(payload["target_std"], dtype=torch.float32, device=device).view(-1).clamp_min(1.0e-6)
        if int(mean.numel()) != int(x_dim) or int(std.numel()) != int(x_dim):
            raise RuntimeError(
                f"cached target stats under {stats_path} do not match x_dim={x_dim}: "
                f"mean={tuple(mean.shape)} std={tuple(std.shape)}"
            )
        print(f"loaded target stats from {stats_path}")
        return mean.contiguous(), std.contiguous()

    print("estimating target normalization from train split")
    mean, std = estimate_target_normalization(train_loader, device=device)
    torch.save(
        {
            "target_mean": mean.detach().cpu(),
            "target_std": std.detach().cpu(),
            "x_dim": int(x_dim),
        },
        stats_path,
    )
    print(f"saved target stats to {stats_path}")
    return mean.contiguous(), std.contiguous()
