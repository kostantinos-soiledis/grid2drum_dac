"""Vendored direct-PCA sequence regressor (inference only) for the HF Space.

Extracted from standalone_direct_pca_regressor.py: the seconds-aligned frontend,
its helper functions, and DirectPCASequenceRegressor + DirectRegressorConfig.
Training/dataset/CLI code is intentionally omitted. Do not edit here; re-vendor
from the source module if the model definition changes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def lengths_to_mask(lengths_b: torch.Tensor, *, max_len: int | None = None) -> torch.Tensor:
    lengths = torch.as_tensor(lengths_b, dtype=torch.long).view(-1)
    if int(lengths.numel()) <= 0:
        return torch.zeros((0, int(max_len or 0)), dtype=torch.bool, device=lengths.device)
    resolved_max = int(max_len) if max_len is not None else int(lengths.max().item())
    steps = torch.arange(resolved_max, device=lengths.device).view(1, -1)
    return steps < lengths.view(-1, 1)


def apply_seq_mask(x: torch.Tensor, mask_bt: torch.Tensor) -> torch.Tensor:
    return x * mask_bt.unsqueeze(-1).to(dtype=x.dtype)


def normalize_latent(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean.view(1, 1, -1)) / std.clamp_min(1.0e-8).view(1, 1, -1)


def denormalize_latent(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x * std.clamp_min(1.0e-8).view(1, 1, -1)) + mean.view(1, 1, -1)


def sinusoidal_time_positions(times_sec_bt: torch.Tensor, dim: int, *, rate_hz: float) -> torch.Tensor:
    times = torch.as_tensor(times_sec_bt, dtype=torch.float32)
    if int(times.dim()) != 2:
        raise ValueError(f"times_sec_bt must be [B,T], got {tuple(times.shape)}")
    position = times.unsqueeze(-1) * float(max(1.0e-6, float(rate_hz)))
    half = dim // 2
    div = torch.exp(
        torch.arange(0, half, device=times.device, dtype=torch.float32) * (-math.log(10000.0) / max(1, half))
    ).view(1, 1, -1)
    pe = torch.zeros((*times.shape, int(dim)), device=times.device, dtype=torch.float32)
    pe[:, :, 0:half] = torch.sin(position * div)
    pe[:, :, half : 2 * half] = torch.cos(position * div)
    return pe


def sinusoidal_index_positions(length: int, dim: int, device: torch.device) -> torch.Tensor:
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    half = dim // 2
    div = torch.exp(torch.arange(0, half, device=device, dtype=torch.float32) * (-math.log(10000.0) / max(1, half)))
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0:half] = torch.sin(pos * div)
    pe[:, half : 2 * half] = torch.cos(pos * div)
    return pe.unsqueeze(0)


# ------------------------- copied seconds frontend -------------------------


VALID_VARIANTS = ("linear", "cnn", "bilstm", "hybrid")
VALID_PADDING_MODES = ("zeros", "reflect", "replicate")
VALID_FRONTEND_OUTPUTS = ("base", "feat")
DEFAULT_FRONTEND_OUTPUT = "feat"
DEFAULT_SAMPLE_STEP_SECONDS = 1.0 / 250.0
DEFAULT_CLASS_LOCAL_DIM = 8


def resolve_frontend_output_dim(*, embed_dim: int, output_kind: str) -> int:
    output_eff = str(output_kind or DEFAULT_FRONTEND_OUTPUT).strip().lower()
    if output_eff not in set(VALID_FRONTEND_OUTPUTS):
        raise ValueError(f"unsupported output_kind={output_kind!r}")
    return int(embed_dim)


def infer_nearest_feature_mask_source(
    *,
    input_dim_source: int,
    class_id_vocab_sizes: Sequence[int] = (),
    source_feature_names: Optional[Sequence[str]] = None,
) -> torch.Tensor:
    feature_dim = int(max(0, int(input_dim_source)))
    mask = torch.zeros((feature_dim,), dtype=torch.bool)
    names = list(source_feature_names or ())
    if names:
        for idx, name in enumerate(names[:feature_dim]):
            if str(name).endswith("_hit") or str(name).endswith("_onset_vel") or str(name).endswith("_onset_count"):
                mask[int(idx)] = True
        if bool(mask.any().item()):
            return mask
    class_count = int(len(list(class_id_vocab_sizes or ())))
    if class_count > 0 and feature_dim == 3 * class_count:
        mask[1::3] = True
        mask[2::3] = True
    elif class_count > 0 and feature_dim == 2 * class_count:
        mask[0::2] = True
    return mask


def _default_class_names(
    *,
    source_feature_names: Sequence[str] = (),
    class_id_vocab_sizes: Sequence[int] = (),
) -> list[str]:
    names = [str(x) for x in list(source_feature_names or ())]
    out: list[str] = []
    for name in names:
        prefix = str(name).split("_", 1)[0].strip()
        if prefix and prefix not in out:
            out.append(prefix)
    class_count = int(len(list(class_id_vocab_sizes or ())))
    if class_count > 0 and len(out) == class_count:
        return out
    if out:
        return out
    return [f"class_{idx}" for idx in range(class_count)]


def resolve_frontend_input_feature_names(
    *,
    input_dim_source: int,
    class_id_vocab_sizes: Sequence[int] = (),
    source_feature_names: Sequence[str] = (),
    class_names: Sequence[str] = (),
    class_local_fusion: bool = False,
    class_local_dim: int = DEFAULT_CLASS_LOCAL_DIM,
) -> list[str]:
    source_names = [str(x) for x in list(source_feature_names or ())]
    vocab_sizes = [int(max(0, int(x))) for x in list(class_id_vocab_sizes or ())]
    class_names_eff = [str(x) for x in list(class_names or ())] or _default_class_names(
        source_feature_names=source_names,
        class_id_vocab_sizes=vocab_sizes,
    )
    if bool(class_local_fusion):
        dim = int(max(1, int(class_local_dim)))
        return [f"{class_name}_local{slot}" for class_name in class_names_eff for slot in range(dim)]
    id_names: list[str] = []
    for class_idx, vocab_size in enumerate(vocab_sizes):
        if vocab_size <= 1:
            continue
        class_name = class_names_eff[class_idx] if class_idx < len(class_names_eff) else f"class_{class_idx}"
        id_names.extend([f"{class_name}_id{slot}" for slot in range(vocab_size)])
    if source_names:
        return list(source_names[: int(input_dim_source)]) + id_names
    return [f"feat_{idx}" for idx in range(int(input_dim_source))] + id_names


def _resolve_source_groups_by_class(
    *,
    input_dim_source: int,
    source_feature_names: Sequence[str],
    class_names: Sequence[str],
) -> list[list[int]]:
    names = [str(x) for x in list(source_feature_names or ())]
    class_names_eff = [str(x) for x in list(class_names or ())]
    if names and class_names_eff:
        groups: list[list[int]] = []
        for class_name in class_names_eff:
            prefix = f"{class_name}_"
            group = [idx for idx, name in enumerate(names[: int(input_dim_source)]) if str(name).startswith(prefix)]
            groups.append(group)
        if all(bool(group) for group in groups) and sum(len(group) for group in groups) == int(input_dim_source):
            return groups
    class_count = int(len(class_names_eff))
    if class_count <= 0:
        return [list(range(int(input_dim_source)))]
    base = int(input_dim_source) // class_count
    rem = int(input_dim_source) % class_count
    groups = []
    cursor = 0
    for idx in range(class_count):
        size = base + (1 if idx < rem else 0)
        groups.append(list(range(cursor, cursor + size)))
        cursor += size
    return groups


def _resolve_id_groups_by_class(*, input_dim_source: int, class_id_vocab_sizes: Sequence[int]) -> list[list[int]]:
    offset = int(input_dim_source)
    groups: list[list[int]] = []
    for vocab_size in list(class_id_vocab_sizes or ()):
        vocab = int(max(0, int(vocab_size)))
        size = vocab if vocab > 1 else 0
        groups.append(list(range(offset, offset + size)))
        offset += size
    return groups


def _lengths_from_mask(
    mask_bt: Optional[torch.Tensor],
    *,
    fallback: int,
    batch_size: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if mask_bt is None:
        return torch.full((int(batch_size),), int(fallback), dtype=torch.long, device=device)
    if int(mask_bt.dim()) != 2:
        raise ValueError(f"mask_bt must be [B,T], got {tuple(mask_bt.shape)}")
    return mask_bt.to(device=device, dtype=torch.long).sum(dim=1).clamp_min(1)


def _infer_step_seconds(
    *,
    grid_times_sec_bt: torch.Tensor,
    grid_valid_mask_bt: Optional[torch.Tensor],
    default_step_seconds: float,
) -> torch.Tensor:
    if int(grid_times_sec_bt.dim()) != 2:
        raise ValueError(f"grid_times_sec_bt must be [B,T], got {tuple(grid_times_sec_bt.shape)}")
    batch_size = int(grid_times_sec_bt.shape[0])
    lengths = _lengths_from_mask(
        grid_valid_mask_bt,
        fallback=int(grid_times_sec_bt.shape[1]),
        batch_size=batch_size,
        device=grid_times_sec_bt.device,
    )
    out = torch.full(
        (batch_size,),
        float(max(1.0e-6, float(default_step_seconds))),
        dtype=grid_times_sec_bt.dtype,
        device=grid_times_sec_bt.device,
    )
    for batch_idx in range(batch_size):
        length = int(lengths[batch_idx].item())
        if length <= 1:
            continue
        diffs = grid_times_sec_bt[batch_idx, 1:length].to(dtype=torch.float32) - grid_times_sec_bt[
            batch_idx, : length - 1
        ].to(dtype=torch.float32)
        diffs = diffs[torch.isfinite(diffs) & diffs.gt(1.0e-6)]
        if int(diffs.numel()) > 0:
            out[batch_idx] = diffs.median()
    return out


def _fill_invalid_token_times(token_times_sec_bt: torch.Tensor, valid_mask_bt: Optional[torch.Tensor]) -> torch.Tensor:
    if valid_mask_bt is None:
        return token_times_sec_bt.to(dtype=torch.float32)
    if token_times_sec_bt.shape != valid_mask_bt.shape:
        raise ValueError(
            f"token_times_sec_bt and valid_mask_bt must match, got {tuple(token_times_sec_bt.shape)} / {tuple(valid_mask_bt.shape)}"
        )
    token_times = token_times_sec_bt.to(dtype=torch.float32)
    valid_bt = valid_mask_bt.to(device=token_times.device, dtype=torch.bool)
    if bool(torch.all(valid_bt)):
        return token_times
    lengths = valid_bt.sum(dim=1, dtype=torch.long).clamp_min(1)
    last_idx = (lengths - 1).view(int(token_times.shape[0]), 1)
    last_vals = token_times.gather(dim=1, index=last_idx).expand(-1, int(token_times.shape[1]))
    return torch.where(valid_bt, token_times, last_vals)


def _sample_grid_linear_single(
    *,
    grid_ft: torch.Tensor,
    grid_times_t: torch.Tensor,
    sample_times_tw: torch.Tensor,
) -> torch.Tensor:
    feature_dim = int(grid_ft.shape[0])
    length = int(grid_ft.shape[1])
    if length <= 0:
        return torch.zeros((int(sample_times_tw.shape[0]), feature_dim, int(sample_times_tw.shape[1])), device=grid_ft.device)
    flat = sample_times_tw.reshape(-1).to(device=grid_times_t.device, dtype=grid_times_t.dtype)
    flat = flat.clamp(min=float(grid_times_t[0].item()), max=float(grid_times_t[length - 1].item()))
    idx1 = torch.searchsorted(grid_times_t, flat, right=False).clamp(min=0, max=length - 1)
    idx0 = (idx1 - 1).clamp(min=0, max=length - 1)
    t0 = grid_times_t.index_select(0, idx0)
    t1 = grid_times_t.index_select(0, idx1)
    x0 = grid_ft.index_select(1, idx0)
    x1 = grid_ft.index_select(1, idx1)
    denom = t1 - t0
    weight = torch.where(denom.abs() > 1.0e-8, (flat - t0) / denom, torch.zeros_like(flat)).to(dtype=grid_ft.dtype)
    interp = x0 + ((x1 - x0) * weight.unsqueeze(0))
    return interp.view(feature_dim, int(sample_times_tw.shape[0]), int(sample_times_tw.shape[1])).permute(1, 0, 2).contiguous()


def _sample_grid_nearest_value_single(
    *,
    grid_ft: torch.Tensor,
    grid_times_t: torch.Tensor,
    sample_times_tw: torch.Tensor,
) -> torch.Tensor:
    feature_dim = int(grid_ft.shape[0])
    length = int(grid_ft.shape[1])
    if length <= 0:
        return torch.zeros((int(sample_times_tw.shape[0]), feature_dim, int(sample_times_tw.shape[1])), device=grid_ft.device)
    flat = sample_times_tw.reshape(-1).to(device=grid_times_t.device, dtype=grid_times_t.dtype)
    flat = flat.clamp(min=float(grid_times_t[0].item()), max=float(grid_times_t[length - 1].item()))
    idx_hi = torch.searchsorted(grid_times_t, flat, right=False).clamp(min=0, max=length - 1)
    idx_lo = (idx_hi - 1).clamp(min=0, max=length - 1)
    t_lo = grid_times_t.index_select(0, idx_lo)
    t_hi = grid_times_t.index_select(0, idx_hi)
    nearest_idx = torch.where((flat - t_lo).abs() > (t_hi - flat).abs(), idx_hi, idx_lo)
    sampled = grid_ft.index_select(1, nearest_idx)
    return sampled.view(feature_dim, int(sample_times_tw.shape[0]), int(sample_times_tw.shape[1])).permute(1, 0, 2).contiguous()


def sample_grid_windows_in_seconds(
    *,
    grid_bft: torch.Tensor,
    grid_times_sec_bt: torch.Tensor,
    token_times_sec_bt: torch.Tensor,
    window_radius: int,
    step_seconds: float = 0.0,
    grid_valid_mask_bt: Optional[torch.Tensor] = None,
    valid_mask_bt: Optional[torch.Tensor] = None,
    nearest_feature_mask_f: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if int(grid_bft.dim()) != 3:
        raise ValueError(f"grid_bft must be [B,F,Tg], got {tuple(grid_bft.shape)}")
    batch_size = int(grid_bft.shape[0])
    nearest_mask = None
    if nearest_feature_mask_f is not None:
        nearest_mask = nearest_feature_mask_f.to(device=grid_bft.device, dtype=torch.bool)
    token_times = _fill_invalid_token_times(token_times_sec_bt, valid_mask_bt)
    if float(step_seconds) > 0.0:
        step_b = torch.full((batch_size,), float(step_seconds), dtype=grid_bft.dtype, device=grid_bft.device)
    else:
        step_b = _infer_step_seconds(
            grid_times_sec_bt=grid_times_sec_bt.to(device=grid_bft.device, dtype=torch.float32),
            grid_valid_mask_bt=grid_valid_mask_bt,
            default_step_seconds=DEFAULT_SAMPLE_STEP_SECONDS,
        ).to(dtype=grid_bft.dtype, device=grid_bft.device)
    radius = int(max(0, int(window_radius)))
    offsets = torch.arange(-radius, radius + 1, device=grid_bft.device, dtype=grid_bft.dtype)
    sample_times = token_times.to(dtype=grid_bft.dtype)[:, :, None] + (step_b[:, None, None] * offsets[None, None, :])
    grid_lengths = _lengths_from_mask(grid_valid_mask_bt, fallback=int(grid_bft.shape[2]), batch_size=batch_size, device=grid_bft.device)
    outputs: list[torch.Tensor] = []
    for batch_idx in range(batch_size):
        grid_len = int(grid_lengths[batch_idx].item())
        valid_grid_ft = grid_bft[batch_idx, :, :grid_len].to(dtype=torch.float32)
        valid_times_t = grid_times_sec_bt[batch_idx, :grid_len].to(device=grid_bft.device, dtype=torch.float32)
        sampled_tfw = _sample_grid_linear_single(
            grid_ft=valid_grid_ft,
            grid_times_t=valid_times_t,
            sample_times_tw=sample_times[batch_idx].to(dtype=torch.float32),
        )
        if nearest_mask is not None and bool(nearest_mask.any().item()):
            sampled_nearest = _sample_grid_nearest_value_single(
                grid_ft=valid_grid_ft,
                grid_times_t=valid_times_t,
                sample_times_tw=sample_times[batch_idx].to(dtype=torch.float32),
            )
            sampled_tfw[:, nearest_mask, :] = sampled_nearest[:, nearest_mask, :]
        outputs.append(sampled_tfw.unsqueeze(0))
    return torch.cat(outputs, dim=0)


def _sample_grid_nearest_single(
    *,
    grid_ct: torch.Tensor,
    grid_times_t: torch.Tensor,
    sample_times_tw: torch.Tensor,
) -> torch.Tensor:
    class_dim = int(grid_ct.shape[0])
    length = int(grid_ct.shape[1])
    if length <= 0:
        return torch.full((int(sample_times_tw.shape[0]), class_dim, int(sample_times_tw.shape[1])), -1, dtype=grid_ct.dtype, device=grid_ct.device)
    flat = sample_times_tw.reshape(-1).to(device=grid_times_t.device, dtype=grid_times_t.dtype)
    flat = flat.clamp(min=float(grid_times_t[0].item()), max=float(grid_times_t[length - 1].item()))
    idx_hi = torch.searchsorted(grid_times_t, flat, right=False).clamp(min=0, max=length - 1)
    idx_lo = (idx_hi - 1).clamp(min=0, max=length - 1)
    t_lo = grid_times_t.index_select(0, idx_lo)
    t_hi = grid_times_t.index_select(0, idx_hi)
    nearest_idx = torch.where((flat - t_lo).abs() > (t_hi - flat).abs(), idx_hi, idx_lo)
    sampled = grid_ct.index_select(1, nearest_idx)
    return sampled.view(class_dim, int(sample_times_tw.shape[0]), int(sample_times_tw.shape[1])).permute(1, 0, 2).contiguous()


def sample_grid_id_windows_in_seconds(
    *,
    grid_ids_bct: torch.Tensor,
    grid_times_sec_bt: torch.Tensor,
    token_times_sec_bt: torch.Tensor,
    window_radius: int,
    step_seconds: float = 0.0,
    grid_valid_mask_bt: Optional[torch.Tensor] = None,
    valid_mask_bt: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    batch_size = int(grid_ids_bct.shape[0])
    token_times = _fill_invalid_token_times(token_times_sec_bt, valid_mask_bt)
    if float(step_seconds) > 0.0:
        step_b = torch.full((batch_size,), float(step_seconds), dtype=token_times.dtype, device=token_times.device)
    else:
        step_b = _infer_step_seconds(
            grid_times_sec_bt=grid_times_sec_bt.to(device=token_times.device, dtype=torch.float32),
            grid_valid_mask_bt=grid_valid_mask_bt,
            default_step_seconds=DEFAULT_SAMPLE_STEP_SECONDS,
        ).to(dtype=token_times.dtype, device=token_times.device)
    radius = int(max(0, int(window_radius)))
    offsets = torch.arange(-radius, radius + 1, device=token_times.device, dtype=token_times.dtype)
    sample_times = token_times[:, :, None] + (step_b[:, None, None] * offsets[None, None, :])
    grid_lengths = _lengths_from_mask(grid_valid_mask_bt, fallback=int(grid_ids_bct.shape[2]), batch_size=batch_size, device=grid_ids_bct.device)
    outputs: list[torch.Tensor] = []
    for batch_idx in range(batch_size):
        grid_len = int(grid_lengths[batch_idx].item())
        sampled = _sample_grid_nearest_single(
            grid_ct=grid_ids_bct[batch_idx, :, :grid_len].to(dtype=torch.long),
            grid_times_t=grid_times_sec_bt[batch_idx, :grid_len].to(device=grid_ids_bct.device, dtype=torch.float32),
            sample_times_tw=sample_times[batch_idx].to(dtype=torch.float32),
        )
        outputs.append(sampled.unsqueeze(0))
    return torch.cat(outputs, dim=0)


def expand_sampled_id_windows_onehot(
    sampled_ids_btcw: Optional[torch.Tensor],
    *,
    class_id_vocab_sizes: Sequence[int],
) -> Optional[torch.Tensor]:
    if sampled_ids_btcw is None:
        return None
    parts: list[torch.Tensor] = []
    for class_idx, vocab_size in enumerate(int(max(0, int(x))) for x in list(class_id_vocab_sizes or ())):
        if vocab_size <= 1:
            continue
        ids_btw = sampled_ids_btcw[:, :, class_idx, :].to(dtype=torch.long)
        safe = ids_btw.clamp(min=0, max=vocab_size - 1)
        onehot = F.one_hot(safe, num_classes=vocab_size).to(dtype=torch.float32)
        valid = ids_btw.ge(0) & ids_btw.lt(vocab_size)
        onehot = onehot * valid.unsqueeze(-1).to(dtype=onehot.dtype)
        parts.append(onehot.permute(0, 1, 3, 2).contiguous())
    if not parts:
        return None
    return torch.cat(parts, dim=2)


class SamePadConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True,
        padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        mode = str(padding_mode).strip().lower()
        if mode not in set(VALID_PADDING_MODES):
            raise ValueError(f"unsupported padding_mode={padding_mode!r}")
        self.padding_mode = mode
        self.pad = int(((int(kernel_size) - 1) * int(dilation)) // 2)
        self.conv = nn.Conv1d(
            int(in_channels),
            int(out_channels),
            kernel_size=int(kernel_size),
            stride=1,
            padding=0,
            dilation=int(dilation),
            bias=bool(bias),
        )

    def forward(self, x_bcw: torch.Tensor) -> torch.Tensor:
        if self.pad <= 0:
            return self.conv(x_bcw)
        if self.padding_mode == "zeros":
            x_pad = F.pad(x_bcw, (self.pad, self.pad), mode="constant", value=0.0)
        else:
            mode = self.padding_mode if int(x_bcw.shape[-1]) > self.pad else "replicate"
            x_pad = F.pad(x_bcw, (self.pad, self.pad), mode=mode)
        return self.conv(x_pad)


class ResidualTCNBlock(nn.Module):
    def __init__(self, *, channels: int, dilation: int, padding_mode: str) -> None:
        super().__init__()
        self.conv1 = SamePadConv1d(channels, channels, kernel_size=3, dilation=dilation, bias=False, padding_mode=padding_mode)
        self.conv2 = SamePadConv1d(channels, channels, kernel_size=3, dilation=dilation, bias=True, padding_mode=padding_mode)
        self.norm1 = nn.GroupNorm(1, int(channels))
        self.norm2 = nn.GroupNorm(1, int(channels))

    def forward(self, x_bcw: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.norm1(self.conv1(x_bcw)))
        h = self.norm2(self.conv2(h))
        return F.gelu(x_bcw + h)


class ClassLocalFeatureMixer(nn.Module):
    def __init__(
        self,
        *,
        source_dim: int,
        class_id_vocab_sizes: Sequence[int],
        class_names: Sequence[str],
        source_feature_names: Sequence[str],
        class_local_dim: int,
    ) -> None:
        super().__init__()
        self.class_names = tuple(str(x) for x in list(class_names or ()))
        self.class_count = int(len(self.class_names))
        if self.class_count <= 0:
            raise ValueError("class-local fusion requires non-empty class_names")
        self.source_groups = tuple(
            tuple(int(idx) for idx in group)
            for group in _resolve_source_groups_by_class(
                input_dim_source=int(source_dim),
                source_feature_names=source_feature_names,
                class_names=self.class_names,
            )
        )
        self.id_groups = tuple(
            tuple(int(idx) for idx in group)
            for group in _resolve_id_groups_by_class(input_dim_source=int(source_dim), class_id_vocab_sizes=class_id_vocab_sizes)
        )
        self.class_local_dim = int(max(1, int(class_local_dim)))
        self.mixers = nn.ModuleList()
        for class_idx in range(self.class_count):
            in_dim = len(self.source_groups[class_idx]) + len(self.id_groups[class_idx])
            if in_dim <= 0:
                raise ValueError(f"class-local fusion has empty input for class {self.class_names[class_idx]!r}")
            hidden = max(self.class_local_dim, in_dim)
            self.mixers.append(nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, self.class_local_dim)))
        self.output_dim = self.class_count * self.class_local_dim
        self.output_feature_names = tuple(
            f"{class_name}_local{slot}" for class_name in self.class_names for slot in range(self.class_local_dim)
        )

    def forward(self, windows_btfw: torch.Tensor) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for class_idx, mixer in enumerate(self.mixers):
            feat_idx = list(self.source_groups[class_idx]) + list(self.id_groups[class_idx])
            class_view = windows_btfw[:, :, feat_idx, :].permute(0, 1, 3, 2).contiguous()
            flat = class_view.view(-1, int(class_view.shape[-1]))
            fused = mixer(flat).view(
                int(windows_btfw.shape[0]),
                int(windows_btfw.shape[1]),
                int(windows_btfw.shape[3]),
                self.class_local_dim,
            )
            parts.append(fused.permute(0, 1, 3, 2).contiguous())
        return torch.cat(parts, dim=2).contiguous()


class LinearWindowEncoder(nn.Module):
    def __init__(self, *, input_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(input_dim))
        self.proj = nn.Linear(int(input_dim), int(embed_dim))

    def forward(self, x_bfw: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.proj(self.norm(x_bfw.reshape(int(x_bfw.shape[0]), -1))))


class TemporalCNNEncoder(nn.Module):
    def __init__(self, *, input_dim: int, embed_dim: int, center_index: int, padding_mode: str) -> None:
        super().__init__()
        self.center_index = int(center_index)
        self.conv1 = SamePadConv1d(input_dim, embed_dim, kernel_size=5, dilation=1, bias=False, padding_mode=padding_mode)
        self.conv2 = SamePadConv1d(embed_dim, embed_dim, kernel_size=3, dilation=1, bias=True, padding_mode=padding_mode)
        self.norm = nn.LayerNorm(int(embed_dim))

    def forward(self, x_bfw: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.conv1(x_bfw))
        h = F.gelu(self.conv2(h))
        return self.norm(h[:, :, self.center_index])


class BiLSTMWindowEncoder(nn.Module):
    def __init__(self, *, input_dim: int, embed_dim: int, center_index: int, num_layers: int = 1) -> None:
        super().__init__()
        self.center_index = int(center_index)
        lstm_hidden = int(max(1, math.ceil(int(embed_dim) / 2.0)))
        self.input_norm = nn.LayerNorm(int(input_dim))
        self.encoder = nn.LSTM(
            input_size=int(input_dim),
            hidden_size=lstm_hidden,
            num_layers=int(num_layers),
            batch_first=True,
            bidirectional=True,
            dropout=0.10 if int(num_layers) > 1 else 0.0,
        )
        recurrent_dim = lstm_hidden * 2
        self.proj = nn.Identity() if recurrent_dim == int(embed_dim) else nn.Linear(recurrent_dim, int(embed_dim))
        self.norm = nn.LayerNorm(int(embed_dim))

    def forward(self, x_bfw: torch.Tensor) -> torch.Tensor:
        seq = self.input_norm(x_bfw.permute(0, 2, 1).contiguous())
        out, _ = self.encoder(seq)
        return self.norm(self.proj(out[:, self.center_index, :]))


class HybridTCNBiLSTMEncoder(nn.Module):
    def __init__(self, *, input_dim: int, embed_dim: int, center_index: int, padding_mode: str) -> None:
        super().__init__()
        self.center_index = int(center_index)
        self.stem = SamePadConv1d(input_dim, embed_dim, kernel_size=7, dilation=1, bias=False, padding_mode=padding_mode)
        self.blocks = nn.ModuleList(
            [
                ResidualTCNBlock(channels=embed_dim, dilation=1, padding_mode=padding_mode),
                ResidualTCNBlock(channels=embed_dim, dilation=2, padding_mode=padding_mode),
                ResidualTCNBlock(channels=embed_dim, dilation=4, padding_mode=padding_mode),
            ]
        )
        self.pre_lstm_norm = nn.LayerNorm(int(embed_dim))
        lstm_hidden = int(max(1, math.ceil(int(embed_dim) / 2.0)))
        self.bilstm = nn.LSTM(
            input_size=int(embed_dim),
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.10,
        )
        recurrent_dim = lstm_hidden * 2
        self.proj = nn.Identity() if recurrent_dim == int(embed_dim) else nn.Linear(recurrent_dim, int(embed_dim))
        self.norm = nn.LayerNorm(int(embed_dim))

    def forward(self, x_bfw: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.stem(x_bfw))
        for block in self.blocks:
            h = block(h)
        seq = self.pre_lstm_norm(h.permute(0, 2, 1).contiguous())
        out, _ = self.bilstm(seq)
        return self.norm(self.proj(out[:, self.center_index, :]))


class SecondsSequenceFrontend(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        class_id_vocab_sizes: Sequence[int] = (),
        source_feature_names: Sequence[str] = (),
        class_names: Sequence[str] = (),
        variant: str,
        embed_dim: int,
        window_radius: int,
        padding_mode: str,
        output_kind: str,
        chunk_size: int = 0,
        step_seconds: float = 0.0,
        class_local_fusion: bool = False,
        class_local_dim: int = DEFAULT_CLASS_LOCAL_DIM,
    ) -> None:
        super().__init__()
        variant_eff = str(variant or "").strip().lower()
        if variant_eff not in set(VALID_VARIANTS):
            raise ValueError(f"unsupported variant={variant!r}")
        output_eff = str(output_kind or DEFAULT_FRONTEND_OUTPUT).strip().lower()
        if output_eff not in set(VALID_FRONTEND_OUTPUTS):
            raise ValueError(f"unsupported output_kind={output_kind!r}")
        self.variant = variant_eff
        self.input_dim_source = int(input_dim)
        self.class_id_vocab_sizes = tuple(int(max(0, int(x))) for x in list(class_id_vocab_sizes or ()))
        self.source_feature_names = tuple(str(x) for x in list(source_feature_names or ()))
        self.class_names = tuple(str(x) for x in list(class_names or ())) or tuple(
            _default_class_names(source_feature_names=self.source_feature_names, class_id_vocab_sizes=self.class_id_vocab_sizes)
        )
        self.id_extra_dim = sum(int(vocab) for vocab in self.class_id_vocab_sizes if int(vocab) > 1)
        self.raw_input_dim = self.input_dim_source + self.id_extra_dim
        self.class_local_fusion = bool(class_local_fusion)
        self.class_local_dim = int(max(1, int(class_local_dim)))
        if self.class_local_fusion:
            self.class_local_mixer: Optional[ClassLocalFeatureMixer] = ClassLocalFeatureMixer(
                source_dim=self.input_dim_source,
                class_id_vocab_sizes=self.class_id_vocab_sizes,
                class_names=self.class_names,
                source_feature_names=self.source_feature_names,
                class_local_dim=self.class_local_dim,
            )
            self.input_dim = int(self.class_local_mixer.output_dim)
            self.input_feature_names = tuple(self.class_local_mixer.output_feature_names)
        else:
            self.class_local_mixer = None
            self.input_dim = self.raw_input_dim
            self.input_feature_names = tuple(
                resolve_frontend_input_feature_names(
                    input_dim_source=self.input_dim_source,
                    class_id_vocab_sizes=self.class_id_vocab_sizes,
                    source_feature_names=self.source_feature_names,
                    class_names=self.class_names,
                    class_local_fusion=False,
                )
            )
        self.embed_dim = int(embed_dim)
        self.window_radius = int(max(0, int(window_radius)))
        self.window_len = (2 * self.window_radius) + 1
        self.padding_mode = str(padding_mode or "reflect").strip().lower()
        self.output_kind = output_eff
        self.chunk_size = int(max(0, int(chunk_size)))
        self.step_seconds = float(step_seconds)
        self.output_dim = resolve_frontend_output_dim(embed_dim=embed_dim, output_kind=output_eff)
        nearest_feature_mask = infer_nearest_feature_mask_source(
            input_dim_source=self.input_dim_source,
            class_id_vocab_sizes=self.class_id_vocab_sizes,
            source_feature_names=self.source_feature_names,
        )
        self.register_buffer("nearest_feature_mask_source", nearest_feature_mask, persistent=False)
        center = self.window_radius
        if self.variant == "linear":
            self.encoder = LinearWindowEncoder(input_dim=self.input_dim * self.window_len, embed_dim=embed_dim)
        elif self.variant == "cnn":
            self.encoder = TemporalCNNEncoder(input_dim=self.input_dim, embed_dim=embed_dim, center_index=center, padding_mode=self.padding_mode)
        elif self.variant == "bilstm":
            self.encoder = BiLSTMWindowEncoder(input_dim=self.input_dim, embed_dim=embed_dim, center_index=center, num_layers=1)
        else:
            self.encoder = HybridTCNBiLSTMEncoder(input_dim=self.input_dim, embed_dim=embed_dim, center_index=center, padding_mode=self.padding_mode)
        self.norm = nn.LayerNorm(int(embed_dim))

    def _extract_windows(
        self,
        *,
        grid_bft: torch.Tensor,
        grid_ids_bct: Optional[torch.Tensor],
        grid_times_sec_bt: torch.Tensor,
        token_times_sec_bt: torch.Tensor,
        grid_valid_mask_bt: Optional[torch.Tensor],
        valid_mask_bt: Optional[torch.Tensor],
    ) -> torch.Tensor:
        windows = sample_grid_windows_in_seconds(
            grid_bft=grid_bft,
            grid_times_sec_bt=grid_times_sec_bt,
            token_times_sec_bt=token_times_sec_bt,
            window_radius=self.window_radius,
            step_seconds=self.step_seconds,
            grid_valid_mask_bt=grid_valid_mask_bt,
            valid_mask_bt=valid_mask_bt,
            nearest_feature_mask_f=self.nearest_feature_mask_source,
        )
        if self.id_extra_dim <= 0:
            return windows.contiguous()
        id_extra: Optional[torch.Tensor] = None
        if grid_ids_bct is not None and int(grid_ids_bct.shape[1]) > 0:
            sampled_ids = sample_grid_id_windows_in_seconds(
                grid_ids_bct=grid_ids_bct,
                grid_times_sec_bt=grid_times_sec_bt,
                token_times_sec_bt=token_times_sec_bt,
                window_radius=self.window_radius,
                step_seconds=self.step_seconds,
                grid_valid_mask_bt=grid_valid_mask_bt,
                valid_mask_bt=valid_mask_bt,
            )
            id_extra = expand_sampled_id_windows_onehot(sampled_ids, class_id_vocab_sizes=self.class_id_vocab_sizes)
        if id_extra is None:
            id_extra = torch.zeros((int(windows.shape[0]), int(windows.shape[1]), self.id_extra_dim, int(windows.shape[3])), dtype=windows.dtype, device=windows.device)
        raw_windows = torch.cat([windows, id_extra.to(dtype=windows.dtype)], dim=2).contiguous()
        if self.class_local_mixer is None:
            return raw_windows
        return self.class_local_mixer(raw_windows)

    def _encode_windows(self, windows_bfw: torch.Tensor) -> torch.Tensor:
        base = self.encoder(windows_bfw)
        feat = F.normalize(self.norm(base), dim=-1, eps=1.0e-4)
        return base if self.output_kind == "base" else feat

    def forward(
        self,
        grid_bft: torch.Tensor,
        *,
        grid_ids_bct: Optional[torch.Tensor] = None,
        grid_times_sec_bt: torch.Tensor,
        token_times_sec_bt: torch.Tensor,
        grid_valid_mask_bt: Optional[torch.Tensor] = None,
        valid_mask_bt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        windows = self._extract_windows(
            grid_bft=grid_bft,
            grid_ids_bct=grid_ids_bct,
            grid_times_sec_bt=grid_times_sec_bt,
            token_times_sec_bt=token_times_sec_bt,
            grid_valid_mask_bt=grid_valid_mask_bt,
            valid_mask_bt=valid_mask_bt,
        )
        batch_size, time_steps = int(windows.shape[0]), int(windows.shape[1])
        flat = windows.view(batch_size * time_steps, int(windows.shape[2]), int(windows.shape[3]))
        if self.chunk_size > 0 and int(flat.shape[0]) > self.chunk_size:
            parts = []
            for start in range(0, int(flat.shape[0]), self.chunk_size):
                parts.append(self._encode_windows(flat[start : start + self.chunk_size]))
            out = torch.cat(parts, dim=0)
        else:
            out = self._encode_windows(flat)
        return out.view(batch_size, time_steps, int(out.shape[-1]))


class SecondsMultiScaleFrontend(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        class_id_vocab_sizes: Sequence[int] = (),
        source_feature_names: Sequence[str] = (),
        class_names: Sequence[str] = (),
        variant: str,
        embed_dim: int,
        window_radii: Sequence[int],
        primary_radius: int,
        padding_mode: str,
        output_kind: str,
        chunk_size: int = 0,
        step_seconds: float = 0.0,
        class_local_fusion: bool = False,
        class_local_dim: int = DEFAULT_CLASS_LOCAL_DIM,
    ) -> None:
        super().__init__()
        radii = sorted({int(x) for x in list(window_radii) if int(x) >= 0})
        primary_eff = int(primary_radius)
        if primary_eff not in set(radii):
            radii = sorted(set(radii + [primary_eff]))
        self.window_radii = tuple(radii)
        self.primary_radius = primary_eff
        self.frontends = nn.ModuleDict(
            {
                str(radius): SecondsSequenceFrontend(
                    input_dim=input_dim,
                    class_id_vocab_sizes=class_id_vocab_sizes,
                    source_feature_names=source_feature_names,
                    class_names=class_names,
                    variant=variant,
                    embed_dim=embed_dim,
                    window_radius=radius,
                    padding_mode=padding_mode,
                    output_kind=output_kind,
                    chunk_size=chunk_size,
                    step_seconds=step_seconds,
                    class_local_fusion=class_local_fusion,
                    class_local_dim=class_local_dim,
                )
                for radius in self.window_radii
            }
        )
        self.output_dim = int(self.frontends[str(self.primary_radius)].output_dim)

    def iter_frontends(self) -> Iterable[tuple[int, SecondsSequenceFrontend]]:
        for radius in self.window_radii:
            yield int(radius), self.frontends[str(radius)]

    def forward_multiscale(
        self,
        grid_bft: torch.Tensor,
        *,
        grid_ids_bct: Optional[torch.Tensor] = None,
        grid_times_sec_bt: torch.Tensor,
        token_times_sec_bt: torch.Tensor,
        grid_valid_mask_bt: Optional[torch.Tensor] = None,
        valid_mask_bt: Optional[torch.Tensor] = None,
    ) -> dict[int, torch.Tensor]:
        return {
            radius: frontend(
                grid_bft,
                grid_ids_bct=grid_ids_bct,
                grid_times_sec_bt=grid_times_sec_bt,
                token_times_sec_bt=token_times_sec_bt,
                grid_valid_mask_bt=grid_valid_mask_bt,
                valid_mask_bt=valid_mask_bt,
            )
            for radius, frontend in self.iter_frontends()
        }

    def forward(
        self,
        grid_bft: torch.Tensor,
        *,
        grid_ids_bct: Optional[torch.Tensor] = None,
        grid_times_sec_bt: torch.Tensor,
        token_times_sec_bt: torch.Tensor,
        grid_valid_mask_bt: Optional[torch.Tensor] = None,
        valid_mask_bt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.frontends[str(self.primary_radius)](
            grid_bft,
            grid_ids_bct=grid_ids_bct,
            grid_times_sec_bt=grid_times_sec_bt,
            token_times_sec_bt=token_times_sec_bt,
            grid_valid_mask_bt=grid_valid_mask_bt,
            valid_mask_bt=valid_mask_bt,
        )


def build_seconds_frontend_from_cfg(frontend_cfg: Optional[dict[str, Any]]) -> Optional[nn.Module]:
    if not frontend_cfg:
        return None
    cfg = dict(frontend_cfg)
    multiscale_radii = [int(x) for x in list(cfg.get("multiscale_radii", []) or []) if int(x) >= 0]
    multiscale_enabled = bool(cfg.get("multiscale_enabled", False) or multiscale_radii)
    primary_radius = int(cfg.get("primary_radius", cfg.get("window_radius", 0)) or 0)
    common_kwargs = {
        "input_dim": int(cfg["input_dim_source"]),
        "class_id_vocab_sizes": tuple(int(x) for x in list(cfg.get("class_id_vocab_sizes") or [])),
        "source_feature_names": tuple(str(x) for x in list(cfg.get("source_feature_names") or [])),
        "class_names": tuple(str(x) for x in list(cfg.get("class_names") or [])),
        "variant": str(cfg["variant"]),
        "embed_dim": int(cfg["embed_dim"]),
        "padding_mode": str(cfg.get("padding_mode", "reflect")),
        "output_kind": str(cfg.get("output_kind", DEFAULT_FRONTEND_OUTPUT)),
        "chunk_size": int(cfg.get("chunk_size", 0) or 0),
        "step_seconds": float(cfg.get("step_seconds", 0.0) or 0.0),
        "class_local_fusion": bool(cfg.get("class_local_fusion", False)),
        "class_local_dim": int(cfg.get("class_local_dim", DEFAULT_CLASS_LOCAL_DIM) or DEFAULT_CLASS_LOCAL_DIM),
    }
    if multiscale_enabled:
        if not multiscale_radii:
            multiscale_radii = [primary_radius]
        return SecondsMultiScaleFrontend(window_radii=multiscale_radii, primary_radius=primary_radius, **common_kwargs)
    return SecondsSequenceFrontend(window_radius=int(cfg.get("window_radius", primary_radius)), **common_kwargs)


# ----------------------------- cache dataset -------------------------------




DEFAULT_FRONTEND_RADII = (0, 22, 41, 55)
DEFAULT_FRONTEND_PRIMARY_RADIUS = 22
DEFAULT_FRONTEND_VARIANT = "hybrid"
DEFAULT_FRONTEND_EMBED_DIM = 64
DEFAULT_FRONTEND_OUTPUT_KIND = "feat"
DEFAULT_FRONTEND_PADDING_MODE = "reflect"
DEFAULT_TARGET_TOKEN_RATE_HZ = 86.1328125


@dataclass
class DirectRegressorConfig:
    x_dim: int = 72
    frontend_cfg: dict[str, Any] | None = None
    concat_multiscale_frontend: bool = True
    positional_encoding: str = "seconds"
    positional_rate_hz: float = DEFAULT_TARGET_TOKEN_RATE_HZ
    backbone: str = "transformer"
    d_model: int = 512
    num_layers: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    cond_dropout_prob: float = 0.0


class ResidualSequenceConvBlock(nn.Module):
    def __init__(self, d_model: int, *, dilation: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(d_model))
        self.conv = SamePadConv1d(
            int(d_model),
            int(d_model),
            kernel_size=5,
            dilation=int(dilation),
            bias=True,
            padding_mode="reflect",
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(int(d_model)),
            nn.Linear(int(d_model), int(d_model) * 4),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(d_model) * 4, int(d_model)),
        )
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x_btd: torch.Tensor, mask_bt: torch.Tensor) -> torch.Tensor:
        h = self.norm(x_btd).transpose(1, 2).contiguous()
        h = F.gelu(self.conv(h)).transpose(1, 2).contiguous()
        x_btd = apply_seq_mask(x_btd + self.drop(h), mask_bt)
        x_btd = apply_seq_mask(x_btd + self.drop(self.ff(x_btd)), mask_bt)
        return x_btd


class DirectPCASequenceRegressor(nn.Module):
    def __init__(self, cfg: DirectRegressorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.frontend = build_seconds_frontend_from_cfg(cfg.frontend_cfg)
        if self.frontend is None:
            raise ValueError("frontend_cfg is required")
        if hasattr(self.frontend, "window_radii"):
            self.frontend_scale_radii = tuple(sorted(int(x) for x in list(getattr(self.frontend, "window_radii"))))
            self.frontend_primary_radius = int(getattr(self.frontend, "primary_radius"))
        else:
            self.frontend_primary_radius = int(getattr(self.frontend, "window_radius", 0))
            self.frontend_scale_radii = (self.frontend_primary_radius,)
        self.concat_multiscale_frontend = bool(cfg.concat_multiscale_frontend and hasattr(self.frontend, "forward_multiscale"))
        frontend_output_dim = int(getattr(self.frontend, "output_dim"))
        cond_dim = frontend_output_dim * (len(self.frontend_scale_radii) if self.concat_multiscale_frontend else 1)
        self.positional_encoding = str(cfg.positional_encoding).strip().lower()
        if self.positional_encoding not in {"seconds", "index"}:
            raise ValueError(f"unsupported positional_encoding={cfg.positional_encoding!r}")
        self.cond_proj = nn.Linear(cond_dim, int(cfg.d_model))
        self.in_norm = nn.LayerNorm(int(cfg.d_model))
        self.backbone_kind = str(cfg.backbone).strip().lower()
        if self.backbone_kind == "mlp":
            self.backbone = nn.Sequential(
                nn.LayerNorm(int(cfg.d_model)),
                nn.Linear(int(cfg.d_model), int(cfg.d_model) * int(cfg.mlp_ratio)),
                nn.GELU(),
                nn.Dropout(float(cfg.dropout)),
                nn.Linear(int(cfg.d_model) * int(cfg.mlp_ratio), int(cfg.d_model)),
            )
        elif self.backbone_kind == "tcn":
            dilations = [1, 2, 4, 8, 16, 32]
            self.backbone = nn.ModuleList(
                [
                    ResidualSequenceConvBlock(
                        int(cfg.d_model),
                        dilation=dilations[layer_idx % len(dilations)],
                        dropout=float(cfg.dropout),
                    )
                    for layer_idx in range(int(cfg.num_layers))
                ]
            )
        elif self.backbone_kind == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=int(cfg.d_model),
                nhead=int(cfg.num_heads),
                dim_feedforward=int(float(cfg.mlp_ratio) * int(cfg.d_model)),
                dropout=float(cfg.dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.backbone = nn.TransformerEncoder(layer, num_layers=int(cfg.num_layers), norm=nn.LayerNorm(int(cfg.d_model)))
        else:
            raise ValueError(f"unsupported --backbone={cfg.backbone!r}")
        self.out = nn.Sequential(nn.LayerNorm(int(cfg.d_model)), nn.Linear(int(cfg.d_model), int(cfg.x_dim)))

    def encode_conditioning(
        self,
        *,
        grid: torch.Tensor,
        grid_ids: torch.Tensor | None,
        grid_times_sec: torch.Tensor,
        token_times_sec: torch.Tensor,
        target_valid_mask_bt: torch.Tensor,
        grid_valid_mask_bt: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frontend_kwargs = {
            "grid_ids_bct": grid_ids,
            "grid_times_sec_bt": grid_times_sec,
            "token_times_sec_bt": token_times_sec,
            "grid_valid_mask_bt": grid_valid_mask_bt,
            "valid_mask_bt": target_valid_mask_bt,
        }
        if self.concat_multiscale_frontend:
            scale_features = {
                int(radius): feat
                for radius, feat in dict(self.frontend.forward_multiscale(grid, **frontend_kwargs)).items()
            }
            cond = torch.cat([scale_features[int(radius)] for radius in self.frontend_scale_radii], dim=-1).contiguous()
        else:
            cond = self.frontend(grid, **frontend_kwargs)
        cond_mask = target_valid_mask_bt.to(dtype=torch.bool)
        return apply_seq_mask(cond, cond_mask).contiguous(), cond_mask.contiguous()

    def forward(
        self,
        *,
        grid: torch.Tensor,
        grid_ids: torch.Tensor | None,
        grid_times_sec: torch.Tensor,
        token_times_sec: torch.Tensor,
        target_valid_mask_bt: torch.Tensor,
        grid_valid_mask_bt: torch.Tensor | None,
    ) -> torch.Tensor:
        cond, cond_mask = self.encode_conditioning(
            grid=grid,
            grid_ids=grid_ids,
            grid_times_sec=grid_times_sec,
            token_times_sec=token_times_sec,
            target_valid_mask_bt=target_valid_mask_bt,
            grid_valid_mask_bt=grid_valid_mask_bt,
        )
        if self.training and float(self.cfg.cond_dropout_prob) > 0.0:
            drop_b = torch.rand(int(cond.shape[0]), device=cond.device) < float(self.cfg.cond_dropout_prob)
            if bool(drop_b.any()):
                cond = cond.clone()
                cond[drop_b] = 0.0
        x = self.cond_proj(cond)
        if self.positional_encoding == "seconds":
            pos = sinusoidal_time_positions(token_times_sec.to(device=x.device), int(self.cfg.d_model), rate_hz=float(self.cfg.positional_rate_hz))
        else:
            pos = sinusoidal_index_positions(int(x.shape[1]), int(self.cfg.d_model), x.device)
        x = self.in_norm(x + pos[:, : int(x.shape[1]), :])
        x = apply_seq_mask(x, cond_mask)
        if self.backbone_kind == "transformer":
            x = self.backbone(x, src_key_padding_mask=~cond_mask)
        elif self.backbone_kind == "tcn":
            for block in self.backbone:  # type: ignore[union-attr]
                x = block(x, cond_mask)
        else:
            x = apply_seq_mask(x + self.backbone(x), cond_mask)  # type: ignore[operator]
        return apply_seq_mask(self.out(x), cond_mask)


