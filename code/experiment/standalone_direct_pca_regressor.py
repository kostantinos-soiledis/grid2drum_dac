#!/usr/bin/env python3
"""
Standalone direct PCA sequence regressor for the drum-grid diffusion cache.

This file is mostly standalone; it imports only the shared conditioning
ablation helper from this repo. It expects an already built cache with:

  cache_root/
    config.json
    target_stats.pt              optional, recomputed if absent/mismatched
    pca_basis.pt                 optional, used only for full-latent export
    manifests/{train,validation}.jsonl
    examples/<split>/*.pt

The model predicts normalized target PCA components directly:

  seconds-grid frontend -> sequence backbone -> [B, T, target_dim]

Example:

  python standalone_direct_pca_regressor.py \
    --cache-root cache_4beats_dac44q9_pca72_native_bpmgeom_duration_v1 \
    --out-dir runs_direct/direct_pca_v1 \
    --device cuda:0 \
    --epochs 80 --batch-size 8
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = REPO_ROOT.parent.parent
RUNS_ROOT = PACKAGE_ROOT / "runs"
RESULTS_ROOT = PACKAGE_ROOT / "results"


def _preload_stdlib_inspect() -> None:
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

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from scripts.conditioning_ablation import (
    VALID_CONDITIONING_ABLATIONS,
    apply_conditioning_ablation,
    conditioning_ablation_help,
    normalize_conditioning_ablation,
)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


# ----------------------------- small utilities -----------------------------


def torch_load(path: str | Path, *, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def read_json(path: str | Path) -> dict[str, Any]:
    return dict(json.loads(Path(path).read_text(encoding="utf-8")))


def maybe_read_json(path: str | Path) -> dict[str, Any] | None:
    path_obj = Path(path)
    if not path_obj.is_file():
        return None
    return read_json(path_obj)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: str | Path, payload: Mapping[str, Any]) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path_obj = Path(path)
    if not path_obj.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path_obj.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(dict(json.loads(text)))
    return rows


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def resolve_device(device: str) -> torch.device:
    text = str(device).strip().lower()
    if text in {"", "auto"}:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(text)


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


@dataclass(frozen=True)
class CacheExample:
    example_path: Path
    source_id: str
    source_manifest_index: int
    beat_index: int
    split: str
    class_names: tuple[str, ...]
    class_id_vocab_sizes: tuple[int, ...]
    feature_row_names: tuple[str, ...]
    grid_ft: torch.Tensor
    grid_ids_ft: torch.Tensor
    family_onsets_ft: torch.Tensor
    grid_times_sec_t: torch.Tensor
    token_times_sec_t: torch.Tensor
    grid_num_frames: int
    target_td: torch.Tensor
    target_full_td: torch.Tensor | None
    target_dim: int
    target_full_dim: int
    target_layout: str
    target_num_frames: int
    bpm: float
    duration_sec: float


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        text = raw_line.strip()
        if text:
            rows.append(dict(json.loads(text)))
    return rows


def _normalize_target_payload(payload: Mapping[str, Any], *, fallback_target_dim: int | None = None) -> tuple[torch.Tensor, int]:
    layout = str(payload.get("target_layout") or ("framewise_pca" if payload.get("target_pc_tk") is not None else "framewise_sum")).strip().lower()
    if layout == "framewise_pca":
        target = payload.get("target_pc_tk")
        if target is None:
            raise KeyError("cache payload is missing target_pc_tk for framewise_pca")
        target_t = torch.as_tensor(target, dtype=torch.float32).contiguous()
    else:
        target = payload.get("target_sum_td", payload.get("target_sum_t128"))
        if target is None:
            raise KeyError("cache payload is missing target_sum_td")
        target_t = torch.as_tensor(target, dtype=torch.float32).contiguous()
    target_dim = int(payload.get("target_dim", fallback_target_dim or int(target_t.shape[-1])))
    if int(target_t.dim()) != 2 or int(target_t.shape[-1]) != target_dim:
        raise RuntimeError(f"target shape/dim mismatch: target={tuple(target_t.shape)} target_dim={target_dim}")
    return target_t, target_dim


class DirectCacheDataset(Dataset[CacheExample]):
    def __init__(self, cache_root: str | Path, *, split: str, max_items: int = 0) -> None:
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.split = str(split).strip().lower()
        self.config = maybe_read_json(self.cache_root / "config.json") or {}
        self.target_dim = int(self.config.get("target_dim", 0) or 0)
        manifest_path = self.cache_root / "manifests" / f"{self.split}.jsonl"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"manifest not found: {manifest_path}")
        self.rows = _load_jsonl(manifest_path)
        if int(max_items) > 0:
            self.rows = self.rows[: int(max_items)]
        if not self.rows:
            raise RuntimeError(f"no rows found for split={self.split!r} under {self.cache_root}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> CacheExample:
        row = dict(self.rows[int(index)])
        example_path = (self.cache_root / str(row["out_pt"])).resolve()
        payload = dict(torch_load(example_path, map_location="cpu"))
        target_td, target_dim = _normalize_target_payload(payload, fallback_target_dim=(self.target_dim or None))
        target_num_frames = int(payload.get("target_num_frames", int(target_td.shape[0])))
        if int(target_td.shape[0]) != target_num_frames:
            raise RuntimeError(f"target_num_frames mismatch in {example_path}: {tuple(target_td.shape)} vs {target_num_frames}")
        target_full_payload = payload.get("target_sum_td", payload.get("target_sum_t128"))
        target_full_td = None if target_full_payload is None else torch.as_tensor(target_full_payload, dtype=torch.float32).contiguous()
        target_full_dim = int(payload.get("full_target_dim", int(target_full_td.shape[-1]) if target_full_td is not None else target_dim))

        grid_ft = torch.as_tensor(payload["grid_ft"], dtype=torch.float32).contiguous()
        grid_ids_ft = torch.as_tensor(payload.get("grid_ids_ft"), dtype=torch.long).contiguous()
        family_onsets_ft = torch.as_tensor(payload.get("family_onsets_ft"), dtype=torch.bool).contiguous()
        grid_times_sec_t = torch.as_tensor(payload["grid_times_sec_t"], dtype=torch.float32).contiguous()
        token_times_sec_t = torch.as_tensor(payload["token_times_sec_t"], dtype=torch.float32).contiguous()
        grid_num_frames = int(payload.get("grid_num_frames", int(grid_ft.shape[-1])))
        if int(grid_ft.shape[-1]) != grid_num_frames:
            raise RuntimeError(f"grid_num_frames mismatch in {example_path}: {tuple(grid_ft.shape)} vs {grid_num_frames}")

        return CacheExample(
            example_path=example_path,
            source_id=str(payload.get("source_id") or row.get("source_id") or ""),
            source_manifest_index=int(payload.get("source_manifest_index", row.get("source_manifest_index", -1))),
            beat_index=int(payload.get("beat_index", row.get("beat_index", 0))),
            split=str(payload.get("split") or row.get("split") or self.split),
            class_names=tuple(str(x) for x in list(payload.get("class_names") or [])),
            class_id_vocab_sizes=tuple(int(x) for x in list(payload.get("class_id_vocab_sizes") or [])),
            feature_row_names=tuple(str(x) for x in list(payload.get("feature_row_names") or [])),
            grid_ft=grid_ft,
            grid_ids_ft=grid_ids_ft,
            family_onsets_ft=family_onsets_ft,
            grid_times_sec_t=grid_times_sec_t,
            token_times_sec_t=token_times_sec_t,
            grid_num_frames=grid_num_frames,
            target_td=target_td,
            target_full_td=target_full_td,
            target_dim=int(target_dim),
            target_full_dim=int(target_full_dim),
            target_layout=str(payload.get("target_layout") or "framewise_sum"),
            target_num_frames=target_num_frames,
            bpm=float(payload.get("bpm", 0.0) or 0.0),
            duration_sec=float(payload.get("duration_sec", 0.0) or 0.0),
        )


def collate_cache_examples(items: Sequence[CacheExample]) -> dict[str, Any]:
    if not items:
        raise ValueError("expected non-empty batch")
    batch_size = len(items)
    grid_len = max(int(item.grid_num_frames) for item in items)
    target_len = max(int(item.target_num_frames) for item in items)
    grid_dim = int(items[0].grid_ft.shape[0])
    class_dim = int(items[0].grid_ids_ft.shape[0])
    target_dim = int(items[0].target_dim)
    target_full_dim = int(items[0].target_full_dim)
    if any(int(item.target_dim) != target_dim for item in items):
        raise ValueError("all batch items must share target_dim")

    grid = torch.zeros((batch_size, grid_dim, grid_len), dtype=torch.float32)
    grid_ids = torch.full((batch_size, class_dim, grid_len), -1, dtype=torch.long)
    family_onsets = torch.zeros((batch_size, class_dim, grid_len), dtype=torch.bool)
    grid_mask = torch.zeros((batch_size, grid_len), dtype=torch.bool)
    grid_times = torch.zeros((batch_size, grid_len), dtype=torch.float32)
    token_times = torch.zeros((batch_size, target_len), dtype=torch.float32)
    target = torch.zeros((batch_size, target_len, target_dim), dtype=torch.float32)
    target_full = torch.zeros((batch_size, target_len, target_full_dim), dtype=torch.float32)
    target_mask = torch.zeros((batch_size, target_len), dtype=torch.bool)
    bpm = torch.zeros((batch_size,), dtype=torch.float32)
    duration = torch.zeros((batch_size,), dtype=torch.float32)

    for row_idx, item in enumerate(items):
        gf = int(item.grid_num_frames)
        tf = int(item.target_num_frames)
        grid[row_idx, :, :gf] = item.grid_ft[:, :gf]
        grid_ids[row_idx, :, :gf] = item.grid_ids_ft[:, :gf]
        family_onsets[row_idx, :, :gf] = item.family_onsets_ft[:, :gf]
        grid_mask[row_idx, :gf] = True
        grid_times[row_idx, :gf] = item.grid_times_sec_t[:gf]
        token_times[row_idx, :tf] = item.token_times_sec_t[:tf]
        target[row_idx, :tf, :] = item.target_td[:tf]
        if item.target_full_td is not None and int(item.target_full_td.shape[-1]) == target_full_dim:
            target_full[row_idx, :tf, :] = item.target_full_td[:tf]
        target_mask[row_idx, :tf] = True
        bpm[row_idx] = float(item.bpm)
        duration[row_idx] = float(item.duration_sec)

    return {
        "class_names": list(items[0].class_names),
        "class_id_vocab_sizes": [int(x) for x in list(items[0].class_id_vocab_sizes)],
        "feature_row_names": list(items[0].feature_row_names),
        "grid": grid.contiguous(),
        "grid_ids": grid_ids.contiguous(),
        "family_onsets_bft": family_onsets.contiguous(),
        "grid_valid_mask": grid_mask.contiguous(),
        "grid_times_sec": grid_times.contiguous(),
        "token_times_sec": token_times.contiguous(),
        "target_btd": target.contiguous(),
        "target_full_btd": target_full.contiguous(),
        "target_valid_mask_bt": target_mask.contiguous(),
        "target_dim": target_dim,
        "target_full_dim": target_full_dim,
        "bpm": bpm.contiguous(),
        "duration_sec": duration.contiguous(),
        "source_id": [item.source_id for item in items],
        "example_path": [str(item.example_path) for item in items],
        "source_manifest_index_b": torch.tensor([int(item.source_manifest_index) for item in items], dtype=torch.long),
        "beat_index_b": torch.tensor([int(item.beat_index) for item in items], dtype=torch.long),
        "target_num_frames_b": torch.tensor([int(item.target_num_frames) for item in items], dtype=torch.long),
        "grid_num_frames_b": torch.tensor([int(item.grid_num_frames) for item in items], dtype=torch.long),
    }


def build_loader(
    cache_root: str | Path,
    *,
    split: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    max_items: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = DirectCacheDataset(cache_root, split=split, max_items=max_items)
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        collate_fn=collate_cache_examples,
        persistent_workers=bool(num_workers > 0),
    )


# ----------------------------- direct model --------------------------------


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


def build_frontend_cfg_from_batch(batch: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    radii = [int(x) for x in str(args.frontend_radii).split(",") if str(x).strip()]
    if not radii:
        radii = [int(args.frontend_primary_radius)]
    grid = torch.as_tensor(batch["grid"])
    return {
        "input_dim_source": int(grid.shape[1]),
        "class_id_vocab_sizes": [int(x) for x in list(batch.get("class_id_vocab_sizes") or [])],
        "source_feature_names": [str(x) for x in list(batch.get("feature_row_names") or [])],
        "class_names": [str(x) for x in list(batch.get("class_names") or [])],
        "variant": str(args.frontend_variant),
        "embed_dim": int(args.frontend_embed_dim),
        "output_kind": str(args.frontend_output_kind),
        "multiscale_enabled": bool(len(radii) > 1),
        "multiscale_radii": [int(x) for x in radii],
        "primary_radius": int(args.frontend_primary_radius),
        "window_radius": int(args.frontend_primary_radius),
        "padding_mode": str(args.frontend_padding_mode),
        "step_seconds": float(args.frontend_step_seconds),
        "chunk_size": int(args.frontend_chunk_size),
        "class_local_fusion": bool(args.frontend_class_local_fusion),
        "class_local_dim": int(args.frontend_class_local_dim),
    }


# -------------------------- train/eval/export code -------------------------


def batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device=device, non_blocking=True) if torch.is_tensor(value) else value
    return out


@torch.no_grad()
def estimate_target_normalization(loader: DataLoader, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    sum_d: torch.Tensor | None = None
    sq_sum_d: torch.Tensor | None = None
    n_frames = 0
    for batch in loader:
        target = torch.as_tensor(batch["target_btd"], dtype=torch.float32, device=device)
        mask = torch.as_tensor(batch["target_valid_mask_bt"], dtype=torch.bool, device=device)
        if sum_d is None:
            sum_d = torch.zeros((int(target.shape[-1]),), dtype=torch.float32, device=device)
            sq_sum_d = torch.zeros_like(sum_d)
        valid = target * mask.unsqueeze(-1).to(dtype=target.dtype)
        sum_d += valid.sum(dim=(0, 1))
        sq_sum_d += valid.square().sum(dim=(0, 1))
        n_frames += int(mask.sum().item())
    if sum_d is None or sq_sum_d is None or n_frames <= 0:
        raise RuntimeError("could not compute target normalization from empty loader")
    mean = sum_d / float(n_frames)
    var = (sq_sum_d / float(n_frames)) - mean.square()
    return mean.contiguous(), var.clamp_min(1.0e-6).sqrt().contiguous()


def load_or_compute_target_stats(
    cache_root: Path,
    train_loader: DataLoader,
    *,
    device: torch.device,
    target_dim: int,
    recompute: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    stats_path = cache_root / "target_stats.pt"
    if stats_path.is_file() and not bool(recompute):
        stats = dict(torch_load(stats_path, map_location="cpu"))
        mean = torch.as_tensor(stats.get("target_mean"), dtype=torch.float32, device=device).view(-1)
        std = torch.as_tensor(stats.get("target_std"), dtype=torch.float32, device=device).view(-1).clamp_min(1.0e-6)
        if int(mean.numel()) == int(target_dim) and int(std.numel()) == int(target_dim):
            return mean.contiguous(), std.contiguous()
        print(f"target_stats.pt has wrong dim ({mean.numel()}/{std.numel()} vs {target_dim}); recomputing")
    print("computing target normalization from train split")
    return estimate_target_normalization(train_loader, device=device)


def masked_prediction_loss(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    mask_bt: torch.Tensor,
    *,
    loss_kind: str,
    huber_beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if str(loss_kind) == "huber":
        per_dim = F.smooth_l1_loss(pred_norm, target_norm, reduction="none", beta=float(huber_beta))
    else:
        per_dim = (pred_norm - target_norm).square()
    per_tok = per_dim.mean(dim=-1)
    return per_tok[mask_bt].mean(), per_tok


def train_or_eval_epoch(
    *,
    model: DirectPCASequenceRegressor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    loss_kind: str,
    huber_beta: float,
    desc: str,
    conditioning_ablation: str = "none",
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_mse = 0.0
    total_tokens = 0
    total_batches = 0
    iterator = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True) if tqdm is not None else loader
    ablation_mode = normalize_conditioning_ablation(conditioning_ablation)
    for batch_index, batch_cpu in enumerate(iterator):
        batch_cpu = apply_conditioning_ablation(
            batch_cpu,
            ablation_mode,
            batch_index=int(batch_index),
        )
        batch = batch_to_device(batch_cpu, device)
        target = torch.as_tensor(batch["target_btd"], dtype=torch.float32, device=device)
        mask = torch.as_tensor(batch["target_valid_mask_bt"], dtype=torch.bool, device=device)
        target_norm = normalize_latent(target, target_mean, target_std)
        with torch.set_grad_enabled(train):
            pred_norm = model(
                grid=batch["grid"],
                grid_ids=batch["grid_ids"],
                grid_times_sec=batch["grid_times_sec"],
                token_times_sec=batch["token_times_sec"],
                target_valid_mask_bt=mask,
                grid_valid_mask_bt=batch["grid_valid_mask"],
            )
            loss, per_tok = masked_prediction_loss(
                pred_norm,
                target_norm,
                mask,
                loss_kind=loss_kind,
                huber_beta=float(huber_beta),
            )
            mse = ((pred_norm - target_norm).square().mean(dim=-1))[mask].mean()
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        tokens = int(mask.sum().item())
        total_loss += float(loss.detach().item()) * float(tokens)
        total_mse += float(mse.detach().item()) * float(tokens)
        total_tokens += tokens
        total_batches += 1
    denom = float(max(1, total_tokens))
    return {
        "loss": total_loss / denom,
        "mse": total_mse / denom,
        "tokens": float(total_tokens),
        "batches": float(total_batches),
    }


def load_pca_basis(cache_root: Path) -> dict[str, Any] | None:
    config = maybe_read_json(cache_root / "config.json") or {}
    rel_path = str(config.get("pca_basis_path") or "").strip()
    candidates = []
    if rel_path:
        candidates.append((cache_root / rel_path).resolve())
    candidates.append((cache_root / "pca_basis.pt").resolve())
    for path in candidates:
        if path.is_file():
            payload = dict(torch_load(path, map_location="cpu"))
            payload["mean"] = torch.as_tensor(payload["mean"], dtype=torch.float32).view(-1).contiguous()
            payload["components"] = torch.as_tensor(payload["components"], dtype=torch.float32).contiguous()
            return payload
    return None


def reconstruct_latent_from_pca(latent_btd: torch.Tensor, pca_basis: Mapping[str, Any] | None) -> torch.Tensor:
    if pca_basis is None:
        return latent_btd.contiguous()
    mean = torch.as_tensor(pca_basis["mean"], dtype=latent_btd.dtype, device=latent_btd.device)
    components = torch.as_tensor(pca_basis["components"], dtype=latent_btd.dtype, device=latent_btd.device)
    return (torch.matmul(latent_btd, components) + mean.view(1, 1, -1)).contiguous()


@torch.no_grad()
def export_predictions(
    *,
    model: DirectPCASequenceRegressor,
    loader: DataLoader,
    device: torch.device,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    out_path: Path,
    max_batches: int,
    cache_root: Path,
    export_full_latent: bool,
) -> None:
    model.eval()
    pca_basis = load_pca_basis(cache_root) if bool(export_full_latent) else None
    rows: list[dict[str, Any]] = []
    for batch_idx, batch_cpu in enumerate(loader):
        if int(max_batches) > 0 and int(batch_idx) >= int(max_batches):
            break
        batch = batch_to_device(batch_cpu, device)
        mask = torch.as_tensor(batch["target_valid_mask_bt"], dtype=torch.bool, device=device)
        pred_norm = model(
            grid=batch["grid"],
            grid_ids=batch["grid_ids"],
            grid_times_sec=batch["grid_times_sec"],
            token_times_sec=batch["token_times_sec"],
            target_valid_mask_bt=mask,
            grid_valid_mask_bt=batch["grid_valid_mask"],
        )
        pred = denormalize_latent(pred_norm, target_mean, target_std)
        target = torch.as_tensor(batch["target_btd"], dtype=torch.float32, device=device)
        row = {
            "pred_btd": pred.detach().cpu(),
            "target_btd": target.detach().cpu(),
            "target_valid_mask_bt": mask.detach().cpu(),
            "token_times_sec": batch["token_times_sec"].detach().cpu(),
            "source_id": list(batch["source_id"]),
            "example_path": list(batch["example_path"]),
        }
        if pca_basis is not None:
            row["pred_full_btd"] = reconstruct_latent_from_pca(pred, pca_basis).detach().cpu()
            row["target_full_btd"] = torch.as_tensor(batch["target_full_btd"], dtype=torch.float32).detach().cpu()
        rows.append(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "rows": rows,
            "metadata": {
                "cache_root": str(cache_root),
                "export_full_latent": bool(export_full_latent and pca_basis is not None),
            },
        },
        out_path,
    )


def save_checkpoint(
    path: Path,
    *,
    model: DirectPCASequenceRegressor,
    optimizer: torch.optim.Optimizer,
    cfg: DirectRegressorConfig,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    epoch: int,
    best_val_loss: float,
    run_config: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(cfg),
            "target_mean": target_mean.detach().cpu(),
            "target_std": target_std.detach().cpu(),
            "best_val_loss": float(best_val_loss),
            "run_config": dict(run_config),
        },
        path,
    )


def write_history_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a standalone direct PCA sequence regressor from a built cache.")
    parser.add_argument("--cache-root", type=str, default=str(RUNS_ROOT / "mini_cache"))
    parser.add_argument("--out-dir", type=str, default=str(RUNS_ROOT / "runs_direct" / "direct_pca_regressor"))
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="validation")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-items", type=int, default=0)
    parser.add_argument("--max-val-items", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--loss", type=str, default="huber", choices=("mse", "huber"))
    parser.add_argument("--huber-beta", type=float, default=0.25)
    parser.add_argument("--recompute-target-stats", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from --out-dir/last.pt while preserving optimizer state and history.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default="",
        help="Resume from an explicit checkpoint instead of --out-dir/last.pt.",
    )

    parser.add_argument("--backbone", type=str, default="transformer", choices=("transformer", "tcn", "mlp"))
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cond-dropout-prob", type=float, default=0.0)
    parser.add_argument("--positional-encoding", type=str, default="seconds", choices=("seconds", "index"))
    parser.add_argument("--positional-rate-hz", type=float, default=0.0)

    parser.add_argument("--frontend-variant", type=str, default=DEFAULT_FRONTEND_VARIANT, choices=VALID_VARIANTS)
    parser.add_argument("--frontend-embed-dim", type=int, default=DEFAULT_FRONTEND_EMBED_DIM)
    parser.add_argument("--frontend-output-kind", type=str, default=DEFAULT_FRONTEND_OUTPUT_KIND, choices=VALID_FRONTEND_OUTPUTS)
    parser.add_argument("--frontend-radii", type=str, default=",".join(str(x) for x in DEFAULT_FRONTEND_RADII))
    parser.add_argument("--frontend-primary-radius", type=int, default=DEFAULT_FRONTEND_PRIMARY_RADIUS)
    parser.add_argument("--frontend-padding-mode", type=str, default=DEFAULT_FRONTEND_PADDING_MODE, choices=VALID_PADDING_MODES)
    parser.add_argument("--frontend-step-seconds", type=float, default=0.0)
    parser.add_argument("--frontend-chunk-size", type=int, default=0)
    parser.add_argument("--frontend-class-local-fusion", action="store_true")
    parser.add_argument("--frontend-class-local-dim", type=int, default=DEFAULT_CLASS_LOCAL_DIM)
    parser.add_argument("--no-concat-multiscale-frontend", action="store_true")
    parser.add_argument(
        "--conditioning-ablation",
        type=str,
        default="none",
        choices=VALID_CONDITIONING_ABLATIONS,
        help="Training/validation symbolic conditioning mode. " + conditioning_ablation_help(),
    )

    parser.add_argument("--export-val-predictions", type=int, default=2, help="Number of validation batches to export after training; 0 disables.")
    parser.add_argument("--export-full-latent", action="store_true", help="Also reconstruct full codec latents via pca_basis.pt in exported predictions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conditioning_ablation = normalize_conditioning_ablation(str(args.conditioning_ablation))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    cache_root = Path(args.cache_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    resume_checkpoint_arg = str(args.resume_checkpoint or "").strip()
    resume_requested = bool(args.resume) or bool(resume_checkpoint_arg)
    if bool(args.overwrite) and bool(resume_requested):
        raise ValueError("--overwrite cannot be combined with --resume/--resume-checkpoint")
    resume_checkpoint_path = (
        Path(resume_checkpoint_arg).expanduser().resolve()
        if resume_checkpoint_arg
        else (out_dir / "last.pt").resolve()
    )
    if bool(args.overwrite) and out_dir.exists():
        shutil.rmtree(out_dir)
    existing = [
        path
        for path in (
            out_dir / "last.pt",
            out_dir / "best_direct.pt",
            out_dir / "history.jsonl",
            out_dir / "history.csv",
            out_dir / "run_config.json",
        )
        if path.exists()
    ]
    if bool(resume_requested):
        if not resume_checkpoint_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_checkpoint_path}")
    elif existing:
        artifacts = ", ".join(path.name for path in existing)
        raise RuntimeError(
            f"{out_dir} already has training artifacts ({artifacts}); "
            "pass --resume, --overwrite, or choose another --out-dir"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    history_jsonl = out_dir / "history.jsonl"
    history_csv = out_dir / "history.csv"
    if not bool(resume_requested):
        history_jsonl.write_text("", encoding="utf-8")

    device = resolve_device(str(args.device))
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    pin_memory = device.type == "cuda"

    train_loader = build_loader(
        cache_root,
        split=str(args.train_split),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        max_items=int(args.max_train_items),
        pin_memory=pin_memory,
    )
    val_loader = build_loader(
        cache_root,
        split=str(args.val_split),
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        max_items=int(args.max_val_items),
        pin_memory=pin_memory,
    )

    sample_batch = next(iter(train_loader))
    cache_config = maybe_read_json(cache_root / "config.json") or {}
    frontend_cfg = build_frontend_cfg_from_batch(sample_batch, args)
    target_dim = int(sample_batch["target_btd"].shape[-1])
    target_token_rate_hz = (
        float(args.positional_rate_hz)
        if float(args.positional_rate_hz) > 0.0
        else float(cache_config.get("target_token_rate_hz", cache_config.get("codec_frame_rate", DEFAULT_TARGET_TOKEN_RATE_HZ)))
    )
    cfg = DirectRegressorConfig(
        x_dim=int(target_dim),
        frontend_cfg=frontend_cfg,
        concat_multiscale_frontend=not bool(args.no_concat_multiscale_frontend),
        positional_encoding=str(args.positional_encoding),
        positional_rate_hz=float(target_token_rate_hz),
        backbone=str(args.backbone),
        d_model=int(args.d_model),
        num_layers=int(args.num_layers),
        num_heads=int(args.num_heads),
        mlp_ratio=float(args.mlp_ratio),
        dropout=float(args.dropout),
        cond_dropout_prob=float(args.cond_dropout_prob),
    )
    model = DirectPCASequenceRegressor(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    resume_payload: dict[str, Any] | None = None
    resume_start_epoch = 0
    if bool(resume_requested):
        resume_payload = dict(torch_load(resume_checkpoint_path, map_location="cpu"))
        saved_cfg = dict(resume_payload.get("config") or {})
        requested_cfg = asdict(cfg)
        if saved_cfg != requested_cfg:
            raise ValueError(
                "resume checkpoint model/frontend configuration does not match the requested configuration"
            )
        model.load_state_dict(dict(resume_payload["model_state_dict"]))
        if resume_payload.get("optimizer_state_dict") is None:
            raise KeyError(f"resume checkpoint missing optimizer_state_dict: {resume_checkpoint_path}")
        optimizer.load_state_dict(dict(resume_payload["optimizer_state_dict"]))
        resume_start_epoch = int(resume_payload["epoch"]) + 1
        target_mean = torch.as_tensor(
            resume_payload["target_mean"],
            dtype=torch.float32,
            device=device,
        ).view(-1).contiguous()
        target_std = (
            torch.as_tensor(
                resume_payload["target_std"],
                dtype=torch.float32,
                device=device,
            )
            .view(-1)
            .clamp_min(1.0e-6)
            .contiguous()
        )
    else:
        target_mean, target_std = load_or_compute_target_stats(
            cache_root,
            train_loader,
            device=device,
            target_dim=int(target_dim),
            recompute=bool(args.recompute_target_stats),
        )

    run_config = {
        "cache_root": str(cache_root),
        "out_dir": str(out_dir),
        "train_split": str(args.train_split),
        "val_split": str(args.val_split),
        "device": str(device),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "num_workers": int(args.num_workers),
        "resume": bool(resume_requested),
        "resume_checkpoint": str(resume_checkpoint_path) if bool(resume_requested) else "",
        "resume_start_epoch": int(resume_start_epoch),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "conditioning_ablation": str(conditioning_ablation),
        "loss": str(args.loss),
        "huber_beta": float(args.huber_beta),
        "model_cfg": asdict(cfg),
        "frontend_cfg": dict(frontend_cfg),
        "cache_config": dict(cache_config),
        "num_parameters": int(sum(int(p.numel()) for p in model.parameters())),
        "target_dim": int(target_dim),
        "target_token_rate_hz": float(target_token_rate_hz),
    }
    write_json(out_dir / "run_config.json", run_config)
    print(
        "training direct PCA regressor: "
        f"target_dim={target_dim} backbone={cfg.backbone} params={run_config['num_parameters']} "
        f"train_batches={len(train_loader)} val_batches={len(val_loader)} device={device}"
    )

    history_rows: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    if bool(resume_requested):
        assert resume_payload is not None
        history_rows = [
            row
            for row in read_jsonl(history_jsonl)
            if int(row.get("epoch", -1)) < int(resume_start_epoch)
        ]
        write_jsonl(history_jsonl, history_rows)
        write_history_csv(history_csv, history_rows)
        best_val_loss = float(resume_payload.get("best_val_loss", best_val_loss))
        print(
            "resumed direct checkpoint: "
            f"path={resume_checkpoint_path} next_epoch={resume_start_epoch} "
            f"best_val_loss={best_val_loss:.6f}"
        )
    if int(resume_start_epoch) >= int(args.epochs):
        print(
            f"checkpoint already reached epoch {int(resume_start_epoch) - 1}; "
            f"--epochs={int(args.epochs)} leaves nothing to train"
        )
        return
    for epoch in range(int(resume_start_epoch), int(args.epochs)):
        epoch_t0 = time.time()
        train_metrics = train_or_eval_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            target_mean=target_mean,
            target_std=target_std,
            loss_kind=str(args.loss),
            huber_beta=float(args.huber_beta),
            desc=f"train[{epoch}]",
            conditioning_ablation=str(conditioning_ablation),
        )
        with torch.no_grad():
            val_metrics = train_or_eval_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                device=device,
                target_mean=target_mean,
                target_std=target_std,
                loss_kind=str(args.loss),
                huber_beta=float(args.huber_beta),
                desc=f"val[{epoch}]",
                conditioning_ablation=str(conditioning_ablation),
            )
        elapsed = time.time() - epoch_t0
        val_loss = float(val_metrics["loss"])
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
        row = {
            "epoch": int(epoch),
            "train_loss": float(train_metrics["loss"]),
            "train_mse": float(train_metrics["mse"]),
            "train_tokens": int(train_metrics["tokens"]),
            "val_loss": float(val_metrics["loss"]),
            "val_mse": float(val_metrics["mse"]),
            "val_tokens": int(val_metrics["tokens"]),
            "best_val_loss": float(best_val_loss),
            "elapsed_sec": float(elapsed),
            "checkpoint_improved": bool(improved),
        }
        append_jsonl(history_jsonl, row)
        history_rows.append(row)
        write_history_csv(history_csv, history_rows)
        save_checkpoint(
            out_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            cfg=cfg,
            target_mean=target_mean,
            target_std=target_std,
            epoch=int(epoch),
            best_val_loss=float(best_val_loss),
            run_config=run_config,
        )
        if improved:
            save_checkpoint(
                out_dir / "best_direct.pt",
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                target_mean=target_mean,
                target_std=target_std,
                epoch=int(epoch),
                best_val_loss=float(best_val_loss),
                run_config=run_config,
            )
        print(
            f"epoch={epoch:03d} train_loss={row['train_loss']:.6f} train_mse={row['train_mse']:.6f} "
            f"val_loss={row['val_loss']:.6f} val_mse={row['val_mse']:.6f} "
            f"best={best_val_loss:.6f} improved={improved} elapsed={elapsed:.1f}s"
        )

    if int(args.export_val_predictions) > 0:
        export_predictions(
            model=model,
            loader=val_loader,
            device=device,
            target_mean=target_mean,
            target_std=target_std,
            out_path=out_dir / "validation_predictions.pt",
            max_batches=int(args.export_val_predictions),
            cache_root=cache_root,
            export_full_latent=bool(args.export_full_latent),
        )
        print(f"wrote {out_dir / 'validation_predictions.pt'}")


if __name__ == "__main__":
    main()
