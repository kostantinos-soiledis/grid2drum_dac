"""Seconds-aware local frontends for beat-grid to token-time conditioning."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


VALID_VARIANTS = ("linear", "cnn", "bilstm", "hybrid")
VALID_PADDING_MODES = ("zeros", "reflect", "replicate")
VALID_FRONTEND_OUTPUTS = ("base", "feat")
DEFAULT_FRONTEND_OUTPUT = "feat"
DEFAULT_SAMPLE_STEP_SECONDS = 1.0 / 250.0
DEFAULT_CLASS_LOCAL_DIM = 8


TorchModuleBase = nn.Module if nn is not None else object

FRONTEND_FEATURE_NORMALIZE_EPS = 1.0e-4


def require_torch() -> None:
    if torch is None or nn is None or F is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required for seconds_frontend")


def resolve_frontend_output_dim(*, embed_dim: int, output_kind: str) -> int:
    output_eff = str(output_kind or DEFAULT_FRONTEND_OUTPUT).strip().lower()
    if str(output_eff) not in set(VALID_FRONTEND_OUTPUTS):
        raise ValueError(f"unsupported output_kind={output_kind!r}")
    return int(embed_dim)


def infer_nearest_feature_mask_source(
    *,
    input_dim_source: int,
    class_id_vocab_sizes: Sequence[int] = (),
    source_feature_names: Optional[Sequence[str]] = None,
) -> "torch.Tensor":
    require_torch()
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
    if int(class_count) > 0 and int(feature_dim) == int(3 * class_count):
        mask[1::3] = True
        mask[2::3] = True
    elif int(class_count) > 0 and int(feature_dim) == int(2 * class_count):
        mask[0::2] = True
    return mask


def _default_class_names(
    *,
    source_feature_names: Sequence[str] = (),
    class_id_vocab_sizes: Sequence[int] = (),
) -> List[str]:
    names = [str(x) for x in list(source_feature_names or ())]
    out: List[str] = []
    for name in names:
        prefix = str(name).split("_", 1)[0].strip()
        if prefix and prefix not in out:
            out.append(prefix)
    class_count = int(len(list(class_id_vocab_sizes or ())))
    if int(class_count) > 0 and int(len(out)) == int(class_count):
        return out
    if out:
        return out
    return [f"class_{idx}" for idx in range(int(class_count))]


def resolve_frontend_input_feature_names(
    *,
    input_dim_source: int,
    class_id_vocab_sizes: Sequence[int] = (),
    source_feature_names: Sequence[str] = (),
    class_names: Sequence[str] = (),
    class_local_fusion: bool = False,
    class_local_dim: int = DEFAULT_CLASS_LOCAL_DIM,
) -> List[str]:
    source_names = [str(x) for x in list(source_feature_names or ())]
    vocab_sizes = [int(max(0, int(x))) for x in list(class_id_vocab_sizes or ())]
    class_names_eff = [str(x) for x in list(class_names or ())] or _default_class_names(
        source_feature_names=source_names,
        class_id_vocab_sizes=vocab_sizes,
    )
    if bool(class_local_fusion):
        dim = int(max(1, int(class_local_dim)))
        return [
            f"{str(class_name)}_local{int(slot)}"
            for class_name in list(class_names_eff)
            for slot in range(int(dim))
        ]
    id_names: List[str] = []
    for class_idx, vocab_size in enumerate(vocab_sizes):
        if int(vocab_size) <= 1:
            continue
        class_name = (
            str(class_names_eff[int(class_idx)])
            if int(class_idx) < int(len(class_names_eff))
            else f"class_{int(class_idx)}"
        )
        id_names.extend([f"{class_name}_id{int(slot)}" for slot in range(int(vocab_size))])
    if source_names:
        return list(source_names[: int(input_dim_source)]) + id_names
    return [f"feat_{idx}" for idx in range(int(input_dim_source))] + id_names


def _resolve_source_groups_by_class(
    *,
    input_dim_source: int,
    source_feature_names: Sequence[str],
    class_names: Sequence[str],
) -> List[List[int]]:
    names = [str(x) for x in list(source_feature_names or ())]
    class_names_eff = [str(x) for x in list(class_names or ())]
    if names and class_names_eff:
        groups: List[List[int]] = []
        for class_name in list(class_names_eff):
            prefix = f"{str(class_name)}_"
            group = [idx for idx, name in enumerate(names[: int(input_dim_source)]) if str(name).startswith(prefix)]
            groups.append(group)
        if all(bool(group) for group in groups) and int(sum(len(group) for group in groups)) == int(input_dim_source):
            return groups
    class_count = int(len(class_names_eff))
    if int(class_count) <= 0:
        return [list(range(int(input_dim_source)))]
    base = int(input_dim_source) // int(class_count)
    rem = int(input_dim_source) % int(class_count)
    groups = []
    cursor = 0
    for idx in range(int(class_count)):
        size = int(base + (1 if int(idx) < int(rem) else 0))
        groups.append(list(range(int(cursor), int(cursor) + int(size))))
        cursor += int(size)
    return groups


def _resolve_id_groups_by_class(
    *,
    input_dim_source: int,
    class_id_vocab_sizes: Sequence[int],
) -> List[List[int]]:
    offset = int(input_dim_source)
    groups: List[List[int]] = []
    for vocab_size in list(class_id_vocab_sizes or ()):
        vocab = int(max(0, int(vocab_size)))
        size = int(vocab if int(vocab) > 1 else 0)
        groups.append(list(range(int(offset), int(offset) + int(size))))
        offset += int(size)
    return groups


def _lengths_from_mask(
    mask_bt: Optional["torch.Tensor"],
    *,
    fallback: int,
    batch_size: int,
    device: Optional["torch.device"] = None,
) -> "torch.Tensor":
    require_torch()
    if mask_bt is None:
        return torch.full((int(batch_size),), int(fallback), dtype=torch.long, device=device)
    if mask_bt.dim() != 2:
        raise ValueError(f"mask_bt must be [B,T], got {tuple(mask_bt.shape)}")
    return mask_bt.to(device=device, dtype=torch.long).sum(dim=1).clamp_min(1)


def _infer_step_seconds(
    *,
    grid_times_sec_bt: "torch.Tensor",
    grid_valid_mask_bt: Optional["torch.Tensor"],
    default_step_seconds: float,
) -> "torch.Tensor":
    require_torch()
    if grid_times_sec_bt.dim() != 2:
        raise ValueError(f"grid_times_sec_bt must be [B,T], got {tuple(grid_times_sec_bt.shape)}")
    batch_size = int(grid_times_sec_bt.shape[0])
    lengths = _lengths_from_mask(
        grid_valid_mask_bt,
        fallback=int(grid_times_sec_bt.shape[1]),
        batch_size=int(batch_size),
        device=grid_times_sec_bt.device,
    )
    out = torch.full(
        (batch_size,),
        float(max(1.0e-6, float(default_step_seconds))),
        dtype=grid_times_sec_bt.dtype,
        device=grid_times_sec_bt.device,
    )
    for batch_idx in range(batch_size):
        length = int(lengths[int(batch_idx)].item())
        if int(length) <= 1:
            continue
        times = grid_times_sec_bt[int(batch_idx), : int(length)].to(dtype=torch.float32)
        diffs = times[1:] - times[:-1]
        diffs = diffs[torch.isfinite(diffs) & diffs.gt(1.0e-6)]
        if int(diffs.numel()) <= 0:
            continue
        out[int(batch_idx)] = diffs.median()
    return out


def _fill_invalid_token_times(
    token_times_sec_bt: "torch.Tensor",
    valid_mask_bt: Optional["torch.Tensor"],
) -> "torch.Tensor":
    require_torch()
    if valid_mask_bt is None:
        return token_times_sec_bt.to(dtype=torch.float32)
    if token_times_sec_bt.dim() != 2 or valid_mask_bt.dim() != 2:
        raise ValueError(
            f"token_times_sec_bt and valid_mask_bt must both be [B,T], got {tuple(token_times_sec_bt.shape)} / {tuple(valid_mask_bt.shape)}"
        )
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
    grid_ft: "torch.Tensor",
    grid_times_t: "torch.Tensor",
    sample_times_tw: "torch.Tensor",
) -> "torch.Tensor":
    require_torch()
    if grid_ft.dim() != 2 or grid_times_t.dim() != 1 or sample_times_tw.dim() != 2:
        raise ValueError(
            f"expected grid_ft [F,T], grid_times_t [T], sample_times_tw [S,W]; got {tuple(grid_ft.shape)}, {tuple(grid_times_t.shape)}, {tuple(sample_times_tw.shape)}"
        )
    feature_dim = int(grid_ft.shape[0])
    length = int(grid_ft.shape[1])
    if int(length) <= 0:
        return torch.zeros(
            (int(sample_times_tw.shape[0]), feature_dim, int(sample_times_tw.shape[1])),
            dtype=grid_ft.dtype,
            device=grid_ft.device,
        )
    flat = sample_times_tw.reshape(-1).to(device=grid_times_t.device, dtype=grid_times_t.dtype)
    lo = grid_times_t[0]
    hi = grid_times_t[int(length) - 1]
    flat = flat.clamp(min=float(lo.item()), max=float(hi.item()))
    idx1 = torch.searchsorted(grid_times_t, flat, right=False).clamp(min=0, max=int(length) - 1)
    idx0 = (idx1 - 1).clamp(min=0, max=int(length) - 1)
    t0 = grid_times_t.index_select(0, idx0)
    t1 = grid_times_t.index_select(0, idx1)
    x0 = grid_ft.index_select(1, idx0)
    x1 = grid_ft.index_select(1, idx1)
    denom = t1 - t0
    weight = torch.where(
        denom.abs() > 1.0e-8,
        (flat - t0) / denom,
        torch.zeros_like(flat),
    ).to(dtype=grid_ft.dtype)
    interp = x0 + ((x1 - x0) * weight.unsqueeze(0))
    return interp.view(feature_dim, int(sample_times_tw.shape[0]), int(sample_times_tw.shape[1])).permute(1, 0, 2).contiguous()


def _sample_grid_nearest_value_single(
    *,
    grid_ft: "torch.Tensor",
    grid_times_t: "torch.Tensor",
    sample_times_tw: "torch.Tensor",
) -> "torch.Tensor":
    require_torch()
    if grid_ft.dim() != 2 or grid_times_t.dim() != 1 or sample_times_tw.dim() != 2:
        raise ValueError(
            f"expected grid_ft [F,T], grid_times_t [T], sample_times_tw [S,W]; got {tuple(grid_ft.shape)}, {tuple(grid_times_t.shape)}, {tuple(sample_times_tw.shape)}"
        )
    feature_dim = int(grid_ft.shape[0])
    length = int(grid_ft.shape[1])
    if int(length) <= 0:
        return torch.zeros(
            (int(sample_times_tw.shape[0]), feature_dim, int(sample_times_tw.shape[1])),
            dtype=grid_ft.dtype,
            device=grid_ft.device,
        )
    flat = sample_times_tw.reshape(-1).to(device=grid_times_t.device, dtype=grid_times_t.dtype)
    lo = grid_times_t[0]
    hi = grid_times_t[int(length) - 1]
    flat = flat.clamp(min=float(lo.item()), max=float(hi.item()))
    idx_hi = torch.searchsorted(grid_times_t, flat, right=False).clamp(min=0, max=int(length) - 1)
    idx_lo = (idx_hi - 1).clamp(min=0, max=int(length) - 1)
    t_lo = grid_times_t.index_select(0, idx_lo)
    t_hi = grid_times_t.index_select(0, idx_hi)
    choose_hi = (flat - t_lo).abs() > (t_hi - flat).abs()
    nearest_idx = torch.where(choose_hi, idx_hi, idx_lo)
    sampled = grid_ft.index_select(1, nearest_idx)
    return sampled.view(feature_dim, int(sample_times_tw.shape[0]), int(sample_times_tw.shape[1])).permute(1, 0, 2).contiguous()


def sample_grid_windows_in_seconds(
    *,
    grid_bft: "torch.Tensor",
    grid_times_sec_bt: "torch.Tensor",
    token_times_sec_bt: "torch.Tensor",
    window_radius: int,
    step_seconds: float = 0.0,
    grid_valid_mask_bt: Optional["torch.Tensor"] = None,
    valid_mask_bt: Optional["torch.Tensor"] = None,
    nearest_feature_mask_f: Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    require_torch()
    if grid_bft.dim() != 3:
        raise ValueError(f"grid_bft must be [B,F,Tg], got {tuple(grid_bft.shape)}")
    if grid_times_sec_bt.dim() != 2 or int(grid_times_sec_bt.shape[0]) != int(grid_bft.shape[0]) or int(grid_times_sec_bt.shape[1]) != int(grid_bft.shape[2]):
        raise ValueError(
            f"grid_times_sec_bt must match [B,Tg]=({int(grid_bft.shape[0])},{int(grid_bft.shape[2])}), got {tuple(grid_times_sec_bt.shape)}"
        )
    if token_times_sec_bt.dim() != 2 or int(token_times_sec_bt.shape[0]) != int(grid_bft.shape[0]):
        raise ValueError(f"token_times_sec_bt must be [B,Tt], got {tuple(token_times_sec_bt.shape)}")
    batch_size = int(grid_bft.shape[0])
    nearest_mask = None
    if nearest_feature_mask_f is not None:
        if nearest_feature_mask_f.dim() != 1 or int(nearest_feature_mask_f.shape[0]) != int(grid_bft.shape[1]):
            raise ValueError(
                f"nearest_feature_mask_f must be [F]={int(grid_bft.shape[1])}, got {tuple(nearest_feature_mask_f.shape)}"
            )
        nearest_mask = nearest_feature_mask_f.to(device=grid_bft.device, dtype=torch.bool)
    token_times = _fill_invalid_token_times(token_times_sec_bt, valid_mask_bt)
    if float(step_seconds) > 0.0:
        step_b = torch.full(
            (batch_size,),
            float(step_seconds),
            dtype=grid_bft.dtype,
            device=grid_bft.device,
        )
    else:
        step_b = _infer_step_seconds(
            grid_times_sec_bt=grid_times_sec_bt.to(device=grid_bft.device, dtype=torch.float32),
            grid_valid_mask_bt=grid_valid_mask_bt,
            default_step_seconds=DEFAULT_SAMPLE_STEP_SECONDS,
        ).to(dtype=grid_bft.dtype, device=grid_bft.device)
    radius = int(max(0, int(window_radius)))
    offsets = torch.arange(-radius, radius + 1, device=grid_bft.device, dtype=grid_bft.dtype)
    sample_times = token_times.to(dtype=grid_bft.dtype)[:, :, None] + (step_b[:, None, None] * offsets[None, None, :])
    outputs: List["torch.Tensor"] = []
    grid_lengths = _lengths_from_mask(
        grid_valid_mask_bt,
        fallback=int(grid_bft.shape[2]),
        batch_size=int(batch_size),
        device=grid_bft.device,
    )
    for batch_idx in range(batch_size):
        grid_len = int(grid_lengths[int(batch_idx)].item())
        valid_grid_ft = grid_bft[int(batch_idx), :, : int(grid_len)].to(dtype=torch.float32)
        valid_times_t = grid_times_sec_bt[int(batch_idx), : int(grid_len)].to(device=grid_bft.device, dtype=torch.float32)
        sampled_tfw = _sample_grid_linear_single(
            grid_ft=valid_grid_ft,
            grid_times_t=valid_times_t,
            sample_times_tw=sample_times[int(batch_idx)].to(dtype=torch.float32),
        )
        if nearest_mask is not None and bool(nearest_mask.any().item()):
            sampled_nearest_tfw = _sample_grid_nearest_value_single(
                grid_ft=valid_grid_ft,
                grid_times_t=valid_times_t,
                sample_times_tw=sample_times[int(batch_idx)].to(dtype=torch.float32),
            )
            sampled_tfw[:, nearest_mask, :] = sampled_nearest_tfw[:, nearest_mask, :]
        outputs.append(sampled_tfw.unsqueeze(0))
    return torch.cat(outputs, dim=0)


def sample_grid_centers_in_seconds(
    *,
    grid_bft: "torch.Tensor",
    grid_times_sec_bt: "torch.Tensor",
    token_times_sec_bt: "torch.Tensor",
    grid_valid_mask_bt: Optional["torch.Tensor"] = None,
    valid_mask_bt: Optional["torch.Tensor"] = None,
    nearest_feature_mask_f: Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    require_torch()
    windows = sample_grid_windows_in_seconds(
        grid_bft=grid_bft,
        grid_times_sec_bt=grid_times_sec_bt,
        token_times_sec_bt=token_times_sec_bt,
        window_radius=0,
        step_seconds=0.0,
        grid_valid_mask_bt=grid_valid_mask_bt,
        valid_mask_bt=valid_mask_bt,
        nearest_feature_mask_f=nearest_feature_mask_f,
    )
    return windows[:, :, :, 0].contiguous()


def _sample_grid_nearest_single(
    *,
    grid_ct: "torch.Tensor",
    grid_times_t: "torch.Tensor",
    sample_times_tw: "torch.Tensor",
) -> "torch.Tensor":
    require_torch()
    if grid_ct.dim() != 2 or grid_times_t.dim() != 1 or sample_times_tw.dim() != 2:
        raise ValueError(
            f"expected grid_ct [C,T], grid_times_t [T], sample_times_tw [S,W]; got {tuple(grid_ct.shape)}, {tuple(grid_times_t.shape)}, {tuple(sample_times_tw.shape)}"
        )
    class_dim = int(grid_ct.shape[0])
    length = int(grid_ct.shape[1])
    if int(length) <= 0:
        return torch.full(
            (int(sample_times_tw.shape[0]), class_dim, int(sample_times_tw.shape[1])),
            -1,
            dtype=grid_ct.dtype,
            device=grid_ct.device,
        )
    flat = sample_times_tw.reshape(-1).to(device=grid_times_t.device, dtype=grid_times_t.dtype)
    lo = grid_times_t[0]
    hi = grid_times_t[int(length) - 1]
    flat = flat.clamp(min=float(lo.item()), max=float(hi.item()))
    idx_hi = torch.searchsorted(grid_times_t, flat, right=False).clamp(min=0, max=int(length) - 1)
    idx_lo = (idx_hi - 1).clamp(min=0, max=int(length) - 1)
    t_lo = grid_times_t.index_select(0, idx_lo)
    t_hi = grid_times_t.index_select(0, idx_hi)
    choose_hi = (flat - t_lo).abs() > (t_hi - flat).abs()
    nearest_idx = torch.where(choose_hi, idx_hi, idx_lo)
    sampled = grid_ct.index_select(1, nearest_idx)
    return sampled.view(class_dim, int(sample_times_tw.shape[0]), int(sample_times_tw.shape[1])).permute(1, 0, 2).contiguous()


def sample_grid_id_windows_in_seconds(
    *,
    grid_ids_bct: "torch.Tensor",
    grid_times_sec_bt: "torch.Tensor",
    token_times_sec_bt: "torch.Tensor",
    window_radius: int,
    step_seconds: float = 0.0,
    grid_valid_mask_bt: Optional["torch.Tensor"] = None,
    valid_mask_bt: Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    require_torch()
    if grid_ids_bct.dim() != 3:
        raise ValueError(f"grid_ids_bct must be [B,C,Tg], got {tuple(grid_ids_bct.shape)}")
    if grid_times_sec_bt.dim() != 2 or int(grid_times_sec_bt.shape[0]) != int(grid_ids_bct.shape[0]) or int(grid_times_sec_bt.shape[1]) != int(grid_ids_bct.shape[2]):
        raise ValueError(
            f"grid_times_sec_bt must match [B,Tg]=({int(grid_ids_bct.shape[0])},{int(grid_ids_bct.shape[2])}), got {tuple(grid_times_sec_bt.shape)}"
        )
    if token_times_sec_bt.dim() != 2 or int(token_times_sec_bt.shape[0]) != int(grid_ids_bct.shape[0]):
        raise ValueError(f"token_times_sec_bt must be [B,Tt], got {tuple(token_times_sec_bt.shape)}")
    batch_size = int(grid_ids_bct.shape[0])
    token_times = _fill_invalid_token_times(token_times_sec_bt, valid_mask_bt)
    if float(step_seconds) > 0.0:
        step_b = torch.full(
            (batch_size,),
            float(step_seconds),
            dtype=token_times.dtype,
            device=token_times.device,
        )
    else:
        step_b = _infer_step_seconds(
            grid_times_sec_bt=grid_times_sec_bt.to(device=token_times.device, dtype=torch.float32),
            grid_valid_mask_bt=grid_valid_mask_bt,
            default_step_seconds=DEFAULT_SAMPLE_STEP_SECONDS,
        ).to(dtype=token_times.dtype, device=token_times.device)
    radius = int(max(0, int(window_radius)))
    offsets = torch.arange(-radius, radius + 1, device=token_times.device, dtype=token_times.dtype)
    sample_times = token_times.to(dtype=token_times.dtype)[:, :, None] + (step_b[:, None, None] * offsets[None, None, :])
    outputs: List["torch.Tensor"] = []
    grid_lengths = _lengths_from_mask(
        grid_valid_mask_bt,
        fallback=int(grid_ids_bct.shape[2]),
        batch_size=int(batch_size),
        device=grid_ids_bct.device,
    )
    for batch_idx in range(batch_size):
        grid_len = int(grid_lengths[int(batch_idx)].item())
        valid_grid_ct = grid_ids_bct[int(batch_idx), :, : int(grid_len)].to(dtype=torch.long)
        valid_times_t = grid_times_sec_bt[int(batch_idx), : int(grid_len)].to(device=grid_ids_bct.device, dtype=torch.float32)
        sampled_tcw = _sample_grid_nearest_single(
            grid_ct=valid_grid_ct,
            grid_times_t=valid_times_t,
            sample_times_tw=sample_times[int(batch_idx)].to(dtype=torch.float32),
        )
        outputs.append(sampled_tcw.unsqueeze(0))
    return torch.cat(outputs, dim=0)


def expand_sampled_id_windows_onehot(
    sampled_ids_btcw: Optional["torch.Tensor"],
    *,
    class_id_vocab_sizes: Sequence[int],
) -> Optional["torch.Tensor"]:
    require_torch()
    if sampled_ids_btcw is None:
        return None
    if sampled_ids_btcw.dim() != 4:
        raise ValueError(f"sampled_ids_btcw must be [B,T,C,W], got {tuple(sampled_ids_btcw.shape)}")
    vocab_sizes = tuple(int(max(0, int(x))) for x in list(class_id_vocab_sizes or ()))
    if int(sampled_ids_btcw.shape[2]) != int(len(vocab_sizes)):
        raise ValueError(
            f"class_id_vocab_sizes must match sampled id class count, got {len(vocab_sizes)} vs {tuple(sampled_ids_btcw.shape)}"
        )
    parts: List["torch.Tensor"] = []
    for class_idx, vocab_size in enumerate(vocab_sizes):
        if int(vocab_size) <= 1:
            continue
        ids_btw = sampled_ids_btcw[:, :, int(class_idx), :].to(dtype=torch.long)
        safe = ids_btw.clamp(min=0, max=int(vocab_size) - 1)
        onehot = F.one_hot(safe, num_classes=int(vocab_size)).to(dtype=torch.float32)
        valid = ids_btw.ge(0) & ids_btw.lt(int(vocab_size))
        onehot = onehot * valid.unsqueeze(-1).to(dtype=onehot.dtype)
        parts.append(onehot.permute(0, 1, 3, 2).contiguous())
    if not parts:
        return None
    return torch.cat(parts, dim=2)


class SamePadConv1d(TorchModuleBase):
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
        require_torch()
        super().__init__()
        mode = str(padding_mode).strip().lower()
        if str(mode) not in set(VALID_PADDING_MODES):
            raise ValueError(f"unsupported padding_mode={padding_mode!r}")
        self.padding_mode = str(mode)
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

    def forward(self, x_bcw: "torch.Tensor") -> "torch.Tensor":
        if int(self.pad) <= 0:
            return self.conv(x_bcw)
        mode = str(self.padding_mode)
        if str(mode) == "zeros":
            x_pad = F.pad(x_bcw, (int(self.pad), int(self.pad)), mode="constant", value=0.0)
        else:
            eff_mode = str(mode)
            if int(x_bcw.shape[-1]) <= int(self.pad):
                eff_mode = "replicate"
            x_pad = F.pad(x_bcw, (int(self.pad), int(self.pad)), mode=str(eff_mode))
        return self.conv(x_pad)


class ResidualTCNBlock(TorchModuleBase):
    def __init__(self, *, channels: int, dilation: int, padding_mode: str) -> None:
        require_torch()
        super().__init__()
        self.conv1 = SamePadConv1d(
            int(channels),
            int(channels),
            kernel_size=3,
            dilation=int(dilation),
            bias=False,
            padding_mode=str(padding_mode),
        )
        self.conv2 = SamePadConv1d(
            int(channels),
            int(channels),
            kernel_size=3,
            dilation=int(dilation),
            bias=True,
            padding_mode=str(padding_mode),
        )
        self.norm1 = nn.GroupNorm(1, int(channels))
        self.norm2 = nn.GroupNorm(1, int(channels))

    def forward(self, x_bcw: "torch.Tensor") -> "torch.Tensor":
        h = F.gelu(self.norm1(self.conv1(x_bcw)))
        h = self.norm2(self.conv2(h))
        return F.gelu(x_bcw + h)


class ClassLocalFeatureMixer(TorchModuleBase):
    def __init__(
        self,
        *,
        source_dim: int,
        class_id_vocab_sizes: Sequence[int],
        class_names: Sequence[str],
        source_feature_names: Sequence[str],
        class_local_dim: int,
    ) -> None:
        require_torch()
        super().__init__()
        self.class_names = tuple(str(x) for x in list(class_names or ()))
        self.class_count = int(len(self.class_names))
        if int(self.class_count) <= 0:
            raise ValueError("class-local fusion requires non-empty class_names")
        self.source_groups = tuple(
            tuple(int(idx) for idx in list(group))
            for group in _resolve_source_groups_by_class(
                input_dim_source=int(source_dim),
                source_feature_names=source_feature_names,
                class_names=self.class_names,
            )
        )
        self.id_groups = tuple(
            tuple(int(idx) for idx in list(group))
            for group in _resolve_id_groups_by_class(
                input_dim_source=int(source_dim),
                class_id_vocab_sizes=class_id_vocab_sizes,
            )
        )
        if int(len(self.source_groups)) != int(self.class_count) or int(len(self.id_groups)) != int(self.class_count):
            raise ValueError("class-local fusion group count must match class count")
        self.class_local_dim = int(max(1, int(class_local_dim)))
        self.mixers = nn.ModuleList()
        for class_idx in range(int(self.class_count)):
            in_dim = int(len(self.source_groups[int(class_idx)]) + len(self.id_groups[int(class_idx)]))
            if int(in_dim) <= 0:
                raise ValueError(f"class-local fusion requires positive in_dim for class {self.class_names[int(class_idx)]}")
            hidden = int(max(self.class_local_dim, int(in_dim)))
            self.mixers.append(
                nn.Sequential(
                    nn.LayerNorm(int(in_dim)),
                    nn.Linear(int(in_dim), int(hidden)),
                    nn.GELU(),
                    nn.Linear(int(hidden), int(self.class_local_dim)),
                )
            )
        self.output_dim = int(self.class_count * self.class_local_dim)
        self.output_feature_names = tuple(
            f"{str(class_name)}_local{int(slot)}"
            for class_name in list(self.class_names)
            for slot in range(int(self.class_local_dim))
        )

    def forward(self, windows_btfw: "torch.Tensor") -> "torch.Tensor":
        if windows_btfw.dim() != 4:
            raise ValueError(f"windows_btfw must be [B,T,F,W], got {tuple(windows_btfw.shape)}")
        parts: List["torch.Tensor"] = []
        for class_idx, mixer in enumerate(list(self.mixers)):
            feat_idx = list(self.source_groups[int(class_idx)]) + list(self.id_groups[int(class_idx)])
            class_view = windows_btfw[:, :, feat_idx, :].permute(0, 1, 3, 2).contiguous()
            flat = class_view.view(-1, int(class_view.shape[-1]))
            fused = mixer(flat).view(
                int(windows_btfw.shape[0]),
                int(windows_btfw.shape[1]),
                int(windows_btfw.shape[3]),
                int(self.class_local_dim),
            )
            parts.append(fused.permute(0, 1, 3, 2).contiguous())
        return torch.cat(parts, dim=2).contiguous()


class LinearWindowEncoder(TorchModuleBase):
    def __init__(self, *, input_dim: int, embed_dim: int) -> None:
        require_torch()
        super().__init__()
        self.norm = nn.LayerNorm(int(input_dim))
        self.proj = nn.Linear(int(input_dim), int(embed_dim))

    def forward(self, x_bfw: "torch.Tensor", *, return_debug: bool = False):
        flat = x_bfw.reshape(int(x_bfw.shape[0]), -1)
        z = F.gelu(self.proj(self.norm(flat)))
        if not bool(return_debug):
            return z
        return z, {"layer_maps": {"input": x_bfw, "linear_embed": z[:, :, None]}}


class TemporalCNNEncoder(TorchModuleBase):
    def __init__(self, *, input_dim: int, embed_dim: int, center_index: int, padding_mode: str) -> None:
        require_torch()
        super().__init__()
        self.center_index = int(center_index)
        self.conv1 = SamePadConv1d(
            int(input_dim),
            int(embed_dim),
            kernel_size=5,
            dilation=1,
            bias=False,
            padding_mode=str(padding_mode),
        )
        self.conv2 = SamePadConv1d(
            int(embed_dim),
            int(embed_dim),
            kernel_size=3,
            dilation=1,
            bias=True,
            padding_mode=str(padding_mode),
        )
        self.norm = nn.LayerNorm(int(embed_dim))

    def forward(self, x_bfw: "torch.Tensor", *, return_debug: bool = False):
        h = F.gelu(self.conv1(x_bfw))
        h = F.gelu(self.conv2(h))
        z = self.norm(h[:, :, int(self.center_index)])
        if not bool(return_debug):
            return z
        return z, {"layer_maps": {"input": x_bfw, "cnn_center": z[:, :, None]}}


class BiLSTMWindowEncoder(TorchModuleBase):
    def __init__(self, *, input_dim: int, embed_dim: int, center_index: int, num_layers: int = 1) -> None:
        require_torch()
        super().__init__()
        self.center_index = int(center_index)
        lstm_hidden = int(max(1, math.ceil(int(embed_dim) / 2.0)))
        self.input_norm = nn.LayerNorm(int(input_dim))
        self.encoder = nn.LSTM(
            input_size=int(input_dim),
            hidden_size=int(lstm_hidden),
            num_layers=int(num_layers),
            batch_first=True,
            bidirectional=True,
            dropout=0.10 if int(num_layers) > 1 else 0.0,
        )
        recurrent_dim = int(lstm_hidden * 2)
        self.proj = nn.Identity() if int(recurrent_dim) == int(embed_dim) else nn.Linear(int(recurrent_dim), int(embed_dim))
        self.norm = nn.LayerNorm(int(embed_dim))

    def forward(self, x_bfw: "torch.Tensor", *, return_debug: bool = False):
        seq = x_bfw.permute(0, 2, 1).contiguous()
        seq = self.input_norm(seq)
        out, _ = self.encoder(seq)
        z = self.norm(self.proj(out[:, int(self.center_index), :]))
        if not bool(return_debug):
            return z
        return z, {"layer_maps": {"input": x_bfw, "bilstm_center": z[:, :, None]}}


class HybridTCNBiLSTMEncoder(TorchModuleBase):
    def __init__(self, *, input_dim: int, embed_dim: int, center_index: int, padding_mode: str) -> None:
        require_torch()
        super().__init__()
        self.center_index = int(center_index)
        self.stem = SamePadConv1d(
            int(input_dim),
            int(embed_dim),
            kernel_size=7,
            dilation=1,
            bias=False,
            padding_mode=str(padding_mode),
        )
        self.blocks = nn.ModuleList(
            [
                ResidualTCNBlock(channels=int(embed_dim), dilation=1, padding_mode=str(padding_mode)),
                ResidualTCNBlock(channels=int(embed_dim), dilation=2, padding_mode=str(padding_mode)),
                ResidualTCNBlock(channels=int(embed_dim), dilation=4, padding_mode=str(padding_mode)),
            ]
        )
        self.pre_lstm_norm = nn.LayerNorm(int(embed_dim))
        lstm_hidden = int(max(1, math.ceil(int(embed_dim) / 2.0)))
        self.bilstm = nn.LSTM(
            input_size=int(embed_dim),
            hidden_size=int(lstm_hidden),
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.10,
        )
        recurrent_dim = int(lstm_hidden * 2)
        self.proj = nn.Identity() if int(recurrent_dim) == int(embed_dim) else nn.Linear(int(recurrent_dim), int(embed_dim))
        self.norm = nn.LayerNorm(int(embed_dim))

    def forward(self, x_bfw: "torch.Tensor", *, return_debug: bool = False):
        h = F.gelu(self.stem(x_bfw))
        for block in list(self.blocks):
            h = block(h)
        seq = self.pre_lstm_norm(h.permute(0, 2, 1).contiguous())
        out, _ = self.bilstm(seq)
        z = self.norm(self.proj(out[:, int(self.center_index), :]))
        if not bool(return_debug):
            return z
        return z, {"layer_maps": {"input": x_bfw, "hybrid_center": z[:, :, None]}}


class SecondsSequenceFrontend(TorchModuleBase):
    """Apply the sibling local frontend family on token-centered windows in seconds."""

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
        require_torch()
        super().__init__()
        variant_eff = str(variant or "").strip().lower()
        if str(variant_eff) not in set(VALID_VARIANTS):
            raise ValueError(f"unsupported variant={variant!r}")
        output_eff = str(output_kind or DEFAULT_FRONTEND_OUTPUT).strip().lower()
        if str(output_eff) not in set(VALID_FRONTEND_OUTPUTS):
            raise ValueError(f"unsupported output_kind={output_kind!r}")
        self.variant = str(variant_eff)
        self.input_dim_source = int(input_dim)
        self.class_id_vocab_sizes = tuple(int(max(0, int(x))) for x in list(class_id_vocab_sizes or ()))
        self.source_feature_names = tuple(str(x) for x in list(source_feature_names or ()))
        self.class_names = tuple(str(x) for x in list(class_names or ())) or tuple(
            _default_class_names(
                source_feature_names=self.source_feature_names,
                class_id_vocab_sizes=self.class_id_vocab_sizes,
            )
        )
        self.id_extra_dim = int(sum(int(vocab) for vocab in list(self.class_id_vocab_sizes) if int(vocab) > 1))
        self.raw_input_dim = int(self.input_dim_source) + int(self.id_extra_dim)
        self.class_local_fusion = bool(class_local_fusion)
        self.class_local_dim = int(max(1, int(class_local_dim)))
        self.class_local_mixer: Optional[ClassLocalFeatureMixer]
        if bool(self.class_local_fusion):
            self.class_local_mixer = ClassLocalFeatureMixer(
                source_dim=int(self.input_dim_source),
                class_id_vocab_sizes=self.class_id_vocab_sizes,
                class_names=self.class_names,
                source_feature_names=self.source_feature_names,
                class_local_dim=int(self.class_local_dim),
            )
            self.input_dim = int(self.class_local_mixer.output_dim)
            self.input_feature_names = tuple(self.class_local_mixer.output_feature_names)
        else:
            self.class_local_mixer = None
            self.input_dim = int(self.raw_input_dim)
            self.input_feature_names = tuple(
                resolve_frontend_input_feature_names(
                    input_dim_source=int(self.input_dim_source),
                    class_id_vocab_sizes=self.class_id_vocab_sizes,
                    source_feature_names=self.source_feature_names,
                    class_names=self.class_names,
                    class_local_fusion=False,
                )
            )
        self.embed_dim = int(embed_dim)
        self.window_radius = int(max(0, int(window_radius)))
        self.window_len = int((2 * self.window_radius) + 1)
        self.padding_mode = str(padding_mode or "reflect").strip().lower()
        self.output_kind = str(output_eff)
        self.chunk_size = int(max(0, int(chunk_size)))
        self.step_seconds = float(step_seconds)
        self.output_dim = int(resolve_frontend_output_dim(embed_dim=int(embed_dim), output_kind=str(output_eff)))
        nearest_feature_mask = infer_nearest_feature_mask_source(
            input_dim_source=int(self.input_dim_source),
            class_id_vocab_sizes=self.class_id_vocab_sizes,
            source_feature_names=self.source_feature_names,
        )
        self.register_buffer("nearest_feature_mask_source", nearest_feature_mask, persistent=False)
        center_index = int(self.window_radius)
        if str(self.variant) == "linear":
            self.encoder = LinearWindowEncoder(input_dim=int(self.input_dim) * int(self.window_len), embed_dim=int(embed_dim))
        elif str(self.variant) == "cnn":
            self.encoder = TemporalCNNEncoder(
                input_dim=int(self.input_dim),
                embed_dim=int(embed_dim),
                center_index=int(center_index),
                padding_mode=str(self.padding_mode),
            )
        elif str(self.variant) == "bilstm":
            self.encoder = BiLSTMWindowEncoder(
                input_dim=int(self.input_dim),
                embed_dim=int(embed_dim),
                center_index=int(center_index),
                num_layers=1,
            )
        else:
            self.encoder = HybridTCNBiLSTMEncoder(
                input_dim=int(self.input_dim),
                embed_dim=int(embed_dim),
                center_index=int(center_index),
                padding_mode=str(self.padding_mode),
            )
        self.norm = nn.LayerNorm(int(embed_dim))

    def _extract_windows(
        self,
        *,
        grid_bft: "torch.Tensor",
        grid_ids_bct: Optional["torch.Tensor"],
        grid_times_sec_bt: "torch.Tensor",
        token_times_sec_bt: "torch.Tensor",
        grid_valid_mask_bt: Optional["torch.Tensor"],
        valid_mask_bt: Optional["torch.Tensor"],
    ) -> "torch.Tensor":
        windows_btfw = sample_grid_windows_in_seconds(
            grid_bft=grid_bft,
            grid_times_sec_bt=grid_times_sec_bt,
            token_times_sec_bt=token_times_sec_bt,
            window_radius=int(self.window_radius),
            step_seconds=float(self.step_seconds),
            grid_valid_mask_bt=grid_valid_mask_bt,
            valid_mask_bt=valid_mask_bt,
            nearest_feature_mask_f=self.nearest_feature_mask_source,
        )
        if int(self.id_extra_dim) <= 0:
            return windows_btfw.contiguous()
        id_extra_btfw: Optional["torch.Tensor"] = None
        if grid_ids_bct is not None and int(grid_ids_bct.shape[1]) > 0:
            sampled_ids = sample_grid_id_windows_in_seconds(
                grid_ids_bct=grid_ids_bct,
                grid_times_sec_bt=grid_times_sec_bt,
                token_times_sec_bt=token_times_sec_bt,
                window_radius=int(self.window_radius),
                step_seconds=float(self.step_seconds),
                grid_valid_mask_bt=grid_valid_mask_bt,
                valid_mask_bt=valid_mask_bt,
            )
            id_extra_btfw = expand_sampled_id_windows_onehot(
                sampled_ids,
                class_id_vocab_sizes=self.class_id_vocab_sizes,
            )
        if id_extra_btfw is None:
            id_extra_btfw = torch.zeros(
                (int(windows_btfw.shape[0]), int(windows_btfw.shape[1]), int(self.id_extra_dim), int(windows_btfw.shape[3])),
                dtype=windows_btfw.dtype,
                device=windows_btfw.device,
            )
        raw_windows = torch.cat([windows_btfw, id_extra_btfw.to(dtype=windows_btfw.dtype)], dim=2).contiguous()
        if self.class_local_mixer is None:
            return raw_windows
        return self.class_local_mixer(raw_windows)

    def _encode_windows(self, windows_bfw: "torch.Tensor") -> "torch.Tensor":
        base = self.encoder(windows_bfw, return_debug=False)
        feat = F.normalize(self.norm(base), dim=-1, eps=float(FRONTEND_FEATURE_NORMALIZE_EPS))
        if str(self.output_kind) == "base":
            return base
        return feat

    def forward(
        self,
        grid_bft: "torch.Tensor",
        *,
        grid_ids_bct: Optional["torch.Tensor"] = None,
        grid_times_sec_bt: "torch.Tensor",
        token_times_sec_bt: "torch.Tensor",
        grid_valid_mask_bt: Optional["torch.Tensor"] = None,
        valid_mask_bt: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        windows_btfw = self._extract_windows(
            grid_bft=grid_bft,
            grid_ids_bct=grid_ids_bct,
            grid_times_sec_bt=grid_times_sec_bt,
            token_times_sec_bt=token_times_sec_bt,
            grid_valid_mask_bt=grid_valid_mask_bt,
            valid_mask_bt=valid_mask_bt,
        )
        batch_size, time_steps = int(windows_btfw.shape[0]), int(windows_btfw.shape[1])
        flat_bfw = windows_btfw.view(batch_size * time_steps, int(windows_btfw.shape[2]), int(windows_btfw.shape[3]))
        if int(self.chunk_size) > 0 and int(flat_bfw.shape[0]) > int(self.chunk_size):
            parts: List["torch.Tensor"] = []
            for start in range(0, int(flat_bfw.shape[0]), int(self.chunk_size)):
                end = min(int(flat_bfw.shape[0]), int(start) + int(self.chunk_size))
                parts.append(self._encode_windows(flat_bfw[int(start) : int(end)]))
            out_bd = torch.cat(parts, dim=0)
        else:
            out_bd = self._encode_windows(flat_bfw)
        return out_bd.view(batch_size, time_steps, int(out_bd.shape[-1]))


class SecondsMultiScaleFrontend(TorchModuleBase):
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
        require_torch()
        super().__init__()
        radii = sorted({int(x) for x in list(window_radii) if int(x) >= 0})
        primary_eff = int(primary_radius)
        if int(primary_eff) not in set(radii):
            radii.append(int(primary_eff))
            radii = sorted(set(radii))
        self.variant = str(variant or "").strip().lower()
        self.input_dim_source = int(input_dim)
        self.class_id_vocab_sizes = tuple(int(max(0, int(x))) for x in list(class_id_vocab_sizes or ()))
        self.source_feature_names = tuple(str(x) for x in list(source_feature_names or ()))
        self.class_names = tuple(str(x) for x in list(class_names or ())) or tuple(
            _default_class_names(
                source_feature_names=self.source_feature_names,
                class_id_vocab_sizes=self.class_id_vocab_sizes,
            )
        )
        self.embed_dim = int(embed_dim)
        self.padding_mode = str(padding_mode or "reflect").strip().lower()
        self.output_kind = str(output_kind or DEFAULT_FRONTEND_OUTPUT).strip().lower()
        self.chunk_size = int(chunk_size)
        self.step_seconds = float(step_seconds)
        self.window_radii = tuple(int(x) for x in list(radii))
        self.primary_radius = int(primary_eff)
        self.class_local_fusion = bool(class_local_fusion)
        self.class_local_dim = int(max(1, int(class_local_dim)))
        self.frontends = nn.ModuleDict(
            {
                str(int(radius)): SecondsSequenceFrontend(
                    input_dim=int(input_dim),
                    class_id_vocab_sizes=tuple(self.class_id_vocab_sizes),
                    source_feature_names=tuple(self.source_feature_names),
                    class_names=tuple(self.class_names),
                    variant=str(variant),
                    embed_dim=int(embed_dim),
                    window_radius=int(radius),
                    padding_mode=str(padding_mode),
                    output_kind=str(output_kind),
                    chunk_size=int(chunk_size),
                    step_seconds=float(step_seconds),
                    class_local_fusion=bool(self.class_local_fusion),
                    class_local_dim=int(self.class_local_dim),
                )
                for radius in list(self.window_radii)
            }
        )
        self.input_dim = int(self.frontends[str(int(self.primary_radius))].input_dim)
        self.output_dim = int(self.frontends[str(int(self.primary_radius))].output_dim)
        self.input_feature_names = tuple(self.frontends[str(int(self.primary_radius))].input_feature_names)

    def iter_frontends(self) -> Iterable[Tuple[int, SecondsSequenceFrontend]]:
        for radius in list(self.window_radii):
            yield int(radius), self.frontends[str(int(radius))]

    def forward_multiscale(
        self,
        grid_bft: "torch.Tensor",
        *,
        grid_ids_bct: Optional["torch.Tensor"] = None,
        grid_times_sec_bt: "torch.Tensor",
        token_times_sec_bt: "torch.Tensor",
        grid_valid_mask_bt: Optional["torch.Tensor"] = None,
        valid_mask_bt: Optional["torch.Tensor"] = None,
    ) -> Dict[int, "torch.Tensor"]:
        return {
            int(radius): frontend(
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
        grid_bft: "torch.Tensor",
        *,
        grid_ids_bct: Optional["torch.Tensor"] = None,
        grid_times_sec_bt: "torch.Tensor",
        token_times_sec_bt: "torch.Tensor",
        grid_valid_mask_bt: Optional["torch.Tensor"] = None,
        valid_mask_bt: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        return self.frontends[str(int(self.primary_radius))](
            grid_bft,
            grid_ids_bct=grid_ids_bct,
            grid_times_sec_bt=grid_times_sec_bt,
            token_times_sec_bt=token_times_sec_bt,
            grid_valid_mask_bt=grid_valid_mask_bt,
            valid_mask_bt=valid_mask_bt,
        )


def build_seconds_frontend_from_cfg(frontend_cfg: Optional[Dict[str, Any]]) -> Optional[Any]:
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
    if bool(multiscale_enabled):
        if not multiscale_radii:
            multiscale_radii = [int(primary_radius)]
        return SecondsMultiScaleFrontend(
            window_radii=list(multiscale_radii),
            primary_radius=int(primary_radius),
            **common_kwargs,
        )
    return SecondsSequenceFrontend(
        window_radius=int(cfg.get("window_radius", primary_radius)),
        **common_kwargs,
    )
