#!/usr/bin/env python3
"""Shared symbolic-conditioning ablations for frontend sensitivity checks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


VALID_CONDITIONING_ABLATIONS = ("none", "zero", "shuffle", "phase_shift", "family_swap")


def normalize_conditioning_ablation(mode: str | None) -> str:
    mode_eff = str(mode or "none").strip().lower().replace("-", "_")
    aliases = {
        "off": "none",
        "identity": "none",
        "zeros": "zero",
        "zero_cond": "zero",
        "shuffled": "shuffle",
        "batch_shuffle": "shuffle",
        "phase": "phase_shift",
        "timeshift": "phase_shift",
        "time_shift": "phase_shift",
        "family": "family_swap",
        "familyswap": "family_swap",
    }
    mode_eff = aliases.get(mode_eff, mode_eff)
    if mode_eff not in set(VALID_CONDITIONING_ABLATIONS):
        raise ValueError(
            f"unsupported conditioning ablation {mode!r}; "
            f"expected one of {', '.join(VALID_CONDITIONING_ABLATIONS)}"
        )
    return mode_eff


def conditioning_ablation_help() -> str:
    return (
        "Frontend conditioning ablation: none keeps the batch unchanged; zero clears grid values and IDs; "
        "shuffle cyclically pairs each target with another example's grid/time conditioning; phase_shift "
        "rolls each grid in time while preserving its marginal events; family_swap reverses drum-family lanes."
    )


def _clone_mapping(batch: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in dict(batch).items()}


def _clone_tensor(value: Any) -> torch.Tensor | None:
    if value is None or not torch.is_tensor(value):
        return None
    return value.clone().contiguous()


def _roll_batch_tensor(value: Any, *, shift: int) -> torch.Tensor | None:
    tensor = _clone_tensor(value)
    if tensor is None or int(tensor.dim()) <= 0 or int(tensor.shape[0]) <= 1:
        return tensor
    return torch.roll(tensor, shifts=int(shift), dims=0).contiguous()


def _valid_lengths(mask: torch.Tensor | None, *, batch_size: int, fallback: int) -> list[int]:
    if mask is None or int(mask.dim()) != 2:
        return [int(fallback)] * int(batch_size)
    lengths = torch.as_tensor(mask, dtype=torch.bool).sum(dim=1).detach().cpu().tolist()
    return [max(1, min(int(fallback), int(length))) for length in lengths]


def _phase_shift_tensor(value: Any, *, grid_valid_mask: torch.Tensor | None) -> torch.Tensor | None:
    tensor = _clone_tensor(value)
    if tensor is None or int(tensor.dim()) < 2:
        return tensor
    batch_size = int(tensor.shape[0])
    time_dim = int(tensor.dim()) - 1
    time_len = int(tensor.shape[-1])
    lengths = _valid_lengths(grid_valid_mask, batch_size=batch_size, fallback=time_len)
    for batch_idx, valid_len in enumerate(lengths):
        if int(valid_len) <= 1:
            continue
        shift = max(1, int(valid_len) // 2)
        view = tensor[int(batch_idx), ..., : int(valid_len)]
        tensor[int(batch_idx), ..., : int(valid_len)] = torch.roll(view, shifts=int(shift), dims=int(time_dim - 1))
    return tensor.contiguous()


def _swap_family_tensor(value: Any, *, family_count: int | None = None) -> torch.Tensor | None:
    tensor = _clone_tensor(value)
    if tensor is None or int(tensor.dim()) < 3:
        return tensor
    channels = int(tensor.shape[1])
    if family_count is None:
        family_count = int(channels)
    family_count = int(max(0, int(family_count)))
    if family_count <= 1:
        return tensor
    if int(channels) == int(family_count):
        return tensor.flip(dims=(1,)).contiguous()
    if int(channels) % int(family_count) != 0:
        return tensor
    lanes_per_family = int(channels) // int(family_count)
    shape = list(tensor.shape)
    reshaped = tensor.reshape(shape[0], int(family_count), int(lanes_per_family), *shape[2:])
    return reshaped.flip(dims=(1,)).reshape(shape).contiguous()


def apply_conditioning_ablation(
    batch: Mapping[str, Any],
    mode: str | None,
    *,
    batch_index: int = 0,
) -> dict[str, Any]:
    """Return a shallow batch copy with symbolic frontend conditioning ablated."""
    mode_eff = normalize_conditioning_ablation(mode)
    out = _clone_mapping(batch)
    if mode_eff == "none":
        return out

    grid = out.get("grid")
    if not torch.is_tensor(grid):
        raise KeyError("conditioning ablation requires batch['grid'] to be a tensor")
    batch_size = int(grid.shape[0])

    if mode_eff == "zero":
        out["grid"] = torch.zeros_like(grid).contiguous()
        if torch.is_tensor(out.get("grid_ids")):
            out["grid_ids"] = torch.full_like(out["grid_ids"], -1).contiguous()
        if torch.is_tensor(out.get("family_onsets_bft")):
            out["family_onsets_bft"] = torch.zeros_like(out["family_onsets_bft"]).contiguous()
        return out

    if mode_eff == "shuffle":
        if int(batch_size) > 1:
            shift = 1 + (int(batch_index) % max(1, int(batch_size) - 1))
            for key in ("grid", "grid_ids", "grid_times_sec", "grid_valid_mask", "family_onsets_bft"):
                if torch.is_tensor(out.get(key)):
                    out[key] = _roll_batch_tensor(out[key], shift=int(shift))
            return out
        mode_eff = "phase_shift"

    if mode_eff == "phase_shift":
        grid_valid_mask = out.get("grid_valid_mask") if torch.is_tensor(out.get("grid_valid_mask")) else None
        for key in ("grid", "grid_ids", "family_onsets_bft"):
            if torch.is_tensor(out.get(key)):
                out[key] = _phase_shift_tensor(out[key], grid_valid_mask=grid_valid_mask)
        return out

    if mode_eff == "family_swap":
        family_count = int(out["grid_ids"].shape[1]) if torch.is_tensor(out.get("grid_ids")) else None
        out["grid"] = _swap_family_tensor(out["grid"], family_count=family_count)
        if torch.is_tensor(out.get("grid_ids")):
            out["grid_ids"] = _swap_family_tensor(out["grid_ids"])
        if torch.is_tensor(out.get("family_onsets_bft")):
            out["family_onsets_bft"] = _swap_family_tensor(out["family_onsets_bft"])
        return out

    raise AssertionError(f"unhandled conditioning ablation: {mode_eff}")
