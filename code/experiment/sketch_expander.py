from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.diffusion_cache_utils import FAMILY_STATE_FAMILY_NAMES, FAMILY_STATE_ID_VOCAB_SIZES
from data.sketch_dataset import (
    DEFAULT_NUM_STEPS,
    DEFAULT_SKETCH_MAX_SLOTS,
    HIHAT_CLOSED_CLASS_IDS,
    HIHAT_OPEN_CLASS_IDS,
    HIHAT_PEDAL_CLASS_IDS,
    KICK_GHOST_VELOCITY_THRESHOLD,
    ORNAMENT_BUDGET_GROUP_NAMES,
    ORNAMENT_BUDGET_MAX_COUNTS,
    ORNAMENT_BUDGET_GROUP_NAMES_V2,
    ORNAMENT_BUDGET_MAX_COUNTS_V2,
    FEEL_STYLE_VALUES,
    FILL_ACCENT_SHAPE_VALUES,
    SKETCH_CONTROL_NAMES,
    SKETCH_FAMILY_NAMES,
    SNARE_BACKBEAT_STEPS,
    SNARE_GHOST_VELOCITY_THRESHOLD,
    SNARE_STRONG_VELOCITY_THRESHOLD,
    TOM_DIRECTION_VALUES,
    decode_sketch_controls,
    is_legacy_sketch_control_names,
    is_v3_sketch_control_names,
    is_v4_sketch_control_names,
)


DEFAULT_CLASS_ID_BY_FAMILY: dict[str, int] = {
    "kick": 0,
    "snare": 0,
    "tom_high": 0,
    "tom_mid": 0,
    "tom_floor": 0,
    "hihat": 2,
    "crash": 0,
    "ride": 0,
}


@dataclass
class SketchExpanderConfig:
    num_steps: int = DEFAULT_NUM_STEPS
    max_slots: int = DEFAULT_SKETCH_MAX_SLOTS
    sketch_family_names: tuple[str, ...] = SKETCH_FAMILY_NAMES
    class_names: tuple[str, ...] = FAMILY_STATE_FAMILY_NAMES
    class_id_vocab_sizes: tuple[int, ...] = FAMILY_STATE_ID_VOCAB_SIZES
    control_names: tuple[str, ...] = SKETCH_CONTROL_NAMES
    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    budget_group_names: tuple[str, ...] = ORNAMENT_BUDGET_GROUP_NAMES
    budget_max_counts: tuple[int, ...] = ORNAMENT_BUDGET_MAX_COUNTS
    fill_start_classes: int = DEFAULT_NUM_STEPS + 1
    fill_length_classes: int = DEFAULT_NUM_STEPS + 1
    tom_direction_classes: int = len(TOM_DIRECTION_VALUES)
    fill_accent_shape_classes: int = len(FILL_ACCENT_SHAPE_VALUES)


def _sinusoidal_positions(length: int, dim: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    half = int(dim) // 2
    div_term = torch.exp(
        torch.arange(0, half, device=device, dtype=torch.float32) * (-math.log(10000.0) / max(1, half))
    )
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, :half] = torch.sin(position * div_term)
    pe[:, half : 2 * half] = torch.cos(position * div_term)
    return pe


class SketchExpander(nn.Module):
    def __init__(self, cfg: SketchExpanderConfig):
        super().__init__()
        self.cfg = cfg
        self.num_steps = int(cfg.num_steps)
        self.max_slots = int(cfg.max_slots)
        self.class_names = tuple(str(x) for x in cfg.class_names)
        self.sketch_family_names = tuple(str(x) for x in cfg.sketch_family_names)
        self.class_id_vocab_sizes = tuple(int(x) for x in cfg.class_id_vocab_sizes)
        if int(len(self.class_names)) != int(len(self.class_id_vocab_sizes)):
            raise ValueError("class_names and class_id_vocab_sizes must have the same length")
        self.budget_group_names = tuple(str(x) for x in cfg.budget_group_names)
        self.budget_max_counts = tuple(int(x) for x in cfg.budget_max_counts)
        if int(len(self.budget_group_names)) != int(len(self.budget_max_counts)):
            raise ValueError("budget_group_names and budget_max_counts must have the same length")
        input_dim = int(2 * len(self.sketch_family_names) + len(tuple(cfg.control_names)))
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, int(cfg.d_model)),
            nn.GELU(),
            nn.Dropout(float(cfg.dropout)),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=int(cfg.d_model),
            nhead=int(cfg.num_heads),
            dim_feedforward=int(round(float(cfg.mlp_ratio) * float(cfg.d_model))),
            dropout=float(cfg.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(cfg.num_layers))
        self.family_embed = nn.Embedding(int(len(self.class_names)), int(cfg.d_model))
        self.slot_embed = nn.Embedding(int(self.max_slots), int(cfg.d_model))
        self.event_norm = nn.LayerNorm(int(cfg.d_model))
        self.presence_head = nn.Linear(int(cfg.d_model), 1)
        self.velocity_head = nn.Linear(int(cfg.d_model), 1)
        self.offset_head = nn.Linear(int(cfg.d_model), 1)
        self.class_head = nn.Linear(int(cfg.d_model), int(max(self.class_id_vocab_sizes)))
        budget_class_count = int(max(self.budget_max_counts, default=0)) + 1
        self.budget_count_head = (
            nn.Linear(int(cfg.d_model), int(len(self.budget_group_names)) * int(budget_class_count))
            if int(len(self.budget_group_names)) > 0
            else None
        )
        self.fill_start_head = nn.Linear(int(cfg.d_model), int(max(1, cfg.fill_start_classes)))
        self.fill_length_head = nn.Linear(int(cfg.d_model), int(max(1, cfg.fill_length_classes)))
        self.tom_direction_head = nn.Linear(int(cfg.d_model), int(max(1, cfg.tom_direction_classes)))
        self.fill_accent_shape_head = nn.Linear(int(cfg.d_model), int(max(1, cfg.fill_accent_shape_classes)))

    def forward(
        self,
        sketch_hits: torch.Tensor,
        sketch_vel: torch.Tensor,
        controls: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hits = torch.as_tensor(sketch_hits, dtype=torch.float32)
        vel = torch.as_tensor(sketch_vel, dtype=torch.float32, device=hits.device)
        ctrl = torch.as_tensor(controls, dtype=torch.float32, device=hits.device)
        if int(hits.dim()) != 3:
            raise ValueError(f"sketch_hits must be [B,3,16], got {tuple(hits.shape)}")
        if tuple(vel.shape) != tuple(hits.shape):
            raise ValueError(f"sketch_vel must match sketch_hits, got {tuple(vel.shape)} / {tuple(hits.shape)}")
        if int(ctrl.dim()) != 2 or int(ctrl.shape[0]) != int(hits.shape[0]):
            raise ValueError(f"controls must be [B,C], got {tuple(ctrl.shape)}")
        if int(hits.shape[1]) != int(len(self.sketch_family_names)) or int(hits.shape[2]) != int(self.num_steps):
            raise ValueError(
                f"expected sketch shape [B,{len(self.sketch_family_names)},{self.num_steps}], got {tuple(hits.shape)}"
            )

        batch_size = int(hits.shape[0])
        step_input = torch.cat(
            [
                hits.transpose(1, 2).contiguous(),
                vel.transpose(1, 2).contiguous(),
                ctrl[:, None, :].expand(-1, int(self.num_steps), -1),
            ],
            dim=-1,
        )
        x = self.input_proj(step_input)
        x = x + _sinusoidal_positions(int(self.num_steps), int(self.cfg.d_model), x.device).unsqueeze(0)
        h = self.encoder(x)

        family_ids = torch.arange(int(len(self.class_names)), device=h.device, dtype=torch.long)
        slot_ids = torch.arange(int(self.max_slots), device=h.device, dtype=torch.long)
        family_emb = self.family_embed(family_ids).view(1, 1, int(len(self.class_names)), 1, int(self.cfg.d_model))
        slot_emb = self.slot_embed(slot_ids).view(1, 1, 1, int(self.max_slots), int(self.cfg.d_model))
        z = h.view(batch_size, int(self.num_steps), 1, 1, int(self.cfg.d_model)) + family_emb + slot_emb
        z = self.event_norm(z)
        # Network works [B,T,F,S,D]; public tensors use [B,F,T,S].
        presence_logits = self.presence_head(z).squeeze(-1).permute(0, 2, 1, 3).contiguous()
        velocity = torch.sigmoid(self.velocity_head(z).squeeze(-1)).permute(0, 2, 1, 3).contiguous()
        offset = (0.5 * torch.tanh(self.offset_head(z).squeeze(-1))).permute(0, 2, 1, 3).contiguous()
        class_logits = self.class_head(z).permute(0, 2, 1, 3, 4).contiguous()
        pooled = h.mean(dim=1)
        if self.budget_count_head is not None:
            budget_class_count = int(max(self.budget_max_counts, default=0)) + 1
            budget_logits = self.budget_count_head(pooled).view(
                batch_size,
                int(len(self.budget_group_names)),
                int(budget_class_count),
            ).contiguous()
        else:
            budget_logits = h.new_zeros((batch_size, 0, 0))
        return {
            "presence_logits": presence_logits,
            "velocity": velocity,
            "offset": offset,
            "class_logits": class_logits,
            "budget_logits": budget_logits,
            "fill_start_logits": self.fill_start_head(pooled).contiguous(),
            "fill_length_logits": self.fill_length_head(pooled).contiguous(),
            "tom_direction_logits": self.tom_direction_head(pooled).contiguous(),
            "fill_accent_shape_logits": self.fill_accent_shape_head(pooled).contiguous(),
        }

    def to_config_dict(self) -> dict[str, Any]:
        payload = asdict(self.cfg)
        payload["sketch_family_names"] = list(self.sketch_family_names)
        payload["class_names"] = list(self.class_names)
        payload["class_id_vocab_sizes"] = [int(x) for x in self.class_id_vocab_sizes]
        payload["control_names"] = list(self.cfg.control_names)
        payload["budget_group_names"] = list(self.budget_group_names)
        payload["budget_max_counts"] = [int(x) for x in self.budget_max_counts]
        return payload


def _max_counts_for_budget_names(
    budget_group_names: Sequence[str] | None = None,
    budget_max_counts: Sequence[int] | None = None,
) -> dict[str, int]:
    names = tuple(str(name) for name in list(budget_group_names or ORNAMENT_BUDGET_GROUP_NAMES))
    if budget_max_counts is not None and int(len(tuple(budget_max_counts))) == int(len(names)):
        return {name: int(max_count) for name, max_count in zip(names, budget_max_counts, strict=True)}
    if names == ORNAMENT_BUDGET_GROUP_NAMES_V2:
        return {
            name: int(max_count)
            for name, max_count in zip(ORNAMENT_BUDGET_GROUP_NAMES_V2, ORNAMENT_BUDGET_MAX_COUNTS_V2, strict=True)
        }
    default_counts = {
        name: int(max_count)
        for name, max_count in zip(ORNAMENT_BUDGET_GROUP_NAMES, ORNAMENT_BUDGET_MAX_COUNTS, strict=True)
    }
    legacy_counts = {
        name: int(max_count)
        for name, max_count in zip(ORNAMENT_BUDGET_GROUP_NAMES_V2, ORNAMENT_BUDGET_MAX_COUNTS_V2, strict=True)
    }
    return {name: int(default_counts.get(name, legacy_counts.get(name, 0))) for name in names}


def _cap_from_thresholds(value: float, thresholds: Sequence[tuple[float, int]]) -> int:
    if not thresholds:
        return 0
    value_f = float(value)
    for upper_bound, cap in thresholds:
        if value_f < float(upper_bound):
            return int(cap)
    return int(thresholds[-1][1])


def _budget_caps_from_decoded_controls(
    decoded: Mapping[str, Any],
    *,
    variation: float = 0.0,
    budget_group_names: Sequence[str] | None = None,
    budget_max_counts: Sequence[int] | None = None,
) -> dict[str, int]:
    variation_amount = float(max(0.0, min(1.0, float(variation))))
    names = tuple(str(name) for name in list(budget_group_names or ORNAMENT_BUDGET_GROUP_NAMES))
    max_counts = _max_counts_for_budget_names(names, budget_max_counts)
    version = str(decoded.get("version", "v1" if bool(decoded.get("legacy", False)) else "v2"))
    if version in {"v1", "v3", "v4"}:
        ghost_density = float(decoded.get("ghost_density", 0.0))
        snare_ghost_density = float(decoded.get("snare_ghost_density", ghost_density))
        snare_roll_density = float(decoded.get("snare_roll_density", snare_ghost_density))
        kick_ghost_density = float(decoded.get("kick_ghost_density", max(0.0, (ghost_density - 0.25) / 0.75)))
        hihat_density = float(decoded.get("hihat_density", 0.5))
        hihat_openness = float(decoded.get("hihat_openness", 0.0))
        fill_density = float(decoded.get("fill_density", 0.0))
        kick_ghost = _cap_from_thresholds(kick_ghost_density, ((0.20, 0), (0.60, 1), (float("inf"), 2)))
        snare_ghost = _cap_from_thresholds(
            snare_ghost_density,
            ((0.18, 0), (0.35, 1), (0.55, 2), (0.80, 3), (float("inf"), 5)),
        )
        snare_roll = _cap_from_thresholds(
            snare_roll_density,
            ((0.20, 0), (0.45, 1), (0.70, 2), (float("inf"), 4)),
        )
        off16_hat = _cap_from_thresholds(
            hihat_density,
            ((0.10, 0), (0.35, 1), (0.80, 2), (0.90, 4), (float("inf"), 8)),
        )
        open_hat = _cap_from_thresholds(
            hihat_openness,
            ((0.12, 0), (0.20, 1), (0.35, 2), (0.55, 1), (0.75, 4), (float("inf"), 5)),
        )
        tom_fill = _cap_from_thresholds(
            fill_density,
            ((0.20, 0), (0.35, 1), (0.50, 2), (0.70, 3), (float("inf"), 7)),
        )
        crash = 0 if fill_density <= 0.25 else (1 if fill_density < 0.75 else 2)
        fill_role = str(decoded.get("fill_role", ""))
        if version == "v4" and fill_role == "ride":
            ride = _cap_from_thresholds(fill_density, ((0.20, 2), (0.60, 6), (float("inf"), 8)))
        elif version == "v4" and fill_role == "ride_plus_fill":
            ride = _cap_from_thresholds(fill_density, ((0.20, 2), (0.70, 5), (float("inf"), 7)))
        else:
            ride = _cap_from_thresholds(fill_density, ((0.45, 0), (0.70, 6), (0.85, 3), (float("inf"), 2)))
        tom_crash = max(tom_fill, crash)
    else:
        snare_style = str(decoded.get("snare_style", "plain"))
        hat_rate = str(decoded.get("hat_rate", "eighths"))
        hat_color = str(decoded.get("hat_color", "closed"))
        fill_role = str(decoded.get("fill_role", "none"))
        snare_caps = {
            "plain": (0, 0),
            "ghosts": (2, 0),
            "drags_rolls": (0, 2),
            "ghosts_plus_rolls": (4, 3),
        }
        snare_ghost, snare_roll = snare_caps.get(snare_style, (0, 0))
        off16_hat = {
            "none": 0,
            "eighths": 1,
            "sparse_syncopated": 3,
            "sixteenths": 8,
        }.get(hat_rate, 1)
        open_hat = {"closed": 0, "pedal": 0, "open": 4}.get(hat_color, 0)
        if fill_role == "ride":
            ride, tom_crash = 8, 0
        elif fill_role == "tom_crash_fill":
            ride, tom_crash = 0, 7
        elif fill_role == "ride_plus_fill":
            ride, tom_crash = 7, 7
        else:
            ride, tom_crash = 0, 0
        ghost_density = float(decoded.get("ghost_density", 0.0))
        kick_ghost_density = float(decoded.get("kick_ghost_density", max(0.0, (ghost_density - 0.25) / 0.75)))
        kick_ghost = 0 if kick_ghost_density < 0.20 else (1 if kick_ghost_density < 0.60 else 2)
        tom_fill = tom_crash
        crash = 0 if tom_crash <= 0 else (1 if tom_crash < 7 else 2)

    headroom = 1 if variation_amount >= 0.60 else 0
    kick_headroom = 0
    if kick_ghost >= 2 and variation_amount >= 0.60:
        kick_headroom = 1
    if kick_ghost >= 2 and variation_amount >= 0.85:
        kick_headroom = 2
    caps = {
        "kick_ghost": kick_ghost + kick_headroom,
        "snare_ghost": snare_ghost + (headroom if snare_ghost > 0 else 0),
        "snare_roll_drag": snare_roll + (headroom if snare_roll > 0 else 0),
        "snare_roll_run": snare_roll + (headroom if snare_roll > 0 else 0),
        "off16_hat": off16_hat,
        "open_hat": open_hat + (2 if variation_amount >= 0.50 and open_hat > 0 else 0),
        "tom_fill": tom_fill + (headroom if tom_fill > 0 else 0),
        "crash": crash + (1 if variation_amount >= 0.50 and crash > 0 else 0),
        "ride": ride + (headroom if ride > 0 else 0),
        "tom_crash": tom_crash + (headroom if tom_crash > 0 else 0),
    }
    return {
        name: int(max(0, min(int(caps.get(name, 0)), int(max_counts.get(name, 0)))))
        for name in names
    }


def _batched_ornament_target_masks(
    target_bool: torch.Tensor,
    target_velocity: torch.Tensor,
    target_class_id: torch.Tensor,
    sketch_hits: torch.Tensor | None,
    budget_group_names: Sequence[str] | None = None,
) -> dict[str, torch.Tensor]:
    device = target_bool.device
    group_names = tuple(str(name) for name in list(budget_group_names or ORNAMENT_BUDGET_GROUP_NAMES))
    masks = {name: torch.zeros_like(target_bool, dtype=torch.bool) for name in group_names}
    if int(target_bool.shape[1]) < int(len(FAMILY_STATE_FAMILY_NAMES)):
        return masks
    family_to_idx = {name: idx for idx, name in enumerate(FAMILY_STATE_FAMILY_NAMES)}
    kick_idx = int(family_to_idx["kick"])
    snare_idx = int(family_to_idx["snare"])
    hihat_idx = int(family_to_idx["hihat"])
    steps = int(target_bool.shape[2])
    slots = int(target_bool.shape[3])
    kick_anchor = torch.zeros((target_bool.shape[0], steps, slots), dtype=torch.bool, device=device)
    if sketch_hits is not None and int(slots) > 0 and "kick" in SKETCH_FAMILY_NAMES:
        sketch = torch.as_tensor(sketch_hits, dtype=torch.float32, device=device).gt(0.5)
        if int(sketch.dim()) == 3:
            kick_anchor[:, :, 0] = sketch[:, int(SKETCH_FAMILY_NAMES.index("kick")), :steps]
    kick_active = target_bool[:, kick_idx]
    kick_extra = torch.zeros_like(kick_active)
    if int(slots) > 1:
        kick_extra[:, :, 1:] = kick_active[:, :, 1:]
    if "kick_ghost" in masks:
        masks["kick_ghost"][:, kick_idx] = (
            (
                kick_active
                & target_velocity[:, kick_idx].le(float(KICK_GHOST_VELOCITY_THRESHOLD))
            )
            | kick_extra
        ) & ~kick_anchor

    nonbackbeat = torch.ones((steps, 1), dtype=torch.bool, device=device)
    for backbeat_step in SNARE_BACKBEAT_STEPS:
        if 0 <= int(backbeat_step) < int(steps):
            nonbackbeat[int(backbeat_step), 0] = False
    snare_anchor = torch.zeros((target_bool.shape[0], steps, slots), dtype=torch.bool, device=device)
    if sketch_hits is not None and int(slots) > 0:
        sketch = torch.as_tensor(sketch_hits, dtype=torch.float32, device=device).gt(0.5)
        if int(sketch.dim()) == 3 and "snare" in SKETCH_FAMILY_NAMES:
            snare_sketch_idx = int(SKETCH_FAMILY_NAMES.index("snare"))
            snare_anchor[:, :, 0] = sketch[:, snare_sketch_idx, :steps]
    snare_active = target_bool[:, snare_idx]
    snare_ghost_local = (
        snare_active
        & target_class_id[:, snare_idx].eq(0)
        & target_velocity[:, snare_idx].le(float(SNARE_GHOST_VELOCITY_THRESHOLD))
        & nonbackbeat.view(1, steps, 1)
        & ~snare_anchor
    )
    if "snare_ghost" in masks:
        masks["snare_ghost"][:, snare_idx] = snare_ghost_local
    snare_roll = snare_active & ~snare_ghost_local & ~snare_anchor
    if int(slots) > 1:
        snare_roll[:, :, 0] = False
    else:
        snare_roll.zero_()
    if "snare_roll_drag" in masks:
        masks["snare_roll_drag"][:, snare_idx] = snare_roll
    snare_run_steps = (
        snare_active
        & target_velocity[:, snare_idx].ge(float(SNARE_STRONG_VELOCITY_THRESHOLD))
        & nonbackbeat.view(1, steps, 1)
    ).any(dim=-1)
    snare_adjacent = torch.zeros_like(snare_run_steps, dtype=torch.bool)
    if int(steps) > 1:
        snare_adjacent[:, :-1] |= snare_run_steps[:, 1:]
        snare_adjacent[:, 1:] |= snare_run_steps[:, :-1]
    snare_roll_run = (
        snare_active
        & ~snare_ghost_local
        & ~snare_anchor
        & nonbackbeat.view(1, steps, 1)
        & snare_adjacent.view(snare_active.shape[0], steps, 1)
        & target_velocity[:, snare_idx].ge(float(SNARE_STRONG_VELOCITY_THRESHOLD))
    )
    if int(slots) > 1:
        snare_roll_run[:, :, 1:] = False
    if "snare_roll_run" in masks:
        masks["snare_roll_run"][:, snare_idx] = snare_roll_run

    hihat_active = target_bool[:, hihat_idx]
    off16 = torch.zeros((steps, 1), dtype=torch.bool, device=device)
    off16[1::2, 0] = True
    if "off16_hat" in masks:
        masks["off16_hat"][:, hihat_idx] = hihat_active & off16.view(1, steps, 1)
    hihat_open = hihat_active & torch.zeros_like(hihat_active, dtype=torch.bool)
    for open_id in HIHAT_OPEN_CLASS_IDS:
        hihat_open |= hihat_active & target_class_id[:, hihat_idx].eq(int(open_id))
    if "open_hat" in masks:
        masks["open_hat"][:, hihat_idx] = hihat_open
    if "ride" in family_to_idx and "ride" in masks:
        masks["ride"][:, int(family_to_idx["ride"])] = target_bool[:, int(family_to_idx["ride"])]
    if "tom_fill" in masks:
        for family_name in ("tom_high", "tom_mid", "tom_floor"):
            if family_name in family_to_idx:
                masks["tom_fill"][:, int(family_to_idx[family_name])] = target_bool[:, int(family_to_idx[family_name])]
    if "crash" in masks and "crash" in family_to_idx:
        masks["crash"][:, int(family_to_idx["crash"])] = target_bool[:, int(family_to_idx["crash"])]
    if "tom_crash" in masks:
        for family_name in ("tom_high", "tom_mid", "tom_floor", "crash"):
            if family_name in family_to_idx:
                masks["tom_crash"][:, int(family_to_idx[family_name])] = target_bool[:, int(family_to_idx[family_name])]
    return masks


def _batched_ornament_candidate_masks(
    shape: torch.Size,
    *,
    sketch_hits: torch.Tensor | None,
    device: torch.device,
    budget_group_names: Sequence[str] | None = None,
) -> dict[str, torch.Tensor]:
    group_names = tuple(str(name) for name in list(budget_group_names or ORNAMENT_BUDGET_GROUP_NAMES))
    masks = {name: torch.zeros(shape, dtype=torch.bool, device=device) for name in group_names}
    if int(shape[1]) < int(len(FAMILY_STATE_FAMILY_NAMES)):
        return masks
    family_to_idx = {name: idx for idx, name in enumerate(FAMILY_STATE_FAMILY_NAMES)}
    batch_size, _family_count, steps, slots = [int(x) for x in shape]
    kick_idx = int(family_to_idx["kick"])
    snare_idx = int(family_to_idx["snare"])
    hihat_idx = int(family_to_idx["hihat"])
    nonbackbeat = torch.ones((steps,), dtype=torch.bool, device=device)
    for backbeat_step in SNARE_BACKBEAT_STEPS:
        if 0 <= int(backbeat_step) < int(steps):
            nonbackbeat[int(backbeat_step)] = False
    snare_anchor = torch.zeros((batch_size, steps), dtype=torch.bool, device=device)
    kick_anchor = torch.zeros((batch_size, steps), dtype=torch.bool, device=device)
    if sketch_hits is not None and "kick" in SKETCH_FAMILY_NAMES:
        sketch = torch.as_tensor(sketch_hits, dtype=torch.float32, device=device).gt(0.5)
        if int(sketch.dim()) == 3:
            kick_anchor = sketch[:, int(SKETCH_FAMILY_NAMES.index("kick")), :steps]
    if "kick_ghost" in masks:
        masks["kick_ghost"][:, kick_idx, :, 0] = ~kick_anchor
        if int(slots) > 1:
            masks["kick_ghost"][:, kick_idx, :, 1:] = True
    if sketch_hits is not None and "snare" in SKETCH_FAMILY_NAMES:
        sketch = torch.as_tensor(sketch_hits, dtype=torch.float32, device=device).gt(0.5)
        if int(sketch.dim()) == 3:
            snare_anchor = sketch[:, int(SKETCH_FAMILY_NAMES.index("snare")), :steps]
    if "snare_ghost" in masks:
        masks["snare_ghost"][:, snare_idx, :, 0] = nonbackbeat.view(1, steps) & ~snare_anchor
    if int(slots) > 1 and "snare_roll_drag" in masks:
        masks["snare_roll_drag"][:, snare_idx, :, 1:] = True
    if "snare_roll_run" in masks:
        masks["snare_roll_run"][:, snare_idx, :, 0] = nonbackbeat.view(1, steps) & ~snare_anchor
    off16 = torch.zeros((steps,), dtype=torch.bool, device=device)
    off16[1::2] = True
    if "off16_hat" in masks:
        masks["off16_hat"][:, hihat_idx, :, 0] = off16.view(1, steps)
    if "open_hat" in masks:
        masks["open_hat"][:, hihat_idx, :, 0] = True
    if "ride" in family_to_idx and "ride" in masks:
        masks["ride"][:, int(family_to_idx["ride"]), :, 0] = True
    if "tom_fill" in masks:
        for family_name in ("tom_high", "tom_mid", "tom_floor"):
            if family_name in family_to_idx:
                masks["tom_fill"][:, int(family_to_idx[family_name]), :, 0] = True
    if "crash" in masks and "crash" in family_to_idx:
        masks["crash"][:, int(family_to_idx["crash"]), :, 0] = True
    if "tom_crash" in masks:
        for family_name in ("tom_high", "tom_mid", "tom_floor", "crash"):
            if family_name in family_to_idx:
                masks["tom_crash"][:, int(family_to_idx[family_name]), :, 0] = True
    return masks


def sketch_expander_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    class_id_vocab_sizes: Sequence[int] = FAMILY_STATE_ID_VOCAB_SIZES,
    presence_pos_weight: Sequence[float] | None = None,
    velocity_weight: float = 2.0,
    offset_weight: float = 1.0,
    class_weight: float = 0.5,
    budget_weight: float = 1.0,
    placement_weight: float = 0.35,
    over_budget_weight: float = 0.20,
    phrase_weight: float = 0.35,
) -> dict[str, torch.Tensor]:
    logits = torch.as_tensor(outputs["presence_logits"], dtype=torch.float32)
    pred_velocity = torch.as_tensor(outputs["velocity"], dtype=torch.float32, device=logits.device)
    pred_offset = torch.as_tensor(outputs["offset"], dtype=torch.float32, device=logits.device)
    class_logits = torch.as_tensor(outputs["class_logits"], dtype=torch.float32, device=logits.device)
    budget_logits_raw = outputs.get("budget_logits")
    budget_logits = (
        torch.as_tensor(budget_logits_raw, dtype=torch.float32, device=logits.device)
        if budget_logits_raw is not None
        else None
    )
    target_presence = torch.as_tensor(batch["target_presence"], dtype=torch.float32, device=logits.device)
    target_velocity = torch.as_tensor(batch["target_velocity"], dtype=torch.float32, device=logits.device)
    target_offset = torch.as_tensor(batch["target_offset"], dtype=torch.float32, device=logits.device)
    target_class_id = torch.as_tensor(batch["target_class_id"], dtype=torch.long, device=logits.device)
    if tuple(logits.shape) != tuple(target_presence.shape):
        raise ValueError(f"presence shape mismatch: {tuple(logits.shape)} / {tuple(target_presence.shape)}")

    family_count = int(target_presence.shape[1])
    target_bool = target_presence.gt(0.5)
    budget_group_names = tuple(str(name) for name in list(batch.get("ornament_budget_group_names") or ORNAMENT_BUDGET_GROUP_NAMES))
    budget_max_counts_map = _max_counts_for_budget_names(
        budget_group_names,
        batch.get("ornament_budget_max_counts"),
    )
    budget_max_counts = tuple(int(budget_max_counts_map[str(name)]) for name in budget_group_names)
    sketch_hits_for_masks = batch.get("sketch_hits")
    if sketch_hits_for_masks is not None:
        sketch_hits_for_masks = torch.as_tensor(sketch_hits_for_masks, dtype=torch.float32, device=logits.device)
    group_target_masks = _batched_ornament_target_masks(
        target_bool,
        target_velocity,
        target_class_id,
        sketch_hits_for_masks,
        budget_group_names=budget_group_names,
    )
    group_candidate_masks = _batched_ornament_candidate_masks(
        target_bool.shape,
        sketch_hits=sketch_hits_for_masks,
        device=logits.device,
        budget_group_names=budget_group_names,
    )
    snare_slot_mask = torch.zeros_like(target_bool)
    zero_mask = torch.zeros_like(target_bool, dtype=torch.bool)
    kick_ghost_mask = group_target_masks.get("kick_ghost", zero_mask)
    snare_ghost_mask = group_target_masks.get("snare_ghost", zero_mask)
    hihat_open_mask = group_target_masks.get("open_hat", zero_mask)
    if int(family_count) >= int(len(FAMILY_STATE_FAMILY_NAMES)):
        family_to_idx = {name: idx for idx, name in enumerate(FAMILY_STATE_FAMILY_NAMES)}
        snare_idx = int(family_to_idx["snare"])
        hihat_idx = int(family_to_idx["hihat"])
        snare_slot_mask[:, snare_idx, :, 1:] = True
    if presence_pos_weight is None:
        weights = [1.7, 1.8, 2.8, 2.8, 2.8, 1.5, 2.4, 2.0]
        if int(len(weights)) != int(family_count):
            weights = [1.8] * int(family_count)
    else:
        weights = [float(x) for x in list(presence_pos_weight)]
    pos_weight = torch.as_tensor(weights, dtype=torch.float32, device=logits.device).view(1, family_count, 1, 1)
    bce = F.binary_cross_entropy_with_logits(logits, target_presence, reduction="none")
    presence_weight = torch.where(target_bool, pos_weight.expand_as(target_presence), torch.ones_like(target_presence))
    controls_for_weight = batch.get("controls")
    control_names_for_weight = tuple(str(name) for name in list(batch.get("control_names") or SKETCH_CONTROL_NAMES))
    if controls_for_weight is not None and "fill_density" in control_names_for_weight and int(family_count) >= int(len(FAMILY_STATE_FAMILY_NAMES)):
        controls_tensor = torch.as_tensor(controls_for_weight, dtype=torch.float32, device=logits.device)
        fill_idx = int(control_names_for_weight.index("fill_density"))
        high_fill = controls_tensor[:, int(fill_idx)].ge(0.70).view(-1, 1, 1, 1)
        family_to_idx = {name: idx for idx, name in enumerate(FAMILY_STATE_FAMILY_NAMES)}
        tom_family_mask = torch.zeros((1, family_count, 1, 1), dtype=torch.bool, device=logits.device)
        for family_name in ("tom_high", "tom_mid", "tom_floor"):
            if str(family_name) in family_to_idx and int(family_to_idx[str(family_name)]) < int(family_count):
                tom_family_mask[:, int(family_to_idx[str(family_name)]), :, :] = True
        presence_weight = presence_weight + (
            target_bool.to(dtype=torch.float32)
            * high_fill.to(dtype=torch.float32)
            * tom_family_mask.to(dtype=torch.float32)
            * 1.25
        )
    presence_loss = (bce * presence_weight).mean()

    active = target_bool
    active_count = active.to(dtype=torch.float32).sum().clamp_min(1.0)
    velocity_weight_map = active.to(dtype=torch.float32)
    velocity_weight_map = velocity_weight_map + ((active & kick_ghost_mask).to(dtype=torch.float32) * 3.0)
    velocity_weight_map = velocity_weight_map + ((active & snare_ghost_mask).to(dtype=torch.float32) * 3.0)
    velocity_weight_map = velocity_weight_map + ((active & group_target_masks.get("snare_roll_drag", zero_mask)).to(dtype=torch.float32) * 2.0)
    velocity_weight_map = velocity_weight_map + ((active & group_target_masks.get("snare_roll_run", zero_mask)).to(dtype=torch.float32) * 1.5)
    velocity_weight_map = velocity_weight_map + ((active & hihat_open_mask).to(dtype=torch.float32) * 2.0)
    velocity_loss = (F.smooth_l1_loss(pred_velocity, target_velocity, reduction="none") * velocity_weight_map).sum()
    velocity_loss = velocity_loss / velocity_weight_map.sum().clamp_min(1.0)
    offset_loss = (
        F.smooth_l1_loss(pred_offset, target_offset, reduction="none") * active.to(dtype=torch.float32)
    ).sum() / active_count

    vocab_sizes = [int(x) for x in list(class_id_vocab_sizes)]
    ce_sum = logits.new_zeros(())
    ce_count = logits.new_zeros(())
    for family_idx, vocab_size in enumerate(vocab_sizes[:family_count]):
        if int(vocab_size) <= 1:
            continue
        mask = active[:, int(family_idx), :, :]
        if not bool(mask.any().item()):
            continue
        logits_f = class_logits[:, int(family_idx), :, :, : int(vocab_size)][mask]
        target_f = target_class_id[:, int(family_idx), :, :][mask].clamp(min=0, max=int(vocab_size) - 1)
        ce_items = F.cross_entropy(logits_f, target_f, reduction="none")
        ce_weights = torch.ones_like(ce_items)
        if int(family_count) >= int(len(FAMILY_STATE_FAMILY_NAMES)):
            family_name = str(FAMILY_STATE_FAMILY_NAMES[int(family_idx)])
            if family_name == "hihat":
                ce_weights = torch.where((target_f.eq(0) | target_f.eq(1)), ce_weights * 3.0, ce_weights)
        ce_sum = ce_sum + (ce_items * ce_weights).sum()
        ce_count = ce_count + ce_weights.sum()
    class_loss = ce_sum / ce_count.clamp_min(1.0)

    target_budget_raw = batch.get("target_ornament_budget")
    if target_budget_raw is not None:
        target_budget = torch.as_tensor(target_budget_raw, dtype=torch.long, device=logits.device)
    else:
        target_budget_values = []
        for group_name, max_count in zip(budget_group_names, budget_max_counts, strict=True):
            counts = group_target_masks[str(group_name)].to(dtype=torch.long).sum(dim=(1, 2, 3)).clamp(max=int(max_count))
            target_budget_values.append(counts)
        target_budget = torch.stack(target_budget_values, dim=1).contiguous()
    budget_loss = logits.new_zeros(())
    placement_loss = logits.new_zeros(())
    over_budget_loss = logits.new_zeros(())
    budget_pred = torch.zeros_like(target_budget)
    budget_score = logits.new_zeros(())
    budget_exact_acc = logits.new_zeros(())
    budget_mae = logits.new_zeros(())
    if budget_logits is not None and int(budget_logits.dim()) == 3 and int(budget_logits.shape[1]) > 0:
        budget_loss_sum = logits.new_zeros(())
        budget_loss_count = logits.new_zeros(())
        expected_counts: list[torch.Tensor] = []
        group_count = min(int(budget_logits.shape[1]), int(len(budget_group_names)), int(target_budget.shape[1]))
        for group_idx in range(group_count):
            group_name = str(budget_group_names[int(group_idx)])
            group_weight = {
                "kick_ghost": 1.8,
                "snare_ghost": 1.2,
                "snare_roll_drag": 1.2,
                "snare_roll_run": 1.2,
                "tom_fill": 2.2,
                "tom_crash": 2.0,
                "crash": 1.4,
            }.get(group_name, 1.0)
            max_count = int(budget_max_counts[int(group_idx)])
            logits_g = budget_logits[:, int(group_idx), : int(max_count) + 1]
            target_g = target_budget[:, int(group_idx)].clamp(min=0, max=int(max_count))
            budget_loss_sum = budget_loss_sum + (float(group_weight) * F.cross_entropy(logits_g, target_g, reduction="sum"))
            budget_loss_count = budget_loss_count + torch.as_tensor(
                float(group_weight) * float(target_g.numel()),
                dtype=torch.float32,
                device=logits.device,
            )
            probs_g = F.softmax(logits_g, dim=-1)
            count_values = torch.arange(int(max_count) + 1, dtype=torch.float32, device=logits.device)
            expected_counts.append((probs_g * count_values.view(1, -1)).sum(dim=-1))
            budget_pred[:, int(group_idx)] = torch.argmax(logits_g, dim=-1)
        budget_loss = budget_loss_sum / budget_loss_count.clamp_min(1.0)

        placement_sum = logits.new_zeros(())
        placement_count = logits.new_zeros(())
        for group_idx, group_name in enumerate(budget_group_names[:group_count]):
            candidate_mask = group_candidate_masks[str(group_name)]
            target_mask = group_target_masks[str(group_name)] & candidate_mask
            group_weight = {
                "kick_ghost": 1.5,
                "snare_roll_run": 1.2,
                "tom_fill": 1.8,
                "tom_crash": 1.6,
                "crash": 1.3,
            }.get(str(group_name), 1.0)
            for batch_idx in range(int(logits.shape[0])):
                cand_i = candidate_mask[int(batch_idx)]
                target_i = target_mask[int(batch_idx)]
                if not bool(cand_i.any().item()) or not bool(target_i.any().item()):
                    continue
                logits_i = logits[int(batch_idx)][cand_i]
                target_i_f = target_i[cand_i].to(dtype=torch.float32)
                if float(target_i_f.sum().item()) <= 0.0:
                    continue
                log_probs = F.log_softmax(logits_i.view(1, -1), dim=-1).view(-1)
                target_dist = target_i_f / target_i_f.sum().clamp_min(1.0)
                placement_sum = placement_sum + (float(group_weight) * (-(target_dist * log_probs).sum()))
                placement_count = placement_count + logits.new_tensor(float(group_weight))
        placement_loss = placement_sum / placement_count.clamp_min(1.0)

        if expected_counts:
            expected = torch.stack(expected_counts, dim=1)
            caps_rows: list[list[float]] = []
            controls_for_caps = batch.get("controls")
            controls_tensor = (
                torch.as_tensor(controls_for_caps, dtype=torch.float32, device=logits.device)
                if controls_for_caps is not None
                else torch.zeros((int(logits.shape[0]), len(SKETCH_CONTROL_NAMES)), dtype=torch.float32, device=logits.device)
            )
            control_names = batch.get("control_names") or SKETCH_CONTROL_NAMES
            for batch_idx in range(int(logits.shape[0])):
                decoded = decode_sketch_controls(controls_tensor[int(batch_idx)].detach().cpu(), control_names=control_names)
                caps = _budget_caps_from_decoded_controls(
                    decoded,
                    variation=0.0,
                    budget_group_names=budget_group_names[:group_count],
                    budget_max_counts=budget_max_counts[:group_count],
                )
                caps_rows.append([float(caps[str(name)]) for name in budget_group_names[:group_count]])
            caps_t = torch.tensor(caps_rows, dtype=torch.float32, device=logits.device)
            over_budget_loss = F.smooth_l1_loss(torch.minimum(expected, expected.new_tensor(99.0)), caps_t, reduction="none")
            over_budget_loss = (over_budget_loss * expected.gt(caps_t).to(dtype=torch.float32)).mean()
        with torch.no_grad():
            max_counts = torch.as_tensor(
                budget_max_counts[: int(target_budget.shape[1])],
                dtype=torch.float32,
                device=logits.device,
            ).view(1, -1).clamp_min(1.0)
            budget_abs = (budget_pred[:, : int(max_counts.shape[1])].to(dtype=torch.float32) - target_budget[:, : int(max_counts.shape[1])].to(dtype=torch.float32)).abs()
            budget_mae = budget_abs.mean()
            budget_score = (1.0 - (budget_abs / max_counts).mean()).clamp(0.0, 1.0)
            budget_exact_acc = budget_pred[:, : int(max_counts.shape[1])].eq(target_budget[:, : int(max_counts.shape[1])]).to(dtype=torch.float32).mean()

    phrase_loss = logits.new_zeros(())
    fill_start_acc = logits.new_zeros(())
    fill_length_acc = logits.new_zeros(())
    tom_direction_acc = logits.new_zeros(())
    fill_accent_shape_acc = logits.new_zeros(())

    def _phrase_ce(output_key: str, target_key: str) -> tuple[torch.Tensor, torch.Tensor]:
        raw_logits = outputs.get(output_key)
        raw_target = batch.get(target_key)
        if raw_logits is None or raw_target is None:
            return logits.new_zeros(()), logits.new_zeros(())
        phrase_logits = torch.as_tensor(raw_logits, dtype=torch.float32, device=logits.device)
        phrase_target = torch.as_tensor(raw_target, dtype=torch.long, device=logits.device).view(-1)
        if int(phrase_logits.dim()) != 2 or int(phrase_logits.shape[0]) != int(phrase_target.shape[0]):
            return logits.new_zeros(()), logits.new_zeros(())
        phrase_target = phrase_target.clamp(min=0, max=int(phrase_logits.shape[1]) - 1)
        loss_value = F.cross_entropy(phrase_logits, phrase_target)
        acc_value = phrase_logits.argmax(dim=-1).eq(phrase_target).to(dtype=torch.float32).mean()
        return loss_value, acc_value

    fill_start_loss, fill_start_acc = _phrase_ce("fill_start_logits", "target_fill_start")
    fill_length_loss, fill_length_acc = _phrase_ce("fill_length_logits", "target_fill_length")
    tom_direction_loss, tom_direction_acc = _phrase_ce("tom_direction_logits", "target_tom_direction")
    fill_accent_shape_loss, fill_accent_shape_acc = _phrase_ce("fill_accent_shape_logits", "target_fill_accent_shape")
    phrase_loss = 0.35 * (fill_start_loss + fill_length_loss) + 0.15 * tom_direction_loss + 0.15 * fill_accent_shape_loss

    total = (
        presence_loss
        + (float(velocity_weight) * velocity_loss)
        + (float(offset_weight) * offset_loss)
        + (float(class_weight) * class_loss)
        + (float(budget_weight) * budget_loss)
        + (float(placement_weight) * placement_loss)
        + (float(over_budget_weight) * over_budget_loss)
        + (float(phrase_weight) * phrase_loss)
    )
    with torch.no_grad():
        pred_presence = logits.sigmoid().ge(0.5)
        tp = (pred_presence & target_bool).to(dtype=torch.float32).sum()
        fp = (pred_presence & ~target_bool).to(dtype=torch.float32).sum()
        fn = (~pred_presence & target_bool).to(dtype=torch.float32).sum()
        precision = tp / (tp + fp).clamp_min(1.0)
        recall = tp / (tp + fn).clamp_min(1.0)
        f1 = (2.0 * precision * recall) / (precision + recall).clamp_min(1.0e-8)
        velocity_mae = ((pred_velocity - target_velocity).abs() * active.to(dtype=torch.float32)).sum() / active_count
        offset_mae = ((pred_offset - target_offset).abs() * active.to(dtype=torch.float32)).sum() / active_count

        def _target_recall(mask: torch.Tensor) -> torch.Tensor:
            mask_bool = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
            target_mask = target_bool & mask_bool
            hit = (pred_presence & target_mask).to(dtype=torch.float32).sum()
            total = target_mask.to(dtype=torch.float32).sum()
            return hit / total.clamp_min(1.0)

        def _masked_mae(mask: torch.Tensor) -> torch.Tensor:
            mask_bool = target_bool & torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
            denom = mask_bool.to(dtype=torch.float32).sum().clamp_min(1.0)
            return ((pred_velocity - target_velocity).abs() * mask_bool.to(dtype=torch.float32)).sum() / denom

        def _masked_bias(mask: torch.Tensor) -> torch.Tensor:
            mask_bool = target_bool & torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
            denom = mask_bool.to(dtype=torch.float32).sum().clamp_min(1.0)
            return ((pred_velocity - target_velocity) * mask_bool.to(dtype=torch.float32)).sum() / denom

        snare_slot_gt0_recall = logits.new_zeros(())
        kick_ghost_recall = logits.new_zeros(())
        snare_ghost_recall = logits.new_zeros(())
        snare_roll_drag_recall = logits.new_zeros(())
        snare_roll_run_recall = logits.new_zeros(())
        snare_anchor_class_acc = logits.new_zeros(())
        hihat_anchor_class_acc = logits.new_zeros(())
        hihat_open_class_acc = logits.new_zeros(())
        tom_fill_recall = logits.new_zeros(())
        crash_recall = logits.new_zeros(())
        tom_crash_recall = logits.new_zeros(())
        kick_ghost_velocity_mae = logits.new_zeros(())
        kick_ghost_velocity_bias = logits.new_zeros(())
        snare_ghost_velocity_mae = logits.new_zeros(())
        snare_ghost_velocity_bias = logits.new_zeros(())
        snare_slot_gt0_velocity_mae = logits.new_zeros(())
        hihat_open_velocity_mae = logits.new_zeros(())
        hihat_open_velocity_bias = logits.new_zeros(())
        tom_velocity_mae = logits.new_zeros(())
        crash_velocity_mae = logits.new_zeros(())
        if int(family_count) >= int(len(FAMILY_STATE_FAMILY_NAMES)):
            family_to_idx = {name: idx for idx, name in enumerate(FAMILY_STATE_FAMILY_NAMES)}
            snare_idx = int(family_to_idx["snare"])
            hihat_idx = int(family_to_idx["hihat"])
            snare_slot_gt0_recall = _target_recall(snare_slot_mask)
            kick_ghost_recall = _target_recall(kick_ghost_mask)
            kick_ghost_velocity_mae = _masked_mae(kick_ghost_mask)
            kick_ghost_velocity_bias = _masked_bias(kick_ghost_mask)
            snare_ghost_recall = _target_recall(snare_ghost_mask)
            snare_ghost_velocity_mae = _masked_mae(snare_ghost_mask)
            snare_ghost_velocity_bias = _masked_bias(snare_ghost_mask)
            snare_roll_drag_mask = group_target_masks.get("snare_roll_drag", zero_mask)
            snare_roll_run_mask = group_target_masks.get("snare_roll_run", zero_mask)
            snare_roll_drag_recall = _target_recall(snare_roll_drag_mask)
            snare_roll_run_recall = _target_recall(snare_roll_run_mask)
            snare_slot_gt0_velocity_mae = _masked_mae(snare_slot_mask)
            hihat_open_velocity_mae = _masked_mae(hihat_open_mask)
            hihat_open_velocity_bias = _masked_bias(hihat_open_mask)

            sketch_hits_raw = batch.get("sketch_hits")
            if sketch_hits_raw is not None:
                sketch_hits = torch.as_tensor(sketch_hits_raw, dtype=torch.float32, device=logits.device).gt(0.5)
                if int(sketch_hits.dim()) == 3:
                    sketch_names = tuple(str(x) for x in list(batch.get("sketch_family_names") or SKETCH_FAMILY_NAMES))
                    for family_name, metric_name in (("snare", "snare_anchor"), ("hihat", "hihat_anchor")):
                        if str(family_name) not in sketch_names:
                            continue
                        family_idx = int(family_to_idx[str(family_name)])
                        vocab_size = int(class_id_vocab_sizes[family_idx]) if family_idx < len(class_id_vocab_sizes) else 1
                        if int(vocab_size) <= 1:
                            continue
                        sketch_idx = int(sketch_names.index(str(family_name)))
                        step_count = int(min(int(target_bool.shape[2]), int(sketch_hits.shape[-1])))
                        anchor_mask = (
                            target_bool[:, family_idx, :step_count, 0]
                            & sketch_hits[:, sketch_idx, :step_count]
                        )
                        if not bool(anchor_mask.any().item()):
                            continue
                        pred_class = class_logits[:, family_idx, :step_count, 0, : int(vocab_size)].argmax(dim=-1)
                        acc = (
                            pred_class[anchor_mask]
                            == target_class_id[:, family_idx, :step_count, 0][anchor_mask]
                        ).to(dtype=torch.float32).mean()
                        if metric_name == "snare_anchor":
                            snare_anchor_class_acc = acc
                        else:
                            hihat_anchor_class_acc = acc

            hihat_open_hit = pred_presence & hihat_open_mask
            if bool(hihat_open_hit.any().item()):
                pred_hihat_class = class_logits[:, hihat_idx, :, :, : int(class_id_vocab_sizes[hihat_idx])].argmax(dim=-1)
                hihat_open_class_acc = (
                    pred_hihat_class[hihat_open_hit[:, hihat_idx]]
                    == target_class_id[:, hihat_idx][hihat_open_hit[:, hihat_idx]]
                ).to(dtype=torch.float32).mean()

            tom_fill_mask = torch.zeros_like(target_bool)
            for family_name in ("tom_high", "tom_mid", "tom_floor"):
                tom_fill_mask[:, int(family_to_idx[family_name])] = True
            crash_mask = torch.zeros_like(target_bool)
            crash_mask[:, int(family_to_idx["crash"])] = True
            tom_crash_mask = tom_fill_mask | crash_mask
            tom_fill_recall = _target_recall(tom_fill_mask)
            crash_recall = _target_recall(crash_mask)
            tom_crash_recall = _target_recall(tom_crash_mask)
            tom_velocity_mae = _masked_mae(tom_fill_mask)
            crash_velocity_mae = _masked_mae(crash_mask)
    return {
        "loss": total,
        "presence_loss": presence_loss.detach(),
        "velocity_loss": velocity_loss.detach(),
        "offset_loss": offset_loss.detach(),
        "class_loss": class_loss.detach(),
        "budget_loss": budget_loss.detach(),
        "placement_loss": placement_loss.detach(),
        "over_budget_loss": over_budget_loss.detach(),
        "phrase_loss": phrase_loss.detach(),
        "fill_start_acc": fill_start_acc.detach(),
        "fill_length_acc": fill_length_acc.detach(),
        "tom_direction_acc": tom_direction_acc.detach(),
        "fill_accent_shape_acc": fill_accent_shape_acc.detach(),
        "budget_mae": budget_mae.detach(),
        "budget_score": budget_score.detach(),
        "budget_exact_acc": budget_exact_acc.detach(),
        "event_f1": f1.detach(),
        "event_precision": precision.detach(),
        "event_recall": recall.detach(),
        "velocity_mae": velocity_mae.detach(),
        "offset_mae": offset_mae.detach(),
        "snare_slot_gt0_recall": snare_slot_gt0_recall.detach(),
        "kick_ghost_recall": kick_ghost_recall.detach(),
        "snare_ghost_recall": snare_ghost_recall.detach(),
        "snare_roll_drag_recall": snare_roll_drag_recall.detach(),
        "snare_roll_run_recall": snare_roll_run_recall.detach(),
        "snare_anchor_class_acc": snare_anchor_class_acc.detach(),
        "hihat_anchor_class_acc": hihat_anchor_class_acc.detach(),
        "hihat_open_class_acc": hihat_open_class_acc.detach(),
        "tom_fill_recall": tom_fill_recall.detach(),
        "crash_recall": crash_recall.detach(),
        "tom_crash_recall": tom_crash_recall.detach(),
        "kick_ghost_velocity_mae": kick_ghost_velocity_mae.detach(),
        "kick_ghost_velocity_bias": kick_ghost_velocity_bias.detach(),
        "snare_ghost_velocity_mae": snare_ghost_velocity_mae.detach(),
        "snare_ghost_velocity_bias": snare_ghost_velocity_bias.detach(),
        "snare_slot_gt0_velocity_mae": snare_slot_gt0_velocity_mae.detach(),
        "hihat_open_velocity_mae": hihat_open_velocity_mae.detach(),
        "hihat_open_velocity_bias": hihat_open_velocity_bias.detach(),
        "tom_velocity_mae": tom_velocity_mae.detach(),
        "crash_velocity_mae": crash_velocity_mae.detach(),
    }


def _threshold_for_family(
    family: str,
    *,
    slot: int,
    step: int | None = None,
    controls: torch.Tensor,
    control_names: Sequence[str] | None = None,
) -> float:
    ctrl = torch.as_tensor(controls, dtype=torch.float32).view(-1)
    names = tuple(str(name) for name in list(control_names or ()))
    legacy = bool(is_legacy_sketch_control_names(names) or (not names and int(ctrl.numel()) == 5))
    if legacy:
        ghost_density = float(ctrl[2].item()) if int(ctrl.numel()) > 2 else 0.0
        hihat_density = float(ctrl[3].item()) if int(ctrl.numel()) > 3 else 0.5
        fill_density = float(ctrl[4].item()) if int(ctrl.numel()) > 4 else 0.0
        if str(family) == "hihat":
            return float(max(0.15, min(0.85, 0.55 - ((float(hihat_density) - 0.5) * 0.6))))
        if str(family) in {"tom_high", "tom_mid", "tom_floor", "crash", "ride"}:
            return float(max(0.15, min(0.9, 0.58 - ((float(fill_density) - 0.2) * 0.7))))
        if str(family) == "snare" and int(slot) > 0:
            return float(max(0.2, min(0.9, 0.60 - ((float(ghost_density) - 0.25) * 0.5))))
        return 0.5

    decoded = decode_sketch_controls(ctrl, control_names=names or None)
    if str(decoded.get("version", "")) in {"v3", "v4"}:
        ghost_density = float(decoded.get("ghost_density", 0.0))
        kick_ghost_density = float(decoded.get("kick_ghost_density", max(0.0, (ghost_density - 0.25) / 0.75)))
        snare_ghost_density = float(decoded.get("snare_ghost_density", ghost_density))
        snare_roll_density = float(decoded.get("snare_roll_density", snare_ghost_density))
        hihat_density = float(decoded.get("hihat_density", 0.5))
        fill_density = float(decoded.get("fill_density", 0.0))
        step_idx = int(step) if step is not None else -1
        if str(family) == "kick" and (int(step_idx) < 0 or int(step_idx) not in {0, 8} or int(slot) > 0):
            return float(max(0.14, min(0.92, 0.72 - ((kick_ghost_density - 0.25) * 0.90))))
        if str(family) == "hihat":
            return float(max(0.18, min(0.92, 0.62 - ((hihat_density - 0.35) * 0.65))))
        if str(family) == "ride":
            return float(max(0.22, min(0.92, 0.78 - ((fill_density - 0.45) * 0.70))))
        if str(family) in {"tom_high", "tom_mid", "tom_floor"}:
            return float(max(0.18, min(0.92, 0.70 - ((fill_density - 0.20) * 0.85))))
        if str(family) == "crash":
            return float(max(0.25, min(0.94, 0.82 - ((fill_density - 0.35) * 0.70))))
        if str(family) == "snare":
            if int(slot) > 0:
                return float(max(0.16, min(0.92, 0.74 - ((snare_roll_density - 0.30) * 0.95))))
            if int(step_idx) >= 0 and int(step_idx) not in set(SNARE_BACKBEAT_STEPS):
                return float(max(0.16, min(0.92, 0.66 - ((snare_ghost_density - 0.20) * 0.85))))
        return 0.5
    snare_style = str(decoded.get("snare_style", "plain"))
    hat_rate = str(decoded.get("hat_rate", "eighths"))
    fill_role = str(decoded.get("fill_role", "none"))
    step_idx = int(step) if step is not None else -1
    if str(family) == "hihat":
        return {
            "none": 0.92,
            "sparse_syncopated": 0.58,
            "eighths": 0.50,
            "sixteenths": 0.32,
        }.get(hat_rate, 0.50)
    if str(family) == "ride":
        return 0.40 if fill_role in {"ride", "ride_plus_fill"} else 0.86
    if str(family) in {"tom_high", "tom_mid", "tom_floor", "crash"}:
        return 0.36 if fill_role in {"tom_crash_fill", "ride_plus_fill"} else 0.88
    if str(family) == "snare":
        if int(slot) > 0:
            return {
                "plain": 0.88,
                "ghosts": 0.48,
                "drags_rolls": 0.24,
                "ghosts_plus_rolls": 0.22,
            }.get(snare_style, 0.88)
        if int(step_idx) >= 0 and int(step_idx) not in set(SNARE_BACKBEAT_STEPS):
            return {
                "plain": 0.72,
                "ghosts": 0.30,
                "drags_rolls": 0.50,
                "ghosts_plus_rolls": 0.28,
            }.get(snare_style, 0.72)
    return 0.5


def apply_feel_to_events(
    events: Sequence[Mapping[str, Any]],
    controls: torch.Tensor,
    *,
    control_names: Sequence[str] | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    ctrl = torch.as_tensor(controls, dtype=torch.float32).view(-1)
    decoded = decode_sketch_controls(ctrl, control_names=control_names)
    if str(decoded.get("version", "")) == "v4":
        feel_style = str(decoded.get("feel_style", "straight"))
        feel_amount = float(max(0.0, min(1.0, float(decoded.get("feel_amount", 0.0)))))
    else:
        swing = float(decoded.get("swing", 0.0))
        humanize = float(decoded.get("humanize", 0.0))
        if swing >= 0.18:
            feel_style = "swing"
        elif swing <= -0.18:
            feel_style = "pushed"
        elif humanize >= 0.55:
            feel_style = "laid_back"
        else:
            feel_style = "straight"
        feel_amount = float(max(0.0, min(1.0, max(abs(swing), humanize))))
    rng = random.Random(int(seed))
    out: list[dict[str, Any]] = []
    for event in list(events):
        item = dict(event)
        step = int(item.get("step", 0))
        offset = float(item.get("offset", 0.0))
        family = str(item.get("family", ""))
        forced = bool(item.get("forced", False))
        ornament = (not forced) or int(item.get("slot", 0)) > 0 or family in {"tom_high", "tom_mid", "tom_floor", "crash", "ride"}
        if feel_style == "swing":
            if int(step) % 2 == 1:
                offset += 0.20 * feel_amount
            elif int(step) % 4 in {2, 3}:
                offset += 0.05 * feel_amount
        elif feel_style == "pushed":
            if ornament or int(step) % 4 in {1, 3}:
                offset -= 0.14 * feel_amount
            else:
                offset -= 0.04 * feel_amount
        elif feel_style == "laid_back":
            if family == "snare" and int(step) in set(SNARE_BACKBEAT_STEPS):
                offset += 0.13 * feel_amount
            elif ornament or int(step) % 4 in {2, 3}:
                offset += 0.10 * feel_amount
            else:
                offset += 0.035 * feel_amount
        jitter_std = (0.12 if ornament else 0.055) * feel_amount
        if jitter_std > 0.0:
            offset += rng.gauss(0.0, float(jitter_std))
        velocity = float(item.get("velocity", 0.0) or 0.0)
        if feel_amount > 0.0:
            vel_std = (0.12 if ornament else 0.045) * feel_amount
            velocity = max(0.01, min(1.0, velocity + rng.gauss(0.0, float(vel_std))))
            if feel_style == "pushed" and ornament:
                velocity = max(0.01, min(1.0, velocity + (0.035 * feel_amount)))
            elif feel_style == "laid_back" and ornament:
                velocity = max(0.01, min(1.0, velocity - (0.020 * feel_amount)))
        item["offset"] = float(max(-0.49, min(0.49, offset)))
        item["velocity"] = float(max(0.01, min(1.0, velocity)))
        out.append(item)
    return out


def _value_from_intensity(control: float, low: float, mid: float, high: float) -> float:
    amount = float(max(0.0, min(1.0, float(control))))
    if amount <= 0.5:
        return float(low) + ((float(mid) - float(low)) * (amount / 0.5))
    return float(mid) + ((float(high) - float(mid)) * ((amount - 0.5) / 0.5))


def _blend_clamped(value: float, target: float, *, blend: float, lo: float, hi: float) -> float:
    blended = (float(value) * (1.0 - float(blend))) + (float(target) * float(blend))
    return float(max(float(lo), min(float(hi), blended)))


def _tom_contour_velocity(
    *,
    event: Mapping[str, Any],
    decoded_controls: Mapping[str, Any],
) -> float:
    start = int(round(float(decoded_controls.get("fill_start", 0.0)) * float(DEFAULT_NUM_STEPS - 1)))
    length = int(max(1, round(float(decoded_controls.get("fill_length", 0.0)) * float(DEFAULT_NUM_STEPS))))
    step = int(event.get("step", 0))
    pos = max(0, min(int(length) - 1, int(step) - int(start)))
    denom = float(max(1, int(length) - 1))
    x = float(pos) / denom
    shape = str(decoded_controls.get("fill_accent_shape", "flat"))
    base = 0.54 + (0.20 * float(decoded_controls.get("fill_density", 0.0)))
    if shape == "ramp_up":
        return float(base - 0.12 + (0.24 * x))
    if shape == "ramp_down":
        return float(base + 0.12 - (0.24 * x))
    if shape == "peak_end":
        return float(base - 0.08 + (0.32 * (x ** 1.5)))
    return float(base)


def _select_class_id_from_scores(
    scores: torch.Tensor,
    *,
    family: str,
    decoded_controls: Mapping[str, Any],
) -> int:
    adjusted = torch.as_tensor(scores, dtype=torch.float32).clone()
    if str(family) == "hihat" and str(decoded_controls.get("version", "")) in {"v3", "v4"}:
        openness = float(decoded_controls.get("hihat_openness", 0.0))
        if openness >= 0.55:
            for offset_idx, class_id in enumerate(HIHAT_OPEN_CLASS_IDS):
                if int(class_id) < int(adjusted.numel()):
                    adjusted[int(class_id)] += 0.70 + (0.30 * openness) - (0.15 * int(offset_idx))
        elif openness >= 0.25:
            for class_id in HIHAT_PEDAL_CLASS_IDS:
                if int(class_id) < int(adjusted.numel()):
                    adjusted[int(class_id)] += 0.55
        else:
            for offset_idx, class_id in enumerate(HIHAT_CLOSED_CLASS_IDS):
                if int(class_id) < int(adjusted.numel()):
                    adjusted[int(class_id)] += 0.35 - (0.10 * int(offset_idx))
    if str(family) == "hihat" and "hat_color" in decoded_controls and not bool(decoded_controls.get("legacy", False)):
        hat_color = str(decoded_controls.get("hat_color", "closed"))
        if hat_color == "open":
            for offset_idx, class_id in enumerate(HIHAT_OPEN_CLASS_IDS):
                if int(class_id) < int(adjusted.numel()):
                    adjusted[int(class_id)] += 0.65 - (0.15 * int(offset_idx))
        elif hat_color == "pedal":
            for class_id in HIHAT_PEDAL_CLASS_IDS:
                if int(class_id) < int(adjusted.numel()):
                    adjusted[int(class_id)] += 0.60
        elif hat_color == "closed":
            for offset_idx, class_id in enumerate(HIHAT_CLOSED_CLASS_IDS):
                if int(class_id) < int(adjusted.numel()):
                    adjusted[int(class_id)] += 0.35 - (0.10 * int(offset_idx))
    return int(torch.argmax(adjusted).item())


def calibrate_decoded_event_velocities(
    events: Sequence[Mapping[str, Any]],
    controls: torch.Tensor,
    *,
    control_names: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    ctrl = torch.as_tensor(controls, dtype=torch.float32).view(-1)
    decoded = decode_sketch_controls(ctrl, control_names=control_names)
    if bool(decoded.get("legacy", False)):
        snare_intensity = float(decoded.get("ghost_density", 0.5))
        kick_intensity = snare_intensity
        roll_intensity = snare_intensity
        open_hat_intensity = 0.5
        snare_style = "ghosts_plus_rolls" if snare_intensity >= 0.70 else ("ghosts" if snare_intensity >= 0.25 else "plain")
        hat_color = "closed"
    elif str(decoded.get("version", "")) == "v3":
        snare_intensity = float(decoded.get("ghost_density", 0.0))
        kick_intensity = float(decoded.get("kick_ghost_density", max(0.0, (snare_intensity - 0.25) / 0.75)))
        roll_intensity = float(decoded.get("snare_roll_density", snare_intensity))
        open_hat_intensity = float(decoded.get("hihat_openness", 0.0))
        snare_style = "ghosts_plus_rolls" if snare_intensity >= 0.70 else ("ghosts" if snare_intensity >= 0.25 else "plain")
        hat_color = "open" if open_hat_intensity >= 0.55 else ("pedal" if open_hat_intensity >= 0.25 else "closed")
    elif str(decoded.get("version", "")) == "v4":
        snare_intensity = float(decoded.get("snare_ghost_density", 0.0))
        kick_intensity = float(decoded.get("kick_ghost_density", 0.0))
        roll_intensity = float(decoded.get("snare_roll_density", 0.0))
        open_hat_intensity = float(decoded.get("hihat_openness", 0.0))
        if roll_intensity >= 0.45 and snare_intensity >= 0.25:
            snare_style = "ghosts_plus_rolls"
        elif roll_intensity >= 0.45:
            snare_style = "drags_rolls"
        elif snare_intensity >= 0.25:
            snare_style = "ghosts"
        else:
            snare_style = "plain"
        hat_color = "open" if open_hat_intensity >= 0.55 else ("pedal" if open_hat_intensity >= 0.25 else "closed")
    else:
        snare_intensity = float(decoded.get("snare_ornament_intensity", 0.5))
        kick_intensity = snare_intensity
        roll_intensity = snare_intensity
        open_hat_intensity = float(decoded.get("open_hat_intensity", 0.5))
        snare_style = str(decoded.get("snare_style", "plain"))
        hat_color = str(decoded.get("hat_color", "closed"))
    kick_ghost_target = _value_from_intensity(kick_intensity, 0.050, 0.08, 0.13)
    ghost_target = _value_from_intensity(snare_intensity, 0.08, 0.16, 0.24)
    roll_target = _value_from_intensity(roll_intensity, 0.25, 0.45, 0.75)
    open_hat_target = _value_from_intensity(open_hat_intensity, 0.45, 0.70, 1.0)
    out: list[dict[str, Any]] = []
    for event in list(events):
        item = dict(event)
        family = str(item.get("family", ""))
        step = int(item.get("step", 0))
        slot = int(item.get("slot", 0))
        class_id = int(item.get("class_id", 0))
        velocity = float(item.get("velocity", 0.0) or 0.0)
        if family == "kick":
            if not bool(item.get("forced", False)) and (velocity <= 0.45 or snare_intensity >= 0.35):
                item["velocity"] = _blend_clamped(velocity, kick_ghost_target, blend=0.80, lo=0.03, hi=0.22)
        elif family == "snare":
            is_backbeat = int(step) in set(SNARE_BACKBEAT_STEPS)
            is_roll = int(slot) > 0
            is_ghost = (
                not is_roll
                and not is_backbeat
                and int(class_id) == 0
                and (velocity <= 0.45 or snare_style in {"ghosts", "ghosts_plus_rolls"})
            )
            if not is_roll and not is_ghost and snare_style in {"drags_rolls", "ghosts_plus_rolls"} and not is_backbeat:
                is_roll = True
            if is_roll:
                item["velocity"] = _blend_clamped(velocity, roll_target, blend=0.65, lo=0.12, hi=0.90)
            elif is_ghost:
                item["velocity"] = _blend_clamped(velocity, ghost_target, blend=0.70, lo=0.04, hi=0.28)
        elif family == "hihat":
            if int(class_id) in set(HIHAT_OPEN_CLASS_IDS):
                item["velocity"] = _blend_clamped(velocity, open_hat_target, blend=0.65, lo=0.30, hi=1.0)
            elif hat_color == "open" and velocity >= 0.58:
                item["class_id"] = int(HIHAT_OPEN_CLASS_IDS[0])
                item["velocity"] = _blend_clamped(velocity, open_hat_target, blend=0.50, lo=0.30, hi=1.0)
        elif family in {"tom_high", "tom_mid", "tom_floor"} and str(decoded.get("version", "")) == "v4":
            target = _tom_contour_velocity(event=item, decoded_controls=decoded)
            item["velocity"] = _blend_clamped(velocity, target, blend=0.60, lo=0.18, hi=1.0)
        out.append(item)
    return out


def _sample_index_from_weights(weights: Sequence[float], rng: random.Random) -> int:
    total = float(sum(max(0.0, float(weight)) for weight in weights))
    if total <= 0.0:
        return int(rng.randrange(len(weights)))
    cursor = rng.random() * total
    running = 0.0
    for idx, weight in enumerate(weights):
        running += max(0.0, float(weight))
        if cursor <= running:
            return int(idx)
    return int(len(weights) - 1)


def _select_budgeted_events(
    candidates: Sequence[Mapping[str, Any]],
    count: int,
    *,
    rng: random.Random,
    pattern_variation: float,
) -> list[dict[str, Any]]:
    limit = int(max(0, min(int(count), int(len(candidates)))))
    if int(limit) <= 0:
        return []
    ordered = sorted((dict(item) for item in candidates), key=lambda item: -float(item.get("selection_score", 0.0)))
    variation = float(max(0.0, min(1.0, float(pattern_variation))))
    if variation <= 0.05:
        return ordered[:limit]
    pool = list(ordered)
    selected: list[dict[str, Any]] = []
    exponent = max(0.35, 1.0 - (0.75 * float(variation)))
    while pool and int(len(selected)) < int(limit):
        weights = [max(1.0e-5, float(item.get("selection_score", 0.0))) ** exponent for item in pool]
        chosen_idx = _sample_index_from_weights(weights, rng)
        selected.append(pool.pop(int(chosen_idx)))
    return selected


def _min_probability_for_budget_group(group_name: str, decoded_controls: Mapping[str, Any]) -> float:
    if str(decoded_controls.get("version", "")) not in {"v3", "v4"}:
        return 0.0
    ghost_density = float(decoded_controls.get("ghost_density", 0.0))
    hihat_density = float(decoded_controls.get("hihat_density", 0.5))
    fill_density = float(decoded_controls.get("fill_density", 0.0))
    base = {
        "kick_ghost": 0.14,
        "snare_ghost": 0.24,
        "snare_roll_drag": 0.28,
        "snare_roll_run": 0.26,
        "off16_hat": 0.22,
        "tom_fill": 0.32,
        "crash": 0.36,
        "ride": 0.34,
        "tom_crash": 0.34,
    }.get(str(group_name), 0.0)
    if str(group_name) in {"kick_ghost", "snare_ghost", "snare_roll_drag", "snare_roll_run"} and ghost_density >= 0.85:
        base -= 0.04
    elif str(group_name) == "off16_hat" and hihat_density >= 0.85:
        base -= 0.04
    elif str(group_name) in {"tom_fill", "crash", "ride", "tom_crash"} and fill_density >= 0.85:
        base -= 0.05
    floor = 0.10 if str(group_name) == "kick_ghost" else 0.12
    return float(max(float(floor), min(0.45, base)))


def _budget_zero_is_confident(
    budget_logits: torch.Tensor,
    *,
    group_idx: int,
    max_count: int,
    margin: float = 10.0,
) -> bool:
    logits_g = torch.as_tensor(
        budget_logits[int(group_idx), : int(max_count) + 1],
        dtype=torch.float32,
    )
    if int(logits_g.numel()) <= 1:
        return True
    return bool(float(logits_g[0].item()) >= float(logits_g[1:].max().item()) + float(margin))


def _budget_counts_from_logits(
    budget_logits: torch.Tensor,
    *,
    decoded_controls: Mapping[str, Any],
    budget_group_names: Sequence[str] | None = None,
    budget_max_counts: Sequence[int] | None = None,
    rng: random.Random,
    pattern_variation: float,
) -> dict[str, int]:
    variation = float(max(0.0, min(1.0, float(pattern_variation))))
    group_names = tuple(str(name) for name in list(budget_group_names or ORNAMENT_BUDGET_GROUP_NAMES))
    max_counts_map = _max_counts_for_budget_names(group_names, budget_max_counts)
    max_counts = tuple(int(max_counts_map[str(name)]) for name in group_names)
    caps = _budget_caps_from_decoded_controls(
        decoded_controls,
        variation=variation,
        budget_group_names=group_names,
        budget_max_counts=max_counts,
    )
    counts: dict[str, int] = {name: 0 for name in group_names}
    if int(budget_logits.dim()) != 2 or int(budget_logits.shape[0]) <= 0:
        return counts
    group_count = min(int(budget_logits.shape[0]), int(len(group_names)))
    group_index = {name: idx for idx, name in enumerate(group_names[:group_count])}
    for group_idx, group_name in enumerate(group_names[:group_count]):
        max_count = int(max_counts[int(group_idx)])
        logits_g = torch.as_tensor(budget_logits[int(group_idx), : int(max_count) + 1], dtype=torch.float32)
        if variation >= 0.45:
            temperature = 0.75 + (0.75 * variation)
            probs = F.softmax(logits_g / float(temperature), dim=-1).detach().cpu().tolist()
            count = _sample_index_from_weights([float(x) for x in probs], rng)
        else:
            count = int(torch.argmax(logits_g).item())
        counts[str(group_name)] = int(max(0, min(int(count), int(caps.get(str(group_name), 0)))))
    if str(decoded_controls.get("version", "")) in {"v3", "v4"}:
        fill_density = float(decoded_controls.get("fill_density", 0.0))
        if "tom_fill" in counts and fill_density >= 0.70:
            tom_idx = int(group_index.get("tom_fill", -1))
            tom_cap = int(caps.get("tom_fill", 0))
            confident_zero = (
                tom_idx >= 0
                and _budget_zero_is_confident(
                    budget_logits,
                    group_idx=int(tom_idx),
                    max_count=int(max_counts[int(tom_idx)]),
                )
            )
            if tom_idx >= 0 and tom_cap > 0 and not bool(confident_zero):
                forced_toms = 1 if fill_density < 0.85 else (2 if fill_density < 0.95 else 3)
                counts["tom_fill"] = int(max(int(counts["tom_fill"]), min(int(tom_cap), int(forced_toms))))
        ghost_density = float(decoded_controls.get("ghost_density", 0.0))
        kick_ghost_density = float(decoded_controls.get("kick_ghost_density", max(0.0, (ghost_density - 0.25) / 0.75)))
        if "kick_ghost" in counts and kick_ghost_density >= 0.55:
            kick_idx = int(group_index.get("kick_ghost", -1))
            kick_cap = int(caps.get("kick_ghost", 0))
            confident_zero = (
                kick_idx >= 0
                and _budget_zero_is_confident(
                    budget_logits,
                    group_idx=int(kick_idx),
                    max_count=int(max_counts[int(kick_idx)]),
                )
            )
            if kick_idx >= 0 and kick_cap > 0 and not bool(confident_zero):
                forced_kicks = 1 if kick_ghost_density < 0.90 else (2 if kick_ghost_density < 0.98 else 3)
                counts["kick_ghost"] = int(max(int(counts["kick_ghost"]), min(int(kick_cap), int(forced_kicks))))
        snare_roll_density = float(decoded_controls.get("snare_roll_density", ghost_density))
        if "snare_roll_drag" in counts and snare_roll_density >= 0.55:
            roll_idx = int(group_index.get("snare_roll_drag", -1))
            roll_cap = int(caps.get("snare_roll_drag", 0))
            confident_zero = (
                roll_idx >= 0
                and _budget_zero_is_confident(
                    budget_logits,
                    group_idx=int(roll_idx),
                    max_count=int(max_counts[int(roll_idx)]),
                )
            )
            if roll_idx >= 0 and roll_cap > 0 and not bool(confident_zero):
                forced_rolls = 1 if snare_roll_density < 0.85 else 2
                counts["snare_roll_drag"] = int(max(int(counts["snare_roll_drag"]), min(int(roll_cap), int(forced_rolls))))
        if "snare_roll_run" in counts and snare_roll_density >= 0.45:
            run_idx = int(group_index.get("snare_roll_run", -1))
            run_cap = int(caps.get("snare_roll_run", 0))
            confident_zero = (
                run_idx >= 0
                and _budget_zero_is_confident(
                    budget_logits,
                    group_idx=int(run_idx),
                    max_count=int(max_counts[int(run_idx)]),
                )
            )
            if run_idx >= 0 and run_cap > 0 and not bool(confident_zero):
                forced_runs = 1 if snare_roll_density < 0.75 else (2 if snare_roll_density < 0.95 else 3)
                counts["snare_roll_run"] = int(max(int(counts["snare_roll_run"]), min(int(run_cap), int(forced_runs))))
    return counts


def _event_from_cell(
    *,
    batch_idx: int,
    family_idx: int,
    family: str,
    step_idx: int,
    slot_idx: int,
    vocab_size: int,
    probs: torch.Tensor,
    velocity: torch.Tensor,
    offset: torch.Tensor,
    class_logits: torch.Tensor,
    decoded_controls: Mapping[str, Any],
) -> dict[str, Any]:
    class_id = 0
    if int(vocab_size) > 1:
        class_id = _select_class_id_from_scores(
            class_logits[int(batch_idx), int(family_idx), int(step_idx), int(slot_idx), : int(vocab_size)],
            family=str(family),
            decoded_controls=decoded_controls,
        )
    prob = float(probs[int(batch_idx), int(family_idx), int(step_idx), int(slot_idx)].item())
    return {
        "family": str(family),
        "step": int(step_idx),
        "slot": int(slot_idx),
        "probability": float(prob),
        "selection_score": float(prob),
        "velocity": float(velocity[int(batch_idx), int(family_idx), int(step_idx), int(slot_idx)].item()),
        "offset": float(offset[int(batch_idx), int(family_idx), int(step_idx), int(slot_idx)].item()),
        "class_id": int(class_id),
        "forced": False,
    }


def _class_id_from_cell_or_default(
    *,
    batch_idx: int,
    family_idx: int,
    family: str,
    step_idx: int,
    slot_idx: int,
    vocab_size: int,
    class_logits: torch.Tensor,
    decoded_controls: Mapping[str, Any],
) -> int:
    if int(vocab_size) <= 1:
        return int(DEFAULT_CLASS_ID_BY_FAMILY.get(str(family), 0))
    if int(class_logits.dim()) != 5:
        return int(DEFAULT_CLASS_ID_BY_FAMILY.get(str(family), 0))
    if not (
        0 <= int(batch_idx) < int(class_logits.shape[0])
        and 0 <= int(family_idx) < int(class_logits.shape[1])
        and 0 <= int(step_idx) < int(class_logits.shape[2])
        and 0 <= int(slot_idx) < int(class_logits.shape[3])
    ):
        return int(DEFAULT_CLASS_ID_BY_FAMILY.get(str(family), 0))
    scores = class_logits[int(batch_idx), int(family_idx), int(step_idx), int(slot_idx), : int(vocab_size)]
    if int(scores.numel()) <= 0 or float((scores.max() - scores.min()).item()) <= 1.0e-6:
        return int(DEFAULT_CLASS_ID_BY_FAMILY.get(str(family), 0))
    return _select_class_id_from_scores(
        scores,
        family=str(family),
        decoded_controls=decoded_controls,
    )


def _phrase_plan_from_outputs(
    *,
    outputs: Mapping[str, torch.Tensor],
    batch_idx: int,
    decoded_controls: Mapping[str, Any],
    rng: random.Random,
    pattern_variation: float,
) -> dict[str, Any]:
    variation = float(max(0.0, min(1.0, float(pattern_variation))))

    def _choice_from_logits(key: str, default_idx: int) -> int:
        raw = outputs.get(key)
        if raw is None:
            return int(default_idx)
        logits = torch.as_tensor(raw, dtype=torch.float32).detach().cpu()
        if int(logits.dim()) != 2 or int(batch_idx) >= int(logits.shape[0]):
            return int(default_idx)
        row = logits[int(batch_idx)]
        if variation >= 0.45 and int(row.numel()) > 1:
            probs = F.softmax(row / float(0.8 + (0.9 * variation)), dim=-1).tolist()
            return _sample_index_from_weights([float(x) for x in probs], rng)
        return int(torch.argmax(row).item())

    default_start = 0
    fill_start = float(decoded_controls.get("fill_start", 0.0))
    fill_length = float(decoded_controls.get("fill_length", 0.0))
    if fill_length > 0.0:
        default_start = int(round(fill_start * float(DEFAULT_NUM_STEPS - 1))) + 1
    start_class = _choice_from_logits("fill_start_logits", default_start)
    length_class = _choice_from_logits("fill_length_logits", int(round(fill_length * float(DEFAULT_NUM_STEPS))))
    direction_default = TOM_DIRECTION_VALUES.index(str(decoded_controls.get("tom_direction", "none"))) if str(decoded_controls.get("tom_direction", "none")) in TOM_DIRECTION_VALUES else 0
    accent_default = FILL_ACCENT_SHAPE_VALUES.index(str(decoded_controls.get("fill_accent_shape", "flat"))) if str(decoded_controls.get("fill_accent_shape", "flat")) in FILL_ACCENT_SHAPE_VALUES else 0
    direction_idx = _choice_from_logits("tom_direction_logits", direction_default)
    accent_idx = _choice_from_logits("fill_accent_shape_logits", accent_default)
    start_step = int(max(0, min(DEFAULT_NUM_STEPS - 1, int(start_class) - 1))) if int(start_class) > 0 else int(default_start - 1 if default_start > 0 else 12)
    length = int(max(1, min(DEFAULT_NUM_STEPS, int(length_class) if int(length_class) > 0 else max(2, round(fill_length * DEFAULT_NUM_STEPS)))))
    if variation >= 0.65:
        start_step = int(max(0, min(DEFAULT_NUM_STEPS - 1, start_step + rng.choice([-2, -1, 0, 1, 2]))))
        length = int(max(1, min(DEFAULT_NUM_STEPS, length + rng.choice([-1, 0, 1]))))
    return {
        "start_step": int(start_step),
        "length": int(length),
        "tom_direction": TOM_DIRECTION_VALUES[int(max(0, min(len(TOM_DIRECTION_VALUES) - 1, direction_idx)))],
        "fill_accent_shape": FILL_ACCENT_SHAPE_VALUES[int(max(0, min(len(FILL_ACCENT_SHAPE_VALUES) - 1, accent_idx)))],
    }


def _tom_phrase_bonus(*, family: str, step: int, phrase_pos: int, phrase_plan: Mapping[str, Any]) -> float:
    direction = str(phrase_plan.get("tom_direction", "down"))
    if direction == "up":
        ordered = ("tom_floor", "tom_mid", "tom_high", "tom_high")
    elif direction == "mixed":
        ordered = ("tom_mid", "tom_high", "tom_floor", "tom_mid")
    else:
        ordered = ("tom_high", "tom_mid", "tom_floor", "tom_floor")
    desired = ordered[int(min(max(0, int(phrase_pos)), len(ordered) - 1))]
    family_bonus = 0.35 if str(family) == str(desired) else 0.0
    step_idx = int(step)
    start_step = int(phrase_plan.get("start_step", 12))
    position_bonus = 0.0
    distance = abs(int(step_idx) - (int(start_step) + int(phrase_pos)))
    position_bonus += max(0.0, 0.30 - (0.10 * float(distance)))
    if int(step_idx) % 4 in {2, 3}:
        position_bonus += 0.08
    return float(family_bonus + position_bonus)


def _tom_phrase_events(
    candidates: Sequence[Mapping[str, Any]],
    count: int,
    *,
    rng: random.Random,
    phrase_plan: Mapping[str, Any] | None = None,
    pattern_variation: float = 0.0,
) -> list[dict[str, Any]]:
    tom_candidates = [
        dict(candidate)
        for candidate in list(candidates)
        if str(candidate.get("family", "")).startswith("tom_")
    ]
    limit = int(max(0, min(int(count), int(len(tom_candidates)))))
    if int(limit) <= 0:
        return []
    plan = dict(phrase_plan or {})
    phrase_len = int(min(max(1, int(plan.get("length", min(4, max(2, int(limit)))))), max(1, int(limit))))
    by_step: dict[int, list[dict[str, Any]]] = {}
    for candidate in tom_candidates:
        step = int(candidate.get("step", 0))
        by_step.setdefault(int(step), []).append(candidate)
    best_score = float("-inf")
    best_events: list[dict[str, Any]] = []
    max_step = max(0, 16 - int(phrase_len))
    hinted_start = int(max(0, min(int(max_step), int(plan.get("start_step", max_step)))))
    start_steps = [hinted_start]
    radius = 2 + int(round(3.0 * max(0.0, min(1.0, float(pattern_variation)))))
    for delta in range(1, radius + 1):
        for candidate_start in (hinted_start - delta, hinted_start + delta):
            if 0 <= int(candidate_start) <= int(max_step) and int(candidate_start) not in start_steps:
                start_steps.append(int(candidate_start))
    for start_step in start_steps:
        events: list[dict[str, Any]] = []
        score = 0.0
        for phrase_pos in range(int(phrase_len)):
            step = int(start_step) + int(phrase_pos)
            step_candidates = by_step.get(int(step), [])
            if not step_candidates:
                score -= 2.0
                continue
            ranked = sorted(
                step_candidates,
                key=lambda item: -(
                    float(item.get("selection_score", 0.0))
                    + _tom_phrase_bonus(
                        family=str(item.get("family", "")),
                        step=int(step),
                        phrase_pos=int(phrase_pos),
                        phrase_plan=plan,
                    )
                ),
            )
            chosen = dict(ranked[0])
            chosen["selection_score"] = float(chosen.get("selection_score", 0.0)) + 0.35 + (
                0.10 * float(phrase_len - phrase_pos)
            )
            contour = _tom_contour_velocity(event=chosen, decoded_controls={
                "fill_start": float(start_step) / float(max(1, DEFAULT_NUM_STEPS - 1)),
                "fill_length": float(phrase_len) / float(DEFAULT_NUM_STEPS),
                "fill_accent_shape": str(plan.get("fill_accent_shape", "flat")),
                "fill_density": 1.0,
            })
            chosen["velocity"] = _blend_clamped(float(chosen.get("velocity", 0.0)), contour, blend=0.50, lo=0.18, hi=1.0)
            events.append(chosen)
            score += float(chosen["selection_score"])
        if int(len(events)) > int(len(best_events)) or (
            int(len(events)) == int(len(best_events)) and float(score) > float(best_score)
        ):
            best_score = float(score) + (rng.random() * 1.0e-6)
            best_events = events
    if int(len(best_events)) < int(limit):
        seen = {
            (str(event.get("family")), int(event.get("step", -1)), int(event.get("slot", -1)))
            for event in best_events
        }
        for candidate in _select_budgeted_events(
            tom_candidates,
            int(limit),
            rng=rng,
            pattern_variation=0.0,
        ):
            key = (str(candidate.get("family")), int(candidate.get("step", -1)), int(candidate.get("slot", -1)))
            if key in seen:
                continue
            best_events.append(dict(candidate))
            seen.add(key)
            if int(len(best_events)) >= int(limit):
                break
    return best_events[:limit]


@torch.no_grad()
def decode_event_plan(
    outputs: Mapping[str, torch.Tensor],
    *,
    sketch_hits: torch.Tensor,
    sketch_vel: torch.Tensor,
    controls: torch.Tensor,
    class_names: Sequence[str] = FAMILY_STATE_FAMILY_NAMES,
    class_id_vocab_sizes: Sequence[int] = FAMILY_STATE_ID_VOCAB_SIZES,
    control_names: Sequence[str] | None = None,
    budget_group_names: Sequence[str] | None = None,
    budget_max_counts: Sequence[int] | None = None,
    seed: int = 0,
    pattern_variation: float = 0.0,
) -> list[list[dict[str, Any]]]:
    logits = torch.as_tensor(outputs["presence_logits"], dtype=torch.float32).detach().cpu()
    velocity = torch.as_tensor(outputs["velocity"], dtype=torch.float32).detach().cpu()
    offset = torch.as_tensor(outputs["offset"], dtype=torch.float32).detach().cpu()
    class_logits = torch.as_tensor(outputs["class_logits"], dtype=torch.float32).detach().cpu()
    budget_logits_raw = outputs.get("budget_logits")
    budget_logits = (
        torch.as_tensor(budget_logits_raw, dtype=torch.float32).detach().cpu()
        if budget_logits_raw is not None
        else None
    )
    hits = torch.as_tensor(sketch_hits, dtype=torch.float32).detach().cpu()
    vel = torch.as_tensor(sketch_vel, dtype=torch.float32).detach().cpu()
    ctrl = torch.as_tensor(controls, dtype=torch.float32).detach().cpu()
    names = tuple(str(x) for x in list(class_names))
    vocab_sizes = tuple(int(x) for x in list(class_id_vocab_sizes))
    control_names_eff = tuple(str(x) for x in list(control_names or ()))
    budget_group_names_eff = tuple(str(x) for x in list(budget_group_names or ORNAMENT_BUDGET_GROUP_NAMES))
    if budget_logits is not None and int(budget_logits.dim()) == 3:
        if budget_group_names is None and int(budget_logits.shape[1]) == int(len(ORNAMENT_BUDGET_GROUP_NAMES_V2)):
            budget_group_names_eff = ORNAMENT_BUDGET_GROUP_NAMES_V2
    budget_max_counts_eff_map = _max_counts_for_budget_names(budget_group_names_eff, budget_max_counts)
    budget_max_counts_eff = tuple(int(budget_max_counts_eff_map[str(name)]) for name in budget_group_names_eff)
    sketch_index = {name: idx for idx, name in enumerate(SKETCH_FAMILY_NAMES)}
    family_index = {name: idx for idx, name in enumerate(names)}

    batch_events: list[list[dict[str, Any]]] = []
    probs = logits.sigmoid()
    for batch_idx in range(int(logits.shape[0])):
        events: list[dict[str, Any]] = []
        decoded_controls = decode_sketch_controls(
            ctrl[int(batch_idx)],
            control_names=control_names_eff or None,
        )
        rng = random.Random(int(seed) + (104729 * int(batch_idx)))
        phrase_plan = _phrase_plan_from_outputs(
            outputs=outputs if str(decoded_controls.get("version", "")) == "v4" else {},
            batch_idx=int(batch_idx),
            decoded_controls=decoded_controls,
            rng=rng,
            pattern_variation=float(pattern_variation),
        )
        use_budget_decode = bool(
            budget_logits is not None
            and int(budget_logits.dim()) == 3
            and int(budget_logits.shape[1]) > 0
            and not bool(decoded_controls.get("legacy", False))
        )
        open_hat_budget = 0
        if use_budget_decode:
            budget_counts = _budget_counts_from_logits(
                budget_logits[int(batch_idx)],
                decoded_controls=decoded_controls,
                budget_group_names=budget_group_names_eff,
                budget_max_counts=budget_max_counts_eff,
                rng=rng,
                pattern_variation=float(pattern_variation),
            )
            open_hat_budget = int(budget_counts.get("open_hat", 0))
            candidates_by_group: dict[str, list[dict[str, Any]]] = {
                name: [] for name in budget_group_names_eff if str(name) != "open_hat"
            }
            for family_idx, family in enumerate(names):
                vocab_size = int(vocab_sizes[int(family_idx)])
                for step_idx in range(int(logits.shape[2])):
                    for slot_idx in range(int(logits.shape[3])):
                        family_s = str(family)
                        is_anchor = False
                        if family_s in sketch_index and int(slot_idx) == 0:
                            sketch_family_idx = int(sketch_index[family_s])
                            is_anchor = float(hits[int(batch_idx), int(sketch_family_idx), int(step_idx)].item()) > 0.0
                        group_name = ""
                        if family_s == "kick" and "kick_ghost" in budget_counts:
                            if int(slot_idx) > 0 or not bool(is_anchor):
                                group_name = "kick_ghost"
                        elif family_s == "snare":
                            if int(slot_idx) > 0:
                                group_name = "snare_roll_drag"
                            elif int(step_idx) not in set(SNARE_BACKBEAT_STEPS) and not bool(is_anchor):
                                snare_velocity = float(
                                    velocity[int(batch_idx), int(family_idx), int(step_idx), int(slot_idx)].item()
                                )
                                if (
                                    "snare_roll_run" in budget_counts
                                    and int(budget_counts.get("snare_roll_run", 0)) > 0
                                    and snare_velocity > float(SNARE_GHOST_VELOCITY_THRESHOLD)
                                ):
                                    group_name = "snare_roll_run"
                                else:
                                    group_name = "snare_ghost"
                        elif family_s == "hihat" and int(step_idx) % 2 == 1 and int(slot_idx) == 0:
                            group_name = "off16_hat"
                        elif family_s == "ride" and int(slot_idx) == 0:
                            group_name = "ride"
                        elif family_s in {"tom_high", "tom_mid", "tom_floor"} and int(slot_idx) == 0:
                            group_name = "tom_fill" if "tom_fill" in budget_counts else "tom_crash"
                        elif family_s == "crash" and int(slot_idx) == 0:
                            group_name = "crash" if "crash" in budget_counts else "tom_crash"
                        elif family_s in {"tom_high", "tom_mid", "tom_floor", "crash"} and int(slot_idx) == 0:
                            group_name = "tom_crash"
                        if not group_name or int(budget_counts.get(group_name, 0)) <= 0:
                            continue
                        event = _event_from_cell(
                            batch_idx=int(batch_idx),
                            family_idx=int(family_idx),
                            family=family_s,
                            step_idx=int(step_idx),
                            slot_idx=int(slot_idx),
                            vocab_size=int(vocab_size),
                            probs=probs,
                            velocity=velocity,
                            offset=offset,
                            class_logits=class_logits,
                            decoded_controls=decoded_controls,
                        )
                        if float(event["probability"]) < _min_probability_for_budget_group(
                            str(group_name),
                            decoded_controls,
                        ):
                            continue
                        if group_name == "snare_ghost":
                            event["selection_score"] = float(event["probability"]) * (1.05 - (0.50 * float(event["velocity"])))
                        elif group_name == "snare_roll_run":
                            adjacency = 0.12 if int(step_idx) % 4 in {1, 3} else 0.04
                            event["selection_score"] = float(event["probability"]) * (
                                0.95 + (0.25 * float(event["velocity"])) + float(adjacency)
                            )
                        elif group_name == "kick_ghost":
                            syncopation = 0.12 if int(step_idx) % 4 in {1, 3} else 0.0
                            event["selection_score"] = float(event["probability"]) * (
                                1.10 - (0.45 * float(event["velocity"])) + float(syncopation)
                            )
                        elif group_name == "tom_fill":
                            family_rank = {"tom_high": 0, "tom_mid": 1, "tom_floor": 2}.get(family_s, 1)
                            if str(phrase_plan.get("tom_direction")) == "up":
                                desired_rank = max(0, 2 - int(round((float(step_idx) / max(1.0, float(logits.shape[2] - 1))) * 2.0)))
                            else:
                                desired_rank = int(round((float(step_idx) / max(1.0, float(logits.shape[2] - 1))) * 2.0))
                            phrase_bias = 0.12 if int(step_idx) % 4 in {2, 3} else 0.0
                            phrase_distance = abs(int(step_idx) - int(phrase_plan.get("start_step", 12)))
                            late_bias = max(0.0, 0.26 - (0.04 * float(phrase_distance)))
                            event["selection_score"] = float(event["probability"]) * (
                                1.0
                                + (0.08 * (2.0 - abs(float(family_rank - desired_rank))))
                                + float(phrase_bias)
                                + float(late_bias)
                            )
                        candidates_by_group[str(group_name)].append(event)
            selected_by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
            for group_name, candidates in candidates_by_group.items():
                for event in _select_budgeted_events(
                    candidates,
                    int(budget_counts.get(str(group_name), 0)),
                    rng=rng,
                    pattern_variation=float(pattern_variation),
                ):
                    key = (str(event["family"]), int(event["step"]), int(event["slot"]))
                    current = selected_by_key.get(key)
                    if current is None or float(event.get("selection_score", 0.0)) > float(current.get("selection_score", 0.0)):
                        selected_by_key[key] = dict(event)
            tom_budget = int(budget_counts.get("tom_fill", budget_counts.get("tom_crash", 0)))
            tom_candidates = candidates_by_group.get("tom_fill") or candidates_by_group.get("tom_crash") or []
            for event in _tom_phrase_events(
                tom_candidates,
                int(tom_budget),
                rng=rng,
                phrase_plan=phrase_plan,
                pattern_variation=float(pattern_variation),
            ):
                key = (str(event["family"]), int(event["step"]), int(event["slot"]))
                current = selected_by_key.get(key)
                if current is None or float(event.get("selection_score", 0.0)) > float(current.get("selection_score", 0.0)):
                    selected_by_key[key] = dict(event)
            extra_cap = int(15 + round(8.0 * max(0.0, min(1.0, float(pattern_variation)))))
            selected_events = sorted(
                selected_by_key.values(),
                key=lambda item: -float(item.get("selection_score", 0.0)),
            )[:extra_cap]
            events.extend(selected_events)
        else:
            for family_idx, family in enumerate(names):
                vocab_size = int(vocab_sizes[int(family_idx)])
                for step_idx in range(int(logits.shape[2])):
                    for slot_idx in range(int(logits.shape[3])):
                        threshold = _threshold_for_family(
                            str(family),
                            slot=int(slot_idx),
                            step=int(step_idx),
                            controls=ctrl[int(batch_idx)],
                            control_names=control_names_eff,
                        )
                        prob = float(probs[int(batch_idx), int(family_idx), int(step_idx), int(slot_idx)].item())
                        if prob < float(threshold):
                            continue
                        events.append(
                            _event_from_cell(
                                batch_idx=int(batch_idx),
                                family_idx=int(family_idx),
                                family=str(family),
                                step_idx=int(step_idx),
                                slot_idx=int(slot_idx),
                                vocab_size=int(vocab_size),
                                probs=probs,
                                velocity=velocity,
                                offset=offset,
                                class_logits=class_logits,
                                decoded_controls=decoded_controls,
                            )
                        )

        for family in SKETCH_FAMILY_NAMES:
            if family not in sketch_index or family not in family_index:
                continue
            sketch_family_idx = int(sketch_index[family])
            for step_idx in range(int(hits.shape[-1])):
                if float(hits[int(batch_idx), int(sketch_family_idx), int(step_idx)].item()) <= 0.0:
                    continue
                existing = [
                    event
                    for event in events
                    if str(event.get("family")) == str(family)
                    and int(event.get("step", -1)) == int(step_idx)
                    and int(event.get("slot", 0)) == 0
                ]
                velocity_value = float(vel[int(batch_idx), int(sketch_family_idx), int(step_idx)].item())
                if float(velocity_value) <= 0.0:
                    velocity_value = 0.8
                if existing:
                    chosen = sorted(existing, key=lambda item: -float(item.get("probability", 0.0)))[0]
                    chosen["velocity"] = float(max(float(chosen.get("velocity", 0.0)), float(velocity_value)))
                    family_idx = int(family_index[str(family)])
                    vocab_size = int(vocab_sizes[int(family_idx)]) if int(family_idx) < int(len(vocab_sizes)) else 1
                    chosen["class_id"] = int(
                        _class_id_from_cell_or_default(
                            batch_idx=int(batch_idx),
                            family_idx=int(family_idx),
                            family=str(family),
                            step_idx=int(step_idx),
                            slot_idx=0,
                            vocab_size=int(vocab_size),
                            class_logits=class_logits,
                            decoded_controls=decoded_controls,
                        )
                    )
                    chosen["forced"] = True
                else:
                    family_idx = int(family_index[str(family)])
                    vocab_size = int(vocab_sizes[int(family_idx)]) if int(family_idx) < int(len(vocab_sizes)) else 1
                    events.append(
                        {
                            "family": str(family),
                            "step": int(step_idx),
                            "slot": 0,
                            "probability": 1.0,
                            "velocity": float(max(0.0, min(1.0, velocity_value))),
                            "offset": 0.0,
                            "class_id": int(
                                _class_id_from_cell_or_default(
                                    batch_idx=int(batch_idx),
                                    family_idx=int(family_idx),
                                    family=str(family),
                                    step_idx=int(step_idx),
                                    slot_idx=0,
                                    vocab_size=int(vocab_size),
                                    class_logits=class_logits,
                                    decoded_controls=decoded_controls,
                                )
                            ),
                            "forced": True,
                        }
                    )
        if use_budget_decode and int(open_hat_budget) > 0 and "hihat" in family_index:
            hihat_idx = int(family_index["hihat"])
            hihat_vocab = int(vocab_sizes[int(hihat_idx)])
            openness = float(decoded_controls.get("hihat_openness", 0.0))
            min_open_class_prob = 0.25 if openness >= 0.55 else (0.35 if openness >= 0.25 else 0.50)
            open_candidates: list[dict[str, Any]] = []
            for event_idx, event in enumerate(events):
                if str(event.get("family")) != "hihat":
                    continue
                step_idx = int(event.get("step", 0))
                slot_idx = int(event.get("slot", 0))
                if not (0 <= step_idx < int(logits.shape[2]) and 0 <= slot_idx < int(logits.shape[3])):
                    continue
                open_score = float(event.get("probability", 1.0))
                open_class_prob = 1.0
                if int(hihat_vocab) > 1:
                    class_prob = F.softmax(
                        class_logits[int(batch_idx), int(hihat_idx), int(step_idx), int(slot_idx), : int(hihat_vocab)],
                        dim=-1,
                    )
                    open_probs = [
                        float(class_prob[int(class_id)].item())
                        for class_id in HIHAT_OPEN_CLASS_IDS
                        if int(class_id) < int(class_prob.numel())
                    ]
                    if open_probs:
                        open_class_prob = float(max(open_probs))
                        if float(open_class_prob) < float(min_open_class_prob):
                            continue
                        open_score += float(open_class_prob)
                item = dict(event)
                item["_event_idx"] = int(event_idx)
                item["selection_score"] = float(open_score)
                open_candidates.append(item)
            for selected in _select_budgeted_events(
                open_candidates,
                int(open_hat_budget),
                rng=rng,
                pattern_variation=float(pattern_variation),
            ):
                event_idx = int(selected["_event_idx"])
                events[event_idx]["class_id"] = int(HIHAT_OPEN_CLASS_IDS[0])
                events[event_idx]["probability"] = float(max(float(events[event_idx].get("probability", 0.0)), 0.75))

        events = calibrate_decoded_event_velocities(
            events,
            ctrl[int(batch_idx)],
            control_names=control_names_eff or None,
        )
        events = apply_feel_to_events(
            events,
            ctrl[int(batch_idx)],
            control_names=control_names_eff or None,
            seed=int(seed) + int(batch_idx),
        )
        events = sorted(
            events,
            key=lambda item: (
                int(item.get("step", 0)),
                list(names).index(str(item.get("family"))) if str(item.get("family")) in names else 999,
                int(item.get("slot", 0)),
            ),
        )
        batch_events.append(events)
    return batch_events
