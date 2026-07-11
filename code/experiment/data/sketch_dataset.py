from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from data.diffusion_cache_utils import (
    FAMILY_STATE_FAMILY_NAMES,
    FAMILY_STATE_ID_VOCAB_SIZES,
)


SKETCH_FAMILY_NAMES: tuple[str, ...] = ("kick", "snare", "hihat")
SKETCH_CONTROL_NAMES_V1: tuple[str, ...] = (
    "swing",
    "humanize",
    "ghost_density",
    "hihat_density",
    "fill_density",
)
LEGACY_SKETCH_CONTROL_NAMES: tuple[str, ...] = SKETCH_CONTROL_NAMES_V1
SKETCH_CONTROL_NAMES_V3: tuple[str, ...] = (
    "swing",
    "humanize",
    "ghost_density",
    "hihat_density",
    "hihat_openness",
    "fill_density",
)
FEEL_STYLE_VALUES: tuple[str, ...] = ("straight", "pushed", "laid_back", "swing")
SKETCH_FEEL_CONTROL_NAMES: tuple[str, ...] = ("swing", "humanize")
SKETCH_ORNAMENT_CONTROL_NAMES: tuple[str, ...] = (
    "snare_style",
    "hat_rate",
    "hat_color",
    "fill_role",
)
SKETCH_INTENSITY_CONTROL_NAMES: tuple[str, ...] = (
    "snare_ornament_intensity",
    "open_hat_intensity",
)
SNARE_STYLE_VALUES: tuple[str, ...] = ("plain", "ghosts", "drags_rolls", "ghosts_plus_rolls")
HAT_RATE_VALUES: tuple[str, ...] = ("none", "sparse_syncopated", "eighths", "sixteenths")
HAT_COLOR_VALUES: tuple[str, ...] = ("closed", "pedal", "open")
FILL_ROLE_VALUES: tuple[str, ...] = ("none", "ride", "tom_crash_fill", "ride_plus_fill")
TOM_DIRECTION_VALUES: tuple[str, ...] = ("none", "up", "down", "mixed")
FILL_ACCENT_SHAPE_VALUES: tuple[str, ...] = ("flat", "ramp_up", "ramp_down", "peak_end")
SKETCH_PUBLIC_CONTROL_NAMES_V2: tuple[str, ...] = (
    *SKETCH_FEEL_CONTROL_NAMES,
    *SKETCH_ORNAMENT_CONTROL_NAMES,
    *SKETCH_INTENSITY_CONTROL_NAMES,
)
SKETCH_CONTROL_NAMES_V2_17: tuple[str, ...] = (
    *SKETCH_FEEL_CONTROL_NAMES,
    *(f"snare_style_{value}" for value in SNARE_STYLE_VALUES),
    *(f"hat_rate_{value}" for value in HAT_RATE_VALUES),
    *(f"hat_color_{value}" for value in HAT_COLOR_VALUES),
    *(f"fill_role_{value}" for value in FILL_ROLE_VALUES),
)
SKETCH_CONTROL_NAMES_V2: tuple[str, ...] = (
    *SKETCH_CONTROL_NAMES_V2_17,
    *SKETCH_INTENSITY_CONTROL_NAMES,
)
SKETCH_CONTROL_NAMES_V4: tuple[str, ...] = (
    *(f"feel_style_{value}" for value in FEEL_STYLE_VALUES),
    "feel_amount",
    "kick_ghost_density",
    "snare_ghost_density",
    "snare_roll_density",
    "hihat_density",
    "hihat_openness",
    "fill_density",
    *(f"fill_role_{value}" for value in FILL_ROLE_VALUES),
    "fill_start",
    "fill_length",
    *(f"tom_direction_{value}" for value in TOM_DIRECTION_VALUES),
    *(f"fill_accent_shape_{value}" for value in FILL_ACCENT_SHAPE_VALUES),
)
SKETCH_PUBLIC_CONTROL_NAMES: tuple[str, ...] = (
    "feel_style",
    "feel_amount",
    "ghost_density",
    "kick_ghost_density",
    "snare_ghost_density",
    "snare_roll_density",
    "hihat_density",
    "hihat_openness",
    "fill_density",
    "fill_role",
    "fill_start",
    "fill_length",
    "tom_direction",
    "fill_accent_shape",
)
SKETCH_CONTROL_NAMES: tuple[str, ...] = SKETCH_CONTROL_NAMES_V4
FILL_PHRASE_TARGET_NAMES: tuple[str, ...] = (
    "fill_start",
    "fill_length",
    "tom_direction",
    "fill_accent_shape",
)
DEFAULT_SKETCH_MAX_SLOTS = 3
DEFAULT_NUM_STEPS = 16
SNARE_BACKBEAT_STEPS: tuple[int, ...] = (4, 12)
KICK_BACKBEAT_STEPS: tuple[int, ...] = (0, 4, 8, 12)
KICK_GHOST_VELOCITY_THRESHOLD = 0.22
SNARE_GHOST_VELOCITY_THRESHOLD = 0.25
SNARE_STRONG_VELOCITY_THRESHOLD = 0.35
HIHAT_OPEN_CLASS_IDS: tuple[int, ...] = (0, 1)
HIHAT_CLOSED_CLASS_IDS: tuple[int, ...] = (2, 3)
HIHAT_PEDAL_CLASS_IDS: tuple[int, ...] = (4,)
ORNAMENT_BUDGET_GROUP_NAMES_V2: tuple[str, ...] = (
    "snare_ghost",
    "snare_roll_drag",
    "off16_hat",
    "open_hat",
    "ride",
    "tom_crash",
)
ORNAMENT_BUDGET_MAX_COUNTS_V2: tuple[int, ...] = (6, 4, 8, 6, 8, 8)
ORNAMENT_BUDGET_GROUP_NAMES: tuple[str, ...] = (
    "kick_ghost",
    "snare_ghost",
    "snare_roll_drag",
    "snare_roll_run",
    "off16_hat",
    "open_hat",
    "tom_fill",
    "crash",
    "ride",
)
ORNAMENT_BUDGET_MAX_COUNTS: tuple[int, ...] = (4, 6, 4, 8, 8, 6, 8, 4, 8)


@dataclass(frozen=True)
class SketchTrainingExample:
    example_path: Path
    source_id: str
    split: str
    beat_index: int
    bpm: float
    duration_sec: float
    sketch_hits: torch.Tensor
    sketch_vel: torch.Tensor
    controls: torch.Tensor
    target_presence: torch.Tensor
    target_velocity: torch.Tensor
    target_offset: torch.Tensor
    target_class_id: torch.Tensor
    target_count: torch.Tensor
    target_ornament_budget: torch.Tensor
    target_fill_start: torch.Tensor
    target_fill_length: torch.Tensor
    target_tom_direction: torch.Tensor
    target_fill_accent_shape: torch.Tensor


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            text = str(raw_line).strip()
            if text:
                rows.append(dict(json.loads(text)))
    return rows


def _family_index_map(class_names: Sequence[str] | None = None) -> dict[str, int]:
    names = tuple(str(x) for x in (class_names or FAMILY_STATE_FAMILY_NAMES))
    return {name: idx for idx, name in enumerate(names)}


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(float(lo), min(float(hi), float(value))))


def _normalize_piecewise(value: float, low: float, mid: float, high: float) -> float:
    value_f = float(value)
    low_f = float(low)
    mid_f = float(mid)
    high_f = float(high)
    if value_f <= mid_f:
        denom = max(1.0e-6, mid_f - low_f)
        return _clamp(0.5 * ((value_f - low_f) / denom), 0.0, 1.0)
    denom = max(1.0e-6, high_f - mid_f)
    return _clamp(0.5 + (0.5 * ((value_f - mid_f) / denom)), 0.0, 1.0)


def _normalize_control_choice(value: Any, *, choices: Sequence[str], default: str) -> str:
    aliases = {
        "16th": "sixteenths",
        "16ths": "sixteenths",
        "sixteenth": "sixteenths",
        "8th": "eighths",
        "8ths": "eighths",
        "eighth": "eighths",
        "sparse": "sparse_syncopated",
        "syncopated": "sparse_syncopated",
        "drag": "drags_rolls",
        "drags": "drags_rolls",
        "roll": "drags_rolls",
        "rolls": "drags_rolls",
        "drags_and_rolls": "drags_rolls",
        "ghost_rolls": "ghosts_plus_rolls",
        "ghosts_rolls": "ghosts_plus_rolls",
        "ghosts_and_rolls": "ghosts_plus_rolls",
        "fill": "tom_crash_fill",
        "tom_fill": "tom_crash_fill",
        "crash_fill": "tom_crash_fill",
        "ride_fill": "ride_plus_fill",
        "both": "ride_plus_fill",
        "laidback": "laid_back",
        "laid-back": "laid_back",
        "late": "laid_back",
        "ahead": "pushed",
        "push": "pushed",
        "upward": "up",
        "ascending": "up",
        "downward": "down",
        "descending": "down",
        "peak": "peak_end",
        "end_peak": "peak_end",
        "crescendo": "ramp_up",
        "decrescendo": "ramp_down",
    }
    text = str(value if value is not None else default).strip().lower().replace("-", "_").replace(" ", "_")
    text = aliases.get(text, text)
    allowed = tuple(str(choice) for choice in choices)
    return text if text in allowed else str(default)


def _one_hot(choice: str, choices: Sequence[str]) -> list[float]:
    selected = str(choice)
    return [1.0 if str(value) == selected else 0.0 for value in choices]


def _names_tuple(control_names: Sequence[str] | None) -> tuple[str, ...]:
    return tuple(str(name) for name in list(control_names or ()))


def is_v1_sketch_control_names(control_names: Sequence[str] | None) -> bool:
    names = _names_tuple(control_names)
    return bool(names == SKETCH_CONTROL_NAMES_V1)


def is_v2_sketch_control_names(control_names: Sequence[str] | None) -> bool:
    names = _names_tuple(control_names)
    if names == SKETCH_CONTROL_NAMES_V4 or "feel_amount" in names:
        return False
    return bool(
        names in {SKETCH_CONTROL_NAMES_V2, SKETCH_CONTROL_NAMES_V2_17}
        or any(str(name).startswith(("snare_style_", "hat_rate_", "hat_color_", "fill_role_")) for name in names)
    )


def is_v3_sketch_control_names(control_names: Sequence[str] | None) -> bool:
    names = _names_tuple(control_names)
    return bool(names == SKETCH_CONTROL_NAMES_V3 or ("hihat_openness" in names and not is_v4_sketch_control_names(names)))


def is_v4_sketch_control_names(control_names: Sequence[str] | None) -> bool:
    names = _names_tuple(control_names)
    return bool(
        names == SKETCH_CONTROL_NAMES_V4
        or "feel_amount" in names
        or any(str(name).startswith(("feel_style_", "tom_direction_", "fill_accent_shape_")) for name in names)
    )


def is_legacy_sketch_control_names(control_names: Sequence[str] | None) -> bool:
    return is_v1_sketch_control_names(control_names)


def _ghost_density_from_v2(snare_style: Any, snare_ornament_intensity: float | None) -> float:
    style = _normalize_control_choice(snare_style, choices=SNARE_STYLE_VALUES, default="plain")
    intensity = _clamp(0.5 if snare_ornament_intensity is None else float(snare_ornament_intensity), 0.0, 1.0)
    if style == "ghosts":
        return _clamp(0.35 + (0.35 * intensity), 0.0, 1.0)
    if style == "drags_rolls":
        return _clamp(0.65 + (0.20 * intensity), 0.0, 1.0)
    if style == "ghosts_plus_rolls":
        return _clamp(0.72 + (0.28 * intensity), 0.0, 1.0)
    return 0.0


def split_ghost_densities(ghost_density: float) -> dict[str, float]:
    ghost = _clamp(float(ghost_density), 0.0, 1.0)
    return {
        "snare_ghost_density": ghost,
        "kick_ghost_density": _clamp((ghost - 0.25) / 0.75, 0.0, 1.0),
        "snare_roll_density": _clamp((ghost - 0.55) / 0.45, 0.0, 1.0),
    }


def _feel_from_swing_humanize(swing: float, humanize: float) -> tuple[str, float]:
    swing_f = _clamp(float(swing), -1.0, 1.0)
    humanize_f = _clamp(float(humanize), 0.0, 1.0)
    if swing_f >= 0.18:
        style = "swing"
    elif swing_f <= -0.18:
        style = "pushed"
    elif humanize_f >= 0.55:
        style = "laid_back"
    else:
        style = "straight"
    return style, _clamp(max(abs(swing_f), humanize_f), 0.0, 1.0)


def _control_choice_from_one_hot(
    raw: Mapping[str, Any],
    *,
    prefix: str,
    choices: Sequence[str],
    default: str,
) -> str:
    for choice in choices:
        key = f"{prefix}_{choice}"
        if float(raw.get(key, 0.0) or 0.0) >= 0.5:
            return str(choice)
    return str(default)


def _hihat_density_from_v2(hat_rate: Any) -> float:
    rate = _normalize_control_choice(hat_rate, choices=HAT_RATE_VALUES, default="eighths")
    return {
        "none": 0.0,
        "sparse_syncopated": 0.28,
        "eighths": 0.52,
        "sixteenths": 0.88,
    }.get(rate, 0.52)


def _hihat_openness_from_v2(hat_color: Any, open_hat_intensity: float | None) -> float:
    color = _normalize_control_choice(hat_color, choices=HAT_COLOR_VALUES, default="closed")
    intensity = _clamp(0.5 if open_hat_intensity is None else float(open_hat_intensity), 0.0, 1.0)
    if color == "open":
        return _clamp(max(0.55, intensity), 0.0, 1.0)
    if color == "pedal":
        return _clamp(0.28 + (0.20 * intensity), 0.0, 1.0)
    return 0.0


def _fill_density_from_v2(fill_role: Any) -> float:
    role = _normalize_control_choice(fill_role, choices=FILL_ROLE_VALUES, default="none")
    return {
        "none": 0.0,
        "ride": 0.45,
        "tom_crash_fill": 0.65,
        "ride_plus_fill": 0.88,
    }.get(role, 0.0)


def encode_sketch_controls(
    *,
    swing: float = 0.0,
    humanize: float = 0.0,
    ghost_density: float | None = None,
    hihat_density: float | None = None,
    hihat_openness: float | None = None,
    fill_density: float | None = None,
    snare_style: str = "plain",
    hat_rate: str = "eighths",
    hat_color: str = "closed",
    fill_role: str = "none",
    snare_ornament_intensity: float | None = None,
    open_hat_intensity: float | None = None,
) -> torch.Tensor:
    values = [
        _clamp(float(swing), -1.0, 1.0),
        _clamp(float(humanize), 0.0, 1.0),
        _clamp(
            _ghost_density_from_v2(snare_style, snare_ornament_intensity)
            if ghost_density is None
            else float(ghost_density),
            0.0,
            1.0,
        ),
        _clamp(_hihat_density_from_v2(hat_rate) if hihat_density is None else float(hihat_density), 0.0, 1.0),
        _clamp(
            _hihat_openness_from_v2(hat_color, open_hat_intensity)
            if hihat_openness is None
            else float(hihat_openness),
            0.0,
            1.0,
        ),
        _clamp(_fill_density_from_v2(fill_role) if fill_density is None else float(fill_density), 0.0, 1.0),
    ]
    return torch.tensor(values, dtype=torch.float32).contiguous()


def encode_v2_sketch_controls(
    *,
    swing: float = 0.0,
    humanize: float = 0.0,
    snare_style: str = "plain",
    hat_rate: str = "eighths",
    hat_color: str = "closed",
    fill_role: str = "none",
    snare_ornament_intensity: float = 0.5,
    open_hat_intensity: float = 0.5,
) -> torch.Tensor:
    snare_style_eff = _normalize_control_choice(snare_style, choices=SNARE_STYLE_VALUES, default="plain")
    hat_rate_eff = _normalize_control_choice(hat_rate, choices=HAT_RATE_VALUES, default="eighths")
    hat_color_eff = _normalize_control_choice(hat_color, choices=HAT_COLOR_VALUES, default="closed")
    fill_role_eff = _normalize_control_choice(fill_role, choices=FILL_ROLE_VALUES, default="none")
    values = [
        _clamp(float(swing), -1.0, 1.0),
        _clamp(float(humanize), 0.0, 1.0),
        *_one_hot(snare_style_eff, SNARE_STYLE_VALUES),
        *_one_hot(hat_rate_eff, HAT_RATE_VALUES),
        *_one_hot(hat_color_eff, HAT_COLOR_VALUES),
        *_one_hot(fill_role_eff, FILL_ROLE_VALUES),
        _clamp(float(snare_ornament_intensity), 0.0, 1.0),
        _clamp(float(open_hat_intensity), 0.0, 1.0),
    ]
    return torch.tensor(values, dtype=torch.float32).contiguous()


def encode_v4_sketch_controls(
    *,
    feel_style: str = "straight",
    feel_amount: float = 0.0,
    kick_ghost_density: float | None = None,
    snare_ghost_density: float | None = None,
    snare_roll_density: float | None = None,
    ghost_density: float | None = None,
    hihat_density: float = 0.52,
    hihat_openness: float = 0.0,
    fill_density: float = 0.0,
    fill_role: str = "none",
    fill_start: float = 0.0,
    fill_length: float = 0.0,
    tom_direction: str = "none",
    fill_accent_shape: str = "flat",
) -> torch.Tensor:
    ghost_defaults = split_ghost_densities(0.0 if ghost_density is None else float(ghost_density))
    feel_style_eff = _normalize_control_choice(feel_style, choices=FEEL_STYLE_VALUES, default="straight")
    fill_role_eff = _normalize_control_choice(fill_role, choices=FILL_ROLE_VALUES, default="none")
    tom_direction_eff = _normalize_control_choice(tom_direction, choices=TOM_DIRECTION_VALUES, default="none")
    fill_accent_eff = _normalize_control_choice(fill_accent_shape, choices=FILL_ACCENT_SHAPE_VALUES, default="flat")
    values = [
        *_one_hot(feel_style_eff, FEEL_STYLE_VALUES),
        _clamp(float(feel_amount), 0.0, 1.0),
        _clamp(
            float(ghost_defaults["kick_ghost_density"] if kick_ghost_density is None else kick_ghost_density),
            0.0,
            1.0,
        ),
        _clamp(
            float(ghost_defaults["snare_ghost_density"] if snare_ghost_density is None else snare_ghost_density),
            0.0,
            1.0,
        ),
        _clamp(
            float(ghost_defaults["snare_roll_density"] if snare_roll_density is None else snare_roll_density),
            0.0,
            1.0,
        ),
        _clamp(float(hihat_density), 0.0, 1.0),
        _clamp(float(hihat_openness), 0.0, 1.0),
        _clamp(float(fill_density), 0.0, 1.0),
        *_one_hot(fill_role_eff, FILL_ROLE_VALUES),
        _clamp(float(fill_start), 0.0, 1.0),
        _clamp(float(fill_length), 0.0, 1.0),
        *_one_hot(tom_direction_eff, TOM_DIRECTION_VALUES),
        *_one_hot(fill_accent_eff, FILL_ACCENT_SHAPE_VALUES),
    ]
    return torch.tensor(values, dtype=torch.float32).contiguous()


def _choice_from_encoded(
    values: torch.Tensor,
    *,
    start: int,
    choices: Sequence[str],
    default: str,
) -> str:
    width = int(len(tuple(choices)))
    if int(values.numel()) < int(start) + int(width):
        return str(default)
    group = values[int(start) : int(start) + int(width)]
    if int(group.numel()) <= 0:
        return str(default)
    return str(tuple(choices)[int(torch.argmax(group).item())])


def decode_sketch_controls(
    controls: torch.Tensor | Sequence[float],
    *,
    control_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    ctrl = torch.as_tensor(controls, dtype=torch.float32).view(-1)
    names = tuple(str(name) for name in list(control_names or ()))
    if is_v4_sketch_control_names(names) or (not names and int(ctrl.numel()) == int(len(SKETCH_CONTROL_NAMES_V4))):
        feel_style = _choice_from_encoded(ctrl, start=0, choices=FEEL_STYLE_VALUES, default="straight")
        fill_role = _choice_from_encoded(ctrl, start=11, choices=FILL_ROLE_VALUES, default="none")
        tom_direction = _choice_from_encoded(ctrl, start=17, choices=TOM_DIRECTION_VALUES, default="none")
        fill_accent_shape = _choice_from_encoded(ctrl, start=21, choices=FILL_ACCENT_SHAPE_VALUES, default="flat")
        feel_amount = _clamp(float(ctrl[4].item()) if int(ctrl.numel()) > 4 else 0.0, 0.0, 1.0)
        kick_ghost_density = _clamp(float(ctrl[5].item()) if int(ctrl.numel()) > 5 else 0.0, 0.0, 1.0)
        snare_ghost_density = _clamp(float(ctrl[6].item()) if int(ctrl.numel()) > 6 else 0.0, 0.0, 1.0)
        snare_roll_density = _clamp(float(ctrl[7].item()) if int(ctrl.numel()) > 7 else 0.0, 0.0, 1.0)
        ghost_density = _clamp(
            max(
                float(snare_ghost_density),
                0.25 + (0.75 * float(kick_ghost_density)) if kick_ghost_density > 0.0 else 0.0,
                0.55 + (0.45 * float(snare_roll_density)) if snare_roll_density > 0.0 else 0.0,
            ),
            0.0,
            1.0,
        )
        swing = {
            "pushed": -0.35,
            "laid_back": 0.10,
            "swing": 0.35,
        }.get(str(feel_style), 0.0) * float(feel_amount)
        humanize = float(feel_amount)
        return {
            "feel_style": str(feel_style),
            "feel_amount": float(feel_amount),
            "swing": _clamp(float(swing), -1.0, 1.0),
            "humanize": _clamp(float(humanize), 0.0, 1.0),
            "ghost_density": float(ghost_density),
            "kick_ghost_density": float(kick_ghost_density),
            "snare_ghost_density": float(snare_ghost_density),
            "snare_roll_density": float(snare_roll_density),
            "hihat_density": _clamp(float(ctrl[8].item()) if int(ctrl.numel()) > 8 else 0.52, 0.0, 1.0),
            "hihat_openness": _clamp(float(ctrl[9].item()) if int(ctrl.numel()) > 9 else 0.0, 0.0, 1.0),
            "fill_density": _clamp(float(ctrl[10].item()) if int(ctrl.numel()) > 10 else 0.0, 0.0, 1.0),
            "fill_role": str(fill_role),
            "fill_start": _clamp(float(ctrl[15].item()) if int(ctrl.numel()) > 15 else 0.0, 0.0, 1.0),
            "fill_length": _clamp(float(ctrl[16].item()) if int(ctrl.numel()) > 16 else 0.0, 0.0, 1.0),
            "tom_direction": str(tom_direction),
            "fill_accent_shape": str(fill_accent_shape),
            "version": "v4",
            "legacy": False,
        }
    if is_v1_sketch_control_names(names) or (not names and int(ctrl.numel()) == int(len(SKETCH_CONTROL_NAMES_V1))):
        values = [float(ctrl[idx].item()) if int(ctrl.numel()) > idx else 0.0 for idx in range(5)]
        ghost_density = _clamp(values[2], 0.0, 1.0)
        return {
            "swing": _clamp(values[0], -1.0, 1.0),
            "humanize": _clamp(values[1], 0.0, 1.0),
            "ghost_density": ghost_density,
            **split_ghost_densities(ghost_density),
            "hihat_density": _clamp(values[3], 0.0, 1.0),
            "hihat_openness": 0.0,
            "fill_density": _clamp(values[4], 0.0, 1.0),
            "version": "v1",
            "legacy": True,
        }
    if is_v3_sketch_control_names(names) or (not names and int(ctrl.numel()) == int(len(SKETCH_CONTROL_NAMES_V3))):
        ghost_density = _clamp(float(ctrl[2].item()) if int(ctrl.numel()) > 2 else 0.0, 0.0, 1.0)
        return {
            "swing": _clamp(float(ctrl[0].item()) if int(ctrl.numel()) > 0 else 0.0, -1.0, 1.0),
            "humanize": _clamp(float(ctrl[1].item()) if int(ctrl.numel()) > 1 else 0.0, 0.0, 1.0),
            "ghost_density": ghost_density,
            **split_ghost_densities(ghost_density),
            "hihat_density": _clamp(float(ctrl[3].item()) if int(ctrl.numel()) > 3 else 0.5, 0.0, 1.0),
            "hihat_openness": _clamp(float(ctrl[4].item()) if int(ctrl.numel()) > 4 else 0.0, 0.0, 1.0),
            "fill_density": _clamp(float(ctrl[5].item()) if int(ctrl.numel()) > 5 else 0.0, 0.0, 1.0),
            "version": "v3",
            "legacy": False,
        }
    v2_width = int(len(SKETCH_CONTROL_NAMES_V2))
    v2_17_width = int(len(SKETCH_CONTROL_NAMES_V2_17))
    if is_v2_sketch_control_names(names) or (not names and int(ctrl.numel()) in {v2_width, v2_17_width}):
        snare_style = _choice_from_encoded(ctrl, start=2, choices=SNARE_STYLE_VALUES, default="plain")
        hat_rate = _choice_from_encoded(ctrl, start=6, choices=HAT_RATE_VALUES, default="eighths")
        hat_color = _choice_from_encoded(ctrl, start=10, choices=HAT_COLOR_VALUES, default="closed")
        fill_role = _choice_from_encoded(ctrl, start=13, choices=FILL_ROLE_VALUES, default="none")
        snare_ornament_intensity = _clamp(float(ctrl[17].item()) if int(ctrl.numel()) > 17 else 0.5, 0.0, 1.0)
        open_hat_intensity = _clamp(float(ctrl[18].item()) if int(ctrl.numel()) > 18 else 0.5, 0.0, 1.0)
        ghost_density = _ghost_density_from_v2(snare_style, snare_ornament_intensity)
        return {
            "swing": _clamp(float(ctrl[0].item()) if int(ctrl.numel()) > 0 else 0.0, -1.0, 1.0),
            "humanize": _clamp(float(ctrl[1].item()) if int(ctrl.numel()) > 1 else 0.0, 0.0, 1.0),
            "snare_style": snare_style,
            "hat_rate": hat_rate,
            "hat_color": hat_color,
            "fill_role": fill_role,
            "snare_ornament_intensity": snare_ornament_intensity,
            "open_hat_intensity": open_hat_intensity,
            "ghost_density": ghost_density,
            **split_ghost_densities(ghost_density),
            "hihat_density": _hihat_density_from_v2(hat_rate),
            "hihat_openness": _hihat_openness_from_v2(hat_color, open_hat_intensity),
            "fill_density": _fill_density_from_v2(fill_role),
            "version": "v2",
            "legacy": False,
        }
    ghost_density = _clamp(float(ctrl[2].item()) if int(ctrl.numel()) > 2 else 0.0, 0.0, 1.0)
    return {
        "swing": _clamp(float(ctrl[0].item()) if int(ctrl.numel()) > 0 else 0.0, -1.0, 1.0),
        "humanize": _clamp(float(ctrl[1].item()) if int(ctrl.numel()) > 1 else 0.0, 0.0, 1.0),
        "ghost_density": ghost_density,
        **split_ghost_densities(ghost_density),
        "hihat_density": _clamp(float(ctrl[3].item()) if int(ctrl.numel()) > 3 else 0.5, 0.0, 1.0),
        "hihat_openness": _clamp(float(ctrl[4].item()) if int(ctrl.numel()) > 4 else 0.0, 0.0, 1.0),
        "fill_density": _clamp(float(ctrl[5].item()) if int(ctrl.numel()) > 5 else 0.0, 0.0, 1.0),
        "version": "v3",
        "legacy": False,
    }


def sketch_controls_to_public_dict(
    controls: torch.Tensor | Sequence[float],
    *,
    control_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    decoded = decode_sketch_controls(controls, control_names=control_names)
    if str(decoded.get("version", "")) == "v4":
        return {
            "feel_style": str(decoded["feel_style"]),
            "feel_amount": float(decoded["feel_amount"]),
            "ghost_density": float(decoded["ghost_density"]),
            "kick_ghost_density": float(decoded["kick_ghost_density"]),
            "snare_ghost_density": float(decoded["snare_ghost_density"]),
            "snare_roll_density": float(decoded["snare_roll_density"]),
            "hihat_density": float(decoded["hihat_density"]),
            "hihat_openness": float(decoded["hihat_openness"]),
            "fill_density": float(decoded["fill_density"]),
            "fill_role": str(decoded["fill_role"]),
            "fill_start": float(decoded["fill_start"]),
            "fill_length": float(decoded["fill_length"]),
            "tom_direction": str(decoded["tom_direction"]),
            "fill_accent_shape": str(decoded["fill_accent_shape"]),
        }
    if bool(decoded.get("legacy", False)):
        return {
            "swing": float(decoded["swing"]),
            "humanize": float(decoded["humanize"]),
            "ghost_density": float(decoded["ghost_density"]),
            "hihat_density": float(decoded["hihat_density"]),
            "fill_density": float(decoded["fill_density"]),
        }
    if str(decoded.get("version", "")) == "v2":
        return {
            "swing": float(decoded["swing"]),
            "humanize": float(decoded["humanize"]),
            "snare_style": str(decoded["snare_style"]),
            "hat_rate": str(decoded["hat_rate"]),
            "hat_color": str(decoded["hat_color"]),
            "fill_role": str(decoded["fill_role"]),
            "snare_ornament_intensity": float(decoded["snare_ornament_intensity"]),
            "open_hat_intensity": float(decoded["open_hat_intensity"]),
        }
    return {
        "swing": float(decoded["swing"]),
        "humanize": float(decoded["humanize"]),
        "ghost_density": float(decoded["ghost_density"]),
        "hihat_density": float(decoded["hihat_density"]),
        "hihat_openness": float(decoded["hihat_openness"]),
        "fill_density": float(decoded["fill_density"]),
    }


def _legacy_controls_from_mapping(raw: Mapping[str, Any]) -> torch.Tensor:
    snare_style = _normalize_control_choice(raw.get("snare_style", "plain"), choices=SNARE_STYLE_VALUES, default="plain")
    hat_rate = _normalize_control_choice(raw.get("hat_rate", "eighths"), choices=HAT_RATE_VALUES, default="eighths")
    fill_role = _normalize_control_choice(raw.get("fill_role", "none"), choices=FILL_ROLE_VALUES, default="none")
    ghost_default = {
        "plain": 0.0,
        "ghosts": 0.65,
        "drags_rolls": 0.25,
        "ghosts_plus_rolls": 0.75,
    }[snare_style]
    hat_default = {
        "none": 0.0,
        "sparse_syncopated": 0.25,
        "eighths": 0.5,
        "sixteenths": 0.85,
    }[hat_rate]
    fill_default = {
        "none": 0.0,
        "ride": 0.35,
        "tom_crash_fill": 0.55,
        "ride_plus_fill": 0.8,
    }[fill_role]
    return torch.tensor(
        [
            _clamp(float(raw.get("swing", 0.0) or 0.0), -1.0, 1.0),
            _clamp(float(raw.get("humanize", 0.0) or 0.0), 0.0, 1.0),
            _clamp(float(raw.get("ghost_density", ghost_default) or 0.0), 0.0, 1.0),
            _clamp(float(raw.get("hihat_density", hat_default) or 0.0), 0.0, 1.0),
            _clamp(float(raw.get("fill_density", fill_default) or 0.0), 0.0, 1.0),
        ],
        dtype=torch.float32,
    ).contiguous()


def control_tensor_from_public_controls(
    raw_controls: Mapping[str, Any] | Sequence[float] | torch.Tensor | None,
    *,
    control_names: Sequence[str] | None = SKETCH_CONTROL_NAMES,
) -> torch.Tensor:
    names = tuple(str(name) for name in list(control_names or SKETCH_CONTROL_NAMES))
    if raw_controls is None:
        raw_controls = {}
    if not isinstance(raw_controls, Mapping):
        values = [float(x) for x in torch.as_tensor(raw_controls, dtype=torch.float32).view(-1).tolist()]
        if int(len(values)) == int(len(names)):
            return torch.tensor(values, dtype=torch.float32).contiguous()
        if int(len(values)) == int(len(SKETCH_CONTROL_NAMES_V4)):
            raw_controls = dict(zip(SKETCH_CONTROL_NAMES_V4, values))
        elif int(len(values)) == int(len(SKETCH_CONTROL_NAMES_V1)):
            raw_controls = dict(zip(SKETCH_CONTROL_NAMES_V1, values))
        elif int(len(values)) == int(len(SKETCH_CONTROL_NAMES_V3)):
            raw_controls = dict(zip(SKETCH_CONTROL_NAMES_V3, values))
        elif int(len(values)) == int(len(SKETCH_CONTROL_NAMES_V2_17)):
            raw_controls = dict(zip(SKETCH_CONTROL_NAMES_V2_17, values))
        elif int(len(values)) == int(len(SKETCH_CONTROL_NAMES_V2)):
            raw_controls = dict(zip(SKETCH_CONTROL_NAMES_V2, values))
        else:
            raise ValueError(f"controls list must have {len(names)} values, got {len(values)}")
    raw = dict(raw_controls)
    if is_v1_sketch_control_names(names):
        return _legacy_controls_from_mapping(raw)
    if is_v2_sketch_control_names(names):
        if all(name in raw for name in names):
            return torch.tensor([float(raw[name]) for name in names], dtype=torch.float32).contiguous()
        if all(name in raw for name in SKETCH_CONTROL_NAMES_V2_17):
            values = [float(raw[name]) for name in SKETCH_CONTROL_NAMES_V2_17]
            if names == SKETCH_CONTROL_NAMES_V2_17:
                return torch.tensor(values, dtype=torch.float32).contiguous()
            return torch.tensor(
                [
                    *values,
                    _clamp(float(raw.get("snare_ornament_intensity", 0.5) or 0.5), 0.0, 1.0),
                    _clamp(float(raw.get("open_hat_intensity", 0.5) or 0.5), 0.0, 1.0),
                ],
                dtype=torch.float32,
            ).contiguous()
        if all(name in raw for name in SKETCH_CONTROL_NAMES_V3):
            ghost_density = float(raw.get("ghost_density", 0.0) or 0.0)
            hihat_density = float(raw.get("hihat_density", 0.5) or 0.5)
            hihat_openness = float(raw.get("hihat_openness", 0.0) or 0.0)
            fill_density = float(raw.get("fill_density", 0.0) or 0.0)
            raw.setdefault("snare_style", "ghosts_plus_rolls" if ghost_density >= 0.70 else ("ghosts" if ghost_density >= 0.25 else "plain"))
            raw.setdefault("hat_rate", "sixteenths" if hihat_density >= 0.70 else ("eighths" if hihat_density >= 0.35 else ("sparse_syncopated" if hihat_density > 0.0 else "none")))
            raw.setdefault("hat_color", "open" if hihat_openness >= 0.55 else ("pedal" if hihat_openness >= 0.25 else "closed"))
            raw.setdefault("fill_role", "ride_plus_fill" if fill_density >= 0.78 else ("tom_crash_fill" if fill_density > 0.0 else "none"))
            raw.setdefault("snare_ornament_intensity", ghost_density)
            raw.setdefault("open_hat_intensity", hihat_openness)
        encoded_v2 = encode_v2_sketch_controls(
            swing=float(raw.get("swing", 0.0) or 0.0),
            humanize=float(raw.get("humanize", 0.0) or 0.0),
            snare_style=str(raw.get("snare_style", "plain")),
            hat_rate=str(raw.get("hat_rate", "eighths")),
            hat_color=str(raw.get("hat_color", "closed")),
            fill_role=str(raw.get("fill_role", "none")),
            snare_ornament_intensity=float(raw.get("snare_ornament_intensity", 0.5) or 0.5),
            open_hat_intensity=float(raw.get("open_hat_intensity", 0.5) or 0.5),
        )
        if names == SKETCH_CONTROL_NAMES_V2_17:
            return encoded_v2[: len(SKETCH_CONTROL_NAMES_V2_17)].contiguous()
        return encoded_v2.contiguous()
    if is_v4_sketch_control_names(names):
        if all(name in raw for name in names):
            return torch.tensor([float(raw[name]) for name in names], dtype=torch.float32).contiguous()
        if all(name in raw for name in SKETCH_CONTROL_NAMES_V4):
            return torch.tensor([float(raw[name]) for name in SKETCH_CONTROL_NAMES_V4], dtype=torch.float32).contiguous()
        if all(name in raw for name in SKETCH_CONTROL_NAMES_V2):
            decoded_v2 = decode_sketch_controls(
                [float(raw[name]) for name in SKETCH_CONTROL_NAMES_V2],
                control_names=SKETCH_CONTROL_NAMES_V2,
            )
            for key in SKETCH_CONTROL_NAMES_V3:
                raw.setdefault(key, decoded_v2.get(key, 0.0))
            raw.setdefault("fill_role", decoded_v2.get("fill_role", "none"))
        elif all(name in raw for name in SKETCH_CONTROL_NAMES_V2_17):
            decoded_v2 = decode_sketch_controls(
                [float(raw[name]) for name in SKETCH_CONTROL_NAMES_V2_17],
                control_names=SKETCH_CONTROL_NAMES_V2_17,
            )
            for key in SKETCH_CONTROL_NAMES_V3:
                raw.setdefault(key, decoded_v2.get(key, 0.0))
            raw.setdefault("fill_role", decoded_v2.get("fill_role", "none"))
        if str(raw.get("feel_style", "")).strip():
            feel_style, feel_amount = (
                _normalize_control_choice(raw.get("feel_style", "straight"), choices=FEEL_STYLE_VALUES, default="straight"),
                _clamp(float(raw.get("feel_amount", raw.get("humanize", 0.0)) or 0.0), 0.0, 1.0),
            )
        else:
            feel_style, feel_amount = _feel_from_swing_humanize(
                float(raw.get("swing", 0.0) or 0.0),
                float(raw.get("humanize", raw.get("feel_amount", 0.0)) or 0.0),
            )
            if "feel_amount" in raw:
                feel_amount = _clamp(float(raw.get("feel_amount", feel_amount) or 0.0), 0.0, 1.0)
        ghost_density = _clamp(float(raw.get("ghost_density", 0.0) or 0.0), 0.0, 1.0)
        split = split_ghost_densities(ghost_density)
        fill_density = _clamp(float(raw.get("fill_density", _fill_density_from_v2(raw.get("fill_role", "none"))) or 0.0), 0.0, 1.0)
        fill_role = raw.get("fill_role")
        if fill_role is None:
            fill_role = "ride_plus_fill" if fill_density >= 0.78 else ("tom_crash_fill" if fill_density > 0.0 else "none")
        encoded_v4 = encode_v4_sketch_controls(
            feel_style=feel_style,
            feel_amount=feel_amount,
            ghost_density=ghost_density,
            kick_ghost_density=raw.get("kick_ghost_density", split["kick_ghost_density"]),
            snare_ghost_density=raw.get("snare_ghost_density", split["snare_ghost_density"]),
            snare_roll_density=raw.get("snare_roll_density", split["snare_roll_density"]),
            hihat_density=float(raw.get("hihat_density", _hihat_density_from_v2(raw.get("hat_rate", "eighths"))) or 0.0),
            hihat_openness=float(raw.get("hihat_openness", _hihat_openness_from_v2(raw.get("hat_color", "closed"), raw.get("open_hat_intensity"))) or 0.0),
            fill_density=fill_density,
            fill_role=str(fill_role),
            fill_start=float(raw.get("fill_start", 0.0) or 0.0),
            fill_length=float(raw.get("fill_length", 0.0) or 0.0),
            tom_direction=str(raw.get("tom_direction", "none")),
            fill_accent_shape=str(raw.get("fill_accent_shape", "flat")),
        )
        if names == SKETCH_CONTROL_NAMES_V4:
            return encoded_v4.contiguous()
        projected = {name: float(encoded_v4[idx].item()) for idx, name in enumerate(SKETCH_CONTROL_NAMES_V4)}
        return torch.tensor([float(projected.get(name, 0.0)) for name in names], dtype=torch.float32).contiguous()
    if all(name in raw for name in names):
        return torch.tensor([float(raw[name]) for name in names], dtype=torch.float32).contiguous()
    if all(name in raw for name in SKETCH_CONTROL_NAMES_V2):
        decoded_v2 = decode_sketch_controls(
            [float(raw[name]) for name in SKETCH_CONTROL_NAMES_V2],
            control_names=SKETCH_CONTROL_NAMES_V2,
        )
        for key in SKETCH_CONTROL_NAMES_V3:
            raw.setdefault(key, decoded_v2.get(key, 0.0))
    elif all(name in raw for name in SKETCH_CONTROL_NAMES_V2_17):
        decoded_v2 = decode_sketch_controls(
            [float(raw[name]) for name in SKETCH_CONTROL_NAMES_V2_17],
            control_names=SKETCH_CONTROL_NAMES_V2_17,
        )
        for key in SKETCH_CONTROL_NAMES_V3:
            raw.setdefault(key, decoded_v2.get(key, 0.0))
    ghost_density_raw = raw.get("ghost_density")
    hihat_density_raw = raw.get("hihat_density")
    hihat_openness_raw = raw.get("hihat_openness")
    fill_density_raw = raw.get("fill_density")
    encoded_v3 = encode_sketch_controls(
        swing=float(raw.get("swing", 0.0) or 0.0),
        humanize=float(raw.get("humanize", 0.0) or 0.0),
        ghost_density=None if ghost_density_raw is None else float(ghost_density_raw),
        hihat_density=None if hihat_density_raw is None else float(hihat_density_raw),
        hihat_openness=None if hihat_openness_raw is None else float(hihat_openness_raw),
        fill_density=None if fill_density_raw is None else float(fill_density_raw),
        snare_style=str(raw.get("snare_style", "plain")),
        hat_rate=str(raw.get("hat_rate", "eighths")),
        hat_color=str(raw.get("hat_color", "closed")),
        fill_role=str(raw.get("fill_role", "none")),
        snare_ornament_intensity=None if raw.get("snare_ornament_intensity") is None else float(raw["snare_ornament_intensity"]),
        open_hat_intensity=None if raw.get("open_hat_intensity") is None else float(raw["open_hat_intensity"]),
    )
    if names == SKETCH_CONTROL_NAMES_V3:
        return encoded_v3.contiguous()
    projected = {name: float(encoded_v3[idx].item()) for idx, name in enumerate(SKETCH_CONTROL_NAMES_V3)}
    return torch.tensor([float(projected.get(name, 0.0)) for name in names], dtype=torch.float32).contiguous()


def _stacked_onset_vel_from_grid(grid_ft: torch.Tensor) -> torch.Tensor:
    grid = torch.as_tensor(grid_ft, dtype=torch.float32)
    if int(grid.dim()) != 2:
        raise ValueError(f"grid_ft must be [24,T], got {tuple(grid.shape)}")
    family_count = int(len(FAMILY_STATE_FAMILY_NAMES))
    expected_rows = int(3 * family_count)
    if int(grid.shape[0]) != int(expected_rows):
        raise ValueError(f"expected {expected_rows} grid rows, got {tuple(grid.shape)}")
    return grid[1::3, :].contiguous()


def _step_index_for_time(step_boundaries_t: torch.Tensor, time_sec: float) -> int:
    boundaries = torch.as_tensor(step_boundaries_t, dtype=torch.float32).view(-1)
    if int(boundaries.numel()) < 2:
        return 0
    step_anchors = boundaries[:-1]
    distances = (step_anchors - float(time_sec)).abs()
    return int(torch.argmin(distances).item())


def extract_event_targets_from_payload(
    payload: Mapping[str, Any],
    *,
    max_slots: int = DEFAULT_SKETCH_MAX_SLOTS,
) -> dict[str, torch.Tensor]:
    class_names = tuple(str(x) for x in list(payload.get("class_names") or FAMILY_STATE_FAMILY_NAMES))
    family_count = int(len(class_names))
    num_steps = DEFAULT_NUM_STEPS
    slots = int(max(1, int(max_slots)))

    family_onsets_ft = torch.as_tensor(payload["family_onsets_ft"], dtype=torch.bool)
    grid_ids_ft = torch.as_tensor(payload["grid_ids_ft"], dtype=torch.long)
    grid_times_sec_t = torch.as_tensor(payload["grid_times_sec_t"], dtype=torch.float32).view(-1)
    step_boundaries_sec_t = torch.as_tensor(payload["step_boundaries_sec_rel"], dtype=torch.float32).view(-1)
    onset_vel_ft = _stacked_onset_vel_from_grid(torch.as_tensor(payload["grid_ft"], dtype=torch.float32))

    if int(step_boundaries_sec_t.numel()) != int(num_steps + 1):
        raise ValueError(f"step_boundaries_sec_rel must have {num_steps + 1} entries")
    if tuple(family_onsets_ft.shape) != tuple(grid_ids_ft.shape):
        raise ValueError(
            f"family_onsets_ft and grid_ids_ft must match, got {tuple(family_onsets_ft.shape)} / {tuple(grid_ids_ft.shape)}"
        )
    if tuple(family_onsets_ft.shape) != tuple(onset_vel_ft.shape):
        raise ValueError(
            f"family_onsets_ft and onset_vel_ft must match, got {tuple(family_onsets_ft.shape)} / {tuple(onset_vel_ft.shape)}"
        )
    if int(family_onsets_ft.shape[0]) != int(family_count):
        raise ValueError(f"class_names count does not match onset rows: {family_count} vs {tuple(family_onsets_ft.shape)}")

    target_presence = torch.zeros((family_count, num_steps, slots), dtype=torch.float32)
    target_velocity = torch.zeros_like(target_presence)
    target_offset = torch.zeros_like(target_presence)
    target_class_id = torch.zeros((family_count, num_steps, slots), dtype=torch.long)
    target_count = torch.zeros((family_count, num_steps), dtype=torch.long)

    for family_idx in range(family_count):
        hit_frames = torch.nonzero(family_onsets_ft[int(family_idx)], as_tuple=False).view(-1)
        by_step: list[list[tuple[float, int, float, int]]] = [[] for _ in range(num_steps)]
        for frame_idx_t in hit_frames:
            frame_idx = int(frame_idx_t.item())
            if frame_idx < 0 or frame_idx >= int(grid_times_sec_t.numel()):
                continue
            time_sec = float(grid_times_sec_t[int(frame_idx)].item())
            step_idx = _step_index_for_time(step_boundaries_sec_t, time_sec)
            if not (0 <= int(step_idx) < int(num_steps)):
                continue
            velocity = float(onset_vel_ft[int(family_idx), int(frame_idx)].item())
            class_id = int(grid_ids_ft[int(family_idx), int(frame_idx)].item())
            by_step[int(step_idx)].append((float(velocity), int(frame_idx), float(time_sec), int(max(0, class_id))))

        for step_idx, events in enumerate(by_step):
            if not events:
                continue
            # Keep the strongest events in dense bins. Frame index is the stable tie-breaker.
            events_sorted = sorted(events, key=lambda item: (-float(item[0]), int(item[1])))[:slots]
            target_count[int(family_idx), int(step_idx)] = int(min(len(events), slots))
            step_start = float(step_boundaries_sec_t[int(step_idx)].item())
            step_end = float(step_boundaries_sec_t[int(step_idx) + 1].item())
            step_span = max(1.0e-6, float(step_end) - float(step_start))
            for slot_idx, (velocity, _frame_idx, time_sec, class_id) in enumerate(events_sorted):
                target_presence[int(family_idx), int(step_idx), int(slot_idx)] = 1.0
                target_velocity[int(family_idx), int(step_idx), int(slot_idx)] = float(max(0.0, min(1.0, velocity)))
                offset = (float(time_sec) - float(step_start)) / float(step_span)
                target_offset[int(family_idx), int(step_idx), int(slot_idx)] = float(max(-0.5, min(0.5, offset)))
                target_class_id[int(family_idx), int(step_idx), int(slot_idx)] = int(class_id)

    return {
        "target_presence": target_presence.contiguous(),
        "target_velocity": target_velocity.contiguous(),
        "target_offset": target_offset.contiguous(),
        "target_class_id": target_class_id.contiguous(),
        "target_count": target_count.contiguous(),
    }


def derive_fill_phrase_targets(
    target_presence: torch.Tensor,
    target_velocity: torch.Tensor,
    *,
    class_names: Sequence[str] | None = None,
) -> dict[str, torch.Tensor]:
    names = tuple(str(x) for x in (class_names or FAMILY_STATE_FAMILY_NAMES))
    index = _family_index_map(names)
    presence = torch.as_tensor(target_presence, dtype=torch.float32).gt(0.5)
    velocity = torch.as_tensor(target_velocity, dtype=torch.float32)
    tom_rank_by_name = {"tom_high": 0, "tom_mid": 1, "tom_floor": 2}
    events: list[tuple[int, int, float]] = []
    crash_steps: set[int] = set()
    ride_steps: set[int] = set()

    for family_name, rank in tom_rank_by_name.items():
        if str(family_name) not in index:
            continue
        family_idx = int(index[str(family_name)])
        active_positions = torch.nonzero(presence[family_idx], as_tuple=False)
        for position in active_positions:
            step_idx = int(position[0].item())
            slot_idx = int(position[1].item())
            events.append((int(step_idx), int(rank), float(velocity[family_idx, step_idx, slot_idx].item())))
    if "crash" in index:
        crash_idx = int(index["crash"])
        crash_steps = {int(pos[0].item()) for pos in torch.nonzero(presence[crash_idx], as_tuple=False)}
    if "ride" in index:
        ride_idx = int(index["ride"])
        ride_steps = {int(pos[0].item()) for pos in torch.nonzero(presence[ride_idx], as_tuple=False)}

    if events:
        events_sorted = sorted(events, key=lambda item: (int(item[0]), int(item[1])))
        first_step = int(events_sorted[0][0])
        last_step = int(max(step for step, _rank, _vel in events_sorted))
        phrase_len = int(max(1, min(DEFAULT_NUM_STEPS, (last_step - first_step) + 1)))
        ranks = [int(rank) for _step, rank, _vel in events_sorted]
        velocities = [float(vel) for _step, _rank, vel in events_sorted]
        if int(ranks[-1]) > int(ranks[0]):
            direction = "down"
        elif int(ranks[-1]) < int(ranks[0]):
            direction = "up"
        elif int(len(set(ranks))) > 1:
            direction = "mixed"
        else:
            direction = "none"
        first_vel = float(velocities[0])
        last_vel = float(velocities[-1])
        peak_vel = float(max(velocities))
        if last_vel >= max(first_vel + 0.08, peak_vel - 0.03):
            accent_shape = "peak_end"
        elif last_vel >= first_vel + 0.08:
            accent_shape = "ramp_up"
        elif first_vel >= last_vel + 0.08:
            accent_shape = "ramp_down"
        else:
            accent_shape = "flat"
    else:
        first_step = 0
        phrase_len = 0
        direction = "none"
        accent_shape = "flat"

    role = "none"
    if events and ride_steps:
        role = "ride_plus_fill"
    elif events or crash_steps:
        role = "tom_crash_fill"
    elif ride_steps:
        role = "ride"

    return {
        "fill_start": torch.tensor(int(first_step + 1 if phrase_len > 0 else 0), dtype=torch.long),
        "fill_length": torch.tensor(int(max(0, min(DEFAULT_NUM_STEPS, phrase_len))), dtype=torch.long),
        "tom_direction": torch.tensor(int(TOM_DIRECTION_VALUES.index(direction)), dtype=torch.long),
        "fill_accent_shape": torch.tensor(int(FILL_ACCENT_SHAPE_VALUES.index(accent_shape)), dtype=torch.long),
        "fill_role": torch.tensor(int(FILL_ROLE_VALUES.index(role)), dtype=torch.long),
    }


def derive_controls_from_event_targets(
    target_presence: torch.Tensor,
    target_velocity: torch.Tensor,
    target_offset: torch.Tensor,
    target_class_id: torch.Tensor,
    *,
    class_names: Sequence[str] | None = None,
) -> torch.Tensor:
    names = tuple(str(x) for x in (class_names or FAMILY_STATE_FAMILY_NAMES))
    index = _family_index_map(names)
    presence = torch.as_tensor(target_presence, dtype=torch.float32)
    velocity = torch.as_tensor(target_velocity, dtype=torch.float32)
    offset = torch.as_tensor(target_offset, dtype=torch.float32)
    class_id = torch.as_tensor(target_class_id, dtype=torch.long)

    def _idx(name: str) -> int:
        if str(name) not in index:
            raise KeyError(f"missing family {name!r} in class_names={names}")
        return int(index[str(name)])

    ksh_idx = [_idx("kick"), _idx("snare"), _idx("hihat")]
    active_ksh = presence[ksh_idx].gt(0.5)
    active_offsets = offset[ksh_idx][active_ksh]
    odd_step_offsets: list[torch.Tensor] = []
    for step_idx in range(DEFAULT_NUM_STEPS):
        if int(step_idx) % 2 != 1:
            continue
        step_active = active_ksh[:, int(step_idx), :]
        if bool(step_active.any().item()):
            odd_step_offsets.append(offset[ksh_idx, int(step_idx), :][step_active])
    if odd_step_offsets:
        swing = torch.cat(odd_step_offsets).mean().mul(3.0).clamp(-1.0, 1.0)
    else:
        swing = torch.tensor(0.0, dtype=torch.float32)
    if int(active_offsets.numel()) > 1:
        humanize = active_offsets.std(unbiased=False).mul(4.0).clamp(0.0, 1.0)
    else:
        humanize = torch.tensor(0.0, dtype=torch.float32)
    feel_style, feel_amount = _feel_from_swing_humanize(float(swing.item()), float(humanize.item()))
    if int(active_offsets.numel()) > 0:
        mean_offset = float(active_offsets.mean().item())
        if mean_offset <= -0.045:
            feel_style = "pushed"
            feel_amount = max(float(feel_amount), _clamp(abs(mean_offset) * 5.0, 0.0, 1.0))
        elif mean_offset >= 0.045 and str(feel_style) != "swing":
            feel_style = "laid_back"
            feel_amount = max(float(feel_amount), _clamp(abs(mean_offset) * 5.0, 0.0, 1.0))

    kick_idx = _idx("kick")
    kick_active = presence[kick_idx].gt(0.5)
    kick_slot_extra = torch.zeros_like(kick_active, dtype=torch.bool)
    if int(kick_slot_extra.shape[-1]) > 1:
        kick_slot_extra[:, 1:] = kick_active[:, 1:]
    kick_nonbackbeat = torch.ones((DEFAULT_NUM_STEPS, 1), dtype=torch.bool)
    for backbeat_step in KICK_BACKBEAT_STEPS:
        if 0 <= int(backbeat_step) < int(DEFAULT_NUM_STEPS):
            kick_nonbackbeat[int(backbeat_step), 0] = False
    kick_ghost = (
        kick_active
        & velocity[kick_idx].le(float(KICK_GHOST_VELOCITY_THRESHOLD))
        & kick_nonbackbeat
    ) | kick_slot_extra

    snare_idx = _idx("snare")
    snare_active = presence[snare_idx].gt(0.5)
    nonbackbeat = torch.ones((DEFAULT_NUM_STEPS, 1), dtype=torch.bool)
    for backbeat_step in SNARE_BACKBEAT_STEPS:
        if 0 <= int(backbeat_step) < int(DEFAULT_NUM_STEPS):
            nonbackbeat[int(backbeat_step), 0] = False
    snare_ghost = (
        snare_active
        & class_id[snare_idx].eq(0)
        & velocity[snare_idx].le(float(SNARE_GHOST_VELOCITY_THRESHOLD))
        & nonbackbeat
    )
    snare_slot_extra = torch.zeros_like(snare_active, dtype=torch.bool)
    if int(snare_slot_extra.shape[-1]) > 1:
        snare_slot_extra[:, 1:] = snare_active[:, 1:]
    snare_run_steps = (
        snare_active
        & velocity[snare_idx].ge(float(SNARE_STRONG_VELOCITY_THRESHOLD))
        & nonbackbeat
    ).any(dim=-1)
    snare_adjacent = torch.zeros((DEFAULT_NUM_STEPS,), dtype=torch.bool)
    if int(DEFAULT_NUM_STEPS) > 1:
        snare_adjacent[:-1] |= snare_run_steps[1:]
        snare_adjacent[1:] |= snare_run_steps[:-1]
    snare_roll_run = (
        snare_active
        & nonbackbeat
        & snare_adjacent.view(DEFAULT_NUM_STEPS, 1)
        & velocity[snare_idx].ge(float(SNARE_STRONG_VELOCITY_THRESHOLD))
    )
    snare_roll_run[:, 1:] = False
    snare_roll_or_drag = (snare_slot_extra & ~snare_ghost) | snare_roll_run

    kick_ghost_density = min(1.0, float(kick_ghost.sum().item()) / 3.0)
    snare_ghost_density = min(1.0, float(snare_ghost.sum().item()) / 4.0)
    snare_roll_density = min(1.0, float(snare_roll_or_drag.sum().item()) / 3.0)
    ghost_score = 0.0
    ghost_score += 0.75 * float(kick_ghost_density)
    ghost_score += 0.85 * float(snare_ghost_density)
    ghost_score += 1.00 * float(snare_roll_density)
    ghost_density = _clamp(ghost_score, 0.0, 1.0)
    if bool(snare_roll_or_drag.any().item()):
        ghost_density = max(float(ghost_density), 0.65)

    hihat_idx = _idx("hihat")
    hihat_active = presence[hihat_idx].gt(0.5)
    hihat_steps = hihat_active.any(dim=-1)
    hihat_step_count = int(hihat_steps.sum().item())
    off16_count = int(hihat_steps[1::2].sum().item())
    hat_coverage = float(hihat_step_count) / float(DEFAULT_NUM_STEPS)
    off16_coverage = float(off16_count) / 8.0
    hihat_density = _clamp((1.35 * float(hat_coverage)) + (0.45 * float(off16_coverage)), 0.0, 1.0)

    hihat_class = class_id[hihat_idx]
    hihat_total = int(hihat_active.sum().item())
    hihat_open = hihat_active & torch.zeros_like(hihat_active, dtype=torch.bool)
    hihat_pedal = hihat_active & torch.zeros_like(hihat_active, dtype=torch.bool)
    for open_id in HIHAT_OPEN_CLASS_IDS:
        hihat_open |= hihat_active & hihat_class.eq(int(open_id))
    for pedal_id in HIHAT_PEDAL_CLASS_IDS:
        hihat_pedal |= hihat_active & hihat_class.eq(int(pedal_id))
    open_ratio = float(hihat_open.to(dtype=torch.float32).sum().item()) / float(max(1, hihat_total))
    pedal_ratio = float(hihat_pedal.to(dtype=torch.float32).sum().item()) / float(max(1, hihat_total))
    open_intensity = (
        _normalize_piecewise(float(velocity[hihat_idx][hihat_open].mean().item()), 0.45, 0.70, 1.0)
        if bool(hihat_open.any().item())
        else 0.0
    )
    hihat_openness = _clamp(max(0.45 * float(pedal_ratio), 0.65 * float(open_ratio), 0.75 * float(open_intensity)), 0.0, 1.0)

    tom_count = 0
    crash_count = 0
    ride_count = 0
    for family_name in ("tom_high", "tom_mid", "tom_floor"):
        if family_name in index:
            tom_count += int(presence[_idx(family_name)].gt(0.5).sum().item())
    if "crash" in index:
        crash_count = int(presence[_idx("crash")].gt(0.5).sum().item())
    if "ride" in index:
        ride_count = int(presence[_idx("ride")].gt(0.5).sum().item())
    fill_density = _clamp(
        max(
            float(tom_count) / 5.0,
            0.55 * min(1.0, float(crash_count) / 2.0),
            0.65 * min(1.0, float(ride_count) / 5.0),
        ),
        0.0,
        1.0,
    )
    phrase = derive_fill_phrase_targets(presence, velocity, class_names=names)
    phrase_start_class = int(phrase["fill_start"].item())
    phrase_length_class = int(phrase["fill_length"].item())
    fill_role = FILL_ROLE_VALUES[int(phrase["fill_role"].item())]
    tom_direction = TOM_DIRECTION_VALUES[int(phrase["tom_direction"].item())]
    fill_accent_shape = FILL_ACCENT_SHAPE_VALUES[int(phrase["fill_accent_shape"].item())]

    return encode_v4_sketch_controls(
        feel_style=str(feel_style),
        feel_amount=float(feel_amount),
        ghost_density=float(ghost_density),
        kick_ghost_density=float(kick_ghost_density),
        snare_ghost_density=float(snare_ghost_density),
        snare_roll_density=float(snare_roll_density),
        hihat_density=float(hihat_density),
        hihat_openness=float(hihat_openness),
        fill_density=float(fill_density),
        fill_role=str(fill_role),
        fill_start=(float(phrase_start_class - 1) / float(max(1, DEFAULT_NUM_STEPS - 1))) if phrase_start_class > 0 else 0.0,
        fill_length=float(phrase_length_class) / float(DEFAULT_NUM_STEPS),
        tom_direction=str(tom_direction),
        fill_accent_shape=str(fill_accent_shape),
    )


def _coarse_sketch_from_event_targets(
    target_presence: torch.Tensor,
    target_velocity: torch.Tensor,
    *,
    class_names: Sequence[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    names = tuple(str(x) for x in (class_names or FAMILY_STATE_FAMILY_NAMES))
    index = _family_index_map(names)
    presence = torch.as_tensor(target_presence, dtype=torch.float32)
    velocity = torch.as_tensor(target_velocity, dtype=torch.float32)
    sketch_hits = torch.zeros((len(SKETCH_FAMILY_NAMES), DEFAULT_NUM_STEPS), dtype=torch.float32)
    sketch_vel = torch.zeros_like(sketch_hits)

    def _idx(name: str) -> int:
        if str(name) not in index:
            raise KeyError(f"missing family {name!r} in class_names={names}")
        return int(index[str(name)])

    kick_idx = _idx("kick")
    kick_sketch_idx = SKETCH_FAMILY_NAMES.index("kick")
    for step_idx in range(DEFAULT_NUM_STEPS):
        active = presence[kick_idx, int(step_idx)].gt(0.5)
        if bool(active.any().item()):
            step_velocities = velocity[kick_idx, int(step_idx)][active]
            if (
                int(step_idx) not in set(KICK_BACKBEAT_STEPS)
                and float(step_velocities.max().item()) <= float(KICK_GHOST_VELOCITY_THRESHOLD)
            ):
                continue
            sketch_hits[kick_sketch_idx, int(step_idx)] = 1.0
            sketch_vel[kick_sketch_idx, int(step_idx)] = float(step_velocities.max().item())

    snare_idx = _idx("snare")
    snare_sketch_idx = SKETCH_FAMILY_NAMES.index("snare")
    snare_nonbackbeat = torch.ones((DEFAULT_NUM_STEPS, 1), dtype=torch.bool)
    for backbeat_step in SNARE_BACKBEAT_STEPS:
        if 0 <= int(backbeat_step) < int(DEFAULT_NUM_STEPS):
            snare_nonbackbeat[int(backbeat_step), 0] = False
    snare_step_run = (
        presence[snare_idx].gt(0.5)
        & velocity[snare_idx].ge(float(SNARE_STRONG_VELOCITY_THRESHOLD))
        & snare_nonbackbeat
    ).any(dim=-1)
    snare_step_adjacent = torch.zeros((DEFAULT_NUM_STEPS,), dtype=torch.bool)
    if int(DEFAULT_NUM_STEPS) > 1:
        snare_step_adjacent[:-1] |= snare_step_run[1:]
        snare_step_adjacent[1:] |= snare_step_run[:-1]
    for step_idx in range(DEFAULT_NUM_STEPS):
        active = presence[snare_idx, int(step_idx)].gt(0.5)
        if not bool(active.any().item()):
            continue
        step_velocities = velocity[snare_idx, int(step_idx)][active]
        strong = step_velocities.ge(float(SNARE_STRONG_VELOCITY_THRESHOLD))
        if int(step_idx) not in set(SNARE_BACKBEAT_STEPS) and not bool(strong.any().item()):
            continue
        if (
            int(step_idx) not in set(SNARE_BACKBEAT_STEPS)
            and bool(strong.any().item())
            and bool(snare_step_adjacent[int(step_idx)].item())
        ):
            continue
        chosen_velocities = step_velocities[strong] if bool(strong.any().item()) else step_velocities
        sketch_hits[snare_sketch_idx, int(step_idx)] = 1.0
        sketch_vel[snare_sketch_idx, int(step_idx)] = float(chosen_velocities.max().item())

    hihat_idx = _idx("hihat")
    hihat_sketch_idx = SKETCH_FAMILY_NAMES.index("hihat")
    for step_idx in range(0, DEFAULT_NUM_STEPS, 2):
        active = presence[hihat_idx, int(step_idx)].gt(0.5)
        if bool(active.any().item()):
            sketch_hits[hihat_sketch_idx, int(step_idx)] = 1.0
            sketch_vel[hihat_sketch_idx, int(step_idx)] = float(velocity[hihat_idx, int(step_idx)][active].max().item())

    return sketch_hits.contiguous(), sketch_vel.clamp(0.0, 1.0).contiguous()


def derive_ornament_budget_targets(
    target_presence: torch.Tensor,
    target_velocity: torch.Tensor,
    target_class_id: torch.Tensor,
    sketch_hits: torch.Tensor | None = None,
    *,
    class_names: Sequence[str] | None = None,
) -> torch.Tensor:
    names = tuple(str(x) for x in (class_names or FAMILY_STATE_FAMILY_NAMES))
    index = _family_index_map(names)
    presence = torch.as_tensor(target_presence, dtype=torch.float32).gt(0.5)
    velocity = torch.as_tensor(target_velocity, dtype=torch.float32)
    class_id = torch.as_tensor(target_class_id, dtype=torch.long)
    sketch = (
        torch.as_tensor(sketch_hits, dtype=torch.float32).gt(0.5)
        if sketch_hits is not None
        else torch.zeros((len(SKETCH_FAMILY_NAMES), DEFAULT_NUM_STEPS), dtype=torch.bool)
    )

    def _idx(name: str) -> int:
        if str(name) not in index:
            raise KeyError(f"missing family {name!r} in class_names={names}")
        return int(index[str(name)])

    def _sketch_anchor_mask(family: str) -> torch.Tensor:
        mask = torch.zeros_like(presence[_idx(family)])
        if str(family) in SKETCH_FAMILY_NAMES and int(mask.shape[-1]) > 0:
            sketch_idx = int(SKETCH_FAMILY_NAMES.index(str(family)))
            mask[:, 0] = sketch[int(sketch_idx), : int(mask.shape[0])]
        return mask

    kick_idx = _idx("kick")
    kick_active = presence[kick_idx]
    kick_anchor = _sketch_anchor_mask("kick")
    kick_slot_extra = torch.zeros_like(kick_active, dtype=torch.bool)
    if int(kick_slot_extra.shape[-1]) > 1:
        kick_slot_extra[:, 1:] = kick_active[:, 1:]
    kick_ghost = (
        (
            kick_active
            & velocity[kick_idx].le(float(KICK_GHOST_VELOCITY_THRESHOLD))
        )
        | kick_slot_extra
    ) & ~kick_anchor

    snare_idx = _idx("snare")
    snare_active = presence[snare_idx]
    snare_anchor = _sketch_anchor_mask("snare")
    nonbackbeat = torch.ones((DEFAULT_NUM_STEPS, 1), dtype=torch.bool)
    for backbeat_step in SNARE_BACKBEAT_STEPS:
        if 0 <= int(backbeat_step) < int(nonbackbeat.shape[0]):
            nonbackbeat[int(backbeat_step), 0] = False
    snare_ghost = (
        snare_active
        & class_id[snare_idx].eq(0)
        & velocity[snare_idx].le(float(SNARE_GHOST_VELOCITY_THRESHOLD))
        & nonbackbeat
        & ~snare_anchor
    )
    snare_roll_drag = snare_active & ~snare_ghost & ~snare_anchor
    if int(snare_roll_drag.shape[-1]) > 1:
        snare_roll_drag[:, 0] = False
    else:
        snare_roll_drag.zero_()
    snare_step_run = (
        snare_active
        & velocity[snare_idx].ge(float(SNARE_STRONG_VELOCITY_THRESHOLD))
        & nonbackbeat
    ).any(dim=-1)
    snare_step_adjacent = torch.zeros((DEFAULT_NUM_STEPS,), dtype=torch.bool)
    if int(DEFAULT_NUM_STEPS) > 1:
        snare_step_adjacent[:-1] |= snare_step_run[1:]
        snare_step_adjacent[1:] |= snare_step_run[:-1]
    snare_roll_run = (
        snare_active
        & ~snare_ghost
        & ~snare_anchor
        & nonbackbeat
        & snare_step_adjacent.view(DEFAULT_NUM_STEPS, 1)
        & velocity[snare_idx].ge(float(SNARE_STRONG_VELOCITY_THRESHOLD))
    )
    if int(snare_roll_run.shape[-1]) > 1:
        snare_roll_run[:, 1:] = False

    hihat_idx = _idx("hihat")
    hihat_active = presence[hihat_idx]
    off16_steps = torch.zeros((DEFAULT_NUM_STEPS, 1), dtype=torch.bool)
    off16_steps[1::2, 0] = True
    off16_hat = hihat_active & off16_steps
    hihat_open = hihat_active & torch.zeros_like(hihat_active, dtype=torch.bool)
    for open_id in HIHAT_OPEN_CLASS_IDS:
        hihat_open |= hihat_active & class_id[hihat_idx].eq(int(open_id))

    tom_fill = torch.zeros_like(hihat_active)
    for family_name in ("tom_high", "tom_mid", "tom_floor"):
        if str(family_name) in index:
            tom_fill |= presence[_idx(family_name)]
    crash = presence[_idx("crash")] if "crash" in index else torch.zeros_like(hihat_active)
    ride = presence[_idx("ride")] if "ride" in index else torch.zeros_like(hihat_active)

    raw_counts = [
        int(kick_ghost.sum().item()),
        int(snare_ghost.sum().item()),
        int(snare_roll_drag.sum().item()),
        int(snare_roll_run.sum().item()),
        int(off16_hat.sum().item()),
        int(hihat_open.sum().item()),
        int(tom_fill.sum().item()),
        int(crash.sum().item()),
        int(ride.sum().item()),
    ]
    capped = [
        int(max(0, min(int(count), int(max_count))))
        for count, max_count in zip(raw_counts, ORNAMENT_BUDGET_MAX_COUNTS, strict=True)
    ]
    return torch.tensor(capped, dtype=torch.long).contiguous()


def extract_sketch_training_example_from_payload(
    payload: Mapping[str, Any],
    *,
    example_path: str | Path = "",
    max_slots: int = DEFAULT_SKETCH_MAX_SLOTS,
) -> SketchTrainingExample:
    class_names = tuple(str(x) for x in list(payload.get("class_names") or FAMILY_STATE_FAMILY_NAMES))
    targets = extract_event_targets_from_payload(payload, max_slots=int(max_slots))
    controls = derive_controls_from_event_targets(
        targets["target_presence"],
        targets["target_velocity"],
        targets["target_offset"],
        targets["target_class_id"],
        class_names=class_names,
    )
    sketch_hits, sketch_vel = _coarse_sketch_from_event_targets(
        targets["target_presence"],
        targets["target_velocity"],
        class_names=class_names,
    )
    target_ornament_budget = derive_ornament_budget_targets(
        targets["target_presence"],
        targets["target_velocity"],
        targets["target_class_id"],
        sketch_hits,
        class_names=class_names,
    )
    fill_phrase_targets = derive_fill_phrase_targets(
        targets["target_presence"],
        targets["target_velocity"],
        class_names=class_names,
    )

    return SketchTrainingExample(
        example_path=Path(example_path) if str(example_path) else Path(),
        source_id=str(payload.get("source_id") or ""),
        split=str(payload.get("split") or ""),
        beat_index=int(payload.get("beat_index", 0)),
        bpm=float(payload.get("bpm", 0.0) or 0.0),
        duration_sec=float(payload.get("duration_sec", 0.0) or 0.0),
        sketch_hits=sketch_hits.contiguous(),
        sketch_vel=sketch_vel.contiguous(),
        controls=controls.contiguous(),
        target_presence=targets["target_presence"],
        target_velocity=targets["target_velocity"],
        target_offset=targets["target_offset"],
        target_class_id=targets["target_class_id"],
        target_count=targets["target_count"],
        target_ornament_budget=target_ornament_budget,
        target_fill_start=fill_phrase_targets["fill_start"],
        target_fill_length=fill_phrase_targets["fill_length"],
        target_tom_direction=fill_phrase_targets["tom_direction"],
        target_fill_accent_shape=fill_phrase_targets["fill_accent_shape"],
    )


class SketchConditioningDataset(Dataset[SketchTrainingExample]):
    def __init__(
        self,
        cache_root: str | Path,
        *,
        split: str = "train",
        max_items: int = 0,
        max_slots: int = DEFAULT_SKETCH_MAX_SLOTS,
    ) -> None:
        super().__init__()
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.split = str(split).strip().lower()
        self.max_slots = int(max(1, int(max_slots)))
        manifest_path = self.cache_root / "manifests" / f"{self.split}.jsonl"
        self.rows = _load_jsonl(manifest_path)
        if int(max_items) > 0:
            self.rows = self.rows[: int(max_items)]
        if not self.rows:
            raise RuntimeError(f"no rows found for split={self.split!r} under {self.cache_root}")

    def __len__(self) -> int:
        return int(len(self.rows))

    def __getitem__(self, index: int) -> SketchTrainingExample:
        row = dict(self.rows[int(index)])
        example_path = (self.cache_root / str(row["out_pt"])).resolve()
        payload = dict(torch.load(example_path, map_location="cpu", weights_only=False))
        return extract_sketch_training_example_from_payload(
            payload,
            example_path=example_path,
            max_slots=int(self.max_slots),
        )


def collate_sketch_examples(items: Sequence[SketchTrainingExample]) -> dict[str, Any]:
    if not items:
        raise ValueError("expected at least one SketchTrainingExample")
    return {
        "sketch_hits": torch.stack([item.sketch_hits for item in items], dim=0).contiguous(),
        "sketch_vel": torch.stack([item.sketch_vel for item in items], dim=0).contiguous(),
        "controls": torch.stack([item.controls for item in items], dim=0).contiguous(),
        "target_presence": torch.stack([item.target_presence for item in items], dim=0).contiguous(),
        "target_velocity": torch.stack([item.target_velocity for item in items], dim=0).contiguous(),
        "target_offset": torch.stack([item.target_offset for item in items], dim=0).contiguous(),
        "target_class_id": torch.stack([item.target_class_id for item in items], dim=0).contiguous(),
        "target_count": torch.stack([item.target_count for item in items], dim=0).contiguous(),
        "target_ornament_budget": torch.stack([item.target_ornament_budget for item in items], dim=0).contiguous(),
        "target_fill_start": torch.stack([item.target_fill_start for item in items], dim=0).contiguous(),
        "target_fill_length": torch.stack([item.target_fill_length for item in items], dim=0).contiguous(),
        "target_tom_direction": torch.stack([item.target_tom_direction for item in items], dim=0).contiguous(),
        "target_fill_accent_shape": torch.stack([item.target_fill_accent_shape for item in items], dim=0).contiguous(),
        "bpm": torch.tensor([float(item.bpm) for item in items], dtype=torch.float32),
        "duration_sec": torch.tensor([float(item.duration_sec) for item in items], dtype=torch.float32),
        "source_id": [str(item.source_id) for item in items],
        "split": [str(item.split) for item in items],
        "beat_index": torch.tensor([int(item.beat_index) for item in items], dtype=torch.long),
        "example_path": [str(item.example_path) for item in items],
        "class_names": list(FAMILY_STATE_FAMILY_NAMES),
        "class_id_vocab_sizes": [int(x) for x in FAMILY_STATE_ID_VOCAB_SIZES],
        "sketch_family_names": list(SKETCH_FAMILY_NAMES),
        "control_names": list(SKETCH_CONTROL_NAMES),
        "public_control_names": list(SKETCH_PUBLIC_CONTROL_NAMES),
        "ornament_budget_group_names": list(ORNAMENT_BUDGET_GROUP_NAMES),
        "ornament_budget_max_counts": [int(x) for x in ORNAMENT_BUDGET_MAX_COUNTS],
    }


def build_sketch_dataloader(
    cache_root: str | Path,
    *,
    split: str = "train",
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    max_items: int = 0,
    max_slots: int = DEFAULT_SKETCH_MAX_SLOTS,
    pin_memory: bool = False,
) -> DataLoader:
    dataset = SketchConditioningDataset(
        cache_root,
        split=split,
        max_items=int(max_items),
        max_slots=int(max_slots),
    )
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        collate_fn=collate_sketch_examples,
    )
