#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import gc
import json
import math
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parent
RUNS_ROOT = Path(os.environ.get("DRUMTOGRID_RUNS_ROOT", REPO_ROOT.parent.parent / "runs")).resolve()
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("DO_NOT_TRACK", "1")


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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
from dataclasses import dataclass
from matplotlib.figure import Figure

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover - exercised by direct launch environments.
    raise SystemExit("listen_ui.py requires gradio. Install it in the active environment.") from exc

from data.diffusion_cache_utils import (
    FAMILY_STATE_FAMILY_NAMES,
    FAMILY_STATE_FEATURE_ROW_NAMES,
)
from data.encodec_utils import (
    load_audio_codec_model,
    load_target_pca_basis,
    resolve_codec_metadata_from_cache_config,
    resolve_codec_metadata_from_payload,
    resolve_device,
    resolve_target_layout_from_cache_config,
    resolve_target_pca_basis_path_from_cache_config,
)
from data.sketch_dataset import (
    FEEL_STYLE_VALUES,
    FILL_ACCENT_SHAPE_VALUES,
    FILL_ROLE_VALUES,
    HAT_COLOR_VALUES,
    HAT_RATE_VALUES,
    LEGACY_SKETCH_CONTROL_NAMES,
    SKETCH_CONTROL_NAMES,
    SKETCH_FAMILY_NAMES,
    SNARE_STYLE_VALUES,
    TOM_DIRECTION_VALUES,
    control_tensor_from_public_controls,
    is_legacy_sketch_control_names,
    sketch_controls_to_public_dict,
)
from data.sketch_render import DEFAULT_GRID_FRAME_RATE, DEFAULT_NUM_BEATS, build_diffusion_batch_from_events
from io_utils import save_audio, write_json
from model import (
    DEFAULT_BEAT_CROSSFADE_MS,
    DEFAULT_INFERENCE_GUIDANCE_SCALE,
    DEFAULT_INFERENCE_NUM_BEATS,
    DEFAULT_SAMPLE_X0_CLIP_NORM,
    _prepare_batch_tensors,
    apply_beat_crossfade,
    decode_latent_to_audio,
    denormalize_latent,
    resolve_codec_hop_length,
    resolve_inference_geometry,
    resolve_target_token_rate_hz,
    sample_ddpm,
    stitch_audio_segments_with_crossfade,
)
from scripts.sketch_diffusion_infer import (
    _load_diffusion_state,
    _load_sketch_expander,
    _resolve_diffusion_checkpoint,
    _samples_per_latent_frame,
)
from sketch_expander import decode_event_plan
from direct_regressor import DirectPCASequenceRegressor, DirectRegressorConfig


def _default_sketch_checkpoint() -> Path:
    for path in (
        RUNS_ROOT / "sketch_expander_dac44_native_v5" / "best_sketch_expander.pt",
        RUNS_ROOT / "sketch_expander_dac44_native_v5" / "last_sketch_expander.pt",
    ):
        if path.is_file():
            return path
    return RUNS_ROOT / "sketch_expander_dac44_native_v5" / "best_sketch_expander.pt"


def _default_diffusion_train_dir() -> Path:
    for path in (
        RUNS_ROOT / "runs_dac" / "dac_25steps",
        RUNS_ROOT / "runs_dac_ce" / "dac_25steps",
    ):
        if path.is_dir():
            return path
    return RUNS_ROOT / "runs_dac" / "dac_25steps"


DEFAULT_SKETCH_CHECKPOINT = _default_sketch_checkpoint()
MIN_UI_GUIDANCE_SCALE = 1.0
MAX_UI_GUIDANCE_SCALE = 5.0
DEFAULT_UI_GUIDANCE_SCALE = DEFAULT_INFERENCE_GUIDANCE_SCALE
# Guidance lives in the Advanced section; default a touch above training (1.0) so
# the first listen follows the sketch a little harder without inviting artifacts.
DEFAULT_ADVANCED_GUIDANCE_SCALE = 1.5
DEFAULT_DIFFUSION_TRAIN_DIR = _default_diffusion_train_dir()
DEFAULT_CACHE_ROOT = RUNS_ROOT / "mini_cache"
DEFAULT_OUT_DIR = REPO_ROOT / ".listen_ui_runs"
DEFAULT_MAX_RUN_DIRS = 12
STEP_HEADERS = [str(idx + 1) for idx in range(16)]
HIT_HEADERS = ["family", *STEP_HEADERS]


def _default_hits_table() -> list[list[Any]]:
    patterns = {
        "kick": {0, 8},
        "snare": {4, 12},
        "hihat": set(range(0, 16, 2)),
    }
    return [
        [family, *[bool(step in patterns[family]) for step in range(16)]]
        for family in SKETCH_FAMILY_NAMES
    ]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive sketch-to-grid diffusion UI.")
    parser.add_argument("--sketch-checkpoint", type=str, default=str(DEFAULT_SKETCH_CHECKPOINT))
    parser.add_argument("--diffusion-checkpoint", type=str, default="")
    parser.add_argument("--diffusion-train-dir", type=str, default=str(DEFAULT_DIFFUSION_TRAIN_DIR))
    parser.add_argument("--cache-root", type=str, default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--max-run-dirs",
        type=int,
        default=DEFAULT_MAX_RUN_DIRS,
        help="Maximum generated UI run directories to keep; use 0 to disable cleanup.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--grid-only", action="store_true", help="Disable audio generation and preview rendered grids only.")
    parser.add_argument("--num-beats", type=int, default=DEFAULT_INFERENCE_NUM_BEATS)
    parser.add_argument("--target-token-rate-hz", type=float, default=0.0)
    parser.add_argument("--grid-frame-rate", type=float, default=DEFAULT_GRID_FRAME_RATE)
    parser.add_argument("--guidance-scale", type=float, default=DEFAULT_UI_GUIDANCE_SCALE)
    parser.add_argument("--x0-clip-norm", type=float, default=DEFAULT_SAMPLE_X0_CLIP_NORM)
    parser.add_argument("--beat-crossfade-ms", type=float, default=DEFAULT_BEAT_CROSSFADE_MS)
    parser.add_argument("--chunk-crossfade-ms", type=float, default=25.0)
    parser.add_argument("--server-name", type=str, default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--show-error", action="store_true")
    return parser.parse_args(argv)


def _resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _require_file(path: str | Path, *, label: str) -> Path:
    resolved = _resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _require_dir(path: str | Path, *, label: str) -> Path:
    resolved = _resolve_path(path)
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _display_path(path: str | Path) -> str:
    resolved = _resolve_path(path)
    try:
        return str(resolved.relative_to(RUNS_ROOT))
    except ValueError:
        pass
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _resolve_display_or_path(value: str | Path) -> Path:
    raw = str(value).strip()
    if not raw:
        raise ValueError("empty checkpoint path")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = RUNS_ROOT / path
    return path.resolve()


def _discover_diffusion_checkpoints(*, explicit: str = "", train_dir: str | Path = DEFAULT_DIFFUSION_TRAIN_DIR) -> list[str]:
    candidates: list[Path] = []
    if str(explicit).strip():
        candidates.append(_resolve_display_or_path(str(explicit)))
    train_root = _resolve_path(train_dir)
    for name in ("best_diffusion.pt", "best.pt", "last.pt"):
        candidates.append(train_root / name)
    search_dirs = [RUNS_ROOT, RUNS_ROOT / "runs", RUNS_ROOT / "runs_dac", RUNS_ROOT / "runs_dac_ce"]
    for search_root in search_dirs:
        if not search_root.is_dir():
            continue
        for pattern in ("dac_*", "model_train*"):
            for run_dir in sorted(search_root.glob(pattern)):
                if not run_dir.is_dir():
                    continue
                for name in ("best_diffusion.pt", "best.pt", "last.pt"):
                    candidates.append(run_dir / name)
    direct_root = RUNS_ROOT / "runs_direct"
    if direct_root.is_dir():
        for run_dir in sorted(direct_root.glob("direct_*")):
            candidates.append(run_dir / "best_direct.pt")
    seen: set[Path] = set()
    choices: list[str] = []
    for candidate in candidates:
        path = _resolve_path(candidate)
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        choices.append(_display_path(path))
    return choices


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _as_jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_json_default))


def _coerce_rows(value: Any, *, fallback: Sequence[Sequence[Any]]) -> list[list[Any]]:
    if value is None:
        return [list(row) for row in fallback]
    if hasattr(value, "values"):
        return [list(row) for row in value.values.tolist()]
    if isinstance(value, Mapping) and "data" in value:
        return [list(row) for row in value["data"]]
    rows = list(value)
    return [list(row) if isinstance(row, (list, tuple)) else [row] for row in rows]


def _cell_is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def _cell_to_hit(value: Any) -> float:
    if _cell_is_missing(value):
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "t", "yes", "y", "on", "x", "hit"}:
            return 1.0
        if text in {"0", "false", "f", "no", "n", "off", "-"}:
            return 0.0
    try:
        return 1.0 if float(value) > 0.0 else 0.0
    except (TypeError, ValueError):
        return 0.0


def _table_start_col(rows: Sequence[Sequence[Any]]) -> int:
    if not rows:
        return 0
    first_row = list(rows[0])
    if len(first_row) >= 17:
        return 1
    return 0


def _hits_table_to_tensor(hits_table: Any) -> torch.Tensor:
    hit_rows = _coerce_rows(hits_table, fallback=_default_hits_table())
    hit_start = _table_start_col(hit_rows)
    hits = torch.zeros((len(SKETCH_FAMILY_NAMES), 16), dtype=torch.float32)
    for family_idx in range(len(SKETCH_FAMILY_NAMES)):
        hit_row = list(hit_rows[family_idx]) if family_idx < len(hit_rows) else []
        for step_idx in range(16):
            hit_value = hit_row[hit_start + step_idx] if hit_start + step_idx < len(hit_row) else 0
            hits[family_idx, step_idx] = float(_cell_to_hit(hit_value))
    return hits.contiguous()


def _derive_velocity_matrix(
    hits: torch.Tensor,
    *,
    velocity: float,
    variation: float,
    seed: int,
) -> torch.Tensor:
    hits_t = torch.as_tensor(hits, dtype=torch.float32)
    global_velocity = float(max(0.0, min(1.0, float(velocity))))
    base = torch.tensor(
        [
            float(global_velocity * 1.02),
            float(global_velocity),
            float(global_velocity * 0.72),
        ],
        dtype=torch.float32,
    ).clamp(0.05, 1.0).view(3, 1)
    variation_amount = float(max(0.0, min(1.0, float(variation))))
    multipliers = torch.ones((3, 16), dtype=torch.float32)
    for step_idx in range(16):
        beat_pos = int(step_idx) % 4
        if beat_pos == 0:
            multipliers[0, step_idx] += 0.06
            multipliers[2, step_idx] += 0.06
        elif beat_pos == 2:
            multipliers[0, step_idx] += 0.03
        else:
            multipliers[2, step_idx] -= 0.03

        if step_idx in {4, 12}:
            multipliers[1, step_idx] += 0.05
        else:
            # User-placed extra snares are usually ghost notes in a coarse sketch.
            multipliers[1, step_idx] *= 0.62

    if variation_amount > 0.0:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + 9176)
        jitter = torch.randn((3, 16), generator=generator, dtype=torch.float32) * (0.055 * variation_amount)
        multipliers = multipliers + jitter
    velocities = (base * multipliers).clamp(0.05, 1.0) * hits_t
    return velocities.contiguous()


def _resolve_output_beats(value: Any) -> int:
    try:
        beats = int(round(float(value)))
    except (TypeError, ValueError):
        beats = DEFAULT_NUM_BEATS
    beats = max(DEFAULT_NUM_BEATS, min(64, int(beats)))
    chunks = max(1, int(math.ceil(float(beats) / float(DEFAULT_NUM_BEATS))))
    return int(chunks * DEFAULT_NUM_BEATS)


def _resolve_guidance_scale(value: Any) -> float:
    try:
        guidance = float(value)
    except (TypeError, ValueError):
        guidance = float(DEFAULT_UI_GUIDANCE_SCALE)
    if not math.isfinite(guidance):
        guidance = float(DEFAULT_UI_GUIDANCE_SCALE)
    return float(max(MIN_UI_GUIDANCE_SCALE, min(MAX_UI_GUIDANCE_SCALE, guidance)))


def _vary_controls_for_chunk(
    controls: torch.Tensor,
    *,
    control_names: Sequence[str],
    chunk_idx: int,
    chunk_count: int,
    pattern_variation: float,
    seed: int,
) -> torch.Tensor:
    ctrl = torch.as_tensor(controls, dtype=torch.float32).view(-1).clone()
    names = tuple(str(name) for name in list(control_names or ()))
    variation = float(max(0.0, min(1.0, float(pattern_variation))))
    if variation <= 0.0 or int(chunk_count) <= 1:
        return ctrl.contiguous()
    rng = random.Random(int(seed) + (1009 * int(chunk_idx)))
    if not bool(is_legacy_sketch_control_names(names) or (not names and int(ctrl.numel()) == len(LEGACY_SKETCH_CONTROL_NAMES))):
        public = sketch_controls_to_public_dict(ctrl, control_names=names or SKETCH_CONTROL_NAMES)
        if "feel_amount" in public:
            public["feel_amount"] = float(max(0.0, min(1.0, float(public.get("feel_amount", 0.0)) + rng.uniform(-0.10, 0.14) * variation)))
            if rng.random() < 0.18 * variation:
                public["feel_style"] = rng.choice(list(FEEL_STYLE_VALUES))
            for key, span in (
                ("kick_ghost_density", 0.12),
                ("snare_ghost_density", 0.12),
                ("snare_roll_density", 0.14),
                ("hihat_density", 0.08),
                ("hihat_openness", 0.07),
                ("fill_density", 0.12),
            ):
                public[key] = float(max(0.0, min(1.0, float(public.get(key, 0.0)) + rng.uniform(-span, span) * variation)))
            if int(chunk_idx) == int(chunk_count) - 1 and float(public.get("fill_density", 0.0)) >= 0.20:
                public["fill_density"] = float(max(0.0, min(1.0, float(public.get("fill_density", 0.0)) + (0.10 * variation))))
                public["fill_start"] = float(max(0.0, min(1.0, float(public.get("fill_start", 0.70)) + rng.uniform(-0.10, 0.08) * variation)))
            if rng.random() < 0.25 * variation:
                public["tom_direction"] = rng.choice([value for value in TOM_DIRECTION_VALUES if value != "none"])
            if rng.random() < 0.25 * variation:
                public["fill_accent_shape"] = rng.choice(list(FILL_ACCENT_SHAPE_VALUES))
        elif "hihat_openness" in public:
            public["swing"] = float(max(-1.0, min(1.0, float(public.get("swing", 0.0)) + rng.uniform(-0.12, 0.12) * variation)))
            public["humanize"] = float(max(0.0, min(1.0, float(public.get("humanize", 0.0)) + rng.uniform(-0.08, 0.12) * variation)))
            base_fill_density = float(public.get("fill_density", 0.0))
            for key, span in (
                ("ghost_density", 0.08),
                ("hihat_density", 0.07),
                ("hihat_openness", 0.06),
                ("fill_density", 0.08),
            ):
                public[key] = float(max(0.0, min(1.0, float(public.get(key, 0.0)) + rng.uniform(-span, span) * variation)))
            if int(chunk_idx) == int(chunk_count) - 1 and float(base_fill_density) >= 0.20:
                public["fill_density"] = float(
                    max(0.0, min(1.0, float(public.get("fill_density", 0.0)) + (0.06 * variation)))
                )
        else:
            public["swing"] = float(max(-1.0, min(1.0, float(public.get("swing", 0.0)) + rng.uniform(-0.12, 0.12) * variation)))
            public["humanize"] = float(max(0.0, min(1.0, float(public.get("humanize", 0.0)) + rng.uniform(-0.08, 0.12) * variation)))
            if rng.random() < 0.12 * variation:
                public["hat_rate"] = rng.choice(list(HAT_RATE_VALUES))
            if rng.random() < 0.10 * variation:
                public["snare_style"] = rng.choice(list(SNARE_STYLE_VALUES))
            if int(chunk_idx) == int(chunk_count) - 1 and rng.random() < 0.30 * variation:
                fill_role = str(public.get("fill_role", "none"))
                public["fill_role"] = "ride_plus_fill" if fill_role == "ride" else "tom_crash_fill"
            for key in ("snare_ornament_intensity", "open_hat_intensity"):
                public[key] = float(max(0.0, min(1.0, float(public.get(key, 0.5)) + rng.uniform(-0.10, 0.10) * variation)))
        return control_tensor_from_public_controls(public, control_names=names or SKETCH_CONTROL_NAMES)
    ctrl[0] = float(max(-1.0, min(1.0, float(ctrl[0].item()) + rng.uniform(-0.12, 0.12) * variation)))
    ctrl[1] = float(max(0.0, min(1.0, float(ctrl[1].item()) + rng.uniform(-0.08, 0.12) * variation)))
    ctrl[2] = float(max(0.0, min(1.0, float(ctrl[2].item()) + rng.uniform(-0.20, 0.20) * variation)))
    ctrl[3] = float(max(0.0, min(1.0, float(ctrl[3].item()) + rng.uniform(-0.15, 0.15) * variation)))
    fill_bias = rng.uniform(-0.18, 0.22) * variation
    if int(chunk_idx) == int(chunk_count) - 1:
        fill_bias += 0.18 * variation
    ctrl[4] = float(max(0.0, min(1.0, float(ctrl[4].item()) + fill_bias)))
    return ctrl.contiguous()


@torch.no_grad()
def _decode_event_plan_variant(
    outputs: Mapping[str, torch.Tensor],
    *,
    sketch_hits: torch.Tensor,
    sketch_vel: torch.Tensor,
    controls: torch.Tensor,
    class_names: Sequence[str],
    class_id_vocab_sizes: Sequence[int],
    control_names: Sequence[str],
    budget_group_names: Sequence[str],
    budget_max_counts: Sequence[int],
    seed: int,
    chunk_idx: int,
    pattern_variation: float,
) -> list[dict[str, Any]]:
    hits_b = torch.as_tensor(sketch_hits, dtype=torch.float32).detach().cpu()
    vel_b = torch.as_tensor(sketch_vel, dtype=torch.float32).detach().cpu()
    ctrl_b = torch.as_tensor(controls, dtype=torch.float32).detach().cpu()
    if int(hits_b.dim()) == 2:
        hits_b = hits_b.unsqueeze(0)
    if int(vel_b.dim()) == 2:
        vel_b = vel_b.unsqueeze(0)
    if int(ctrl_b.dim()) == 1:
        ctrl_b = ctrl_b.unsqueeze(0)
    return decode_event_plan(
        outputs,
        sketch_hits=hits_b,
        sketch_vel=vel_b,
        controls=ctrl_b,
        class_names=class_names,
        class_id_vocab_sizes=class_id_vocab_sizes,
        control_names=control_names,
        budget_group_names=budget_group_names,
        budget_max_counts=budget_max_counts,
        seed=int(seed) + (7919 * int(chunk_idx)),
        pattern_variation=float(pattern_variation),
    )[0]


def _controls_tensor(
    feel_style: str,
    feel_amount: float,
    ghost_density: float,
    kick_ghost_density: float,
    snare_ghost_density: float,
    hihat_density: float,
    hihat_openness: float,
    fill_density: float,
    fill_shape: str,
    *,
    control_names: Sequence[str],
) -> torch.Tensor:
    fill_shape_s = str(fill_shape or "down")
    tom_direction = fill_shape_s if fill_shape_s in set(TOM_DIRECTION_VALUES) else "down"
    fill_accent_shape = fill_shape_s if fill_shape_s in set(FILL_ACCENT_SHAPE_VALUES) else "peak_end"
    feel_style_s = str(feel_style or "straight")
    swing = {"pushed": -0.35, "laid_back": 0.10, "swing": 0.35}.get(feel_style_s, 0.0) * float(feel_amount)
    # The main "Ghosts" knob is the master ghost intensity and drives BOTH snare
    # and kick ghosts. The dedicated advanced knobs are "Auto" (encoded as any
    # value < 0) by default and only take over their family when moved to >= 0.
    master_ghost = float(max(0.0, min(1.0, float(ghost_density))))
    snare_ghost_eff = (
        master_ghost
        if float(snare_ghost_density) < 0.0
        else float(max(0.0, min(1.0, float(snare_ghost_density))))
    )
    kick_ghost_eff = (
        float(max(0.0, min(1.0, (master_ghost - 0.30) / 0.70)))  # kicks stay sparser than snares
        if float(kick_ghost_density) < 0.0
        else float(max(0.0, min(1.0, float(kick_ghost_density))))
    )
    snare_roll_eff = float(max(0.0, min(1.0, (snare_ghost_eff - 0.45) / 0.55)))
    return control_tensor_from_public_controls(
        {
            "feel_style": str(feel_style),
            "feel_amount": float(feel_amount),
            "swing": float(swing),
            "humanize": float(feel_amount),
            "ghost_density": master_ghost,
            "kick_ghost_density": kick_ghost_eff,
            "snare_ghost_density": snare_ghost_eff,
            "snare_roll_density": snare_roll_eff,
            "hihat_density": float(hihat_density),
            "hihat_openness": float(hihat_openness),
            "fill_density": float(fill_density),
            "fill_role": "ride_plus_fill" if float(fill_density) >= 0.78 else ("tom_crash_fill" if float(fill_density) > 0.0 else "none"),
            "fill_start": 0.70 if float(fill_density) > 0.0 else 0.0,
            "fill_length": max(0.0, min(1.0, 0.20 + (0.55 * float(fill_density)))),
            "tom_direction": str(tom_direction),
            "fill_accent_shape": str(fill_accent_shape),
        },
        control_names=control_names,
    )


def _inject_crash_events(
    events: list[dict[str, Any]],
    *,
    crash_density: float,
    velocity: float,
    chunk_idx: int,
    chunk_count: int,
) -> list[dict[str, Any]]:
    """Add crash cymbals directly.

    The expander is trained on GMD, where crashes are rare, so it puts almost no
    probability on the crash family (max ~0.07) and never emits one. To give the
    user real control we inject crashes on the downbeat, where crashes musically
    land, with a count/velocity that scales with the ``crash_density`` knob.
    """
    amount = float(max(0.0, min(1.0, float(crash_density))))
    if amount <= 0.0:
        return events
    is_first = int(chunk_idx) == 0
    is_last = int(chunk_idx) == int(chunk_count) - 1
    steps: set[int] = set()
    if is_first or amount >= 0.75 or (is_last and amount >= 0.45):
        steps.add(0)  # crash on the phrase / bar downbeat
    if amount >= 0.90:
        steps.add(8)  # extra half-bar accent at the top of the range
    if not steps:
        return events
    crash_velocity = float(max(0.0, min(1.0, (0.60 + 0.35 * amount) * (0.70 + 0.30 * float(velocity)))))
    existing = {(str(e.get("family")), int(e.get("step", -1))) for e in events}
    for step in sorted(steps):
        if ("crash", int(step)) in existing:
            continue
        events.append(
            {
                "family": "crash",
                "step": int(step),
                "slot": 0,
                "probability": 1.0,
                "velocity": float(crash_velocity),
                "offset": 0.0,
                "class_id": 0,
                "forced": True,
            }
        )
    return events


def _set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _run_name(*, seed: int, audio: bool) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    millis = int((time.time() % 1.0) * 1000.0)
    suffix = "audio" if bool(audio) else "grid"
    return f"{timestamp}_{millis:03d}_seed{int(seed)}_{suffix}"


def _make_run_dir(out_dir: Path, *, seed: int, audio: bool) -> Path:
    base_name = _run_name(seed=int(seed), audio=bool(audio))
    for attempt in range(100):
        name = base_name if int(attempt) == 0 else f"{base_name}_{int(attempt):02d}"
        path = out_dir / name
        try:
            path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return path
    raise FileExistsError(f"could not create a unique run directory under {out_dir}")


def _is_managed_run_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    name = str(path.name)
    return "_seed" in name and ("_grid" in name or "_audio" in name)


def _prune_run_dirs(out_dir: Path, *, max_run_dirs: int, keep: Sequence[Path] = ()) -> None:
    limit = int(max_run_dirs)
    if limit <= 0 or not out_dir.is_dir():
        return
    keep_resolved = {_resolve_path(path) for path in keep}
    candidates: list[Path] = []
    for child in out_dir.iterdir():
        try:
            resolved = child.resolve()
        except OSError:
            continue
        if resolved in keep_resolved or not _is_managed_run_dir(resolved):
            continue
        candidates.append(resolved)
    overflow = int(len(candidates) + len(keep_resolved) - limit)
    if overflow <= 0:
        return
    candidates.sort(key=lambda path: int(path.stat().st_mtime_ns))
    for path in candidates[:overflow]:
        shutil.rmtree(path, ignore_errors=True)


def _release_cuda_memory() -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except RuntimeError:
        pass


def _plot_sketch_and_events(
    *,
    sketch_hits: torch.Tensor,
    sketch_vel: torch.Tensor,
    events: Sequence[Mapping[str, Any]] | Sequence[Sequence[Mapping[str, Any]]],
) -> Figure:
    hits = torch.as_tensor(sketch_hits, dtype=torch.float32).detach().cpu().numpy()
    velocities = torch.as_tensor(sketch_vel, dtype=torch.float32).detach().cpu().numpy()
    raw_events = list(events)
    if raw_events and isinstance(raw_events[0], Mapping):
        event_batches = [list(raw_events)]  # type: ignore[list-item]
    else:
        event_batches = [list(batch) for batch in raw_events]  # type: ignore[arg-type]
    if not event_batches:
        event_batches = [[]]
    chunk_count = int(len(event_batches))
    total_steps = int(16 * chunk_count)
    tiled_hits = np.tile(hits, (1, int(chunk_count)))
    tiled_velocities = np.tile(velocities, (1, int(chunk_count)))
    event_matrix = np.zeros((len(FAMILY_STATE_FAMILY_NAMES), int(total_steps)), dtype=np.float32)
    forced_matrix = np.zeros_like(event_matrix)
    family_index = {name: idx for idx, name in enumerate(FAMILY_STATE_FAMILY_NAMES)}
    scatter_x: list[float] = []
    scatter_y: list[int] = []
    scatter_size: list[float] = []
    scatter_forced: list[bool] = []
    for chunk_idx, chunk_events in enumerate(event_batches):
        for event in chunk_events:
            family = str(event.get("family", ""))
            if family not in family_index:
                continue
            step = int(event.get("step", 0))
            if not 0 <= step < 16:
                continue
            family_idx = int(family_index[family])
            offset = float(max(-0.5, min(0.5, float(event.get("offset", 0.0)))))
            global_step = int(chunk_idx * 16 + step)
            velocity = float(max(0.0, min(1.0, float(event.get("velocity", 0.0)))))
            event_matrix[family_idx, global_step] = max(float(event_matrix[family_idx, global_step]), velocity)
            if bool(event.get("forced", False)):
                forced_matrix[family_idx, global_step] = 1.0
            scatter_x.append(float(global_step) + float(offset))
            scatter_y.append(int(family_idx))
            scatter_size.append(float(28.0 + (72.0 * velocity)))
            scatter_forced.append(bool(event.get("forced", False)))

    fig = Figure(figsize=(13, 7.5), constrained_layout=True)
    axes = fig.subplots(3, 1)
    images = [
        axes[0].imshow(tiled_hits, aspect="auto", interpolation="nearest", cmap="Greys", vmin=0.0, vmax=1.0),
        axes[1].imshow(tiled_velocities, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0.0, vmax=1.0),
        axes[2].imshow(event_matrix, aspect="auto", interpolation="nearest", cmap="magma", vmin=0.0, vmax=1.0),
    ]
    titles = ["Input hits", "Input velocities", "Expanded events"]
    ylabels = [SKETCH_FAMILY_NAMES, SKETCH_FAMILY_NAMES, FAMILY_STATE_FAMILY_NAMES]
    tick_step = 1 if int(total_steps) <= 32 else 4
    xticks = np.arange(0, int(total_steps), int(tick_step))
    xtick_labels = [str((int(idx) % 16) + 1) for idx in xticks]
    for ax, title, labels in zip(axes, titles, ylabels, strict=True):
        ax.set_title(title)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("16th step by chunk")
        for boundary in range(16, int(total_steps), 16):
            ax.axvline(float(boundary) - 0.5, color="black", linewidth=0.8, alpha=0.45)
    if scatter_x:
        colors = ["cyan" if forced else "white" for forced in scatter_forced]
        axes[2].scatter(
            scatter_x,
            scatter_y,
            s=scatter_size,
            c=colors,
            edgecolors="black",
            linewidths=0.6,
            alpha=0.88,
        )
    for family_idx, step_idx in np.argwhere(forced_matrix > 0.0):
        axes[2].text(
            int(step_idx),
            int(family_idx),
            "F",
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            fontweight="bold",
        )
    for ax, image in zip(axes, images, strict=True):
        fig.colorbar(image, ax=ax, fraction=0.018, pad=0.01)
    return fig


def _plot_rendered_grid(batch: Mapping[str, Any], *, chunk_crossfade_ms: float = 0.0) -> Figure:
    grid_bft = torch.as_tensor(batch["grid"], dtype=torch.float32).detach().cpu()
    valid_mask_bt = torch.as_tensor(batch["grid_valid_mask"], dtype=torch.bool).detach().cpu()
    chunks: list[torch.Tensor] = []
    boundaries: list[int] = []
    cursor = 0
    for batch_idx in range(int(grid_bft.shape[0])):
        valid_len = int(valid_mask_bt[int(batch_idx)].sum().item())
        valid_len = max(1, min(valid_len, int(grid_bft.shape[-1])))
        if batch_idx > 0:
            gap = torch.zeros((int(grid_bft.shape[1]), 4), dtype=torch.float32)
            chunks.append(gap)
            cursor += int(gap.shape[-1])
        chunks.append(grid_bft[int(batch_idx), :, :valid_len])
        cursor += int(valid_len)
        boundaries.append(int(cursor))
    grid = torch.cat(chunks, dim=-1) if chunks else grid_bft[0]
    fig = Figure(figsize=(14, 6), constrained_layout=True)
    ax = fig.subplots(1, 1)
    image = ax.imshow(
        grid.numpy(),
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )
    labels = list(batch.get("feature_row_names") or FAMILY_STATE_FEATURE_ROW_NAMES)
    ax.set_title(f"Rendered diffusion conditioning grid ({int(grid_bft.shape[0])} chunk(s))")
    ax.set_xlabel("chunked grid frame")
    ax.set_ylabel("feature row")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    for boundary in boundaries[:-1]:
        ax.axvline(float(boundary), color="black", linewidth=0.8, alpha=0.55)
        frame_rate_raw = batch.get("grid_frame_rate_b")
        frame_rate = (
            float(torch.as_tensor(frame_rate_raw, dtype=torch.float32).view(-1)[0].item())
            if frame_rate_raw is not None
            else float(DEFAULT_GRID_FRAME_RATE)
        )
        crossfade_frames = float(max(0.0, float(chunk_crossfade_ms))) * float(frame_rate) / 1000.0
        if float(crossfade_frames) > 0.0:
            half_width = max(0.5, 0.5 * float(crossfade_frames))
            ax.axvspan(
                float(boundary) - float(half_width),
                float(boundary) + float(half_width),
                color="black",
                alpha=0.08,
                linewidth=0.0,
            )
    fig.colorbar(image, ax=ax, fraction=0.02, pad=0.01)
    return fig


@dataclass
class DiffusionResources:
    checkpoint_path: Path
    model: Any
    diffusion: Any
    target_mean: torch.Tensor
    target_std: torch.Tensor
    payload: dict[str, Any]
    target_token_rate_hz: float
    target_layout: str
    target_pca_basis: Mapping[str, Any] | None
    audio_codec_model: Any
    codec_metadata: Any
    sample_rate: int


@dataclass
class DirectResources:
    """Resources for the deterministic direct-PCA regressor branch.

    Mirrors the subset of DiffusionResources that the shared decode tail in
    ``_generate_audio`` reads (target stats, PCA basis, codec, sample rate),
    minus the diffusion sampler. Inference is a single forward pass.
    """

    checkpoint_path: Path
    model: Any
    target_mean: torch.Tensor
    target_std: torch.Tensor
    target_token_rate_hz: float
    target_pca_basis: Mapping[str, Any] | None
    audio_codec_model: Any
    codec_metadata: Any
    sample_rate: int


def _is_direct_checkpoint(path: Path) -> bool:
    return Path(path).name == "best_direct.pt"


class SketchDiffusionListenApp:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.sketch_checkpoint = _require_file(args.sketch_checkpoint, label="sketch checkpoint")
        self.cache_root = _require_dir(args.cache_root, label="cache root")
        self.out_dir = _resolve_path(args.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_run_dirs = int(args.max_run_dirs)
        _prune_run_dirs(self.out_dir, max_run_dirs=self.max_run_dirs)
        self.resolved_device = resolve_device(str(args.device))
        self.device = torch.device(self.resolved_device)
        if self.device.type == "cuda" and self.device.index is not None:
            torch.cuda.set_device(self.device)
        self.sketch_model = _load_sketch_expander(self.sketch_checkpoint, device=self.device)
        self.diffusion_resources: DiffusionResources | DirectResources | None = None
        self._closed = False
        self.diffusion_checkpoint_choices = _discover_diffusion_checkpoints(
            explicit=str(args.diffusion_checkpoint),
            train_dir=args.diffusion_train_dir,
        )
        self.default_diffusion_checkpoint = self._default_diffusion_checkpoint_value()

    def _default_diffusion_checkpoint_value(self) -> str:
        if str(self.args.diffusion_checkpoint).strip():
            return _display_path(_require_file(self.args.diffusion_checkpoint, label="diffusion checkpoint"))
        _require_dir(self.args.diffusion_train_dir, label="diffusion train dir")
        return _display_path(_resolve_diffusion_checkpoint(self.args.diffusion_checkpoint, self.args.diffusion_train_dir))

    def refresh_diffusion_checkpoint_choices(self) -> list[str]:
        self.diffusion_checkpoint_choices = _discover_diffusion_checkpoints(
            explicit=str(self.args.diffusion_checkpoint),
            train_dir=self.args.diffusion_train_dir,
        )
        if self.default_diffusion_checkpoint not in self.diffusion_checkpoint_choices:
            self.diffusion_checkpoint_choices.insert(0, self.default_diffusion_checkpoint)
        return list(self.diffusion_checkpoint_choices)

    def _resolve_selected_diffusion_checkpoint(self, selection: Any) -> Path:
        text = str(selection or "").strip()
        if not text:
            return _resolve_diffusion_checkpoint(self.args.diffusion_checkpoint, self.args.diffusion_train_dir)
        return _require_file(_resolve_display_or_path(text), label="diffusion checkpoint")

    def _load_diffusion_resources(self, checkpoint_path: Path) -> DiffusionResources:
        model, diffusion, target_mean, target_std, payload = _load_diffusion_state(checkpoint_path, device=self.device)
        codec_metadata = resolve_codec_metadata_from_payload(
            payload,
            fallback=resolve_codec_metadata_from_cache_config(self.cache_root),
        )
        target_layout = str(
            payload.get("target_layout") or resolve_target_layout_from_cache_config(self.cache_root)
        ).strip().lower()
        target_pca_basis = None
        if payload.get("target_pca_basis") is not None:
            target_pca_basis = load_target_pca_basis(payload["target_pca_basis"], device=self.device)
        else:
            basis_path = resolve_target_pca_basis_path_from_cache_config(self.cache_root)
            if basis_path is not None:
                target_pca_basis = load_target_pca_basis(basis_path, device=self.device)
        if target_layout == "framewise_pca" and target_pca_basis is None:
            raise FileNotFoundError("framewise_pca diffusion checkpoint requires a PCA basis")
        audio_codec_model, _codec_device, codec_metadata = load_audio_codec_model(
            device=self.resolved_device,
            metadata=codec_metadata,
        )
        target_token_rate_hz = (
            float(self.args.target_token_rate_hz)
            if float(self.args.target_token_rate_hz) > 0.0
            else resolve_target_token_rate_hz(codec_metadata)
        )
        sample_rate = int(payload.get("sample_rate") or codec_metadata.codec_sample_rate)
        return DiffusionResources(
            checkpoint_path=checkpoint_path,
            model=model,
            diffusion=diffusion,
            target_mean=target_mean,
            target_std=target_std,
            payload=payload,
            target_token_rate_hz=float(target_token_rate_hz),
            target_layout=target_layout,
            target_pca_basis=target_pca_basis,
            audio_codec_model=audio_codec_model,
            codec_metadata=codec_metadata,
            sample_rate=int(sample_rate),
        )

    def _load_direct_resources(self, checkpoint_path: Path) -> DirectResources:
        payload = torch.load(str(checkpoint_path), map_location=self.device, weights_only=False)
        cfg = DirectRegressorConfig(**dict(payload["config"]))
        model = DirectPCASequenceRegressor(cfg).to(self.device).eval()
        model.load_state_dict(dict(payload["model_state_dict"]))
        target_mean = torch.as_tensor(payload["target_mean"], dtype=torch.float32, device=self.device).view(-1)
        target_std = (
            torch.as_tensor(payload["target_std"], dtype=torch.float32, device=self.device).view(-1).clamp_min(1.0e-6)
        )
        codec_metadata = resolve_codec_metadata_from_cache_config(self.cache_root)
        basis_path = resolve_target_pca_basis_path_from_cache_config(self.cache_root)
        target_pca_basis = (
            load_target_pca_basis(basis_path, device=self.device) if basis_path is not None else None
        )
        audio_codec_model, _codec_device, codec_metadata = load_audio_codec_model(
            device=self.resolved_device,
            metadata=codec_metadata,
        )
        target_token_rate_hz = (
            float(self.args.target_token_rate_hz)
            if float(self.args.target_token_rate_hz) > 0.0
            else resolve_target_token_rate_hz(codec_metadata)
        )
        return DirectResources(
            checkpoint_path=checkpoint_path,
            model=model,
            target_mean=target_mean,
            target_std=target_std,
            target_token_rate_hz=float(target_token_rate_hz),
            target_pca_basis=target_pca_basis,
            audio_codec_model=audio_codec_model,
            codec_metadata=codec_metadata,
            sample_rate=int(codec_metadata.codec_sample_rate),
        )

    def close(self) -> None:
        if bool(self._closed):
            return
        self._closed = True
        self.diffusion_resources = None
        self.sketch_model = None
        _release_cuda_memory()

    def _get_diffusion_resources(self, selection: Any) -> DiffusionResources:
        if bool(self.args.grid_only):
            raise RuntimeError("audio generation is disabled because the UI was launched with --grid-only")
        checkpoint_path = self._resolve_selected_diffusion_checkpoint(selection)
        if self.diffusion_resources is not None and self.diffusion_resources.checkpoint_path == checkpoint_path:
            return self.diffusion_resources
        self.diffusion_resources = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        if _is_direct_checkpoint(checkpoint_path):
            self.diffusion_resources = self._load_direct_resources(checkpoint_path)
        else:
            self.diffusion_resources = self._load_diffusion_resources(checkpoint_path)
        return self.diffusion_resources

    def _generate_audio(
        self,
        batch: Mapping[str, Any],
        *,
        run_dir: Path,
        resources: "DiffusionResources | DirectResources",
        guidance_scale: float,
        sample_seed: int,
    ) -> tuple[Path, list[Path], dict[str, Any]]:
        prepared = _prepare_batch_tensors(batch, self.device, require_target=False, require_timing=False)
        geometry = resolve_inference_geometry(
            prepared,
            use_bpm_inference_geometry=True,
            inference_num_beats=DEFAULT_NUM_BEATS,
            target_token_rate_hz=float(resources.target_token_rate_hz),
        )
        with torch.no_grad():
            if isinstance(resources, DirectResources):
                # Deterministic one-shot regression; guidance_scale/sample_seed do not apply.
                # Inputs mirror what sample_ddpm feeds model.encode_conditioning: grid fields
                # from the prepared device batch, timing/masks from the inference geometry.
                latent_norm = resources.model(
                    grid=prepared["grid"],
                    grid_ids=prepared.get("grid_ids"),
                    grid_times_sec=geometry["grid_times_sec"],
                    token_times_sec=geometry["token_times_sec"],
                    target_valid_mask_bt=geometry["target_valid_mask_bt"],
                    grid_valid_mask_bt=prepared["grid_valid_mask"],
                )
            else:
                latent_norm = sample_ddpm(
                    model=resources.model,
                    diffusion=resources.diffusion,
                    batch=batch,
                    device=self.device,
                    guidance_scale=float(guidance_scale),
                    sample_seed=int(sample_seed),
                    x0_clip_norm=float(self.args.x0_clip_norm) if self.args.x0_clip_norm is not None else None,
                    use_bpm_inference_geometry=True,
                    inference_num_beats=DEFAULT_NUM_BEATS,
                    target_token_rate_hz=float(resources.target_token_rate_hz),
                    inference_geometry=geometry,
                )
            latent = denormalize_latent(latent_norm, resources.target_mean, resources.target_std)
            latent = latent * geometry["target_valid_mask_bt"].unsqueeze(-1)
            decoded_audio = decode_latent_to_audio(
                latent,
                resources.audio_codec_model,
                target_pca_basis=resources.target_pca_basis,
            )
        if int(decoded_audio.dim()) == 2:
            decoded_audio = decoded_audio.unsqueeze(1)
        samples_per_frame = _samples_per_latent_frame(
            decoded_num_samples=int(decoded_audio.shape[-1]),
            latent_num_frames=int(geometry["target_valid_mask_bt"].shape[1]),
        )
        chunk_wavs: list[Path] = []
        audio_segments: list[torch.Tensor] = []
        target_frames_by_chunk: list[int] = []
        requested_duration_sec_by_chunk: list[float] = []
        decoded_duration_sec_by_chunk: list[float] = []
        duration_error_ms_by_chunk: list[float] = []
        for chunk_idx in range(int(decoded_audio.shape[0])):
            target_frames = int(geometry["target_num_frames_b"][int(chunk_idx)].item())
            target_frames_by_chunk.append(int(target_frames))
            requested_duration_sec = float(geometry["duration_sec"][int(chunk_idx)].item())
            requested_duration_sec_by_chunk.append(float(requested_duration_sec))
            audio_i = decoded_audio[int(chunk_idx), :, : int(target_frames * samples_per_frame)].detach()
            decoded_duration_sec = float(audio_i.shape[-1]) / float(resources.sample_rate)
            decoded_duration_sec_by_chunk.append(float(decoded_duration_sec))
            duration_error_ms_by_chunk.append(float((decoded_duration_sec - requested_duration_sec) * 1000.0))
            if float(self.args.beat_crossfade_ms) > 0.0:
                beat_mask = geometry["beat_boundaries_valid_mask"][int(chunk_idx)]
                beat_boundaries = geometry["beat_boundaries_sec"][int(chunk_idx)][beat_mask]
                audio_i = apply_beat_crossfade(
                    audio_i,
                    beat_boundaries,
                    sample_rate=int(resources.sample_rate),
                    beat_crossfade_ms=float(self.args.beat_crossfade_ms),
                )
            chunk_path = run_dir / f"chunk_{int(chunk_idx):03d}.wav"
            save_audio(chunk_path, audio_i.unsqueeze(0).cpu(), sample_rate=int(resources.sample_rate))
            chunk_wavs.append(chunk_path)
            audio_segments.append(audio_i.detach().cpu())
        crossfade_samples = int(round(float(resources.sample_rate) * max(0.0, float(self.args.chunk_crossfade_ms)) / 1000.0))
        stitched_audio = stitch_audio_segments_with_crossfade(
            audio_segments,
            crossfade_num_samples=int(crossfade_samples),
        )
        wav_path = run_dir / "output.wav"
        save_audio(wav_path, stitched_audio.unsqueeze(0).cpu(), sample_rate=int(resources.sample_rate))
        return wav_path, chunk_wavs, {
            "diffusion_checkpoint": str(resources.checkpoint_path),
            "decode_mode": "direct_latent",
            "target_num_frames_by_chunk": target_frames_by_chunk,
            "target_token_rate_hz": float(resources.target_token_rate_hz),
            "effective_token_rate_hz": float(resources.target_token_rate_hz),
            "codec_hop_length": resolve_codec_hop_length(resources.codec_metadata),
            "requested_duration_sec_by_chunk": requested_duration_sec_by_chunk,
            "decoded_duration_sec_by_chunk": decoded_duration_sec_by_chunk,
            "duration_error_ms_by_chunk": duration_error_ms_by_chunk,
            "requested_duration_sec": float(sum(requested_duration_sec_by_chunk)),
            "decoded_duration_sec": float(sum(decoded_duration_sec_by_chunk)),
            "duration_error_ms": float((sum(decoded_duration_sec_by_chunk) - sum(requested_duration_sec_by_chunk)) * 1000.0),
            "stitched_duration_sec": float(stitched_audio.shape[-1]) / float(resources.sample_rate),
            "sample_rate": int(resources.sample_rate),
            "beat_crossfade_ms": float(self.args.beat_crossfade_ms),
            "chunk_crossfade_ms": float(self.args.chunk_crossfade_ms),
            "guidance_scale": float(guidance_scale),
            "sample_seed": int(sample_seed),
            "wav": wav_path.name,
            "chunk_wavs": [path.name for path in chunk_wavs],
        }

    def run_once(
        self,
        hits_table: Any,
        velocity: float,
        velocity_variation: float,
        output_beats: float,
        guidance_scale: float,
        pattern_variation: float,
        bpm: float,
        feel_style: str,
        feel_amount: float,
        ghost_density: float,
        kick_ghost_density: float,
        snare_ghost_density: float,
        hihat_density: float,
        hihat_openness: float,
        fill_density: float,
        fill_shape: str,
        crash_density: float,
        seed: int,
        diffusion_checkpoint: Any,
        *,
        make_audio: bool,
    ) -> tuple[str | None, list[str], dict[str, Any], Figure, Figure]:
        bpm_value = float(bpm)
        if not math.isfinite(bpm_value) or bpm_value <= 0.0:
            raise gr.Error(f"bpm must be positive, got {bpm}")
        seed_value = int(seed)
        _set_seed(seed_value)
        output_beats_value = _resolve_output_beats(output_beats)
        guidance_value = _resolve_guidance_scale(guidance_scale)
        chunk_count = int(output_beats_value // DEFAULT_NUM_BEATS)
        sketch_hits = _hits_table_to_tensor(hits_table)
        sketch_vel = _derive_velocity_matrix(
            sketch_hits,
            velocity=float(velocity),
            variation=float(velocity_variation),
            seed=seed_value,
        )
        controls = _controls_tensor(
            str(feel_style),
            float(feel_amount),
            float(ghost_density),
            float(kick_ghost_density),
            float(snare_ghost_density),
            float(hihat_density),
            float(hihat_openness),
            float(fill_density),
            str(fill_shape),
            control_names=self.sketch_model.cfg.control_names,
        )
        event_batches: list[list[dict[str, Any]]] = []
        chunk_controls: list[torch.Tensor] = []
        for chunk_idx in range(int(chunk_count)):
            controls_i = _vary_controls_for_chunk(
                controls,
                control_names=self.sketch_model.cfg.control_names,
                chunk_idx=int(chunk_idx),
                chunk_count=int(chunk_count),
                pattern_variation=float(pattern_variation),
                seed=seed_value,
            )
            chunk_controls.append(controls_i)
            with torch.no_grad():
                outputs = self.sketch_model(
                    sketch_hits.unsqueeze(0).to(device=self.device),
                    sketch_vel.unsqueeze(0).to(device=self.device),
                    controls_i.unsqueeze(0).to(device=self.device),
                )
            chunk_events = _decode_event_plan_variant(
                {key: value.detach().cpu() for key, value in outputs.items()},
                sketch_hits=sketch_hits,
                sketch_vel=sketch_vel,
                controls=controls_i,
                class_names=self.sketch_model.class_names,
                class_id_vocab_sizes=self.sketch_model.class_id_vocab_sizes,
                control_names=self.sketch_model.cfg.control_names,
                budget_group_names=self.sketch_model.budget_group_names,
                budget_max_counts=self.sketch_model.budget_max_counts,
                seed=seed_value,
                chunk_idx=int(chunk_idx),
                pattern_variation=float(pattern_variation),
            )
            chunk_events = _inject_crash_events(
                chunk_events,
                crash_density=float(crash_density),
                velocity=float(velocity),
                chunk_idx=int(chunk_idx),
                chunk_count=int(chunk_count),
            )
            event_batches.append(chunk_events)
        resources = self._get_diffusion_resources(diffusion_checkpoint) if bool(make_audio) else None
        target_token_rate_hz = (
            float(resources.target_token_rate_hz)
            if resources is not None
            else (float(self.args.target_token_rate_hz) if float(self.args.target_token_rate_hz) > 0.0 else None)
        )
        batch = build_diffusion_batch_from_events(
            event_batches,
            bpm=[float(bpm_value)] * int(chunk_count),
            num_beats=DEFAULT_NUM_BEATS,
            grid_frame_rate=float(self.args.grid_frame_rate),
            target_token_rate_hz=target_token_rate_hz,
        )
        run_dir = _make_run_dir(self.out_dir, seed=seed_value, audio=bool(make_audio))
        torch.save(batch, run_dir / "grid_batch.pt")
        chunk_payload = [
            {
                "chunk_index": int(chunk_idx),
                "controls": sketch_controls_to_public_dict(
                    chunk_controls[int(chunk_idx)],
                    control_names=self.sketch_model.cfg.control_names,
                ),
                "encoded_controls": {
                    name: float(chunk_controls[int(chunk_idx)][idx].item())
                    for idx, name in enumerate(self.sketch_model.cfg.control_names)
                },
                "events": event_batches[int(chunk_idx)],
            }
            for chunk_idx in range(int(chunk_count))
        ]
        (run_dir / "events.json").write_text(json.dumps(_as_jsonable(chunk_payload), indent=2, sort_keys=True) + "\n")
        sketch_payload = {
            "hits": sketch_hits,
            "velocities": sketch_vel,
            "bpm": float(bpm_value),
            "output_beats": int(output_beats_value),
            "chunk_count": int(chunk_count),
            "controls": sketch_controls_to_public_dict(
                controls,
                control_names=self.sketch_model.cfg.control_names,
            ),
            "encoded_controls": {
                name: float(controls[idx].item())
                for idx, name in enumerate(self.sketch_model.cfg.control_names)
            },
        }
        write_json(run_dir / "sketch.json", sketch_payload)
        sketch_fig = _plot_sketch_and_events(sketch_hits=sketch_hits, sketch_vel=sketch_vel, events=event_batches)
        grid_fig = _plot_rendered_grid(batch, chunk_crossfade_ms=float(self.args.chunk_crossfade_ms))
        sketch_fig.savefig(run_dir / "sketch_events.png", dpi=150)
        grid_fig.savefig(run_dir / "conditioning_grid.png", dpi=150)
        audio_path: str | None = None
        chunk_audio_paths: list[str] = []
        audio_summary: dict[str, Any] = {}
        if bool(make_audio):
            if resources is None:
                raise RuntimeError("diffusion resources were not loaded")
            wav_path, chunk_wavs, audio_summary = self._generate_audio(
                batch,
                run_dir=run_dir,
                resources=resources,
                guidance_scale=float(guidance_value),
                sample_seed=int(seed_value),
            )
            audio_path = str(wav_path)
            chunk_audio_paths = [str(path) for path in chunk_wavs]
        summary = {
            "run_dir": str(run_dir),
            "sketch_checkpoint": str(self.sketch_checkpoint),
            "cache_root": str(self.cache_root),
            "resolved_device": str(self.resolved_device),
            "grid_only": not bool(make_audio),
            "selected_diffusion_checkpoint": str(diffusion_checkpoint or ""),
            "bpm": float(bpm_value),
            "output_beats": int(output_beats_value),
            "chunk_count": int(chunk_count),
            "chunk_num_beats": int(DEFAULT_NUM_BEATS),
            "pattern_variation": float(pattern_variation),
            "velocity_profile": {
                "velocity": float(velocity),
                "variation": float(velocity_variation),
            },
            "controls": sketch_controls_to_public_dict(
                controls,
                control_names=self.sketch_model.cfg.control_names,
            ),
            "encoded_controls": {
                name: float(controls[idx].item())
                for idx, name in enumerate(self.sketch_model.cfg.control_names)
            },
            "guidance_scale": float(guidance_value),
            "decode_mode": "direct_latent",
            "seed": int(seed_value),
            "sample_seed": int(seed_value),
            "num_events_by_chunk": [int(len(events)) for events in event_batches],
            "grid_num_frames_by_chunk": [int(x.item()) for x in batch["grid_num_frames_b"]],
            "grid_batch": "grid_batch.pt",
            "events": "events.json",
            "sketch_plot": "sketch_events.png",
            "grid_plot": "conditioning_grid.png",
            **audio_summary,
        }
        write_json(run_dir / "summary.json", summary)
        _prune_run_dirs(self.out_dir, max_run_dirs=self.max_run_dirs, keep=[run_dir])
        return audio_path, chunk_audio_paths, {"summary": _as_jsonable(summary), "chunks": _as_jsonable(chunk_payload)}, sketch_fig, grid_fig


def _handle_gradio_error(exc: Exception) -> gr.Error:
    if isinstance(exc, gr.Error):
        return exc
    return gr.Error(str(exc))


def _friendly_ckpt_label(path_str: Any) -> str:
    s = str(path_str)
    if "best_direct.pt" in s or "runs_direct" in s:
        return "Direct regressor (76.5M, deterministic)"
    m = re.search(r"dac_(\d+)steps", s)
    steps = m.group(1) if m else "?"
    family = "RVQ-CE diffusion" if "_ce" in s else "Plain diffusion"
    return f"{family} - {steps} steps"


def _labeled_ckpt_choices(paths: Sequence[str]) -> list[tuple[str, str]]:
    def _key(p: str) -> tuple[int, int]:
        s = str(p)
        if "best_direct.pt" in s or "runs_direct" in s:
            return (2, 0)  # sort the deterministic regressor after all diffusion rows
        m = re.search(r"dac_(\d+)steps", s)
        return (1 if "_ce" in s else 0, int(m.group(1)) if m else 0)
    return [(_friendly_ckpt_label(p), p) for p in sorted(paths, key=_key)]


# Hide Gradio's per-cell three-dot menu button on the Dataframe; the boolean
# checkbox cells stay clickable, but the overlapping menu trigger is removed.
_LISTENER_CSS = ".cell-menu-button{display:none !important;}"


def launch_private(demo: gr.Blocks, **kwargs: Any) -> Any:
    """Launch Gradio with privacy options, tolerating older launch signatures."""
    import inspect

    try:
        accepted = set(inspect.signature(demo.launch).parameters)
    except Exception:
        accepted = set(kwargs)
    launch_kwargs = {key: value for key, value in kwargs.items() if key in accepted}
    return demo.launch(**launch_kwargs)


def build_ui(app: SketchDiffusionListenApp) -> gr.Blocks:
    with gr.Blocks(title="Anonymous Drum Rendering Demo", css=_LISTENER_CSS) as demo:
        gr.Markdown(
            "## 🥁 Grid2Drum-DAC Listener\n"
            "Toggle steps in the **grid** to sketch a one-bar drum pattern, shape the groove with "
            "the main controls on the right, then **Render Grid** to preview what you drew or "
            "**Generate Audio** to hear it. The plot right under the grid shows your pattern and how "
            "it expands into events. Finer controls — guidance, feel, per-instrument ghost/hat/crash "
            "detail, seed, and the model's internal conditioning grid — live under **Advanced settings**."
        )
        with gr.Row():
            # Left: the creative canvas — the grid, with a plot of what it means right below it.
            with gr.Column(scale=3):
                hits_df = gr.Dataframe(
                    value=_default_hits_table(),
                    headers=HIT_HEADERS,
                    datatype=["str", *["bool"] * 16],
                    row_count=(3, "fixed"),
                    col_count=(17, "fixed"),
                    type="array",
                    label="Hits",
                    interactive=True,
                    show_row_numbers=False,
                    static_columns=[0],
                    column_widths=[90, *[44] * 16],
                )
                sketch_plot = gr.Plot(label="Your pattern — hits, velocities and expanded events")
            # Right: the main knobs a listener reaches for most, two per row to keep
            # the column compact.
            with gr.Column(scale=2):
                with gr.Row():
                    velocity = gr.Slider(
                        0.2,
                        1.0,
                        value=0.86,
                        step=0.01,
                        label="Velocity",
                        info="0.2 soft, 0.6 medium, 1.0 hard overall hit strength.",
                    )
                    velocity_variation = gr.Slider(
                        0.0,
                        1.0,
                        value=0.20,
                        step=0.01,
                        label="Velocity variation",
                        info="0 even, 0.2 natural, 0.6+ uneven repeated-hit loudness.",
                    )
                with gr.Row():
                    output_beats = gr.Slider(
                        DEFAULT_NUM_BEATS,
                        32,
                        value=max(DEFAULT_NUM_BEATS, _resolve_output_beats(app.args.num_beats)),
                        step=DEFAULT_NUM_BEATS,
                        label="Output beats",
                        info="4 is one chunk; 8-16 repeats; 32 is longest.",
                    )
                    bpm = gr.Slider(
                        60.0,
                        190.0,
                        value=120.0,
                        step=1.0,
                        label="BPM",
                        info="60 slow, 120 mid, 190 fast; also sets length.",
                    )
                with gr.Row():
                    pattern_variation = gr.Slider(
                        0.0,
                        1.0,
                        value=0.20,
                        step=0.01,
                        label="Pattern variation",
                        info="0 same each chunk, 0.2 subtle, 0.7+ new fills/placements.",
                    )
                    ghost_density = gr.Slider(
                        0.0,
                        1.0,
                        value=0.35,
                        step=0.01,
                        label="Ghosts",
                        info="Soft in-between hits: 0 none, 0.3 light snares, 0.8+ adds kick ghosts too.",
                    )
                with gr.Row():
                    hihat_openness = gr.Slider(
                        0.0,
                        1.0,
                        value=0.0,
                        step=0.01,
                        label="Hat openness",
                        info="0 closed, 0.3 accents, 0.7+ open/washy hats.",
                    )
                    fill_density = gr.Slider(
                        0.0,
                        1.0,
                        value=0.0,
                        step=0.01,
                        label="Fill amount",
                        info="0 none, 0.4 pickups, 0.7+ tom runs/ride, 1 heavy fills.",
                    )
                diffusion_checkpoint = gr.Dropdown(
                    choices=_labeled_ckpt_choices(app.diffusion_checkpoint_choices),
                    value=app.default_diffusion_checkpoint,
                    allow_custom_value=True,
                    label="Diffusion checkpoint",
                    interactive=True,
                )
        with gr.Row():
            preview_btn = gr.Button("Render Grid", variant="secondary")
            audio_btn = gr.Button("Generate Audio", variant="primary", interactive=not bool(app.args.grid_only))
        with gr.Row():
            audio_out = gr.Audio(label="Audio", type="filepath")
            chunk_files = gr.Files(label="Chunk WAVs")
            json_out = gr.JSON(label="Events")
        with gr.Accordion("Advanced settings", open=False):
            with gr.Row():
                guidance_scale = gr.Slider(
                    MIN_UI_GUIDANCE_SCALE,
                    MAX_UI_GUIDANCE_SCALE,
                    value=_resolve_guidance_scale(DEFAULT_ADVANCED_GUIDANCE_SCALE),
                    step=0.05,
                    label="Guidance",
                    info="1 matches training; 2-3 follows conditioning harder; 4-5 may add artifacts.",
                )
                seed = gr.Number(value=1234, precision=0, label="Seed")
            with gr.Row():
                feel_style = gr.Dropdown(
                    choices=list(FEEL_STYLE_VALUES),
                    value="straight",
                    label="Feel style",
                    interactive=True,
                )
                feel_amount = gr.Slider(
                    0.0,
                    1.0,
                    value=0.25,
                    step=0.01,
                    label="Feel amount",
                    info="0 tight, 0.2-0.35 natural, 0.6+ loose timing and velocity.",
                )
            with gr.Row():
                kick_ghost_density = gr.Slider(
                    -1.0,
                    1.0,
                    value=-1.0,
                    step=0.01,
                    label="Kick ghosts (−1 = auto)",
                    info="Auto (−1) follows the main Ghosts knob; set 0-1 to override kick ghosts only.",
                )
                snare_ghost_density = gr.Slider(
                    -1.0,
                    1.0,
                    value=-1.0,
                    step=0.01,
                    label="Snare ghosts (−1 = auto)",
                    info="Auto (−1) follows the main Ghosts knob; set 0-1 to override snare ghosts only.",
                )
            with gr.Row():
                hihat_density = gr.Slider(
                    0.0,
                    1.0,
                    value=0.52,
                    step=0.01,
                    label="Hats",
                    info="0 sparse, 0.5 steady 8ths/16ths, 0.8+ dense hat subdivisions.",
                )
                crash_density = gr.Slider(
                    0.0,
                    1.0,
                    value=0.0,
                    step=0.01,
                    label="Crashes",
                    info="0 none; crashes land on the downbeat, more and louder as you raise it.",
                )
            fill_shape = gr.Dropdown(
                choices=["down", "up", "mixed", "ramp_up", "ramp_down", "peak_end"],
                value="down",
                label="Fill shape",
                interactive=True,
            )
            refresh_checkpoints_btn = gr.Button("Refresh Checkpoints", variant="secondary")
            gr.Markdown(
                "**Conditioning grid** — the seconds-aligned control field the diffusion model "
                "actually sees, rendered from your sketch and the controls above."
            )
            grid_plot = gr.Plot(label="Conditioning Grid")

        inputs = [
            hits_df,
            velocity,
            velocity_variation,
            output_beats,
            guidance_scale,
            pattern_variation,
            bpm,
            feel_style,
            feel_amount,
            ghost_density,
            kick_ghost_density,
            snare_ghost_density,
            hihat_density,
            hihat_openness,
            fill_density,
            fill_shape,
            crash_density,
            seed,
            diffusion_checkpoint,
        ]
        outputs = [audio_out, chunk_files, json_out, sketch_plot, grid_plot]

        def _refresh_checkpoints(current_value: Any) -> gr.Dropdown:
            choices = app.refresh_diffusion_checkpoint_choices()
            value = str(current_value or "").strip() or app.default_diffusion_checkpoint
            if value not in choices:
                choices = [value, *choices]
            return gr.update(choices=_labeled_ckpt_choices(choices), value=value)

        def _preview(*values: Any) -> tuple[str | None, list[str], dict[str, Any], Figure, Figure]:
            try:
                return app.run_once(*values, make_audio=False)
            except Exception as exc:
                raise _handle_gradio_error(exc) from exc

        def _generate(*values: Any) -> tuple[str | None, list[str], dict[str, Any], Figure, Figure]:
            try:
                return app.run_once(*values, make_audio=True)
            except Exception as exc:
                raise _handle_gradio_error(exc) from exc

        refresh_checkpoints_btn.click(
            fn=_refresh_checkpoints,
            inputs=[diffusion_checkpoint],
            outputs=[diffusion_checkpoint],
        )
        preview_btn.click(fn=_preview, inputs=inputs, outputs=outputs)
        audio_btn.click(fn=_generate, inputs=inputs, outputs=outputs)
    return demo


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    app = SketchDiffusionListenApp(args)
    atexit.register(app.close)
    demo = build_ui(app)
    demo.queue()
    server_port = int(args.server_port) if int(args.server_port) > 0 else None
    try:
        launch_private(
            demo,
            server_name=str(args.server_name),
            server_port=server_port,
            inbrowser=bool(args.open_browser),
            share=bool(args.share),
            show_error=False,
            enable_monitoring=False,
            footer_links=[],
        )
    finally:
        try:
            demo.close()
        except Exception:
            pass
        app.close()


if __name__ == "__main__":
    main()
