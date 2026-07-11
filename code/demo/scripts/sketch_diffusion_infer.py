#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = Path(os.environ.get("DRUMTOGRID_RUNS_ROOT", REPO_ROOT.parent.parent / "runs")).resolve()


def _preload_stdlib_inspect() -> None:
    """Avoid the repo-local inspect.py shadowing Python's stdlib inspect."""
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

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import torch

from data.encodec_utils import (
    load_audio_codec_model,
    load_target_pca_basis,
    resolve_codec_metadata_from_cache_config,
    resolve_codec_metadata_from_payload,
    resolve_device,
    resolve_target_layout_from_cache_config,
    resolve_target_pca_basis_path_from_cache_config,
)
from data.diffusion_cache_utils import FAMILY_STATE_FAMILY_NAMES, FAMILY_STATE_ID_VOCAB_SIZES
from data.sketch_dataset import (
    FILL_ROLE_VALUES,
    FEEL_STYLE_VALUES,
    FILL_ACCENT_SHAPE_VALUES,
    HAT_COLOR_VALUES,
    HAT_RATE_VALUES,
    LEGACY_SKETCH_CONTROL_NAMES,
    ORNAMENT_BUDGET_GROUP_NAMES,
    ORNAMENT_BUDGET_GROUP_NAMES_V2,
    ORNAMENT_BUDGET_MAX_COUNTS,
    ORNAMENT_BUDGET_MAX_COUNTS_V2,
    SKETCH_CONTROL_NAMES,
    SKETCH_CONTROL_NAMES_V3,
    SKETCH_CONTROL_NAMES_V2,
    SKETCH_CONTROL_NAMES_V2_17,
    SKETCH_FAMILY_NAMES,
    SNARE_STYLE_VALUES,
    TOM_DIRECTION_VALUES,
    control_tensor_from_public_controls,
    sketch_controls_to_public_dict,
)
from data.sketch_render import DEFAULT_GRID_FRAME_RATE, build_diffusion_batch_from_events
from io_utils import save_audio, write_json
from model import (
    DEFAULT_BEAT_CROSSFADE_MS,
    DEFAULT_INFERENCE_GUIDANCE_SCALE,
    DEFAULT_INFERENCE_NUM_BEATS,
    DEFAULT_SAMPLE_X0_CLIP_NORM,
    ConditionalDiffusionTransformer,
    DiffusionTransformerConfig,
    GaussianDiffusion1D,
    _prepare_batch_tensors,
    apply_beat_crossfade,
    decode_latent_to_audio,
    denormalize_latent,
    resolve_codec_hop_length,
    resolve_inference_geometry,
    resolve_target_token_rate_hz,
    sample_ddpm,
)
try:
    from refiner import (
        DEFAULT_DAC_REFINER_STRENGTH,
        apply_dac_refiner_to_latent,
        load_dac_refiner_checkpoint,
    )
except ModuleNotFoundError as exc:
    if str(exc.name) != "refiner":
        raise
    DEFAULT_DAC_REFINER_STRENGTH = 0.0

    def load_dac_refiner_checkpoint(*_args: Any, **_kwargs: Any) -> tuple[Any, dict[str, Any]]:
        raise RuntimeError("DAC refiner module is unavailable in this workspace")

    def apply_dac_refiner_to_latent(*_args: Any, **_kwargs: Any) -> torch.Tensor:
        raise RuntimeError("DAC refiner module is unavailable in this workspace")
from sketch_expander import SketchExpander, SketchExpanderConfig, decode_event_plan


DEFAULT_SKETCH_GUIDANCE_SCALE = DEFAULT_INFERENCE_GUIDANCE_SCALE


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate diffusion audio from a 16-bin K/S/H sketch.")
    parser.add_argument("--sketch-json", type=str, required=True)
    parser.add_argument("--sketch-checkpoint", type=str, required=True)
    parser.add_argument("--diffusion-checkpoint", type=str, default="")
    parser.add_argument("--diffusion-train-dir", type=str, default=str(RUNS_ROOT / "runs_dac" / "dac_25steps"))
    parser.add_argument(
        "--cache-root",
        type=str,
        default=str(RUNS_ROOT / "mini_cache"),
    )
    parser.add_argument("--out-dir", type=str, default=str(REPO_ROOT / "sketch_diffusion_out"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--bpm", type=float, default=0.0)
    parser.add_argument("--swing", type=float, default=None)
    parser.add_argument("--humanize", type=float, default=None)
    parser.add_argument("--feel-style", type=str, choices=FEEL_STYLE_VALUES, default=None)
    parser.add_argument("--feel-amount", type=float, default=None)
    parser.add_argument("--snare-style", type=str, choices=SNARE_STYLE_VALUES, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--hat-rate", type=str, choices=HAT_RATE_VALUES, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--hat-color", type=str, choices=HAT_COLOR_VALUES, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--fill-role", type=str, choices=FILL_ROLE_VALUES, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--snare-ornament-intensity", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--open-hat-intensity", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--ghost-density", type=float, default=None)
    parser.add_argument("--kick-ghost-density", type=float, default=None)
    parser.add_argument("--snare-ghost-density", type=float, default=None)
    parser.add_argument("--snare-roll-density", type=float, default=None)
    parser.add_argument("--hihat-density", type=float, default=None)
    parser.add_argument("--hihat-openness", type=float, default=None)
    parser.add_argument("--fill-density", type=float, default=None)
    parser.add_argument("--fill-start", type=float, default=None)
    parser.add_argument("--fill-length", type=float, default=None)
    parser.add_argument("--tom-direction", type=str, choices=TOM_DIRECTION_VALUES, default=None)
    parser.add_argument("--fill-accent-shape", type=str, choices=FILL_ACCENT_SHAPE_VALUES, default=None)
    parser.add_argument("--guidance-scale", type=float, default=DEFAULT_SKETCH_GUIDANCE_SCALE)
    parser.add_argument("--x0-clip-norm", type=float, default=DEFAULT_SAMPLE_X0_CLIP_NORM)
    parser.add_argument("--num-beats", type=int, default=DEFAULT_INFERENCE_NUM_BEATS)
    parser.add_argument("--target-token-rate-hz", type=float, default=0.0)
    parser.add_argument("--grid-frame-rate", type=float, default=DEFAULT_GRID_FRAME_RATE)
    parser.add_argument("--beat-crossfade-ms", type=float, default=DEFAULT_BEAT_CROSSFADE_MS)
    parser.add_argument("--refiner-checkpoint", type=str, default="")
    parser.add_argument("--refiner-strength", type=float, default=DEFAULT_DAC_REFINER_STRENGTH)
    parser.add_argument("--disable-refiner", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--grid-only",
        action="store_true",
        help="Only write grid_batch.pt/events.json and skip diffusion/audio.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_json(path: str | Path) -> dict[str, Any]:
    return dict(json.loads(Path(path).expanduser().read_text(encoding="utf-8")))


def _resolve_diffusion_checkpoint(explicit: str, train_dir: str | Path) -> Path:
    if str(explicit).strip():
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"diffusion checkpoint not found: {path}")
        return path
    root = Path(train_dir).expanduser().resolve()
    for name in ("best_diffusion.pt", "best.pt", "last.pt"):
        path = root / name
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"no diffusion checkpoint found under {root}")


def _to_matrix(value: Any, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tuple(tensor.shape) != (len(SKETCH_FAMILY_NAMES), 16):
        raise ValueError(f"{name} must have shape [3,16], got {tuple(tensor.shape)}")
    return tensor.contiguous()


def _controls_from_payload(
    payload: Mapping[str, Any],
    args: argparse.Namespace,
    *,
    control_names: Sequence[str],
) -> torch.Tensor:
    raw = payload.get("controls", {})
    if isinstance(raw, Mapping):
        controls_map: dict[str, Any] = dict(raw)
    else:
        values = [float(x) for x in torch.as_tensor(raw, dtype=torch.float32).view(-1).tolist()]
        if int(len(values)) == int(len(tuple(control_names))):
            controls_map = dict(zip([str(name) for name in control_names], values))
        elif int(len(values)) == int(len(LEGACY_SKETCH_CONTROL_NAMES)):
            controls_map = dict(zip(LEGACY_SKETCH_CONTROL_NAMES, values))
        elif int(len(values)) in {
            int(len(SKETCH_CONTROL_NAMES)),
            int(len(SKETCH_CONTROL_NAMES_V3)),
            int(len(SKETCH_CONTROL_NAMES_V2_17)),
            int(len(SKETCH_CONTROL_NAMES_V2)),
        }:
            controls_map = values
        else:
            raise ValueError(f"controls list must have {len(tuple(control_names))} values, got {len(values)}")
    if not isinstance(controls_map, Mapping):
        controls_tensor = control_tensor_from_public_controls(controls_map, control_names=control_names)
        controls_map = sketch_controls_to_public_dict(controls_tensor, control_names=control_names)
    overrides = {
        "swing": args.swing,
        "humanize": args.humanize,
        "feel_style": args.feel_style,
        "feel_amount": args.feel_amount,
        "snare_style": args.snare_style,
        "hat_rate": args.hat_rate,
        "hat_color": args.hat_color,
        "fill_role": args.fill_role,
        "snare_ornament_intensity": args.snare_ornament_intensity,
        "open_hat_intensity": args.open_hat_intensity,
        "ghost_density": args.ghost_density,
        "kick_ghost_density": args.kick_ghost_density,
        "snare_ghost_density": args.snare_ghost_density,
        "snare_roll_density": args.snare_roll_density,
        "hihat_density": args.hihat_density,
        "hihat_openness": args.hihat_openness,
        "fill_density": args.fill_density,
        "fill_start": args.fill_start,
        "fill_length": args.fill_length,
        "tom_direction": args.tom_direction,
        "fill_accent_shape": args.fill_accent_shape,
    }
    for key, value in overrides.items():
        if value is not None:
            controls_map[str(key)] = value
    return control_tensor_from_public_controls(controls_map, control_names=control_names)


def _load_sketch_inputs(
    payload: Mapping[str, Any],
    args: argparse.Namespace,
    *,
    control_names: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    hits = _to_matrix(payload.get("hits"), name="hits").clamp(0.0, 1.0)
    if payload.get("velocities") is None:
        velocities = torch.where(hits.gt(0.0), torch.full_like(hits, 0.8), torch.zeros_like(hits))
    else:
        velocities = _to_matrix(payload.get("velocities"), name="velocities").clamp(0.0, 1.0)
        velocities = torch.where(hits.gt(0.0), velocities, torch.zeros_like(velocities))
    controls = _controls_from_payload(payload, args, control_names=control_names)
    bpm = float(args.bpm) if float(args.bpm) > 0.0 else float(payload.get("bpm", 120.0) or 120.0)
    if not float(bpm) > 0.0:
        raise ValueError(f"bpm must be positive, got {bpm}")
    return hits, velocities, controls, float(bpm)


def _load_sketch_expander(path: str | Path, *, device: torch.device) -> SketchExpander:
    payload = dict(torch.load(Path(path).expanduser().resolve(), map_location="cpu", weights_only=False))
    cfg_payload = dict(payload.get("config") or {})
    state = dict(payload["model_state_dict"])
    cfg_payload["sketch_family_names"] = tuple(cfg_payload.get("sketch_family_names", SKETCH_FAMILY_NAMES))
    cfg_payload["class_names"] = tuple(cfg_payload.get("class_names", FAMILY_STATE_FAMILY_NAMES))
    cfg_payload["class_id_vocab_sizes"] = tuple(cfg_payload.get("class_id_vocab_sizes", FAMILY_STATE_ID_VOCAB_SIZES))
    if "control_names" not in cfg_payload:
        inferred_control_names: tuple[str, ...] = SKETCH_CONTROL_NAMES
        input_weight = state.get("input_proj.1.weight")
        if isinstance(input_weight, torch.Tensor) and int(input_weight.dim()) == 2:
            sketch_width = 2 * int(len(tuple(cfg_payload["sketch_family_names"])))
            control_width = int(input_weight.shape[1]) - int(sketch_width)
            if control_width == int(len(LEGACY_SKETCH_CONTROL_NAMES)):
                inferred_control_names = tuple(LEGACY_SKETCH_CONTROL_NAMES)
            elif control_width == int(len(SKETCH_CONTROL_NAMES_V2)):
                inferred_control_names = tuple(SKETCH_CONTROL_NAMES_V2)
            elif control_width == int(len(SKETCH_CONTROL_NAMES_V2_17)):
                inferred_control_names = tuple(SKETCH_CONTROL_NAMES_V2_17)
            elif control_width == int(len(SKETCH_CONTROL_NAMES_V3)):
                inferred_control_names = tuple(SKETCH_CONTROL_NAMES_V3)
            elif control_width == int(len(SKETCH_CONTROL_NAMES)):
                inferred_control_names = tuple(SKETCH_CONTROL_NAMES)
        cfg_payload["control_names"] = inferred_control_names
    cfg_payload["control_names"] = tuple(cfg_payload.get("control_names", SKETCH_CONTROL_NAMES))

    budget_weight = state.get("budget_count_head.weight")
    if "budget_group_names" not in cfg_payload:
        if not isinstance(budget_weight, torch.Tensor):
            cfg_payload["budget_group_names"] = ()
            cfg_payload["budget_max_counts"] = ()
        else:
            out_features = int(budget_weight.shape[0])
            v2_features = int(len(ORNAMENT_BUDGET_GROUP_NAMES_V2)) * (int(max(ORNAMENT_BUDGET_MAX_COUNTS_V2)) + 1)
            v3_features = int(len(ORNAMENT_BUDGET_GROUP_NAMES)) * (int(max(ORNAMENT_BUDGET_MAX_COUNTS)) + 1)
            if out_features == v2_features:
                cfg_payload["budget_group_names"] = tuple(ORNAMENT_BUDGET_GROUP_NAMES_V2)
                cfg_payload["budget_max_counts"] = tuple(ORNAMENT_BUDGET_MAX_COUNTS_V2)
            elif out_features == v3_features:
                cfg_payload["budget_group_names"] = tuple(ORNAMENT_BUDGET_GROUP_NAMES)
                cfg_payload["budget_max_counts"] = tuple(ORNAMENT_BUDGET_MAX_COUNTS)
            else:
                cfg_payload["budget_group_names"] = tuple(ORNAMENT_BUDGET_GROUP_NAMES)
                cfg_payload["budget_max_counts"] = tuple(ORNAMENT_BUDGET_MAX_COUNTS)
    cfg_payload["budget_group_names"] = tuple(cfg_payload.get("budget_group_names", ()))
    if "budget_max_counts" not in cfg_payload or int(len(tuple(cfg_payload.get("budget_max_counts") or ()))) != int(len(cfg_payload["budget_group_names"])):
        if cfg_payload["budget_group_names"] == tuple(ORNAMENT_BUDGET_GROUP_NAMES_V2):
            cfg_payload["budget_max_counts"] = tuple(ORNAMENT_BUDGET_MAX_COUNTS_V2)
        elif cfg_payload["budget_group_names"] == tuple(ORNAMENT_BUDGET_GROUP_NAMES):
            cfg_payload["budget_max_counts"] = tuple(ORNAMENT_BUDGET_MAX_COUNTS)
        else:
            cfg_payload["budget_max_counts"] = tuple(8 for _ in cfg_payload["budget_group_names"])
    cfg_payload["budget_max_counts"] = tuple(cfg_payload.get("budget_max_counts", ()))
    cfg = SketchExpanderConfig(**cfg_payload)
    model = SketchExpander(cfg).to(device).eval()
    model.load_state_dict(state, strict=False)
    return model


def _load_diffusion_state(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[ConditionalDiffusionTransformer, GaussianDiffusion1D, torch.Tensor, torch.Tensor, dict[str, Any]]:
    payload = dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False))
    cfg_payload = dict(payload.get("config") or {})
    cfg_payload.setdefault("positional_encoding", "seconds")
    cfg_payload.setdefault("positional_rate_hz", 50.0)
    cfg = DiffusionTransformerConfig(**cfg_payload)
    model = ConditionalDiffusionTransformer(cfg).to(device).eval()
    model.load_state_dict(dict(payload["model_state_dict"]))
    diffusion = GaussianDiffusion1D(num_steps=int(payload.get("num_steps") or 25)).to(device)
    target_mean = torch.as_tensor(payload["target_mean"], dtype=torch.float32, device=device).view(-1)
    target_std = torch.as_tensor(payload["target_std"], dtype=torch.float32, device=device).view(-1).clamp_min(1.0e-6)
    return model, diffusion, target_mean.contiguous(), target_std.contiguous(), payload


def _samples_per_latent_frame(decoded_num_samples: int, latent_num_frames: int) -> int:
    decoded = int(decoded_num_samples)
    frames = int(latent_num_frames)
    if int(decoded) % int(frames) != 0:
        raise ValueError(f"decoded samples {decoded} not divisible by latent frames {frames}")
    return int(decoded // frames)


def main() -> None:
    args = _parse_args()
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir.exists():
        if bool(args.overwrite):
            shutil.rmtree(out_dir)
        elif any(out_dir.iterdir()):
            raise FileExistsError(f"out-dir already exists and is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_device = resolve_device(str(args.device))
    device = torch.device(resolved_device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)

    sketch_model = _load_sketch_expander(args.sketch_checkpoint, device=device)
    sketch_payload = _read_json(args.sketch_json)
    sketch_hits, sketch_vel, controls, bpm = _load_sketch_inputs(
        sketch_payload,
        args,
        control_names=sketch_model.cfg.control_names,
    )
    with torch.no_grad():
        sketch_outputs = sketch_model(
            sketch_hits.unsqueeze(0).to(device=device),
            sketch_vel.unsqueeze(0).to(device=device),
            controls.unsqueeze(0).to(device=device),
        )
    event_batches = decode_event_plan(
        {key: value.detach().cpu() for key, value in sketch_outputs.items()},
        sketch_hits=sketch_hits.unsqueeze(0),
        sketch_vel=sketch_vel.unsqueeze(0),
        controls=controls.unsqueeze(0),
        class_names=sketch_model.class_names,
        class_id_vocab_sizes=sketch_model.class_id_vocab_sizes,
        control_names=sketch_model.cfg.control_names,
        budget_group_names=sketch_model.budget_group_names,
        budget_max_counts=sketch_model.budget_max_counts,
        seed=int(args.seed),
    )
    public_controls = sketch_controls_to_public_dict(controls, control_names=sketch_model.cfg.control_names)

    target_token_rate_hz = float(args.target_token_rate_hz) if float(args.target_token_rate_hz) > 0.0 else None
    batch = build_diffusion_batch_from_events(
        event_batches,
        bpm=float(bpm),
        num_beats=int(args.num_beats),
        grid_frame_rate=float(args.grid_frame_rate),
        target_token_rate_hz=target_token_rate_hz,
    )
    torch.save(batch, out_dir / "grid_batch.pt")
    (out_dir / "events.json").write_text(json.dumps(event_batches[0], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if bool(args.grid_only):
        write_json(
            out_dir / "summary.json",
            {
                "sketch_json": str(Path(args.sketch_json).expanduser().resolve()),
                "sketch_checkpoint": str(Path(args.sketch_checkpoint).expanduser().resolve()),
                "cache_root": str(Path(args.cache_root).expanduser().resolve()),
                "out_dir": str(out_dir),
                "resolved_device": str(resolved_device),
                "grid_only": True,
                "bpm": float(bpm),
                "control_names": list(sketch_model.cfg.control_names),
                "controls": public_controls,
                "encoded_controls": {
                    name: float(controls[idx].item())
                    for idx, name in enumerate(sketch_model.cfg.control_names)
                },
            "guidance_scale": float(args.guidance_scale),
            "sample_seed": int(args.seed),
            "num_events": int(len(event_batches[0])),
                "grid_num_frames": int(batch["grid_num_frames_b"][0].item()),
                "grid_batch": "grid_batch.pt",
                "events": "events.json",
            },
        )
        return

    diffusion_checkpoint = _resolve_diffusion_checkpoint(args.diffusion_checkpoint, args.diffusion_train_dir)
    diffusion_model, diffusion, target_mean, target_std, diffusion_payload = _load_diffusion_state(
        diffusion_checkpoint,
        device=device,
    )
    codec_metadata = resolve_codec_metadata_from_payload(
        diffusion_payload,
        fallback=resolve_codec_metadata_from_cache_config(args.cache_root),
    )
    audio_codec_model, _codec_device, codec_metadata = load_audio_codec_model(device=resolved_device, metadata=codec_metadata)
    target_token_rate_hz = (
        float(args.target_token_rate_hz)
        if float(args.target_token_rate_hz) > 0.0
        else resolve_target_token_rate_hz(codec_metadata)
    )
    if "token_times_sec" not in batch or "target_valid_mask_bt" not in batch:
        batch = build_diffusion_batch_from_events(
            event_batches,
            bpm=float(bpm),
            num_beats=int(args.num_beats),
            grid_frame_rate=float(args.grid_frame_rate),
            target_token_rate_hz=float(target_token_rate_hz),
        )
        torch.save(batch, out_dir / "grid_batch.pt")

    target_layout = str(
        diffusion_payload.get("target_layout")
        or resolve_target_layout_from_cache_config(args.cache_root)
    ).strip().lower()
    target_pca_basis = None
    if diffusion_payload.get("target_pca_basis") is not None:
        target_pca_basis = load_target_pca_basis(diffusion_payload["target_pca_basis"], device=device)
    else:
        basis_path = resolve_target_pca_basis_path_from_cache_config(args.cache_root)
        if basis_path is not None:
            target_pca_basis = load_target_pca_basis(basis_path, device=device)
    if str(target_layout) == "framewise_pca" and target_pca_basis is None:
        raise FileNotFoundError("framewise_pca diffusion checkpoint requires a PCA basis")

    refiner_model = None
    refiner_payload: dict[str, Any] = {}
    if str(args.refiner_checkpoint).strip() and not bool(args.disable_refiner):
        refiner_model, refiner_payload = load_dac_refiner_checkpoint(args.refiner_checkpoint, device=device)

    sample_rate = int(diffusion_payload.get("sample_rate") or codec_metadata.codec_sample_rate)
    prepared = _prepare_batch_tensors(batch, device, require_target=False, require_timing=False)
    geometry = resolve_inference_geometry(
        prepared,
        use_bpm_inference_geometry=True,
        inference_num_beats=int(args.num_beats),
        target_token_rate_hz=float(target_token_rate_hz),
    )
    with torch.no_grad():
        latent_norm = sample_ddpm(
            model=diffusion_model,
            diffusion=diffusion,
            batch=batch,
            device=device,
            guidance_scale=float(args.guidance_scale),
            sample_seed=int(args.seed),
            x0_clip_norm=float(args.x0_clip_norm) if args.x0_clip_norm is not None else None,
            use_bpm_inference_geometry=True,
            inference_num_beats=int(args.num_beats),
            target_token_rate_hz=float(target_token_rate_hz),
            inference_geometry=geometry,
        )
        latent = denormalize_latent(latent_norm, target_mean, target_std)
        latent = latent * geometry["target_valid_mask_bt"].unsqueeze(-1)
        if refiner_model is not None:
            latent = apply_dac_refiner_to_latent(
                refiner_model,
                latent,
                prepared,
                geometry,
                strength=float(args.refiner_strength),
            )
            latent = latent * geometry["target_valid_mask_bt"].unsqueeze(-1)
        decoded_audio = decode_latent_to_audio(latent, audio_codec_model, target_pca_basis=target_pca_basis)
    if int(decoded_audio.dim()) == 2:
        decoded_audio = decoded_audio.unsqueeze(1)
    samples_per_frame = _samples_per_latent_frame(
        decoded_num_samples=int(decoded_audio.shape[-1]),
        latent_num_frames=int(geometry["target_valid_mask_bt"].shape[1]),
    )
    target_frames = int(geometry["target_num_frames_b"][0].item())
    audio_i = decoded_audio[0, :, : int(target_frames * samples_per_frame)].detach()
    requested_duration_sec = float(geometry["duration_sec"][0].item())
    decoded_duration_sec = float(audio_i.shape[-1]) / float(sample_rate)
    duration_error_ms = float((decoded_duration_sec - requested_duration_sec) * 1000.0)
    if float(args.beat_crossfade_ms) > 0.0:
        beat_mask = geometry["beat_boundaries_valid_mask"][0]
        beat_boundaries = geometry["beat_boundaries_sec"][0][beat_mask]
        audio_i = apply_beat_crossfade(
            audio_i,
            beat_boundaries,
            sample_rate=int(sample_rate),
            beat_crossfade_ms=float(args.beat_crossfade_ms),
        )
    save_audio(out_dir / "output.wav", audio_i.unsqueeze(0).cpu(), sample_rate=int(sample_rate))
    write_json(
        out_dir / "summary.json",
        {
            "sketch_json": str(Path(args.sketch_json).expanduser().resolve()),
            "sketch_checkpoint": str(Path(args.sketch_checkpoint).expanduser().resolve()),
            "diffusion_checkpoint": str(diffusion_checkpoint),
            "cache_root": str(Path(args.cache_root).expanduser().resolve()),
            "out_dir": str(out_dir),
            "resolved_device": str(resolved_device),
            "bpm": float(bpm),
            "control_names": list(sketch_model.cfg.control_names),
            "controls": public_controls,
            "encoded_controls": {
                name: float(controls[idx].item())
                for idx, name in enumerate(sketch_model.cfg.control_names)
            },
            "num_events": int(len(event_batches[0])),
            "grid_num_frames": int(batch["grid_num_frames_b"][0].item()),
            "target_num_frames": int(target_frames),
            "target_token_rate_hz": float(target_token_rate_hz),
            "effective_token_rate_hz": float(target_token_rate_hz),
            "guidance_scale": float(args.guidance_scale),
            "sample_seed": int(args.seed),
            "codec_hop_length": resolve_codec_hop_length(codec_metadata),
            "requested_duration_sec": float(requested_duration_sec),
            "decoded_duration_sec": float(decoded_duration_sec),
            "duration_error_ms": float(duration_error_ms),
            "sample_rate": int(sample_rate),
            "refiner_enabled": bool(refiner_model is not None),
            "refiner_checkpoint": str(refiner_payload.get("checkpoint_path") or args.refiner_checkpoint or ""),
            "refiner_strength": float(args.refiner_strength),
            "wav": "output.wav",
            "grid_batch": "grid_batch.pt",
            "events": "events.json",
        },
    )


if __name__ == "__main__":
    main()
