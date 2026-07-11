from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch


FALLBACK_FAMILY_NAMES = (
    "kick",
    "snare",
    "tom_high",
    "tom_mid",
    "tom_floor",
    "hihat",
    "crash",
    "ride",
)

GM_DRUM_NOTES = {
    "kick": 36,
    "snare": 38,
    "tom_high": 50,
    "tom_mid": 47,
    "tom_floor": 43,
    "hihat": 42,
    "crash": 49,
    "ride": 51,
}


def payload_family_names(payload: Mapping[str, Any]) -> list[str]:
    names = [str(value) for value in list(payload.get("class_names") or [])]
    return names or list(FALLBACK_FAMILY_NAMES)


def payload_duration_sec(payload: Mapping[str, Any]) -> float:
    value = float(payload.get("duration_sec", 0.0) or 0.0)
    if value <= 0.0:
        grid_times = torch.as_tensor(payload.get("grid_times_sec_t"), dtype=torch.float32)
        if int(grid_times.numel()) > 0:
            value = float(grid_times.max().item())
    return float(max(value, 1.0e-3))


def _family_velocity(
    *,
    payload: Mapping[str, Any],
    family: str,
    family_idx: int,
    frame_idx: int,
) -> float:
    grid_ft = torch.as_tensor(payload.get("grid_ft"), dtype=torch.float32)
    feature_names = [str(value) for value in list(payload.get("feature_row_names") or [])]
    candidates: list[int] = []
    for name in (f"{family}_onset_vel", f"{family}_state_vel"):
        if name in feature_names:
            candidates.append(int(feature_names.index(name)))
    fallback = int(family_idx) * 3 + 1
    if 0 <= fallback < int(grid_ft.shape[0]):
        candidates.append(fallback)
    for row_idx in candidates:
        if 0 <= int(frame_idx) < int(grid_ft.shape[-1]):
            value = float(grid_ft[int(row_idx), int(frame_idx)].item())
            if value > 0.0:
                return float(max(0.05, min(1.0, value)))
    return 0.75


def grid_payload_to_pianoroll(
    payload: Mapping[str, Any],
    *,
    num_frames: int,
    duration_sec: float | None = None,
    sustain_frames: int = 1,
) -> np.ndarray:
    """Convert the family grid into CTD's official 128-note piano-roll condition."""
    frame_count = int(max(1, int(num_frames)))
    duration = payload_duration_sec(payload) if duration_sec is None else float(duration_sec)
    pr = np.zeros((128, frame_count), dtype=np.float32)
    onsets = torch.as_tensor(payload.get("family_onsets_ft"), dtype=torch.bool)
    grid_times = torch.as_tensor(payload.get("grid_times_sec_t"), dtype=torch.float32)
    if int(onsets.dim()) != 2 or int(grid_times.numel()) <= 0:
        return pr
    family_names = payload_family_names(payload)
    for family_idx, family in enumerate(family_names[: int(onsets.shape[0])]):
        note = int(GM_DRUM_NOTES.get(str(family), 39))
        frames = torch.nonzero(onsets[int(family_idx)], as_tuple=False).flatten().tolist()
        for grid_frame in frames:
            time_sec = float(grid_times[int(grid_frame)].item())
            if time_sec < 0.0:
                continue
            pr_frame = int(round(time_sec / max(duration, 1.0e-8) * float(frame_count - 1)))
            if pr_frame < 0 or pr_frame >= frame_count:
                continue
            velocity = _family_velocity(
                payload=payload,
                family=str(family),
                family_idx=int(family_idx),
                frame_idx=int(grid_frame),
            )
            end = min(frame_count, int(pr_frame) + max(1, int(sustain_frames)))
            pr[int(note), int(pr_frame) : int(end)] = np.maximum(
                pr[int(note), int(pr_frame) : int(end)],
                np.float32(velocity),
            )
    peak = float(np.max(np.abs(pr))) if pr.size else 0.0
    if peak > 0.0:
        pr = pr / peak
    return pr.astype(np.float32, copy=False)


def grid_payload_to_drum_guide_audio(payload: Mapping[str, Any], *, sample_rate: int) -> torch.Tensor:
    """Render a deterministic drum guide audio signal from the grid for public control APIs."""
    from scripts.export_dac_baseline_predictions import _procedural_grid_render

    return _procedural_grid_render(payload, sample_rate=int(sample_rate)).to(dtype=torch.float32).contiguous()


def crop_or_pad_audio(audio_bct: torch.Tensor, *, num_samples: int) -> torch.Tensor:
    audio = torch.as_tensor(audio_bct, dtype=torch.float32)
    if int(audio.dim()) == 2:
        audio = audio.unsqueeze(0)
    if int(audio.dim()) != 3:
        raise ValueError(f"expected audio [B,C,T] or [C,T], got {tuple(audio.shape)}")
    target = int(max(1, int(num_samples)))
    if int(audio.shape[-1]) > target:
        return audio[..., :target].contiguous()
    if int(audio.shape[-1]) < target:
        return torch.nn.functional.pad(audio, (0, target - int(audio.shape[-1]))).contiguous()
    return audio.contiguous()


def mixdown_to_mono(audio_bct: torch.Tensor) -> torch.Tensor:
    audio = torch.as_tensor(audio_bct, dtype=torch.float32)
    if int(audio.dim()) == 2:
        audio = audio.unsqueeze(0)
    if int(audio.shape[1]) > 1:
        audio = audio.mean(dim=1, keepdim=True)
    return audio.contiguous()


def c_major_chords(duration_sec: float, *, segment_sec: float = 2.0) -> list[tuple[str, float]]:
    duration = max(1.0e-3, float(duration_sec))
    step = max(0.25, float(segment_sec))
    times = np.arange(0.0, duration + 1.0e-6, step, dtype=np.float32).tolist()
    if not times:
        times = [0.0]
    return [("C", float(time_sec)) for time_sec in times]
