from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

from data.diffusion_cache_utils import (
    FAMILY_STATE_FEATURE_ROW_NAMES,
    FAMILY_STATE_FAMILY_NAMES,
    FAMILY_STATE_ID_VOCAB_SIZES,
    grid_times_from_fps,
    token_times_from_duration,
)
from data.family_state_cache_utils import render_midi_family_state_grid


DEFAULT_GRID_FRAME_RATE = 250.0
DEFAULT_NUM_BEATS = 4

CLASS_ID_TO_PITCH: dict[str, tuple[int, ...]] = {
    "kick": (36,),
    "snare": (38, 40, 37),
    "tom_high": (48, 50),
    "tom_mid": (45, 47),
    "tom_floor": (41, 58),
    "hihat": (46, 26, 42, 22, 44),
    "crash": (49, 52),
    "ride": (51, 59, 53),
}


def duration_from_bpm(bpm: float, *, num_beats: int = DEFAULT_NUM_BEATS) -> float:
    bpm_eff = float(bpm)
    if not float(bpm_eff) > 0.0:
        raise ValueError(f"bpm must be positive, got {bpm}")
    return (float(max(1, int(num_beats))) * 60.0) / float(bpm_eff)


def step_boundaries_from_duration(duration_sec: float, *, num_steps: int = 16) -> torch.Tensor:
    duration = float(duration_sec)
    if not float(duration) > 0.0:
        raise ValueError(f"duration_sec must be positive, got {duration_sec}")
    return torch.linspace(0.0, float(duration), steps=int(num_steps) + 1, dtype=torch.float32)


def event_time_from_step_offset(
    *,
    step: int,
    offset: float,
    step_boundaries_sec: torch.Tensor,
) -> float:
    step_idx = int(step)
    if not (0 <= int(step_idx) < int(step_boundaries_sec.numel()) - 1):
        raise ValueError(f"step out of range: {step}")
    step_start = float(step_boundaries_sec[int(step_idx)].item())
    step_end = float(step_boundaries_sec[int(step_idx) + 1].item())
    span = max(1.0e-6, float(step_end) - float(step_start))
    return float(float(step_start) + (float(offset) * float(span)))


def _pitch_for_event(family: str, class_id: int) -> int:
    family_name = str(family)
    choices = CLASS_ID_TO_PITCH.get(family_name)
    if not choices:
        return 36
    idx = int(max(0, min(int(class_id), len(choices) - 1)))
    return int(choices[int(idx)])


def _events_to_midi_arrays(
    events: Sequence[Mapping[str, Any]],
    *,
    duration_sec: float,
    time_offset_sec: float = 0.0,
) -> dict[str, np.ndarray]:
    duration = float(duration_sec)
    step_boundaries = step_boundaries_from_duration(float(duration), num_steps=16)
    pitches: list[int] = []
    times: list[float] = []
    velocities: list[float] = []
    upper = float(np.nextafter(np.float32(duration), np.float32(0.0)))
    for event in list(events):
        family = str(event.get("family") or "")
        if family not in set(FAMILY_STATE_FAMILY_NAMES):
            continue
        step = int(event.get("step", 0))
        offset = float(event.get("offset", 0.0))
        time_sec = (
            float(event["time_sec"])
            if event.get("time_sec") is not None
            else event_time_from_step_offset(step=step, offset=offset, step_boundaries_sec=step_boundaries)
        )
        time_sec = float(max(0.0, min(float(upper), float(time_sec))))
        velocity = float(max(0.0, min(1.0, float(event.get("velocity", 0.8)))))
        if float(velocity) <= 0.0:
            continue
        class_id = int(event.get("class_id", 0))
        pitches.append(_pitch_for_event(family, int(class_id)))
        times.append(float(time_offset_sec) + float(time_sec))
        velocities.append(float(velocity))

    if not times:
        return {
            "pitches": np.zeros((0,), dtype=np.int16),
            "times_sec": np.zeros((0,), dtype=np.float32),
            "velocities": np.zeros((0,), dtype=np.float32),
        }
    order = np.argsort(np.asarray(times, dtype=np.float32), kind="stable")
    return {
        "pitches": np.asarray(pitches, dtype=np.int16)[order],
        "times_sec": np.asarray(times, dtype=np.float32)[order],
        "velocities": np.asarray(velocities, dtype=np.float32)[order],
    }


def _concat_midi_arrays(items: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    nonempty = [item for item in list(items) if int(np.asarray(item.get("times_sec", [])).size) > 0]
    if not nonempty:
        return {
            "pitches": np.zeros((0,), dtype=np.int16),
            "times_sec": np.zeros((0,), dtype=np.float32),
            "velocities": np.zeros((0,), dtype=np.float32),
        }
    pitches = np.concatenate([np.asarray(item["pitches"], dtype=np.int16).reshape(-1) for item in nonempty])
    times = np.concatenate([np.asarray(item["times_sec"], dtype=np.float32).reshape(-1) for item in nonempty])
    velocities = np.concatenate([np.asarray(item["velocities"], dtype=np.float32).reshape(-1) for item in nonempty])
    order = np.argsort(times, kind="stable")
    return {
        "pitches": pitches[order].astype(np.int16, copy=False),
        "times_sec": times[order].astype(np.float32, copy=False),
        "velocities": velocities[order].astype(np.float32, copy=False),
    }


def render_events_to_grid(
    events: Sequence[Mapping[str, Any]],
    *,
    bpm: float,
    num_beats: int = DEFAULT_NUM_BEATS,
    grid_frame_rate: float = DEFAULT_GRID_FRAME_RATE,
    start_sec: float = 0.0,
    midi_events: Mapping[str, np.ndarray] | None = None,
) -> dict[str, torch.Tensor | float | int]:
    duration_sec = duration_from_bpm(float(bpm), num_beats=int(num_beats))
    grid_frames = int(max(1, round(float(duration_sec) * float(grid_frame_rate))))
    step_boundaries = step_boundaries_from_duration(float(duration_sec), num_steps=16)
    midi_events_eff = (
        _events_to_midi_arrays(events, duration_sec=float(duration_sec), time_offset_sec=float(start_sec))
        if midi_events is None
        else {
            "pitches": np.asarray(midi_events.get("pitches", np.zeros((0,), dtype=np.int16)), dtype=np.int16),
            "times_sec": np.asarray(midi_events.get("times_sec", np.zeros((0,), dtype=np.float32)), dtype=np.float32),
            "velocities": np.asarray(
                midi_events.get("velocities", np.zeros((0,), dtype=np.float32)),
                dtype=np.float32,
            ),
        }
    )

    grid_np, onset_ids_np, family_onsets_np, family_onset_count_np, _support_np, _salience_np = render_midi_family_state_grid(
        midi_events=midi_events_eff,
        start_sec=float(start_sec),
        end_sec=float(start_sec) + float(duration_sec),
        num_frames=int(grid_frames),
    )
    grid_times = grid_times_from_fps(int(grid_frames), float(grid_frame_rate))
    return {
        "grid": torch.from_numpy(np.asarray(grid_np, dtype=np.float32)).to(dtype=torch.float32),
        "grid_ids": torch.from_numpy(np.asarray(onset_ids_np, dtype=np.int64)).to(dtype=torch.long),
        "family_onsets_bft": torch.from_numpy(np.asarray(family_onsets_np, dtype=np.bool_)).to(dtype=torch.bool),
        "family_onset_count_bft": torch.from_numpy(np.asarray(family_onset_count_np, dtype=np.uint8)).to(dtype=torch.uint8),
        "grid_times_sec": torch.from_numpy(np.asarray(grid_times, dtype=np.float32)).to(dtype=torch.float32),
        "beat_boundaries_sec": torch.linspace(0.0, float(duration_sec), steps=int(num_beats) + 1, dtype=torch.float32),
        "step_boundaries_sec": step_boundaries.contiguous(),
        "duration_sec": float(duration_sec),
        "grid_num_frames": int(grid_frames),
        "grid_frame_rate": float(grid_frame_rate),
    }


def build_diffusion_batch_from_events(
    events_batch: Sequence[Sequence[Mapping[str, Any]]],
    *,
    bpm: float | Sequence[float],
    num_beats: int = DEFAULT_NUM_BEATS,
    grid_frame_rate: float = DEFAULT_GRID_FRAME_RATE,
    target_token_rate_hz: float | None = None,
) -> dict[str, Any]:
    if isinstance(bpm, (int, float)):
        bpm_values = [float(bpm)] * int(len(events_batch))
    else:
        bpm_values = [float(x) for x in list(bpm)]
    if int(len(bpm_values)) != int(len(events_batch)):
        raise ValueError(f"bpm count must match events batch, got {len(bpm_values)} vs {len(events_batch)}")
    durations = [
        duration_from_bpm(float(bpm_values[int(idx)]), num_beats=int(num_beats))
        for idx in range(int(len(events_batch)))
    ]
    start_offsets: list[float] = []
    cursor = 0.0
    for duration in durations:
        start_offsets.append(float(cursor))
        cursor += float(duration)
    global_midi_events = _concat_midi_arrays(
        [
            _events_to_midi_arrays(
                events,
                duration_sec=float(durations[int(idx)]),
                time_offset_sec=float(start_offsets[int(idx)]),
            )
            for idx, events in enumerate(list(events_batch))
        ]
    )
    rendered = [
        render_events_to_grid(
            events,
            bpm=float(bpm_values[int(idx)]),
            num_beats=int(num_beats),
            grid_frame_rate=float(grid_frame_rate),
            start_sec=float(start_offsets[int(idx)]),
            midi_events=global_midi_events,
        )
        for idx, events in enumerate(list(events_batch))
    ]
    batch_size = int(len(rendered))
    if int(batch_size) <= 0:
        raise ValueError("events_batch must not be empty")
    max_grid_len = max(int(item["grid_num_frames"]) for item in rendered)
    grid_dim = int(len(FAMILY_STATE_FEATURE_ROW_NAMES))
    family_dim = int(len(FAMILY_STATE_FAMILY_NAMES))
    grid_bft = torch.zeros((batch_size, grid_dim, max_grid_len), dtype=torch.float32)
    grid_ids_bct = torch.full((batch_size, family_dim, max_grid_len), -1, dtype=torch.long)
    family_onsets_bft = torch.zeros((batch_size, family_dim, max_grid_len), dtype=torch.bool)
    family_onset_count_bft = torch.zeros((batch_size, family_dim, max_grid_len), dtype=torch.uint8)
    grid_valid_mask_bt = torch.zeros((batch_size, max_grid_len), dtype=torch.bool)
    grid_times_sec_bt = torch.zeros((batch_size, max_grid_len), dtype=torch.float32)
    beat_boundaries_sec_bk = torch.zeros((batch_size, int(num_beats) + 1), dtype=torch.float32)
    beat_boundaries_valid_mask_bk = torch.ones((batch_size, int(num_beats) + 1), dtype=torch.bool)
    bpm_b = torch.tensor(bpm_values, dtype=torch.float32)
    duration_sec_b = torch.zeros((batch_size,), dtype=torch.float32)
    grid_frame_rate_b = torch.full((batch_size,), float(grid_frame_rate), dtype=torch.float32)
    grid_num_frames_b = torch.zeros((batch_size,), dtype=torch.long)

    token_times_sec_bt = None
    target_valid_mask_bt = None
    target_num_frames_b = None
    if target_token_rate_hz is not None and float(target_token_rate_hz) > 0.0:
        target_lengths = [
            int(max(1, round(float(item["duration_sec"]) * float(target_token_rate_hz))))
            for item in rendered
        ]
        max_target_len = int(max(target_lengths))
        token_times_sec_bt = torch.zeros((batch_size, max_target_len), dtype=torch.float32)
        target_valid_mask_bt = torch.zeros((batch_size, max_target_len), dtype=torch.bool)
        target_num_frames_b = torch.tensor(target_lengths, dtype=torch.long)

    for batch_idx, item in enumerate(rendered):
        grid_len = int(item["grid_num_frames"])
        grid_bft[int(batch_idx), :, :grid_len] = torch.as_tensor(item["grid"], dtype=torch.float32)
        grid_ids_bct[int(batch_idx), :, :grid_len] = torch.as_tensor(item["grid_ids"], dtype=torch.long)
        family_onsets_bft[int(batch_idx), :, :grid_len] = torch.as_tensor(item["family_onsets_bft"], dtype=torch.bool)
        family_onset_count_bft[int(batch_idx), :, :grid_len] = torch.as_tensor(
            item["family_onset_count_bft"], dtype=torch.uint8
        )
        grid_valid_mask_bt[int(batch_idx), :grid_len] = True
        grid_times_sec_bt[int(batch_idx), :grid_len] = torch.as_tensor(item["grid_times_sec"], dtype=torch.float32)
        beat_boundaries_sec_bk[int(batch_idx)] = torch.as_tensor(item["beat_boundaries_sec"], dtype=torch.float32)
        duration_sec_b[int(batch_idx)] = float(item["duration_sec"])
        grid_num_frames_b[int(batch_idx)] = int(grid_len)
        if token_times_sec_bt is not None and target_valid_mask_bt is not None and target_num_frames_b is not None:
            target_len = int(target_num_frames_b[int(batch_idx)].item())
            token_times = token_times_from_duration(int(target_len), float(item["duration_sec"]))
            token_times_sec_bt[int(batch_idx), :target_len] = torch.from_numpy(token_times).to(dtype=torch.float32)
            target_valid_mask_bt[int(batch_idx), :target_len] = True

    batch: dict[str, Any] = {
        "conditioning_mode": "sketch_expander_rendered",
        "class_names": list(FAMILY_STATE_FAMILY_NAMES),
        "class_id_vocab_sizes": [int(x) for x in FAMILY_STATE_ID_VOCAB_SIZES],
        "feature_row_names": list(FAMILY_STATE_FEATURE_ROW_NAMES),
        "grid": grid_bft.contiguous(),
        "grid_ids": grid_ids_bct.contiguous(),
        "family_onsets_bft": family_onsets_bft.contiguous(),
        "family_onset_count_bft": family_onset_count_bft.contiguous(),
        "grid_valid_mask": grid_valid_mask_bt.contiguous(),
        "grid_times_sec": grid_times_sec_bt.contiguous(),
        "beat_boundaries_sec": beat_boundaries_sec_bk.contiguous(),
        "beat_boundaries_valid_mask": beat_boundaries_valid_mask_bk.contiguous(),
        "bpm": bpm_b.contiguous(),
        "duration_sec": duration_sec_b.contiguous(),
        "grid_frame_rate_b": grid_frame_rate_b.contiguous(),
        "grid_num_frames_b": grid_num_frames_b.contiguous(),
        "source_id": [f"sketch_{idx:03d}" for idx in range(batch_size)],
        "split": ["inference"] * batch_size,
        "beat_index_b": torch.zeros((batch_size,), dtype=torch.long),
        "source_manifest_index_b": torch.full((batch_size,), -1, dtype=torch.long),
    }
    if token_times_sec_bt is not None and target_valid_mask_bt is not None and target_num_frames_b is not None:
        batch["token_times_sec"] = token_times_sec_bt.contiguous()
        batch["target_valid_mask_bt"] = target_valid_mask_bt.contiguous()
        batch["target_num_frames_b"] = target_num_frames_b.contiguous()
    return batch
