from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
import torch


FAMILY_STATE_FAMILY_NAMES: tuple[str, ...] = (
    "kick",
    "snare",
    "tom_high",
    "tom_mid",
    "tom_floor",
    "hihat",
    "crash",
    "ride",
)
FAMILY_STATE_NUMERIC_COMPONENT_NAMES: tuple[str, ...] = (
    "state_vel",
    "onset_vel",
    "onset_count",
)
FAMILY_STATE_FEATURE_ROW_NAMES: tuple[str, ...] = tuple(
    f"{family_name}_{component_name}"
    for family_name in FAMILY_STATE_FAMILY_NAMES
    for component_name in FAMILY_STATE_NUMERIC_COMPONENT_NAMES
)
FAMILY_STATE_ID_VOCAB_SIZES: tuple[int, ...] = (
    1,
    3,
    2,
    2,
    2,
    5,
    2,
    3,
)


def hash_bytes(payload: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def hash_tensor(tensor: torch.Tensor) -> str:
    arr = tensor.detach().to(device="cpu").contiguous().numpy()
    return hash_bytes(arr.tobytes())


def fit_pca(
    x_nd: np.ndarray,
    *,
    pca_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    x = np.asarray(x_nd, dtype=np.float32)
    if int(x.ndim) != 2:
        raise ValueError(f"x_nd must be [N,D], got shape {tuple(x.shape)}")
    if int(x.shape[0]) <= 0:
        raise ValueError("x_nd must have at least one row")
    mean = x.mean(axis=0, keepdims=True)
    xc = x - mean
    _u, s, vt = np.linalg.svd(xc, full_matrices=False)
    var = (s ** 2) / max(1, (xc.shape[0] - 1))
    ratio = var / max(var.sum(), 1.0e-8)
    k = int(min(max(1, int(pca_k)), int(vt.shape[0])))
    components = vt[:k].astype(np.float32, copy=False)
    explained = ratio[:k].astype(np.float32, copy=False)
    return mean.astype(np.float32, copy=False), components, explained, k


def fit_projected_pca(
    x_nd: np.ndarray,
    *,
    projection_basis_dq: np.ndarray,
    pca_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    x = np.asarray(x_nd, dtype=np.float32)
    basis = np.asarray(projection_basis_dq, dtype=np.float32)
    if int(x.ndim) != 2:
        raise ValueError(f"x_nd must be [N,D], got shape {tuple(x.shape)}")
    if int(basis.ndim) != 2:
        raise ValueError(f"projection_basis_dq must be [D,Q], got shape {tuple(basis.shape)}")
    if int(x.shape[0]) <= 0:
        raise ValueError("x_nd must have at least one row")
    if int(x.shape[1]) != int(basis.shape[0]):
        raise ValueError(
            f"x_nd / projection basis dimension mismatch: {tuple(x.shape)} vs {tuple(basis.shape)}"
        )
    mean_full = x.mean(axis=0, keepdims=True)
    coeff = (x - mean_full) @ basis
    coeff_mean = coeff.mean(axis=0, keepdims=True)
    coeff_centered = coeff - coeff_mean
    _u, s, vt = np.linalg.svd(coeff_centered, full_matrices=False)
    var = (s ** 2) / max(1, (coeff_centered.shape[0] - 1))
    ratio = var / max(var.sum(), 1.0e-8)
    k = int(min(max(1, int(pca_k)), int(vt.shape[0])))
    coeff_components = vt[:k].astype(np.float32, copy=False)
    components = (coeff_components @ basis.transpose()).astype(np.float32, copy=False)
    mean = (mean_full + (coeff_mean @ basis.transpose())).astype(np.float32, copy=False)
    explained = ratio[:k].astype(np.float32, copy=False)
    return mean.reshape(-1), components, explained, k


def fix_component_signs(components: np.ndarray) -> np.ndarray:
    fixed = np.asarray(components, dtype=np.float32).copy()
    for idx in range(int(fixed.shape[0])):
        comp = fixed[int(idx)]
        max_idx = int(np.argmax(np.abs(comp)))
        if float(comp[int(max_idx)]) < 0.0:
            fixed[int(idx)] = -comp
    return fixed


def token_times_from_duration(code_frames: int, duration_sec: float) -> np.ndarray:
    frames = int(max(0, int(code_frames)))
    dur = float(max(0.0, float(duration_sec)))
    if frames <= 0 or dur <= 0.0:
        return np.zeros((0,), dtype=np.float32)
    idx = np.arange(frames, dtype=np.float32)
    return ((idx + 0.5) * (dur / float(frames))).astype(np.float32, copy=False)


def grid_times_from_fps(grid_frames: int, fps: float) -> np.ndarray:
    frames = int(max(0, int(grid_frames)))
    fps_eff = float(max(1.0e-6, float(fps)))
    if frames <= 0:
        return np.zeros((0,), dtype=np.float32)
    return ((np.arange(frames, dtype=np.float32) + np.float32(0.5)) / fps_eff).astype(np.float32, copy=False)


def sample_beat_boundaries_from_source(
    *,
    shard: Mapping[str, Any],
    row: Mapping[str, Any],
    duration_sec: float,
) -> np.ndarray:
    duration_eff = float(max(1.0e-6, float(duration_sec)))
    beat_times_raw = shard.get("beat_times_sec")
    if beat_times_raw is None:
        return np.asarray([0.0, float(duration_eff)], dtype=np.float32)
    if torch.is_tensor(beat_times_raw):
        beat_times = beat_times_raw.detach().to(device="cpu", dtype=torch.float32).numpy().reshape(-1)
    else:
        beat_times = np.asarray(beat_times_raw, dtype=np.float32).reshape(-1)
    beat_times = beat_times[np.isfinite(beat_times)]
    boundaries = np.concatenate(([0.0], beat_times.astype(np.float32, copy=False)), dtype=np.float32)
    start_sec = float(row.get("start_sec", 0.0) or 0.0)
    end_sec = float(row.get("end_sec", start_sec + duration_eff) or (start_sec + duration_eff))
    beat_index = int(row.get("beat_index", -1) or -1)
    if row.get("num_beats") is not None:
        num_beats = int(max(1, int(row.get("num_beats", 1) or 1)))
    else:
        beat_index_end = int(row.get("beat_index_end", beat_index) or beat_index)
        num_beats = int(max(1, int(beat_index_end) - int(beat_index) + 1)) if int(beat_index) >= 0 else 1

    if int(beat_index) >= 0 and int(beat_index) + int(num_beats) < int(boundaries.shape[0]):
        sample_boundaries = np.asarray(
            boundaries[int(beat_index) : int(beat_index) + int(num_beats) + 1],
            dtype=np.float32,
        )
    else:
        eps = np.float32(1.0e-4)
        mask = (boundaries >= np.float32(float(start_sec) - float(eps))) & (
            boundaries <= np.float32(float(end_sec) + float(eps))
        )
        sample_boundaries = np.asarray(boundaries[mask], dtype=np.float32)
        sample_boundaries = np.concatenate(
            (
                np.asarray([float(start_sec)], dtype=np.float32),
                sample_boundaries,
                np.asarray([float(end_sec)], dtype=np.float32),
            ),
            dtype=np.float32,
        )

    sample_boundaries = np.clip(
        np.asarray(sample_boundaries, dtype=np.float32) - np.float32(float(start_sec)),
        np.float32(0.0),
        np.float32(float(duration_eff)),
    )
    sample_boundaries = np.sort(sample_boundaries, axis=None).astype(np.float32, copy=False)
    if int(sample_boundaries.size) <= 0:
        return np.asarray([0.0, float(duration_eff)], dtype=np.float32)
    deduped: list[float] = [float(sample_boundaries[0])]
    for value in list(sample_boundaries[1:]):
        if float(value) > float(deduped[-1]) + 1.0e-5:
            deduped.append(float(value))
    if float(deduped[0]) > 1.0e-5:
        deduped.insert(0, 0.0)
    else:
        deduped[0] = 0.0
    if float(deduped[-1]) < float(duration_eff) - 1.0e-5:
        deduped.append(float(duration_eff))
    else:
        deduped[-1] = float(duration_eff)
    if int(len(deduped)) < 2:
        deduped = [0.0, float(duration_eff)]
    return np.asarray(deduped, dtype=np.float32)


def normalize_segment_beat_boundaries_rel(
    boundaries_sec_rel: Sequence[float],
    *,
    expected_num_beats: int,
    duration_sec: float,
    merge_tolerance_sec: float = 1.0e-4,
) -> np.ndarray:
    boundaries = np.asarray(boundaries_sec_rel, dtype=np.float64).reshape(-1)
    expected_count = int(max(1, int(expected_num_beats)) + 1)
    duration = float(max(1.0e-6, float(duration_sec)))
    tol = float(max(1.0e-8, float(merge_tolerance_sec)))
    if int(boundaries.size) <= 0:
        return np.asarray([0.0, duration], dtype=np.float32)

    boundaries = np.clip(boundaries, 0.0, duration)
    merged: list[float] = [0.0]
    for value in list(boundaries):
        v = float(value)
        if float(v) <= float(merged[-1]) + float(tol):
            merged[-1] = max(float(merged[-1]), float(v))
        else:
            merged.append(float(v))
    if float(merged[-1]) < float(duration) - float(tol):
        merged.append(float(duration))
    else:
        merged[-1] = float(duration)
    if int(len(merged)) == int(expected_count):
        return np.asarray(merged, dtype=np.float32)
    raise ValueError(
        f"expected {expected_count} beat boundaries after normalization, got {tuple(np.asarray(merged).shape)} "
        f"from raw boundaries {np.asarray(boundaries_sec_rel, dtype=np.float32).tolist()}"
    )


def stack_family_state_grid(
    *,
    state_vel_ft: np.ndarray,
    onset_vel_ft: np.ndarray,
    onset_count_ft: np.ndarray,
) -> np.ndarray:
    state_vel = np.asarray(state_vel_ft, dtype=np.float32)
    onset_vel = np.asarray(onset_vel_ft, dtype=np.float32)
    onset_count = np.asarray(onset_count_ft, dtype=np.float32)
    if int(state_vel.ndim) != 2:
        raise ValueError(f"state_vel_ft must be [F,T], got {tuple(state_vel.shape)}")
    if tuple(onset_vel.shape) != tuple(state_vel.shape):
        raise ValueError(f"onset_vel_ft must match state_vel_ft, got {tuple(onset_vel.shape)} / {tuple(state_vel.shape)}")
    if tuple(onset_count.shape) != tuple(state_vel.shape):
        raise ValueError(
            f"onset_count_ft must match state_vel_ft, got {tuple(onset_count.shape)} / {tuple(state_vel.shape)}"
        )
    num_families, num_frames = [int(x) for x in list(state_vel.shape)]
    if int(num_families) != int(len(FAMILY_STATE_FAMILY_NAMES)):
        raise ValueError(
            f"expected {len(FAMILY_STATE_FAMILY_NAMES)} families, got {tuple(state_vel.shape)}"
        )
    grid_rows = []
    for family_idx in range(int(num_families)):
        grid_rows.append(state_vel[int(family_idx)])
        grid_rows.append(onset_vel[int(family_idx)])
        grid_rows.append(onset_count[int(family_idx)])
    out = np.stack(grid_rows, axis=0).astype(np.float32, copy=False)
    if tuple(out.shape) != (int(len(FAMILY_STATE_FEATURE_ROW_NAMES)), int(num_frames)):
        raise RuntimeError(f"unexpected stacked family-state grid shape {tuple(out.shape)}")
    return out


def build_segment_beat_boundaries_rel(
    *,
    song_beat_times_sec: Sequence[float],
    beat_index: int,
    num_beats: int,
    start_sec: float,
    end_sec: float,
) -> np.ndarray:
    beat_index = int(beat_index)
    num_beats = int(num_beats)
    start_sec = float(start_sec)
    end_sec = float(end_sec)
    if int(num_beats) <= 0:
        raise ValueError(f"num_beats must be positive, got {num_beats}")
    if not float(end_sec) > float(start_sec):
        raise ValueError(f"end_sec must be greater than start_sec, got {start_sec} -> {end_sec}")

    song_beats = np.asarray(song_beat_times_sec, dtype=np.float64).reshape(-1)
    all_boundaries = np.concatenate((np.asarray([0.0], dtype=np.float64), song_beats), axis=0)
    lo = int(beat_index)
    hi = int(beat_index + num_beats)
    if int(lo) < 0 or int(hi) >= int(all_boundaries.shape[0]):
        raise ValueError(
            f"beat boundary slice [{lo}:{hi}] is out of range for {int(all_boundaries.shape[0])} boundaries"
        )
    boundaries_abs = np.asarray(all_boundaries[int(lo) : int(hi) + 1], dtype=np.float64).copy()
    if int(boundaries_abs.shape[0]) != int(num_beats + 1):
        raise ValueError(
            f"expected {int(num_beats + 1)} boundaries, got shape {tuple(boundaries_abs.shape)}"
        )
    boundaries_abs[0] = float(start_sec)
    boundaries_abs[-1] = float(end_sec)
    boundaries_rel = np.asarray(boundaries_abs - float(start_sec), dtype=np.float32)
    duration_sec = np.float32(float(end_sec) - float(start_sec))
    boundaries_rel[0] = np.float32(0.0)
    boundaries_rel[-1] = duration_sec
    boundaries_rel = np.clip(boundaries_rel, 0.0, float(duration_sec))
    diffs = np.diff(boundaries_rel.astype(np.float64))
    if np.any(diffs <= 0.0):
        raise ValueError(
            f"segment beat boundaries must be strictly increasing, got {boundaries_rel.tolist()}"
        )
    return boundaries_rel.astype(np.float32, copy=False)


def expand_beat_boundaries_to_16th_steps(beat_boundaries_sec_rel: Sequence[float]) -> np.ndarray:
    beat_boundaries = np.asarray(beat_boundaries_sec_rel, dtype=np.float64).reshape(-1)
    if int(beat_boundaries.shape[0]) != 5:
        raise ValueError(f"expected 5 beat boundaries for a 4-beat segment, got {tuple(beat_boundaries.shape)}")
    diffs = np.diff(beat_boundaries)
    if np.any(diffs <= 0.0):
        raise ValueError(f"beat boundaries must be strictly increasing, got {beat_boundaries.tolist()}")
    step_boundaries: list[float] = []
    for beat_idx in range(4):
        beat_start = float(beat_boundaries[int(beat_idx)])
        beat_end = float(beat_boundaries[int(beat_idx) + 1])
        local = np.linspace(beat_start, beat_end, num=5, endpoint=True, dtype=np.float64)
        step_boundaries.extend(local[:-1].tolist())
    step_boundaries.append(float(beat_boundaries[-1]))
    out = np.asarray(step_boundaries, dtype=np.float32)
    if int(out.shape[0]) != 17:
        raise RuntimeError(f"expected 17 step boundaries, got {tuple(out.shape)}")
    return out


def aggregate_grid16_from_conditioning(
    *,
    state_vel_ft: np.ndarray,
    onset_vel_ft: np.ndarray,
    onset_ids_ft: np.ndarray,
    family_onsets_ft: np.ndarray,
    onset_count_ft: np.ndarray,
    step_boundaries_sec_rel: Sequence[float],
    duration_sec: float,
) -> dict[str, np.ndarray]:
    state_vel = np.asarray(state_vel_ft, dtype=np.float32)
    onset_vel = np.asarray(onset_vel_ft, dtype=np.float32)
    onset_ids = np.asarray(onset_ids_ft, dtype=np.int64)
    family_onsets = np.asarray(family_onsets_ft, dtype=np.bool_)
    onset_count = np.asarray(onset_count_ft, dtype=np.uint8)
    step_boundaries = np.asarray(step_boundaries_sec_rel, dtype=np.float64).reshape(-1)
    duration_sec = float(duration_sec)

    if int(state_vel.ndim) != 2:
        raise ValueError(f"state_vel_ft must be [F,T], got {tuple(state_vel.shape)}")
    if tuple(onset_vel.shape) != tuple(state_vel.shape):
        raise ValueError(f"onset_vel_ft must match state_vel_ft, got {tuple(onset_vel.shape)} / {tuple(state_vel.shape)}")
    if tuple(onset_ids.shape) != tuple(state_vel.shape):
        raise ValueError(f"onset_ids_ft must match state_vel_ft, got {tuple(onset_ids.shape)} / {tuple(state_vel.shape)}")
    if tuple(family_onsets.shape) != tuple(state_vel.shape):
        raise ValueError(
            f"family_onsets_ft must match state_vel_ft, got {tuple(family_onsets.shape)} / {tuple(state_vel.shape)}"
        )
    if tuple(onset_count.shape) != tuple(state_vel.shape):
        raise ValueError(
            f"onset_count_ft must match state_vel_ft, got {tuple(onset_count.shape)} / {tuple(state_vel.shape)}"
        )
    if int(step_boundaries.shape[0]) != 17:
        raise ValueError(f"step_boundaries_sec_rel must have 17 values, got {tuple(step_boundaries.shape)}")
    if not float(duration_sec) > 0.0:
        raise ValueError(f"duration_sec must be positive, got {duration_sec}")

    num_families, grid_frames = [int(x) for x in list(state_vel.shape)]
    frame_edges = np.linspace(0.0, float(duration_sec), num=int(grid_frames) + 1, endpoint=True, dtype=np.float64)
    frame_centers = 0.5 * (frame_edges[:-1] + frame_edges[1:])

    grid16_state_vel = np.zeros((int(num_families), 16), dtype=np.float32)
    grid16_onset_vel = np.zeros((int(num_families), 16), dtype=np.float32)
    grid16_onset_count = np.zeros((int(num_families), 16), dtype=np.uint8)
    grid16_onset_ids = np.full((int(num_families), 16), -1, dtype=np.int16)

    for step_idx in range(16):
        step_start = float(step_boundaries[int(step_idx)])
        step_end = float(step_boundaries[int(step_idx) + 1])
        overlap = (frame_edges[1:] > float(step_start)) & (frame_edges[:-1] < float(step_end))
        frame_idx = np.flatnonzero(overlap)
        if int(frame_idx.size) <= 0:
            midpoint = 0.5 * (float(step_start) + float(step_end))
            frame_idx = np.asarray([int(np.argmin(np.abs(frame_centers - float(midpoint))))], dtype=np.int64)

        grid16_state_vel[:, int(step_idx)] = np.max(state_vel[:, frame_idx], axis=1)
        count_sum = onset_count[:, frame_idx].astype(np.int64).sum(axis=1)
        grid16_onset_count[:, int(step_idx)] = np.clip(count_sum, 0, 255).astype(np.uint8, copy=False)

        for family_idx in range(int(num_families)):
            hit_frames = frame_idx[np.flatnonzero(family_onsets[int(family_idx), frame_idx])]
            if int(hit_frames.size) <= 0:
                continue
            hit_vels = onset_vel[int(family_idx), hit_frames]
            sort_key = np.lexsort((hit_frames.astype(np.int64), hit_vels.astype(np.float64)))
            best_frame = int(hit_frames[int(sort_key[-1])])
            grid16_onset_vel[int(family_idx), int(step_idx)] = float(onset_vel[int(family_idx), int(best_frame)])
            grid16_onset_ids[int(family_idx), int(step_idx)] = np.int16(onset_ids[int(family_idx), int(best_frame)])

    return {
        "grid16_state_vel": grid16_state_vel,
        "grid16_onset_vel": grid16_onset_vel,
        "grid16_onset_count": grid16_onset_count,
        "grid16_onset_ids": grid16_onset_ids,
        "step_boundaries_sec_rel": np.asarray(step_boundaries, dtype=np.float32),
    }
