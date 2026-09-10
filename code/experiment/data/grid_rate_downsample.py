"""Load-time grid-rate decimation for the grid-resolution ablation.

The cache is rendered once at 250 Hz. Coarser conditioning rates are obtained by
decimating that rendering here, at __getitem__ time, so the ablation needs no
extra caches and leaves the audio/target side byte-identical across arms.

Frames are partitioned into contiguous groups by target frame count, and each
channel is reduced with the rule that preserves its meaning:

  * state velocity  -> max   (a sustained state stays present)
  * onset velocity  -> max   (a hit keeps its velocity, never averaged away)
  * onset count     -> sum   (two hits merging into one frame is the effect
                              under study, so counts must accumulate)
  * onset flags     -> any
  * class ids       -> value at the loudest onset in the group, else the state

Upsampling is refused: timing precision the 250 Hz raster discarded cannot be
recovered by interpolation, so a finer rate must be re-rendered from MIDI.
"""

from __future__ import annotations

import numpy as np
import torch

# grid_ft rows are [state_vel, onset_vel, onset_count] interleaved per family,
# matching stack_family_state_grid in diffusion_cache_utils.
ROWS_PER_FAMILY = 3
STATE_VEL_OFFSET = 0
ONSET_VEL_OFFSET = 1
ONSET_COUNT_OFFSET = 2


def target_frame_count(*, grid_num_frames: int, duration_sec: float, target_rate_hz: float) -> int:
    """Frame count for target_rate_hz, never exceeding what the cache holds."""
    frames = int(max(0, int(grid_num_frames)))
    if frames <= 0:
        return 0
    duration = float(max(0.0, float(duration_sec)))
    rate = float(target_rate_hz)
    if not rate > 0.0 or not duration > 0.0:
        return frames
    return int(min(int(frames), max(1, int(round(duration * rate)))))


def _group_bounds(num_frames: int, num_groups: int) -> np.ndarray:
    """Contiguous group edges spreading num_frames over num_groups."""
    return np.linspace(0, int(num_frames), num=int(num_groups) + 1).round().astype(np.int64)


def downsample_grid_payload(
    *,
    grid_ft: torch.Tensor,
    grid_ids_ft: torch.Tensor,
    family_onsets_ft: torch.Tensor,
    family_onset_count_ft: torch.Tensor,
    grid_num_frames: int,
    duration_sec: float,
    target_rate_hz: float,
) -> dict[str, object]:
    """Decimate one cached example to target_rate_hz.

    Returns the replacement tensors plus the new frame count / effective rate.
    A target rate at or above the cached rate is a no-op returning the inputs
    unchanged, so the source-rate arm goes through this same code path.
    """
    frames = int(grid_num_frames)
    out_frames = target_frame_count(
        grid_num_frames=frames,
        duration_sec=duration_sec,
        target_rate_hz=target_rate_hz,
    )
    duration = float(max(1.0e-6, float(duration_sec)))

    if int(out_frames) >= int(frames) or int(frames) <= 0:
        # No-op: identical tensors, so the source-rate arm is bit-identical to
        # bypassing this module entirely.
        return {
            "grid_ft": grid_ft,
            "grid_ids_ft": grid_ids_ft,
            "family_onsets_ft": family_onsets_ft,
            "family_onset_count_ft": family_onset_count_ft,
            "grid_num_frames": int(frames),
            "grid_frame_rate": float(frames) / duration,
        }

    bounds = _group_bounds(int(frames), int(out_frames))
    num_families = int(family_onsets_ft.shape[0])

    grid_out = torch.zeros((int(grid_ft.shape[0]), int(out_frames)), dtype=grid_ft.dtype)
    ids_out = torch.zeros((int(grid_ids_ft.shape[0]), int(out_frames)), dtype=grid_ids_ft.dtype)
    onsets_out = torch.zeros((num_families, int(out_frames)), dtype=family_onsets_ft.dtype)
    counts_out = torch.zeros((num_families, int(out_frames)), dtype=torch.int64)

    onset_count_i64 = family_onset_count_ft.to(dtype=torch.int64)

    for out_idx in range(int(out_frames)):
        start = int(bounds[int(out_idx)])
        stop = int(max(int(start) + 1, int(bounds[int(out_idx) + 1])))

        window_grid = grid_ft[:, int(start) : int(stop)]
        # Velocities take the max so a hit is never diluted by neighbouring
        # silence; counts sum so merged hits stay visible to the loss.
        grid_out[:, int(out_idx)] = window_grid.max(dim=1).values
        for family_idx in range(int(num_families)):
            count_row = int(ONSET_COUNT_OFFSET + ROWS_PER_FAMILY * int(family_idx))
            if int(count_row) < int(grid_ft.shape[0]):
                grid_out[int(count_row), int(out_idx)] = window_grid[int(count_row)].sum()

        onsets_out[:, int(out_idx)] = family_onsets_ft[:, int(start) : int(stop)].any(dim=1)
        counts_out[:, int(out_idx)] = onset_count_i64[:, int(start) : int(stop)].sum(dim=1)

        # Ids follow the loudest onset in the group; with no onset the group is
        # a sustained state, so carry the first frame's id.
        window_onset_vel = grid_ft[ONSET_VEL_OFFSET::ROWS_PER_FAMILY, int(start) : int(stop)]
        for family_idx in range(int(ids_out.shape[0])):
            if int(family_idx) < int(window_onset_vel.shape[0]):
                vel_row = window_onset_vel[int(family_idx)]
                if float(vel_row.max()) > 0.0:
                    pick = int(start) + int(torch.argmax(vel_row).item())
                else:
                    pick = int(start)
            else:
                pick = int(start)
            ids_out[int(family_idx), int(out_idx)] = grid_ids_ft[int(family_idx), int(pick)]

    count_dtype = family_onset_count_ft.dtype
    if count_dtype == torch.uint8:
        counts_out = counts_out.clamp_(0, 255)

    return {
        "grid_ft": grid_out.contiguous(),
        "grid_ids_ft": ids_out.contiguous(),
        "family_onsets_ft": onsets_out.contiguous(),
        "family_onset_count_ft": counts_out.to(dtype=count_dtype).contiguous(),
        "grid_num_frames": int(out_frames),
        "grid_frame_rate": float(out_frames) / duration,
    }


def uniform_grid_times(num_frames: int, duration_sec: float) -> torch.Tensor:
    """Frame-centre times, matching _uniform_frame_times_from_duration (bpm geometry)."""
    frames = int(max(0, int(num_frames)))
    duration = float(max(0.0, float(duration_sec)))
    if frames <= 0 or not duration > 0.0:
        return torch.zeros((0,), dtype=torch.float32)
    idx = torch.arange(int(frames), dtype=torch.float32)
    return ((idx + 0.5) * (float(duration) / float(frames))).contiguous()
