from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from data.diffusion_cache_utils import (
    FAMILY_STATE_FEATURE_ROW_NAMES,
    FAMILY_STATE_FAMILY_NAMES,
    FAMILY_STATE_ID_VOCAB_SIZES,
)


FAMILY_STATE_NUMERIC_COMPONENT_NAMES: tuple[str, ...] = (
    "state_vel",
    "onset_vel",
    "onset_count",
)
FAMILY_STATE_INDEX: dict[str, int] = {
    str(name): idx for idx, name in enumerate(FAMILY_STATE_FAMILY_NAMES)
}
FAMILY_STATE_STATE_VEL_ROW_INDICES: tuple[int, ...] = tuple(
    (int(family_idx) * int(len(FAMILY_STATE_NUMERIC_COMPONENT_NAMES))) + 0
    for family_idx in range(int(len(FAMILY_STATE_FAMILY_NAMES)))
)
FAMILY_STATE_ONSET_VEL_ROW_INDICES: tuple[int, ...] = tuple(
    (int(family_idx) * int(len(FAMILY_STATE_NUMERIC_COMPONENT_NAMES))) + 1
    for family_idx in range(int(len(FAMILY_STATE_FAMILY_NAMES)))
)
FAMILY_STATE_ONSET_COUNT_ROW_INDICES: tuple[int, ...] = tuple(
    (int(family_idx) * int(len(FAMILY_STATE_NUMERIC_COMPONENT_NAMES))) + 2
    for family_idx in range(int(len(FAMILY_STATE_FAMILY_NAMES)))
)
CONDITIONING_CACHE_PROFILE_VERSION = 7

_FAMILY_STATE_PITCH_TO_EVENT: dict[int, tuple[str, int]] = {
    35: ("kick", 0),
    36: ("kick", 0),
    37: ("snare", 2),
    38: ("snare", 0),
    40: ("snare", 1),
    48: ("tom_high", 0),
    50: ("tom_high", 1),
    45: ("tom_mid", 0),
    47: ("tom_mid", 1),
    41: ("tom_floor", 0),
    43: ("tom_floor", 0),
    58: ("tom_floor", 1),
    46: ("hihat", 0),
    26: ("hihat", 1),
    42: ("hihat", 2),
    22: ("hihat", 3),
    44: ("hihat", 4),
    49: ("crash", 0),
    57: ("crash", 0),
    52: ("crash", 1),
    55: ("crash", 1),
    51: ("ride", 0),
    59: ("ride", 1),
    53: ("ride", 2),
}
_FAMILY_STATE_ATTACK_MS: dict[str, float] = {
    "kick": 12.0,
    "snare": 12.0,
    "tom_high": 12.0,
    "tom_mid": 12.0,
    "tom_floor": 12.0,
    "hihat": 20.0,
    "crash": 12.0,
    "ride": 12.0,
}
_FAMILY_STATE_BODY_MS: dict[str, float] = {
    "kick": 120.0,
    "snare": 120.0,
    "tom_high": 140.0,
    "tom_mid": 150.0,
    "tom_floor": 160.0,
    "hihat": 110.0,
    "crash": 220.0,
    "ride": 140.0,
}
_FAMILY_STATE_DECAY_MS: dict[str, float] = {
    "kick": 180.0,
    "snare": 220.0,
    "tom_high": 280.0,
    "tom_mid": 320.0,
    "tom_floor": 360.0,
    "hihat": 520.0,
    "crash": 2200.0,
    "ride": 760.0,
}
_FAMILY_STATE_CARRYOVER_MS: dict[str, float] = {
    "kick": 200.0,
    "snare": 220.0,
    "tom_high": 280.0,
    "tom_mid": 320.0,
    "tom_floor": 360.0,
    "hihat": 520.0,
    "crash": 2200.0,
    "ride": 760.0,
}
_FAMILY_STATE_HIHAT_OPEN_IDS = {0, 1}
_FAMILY_STATE_HIHAT_CLOSED_IDS = {2, 3}
_FAMILY_STATE_HIHAT_OPEN_BODY_MS = 110.0
_FAMILY_STATE_HIHAT_OPEN_DECAY_MS = 520.0
_FAMILY_STATE_HIHAT_CLOSED_BODY_MS = 70.0
_FAMILY_STATE_HIHAT_CLOSED_DECAY_MS = 180.0
_FAMILY_STATE_HIHAT_PEDAL_BODY_MS = 50.0
_FAMILY_STATE_HIHAT_PEDAL_DECAY_MS = 120.0
_FAMILY_STATE_SNARE_GHOST_THRESHOLD = 0.25
_FAMILY_STATE_SNARE_GHOST_BODY_SCALE = 0.45
_FAMILY_STATE_SNARE_GHOST_DECAY_SCALE = 0.45
_FAMILY_STATE_SNARE_GHOST_SALIENCE_SCALE = 0.5
_FAMILY_STATE_KICK_GHOST_THRESHOLD = 0.22
_FAMILY_STATE_KICK_GHOST_BODY_SCALE = 0.50
_FAMILY_STATE_KICK_GHOST_DECAY_SCALE = 0.50
_FAMILY_STATE_KICK_GHOST_SALIENCE_SCALE = 0.6


def build_midi_event_cache(pm: Any) -> dict[str, np.ndarray]:
    pitches: list[int] = []
    times_abs: list[float] = []
    velocities: list[float] = []
    for instrument in getattr(pm, "instruments", []):
        if getattr(instrument, "is_drum", True) is False:
            continue
        for note in getattr(instrument, "notes", []):
            pitches.append(int(note.pitch))
            times_abs.append(float(note.start))
            velocities.append(float(note.velocity) / 127.0)
    if not pitches:
        return {
            "pitches": np.zeros((0,), dtype=np.int16),
            "times_sec": np.zeros((0,), dtype=np.float32),
            "velocities": np.zeros((0,), dtype=np.float32),
        }
    times_arr = np.asarray(times_abs, dtype=np.float32)
    order = np.argsort(times_arr, kind="stable")
    return {
        "pitches": np.asarray(pitches, dtype=np.int16)[order],
        "times_sec": times_arr[order],
        "velocities": np.asarray(velocities, dtype=np.float32)[order],
    }


def _normalize_unit(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0).astype(np.float32, copy=False)


def _compose_family_state_numeric_rows(
    *,
    state_vel_ft: np.ndarray,
    onset_vel_ft: np.ndarray,
    onset_count_ft: np.ndarray,
) -> np.ndarray:
    family_dim = int(len(FAMILY_STATE_FAMILY_NAMES))
    num_frames = int(state_vel_ft.shape[-1]) if int(state_vel_ft.ndim) == 2 else 0
    out = np.zeros((int(len(FAMILY_STATE_FEATURE_ROW_NAMES)), int(num_frames)), dtype=np.float32)
    out[np.asarray(FAMILY_STATE_STATE_VEL_ROW_INDICES, dtype=np.int64), :] = np.asarray(state_vel_ft, dtype=np.float32)
    out[np.asarray(FAMILY_STATE_ONSET_VEL_ROW_INDICES, dtype=np.int64), :] = np.asarray(onset_vel_ft, dtype=np.float32)
    out[np.asarray(FAMILY_STATE_ONSET_COUNT_ROW_INDICES, dtype=np.int64), :] = np.asarray(
        onset_count_ft,
        dtype=np.float32,
    )
    if int(state_vel_ft.shape[0]) != int(family_dim):
        raise ValueError(f"state_vel_ft must have {int(family_dim)} family rows, got {tuple(state_vel_ft.shape)}")
    return out


def _event_frame_index_from_time(*, time_sec: float, duration_sec: float, num_frames: int) -> int:
    if int(num_frames) <= 0:
        return 0
    dur = float(max(1.0e-6, float(duration_sec)))
    upper = float(np.nextafter(np.float32(dur), np.float32(0.0)))
    event_time = float(np.clip(float(time_sec), 0.0, upper))
    frame_pos = (event_time / dur) * float(max(1, int(num_frames)))
    return int(np.clip(np.round(frame_pos), 0.0, float(max(0, int(num_frames) - 1))))


def _family_state_event_from_pitch(pitch: int) -> tuple[str, int] | None:
    event = _FAMILY_STATE_PITCH_TO_EVENT.get(int(pitch))
    if event is None:
        return None
    return str(event[0]), int(event[1])


def _family_state_is_ghost(*, family_name: str, class_id: int, velocity: float) -> bool:
    fam = str(family_name)
    vel = float(np.clip(float(velocity), 0.0, 1.0))
    if str(fam) == "snare" and int(class_id) == 0:
        return bool(vel <= float(_FAMILY_STATE_SNARE_GHOST_THRESHOLD))
    if str(fam) == "kick" and int(class_id) == 0:
        return bool(vel <= float(_FAMILY_STATE_KICK_GHOST_THRESHOLD))
    return False


def _family_state_curve_params(
    *,
    family_name: str,
    class_id: int,
    velocity: float,
) -> dict[str, float]:
    fam = str(family_name)
    class_id_eff = int(class_id)
    vel = float(np.clip(float(velocity), 0.0, 1.0))
    attack_ms = float(_FAMILY_STATE_ATTACK_MS.get(str(fam), 12.0))
    body_ms = float(_FAMILY_STATE_BODY_MS.get(str(fam), 120.0))
    decay_ms = float(_FAMILY_STATE_DECAY_MS.get(str(fam), 180.0))
    carryover_ms = float(_FAMILY_STATE_CARRYOVER_MS.get(str(fam), 0.0))
    sustain_salience_scale = 0.6
    onset_salience_scale = 1.0
    force_replace = False
    if str(fam) == "hihat":
        attack_ms = 20.0
        force_replace = True
        if int(class_id_eff) in _FAMILY_STATE_HIHAT_OPEN_IDS:
            body_ms = float(_FAMILY_STATE_HIHAT_OPEN_BODY_MS)
            decay_ms = float(_FAMILY_STATE_HIHAT_OPEN_DECAY_MS)
            carryover_ms = float(_FAMILY_STATE_HIHAT_OPEN_DECAY_MS)
            sustain_salience_scale = 0.2
        elif int(class_id_eff) in _FAMILY_STATE_HIHAT_CLOSED_IDS:
            body_ms = float(_FAMILY_STATE_HIHAT_CLOSED_BODY_MS)
            decay_ms = float(_FAMILY_STATE_HIHAT_CLOSED_DECAY_MS)
            carryover_ms = 180.0
            sustain_salience_scale = 0.35
        else:
            body_ms = float(_FAMILY_STATE_HIHAT_PEDAL_BODY_MS)
            decay_ms = float(_FAMILY_STATE_HIHAT_PEDAL_DECAY_MS)
            carryover_ms = 180.0
            sustain_salience_scale = 0.35
    elif str(fam) == "crash":
        sustain_salience_scale = 0.35
    elif str(fam) == "ride":
        sustain_salience_scale = 0.4
    ghost = _family_state_is_ghost(family_name=str(fam), class_id=int(class_id_eff), velocity=float(vel))
    salience_scale = 1.0
    if bool(ghost) and str(fam) == "snare":
        body_ms *= float(_FAMILY_STATE_SNARE_GHOST_BODY_SCALE)
        decay_ms *= float(_FAMILY_STATE_SNARE_GHOST_DECAY_SCALE)
        salience_scale = float(_FAMILY_STATE_SNARE_GHOST_SALIENCE_SCALE)
    elif bool(ghost) and str(fam) == "kick":
        body_ms *= float(_FAMILY_STATE_KICK_GHOST_BODY_SCALE)
        decay_ms *= float(_FAMILY_STATE_KICK_GHOST_DECAY_SCALE)
        salience_scale = float(_FAMILY_STATE_KICK_GHOST_SALIENCE_SCALE)
    return {
        "attack_sec": 0.001 * float(max(1.0, float(attack_ms))),
        "body_sec": 0.001 * float(max(float(attack_ms), float(body_ms))),
        "decay_sec": 0.001 * float(max(float(attack_ms), float(decay_ms))),
        "carryover_sec": 0.001 * float(max(0.0, float(carryover_ms))),
        "sustain_salience_scale": float(sustain_salience_scale),
        "onset_salience_scale": float(onset_salience_scale),
        "salience_scale": float(salience_scale),
        "force_replace": float(1.0 if bool(force_replace) else 0.0),
    }


def _render_family_state_curve(
    *,
    state_vel_row_t: np.ndarray,
    onset_vel_row_t: np.ndarray,
    onset_id_row_t: np.ndarray,
    onset_count_row_t: np.ndarray,
    support_row_t: np.ndarray,
    onset_row_t: np.ndarray,
    salience_row_t: np.ndarray,
    class_id: int,
    event_time_sec: float,
    velocity_value: float,
    duration_sec: float,
    num_frames: int,
    attack_sec: float,
    body_sec: float,
    decay_sec: float,
    sustain_salience_scale: float,
    onset_salience_scale: float,
    salience_scale: float,
    emit_onset: bool,
    force_replace: bool = False,
) -> None:
    if int(num_frames) <= 0 or float(duration_sec) <= 1.0e-9 or float(velocity_value) <= 0.0:
        return
    duration_eff = float(max(1.0e-6, float(duration_sec)))
    attack_eff = float(max(1.0e-4, float(attack_sec)))
    body_eff = float(max(float(body_sec), float(attack_eff)))
    decay_eff = float(max(float(decay_sec), float(body_eff)))
    upper = float(np.nextafter(np.float32(duration_eff), np.float32(0.0)))
    start_time_sec = float(max(0.0, float(event_time_sec)))
    end_time_sec = float(min(float(upper), float(event_time_sec) + float(decay_eff)))
    if float(end_time_sec) < 0.0 or float(end_time_sec) < float(start_time_sec):
        return
    start_idx = _event_frame_index_from_time(
        time_sec=float(start_time_sec),
        duration_sec=float(duration_eff),
        num_frames=int(num_frames),
    )
    end_idx = _event_frame_index_from_time(
        time_sec=float(max(float(start_time_sec), float(end_time_sec))),
        duration_sec=float(duration_eff),
        num_frames=int(num_frames),
    )
    frame_times_t = (
        np.arange(int(start_idx), int(end_idx) + 1, dtype=np.float32)
        / max(1.0e-6, float(num_frames) / float(duration_eff))
    ).astype(np.float32, copy=False)
    delta_t = np.maximum(0.0, frame_times_t - np.float32(float(event_time_sec)))
    shape = np.zeros_like(delta_t, dtype=np.float32)
    attack_mask = delta_t < np.float32(float(attack_eff))
    if bool(np.any(attack_mask)):
        shape[attack_mask] = np.asarray(
            np.maximum(delta_t[attack_mask] / np.float32(float(attack_eff)), np.float32(0.15)),
            dtype=np.float32,
        )
    body_mask = (~attack_mask) & (delta_t <= np.float32(float(body_eff)))
    if bool(np.any(body_mask)):
        shape[body_mask] = 1.0
    tail_mask = delta_t > np.float32(float(body_eff))
    if bool(np.any(tail_mask)):
        tau = max(1.0e-6, float(max(float(decay_eff) - float(body_eff), 1.0e-4)) / 4.605170186)
        shape[tail_mask] = np.asarray(
            np.exp(-(delta_t[tail_mask] - np.float32(float(body_eff))) / np.float32(float(tau))),
            dtype=np.float32,
        )
    shape = np.clip(shape, 0.0, 1.0).astype(np.float32, copy=False)
    support_curve = np.asarray(shape, dtype=np.float32)
    vel_curve = np.asarray(float(velocity_value) * support_curve, dtype=np.float32)
    dst = slice(int(start_idx), int(end_idx) + 1)
    current_support = np.asarray(support_row_t[dst], dtype=np.float32)
    if bool(force_replace):
        state_vel_row_t[int(start_idx) :] = 0.0
        support_row_t[int(start_idx) :] = 0.0
        update_mask = np.ones_like(support_curve, dtype=np.bool_)
    else:
        update_mask = support_curve >= (current_support - 1.0e-8)
    if bool(np.any(update_mask)):
        support_row_t[dst] = np.where(update_mask, support_curve, current_support).astype(np.float32, copy=False)
        current_vel = np.asarray(state_vel_row_t[dst], dtype=np.float32)
        state_vel_row_t[dst] = np.where(update_mask, vel_curve, current_vel).astype(np.float32, copy=False)
    if bool(emit_onset) and 0.0 <= float(event_time_sec) <= float(duration_eff):
        onset_idx = _event_frame_index_from_time(
            time_sec=float(event_time_sec),
            duration_sec=float(duration_eff),
            num_frames=int(num_frames),
        )
        onset_row_t[int(onset_idx)] = True
        onset_count_row_t[int(onset_idx)] = np.asarray(
            min(255, int(onset_count_row_t[int(onset_idx)]) + 1),
            dtype=np.uint8,
        )
        onset_vel_row_t[int(onset_idx)] = max(float(onset_vel_row_t[int(onset_idx)]), float(velocity_value))
        onset_id_row_t[int(onset_idx)] = int(class_id)
        support_row_t[int(onset_idx)] = max(float(support_row_t[int(onset_idx)]), 1.0e-3)
        state_vel_row_t[int(onset_idx)] = max(float(state_vel_row_t[int(onset_idx)]), float(velocity_value))
        salience_row_t[int(onset_idx)] = max(
            float(salience_row_t[int(onset_idx)]),
            float(velocity_value) * float(onset_salience_scale) * float(salience_scale),
        )
    salience_curve = np.asarray(
        support_curve * float(sustain_salience_scale) * float(salience_scale),
        dtype=np.float32,
    )
    salience_row_t[dst] = np.maximum(
        np.asarray(salience_row_t[dst], dtype=np.float32),
        salience_curve,
    ).astype(np.float32, copy=False)


def render_midi_family_state_grid(
    *,
    midi_events: Mapping[str, np.ndarray],
    start_sec: float,
    end_sec: float,
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    duration_sec = float(max(1.0e-6, float(end_sec) - float(start_sec)))
    num_frames_eff = int(max(0, int(num_frames)))
    family_dim = int(len(FAMILY_STATE_FAMILY_NAMES))
    state_vel = np.zeros((int(family_dim), int(num_frames_eff)), dtype=np.float32)
    onset_vel = np.zeros((int(family_dim), int(num_frames_eff)), dtype=np.float32)
    onset_ids = np.full((int(family_dim), int(num_frames_eff)), -1, dtype=np.int64)
    family_onsets = np.zeros((int(family_dim), int(num_frames_eff)), dtype=np.bool_)
    family_onset_count = np.zeros((int(family_dim), int(num_frames_eff)), dtype=np.uint8)
    support = np.zeros((int(family_dim), int(num_frames_eff)), dtype=np.float32)
    salience = np.zeros((int(family_dim), int(num_frames_eff)), dtype=np.float32)
    if int(num_frames_eff) <= 0:
        return (
            _compose_family_state_numeric_rows(
                state_vel_ft=state_vel,
                onset_vel_ft=onset_vel,
                onset_count_ft=family_onset_count.astype(np.float32, copy=False),
            ),
            onset_ids,
            family_onsets,
            family_onset_count,
            support,
            _normalize_unit(salience),
        )

    times_abs = np.asarray(midi_events.get("times_sec", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    pitches_all = np.asarray(midi_events.get("pitches", np.zeros((0,), dtype=np.int16)), dtype=np.int64).reshape(-1)
    velocities_all = np.asarray(
        midi_events.get("velocities", np.zeros((0,), dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1)
    if int(times_abs.size) <= 0:
        return (
            _compose_family_state_numeric_rows(
                state_vel_ft=state_vel,
                onset_vel_ft=onset_vel,
                onset_count_ft=family_onset_count.astype(np.float32, copy=False),
            ),
            onset_ids,
            family_onsets,
            family_onset_count,
            support,
            _normalize_unit(salience),
        )

    max_lookback_sec = 0.001 * float(max(_FAMILY_STATE_CARRYOVER_MS.values()))
    lo = int(np.searchsorted(times_abs, np.float32(float(start_sec) - float(max_lookback_sec)), side="left"))
    hi = int(np.searchsorted(times_abs, np.float32(float(end_sec)), side="right"))
    if int(hi) <= int(lo):
        return (
            _compose_family_state_numeric_rows(
                state_vel_ft=state_vel,
                onset_vel_ft=onset_vel,
                onset_count_ft=family_onset_count.astype(np.float32, copy=False),
            ),
            onset_ids,
            family_onsets,
            family_onset_count,
            support,
            _normalize_unit(salience),
        )

    family_events: list[list[dict[str, Any]]] = [[] for _ in range(int(family_dim))]
    for pitch, time_abs, velocity in zip(
        pitches_all[lo:hi].tolist(),
        times_abs[lo:hi].tolist(),
        velocities_all[lo:hi].tolist(),
    ):
        event = _family_state_event_from_pitch(int(pitch))
        if event is None:
            continue
        family_name, class_id = event
        params = _family_state_curve_params(
            family_name=str(family_name),
            class_id=int(class_id),
            velocity=float(velocity),
        )
        time_rel = float(time_abs) - float(start_sec)
        if float(time_rel) < 0.0 and (-float(time_rel)) > float(params["carryover_sec"]):
            continue
        family_events[int(FAMILY_STATE_INDEX[str(family_name)])].append(
            {
                "class_id": int(class_id),
                "time_abs": float(time_abs),
                "time_rel": float(time_rel),
                "velocity_value": float(np.clip(float(velocity), 0.0, 1.0)),
                "emit_onset": bool(float(time_rel) >= 0.0),
                **params,
            }
        )

    for family_idx, events in enumerate(family_events):
        for event in sorted(events, key=lambda item: float(item["time_abs"])):
            _render_family_state_curve(
                state_vel_row_t=state_vel[int(family_idx)],
                onset_vel_row_t=onset_vel[int(family_idx)],
                onset_id_row_t=onset_ids[int(family_idx)],
                onset_count_row_t=family_onset_count[int(family_idx)],
                support_row_t=support[int(family_idx)],
                onset_row_t=family_onsets[int(family_idx)],
                salience_row_t=salience[int(family_idx)],
                class_id=int(event["class_id"]),
                event_time_sec=float(event["time_rel"]),
                velocity_value=float(event["velocity_value"]),
                duration_sec=float(duration_sec),
                num_frames=int(num_frames_eff),
                attack_sec=float(event["attack_sec"]),
                body_sec=float(event["body_sec"]),
                decay_sec=float(event["decay_sec"]),
                sustain_salience_scale=float(event["sustain_salience_scale"]),
                onset_salience_scale=float(event["onset_salience_scale"]),
                salience_scale=float(event["salience_scale"]),
                emit_onset=bool(event["emit_onset"]),
                force_replace=bool(int(event["force_replace"])),
            )

    return (
        _compose_family_state_numeric_rows(
            state_vel_ft=_normalize_unit(state_vel),
            onset_vel_ft=_normalize_unit(onset_vel),
            onset_count_ft=np.asarray(family_onset_count, dtype=np.float32),
        ),
        np.asarray(onset_ids, dtype=np.int64),
        np.asarray(family_onsets, dtype=np.bool_),
        np.asarray(family_onset_count, dtype=np.uint8),
        _normalize_unit(support),
        _normalize_unit(salience),
    )
