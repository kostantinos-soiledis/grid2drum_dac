#!/usr/bin/env python3
"""Build a beat-level source cache with generic codec tokenization."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT.parent.parent
RUNS_ROOT = PACKAGE_ROOT / "runs"
RESULTS_ROOT = PACKAGE_ROOT / "results"


def _preload_stdlib_inspect() -> None:
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

from dataclasses import dataclass

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime_compat import apply_runtime_compat

apply_runtime_compat()

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

try:  # pragma: no cover
    import pretty_midi
except Exception:  # pragma: no cover
    pretty_midi = None  # type: ignore

try:  # pragma: no cover
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

from data.audio_codec_utils import (
    DEFAULT_CODEC_MODEL_ID,
    DEFAULT_DAC_CODEC_MODEL_ID,
    DEFAULT_ENCODEC_BANDWIDTH,
    AudioCodecMetadata,
    encode_audio_chunks_to_codes,
    infer_codec_family,
    load_audio_codec_model,
    resolve_device,
)
from data.diffusion_cache_utils import (
    FAMILY_STATE_FEATURE_ROW_NAMES,
    FAMILY_STATE_FAMILY_NAMES,
    FAMILY_STATE_ID_VOCAB_SIZES,
)
from data.family_state_cache_utils import (
    CONDITIONING_CACHE_PROFILE_VERSION,
    build_midi_event_cache,
    render_midi_family_state_grid,
)


LEGACY_CACHE_FORMAT = "gmd-beat-drumgrid-encodec-v9"
GENERIC_CACHE_FORMAT = "gmd-beat-drumgrid-codec-v10"
DEFAULT_ENCODE_BATCH_SIZE = 64
DEFAULT_MIN_BEAT_SAMPLES = 256
DEFAULT_ONSET_WINDOW_MS = 10.0
DEFAULT_ONSET_HOP_MS = 5.0
DEFAULT_ONSET_ABS_THRESHOLD = 0.0025
DEFAULT_ONSET_REL_QUANTILE = 0.90
DEFAULT_ONSET_REL_SCALE = 0.6
DEFAULT_BOUNDARY_HIT_WINDOW_MS = 60.0
DEFAULT_CODES_PAD_VALUE = -1
DEFAULT_TARGET_AUDIO_CONTEXT_MS = 20.0
DEFAULT_PREP_NUM_WORKERS = 0
CACHED_AUDIO_SAMPLE_RATE_32K = 32000
TARGET_AUDIO_SAMPLE_RATE_44K = 44100
MADMOM_AUDIO_SAMPLE_RATE = 44100
DEFAULT_MADMOM_BEAT_FPS = 100
DEFAULT_MADMOM_CORRECT = False
MADMOM_MIN_BPM = 40.0
MADMOM_MAX_BPM = 320.0
DEFAULT_MADMOM_PYTHON = os.environ.get("MADMOM_PYTHON", sys.executable)
RESAMPLE_METHOD = "sinc_interp_kaiser"
RESAMPLE_LOWPASS_FILTER_WIDTH = 64
RESAMPLE_ROLLOFF = 0.9475937167399596
RESAMPLE_BETA = 14.769656459379492


@dataclass(frozen=True)
class SongPrepTask:
    source_root: str
    dataset_root: str
    manifest_index: int
    manifest_rec: dict[str, Any]
    which_audio: str
    beat_times_from: str
    codec_sample_rate: int
    madmom_beat_fps: int
    madmom_correct: bool
    madmom_python: str
    keep_empty_grid_beats: bool
    keep_boundary_mismatch_beats: bool
    boundary_hit_window_ms: float
    onset_window_ms: float
    onset_hop_ms: float
    onset_abs_threshold: float
    onset_rel_quantile: float
    onset_rel_scale: float
    min_beat_samples: int
    beats_per_sample: int
    beat_hop: int
    include_waveform: bool = True


def _progress(iterable: Any, *, desc: str, total: int | None = None) -> Any:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"unsupported json value: {type(value).__name__}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_idx, raw_line in enumerate(handle):
            text = str(raw_line).strip()
            if not text:
                continue
            row = dict(json.loads(text))
            row["_manifest_index"] = int(line_idx)
            rows.append(row)
    return rows


def _parse_indices(text: str | None) -> list[int] | None:
    if text is None:
        return None
    items = [chunk.strip() for chunk in str(text).split(",") if chunk.strip()]
    return [int(item) for item in items] if items else None


def _select_manifest_rows(
    rows: Sequence[dict[str, Any]],
    *,
    start_index: int,
    limit: int | None,
    indices: Sequence[int] | None,
    split: str | None,
) -> list[tuple[int, dict[str, Any]]]:
    if indices is not None:
        selected = [(int(idx), dict(rows[int(idx)])) for idx in indices]
    else:
        lo = max(0, int(start_index))
        hi = len(rows) if limit is None else min(len(rows), lo + max(0, int(limit)))
        selected = [(idx, dict(rows[idx])) for idx in range(lo, hi)]
    if split is None:
        return selected
    split_eff = str(split).strip().lower()
    return [
        (idx, row)
        for idx, row in selected
        if str(row.get("split", "")).strip().lower() == split_eff
    ]


def _dedupe_selected_manifest_rows_by_source_row_id(
    rows: Sequence[tuple[int, dict[str, Any]]],
) -> list[tuple[int, dict[str, Any]]]:
    out: list[tuple[int, dict[str, Any]]] = []
    seen: set[str] = set()
    for manifest_index, row in rows:
        key = str(
            row.get("source_row_id")
            or row.get("source_id")
            or row.get("id")
            or f"manifest_index_{int(manifest_index)}"
        )
        if key in seen:
            continue
        seen.add(key)
        out.append((int(manifest_index), dict(row)))
    return out


def _iter_prefetched_song_preps(
    tasks: Sequence[SongPrepTask],
    *,
    max_workers: int,
) -> Iterable[dict[str, Any]]:
    if not tasks:
        return []
    worker_count = int(max(1, int(max_workers)))
    if int(worker_count) <= 1:
        return (_precompute_song_task(task) for task in tasks)

    def _generator() -> Iterable[dict[str, Any]]:
        task_list = list(tasks)
        next_submit = 0
        next_yield = 0
        try:
            with ProcessPoolExecutor(
                max_workers=int(worker_count),
                mp_context=mp.get_context("spawn"),
            ) as executor:
                pending: dict[Future, int] = {}
                ready: dict[int, dict[str, Any]] = {}

                while next_submit < min(int(worker_count), len(task_list)):
                    future = executor.submit(_precompute_song_task, task_list[int(next_submit)])
                    pending[future] = int(next_submit)
                    next_submit += 1

                while pending:
                    done, _ = wait(set(pending.keys()), return_when=FIRST_COMPLETED)
                    for future in done:
                        task_index = pending.pop(future)
                        ready[int(task_index)] = dict(future.result())
                        if next_submit < len(task_list):
                            next_future = executor.submit(_precompute_song_task, task_list[int(next_submit)])
                            pending[next_future] = int(next_submit)
                            next_submit += 1

                    while next_yield in ready:
                        yield ready.pop(next_yield)
                        next_yield += 1
        except BrokenProcessPool:
            print(
                "[warn] prep worker pool terminated abruptly; "
                f"falling back to in-process prep from task {int(next_yield)}. "
                "For reproducible debugging, rerun with --prep-num-workers 0.",
                file=sys.stderr,
            )
            for task in task_list[int(next_yield) :]:
                yield _precompute_song_task(task)

    return _generator()


def _safe_rel(path: Path, root: Path) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def _mix_to_mono(wav: torch.Tensor) -> torch.Tensor:
    audio = torch.as_tensor(wav, dtype=torch.float32).detach().cpu()
    if int(audio.dim()) == 1:
        audio = audio.unsqueeze(0)
    if int(audio.dim()) != 2:
        raise RuntimeError(f"expected waveform [C,T], got {tuple(audio.shape)}")
    if int(audio.shape[0]) > 1:
        audio = audio.mean(dim=0, keepdim=True)
    return audio.contiguous()


def _resample_waveform_hq(wav: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
    audio = torch.as_tensor(wav, dtype=torch.float32)
    if int(src_sr) == int(dst_sr):
        return audio.contiguous()
    return torchaudio.functional.resample(
        audio,
        int(src_sr),
        int(dst_sr),
        resampling_method=RESAMPLE_METHOD,
        lowpass_filter_width=RESAMPLE_LOWPASS_FILTER_WIDTH,
        rolloff=RESAMPLE_ROLLOFF,
        beta=RESAMPLE_BETA,
    ).contiguous()


def _load_audio_mono(path: Path, *, sample_rate: int | None = None, mixdown: bool = True) -> tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(str(path))
    if bool(mixdown):
        wav = _mix_to_mono(wav)
    else:
        wav = torch.as_tensor(wav, dtype=torch.float32)
        if int(wav.dim()) != 2:
            raise RuntimeError(f"expected waveform [C,T], got {tuple(wav.shape)}")
        wav = wav[:1].contiguous()
    if sample_rate is not None and int(sr) != int(sample_rate):
        wav = _resample_waveform_hq(wav, int(sr), int(sample_rate))
        sr = int(sample_rate)
    return wav.contiguous(), int(sr)


def _resolve_manifest_pt_path(source_root: Path, manifest_rec: Mapping[str, Any]) -> Path:
    pt_rel = str(manifest_rec.get("pt") or "").strip()
    if not pt_rel:
        raise KeyError("manifest row is missing pt")
    pt_path = Path(pt_rel)
    if not pt_path.is_absolute():
        pt_path = source_root / pt_path
    if not pt_path.is_file():
        raise FileNotFoundError(f"PT file not found: {pt_path}")
    return pt_path.resolve()


def _resolve_payload_sample_rate(payload: Mapping[str, Any], *, fallback_sample_rate: int) -> int:
    meta = dict(payload.get("meta") or payload.get("source_meta") or payload.get("source_manifest") or {})
    sample_rate = int(payload.get("sample_rate", meta.get("sample_rate", fallback_sample_rate)) or fallback_sample_rate)
    return int(sample_rate if sample_rate > 0 else fallback_sample_rate)


def _source_meta_with_actual_audio(
    source_meta: Mapping[str, Any],
    *,
    sample_rate: int,
    num_samples: int,
) -> dict[str, Any]:
    out = dict(source_meta)
    old_sample_rate = out.get("sample_rate")
    old_sample_count = out.get("sample_count")
    if old_sample_rate is not None and int(old_sample_rate) != int(sample_rate):
        out.setdefault("upstream_sample_rate", int(old_sample_rate))
    if old_sample_count is not None and int(old_sample_count) != int(num_samples):
        out.setdefault("upstream_sample_count", int(old_sample_count))
    out["sample_rate"] = int(sample_rate)
    out["sample_count"] = int(num_samples)
    out["source_audio_sample_rate"] = int(sample_rate)
    out["source_audio_num_samples"] = int(num_samples)
    out["source_audio_duration_sec"] = float(num_samples) / float(sample_rate)
    return out


def _payload_audio_to_mono(
    payload: Mapping[str, Any],
    *,
    key: str,
    target_sample_rate: int,
) -> torch.Tensor | None:
    wav = payload.get(str(key))
    if not torch.is_tensor(wav):
        return None
    audio = torch.as_tensor(wav, dtype=torch.float32).detach().cpu()
    if int(audio.dim()) == 1:
        audio = audio.unsqueeze(0)
    if int(audio.dim()) != 2:
        return None
    audio = _mix_to_mono(audio)
    sample_rate = _resolve_payload_sample_rate(payload, fallback_sample_rate=int(target_sample_rate))
    if int(sample_rate) != int(target_sample_rate):
        audio = _resample_waveform_hq(audio, int(sample_rate), int(target_sample_rate))
    return audio.contiguous()


def _resolve_audio_relpath(
    manifest_rec: Mapping[str, Any],
    *,
    which_audio: str,
) -> str | None:
    if str(which_audio) == "orig":
        return (
            manifest_rec.get("target_wav")
            or manifest_rec.get("orig_wav")
            or manifest_rec.get("audio_file")
        )
    return (
        manifest_rec.get("rendered_wav")
        or manifest_rec.get("rend_wav")
        or manifest_rec.get("target_wav")
        or manifest_rec.get("orig_wav")
    )


def _load_source_audio_mono(
    payload: Mapping[str, Any],
    *,
    source_root: Path,
    dataset_root: Path | None = None,
    source_meta: Mapping[str, Any] | None = None,
    manifest_rec: Mapping[str, Any],
    which_audio: str,
    target_sample_rate: int,
    prefer_native_audio: bool = False,
) -> tuple[torch.Tensor, int]:
    source_meta_eff = dict(source_meta or {})
    if bool(prefer_native_audio) and str(which_audio) == "orig":
        native_rel = str(source_meta_eff.get("audio_file") or manifest_rec.get("audio_file") or "").strip()
        if native_rel:
            native_path = Path(native_rel)
            if not native_path.is_absolute():
                native_path = (dataset_root if dataset_root is not None else source_root) / native_path
            if native_path.is_file():
                return _load_audio_mono(native_path, sample_rate=int(target_sample_rate))
            raise FileNotFoundError(f"native source audio not found: {native_path}")
        raise KeyError(f"native source audio path missing for {manifest_rec.get('source_id')}")

    payload_keys = (
        ("target_audio", "target_audio_32k")
        if str(which_audio) == "orig"
        else ("rendered_audio", "rendered_audio_32k")
    )
    for key in payload_keys:
        cached = _payload_audio_to_mono(
            payload,
            key=str(key),
            target_sample_rate=int(target_sample_rate),
        )
        if cached is not None:
            return cached, int(target_sample_rate)

    rel = _resolve_audio_relpath(manifest_rec, which_audio=str(which_audio))
    if rel is None:
        raise KeyError(
            f"could not resolve {which_audio!r} audio from payload or manifest for {manifest_rec.get('source_id')}"
        )
    path = Path(str(rel))
    if not path.is_absolute():
        path = source_root / path
    return _load_audio_mono(path, sample_rate=int(target_sample_rate))


def _sanitize_beat_times(beat_times: Any, duration_sec: float) -> np.ndarray:
    bt = np.asarray(beat_times, dtype=np.float64).reshape(-1)
    bt = bt[np.isfinite(bt) & (bt > 0.0)]
    bt = bt[bt <= float(duration_sec) + 1.0e-6]
    if int(bt.size) <= 0:
        return np.zeros((0,), dtype=np.float64)
    deduped: list[float] = [float(bt[0])]
    for value in list(bt[1:]):
        if float(value) > float(deduped[-1]) + 1.0e-6:
            deduped.append(float(value))
    return np.asarray(deduped, dtype=np.float64)


def _apply_madmom_runtime_compat() -> None:
    import collections
    import collections.abc as cabc

    for name in ("MutableSequence", "MutableMapping", "MutableSet"):
        if not hasattr(collections, name) and hasattr(cabc, name):
            setattr(collections, name, getattr(cabc, name))

    for name, val in {
        "float": float,
        "int": int,
        "bool": bool,
        "object": object,
        "complex": complex,
        "str": str,
    }.items():
        if name not in np.__dict__:
            setattr(np, name, val)


def _suppress_madmom_warnings() -> None:
    warnings.filterwarnings("ignore", message=r"pkg_resources is deprecated as an API.*", category=UserWarning)


@lru_cache(maxsize=1)
def _madmom_imports_available() -> bool:
    try:
        _apply_madmom_runtime_compat()
        _suppress_madmom_warnings()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor  # noqa: F401

        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def _get_madmom_rnn_processor() -> Any:
    _apply_madmom_runtime_compat()
    _suppress_madmom_warnings()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from madmom.features.beats import RNNBeatProcessor

    return RNNBeatProcessor()


@lru_cache(maxsize=16)
def _get_madmom_dbn_processor(fps: int, correct: bool) -> Any:
    _apply_madmom_runtime_compat()
    _suppress_madmom_warnings()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from madmom.features.beats import DBNBeatTrackingProcessor

    return DBNBeatTrackingProcessor(
        fps=int(fps),
        min_bpm=MADMOM_MIN_BPM,
        max_bpm=MADMOM_MAX_BPM,
        correct=bool(correct),
    )


def _madmom_detect_beats_local(path_str: str, *, fps: int, correct: bool) -> tuple[float, ...]:
    wav, sr = torchaudio.load(str(path_str))
    wav = _mix_to_mono(wav)
    if int(sr) != int(MADMOM_AUDIO_SAMPLE_RATE):
        wav = _resample_waveform_hq(wav, int(sr), int(MADMOM_AUDIO_SAMPLE_RATE))
    arr = wav.squeeze(0).numpy().astype("float32", copy=False)
    activations = _get_madmom_rnn_processor()(arr)
    beats = _get_madmom_dbn_processor(int(fps), bool(correct))(activations)
    return tuple(float(x) for x in beats.tolist())


@lru_cache(maxsize=128)
def _madmom_detect_beats_cached(
    path_str: str,
    *,
    madmom_python: str = DEFAULT_MADMOM_PYTHON,
    fps: int = DEFAULT_MADMOM_BEAT_FPS,
    correct: bool = DEFAULT_MADMOM_CORRECT,
) -> tuple[float, ...]:
    if _madmom_imports_available():
        return _madmom_detect_beats_local(str(path_str), fps=int(fps), correct=bool(correct))

    script = f"""import collections, collections.abc as cabc
for name in ("MutableSequence", "MutableMapping", "MutableSet"):
    if not hasattr(collections, name) and hasattr(cabc, name):
        setattr(collections, name, getattr(cabc, name))
import json
import numpy as np
for name, val in {{"float": float, "int": int, "bool": bool, "object": object, "complex": complex, "str": str}}.items():
    if name not in np.__dict__:
        setattr(np, name, val)
import sys
import torchaudio
from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
path = sys.argv[1]
fps = int(sys.argv[2])
correct = bool(int(sys.argv[3]))
wav, sr = torchaudio.load(path)
if wav.shape[0] > 1:
    wav = wav.mean(dim=0, keepdim=True)
if sr != {MADMOM_AUDIO_SAMPLE_RATE}:
    wav = torchaudio.transforms.Resample(
        sr,
        {MADMOM_AUDIO_SAMPLE_RATE},
        resampling_method="{RESAMPLE_METHOD}",
        lowpass_filter_width={RESAMPLE_LOWPASS_FILTER_WIDTH},
        rolloff={RESAMPLE_ROLLOFF},
        beta={RESAMPLE_BETA},
    )(wav)
arr = wav.squeeze(0).numpy().astype("float32", copy=False)
activations = RNNBeatProcessor()(arr)
beats = DBNBeatTrackingProcessor(
    fps=fps,
    min_bpm={MADMOM_MIN_BPM},
    max_bpm={MADMOM_MAX_BPM},
    correct=correct,
)(activations)
print(json.dumps([float(x) for x in beats.tolist()]))"""
    cmd = [str(madmom_python), "-c", script, str(path_str), str(int(fps)), "1" if bool(correct) else "0"]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"madmom helper python was not found at {madmom_python!r}. "
            "Set MADMOM_PYTHON to a valid interpreter."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(f"madmom beat detection failed for {path_str}: {stderr or exc}") from exc
    stdout = (proc.stdout or "").strip()
    if not stdout:
        return tuple()
    return tuple(float(x) for x in json.loads(stdout))


def _detect_madmom_beats_from_wav(
    wav_1cn: torch.Tensor,
    sr: int,
    *,
    fps: int,
    correct: bool,
    madmom_python: str = DEFAULT_MADMOM_PYTHON,
) -> np.ndarray:
    wav = torch.as_tensor(wav_1cn, dtype=torch.float32).detach().cpu().contiguous()
    if int(wav.dim()) != 2 or int(wav.shape[0]) != 1:
        raise RuntimeError(f"expected mono audio [1,T], got {tuple(wav.shape)}")
    if _madmom_imports_available():
        wav_local = wav
        if int(sr) != int(MADMOM_AUDIO_SAMPLE_RATE):
            wav_local = _resample_waveform_hq(wav_local, int(sr), int(MADMOM_AUDIO_SAMPLE_RATE))
        arr = wav_local.squeeze(0).cpu().numpy().astype("float32", copy=False)
        activations = _get_madmom_rnn_processor()(arr)
        beats = _get_madmom_dbn_processor(int(fps), bool(correct))(activations)
        return np.asarray(beats, dtype=np.float64)
    fd, tmp_path_str = tempfile.mkstemp(suffix=".wav", prefix="drum_rendering_source_cache_")
    os.close(fd)
    tmp_path = Path(tmp_path_str)
    try:
        torchaudio.save(str(tmp_path), wav, int(sr))
        beats = _madmom_detect_beats_cached(
            str(tmp_path.resolve()),
            madmom_python=str(madmom_python),
            fps=int(fps),
            correct=bool(correct),
        )
        return np.asarray(beats, dtype=np.float64)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def _load_precomputed_beat_times(
    payload: Mapping[str, Any],
    manifest_rec: Mapping[str, Any],
    *,
    which_audio: str,
    duration_sec: float,
    madmom_beat_fps: int | None = None,
    madmom_correct: bool | None = None,
) -> np.ndarray | None:
    audio_analysis = dict(payload.get("audio_analysis") or {})
    variant = dict(audio_analysis.get(str(which_audio)) or {})
    madmom_payload = dict(variant.get("madmom") or {})
    if madmom_payload and madmom_beat_fps is not None and int(madmom_payload.get("fps", -1)) != int(madmom_beat_fps):
        madmom_payload = {}
    if madmom_payload and madmom_correct is not None and bool(madmom_payload.get("correct", not madmom_correct)) != bool(madmom_correct):
        madmom_payload = {}
    for candidate in (
        madmom_payload.get("beat_times_sec"),
        madmom_payload.get("raw_beat_times_sec"),
        variant.get("beat_times_sec"),
        variant.get("raw_beat_times_sec"),
        payload.get("beat_times_sec"),
        payload.get("raw_beat_times_sec"),
        payload.get(f"{which_audio}_beat_times_sec"),
        manifest_rec.get("beat_times_sec"),
        manifest_rec.get(f"{which_audio}_beat_times_sec"),
    ):
        if candidate is None:
            continue
        beat_times = _sanitize_beat_times(candidate, float(duration_sec))
        if int(beat_times.size) > 0:
            return beat_times
    return None


def _resolve_song_beat_times(
    payload: Mapping[str, Any],
    manifest_rec: Mapping[str, Any],
    *,
    source_root: Path,
    duration_sec: float,
    beat_times_from: str,
    requested_sample_rate: int,
    madmom_beat_fps: int,
    madmom_correct: bool,
    preloaded_audio: Mapping[str, tuple[torch.Tensor, int]] | None = None,
    madmom_python: str = DEFAULT_MADMOM_PYTHON,
) -> tuple[np.ndarray, str]:
    beat_times_from_eff = str(beat_times_from).strip().lower()
    precomputed = _load_precomputed_beat_times(
        payload,
        manifest_rec,
        which_audio=str(beat_times_from_eff),
        duration_sec=float(duration_sec),
        madmom_beat_fps=int(madmom_beat_fps),
        madmom_correct=bool(madmom_correct),
    )
    if precomputed is not None:
        return precomputed, "precomputed"

    cached_audio = None if preloaded_audio is None else preloaded_audio.get(str(beat_times_from_eff))
    if cached_audio is None:
        beat_wav, beat_sr = _load_source_audio_mono(
            payload,
            source_root=source_root,
            manifest_rec=manifest_rec,
            which_audio=str(beat_times_from_eff),
            target_sample_rate=int(requested_sample_rate),
        )
    else:
        beat_wav, beat_sr = cached_audio
    detected = _sanitize_beat_times(
        _detect_madmom_beats_from_wav(
            beat_wav,
            int(beat_sr),
            fps=int(madmom_beat_fps),
            correct=bool(madmom_correct),
            madmom_python=str(madmom_python),
        ),
        float(duration_sec),
    )
    if int(detected.size) <= 0:
        raise RuntimeError(
            f"madmom did not detect any beats for which_audio={beat_times_from_eff!r}"
        )
    return detected, "detected"


def _resolve_grid_frame_rate(payload: Mapping[str, Any], manifest_rec: Mapping[str, Any]) -> float:
    for source in (dict(payload.get("meta") or {}), dict(manifest_rec)):
        for key in ("frame_rate", "grid_frame_rate"):
            value = source.get(key)
            if value is None:
                continue
            try:
                value = float(value)
            except Exception:
                continue
            if float(value) > 0.0:
                return float(value)
    raise RuntimeError("missing frame_rate / grid_frame_rate in payload metadata")


def _compute_onset_envelope(
    wav_1cn: torch.Tensor,
    sr: int,
    *,
    window_ms: float,
    hop_ms: float,
    abs_threshold: float,
    rel_quantile: float,
    rel_scale: float,
) -> dict[str, Any]:
    wav = torch.as_tensor(wav_1cn, dtype=torch.float32).contiguous()
    if int(wav.dim()) != 2 or int(wav.shape[0]) != 1:
        raise RuntimeError(f"expected mono audio [1,T], got {tuple(wav.shape)}")
    win = max(16, int(round(float(window_ms) * float(sr) / 1000.0)))
    hop = max(1, int(round(float(hop_ms) * float(sr) / 1000.0)))
    env = F.avg_pool1d(wav.abs().unsqueeze(0), kernel_size=win, stride=hop, ceil_mode=True).squeeze(0).squeeze(0)
    if int(env.numel()) <= 1:
        onset = torch.zeros((0,), dtype=torch.float32)
        onset_times = torch.zeros((0,), dtype=torch.float32)
    else:
        onset = torch.clamp(env[1:] - env[:-1], min=0.0).to(dtype=torch.float32)
        onset_times = (torch.arange(int(onset.numel()), dtype=torch.float32) + 1.0) * (float(hop) / float(sr))
    positive = onset[onset > 0]
    rel_q = float(np.clip(float(rel_quantile), 0.0, 1.0))
    rel_ref = float(torch.quantile(positive, rel_q).item()) if int(positive.numel()) > 0 else 0.0
    threshold = max(float(abs_threshold), float(rel_scale) * float(rel_ref))
    return {
        "times_sec": onset_times,
        "strength": onset,
        "threshold": float(threshold),
    }


def _interval_peak(times_sec: torch.Tensor, values: torch.Tensor, start_sec: float, end_sec: float) -> float:
    if int(values.numel()) <= 0:
        return 0.0
    mask = (times_sec >= float(start_sec)) & (times_sec < float(end_sec))
    if not bool(mask.any()):
        return 0.0
    return float(values[mask].max().item())


def _build_keep_rows(
    *,
    drumgrid: torch.Tensor,
    wav_1cn: torch.Tensor,
    sample_rate: int,
    frame_rate: float,
    beat_times_sec: np.ndarray,
    beats_per_sample: int,
    beat_hop: int,
    min_beat_samples: int,
    drop_empty_grid_beats: bool,
    drop_boundary_mismatch_beats: bool,
    boundary_hit_window_ms: float,
    onset_window_ms: float,
    onset_hop_ms: float,
    onset_abs_threshold: float,
    onset_rel_quantile: float,
    onset_rel_scale: float,
) -> dict[str, Any]:
    onset = _compute_onset_envelope(
        wav_1cn,
        int(sample_rate),
        window_ms=float(onset_window_ms),
        hop_ms=float(onset_hop_ms),
        abs_threshold=float(onset_abs_threshold),
        rel_quantile=float(onset_rel_quantile),
        rel_scale=float(onset_rel_scale),
    )
    boundaries = np.concatenate(([0.0], np.asarray(beat_times_sec, dtype=np.float64)), dtype=np.float64)
    keep_rows: list[dict[str, Any]] = []
    drumgrid_slices: list[torch.Tensor] = []
    dropped_empty_grid_indices: list[int] = []
    dropped_boundary_mismatch_indices: list[int] = []
    dropped_too_short_indices: list[int] = []
    total_num_samples = int(wav_1cn.shape[-1])
    total_num_frames = int(drumgrid.shape[-1])
    num_detected_beats = max(0, int(len(boundaries) - 1))
    for beat_index in range(0, max(0, num_detected_beats - int(beats_per_sample) + 1), int(beat_hop)):
        beat_index_end = int(beat_index + int(beats_per_sample) - 1)
        start_sec = float(boundaries[int(beat_index)])
        end_sec = float(boundaries[int(beat_index) + int(beats_per_sample)])
        if not float(end_sec) > float(start_sec):
            continue
        start_sample = min(total_num_samples, max(0, int(round(float(start_sec) * float(sample_rate)))))
        end_sample = min(total_num_samples, max(start_sample + 1, int(round(float(end_sec) * float(sample_rate)))))
        num_samples = int(end_sample - start_sample)
        if int(num_samples) < int(min_beat_samples):
            dropped_too_short_indices.append(int(beat_index))
            continue
        start_frame = min(total_num_frames, max(0, int(round(float(start_sec) * float(frame_rate)))))
        end_frame = min(total_num_frames, max(start_frame + 1, int(round(float(end_sec) * float(frame_rate)))))
        if int(start_frame) >= int(total_num_frames):
            continue
        drumgrid_slice = drumgrid[:, int(start_frame) : int(end_frame)].contiguous()
        drumgrid_has_hit = bool((drumgrid_slice > 0).any().item())
        onset_peak = _interval_peak(onset["times_sec"], onset["strength"], float(start_sec), float(end_sec))
        detected_audio_onset = bool(float(onset_peak) >= float(onset["threshold"]))
        boundary_window_sec = max(
            0.0,
            min(float(end_sec - start_sec), float(boundary_hit_window_ms) / 1000.0),
        )
        boundary_end_sec = float(start_sec + float(boundary_window_sec))
        start_onset_peak = _interval_peak(onset["times_sec"], onset["strength"], float(start_sec), boundary_end_sec)
        detected_audio_start_onset = bool(float(start_onset_peak) >= float(onset["threshold"]))
        boundary_window_frames = min(
            int(drumgrid_slice.shape[-1]),
            max(1, int(round(float(boundary_window_sec) * float(frame_rate)))) if float(boundary_window_sec) > 0.0 else 0,
        )
        drumgrid_start_hit = bool((drumgrid_slice[:, : int(boundary_window_frames)] > 0).any().item()) if int(boundary_window_frames) > 0 else False
        if bool(drop_empty_grid_beats) and (not bool(drumgrid_has_hit)):
            dropped_empty_grid_indices.append(int(beat_index))
            continue
        if bool(drop_boundary_mismatch_beats) and bool(drumgrid_start_hit) != bool(detected_audio_start_onset):
            dropped_boundary_mismatch_indices.append(int(beat_index))
            continue
        keep_rows.append(
            {
                "beat_index": int(beat_index),
                "beat_index_end": int(beat_index_end),
                "num_beats": int(beats_per_sample),
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
                "duration_sec": float(end_sec - start_sec),
                "sample_start": int(start_sample),
                "sample_end": int(end_sample),
                "sample_count": int(num_samples),
                "grid_start_frame": int(start_frame),
                "grid_end_frame": int(end_frame),
                "grid_num_frames": int(drumgrid_slice.shape[-1]),
                "grid_has_hit": bool(drumgrid_has_hit),
                "grid_start_hit": bool(drumgrid_start_hit),
                "onset_peak": float(onset_peak),
                "audio_onset_detected": bool(detected_audio_onset),
                "start_onset_peak": float(start_onset_peak),
                "audio_start_onset_detected": bool(detected_audio_start_onset),
            }
        )
        drumgrid_slices.append(drumgrid_slice)
    return {
        "keep_rows": keep_rows,
        "drumgrid_slices": drumgrid_slices,
        "dropped_empty_grid_indices": dropped_empty_grid_indices,
        "dropped_boundary_mismatch_indices": dropped_boundary_mismatch_indices,
        "dropped_too_short_indices": dropped_too_short_indices,
        "total_detected_beats": int(num_detected_beats),
        "source_audio_num_samples": int(total_num_samples),
        "source_audio_duration_sec": float(total_num_samples / float(sample_rate)),
        "onset_threshold": float(onset["threshold"]),
    }


def _precompute_song_task(task: SongPrepTask) -> dict[str, Any]:
    source_root = Path(task.source_root)
    dataset_root = None if not str(task.dataset_root).strip() else Path(task.dataset_root)
    manifest_index = int(task.manifest_index)
    manifest_rec = dict(task.manifest_rec)
    source_id = str(
        manifest_rec.get("source_id")
        or manifest_rec.get("id")
        or f"manifest_{int(manifest_index):06d}"
    )
    try:
        payload = dict(
            torch.load(
                _resolve_manifest_pt_path(source_root, manifest_rec),
                map_location="cpu",
                weights_only=False,
            )
        )
        source_meta = dict(payload.get("source_meta") or payload.get("meta") or payload.get("source_manifest") or {})
        wav_codec, _wav_sr = _load_source_audio_mono(
            payload,
            source_root=source_root,
            dataset_root=dataset_root,
            source_meta=source_meta,
            manifest_rec=manifest_rec,
            which_audio=str(task.which_audio),
            target_sample_rate=int(task.codec_sample_rate),
            prefer_native_audio=int(task.codec_sample_rate) != int(_resolve_payload_sample_rate(payload, fallback_sample_rate=int(task.codec_sample_rate))),
        )
        source_meta = _source_meta_with_actual_audio(
            source_meta,
            sample_rate=int(task.codec_sample_rate),
            num_samples=int(wav_codec.shape[-1]),
        )
        drumgrid = payload.get("drumgrid")
        if isinstance(drumgrid, np.ndarray):
            drumgrid = torch.from_numpy(drumgrid)
        if not torch.is_tensor(drumgrid):
            raise RuntimeError("payload is missing drumgrid")
        drumgrid = torch.as_tensor(drumgrid, dtype=torch.float32).contiguous()
        frame_rate = _resolve_grid_frame_rate(payload, manifest_rec)
        preloaded_audio = {
            str(task.which_audio).strip().lower(): (
                torch.as_tensor(wav_codec, dtype=torch.float32).contiguous(),
                int(task.codec_sample_rate),
            )
        }
        beat_times_sec, beat_times_source = _resolve_song_beat_times(
            payload,
            manifest_rec,
            source_root=source_root,
            duration_sec=float(wav_codec.shape[-1]) / float(task.codec_sample_rate),
            beat_times_from=str(task.beat_times_from),
            requested_sample_rate=int(task.codec_sample_rate),
            madmom_beat_fps=int(task.madmom_beat_fps),
            madmom_correct=bool(task.madmom_correct),
            preloaded_audio=preloaded_audio,
            madmom_python=str(task.madmom_python),
        )
        keep_bundle = _build_keep_rows(
            drumgrid=drumgrid,
            wav_1cn=wav_codec,
            sample_rate=int(task.codec_sample_rate),
            frame_rate=float(frame_rate),
            beat_times_sec=beat_times_sec,
            beats_per_sample=int(task.beats_per_sample),
            beat_hop=int(task.beat_hop),
            min_beat_samples=int(task.min_beat_samples),
            drop_empty_grid_beats=not bool(task.keep_empty_grid_beats),
            drop_boundary_mismatch_beats=not bool(task.keep_boundary_mismatch_beats),
            boundary_hit_window_ms=float(task.boundary_hit_window_ms),
            onset_window_ms=float(task.onset_window_ms),
            onset_hop_ms=float(task.onset_hop_ms),
            onset_abs_threshold=float(task.onset_abs_threshold),
            onset_rel_quantile=float(task.onset_rel_quantile),
            onset_rel_scale=float(task.onset_rel_scale),
        )
        return {
            "status": "ok",
            "manifest_index": int(manifest_index),
            "manifest_rec": manifest_rec,
            "source_id": str(source_id),
            "source_meta": source_meta,
            "frame_rate": float(frame_rate),
            "beat_times_sec": [float(x) for x in list(beat_times_sec)],
            "beat_times_source": str(beat_times_source),
            "keep_rows": list(keep_bundle["keep_rows"]),
            "drumgrid_slices": list(keep_bundle["drumgrid_slices"]),
            "dropped_empty_grid_indices": list(keep_bundle["dropped_empty_grid_indices"]),
            "dropped_boundary_mismatch_indices": list(keep_bundle["dropped_boundary_mismatch_indices"]),
            "dropped_too_short_indices": list(keep_bundle["dropped_too_short_indices"]),
            "total_detected_beats": int(keep_bundle["total_detected_beats"]),
            "source_audio_num_samples": int(keep_bundle["source_audio_num_samples"]),
            "source_audio_duration_sec": float(keep_bundle["source_audio_duration_sec"]),
            "onset_threshold": float(keep_bundle["onset_threshold"]),
            "wav_codec": (
                torch.as_tensor(wav_codec, dtype=torch.float32).contiguous()
                if bool(task.include_waveform)
                else None
            ),
        }
    except Exception as exc:
        return {
            "status": "skip",
            "manifest_index": int(manifest_index),
            "manifest_rec": manifest_rec,
            "source_id": str(source_id),
            "reason": str(exc),
        }


def _pack_cached_audio_from_waveform(
    wav_1cn: torch.Tensor,
    *,
    prefix: str,
    sample_rate: int,
    keep_rows: Sequence[Mapping[str, Any]],
    source_audio_start_sec: float,
    context_ms: float,
) -> dict[str, Any]:
    wav = torch.as_tensor(wav_1cn, dtype=torch.float32).contiguous()
    if int(wav.dim()) != 2 or int(wav.shape[0]) != 1:
        raise RuntimeError(f"expected mono waveform [1,T], got {tuple(wav.shape)}")
    if not keep_rows:
        return {}
    context_sec = max(0.0, float(context_ms) / 1000.0)
    total_num_samples = int(wav.shape[-1])
    row_specs: list[tuple[int, int, int, int, int, int]] = []
    max_num_samples = 0
    for row in keep_rows:
        beat_start_abs = float(source_audio_start_sec) + float(row.get("start_sec", 0.0) or 0.0)
        beat_end_abs = float(source_audio_start_sec) + float(row.get("end_sec", 0.0) or 0.0)
        pre_sec = min(float(context_sec), max(0.0, float(beat_start_abs)))
        crop_start_sec = float(max(0.0, float(beat_start_abs) - float(context_sec)))
        crop_end_sec = float(beat_end_abs + float(context_sec))
        crop_start = int(max(0, math.floor(float(crop_start_sec) * float(sample_rate))))
        crop_num_samples = int(max(1, math.ceil(max(0.0, float(crop_end_sec - crop_start_sec)) * float(sample_rate))))
        if int(total_num_samples) > 0:
            if int(crop_start) >= int(total_num_samples):
                crop_start = max(0, int(total_num_samples) - 1)
            crop_num_samples = int(max(1, min(int(crop_num_samples), int(total_num_samples) - int(crop_start))))
            crop_end = int(min(int(total_num_samples), int(crop_start) + int(crop_num_samples)))
        else:
            crop_start = 0
            crop_end = 0
        num_samples = int(max(0, int(crop_end) - int(crop_start)))
        left_context_samples = int(max(0, round(float(pre_sec) * float(sample_rate))))
        beat_num_samples = int(max(1, round(max(0.0, float(beat_end_abs - beat_start_abs)) * float(sample_rate))))
        if int(num_samples) < int(left_context_samples) + int(beat_num_samples):
            beat_num_samples = max(1, min(int(beat_num_samples), int(num_samples) - int(left_context_samples)))
        right_context_samples = int(max(0, int(num_samples) - int(left_context_samples) - int(beat_num_samples)))
        row_specs.append(
            (
                int(crop_start),
                int(crop_end),
                int(left_context_samples),
                int(right_context_samples),
                int(beat_num_samples),
                int(num_samples),
            )
        )
        max_num_samples = max(int(max_num_samples), int(num_samples))

    num_items = int(len(row_specs))
    audio = torch.zeros((int(num_items), 1, int(max_num_samples)), dtype=torch.float32)
    loss_mask = torch.zeros((int(num_items), int(max_num_samples)), dtype=torch.bool)
    left_context = torch.zeros((int(num_items),), dtype=torch.int32)
    right_context = torch.zeros((int(num_items),), dtype=torch.int32)
    beat_num_samples = torch.zeros((int(num_items),), dtype=torch.int32)
    target_num_samples = torch.zeros((int(num_items),), dtype=torch.int32)
    for row_idx, (crop_start, crop_end, left_samples, right_samples, beat_samples, num_samples) in enumerate(row_specs):
        if int(num_samples) > 0:
            audio[int(row_idx), 0, : int(num_samples)] = wav[:, int(crop_start) : int(crop_end)]
        left_context[int(row_idx)] = int(left_samples)
        right_context[int(row_idx)] = int(right_samples)
        beat_num_samples[int(row_idx)] = int(beat_samples)
        target_num_samples[int(row_idx)] = int(num_samples)
        center_end = min(int(num_samples), int(left_samples) + int(beat_samples))
        if int(center_end) > int(left_samples):
            loss_mask[int(row_idx), int(left_samples) : int(center_end)] = True
    return {
        str(prefix): audio,
        f"{prefix}_loss_mask": loss_mask,
        f"{prefix}_sample_rate": int(sample_rate),
        f"{prefix}_left_context_samples": left_context,
        f"{prefix}_right_context_samples": right_context,
        f"{prefix}_beat_num_samples": beat_num_samples,
        f"{prefix}_num_samples": target_num_samples,
        f"{prefix}_context_ms": float(context_ms),
    }


def _build_cached_conditioning_payload(
    *,
    source_meta: Mapping[str, Any],
    keep_rows: Sequence[Mapping[str, Any]],
    dataset_root: Path | None,
) -> dict[str, Any]:
    if dataset_root is None:
        return {}
    if pretty_midi is None:
        raise RuntimeError("pretty_midi is required to build conditioning tensors")
    midi_rel = str(source_meta.get("midi_file") or "").strip()
    if not midi_rel:
        return {}
    midi_path = Path(midi_rel)
    if not midi_path.is_absolute():
        midi_path = dataset_root / midi_path
    if not midi_path.is_file():
        return {}
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    midi_events = build_midi_event_cache(pm)
    source_audio_start_sec = float(source_meta.get("audio_start_anchor_sec", source_meta.get("start_sec", 0.0)) or 0.0)
    state_rows: list[np.ndarray] = []
    onset_vel_rows: list[np.ndarray] = []
    onset_id_rows: list[np.ndarray] = []
    onset_rows: list[np.ndarray] = []
    onset_count_rows: list[np.ndarray] = []
    max_grid_frames = 0
    for row in keep_rows:
        num_frames = int(row.get("grid_num_frames", 0) or 0)
        beat_start_abs = float(source_audio_start_sec) + float(row.get("start_sec", 0.0) or 0.0)
        beat_end_abs = float(source_audio_start_sec) + float(row.get("end_sec", 0.0) or 0.0)
        grid_np, onset_ids_np, family_onsets_np, family_onset_count_np, _support_np, _salience_np = render_midi_family_state_grid(
            midi_events=midi_events,
            start_sec=float(beat_start_abs),
            end_sec=float(beat_end_abs),
            num_frames=int(num_frames),
        )
        state_rows.append(
            np.asarray(grid_np[0::3, :], dtype=np.float32)
        )
        onset_vel_rows.append(
            np.asarray(grid_np[1::3, :], dtype=np.float32)
        )
        onset_id_rows.append(np.asarray(onset_ids_np, dtype=np.int64))
        onset_rows.append(np.asarray(family_onsets_np, dtype=np.bool_))
        onset_count_rows.append(np.asarray(family_onset_count_np, dtype=np.uint8))
        max_grid_frames = max(int(max_grid_frames), int(grid_np.shape[-1]))
    if not state_rows:
        return {}
    num_rows = int(len(state_rows))
    num_families = int(len(FAMILY_STATE_FAMILY_NAMES))
    conditioning_state_vel = torch.zeros((int(num_rows), int(num_families), int(max_grid_frames)), dtype=torch.float32)
    conditioning_onset_vel = torch.zeros((int(num_rows), int(num_families), int(max_grid_frames)), dtype=torch.float32)
    conditioning_onset_ids = torch.full((int(num_rows), int(num_families), int(max_grid_frames)), -1, dtype=torch.int16)
    conditioning_family_onsets = torch.zeros((int(num_rows), int(num_families), int(max_grid_frames)), dtype=torch.bool)
    conditioning_family_onset_count = torch.zeros((int(num_rows), int(num_families), int(max_grid_frames)), dtype=torch.uint8)
    for row_idx, (state_np, onset_vel_np, onset_ids_np, onset_np, onset_count_np) in enumerate(
        zip(state_rows, onset_vel_rows, onset_id_rows, onset_rows, onset_count_rows)
    ):
        grid_frames = int(state_np.shape[-1])
        conditioning_state_vel[int(row_idx), :, : int(grid_frames)] = torch.from_numpy(state_np.astype(np.float32, copy=False))
        conditioning_onset_vel[int(row_idx), :, : int(grid_frames)] = torch.from_numpy(onset_vel_np.astype(np.float32, copy=False))
        conditioning_onset_ids[int(row_idx), :, : int(grid_frames)] = torch.from_numpy(onset_ids_np.astype(np.int16, copy=False))
        conditioning_family_onsets[int(row_idx), :, : int(grid_frames)] = torch.from_numpy(onset_np.astype(np.bool_, copy=False))
        conditioning_family_onset_count[int(row_idx), :, : int(grid_frames)] = torch.from_numpy(onset_count_np.astype(np.uint8, copy=False))
    return {
        "conditioning_cache_config": {
            "mode": "midi_family_state_onset_ids",
            "feature_row_names": list(FAMILY_STATE_FEATURE_ROW_NAMES),
            "class_names": list(FAMILY_STATE_FAMILY_NAMES),
            "class_id_vocab_sizes": [int(x) for x in list(FAMILY_STATE_ID_VOCAB_SIZES)],
            "profile_version": int(CONDITIONING_CACHE_PROFILE_VERSION),
        },
        "conditioning_state_vel": conditioning_state_vel,
        "conditioning_onset_vel": conditioning_onset_vel,
        "conditioning_onset_ids": conditioning_onset_ids,
        "conditioning_family_onsets": conditioning_family_onsets,
        "conditioning_family_onset_count": conditioning_family_onset_count,
    }


def _build_cached_audio_target_payloads(
    *,
    keep_rows: Sequence[Mapping[str, Any]],
    source_root: Path,
    dataset_root: Path | None,
    manifest_rec: Mapping[str, Any],
    source_meta: Mapping[str, Any],
    which_audio: str,
    source_wav_codec: torch.Tensor | None,
    source_codec_sample_rate: int,
    cache_target_audio_32k: bool,
    cache_target_audio_44k: bool,
    target_audio_context_ms: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if bool(cache_target_audio_32k) and keep_rows:
        wav_32k = source_wav_codec
        if wav_32k is not None:
            wav_32k = torch.as_tensor(wav_32k, dtype=torch.float32).contiguous()
            if int(source_codec_sample_rate) != int(CACHED_AUDIO_SAMPLE_RATE_32K):
                wav_32k = _resample_waveform_hq(
                    wav_32k,
                    int(source_codec_sample_rate),
                    int(CACHED_AUDIO_SAMPLE_RATE_32K),
                )
        else:
            payload_like: dict[str, Any] = {}
            wav_32k, _ = _load_source_audio_mono(
                payload_like,
                source_root=source_root,
                manifest_rec=manifest_rec,
                which_audio=str(which_audio),
                target_sample_rate=int(CACHED_AUDIO_SAMPLE_RATE_32K),
            )
        out.update(
            _pack_cached_audio_from_waveform(
                wav_32k,
                prefix="target_audio_32k",
                sample_rate=int(CACHED_AUDIO_SAMPLE_RATE_32K),
                keep_rows=keep_rows,
                source_audio_start_sec=0.0,
                context_ms=float(target_audio_context_ms),
            )
        )
    if bool(cache_target_audio_44k) and keep_rows and dataset_root is not None:
        audio_rel_44k = str(source_meta.get("audio_file") or "").strip()
        source_audio_start_sec = float(source_meta.get("audio_start_anchor_sec", source_meta.get("start_sec", 0.0)) or 0.0)
        if audio_rel_44k:
            audio_path = Path(audio_rel_44k)
            if not audio_path.is_absolute():
                audio_path = dataset_root / audio_path
            wav_44k, _ = _load_audio_mono(audio_path, sample_rate=int(TARGET_AUDIO_SAMPLE_RATE_44K), mixdown=False)
            out.update(
                _pack_cached_audio_from_waveform(
                    wav_44k,
                    prefix="target_audio_44k",
                    sample_rate=int(TARGET_AUDIO_SAMPLE_RATE_44K),
                    keep_rows=keep_rows,
                    source_audio_start_sec=float(source_audio_start_sec),
                    context_ms=float(target_audio_context_ms),
                )
            )
    return out


def _pack_song_payload(
    *,
    source_id: str,
    source_meta: Mapping[str, Any],
    source_manifest_rec: Mapping[str, Any],
    keep_rows: Sequence[Mapping[str, Any]],
    drumgrid_slices: Sequence[torch.Tensor],
    encoded_codes: Sequence[torch.Tensor],
    codec_metadata: AudioCodecMetadata,
    cache_format: str,
    which_audio: str,
    beat_times_from: str,
    madmom_beat_fps: int,
    madmom_correct: bool,
    beat_times_source: str,
    frame_rate: float,
    beat_times_sec: Sequence[float],
    onset_threshold: float,
    conditioning_payload: Mapping[str, Any],
    audio_target_payloads: Mapping[str, Any],
) -> dict[str, Any]:
    if not keep_rows:
        return {
            "cache_format": str(cache_format),
            "codec_metadata": codec_metadata.to_dict(),
            "source_id": str(source_id),
            "which_audio": str(which_audio),
            "beat_times_from": str(beat_times_from),
            "madmom_beat_fps": int(madmom_beat_fps),
            "madmom_correct": bool(madmom_correct),
            "beat_times_source": str(beat_times_source),
            "sample_rate": int(codec_metadata.codec_sample_rate),
            "frame_rate": float(frame_rate),
            "class_names": list(source_manifest_rec.get("classes") or []),
            "beat_times_sec": torch.zeros((0,), dtype=torch.float32),
            "drumgrid": torch.zeros((0, 0, 0), dtype=torch.float16),
            "drumgrid_num_frames": torch.zeros((0,), dtype=torch.int32),
            "codes": torch.zeros((0, int(codec_metadata.codec_num_codebooks), 0), dtype=torch.int16),
            "code_num_frames": torch.zeros((0,), dtype=torch.int32),
            "beat_index": torch.zeros((0,), dtype=torch.int32),
            "beat_start_sec": torch.zeros((0,), dtype=torch.float32),
            "beat_end_sec": torch.zeros((0,), dtype=torch.float32),
            "beat_sample_start": torch.zeros((0,), dtype=torch.int32),
            "beat_sample_end": torch.zeros((0,), dtype=torch.int32),
            "grid_has_hit": torch.zeros((0,), dtype=torch.bool),
            "audio_onset_detected": torch.zeros((0,), dtype=torch.bool),
            "onset_peak": torch.zeros((0,), dtype=torch.float32),
            "source_meta": dict(source_meta),
            **dict(conditioning_payload),
            **dict(audio_target_payloads),
        }
    num_items = int(len(keep_rows))
    num_classes = int(drumgrid_slices[0].shape[0])
    num_codebooks = int(encoded_codes[0].shape[0])
    max_grid_frames = max(int(item.shape[-1]) for item in drumgrid_slices)
    max_code_frames = max(int(item.shape[-1]) for item in encoded_codes)
    drumgrid_padded = torch.zeros((int(num_items), int(num_classes), int(max_grid_frames)), dtype=torch.float16)
    drumgrid_num_frames = torch.zeros((int(num_items),), dtype=torch.int32)
    codes_padded = torch.full((int(num_items), int(num_codebooks), int(max_code_frames)), DEFAULT_CODES_PAD_VALUE, dtype=torch.int16)
    code_num_frames = torch.zeros((int(num_items),), dtype=torch.int32)
    beat_index = torch.zeros((int(num_items),), dtype=torch.int32)
    beat_start_sec = torch.zeros((int(num_items),), dtype=torch.float32)
    beat_end_sec = torch.zeros((int(num_items),), dtype=torch.float32)
    beat_sample_start = torch.zeros((int(num_items),), dtype=torch.int32)
    beat_sample_end = torch.zeros((int(num_items),), dtype=torch.int32)
    grid_has_hit = torch.zeros((int(num_items),), dtype=torch.bool)
    audio_onset_detected = torch.zeros((int(num_items),), dtype=torch.bool)
    onset_peak = torch.zeros((int(num_items),), dtype=torch.float32)
    for row_idx, (row, drumgrid_slice, codes_ct) in enumerate(zip(keep_rows, drumgrid_slices, encoded_codes)):
        grid_frames = int(drumgrid_slice.shape[-1])
        code_frames = int(codes_ct.shape[-1])
        drumgrid_padded[int(row_idx), :, : int(grid_frames)] = drumgrid_slice.to(dtype=torch.float16)
        drumgrid_num_frames[int(row_idx)] = int(grid_frames)
        codes_padded[int(row_idx), :, : int(code_frames)] = torch.as_tensor(codes_ct, dtype=torch.int16)
        code_num_frames[int(row_idx)] = int(code_frames)
        beat_index[int(row_idx)] = int(row["beat_index"])
        beat_start_sec[int(row_idx)] = float(row["start_sec"])
        beat_end_sec[int(row_idx)] = float(row["end_sec"])
        beat_sample_start[int(row_idx)] = int(row["sample_start"])
        beat_sample_end[int(row_idx)] = int(row["sample_end"])
        grid_has_hit[int(row_idx)] = bool(row["grid_has_hit"])
        audio_onset_detected[int(row_idx)] = bool(row["audio_onset_detected"])
        onset_peak[int(row_idx)] = float(row["onset_peak"])
    return {
        "cache_format": str(cache_format),
        "codec_metadata": codec_metadata.to_dict(),
        "source_id": str(source_id),
        "which_audio": str(which_audio),
        "beat_times_from": str(beat_times_from),
        "madmom_beat_fps": int(madmom_beat_fps),
        "madmom_correct": bool(madmom_correct),
        "beat_times_source": str(beat_times_source),
        "sample_rate": int(codec_metadata.codec_sample_rate),
        "frame_rate": float(frame_rate),
        "bandwidth": (
            None if codec_metadata.encodec_bandwidth is None else float(codec_metadata.encodec_bandwidth)
        ),
        "dac_num_quantizers": (
            None if codec_metadata.dac_num_quantizers is None else int(codec_metadata.dac_num_quantizers)
        ),
        "class_names": list(source_manifest_rec.get("classes") or []),
        "beat_times_sec": torch.as_tensor(list(beat_times_sec), dtype=torch.float32),
        "drumgrid": drumgrid_padded,
        "drumgrid_num_frames": drumgrid_num_frames,
        "codes": codes_padded,
        "code_num_frames": code_num_frames,
        "beat_index": beat_index,
        "beat_start_sec": beat_start_sec,
        "beat_end_sec": beat_end_sec,
        "beat_sample_start": beat_sample_start,
        "beat_sample_end": beat_sample_end,
        "grid_has_hit": grid_has_hit,
        "audio_onset_detected": audio_onset_detected,
        "onset_peak": onset_peak,
        "onset_threshold": float(onset_threshold),
        "source_meta": dict(source_meta),
        **dict(conditioning_payload),
        **dict(audio_target_payloads),
    }


def _song_output_paths(out_root: Path, source_id: str) -> tuple[Path, Path]:
    rel = Path(str(source_id)).with_suffix(".pt")
    return out_root / "shards" / rel, out_root / "meta" / rel.with_suffix(".json")


def _make_song_meta_payload(
    *,
    out_root: Path,
    shard_path: Path | None,
    source_id: str,
    source_manifest_index: int,
    source_manifest_rec: Mapping[str, Any],
    source_meta: Mapping[str, Any],
    keep_rows: Sequence[Mapping[str, Any]],
    codec_metadata: AudioCodecMetadata,
    cache_format: str,
    which_audio: str,
    beat_times_from: str,
    madmom_beat_fps: int,
    madmom_correct: bool,
    beat_times_source: str,
    frame_rate: float,
    beat_times_sec: Sequence[float],
    onset_threshold: float,
    total_detected_beats: int,
    dropped_empty_grid_indices: Sequence[int],
    dropped_boundary_mismatch_indices: Sequence[int],
    dropped_too_short_indices: Sequence[int],
    source_audio_num_samples: int,
    source_audio_duration_sec: float,
    cache_target_audio_32k: bool,
    cache_target_audio_44k: bool,
    cache_midi_family_state_ids: bool,
    target_audio_context_ms: float,
    beats_per_sample: int,
    beat_hop: int,
) -> dict[str, Any]:
    shard_rel = None if shard_path is None else _safe_rel(shard_path, out_root)
    rows: list[dict[str, Any]] = []
    for row_in_shard, row in enumerate(keep_rows):
        rows.append(
            {
                "id": (
                    f"{source_id}__beat{int(row['beat_index']):04d}"
                    if int(row.get("num_beats", beats_per_sample)) == 1
                    else f"{source_id}__beats{int(row['beat_index']):04d}_{int(row.get('beat_index_end', row['beat_index'])):04d}"
                ),
                "source_id": str(source_id),
                "segment_type": ("beat" if int(row.get("num_beats", beats_per_sample)) == 1 else "beat_span"),
                "split": source_manifest_rec.get("split"),
                "style": source_manifest_rec.get("style"),
                "drummer": source_manifest_rec.get("drummer"),
                "session": source_manifest_rec.get("session"),
                "dataset_name": source_manifest_rec.get("dataset_name"),
                "kit_name": source_manifest_rec.get("kit_name", source_meta.get("kit_name")),
                "source_row_id": source_manifest_rec.get("source_row_id", source_meta.get("source_row_id")),
                "bpm": source_manifest_rec.get("bpm"),
                "time_signature": source_manifest_rec.get("time_signature"),
                "which_audio": str(which_audio),
                "beat_times_from": str(beat_times_from),
                "madmom_beat_fps": int(madmom_beat_fps),
                "madmom_correct": bool(madmom_correct),
                "beat_times_source": str(beat_times_source),
                "codec_family": str(codec_metadata.codec_family),
                "codec_model_id": str(codec_metadata.codec_model_id),
                "codec_target_dim": int(codec_metadata.codec_target_dim),
                "encodec_bandwidth": (
                    None if codec_metadata.encodec_bandwidth is None else float(codec_metadata.encodec_bandwidth)
                ),
                "dac_num_quantizers": (
                    None if codec_metadata.dac_num_quantizers is None else int(codec_metadata.dac_num_quantizers)
                ),
                "beat_index": int(row["beat_index"]),
                "beat_index_end": int(row.get("beat_index_end", row["beat_index"])),
                "num_beats": int(row.get("num_beats", beats_per_sample)),
                "start_sec": float(row["start_sec"]),
                "end_sec": float(row["end_sec"]),
                "duration_sec": float(row["duration_sec"]),
                "sample_rate": int(codec_metadata.codec_sample_rate),
                "sample_count": int(row["sample_count"]),
                "source_audio_file": source_meta.get("audio_file"),
                "source_audio_sample_rate": int(codec_metadata.codec_sample_rate),
                "frame_rate": float(frame_rate),
                "grid_frame_rate": float(frame_rate),
                "grid_num_frames": int(row["grid_num_frames"]),
                "code_num_frames": int(row.get("code_num_frames", 0)),
                "grid_has_hit": bool(row["grid_has_hit"]),
                "grid_start_hit": bool(row.get("grid_start_hit", False)),
                "audio_onset_detected": bool(row["audio_onset_detected"]),
                "onset_peak": float(row["onset_peak"]),
                "audio_start_onset_detected": bool(row.get("audio_start_onset_detected", False)),
                "start_onset_peak": float(row.get("start_onset_peak", 0.0)),
                "pt": shard_rel,
                "row_in_shard": int(row_in_shard),
                "source_manifest_index": int(source_manifest_index),
                "source_orig_wav": source_manifest_rec.get("orig_wav"),
                "source_rend_wav": source_manifest_rec.get("rend_wav") or source_manifest_rec.get("rendered_wav"),
                "source_pt": source_manifest_rec.get("pt"),
            }
        )
    return {
        "cache_format": str(cache_format),
        "codec_metadata": codec_metadata.to_dict(),
        "source_id": str(source_id),
        "manifest_index": int(source_manifest_index),
        "dataset_name": source_manifest_rec.get("dataset_name", source_meta.get("dataset_name")),
        "kit_name": source_manifest_rec.get("kit_name", source_meta.get("kit_name")),
        "source_row_id": source_manifest_rec.get("source_row_id", source_meta.get("source_row_id")),
        "source_manifest": dict(source_manifest_rec),
        "source_meta": dict(source_meta),
        "config": {
            "which_audio": str(which_audio),
            "beat_times_from": str(beat_times_from),
            "madmom_beat_fps": int(madmom_beat_fps),
            "madmom_correct": bool(madmom_correct),
            "beat_times_source": str(beat_times_source),
            "cache_target_audio_32k": bool(cache_target_audio_32k),
            "cache_target_audio_44k": bool(cache_target_audio_44k),
            "cache_midi_family_state_ids": bool(cache_midi_family_state_ids),
            "target_audio_context_ms": float(target_audio_context_ms),
            "beats_per_sample": int(beats_per_sample),
            "beat_hop": int(beat_hop),
            **codec_metadata.to_dict(),
        },
        "stats": {
            "total_detected_beats": int(total_detected_beats),
            "kept_beats": int(len(rows)),
            "kept_segments": int(len(rows)),
            "dropped_empty_grid_beats": int(len(list(dropped_empty_grid_indices))),
            "dropped_boundary_mismatch_beats": int(len(list(dropped_boundary_mismatch_indices))),
            "dropped_too_short_beats": int(len(list(dropped_too_short_indices))),
            "source_audio_num_samples": int(source_audio_num_samples),
            "source_audio_duration_sec": float(source_audio_duration_sec),
            "onset_threshold": float(onset_threshold),
        },
        "beat_times_sec": [float(x) for x in list(beat_times_sec)],
        "dropped_empty_grid_indices": [int(x) for x in list(dropped_empty_grid_indices)],
        "dropped_boundary_mismatch_indices": [int(x) for x in list(dropped_boundary_mismatch_indices)],
        "dropped_too_short_indices": [int(x) for x in list(dropped_too_short_indices)],
        "shard_pt": shard_rel,
        "rows": rows,
    }


def _iter_all_meta_payloads(meta_root: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    if not meta_root.exists():
        return payloads
    for path in sorted(meta_root.rglob("*.json")):
        try:
            payload = dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
        if payload.get("cache_format") not in {LEGACY_CACHE_FORMAT, GENERIC_CACHE_FORMAT}:
            continue
        payloads.append(payload)
    return payloads


def _rebuild_outputs(out_root: Path) -> dict[str, Any]:
    manifest_rows: list[dict[str, Any]] = []
    summary = {
        "num_song_meta_files": 0,
        "num_examples": 0,
        "num_kept_beats": 0,
        "num_kept_segments": 0,
        "num_detected_beats": 0,
        "num_dropped_empty_grid_beats": 0,
        "num_dropped_boundary_mismatch_beats": 0,
        "num_dropped_too_short_beats": 0,
    }
    for payload in _iter_all_meta_payloads(out_root / "meta"):
        summary["num_song_meta_files"] += 1
        stats = dict(payload.get("stats") or {})
        summary["num_examples"] += 1
        summary["num_kept_beats"] += int(stats.get("kept_beats", 0))
        summary["num_kept_segments"] += int(stats.get("kept_segments", 0))
        summary["num_detected_beats"] += int(stats.get("total_detected_beats", 0))
        summary["num_dropped_empty_grid_beats"] += int(stats.get("dropped_empty_grid_beats", 0))
        summary["num_dropped_boundary_mismatch_beats"] += int(stats.get("dropped_boundary_mismatch_beats", 0))
        summary["num_dropped_too_short_beats"] += int(stats.get("dropped_too_short_beats", 0))
        manifest_rows.extend(list(payload.get("rows") or []))
    manifest_rows.sort(key=lambda row: (str(row.get("source_id", "")), int(row.get("beat_index", -1))))
    manifest_path = out_root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")
    summary["num_examples"] = int(len({str(row.get("source_id", "")) for row in manifest_rows}))
    summary["num_kept_beats"] = int(len(manifest_rows))
    summary["num_kept_segments"] = int(len(manifest_rows))
    _write_json(out_root / "summary.json", summary)
    return summary


def _is_legacy_source_codec(codec_metadata: AudioCodecMetadata) -> bool:
    return (
        str(codec_metadata.codec_family) == "encodec"
        and str(codec_metadata.codec_model_id) == "facebook/encodec_32khz"
        and int(codec_metadata.codec_audio_channels) == 1
        and int(codec_metadata.codec_num_codebooks) == 4
        and int(codec_metadata.codec_target_dim) == 128
        and codec_metadata.encodec_bandwidth is not None
        and abs(float(codec_metadata.encodec_bandwidth) - 2.2) <= 1.0e-6
    )


def _resolve_cache_format(codec_metadata: AudioCodecMetadata) -> str:
    return LEGACY_CACHE_FORMAT if _is_legacy_source_codec(codec_metadata) else GENERIC_CACHE_FORMAT


def _validate_out_root_for_codec(out_root: Path, codec_metadata: AudioCodecMetadata) -> None:
    if _is_legacy_source_codec(codec_metadata):
        return
    if out_root.name == "4beats_v9":
        raise ValueError(
            f"non-default codec {codec_metadata.codec_model_id!r} must not reuse {out_root}; "
            "choose a distinct out-root that encodes the codec and bandwidth/quantizer settings"
        )


def _existing_meta_matches(
    meta_path: Path,
    *,
    codec_metadata: AudioCodecMetadata,
    which_audio: str,
    beat_times_from: str,
    madmom_beat_fps: int,
    madmom_correct: bool,
    beats_per_sample: int,
    beat_hop: int,
    cache_target_audio_32k: bool,
    cache_target_audio_44k: bool,
    cache_midi_family_state_ids: bool,
    target_audio_context_ms: float,
) -> bool:
    if not meta_path.is_file():
        return False
    try:
        payload = dict(json.loads(meta_path.read_text(encoding="utf-8")))
    except Exception:
        return False
    config = dict(payload.get("config") or {})
    if str(config.get("which_audio")) != str(which_audio):
        return False
    if str(config.get("beat_times_from")) != str(beat_times_from):
        return False
    if int(config.get("madmom_beat_fps", -1)) != int(madmom_beat_fps):
        return False
    if bool(config.get("madmom_correct", not madmom_correct)) != bool(madmom_correct):
        return False
    if int(config.get("beats_per_sample", -1)) != int(beats_per_sample):
        return False
    if int(config.get("beat_hop", -1)) != int(beat_hop):
        return False
    if bool(config.get("cache_target_audio_32k", False)) != bool(cache_target_audio_32k):
        return False
    if bool(config.get("cache_target_audio_44k", False)) != bool(cache_target_audio_44k):
        return False
    if bool(config.get("cache_midi_family_state_ids", False)) != bool(cache_midi_family_state_ids):
        return False
    if not np.isclose(float(config.get("target_audio_context_ms", -1.0)), float(target_audio_context_ms)):
        return False
    config_codec_payload = dict(config.get("codec_metadata") or {})
    if not config_codec_payload:
        codec_keys = set(codec_metadata.to_dict().keys())
        config_codec_payload = {key: config.get(key) for key in codec_keys}
    return config_codec_payload == codec_metadata.to_dict()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 4-beat source cache with EnCodec or DAC tokenization.")
    parser.add_argument("--source-root", type=str, required=True)
    parser.add_argument("--out-root", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--which-audio", choices=("orig", "rend"), default="orig")
    parser.add_argument("--beat-times-from", choices=("orig", "rend"), default=None)
    parser.add_argument("--codec-family", choices=("encodec", "dac"), default="encodec")
    parser.add_argument("--codec-model-id", type=str, default=None)
    parser.add_argument("--encodec-bandwidth", type=float, default=DEFAULT_ENCODEC_BANDWIDTH)
    parser.add_argument("--dac-num-quantizers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--encode-batch-size", type=int, default=DEFAULT_ENCODE_BATCH_SIZE)
    parser.add_argument("--prep-num-workers", type=int, default=DEFAULT_PREP_NUM_WORKERS)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--indices", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--dedupe-by-source-row-id", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--n-beats", "--beats-per-sample", dest="beats_per_sample", type=int, default=4)
    parser.add_argument("--beat-hop", type=int, default=4)
    parser.add_argument("--min-beat-samples", type=int, default=DEFAULT_MIN_BEAT_SAMPLES)
    parser.add_argument("--keep-empty-grid-beats", action="store_true")
    parser.add_argument("--keep-boundary-mismatch-beats", action="store_true")
    parser.add_argument("--madmom-beat-fps", type=int, default=DEFAULT_MADMOM_BEAT_FPS)
    parser.add_argument("--madmom-correct", dest="madmom_correct", action="store_true")
    parser.add_argument("--no-madmom-correct", dest="madmom_correct", action="store_false")
    parser.add_argument("--madmom-python", type=str, default=DEFAULT_MADMOM_PYTHON)
    parser.add_argument("--boundary-hit-window-ms", type=float, default=DEFAULT_BOUNDARY_HIT_WINDOW_MS)
    parser.add_argument("--onset-window-ms", type=float, default=DEFAULT_ONSET_WINDOW_MS)
    parser.add_argument("--onset-hop-ms", type=float, default=DEFAULT_ONSET_HOP_MS)
    parser.add_argument("--onset-abs-threshold", type=float, default=DEFAULT_ONSET_ABS_THRESHOLD)
    parser.add_argument("--onset-rel-quantile", type=float, default=DEFAULT_ONSET_REL_QUANTILE)
    parser.add_argument("--onset-rel-scale", type=float, default=DEFAULT_ONSET_REL_SCALE)
    parser.add_argument("--cache-midi-family-state-ids", dest="cache_midi_family_state_ids", action="store_true")
    parser.add_argument("--no-cache-midi-family-state-ids", dest="cache_midi_family_state_ids", action="store_false")
    parser.add_argument("--cache-target-audio-32k", dest="cache_target_audio_32k", action="store_true")
    parser.add_argument("--no-cache-target-audio-32k", dest="cache_target_audio_32k", action="store_false")
    parser.add_argument("--cache-target-audio-44k", dest="cache_target_audio_44k", action="store_true")
    parser.add_argument("--no-cache-target-audio-44k", dest="cache_target_audio_44k", action="store_false")
    parser.add_argument("--target-audio-context-ms", type=float, default=DEFAULT_TARGET_AUDIO_CONTEXT_MS)
    parser.set_defaults(
        madmom_correct=DEFAULT_MADMOM_CORRECT,
        cache_midi_family_state_ids=True,
        cache_target_audio_32k=None,
        cache_target_audio_44k=None,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    dataset_root = None if args.dataset_root is None else Path(args.dataset_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    beat_times_from = (
        str(args.beat_times_from).strip().lower()
        if args.beat_times_from is not None
        else str(args.which_audio).strip().lower()
    )
    codec_family = infer_codec_family(codec_family=str(args.codec_family), codec_model_id=args.codec_model_id)
    codec_model_id = str(
        args.codec_model_id
        or (DEFAULT_DAC_CODEC_MODEL_ID if codec_family == "dac" else DEFAULT_CODEC_MODEL_ID)
    )
    requested_device = resolve_device(str(args.device))
    codec_model, resolved_device, codec_metadata = load_audio_codec_model(
        codec_family=str(codec_family),
        codec_model_id=codec_model_id,
        device=requested_device,
        encodec_bandwidth=float(args.encodec_bandwidth) if str(codec_family) == "encodec" else None,
        dac_num_quantizers=(None if int(args.dac_num_quantizers) <= 0 else int(args.dac_num_quantizers)),
    )
    _validate_out_root_for_codec(out_root, codec_metadata)
    cache_format = _resolve_cache_format(codec_metadata)
    if args.cache_target_audio_32k is None:
        args.cache_target_audio_32k = str(codec_metadata.codec_family).strip().lower() != "dac"
    if args.cache_target_audio_44k is None:
        args.cache_target_audio_44k = False
    if (bool(args.cache_midi_family_state_ids) or bool(args.cache_target_audio_44k)) and dataset_root is None:
        raise ValueError(
            "--dataset-root is required when caching midi family-state conditioning or 44.1 kHz target audio"
        )

    manifest_rows = _load_manifest(source_root / "manifest.jsonl")
    selected = _select_manifest_rows(
        manifest_rows,
        start_index=int(args.start_index),
        limit=args.limit,
        indices=_parse_indices(args.indices),
        split=args.split,
    )
    if bool(args.dedupe_by_source_row_id):
        selected = _dedupe_selected_manifest_rows_by_source_row_id(selected)
    if not selected:
        raise RuntimeError("no source manifest rows selected")

    root_config = {
        "cache_format": str(cache_format),
        "which_audio": str(args.which_audio),
        "beat_times_from": str(beat_times_from),
        "madmom_beat_fps": int(args.madmom_beat_fps),
        "madmom_correct": bool(args.madmom_correct),
        "beats_per_sample": int(args.beats_per_sample),
        "beat_hop": int(args.beat_hop),
        "cache_target_audio_32k": bool(args.cache_target_audio_32k),
        "cache_target_audio_44k": bool(args.cache_target_audio_44k),
        "cache_midi_family_state_ids": bool(args.cache_midi_family_state_ids),
        "target_audio_context_ms": float(args.target_audio_context_ms),
        "codec_metadata": codec_metadata.to_dict(),
        **codec_metadata.to_dict(),
    }
    _write_json(out_root / "config.json", root_config)

    written = 0
    reused = 0
    skipped = 0
    prep_tasks: list[SongPrepTask] = []
    for manifest_index, manifest_rec in selected:
        source_id = str(manifest_rec.get("source_id") or manifest_rec.get("id") or f"manifest_{int(manifest_index):06d}")
        shard_path, meta_path = _song_output_paths(out_root, source_id)
        if not bool(args.overwrite) and _existing_meta_matches(
            meta_path,
            codec_metadata=codec_metadata,
            which_audio=str(args.which_audio),
            beat_times_from=str(beat_times_from),
            madmom_beat_fps=int(args.madmom_beat_fps),
            madmom_correct=bool(args.madmom_correct),
            beats_per_sample=int(args.beats_per_sample),
            beat_hop=int(args.beat_hop),
            cache_target_audio_32k=bool(args.cache_target_audio_32k),
            cache_target_audio_44k=bool(args.cache_target_audio_44k),
            cache_midi_family_state_ids=bool(args.cache_midi_family_state_ids),
            target_audio_context_ms=float(args.target_audio_context_ms),
        ) and (shard_path.exists() or not bool((dict(json.loads(meta_path.read_text(encoding="utf-8"))).get("rows") or []))):
            reused += 1
            continue
        prep_tasks.append(
            SongPrepTask(
                source_root=str(source_root),
                dataset_root="" if dataset_root is None else str(dataset_root),
                manifest_index=int(manifest_index),
                manifest_rec=dict(manifest_rec),
                which_audio=str(args.which_audio),
                beat_times_from=str(beat_times_from),
                codec_sample_rate=int(codec_metadata.codec_sample_rate),
                madmom_beat_fps=int(args.madmom_beat_fps),
                madmom_correct=bool(args.madmom_correct),
                madmom_python=str(args.madmom_python),
                keep_empty_grid_beats=bool(args.keep_empty_grid_beats),
                keep_boundary_mismatch_beats=bool(args.keep_boundary_mismatch_beats),
                boundary_hit_window_ms=float(args.boundary_hit_window_ms),
                onset_window_ms=float(args.onset_window_ms),
                onset_hop_ms=float(args.onset_hop_ms),
                onset_abs_threshold=float(args.onset_abs_threshold),
                onset_rel_quantile=float(args.onset_rel_quantile),
                onset_rel_scale=float(args.onset_rel_scale),
                min_beat_samples=int(args.min_beat_samples),
                beats_per_sample=int(args.beats_per_sample),
                beat_hop=int(args.beat_hop),
                include_waveform=True,
            )
        )

    prepared_iter = _iter_prefetched_song_preps(prep_tasks, max_workers=int(args.prep_num_workers))
    for pre in _progress(prepared_iter, desc="build_source_cache", total=len(prep_tasks)):
        manifest_index = int(pre.get("manifest_index", -1))
        manifest_rec = dict(pre.get("manifest_rec") or {})
        source_id = str(pre.get("source_id") or manifest_rec.get("source_id") or manifest_rec.get("id") or "")
        shard_path, meta_path = _song_output_paths(out_root, source_id)
        if str(pre.get("status")) != "ok":
            skipped += 1
            print(
                f"[skip] manifest_index={manifest_index} source_id={source_id}: {pre.get('reason', 'unknown_error')}",
                file=sys.stderr,
            )
            continue
        try:
            source_meta = dict(pre.get("source_meta") or {})
            wav_codec = torch.as_tensor(pre.get("wav_codec"), dtype=torch.float32).contiguous()
            frame_rate = float(pre.get("frame_rate"))
            beat_times_sec = np.asarray(pre.get("beat_times_sec") or [], dtype=np.float64)
            beat_times_source = str(pre.get("beat_times_source") or "precomputed")
            keep_rows = list(pre.get("keep_rows") or [])
            drumgrid_slices = list(pre.get("drumgrid_slices") or [])
            encoded_codes: list[torch.Tensor] = []
            if keep_rows:
                chunks: list[torch.Tensor] = [
                    wav_codec[:, int(row["sample_start"]) : int(row["sample_end"])].contiguous()
                    for row in keep_rows
                ]
                grouped: dict[int, list[tuple[int, torch.Tensor]]] = {}
                for idx, chunk in enumerate(chunks):
                    grouped.setdefault(int(chunk.shape[-1]), []).append((int(idx), chunk))
                encoded_codes = [torch.empty((int(codec_metadata.codec_num_codebooks), 0), dtype=torch.long)] * int(len(chunks))
                for chunk_len in sorted(grouped.keys()):
                    pairs = list(grouped[int(chunk_len)])
                    for start in range(0, len(pairs), int(max(1, int(args.encode_batch_size)))):
                        batch_pairs = pairs[start : start + int(max(1, int(args.encode_batch_size)))]
                        batch_chunks = [chunk for _row_idx, chunk in batch_pairs]
                        batch_codes = encode_audio_chunks_to_codes(
                            codec_model,
                            batch_chunks,
                            device=resolved_device,
                            metadata=codec_metadata,
                        )
                        for (row_idx, _chunk), codes_ct in zip(batch_pairs, batch_codes):
                            encoded_codes[int(row_idx)] = codes_ct.to(dtype=torch.long).contiguous()
                for row, codes_ct in zip(keep_rows, encoded_codes):
                    row["code_num_frames"] = int(codes_ct.shape[-1])
            conditioning_payload = (
                _build_cached_conditioning_payload(
                    source_meta=source_meta,
                    keep_rows=keep_rows,
                    dataset_root=dataset_root,
                )
                if bool(args.cache_midi_family_state_ids)
                else {}
            )
            audio_target_payloads = _build_cached_audio_target_payloads(
                keep_rows=keep_rows,
                source_root=source_root,
                dataset_root=dataset_root,
                manifest_rec=manifest_rec,
                source_meta=source_meta,
                which_audio=str(args.which_audio),
                source_wav_codec=wav_codec,
                source_codec_sample_rate=int(codec_metadata.codec_sample_rate),
                cache_target_audio_32k=bool(args.cache_target_audio_32k),
                cache_target_audio_44k=bool(args.cache_target_audio_44k),
                target_audio_context_ms=float(args.target_audio_context_ms),
            )
            if keep_rows:
                shard_payload = _pack_song_payload(
                    source_id=source_id,
                    source_meta=source_meta,
                    source_manifest_rec=manifest_rec,
                    keep_rows=keep_rows,
                    drumgrid_slices=drumgrid_slices,
                    encoded_codes=encoded_codes,
                    codec_metadata=codec_metadata,
                    cache_format=cache_format,
                    which_audio=str(args.which_audio),
                    beat_times_from=str(beat_times_from),
                    madmom_beat_fps=int(args.madmom_beat_fps),
                    madmom_correct=bool(args.madmom_correct),
                    beat_times_source=str(beat_times_source),
                    frame_rate=float(frame_rate),
                    beat_times_sec=beat_times_sec.tolist(),
                    onset_threshold=float(pre.get("onset_threshold")),
                    conditioning_payload=conditioning_payload,
                    audio_target_payloads=audio_target_payloads,
                )
                shard_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(shard_payload, shard_path)
            elif shard_path.exists():
                shard_path.unlink()
            meta_payload = _make_song_meta_payload(
                out_root=out_root,
                shard_path=shard_path if keep_rows else None,
                source_id=source_id,
                source_manifest_index=int(manifest_index),
                source_manifest_rec=manifest_rec,
                source_meta=source_meta,
                keep_rows=keep_rows,
                codec_metadata=codec_metadata,
                cache_format=cache_format,
                which_audio=str(args.which_audio),
                beat_times_from=str(beat_times_from),
                madmom_beat_fps=int(args.madmom_beat_fps),
                madmom_correct=bool(args.madmom_correct),
                beat_times_source=str(beat_times_source),
                frame_rate=float(frame_rate),
                beat_times_sec=beat_times_sec.tolist(),
                onset_threshold=float(pre.get("onset_threshold")),
                total_detected_beats=int(pre.get("total_detected_beats")),
                dropped_empty_grid_indices=list(pre.get("dropped_empty_grid_indices") or []),
                dropped_boundary_mismatch_indices=list(pre.get("dropped_boundary_mismatch_indices") or []),
                dropped_too_short_indices=list(pre.get("dropped_too_short_indices") or []),
                source_audio_num_samples=int(pre.get("source_audio_num_samples")),
                source_audio_duration_sec=float(pre.get("source_audio_duration_sec")),
                cache_target_audio_32k=bool(args.cache_target_audio_32k),
                cache_target_audio_44k=bool(args.cache_target_audio_44k),
                cache_midi_family_state_ids=bool(args.cache_midi_family_state_ids),
                target_audio_context_ms=float(args.target_audio_context_ms),
                beats_per_sample=int(args.beats_per_sample),
                beat_hop=int(args.beat_hop),
            )
            _write_json(meta_path, meta_payload)
            written += 1
        except Exception as exc:
            skipped += 1
            print(f"[skip] manifest_index={manifest_index} source_id={source_id}: {exc}", file=sys.stderr)
            continue

    summary = _rebuild_outputs(out_root)
    summary["reuse_count"] = int(reused)
    summary["written_count"] = int(written)
    summary["skipped_count"] = int(skipped)
    _write_json(out_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
