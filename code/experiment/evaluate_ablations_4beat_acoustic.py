#!/usr/bin/env python3
from __future__ import annotations

"""Drum-aware acoustic evaluation for 4-beat ablation exports.

This script evaluates decoded EnCodec 32 kHz prediction WAVs against the cached
32 kHz target clips stored in the 4-beat transformer cache. It intentionally
avoids the 44.1 kHz refiner targets.

Metric design:
- `fadtk` FAD-inf on standardized 32 kHz beat clips for distributional quality
- paired log-mel error for broad time-frequency fidelity
- paired spectral-flux cosine scores for transient timing, both broadband and
  in drum-relevant low/mid/high bands
- band-balance and centroid deltas for spectral/timbral balance
- RMS and crest-factor deltas for loudness and punch

The metric mix is an inference from recent literature rather than a direct
copy-paste of any single benchmark:
- Kilgour et al. (Interspeech 2019) introduced FAD.
- Gui et al. (ICASSP 2024) show FAD depends on embedding/reference choice and
  recommend FAD-inf to reduce sample-size bias.
- Grötschla et al. (2025 preprint / ICASSP 2025) show music-generation quality
  is not captured well by any single automatic metric.
- Recent timbre/perception work continues to highlight spectral brightness and
  temporal-envelope cues as salient dimensions, which motivates centroid,
  band-balance, onset/flux, RMS, and crest-factor views here.
"""

import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import textwrap
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-drum-rendering")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-drum-rendering")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

REPO_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = REPO_ROOT.parent.parent
RUNS_ROOT = PACKAGE_ROOT / "runs"
RESULTS_ROOT = PACKAGE_ROOT / "results"


def _preload_stdlib_inspect() -> None:
    original_path = list(sys.path)
    repo = str(REPO_ROOT)
    sys.path = [path for path in sys.path if path not in {"", repo}]
    try:
        import inspect  # noqa: F401
    finally:
        sys.path = original_path


_preload_stdlib_inspect()

from dataclasses import dataclass

import matplotlib
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as AF
from numpy.lib.scimath import sqrt as scisqrt
from scipy import linalg
from tqdm.auto import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SAMPLE_RATE = 32000
EXPECTED_TARGET_AUDIO_CONTEXT_MS = 20.0
N_FFT = 1024
WIN_LENGTH = 1024
HOP_LENGTH = 256
N_MELS = 80
MEL_FMIN = 20.0
MEL_FMAX = 14000.0
EPS = 1.0e-8
RMS_DB_FLOOR = -80.0
STYLE_PLOT_TOP_K = 12
FAD_INF_STEPS = 25
FAD_INF_MIN_N = 500
FAD_CACHE_VERSION = "v1"
DEFAULT_FAD_SEED = 20260404
DEFAULT_FAD_REPEATS = 8
DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_PERMUTATION_SAMPLES = 2000
DEFAULT_INFERENCE_ALPHA = 0.05
STFT_WINDOW = torch.hann_window(int(WIN_LENGTH))
_MEL_BANK_CACHE: Dict[int, torch.Tensor] = {}
_FFT_FREQ_CACHE: Dict[int, np.ndarray] = {}

BAND_SPECS: Tuple[Tuple[str, float, float, str], ...] = (
    ("low", 30.0, 200.0, "kick-weighted low band"),
    ("mid", 200.0, 2000.0, "snare/tom body band"),
    ("high", 2000.0, 12000.0, "hihat/cymbal brightness band"),
)
BAND_NAMES: Tuple[str, ...] = tuple(spec[0] for spec in BAND_SPECS)
PAIRED_METRIC_COLUMNS: Tuple[str, ...] = (
    "mel_mae_db",
    "onset_flux_cosine",
    "low_flux_cosine",
    "mid_flux_cosine",
    "high_flux_cosine",
    "band_balance_l1",
    "centroid_mae_hz",
    "rms_mae_db",
    "rms_norm_mae_db",
    "crest_mae_db",
)

METRIC_SPECS: Dict[str, Dict[str, Any]] = {
    "fad_inf": {"label": "FAD∞", "better": "lower", "decimals": 3},
    "fad_inf_r2": {"label": "FAD∞ R²", "better": "higher", "decimals": 3},
    "mel_mae_db": {"label": "Mel MAE (dB)", "better": "lower", "decimals": 3},
    "onset_flux_cosine": {"label": "Broad Flux Cos", "better": "higher", "decimals": 3},
    "low_flux_cosine": {"label": "Low Flux Cos", "better": "higher", "decimals": 3},
    "mid_flux_cosine": {"label": "Mid Flux Cos", "better": "higher", "decimals": 3},
    "high_flux_cosine": {"label": "High Flux Cos", "better": "higher", "decimals": 3},
    "band_balance_l1": {"label": "Band Balance L1", "better": "lower", "decimals": 3},
    "centroid_mae_hz": {"label": "Centroid MAE (Hz)", "better": "lower", "decimals": 1},
    "rms_mae_db": {"label": "Raw RMS MAE (dB)", "better": "lower", "decimals": 3},
    "rms_norm_mae_db": {"label": "Peak-Norm RMS MAE (dB)", "better": "lower", "decimals": 3},
    "crest_mae_db": {"label": "Crest MAE (dB)", "better": "lower", "decimals": 3},
    "rtf_end_to_end": {"label": "RTF E2E", "better": "lower", "decimals": 3},
    "audio_sec_per_sec": {"label": "Audio Sec/Sec", "better": "higher", "decimals": 3},
    "total_train_wall_sec": {"label": "Train Wall (s)", "better": "lower", "decimals": 1},
    "time_to_best_checkpoint_sec": {"label": "Time To Best (s)", "better": "lower", "decimals": 1},
    "peak_gpu_mem_allocated_mb": {"label": "Peak GPU Mem (MB)", "better": "lower", "decimals": 1},
    "model_forward_sec_total": {"label": "Model Fwd (s)", "better": "lower", "decimals": 3},
    "codec_decode_sec_total": {"label": "Codec Decode (s)", "better": "lower", "decimals": 3},
    "export_wall_sec_total": {"label": "Export Wall (s)", "better": "lower", "decimals": 3},
    "total_audio_sec_generated": {"label": "Audio Gen (s)", "better": "higher", "decimals": 3},
    "clips_per_sec": {"label": "Clips/Sec", "better": "higher", "decimals": 3},
    "rtf_model_only": {"label": "RTF Model", "better": "lower", "decimals": 3},
    "total_val_wall_sec": {"label": "Val Wall (s)", "better": "lower", "decimals": 1},
    "mean_train_steps_per_sec": {"label": "Train Steps/Sec", "better": "higher", "decimals": 3},
    "mean_train_tokens_per_sec": {"label": "Train Tokens/Sec", "better": "higher", "decimals": 1},
    "mean_train_audio_seconds_per_sec": {"label": "Train Audio Sec/Sec", "better": "higher", "decimals": 3},
    "export_peak_gpu_mem_allocated_mb": {"label": "Export Peak GPU Mem (MB)", "better": "lower", "decimals": 1},
    "export_peak_gpu_mem_reserved_mb": {"label": "Export Peak GPU Res (MB)", "better": "lower", "decimals": 1},
    "peak_gpu_mem_reserved_mb": {"label": "Peak GPU Res (MB)", "better": "lower", "decimals": 1},
}

OVERALL_DISPLAY_METRICS: Tuple[str, ...] = (
    "fad_inf",
    "fad_inf_r2",
    "mel_mae_db",
    "onset_flux_cosine",
    "band_balance_l1",
    "centroid_mae_hz",
    "rms_mae_db",
    "crest_mae_db",
    "rtf_end_to_end",
    "audio_sec_per_sec",
    "total_train_wall_sec",
    "time_to_best_checkpoint_sec",
    "peak_gpu_mem_allocated_mb",
)
BAND_DISPLAY_METRICS: Tuple[str, ...] = (
    "low_flux_cosine",
    "mid_flux_cosine",
    "high_flux_cosine",
    "band_balance_l1",
)
EFFICIENCY_DISPLAY_COLUMNS: Tuple[str, ...] = (
    "model",
    "device_name",
    "batch_size",
    "num_examples",
    "best_checkpoint_epoch",
    "num_parameters",
    "rtf_end_to_end",
    "audio_sec_per_sec",
    "model_forward_sec_total",
    "codec_decode_sec_total",
    "export_wall_sec_total",
    "total_train_wall_sec",
    "total_val_wall_sec",
    "time_to_best_checkpoint_sec",
    "mean_train_steps_per_sec",
    "mean_train_tokens_per_sec",
    "mean_train_audio_seconds_per_sec",
    "peak_gpu_mem_allocated_mb",
    "peak_gpu_mem_reserved_mb",
    "export_peak_gpu_mem_allocated_mb",
    "export_peak_gpu_mem_reserved_mb",
)
INT_DISPLAY_COLUMNS: Tuple[str, ...] = (
    "num_examples",
    "batch_size",
    "best_checkpoint_epoch",
    "num_parameters",
)

LITERATURE_REFERENCES: Tuple[Tuple[str, str], ...] = (
    (
        "Kilgour et al., 2019. Fréchet Audio Distance: A Reference-Free Metric for Evaluating Music Enhancement Algorithms.",
        "https://www.isca-archive.org/interspeech_2019/kilgour19_interspeech.html",
    ),
    (
        "Gui et al., 2024. Adapting Fréchet Audio Distance for Generative Music Evaluation.",
        "https://www.microsoft.com/en-us/research/publication/adapting-frechet-audio-distance-for-generative-music-evaluation/",
    ),
    (
        "Grötschla et al., 2025. Benchmarking Music Generation Models and Metrics via Human Preference Studies.",
        "https://openreview.net/forum?id=105yqGIpVW",
    ),
    (
        "Saitis and Wallmark, 2024. Timbral brightness perception investigated through multimodal interference.",
        "https://pubmed.ncbi.nlm.nih.gov/39090510/",
    ),
    (
        "Auditory and vibrotactile interactions in perception of timbre acoustic features, 2025.",
        "https://pubmed.ncbi.nlm.nih.gov/41168236/",
    ),
)


@dataclass(frozen=True)
class AudioFeatures:
    mel_db: np.ndarray
    flux: np.ndarray
    band_flux: Dict[str, np.ndarray]
    band_energy_ratio: np.ndarray
    centroid_hz: float
    rms_dbfs: float
    rms_norm_dbfs: float
    crest_db: float


@dataclass(frozen=True)
class TargetClip:
    row: Dict[str, Any]
    audio: np.ndarray
    sample_rate: int
    features: AudioFeatures


@dataclass(frozen=True)
class RunPredictions:
    name: str
    run_dir: Path
    manifest_path: Path
    wav_root: Path
    rows_by_dataset_index: Dict[int, Dict[str, Any]]


@dataclass(frozen=True)
class FADOutputs:
    overall_df: pd.DataFrame
    repeat_df: pd.DataFrame
    cache_stats: Dict[str, Any]


@dataclass(frozen=True)
class InferenceOutputs:
    intervals_df: pd.DataFrame
    significance_df: pd.DataFrame


class TargetAudioCache:
    def __init__(self, cache_root: Path) -> None:
        self.cache_root = Path(cache_root).resolve()
        self.manifest_rows = _read_jsonl(self.cache_root / "manifest.jsonl")
        self._shards: Dict[str, Dict[str, Any]] = {}
        self._clips: Dict[int, TargetClip] = {}

    def get_row(self, dataset_index: int) -> Dict[str, Any]:
        if int(dataset_index) < 0 or int(dataset_index) >= len(self.manifest_rows):
            raise IndexError(f"dataset_index out of range: {dataset_index}")
        return dict(self.manifest_rows[int(dataset_index)])

    def get_clip(self, dataset_index: int) -> TargetClip:
        key = int(dataset_index)
        cached = self._clips.get(key)
        if cached is not None:
            return cached

        row = self.get_row(int(dataset_index))
        shard_rel = str(row.get("pt") or "")
        if not shard_rel:
            raise KeyError(f"manifest row {dataset_index} missing shard path 'pt'")
        shard = self._load_shard(shard_rel)
        row_in_shard = int(row.get("row_in_shard", -1))
        if row_in_shard < 0:
            raise KeyError(f"manifest row {dataset_index} missing row_in_shard")

        if "target_audio_32k" not in shard:
            raise KeyError(f"shard {shard_rel} missing target_audio_32k")
        sample_rate = int(shard.get("target_audio_32k_sample_rate", SAMPLE_RATE))

        context_ms = float(shard.get("target_audio_32k_context_ms", EXPECTED_TARGET_AUDIO_CONTEXT_MS))
        if not np.isclose(float(context_ms), float(EXPECTED_TARGET_AUDIO_CONTEXT_MS), atol=1.0e-4):
            raise ValueError(
                f"expected target_audio_32k_context_ms={EXPECTED_TARGET_AUDIO_CONTEXT_MS}, found {context_ms} in {shard_rel}"
            )

        num_samples = int(shard["target_audio_32k_num_samples"][row_in_shard].item())
        beat_num_samples = int(shard["target_audio_32k_beat_num_samples"][row_in_shard].item())
        audio = (
            shard["target_audio_32k"][row_in_shard, 0, : int(num_samples)]
            .detach()
            .to(dtype=torch.float32)
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        mask = (
            shard["target_audio_32k_loss_mask"][row_in_shard, : int(num_samples)]
            .detach()
            .to(dtype=torch.bool)
            .cpu()
            .numpy()
            .astype(bool, copy=False)
        )
        left_context = int(shard["target_audio_32k_left_context_samples"][row_in_shard].item())

        beat_audio = audio[mask]
        if beat_audio.size == 0:
            beat_audio = audio[int(left_context) : int(left_context) + int(beat_num_samples)]
        beat_audio = _pad_or_trim(beat_audio, int(beat_num_samples))
        features = compute_audio_features(beat_audio, sample_rate=int(sample_rate))
        clip = TargetClip(row=row, audio=beat_audio, sample_rate=int(sample_rate), features=features)
        self._clips[key] = clip
        return clip

    def _load_shard(self, shard_rel: str) -> Dict[str, Any]:
        shard_key = str(shard_rel)
        cached = self._shards.get(shard_key)
        if cached is not None:
            return cached
        shard_path = (self.cache_root / shard_key).resolve()
        payload = torch.load(shard_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError(f"unexpected shard payload type at {shard_path}: {type(payload)!r}")
        self._shards[shard_key] = payload
        return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate 4-beat ablation prediction exports with drum-aware acoustic metrics.",
    )
    parser.add_argument(
        "ablations_root",
        type=Path,
        nargs="?",
        default=Path("final_pipeline/ablations_4beat_100epochs"),
        help="Root containing ablation run directories.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("final_pipeline/data/4beats_v9"),
        help="4-beat cache root containing manifest.jsonl and shard .pt files.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output root. Defaults to <ablations_root>/acoustic_eval.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of run directory names to evaluate.",
    )
    parser.add_argument("--max-items", type=int, default=0, help="Optional cap on the number of shared dataset indices.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device used for GPU-backed FAD embedding extraction; paired acoustic metrics run on CPU.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Remove an existing output directory before writing.")
    parser.add_argument("--skip-fad", action="store_true", help="Skip fadtk FAD-inf computation.")
    parser.add_argument("--fad-model", type=str, default="encodec-emb", help="fadtk embedding model to use.")
    parser.add_argument(
        "--fad-python",
        type=str,
        default=sys.executable,
        help="Python executable used to invoke `python -m fadtk.embeds` for missing embedding caches.",
    )
    parser.add_argument("--fad-workers", type=int, default=4, help="fadtk worker count.")
    parser.add_argument(
        "--fad-inf-workers",
        type=int,
        default=4,
        help="CPU worker count for deterministic FAD-inf repeat resampling.",
    )
    parser.add_argument("--fad-seed", type=int, default=DEFAULT_FAD_SEED, help="Base seed for deterministic FAD-inf and inference resampling.")
    parser.add_argument("--fad-repeats", type=int, default=DEFAULT_FAD_REPEATS, help="How many deterministic FAD-inf repeats to average.")
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help="Number of paired bootstrap resamples used for clip-level confidence intervals.",
    )
    parser.add_argument(
        "--permutation-samples",
        type=int,
        default=DEFAULT_PERMUTATION_SAMPLES,
        help="Number of paired sign-flip permutations used for best-vs-rest significance tests.",
    )
    parser.add_argument(
        "--inference-alpha",
        type=float,
        default=DEFAULT_INFERENCE_ALPHA,
        help="Alpha threshold used for confidence intervals and Holm-corrected significance.",
    )
    parser.add_argument("--skip-inference", action="store_true", help="Skip paired clip-level inference statistics.")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    parser.add_argument("--style-plot-top-k", type=int, default=STYLE_PLOT_TOP_K, help="How many styles to plot.")
    return parser.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    return dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(dict(json.loads(line)))
    return rows


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return float(out)


def _safe_divide(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or float(denominator) <= 0.0:
        return None
    return float(numerator) / float(denominator)


def _series_mean(df: pd.DataFrame, column: str) -> Optional[float]:
    if column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _series_max(df: pd.DataFrame, columns: Sequence[str]) -> Optional[float]:
    present = [column for column in columns if column in df.columns]
    if not present:
        return None
    values = pd.to_numeric(df.loc[:, present].stack(dropna=True), errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.max())


def _load_checkpoint_num_parameters(run_dir: Path) -> Tuple[Optional[int], bool]:
    for checkpoint_name in ("best.pt", "last.pt"):
        checkpoint_path = (Path(run_dir) / checkpoint_name).resolve()
        if not checkpoint_path.is_file():
            continue
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception:
            return None, False
        direct_value = checkpoint.get("num_parameters")
        if direct_value is not None:
            parsed = _safe_float(direct_value)
            if parsed is not None:
                return int(round(float(parsed))), True
        model_state = checkpoint.get("model_state")
        if isinstance(model_state, Mapping):
            total = 0
            for tensor in list(model_state.values()):
                if torch.is_tensor(tensor):
                    total += int(tensor.numel())
            if int(total) > 0:
                return int(total), True
        break
    return None, False


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(payload), indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _write_markdown(path: Path, title: str, df: pd.DataFrame) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        body = df.to_markdown(index=False)
    except Exception:
        body = "```\n" + df.to_string(index=False) + "\n```"
    target.write_text(f"# {title}\n\n{body}\n", encoding="utf-8")


def _json_dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _seed_schedule(seed: int, repeats: int) -> List[int]:
    count = int(max(1, int(repeats)))
    start = int(seed)
    return [int(start + idx) for idx in range(count)]


def _percentile_interval(values: Sequence[float], alpha: float) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return (math.nan, math.nan)
    lo = 100.0 * float(alpha) / 2.0
    hi = 100.0 * (1.0 - (float(alpha) / 2.0))
    return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))


def _sample_std(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64).reshape(-1)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _sanitize_name(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "item"
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "item"


def _clip_file_name(dataset_index: int, source_id: str, beat_index: int) -> str:
    return f"{int(dataset_index):06d}__{_sanitize_name(source_id)}__beat_{int(beat_index):04d}.wav"


def _pad_or_trim(audio: np.ndarray, target_len: int) -> np.ndarray:
    wav = np.asarray(audio, dtype=np.float32).reshape(-1)
    tgt = int(max(1, int(target_len)))
    if wav.shape[0] == tgt:
        return wav.astype(np.float32, copy=False)
    if wav.shape[0] > tgt:
        return wav[:tgt].astype(np.float32, copy=False)
    out = np.zeros((tgt,), dtype=np.float32)
    out[: wav.shape[0]] = wav
    return out


def _load_audio_mono(path: Path) -> Tuple[np.ndarray, int]:
    wav, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    audio = np.asarray(wav, dtype=np.float32)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    elif audio.ndim != 1:
        audio = np.asarray(audio).reshape(-1).astype(np.float32, copy=False)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def _cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    n = int(min(xv.shape[0], yv.shape[0]))
    if n <= 0:
        return 0.0
    xv = xv[:n]
    yv = yv[:n]
    denom = float(np.linalg.norm(xv) * np.linalg.norm(yv))
    if not np.isfinite(denom) or denom <= EPS:
        return 0.0
    return float(np.dot(xv, yv) / denom)


def _aligned_mel_mae(pred_mel: np.ndarray, target_mel: np.ndarray) -> float:
    pm = np.asarray(pred_mel, dtype=np.float32)
    tm = np.asarray(target_mel, dtype=np.float32)
    frames = int(min(pm.shape[-1], tm.shape[-1]))
    if frames <= 0:
        return 0.0
    return float(np.mean(np.abs(pm[:, :frames] - tm[:, :frames])))


def _amplitude_to_db(value: float) -> float:
    if not np.isfinite(float(value)):
        return float(RMS_DB_FLOOR)
    return float(max(float(RMS_DB_FLOOR), 20.0 * np.log10(max(float(value), EPS))))


def _mel_filter_bank(sample_rate: int) -> torch.Tensor:
    sr = int(sample_rate)
    cached = _MEL_BANK_CACHE.get(sr)
    if cached is not None:
        return cached
    cached = AF.melscale_fbanks(
        n_freqs=1 + (int(N_FFT) // 2),
        f_min=float(MEL_FMIN),
        f_max=float(min(float(MEL_FMAX), 0.5 * float(sr))),
        n_mels=int(N_MELS),
        sample_rate=int(sr),
        norm=None,
        mel_scale="htk",
    ).to(dtype=torch.float32)
    _MEL_BANK_CACHE[sr] = cached
    return cached


def _fft_frequencies(sample_rate: int) -> np.ndarray:
    sr = int(sample_rate)
    cached = _FFT_FREQ_CACHE.get(sr)
    if cached is not None:
        return cached
    cached = np.fft.rfftfreq(int(N_FFT), d=1.0 / float(sr)).astype(np.float32)
    _FFT_FREQ_CACHE[sr] = cached
    return cached


def _resample_audio(audio: np.ndarray, *, orig_sr: int, target_sr: int) -> np.ndarray:
    if int(orig_sr) == int(target_sr):
        return np.asarray(audio, dtype=np.float32).reshape(-1)
    waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32).reshape(1, -1))
    resampled = AF.resample(
        waveform,
        orig_freq=int(orig_sr),
        new_freq=int(target_sr),
        lowpass_filter_width=64,
        rolloff=0.9475937167399596,
        resampling_method="sinc_interp_kaiser",
        beta=14.769656459379492,
    )
    return resampled.squeeze(0).cpu().numpy().astype(np.float32, copy=False)


def compute_audio_features(audio: np.ndarray, *, sample_rate: int) -> AudioFeatures:
    wav = np.asarray(audio, dtype=np.float32).reshape(-1)
    if wav.size == 0:
        wav = np.zeros((1,), dtype=np.float32)

    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    wav_norm = wav / max(peak, EPS)

    waveform = torch.from_numpy(wav_norm.astype(np.float32, copy=False)).reshape(1, -1)
    if int(waveform.shape[-1]) < int(WIN_LENGTH):
        waveform = torch.nn.functional.pad(waveform, (0, int(WIN_LENGTH) - int(waveform.shape[-1])))

    stft_complex = torch.stft(
        waveform,
        n_fft=int(N_FFT),
        hop_length=int(HOP_LENGTH),
        win_length=int(WIN_LENGTH),
        window=STFT_WINDOW,
        center=True,
        return_complex=True,
    ).squeeze(0)
    mag_t = stft_complex.abs().to(dtype=torch.float32)
    if mag_t.ndim != 2 or int(mag_t.shape[1]) <= 0:
        mag_t = torch.zeros((1 + (int(N_FFT) // 2), 1), dtype=torch.float32)

    power_t = mag_t.pow(2.0)
    mel_t = torch.matmul(_mel_filter_bank(int(sample_rate)).transpose(0, 1), power_t)
    mel_db = (10.0 * torch.log10(torch.clamp(mel_t, min=float(EPS)))).cpu().numpy().astype(np.float32, copy=False)

    diff_t = mag_t[:, 1:] - mag_t[:, :-1]
    if int(diff_t.shape[1]) <= 0:
        flux = np.zeros((1,), dtype=np.float32)
    else:
        flux = torch.clamp(diff_t, min=0.0).mean(dim=0).cpu().numpy().astype(np.float32, copy=False)

    freqs = _fft_frequencies(int(sample_rate))
    band_flux: Dict[str, np.ndarray] = {}
    band_energy: List[float] = []
    for band_name, lo_hz, hi_hz, _desc in BAND_SPECS:
        mask = np.logical_and(freqs >= float(lo_hz), freqs < float(hi_hz))
        if not np.any(mask):
            band_flux[str(band_name)] = np.zeros((1,), dtype=np.float32)
            band_energy.append(0.0)
            continue
        band_mag = mag_t[mask, :]
        if int(diff_t.shape[1]) <= 0:
            band_flux[str(band_name)] = np.zeros((1,), dtype=np.float32)
        else:
            band_diff_t = diff_t[mask, :]
            band_flux[str(band_name)] = (
                torch.clamp(band_diff_t, min=0.0).mean(dim=0).cpu().numpy().astype(np.float32, copy=False)
            )
        band_energy.append(float(band_mag.pow(2.0).mean().item()) if int(band_mag.numel()) > 0 else 0.0)

    band_energy_arr = np.asarray(band_energy, dtype=np.float32)
    band_energy_ratio = band_energy_arr / max(float(np.sum(band_energy_arr)), EPS)

    freq_t = torch.from_numpy(freqs).to(dtype=torch.float32, device=mag_t.device).unsqueeze(1)
    centroid_t = (freq_t * mag_t).sum(dim=0) / torch.clamp(mag_t.sum(dim=0), min=float(EPS))
    centroid_hz = float(centroid_t.mean().item()) if int(centroid_t.numel()) > 0 else 0.0

    wav64 = wav.astype(np.float64, copy=False)
    rms = float(np.sqrt(float(np.mean(np.square(wav64))) + EPS))
    wav_norm64 = wav_norm.astype(np.float64, copy=False)
    rms_norm = float(np.sqrt(float(np.mean(np.square(wav_norm64))) + EPS))
    peak_abs = float(np.max(np.abs(wav64))) if wav64.size else 0.0
    rms_dbfs = _amplitude_to_db(rms)
    rms_norm_dbfs = _amplitude_to_db(rms_norm)
    crest_db = float(max(0.0, _amplitude_to_db(peak_abs) - rms_dbfs))

    return AudioFeatures(
        mel_db=mel_db,
        flux=flux,
        band_flux=band_flux,
        band_energy_ratio=band_energy_ratio.astype(np.float32, copy=False),
        centroid_hz=centroid_hz,
        rms_dbfs=rms_dbfs,
        rms_norm_dbfs=rms_norm_dbfs,
        crest_db=crest_db,
    )


def discover_runs(ablations_root: Path, selected_models: Optional[Sequence[str]]) -> List[RunPredictions]:
    root = Path(ablations_root).resolve()
    selected = {str(name) for name in list(selected_models or [])}
    discovered: List[RunPredictions] = []
    for manifest_path in sorted(root.glob("*/test_set_predictions/manifest.jsonl")):
        run_dir = manifest_path.parent.parent.resolve()
        name = run_dir.name
        if selected and str(name) not in selected:
            continue
        rows = _read_jsonl(manifest_path)
        rows_by_dataset_index: Dict[int, Dict[str, Any]] = {}
        for row in rows:
            dataset_index = int(row.get("dataset_index", -1))
            if dataset_index < 0:
                raise ValueError(f"{manifest_path} contains row without a valid dataset_index")
            if dataset_index in rows_by_dataset_index:
                raise ValueError(f"{manifest_path} contains duplicate dataset_index={dataset_index}")
            rows_by_dataset_index[dataset_index] = dict(row)
        wav_root = manifest_path.parent.resolve()
        discovered.append(
            RunPredictions(
                name=str(name),
                run_dir=run_dir,
                manifest_path=manifest_path.resolve(),
                wav_root=wav_root,
                rows_by_dataset_index=rows_by_dataset_index,
            )
        )

    if selected:
        found = {run.name for run in discovered}
        missing = sorted(set(selected) - found)
        if missing:
            raise FileNotFoundError(f"requested models not found under {root}: {missing}")
    if not discovered:
        raise FileNotFoundError(f"no test_set_predictions manifests found under {root}")
    return discovered


def shared_dataset_indices(runs: Sequence[RunPredictions], max_items: int) -> List[int]:
    shared: Optional[set[int]] = None
    for run in runs:
        dataset_indices = set(int(x) for x in run.rows_by_dataset_index.keys())
        shared = dataset_indices if shared is None else (shared & dataset_indices)
    out = sorted(shared or [])
    if int(max_items) > 0:
        out = out[: int(max_items)]
    if not out:
        raise RuntimeError("no shared dataset_index values across selected runs")
    return out


def evaluate_run(
    run: RunPredictions,
    *,
    dataset_indices: Sequence[int],
    target_cache: TargetAudioCache,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    progress = tqdm(dataset_indices, desc=f"paired_metrics[{run.name}]", unit="clip")
    for dataset_index in progress:
        pred_row = dict(run.rows_by_dataset_index[int(dataset_index)])
        pred_wav_path = (run.wav_root / str(pred_row.get("wav") or "")).resolve()
        if not pred_wav_path.is_file():
            raise FileNotFoundError(f"prediction wav missing: {pred_wav_path}")
        target_clip = target_cache.get_clip(int(dataset_index))
        target_row = dict(target_clip.row)
        target_sr = int(target_clip.sample_rate)
        pred_audio, pred_sr = _load_audio_mono(pred_wav_path)
        if int(pred_sr) != int(target_sr):
            pred_audio = _resample_audio(pred_audio, orig_sr=int(pred_sr), target_sr=int(target_sr))
            pred_sr = int(target_sr)
        target_audio = target_clip.audio
        pred_audio_aligned = _pad_or_trim(pred_audio, int(target_audio.shape[0]))
        pred_features = compute_audio_features(pred_audio_aligned, sample_rate=int(pred_sr))
        tgt_features = target_clip.features

        pred_band_ratio = pred_features.band_energy_ratio
        tgt_band_ratio = tgt_features.band_energy_ratio
        band_balance_l1 = 0.5 * float(np.sum(np.abs(pred_band_ratio - tgt_band_ratio)))

        result: Dict[str, Any] = {
            "model": str(run.name),
            "dataset_index": int(dataset_index),
            "source_id": str(target_row.get("source_id", pred_row.get("source_id", ""))),
            "beat_index": int(target_row.get("beat_index", pred_row.get("beat_index", -1))),
            "split": str(target_row.get("split", pred_row.get("split", ""))),
            "style": str(target_row.get("style", "")),
            "drummer": str(target_row.get("drummer", "")),
            "source_manifest_index": int(target_row.get("source_manifest_index", pred_row.get("source_manifest_index", -1))),
            "pred_wav": str(pred_wav_path),
            "sample_rate": int(target_sr),
            "pred_num_samples": int(pred_audio.shape[0]),
            "target_num_samples": int(target_audio.shape[0]),
            "length_delta_ms": float(1000.0 * float(pred_audio.shape[0] - target_audio.shape[0]) / float(target_sr)),
            "mel_mae_db": _aligned_mel_mae(pred_features.mel_db, tgt_features.mel_db),
            "onset_flux_cosine": _cosine_similarity(pred_features.flux, tgt_features.flux),
            "band_balance_l1": float(band_balance_l1),
            "centroid_mae_hz": float(abs(pred_features.centroid_hz - tgt_features.centroid_hz)),
            "rms_mae_db": float(abs(pred_features.rms_dbfs - tgt_features.rms_dbfs)),
            "rms_norm_mae_db": float(abs(pred_features.rms_norm_dbfs - tgt_features.rms_norm_dbfs)),
            "crest_mae_db": float(abs(pred_features.crest_db - tgt_features.crest_db)),
        }
        for band_idx, band_name in enumerate(BAND_NAMES):
            result[f"{band_name}_flux_cosine"] = _cosine_similarity(
                pred_features.band_flux[str(band_name)],
                tgt_features.band_flux[str(band_name)],
            )
            result[f"pred_{band_name}_ratio"] = float(pred_band_ratio[band_idx])
            result[f"target_{band_name}_ratio"] = float(tgt_band_ratio[band_idx])
            result[f"{band_name}_ratio_mae"] = float(abs(pred_band_ratio[band_idx] - tgt_band_ratio[band_idx]))
        rows.append(result)
    return pd.DataFrame(rows)


def build_display_table(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.loc[:, [col for col in columns if col in df.columns]].copy()
    for column in list(out.columns):
        if column in METRIC_SPECS:
            decimals = int(METRIC_SPECS[column]["decimals"])
            out[column] = out[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.{decimals}f}")
        elif column in INT_DISPLAY_COLUMNS:
            out[column] = out[column].map(lambda value: "" if pd.isna(value) else str(int(round(float(value)))))
    renamed = {column: str(METRIC_SPECS[column]["label"]) for column in out.columns if column in METRIC_SPECS}
    return out.rename(columns=renamed)


def aggregate_overall(
    per_clip_df: pd.DataFrame,
    fad_rows: Optional[pd.DataFrame] = None,
    efficiency_rows: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    metric_columns = [
        "mel_mae_db",
        "onset_flux_cosine",
        "low_flux_cosine",
        "mid_flux_cosine",
        "high_flux_cosine",
        "band_balance_l1",
        "centroid_mae_hz",
        "rms_mae_db",
        "rms_norm_mae_db",
        "crest_mae_db",
    ]
    overall = per_clip_df.groupby("model", as_index=False)[metric_columns].mean(numeric_only=True)
    counts = per_clip_df.groupby("model", as_index=False).size().rename(columns={"size": "num_examples"})
    overall = counts.merge(overall, on="model", how="left")
    if fad_rows is not None and not fad_rows.empty:
        overall = overall.merge(fad_rows, on="model", how="left")
    if efficiency_rows is not None and not efficiency_rows.empty:
        merge_columns = [
            column
            for column in [
                "model",
                "rtf_end_to_end",
                "audio_sec_per_sec",
                "total_train_wall_sec",
                "time_to_best_checkpoint_sec",
                "peak_gpu_mem_allocated_mb",
            ]
            if column in efficiency_rows.columns
        ]
        overall = overall.merge(efficiency_rows.loc[:, merge_columns], on="model", how="left")
    sort_columns = [column for column in ["fad_inf", "mel_mae_db", "band_balance_l1"] if column in overall.columns]
    if sort_columns:
        overall = overall.sort_values(sort_columns, ascending=[True] * len(sort_columns), kind="stable").reset_index(drop=True)
    return overall


def aggregate_efficiency(runs: Sequence[RunPredictions]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    availability: List[Dict[str, Any]] = []
    progress = tqdm(runs, desc="efficiency_summary", unit="model", leave=False)
    for run in progress:
        export_summary_path = run.manifest_path.parent / "summary.json"
        history_path = run.run_dir / "history.csv"
        config_path = run.run_dir / "config.json"

        export_summary = (_read_json(export_summary_path) if export_summary_path.is_file() else {})
        history_df = (pd.read_csv(history_path) if history_path.is_file() else pd.DataFrame())
        config = (_read_json(config_path) if config_path.is_file() else {})

        if not history_df.empty and "epoch" in history_df.columns:
            history_df = history_df.copy()
            history_df["epoch"] = pd.to_numeric(history_df["epoch"], errors="coerce")
            history_df = history_df.sort_values("epoch", kind="stable").reset_index(drop=True)

        best_checkpoint_epoch: Optional[float] = None
        if not history_df.empty and "best_checkpoint_epoch" in history_df.columns:
            best_checkpoint_epoch = _safe_float(history_df["best_checkpoint_epoch"].dropna().iloc[-1]) if not history_df["best_checkpoint_epoch"].dropna().empty else None
        if best_checkpoint_epoch is None and not history_df.empty and "checkpoint_improved" in history_df.columns and "epoch" in history_df.columns:
            improved = history_df[history_df["checkpoint_improved"] == True]  # noqa: E712
            if not improved.empty:
                best_checkpoint_epoch = _safe_float(improved.iloc[-1].get("epoch"))

        total_train_wall_sec = None if history_df.empty else _safe_float(pd.to_numeric(history_df.get("train_elapsed_sec"), errors="coerce").sum()) if "train_elapsed_sec" in history_df.columns else None
        total_val_wall_sec = None if history_df.empty else _safe_float(pd.to_numeric(history_df.get("val_elapsed_sec"), errors="coerce").sum()) if "val_elapsed_sec" in history_df.columns else None

        time_to_best_checkpoint_sec: Optional[float] = None
        if best_checkpoint_epoch is not None and not history_df.empty and "epoch" in history_df.columns:
            upto_best = history_df[pd.to_numeric(history_df["epoch"], errors="coerce") <= float(best_checkpoint_epoch)]
            if not upto_best.empty:
                train_to_best = _safe_float(pd.to_numeric(upto_best.get("train_elapsed_sec"), errors="coerce").sum()) if "train_elapsed_sec" in upto_best.columns else 0.0
                val_to_best = _safe_float(pd.to_numeric(upto_best.get("val_elapsed_sec"), errors="coerce").sum()) if "val_elapsed_sec" in upto_best.columns else 0.0
                time_to_best_checkpoint_sec = _safe_float(float(train_to_best or 0.0) + float(val_to_best or 0.0))

        mean_train_steps_per_sec = _series_mean(history_df, "train_steps_per_sec")
        if mean_train_steps_per_sec is None and not history_df.empty and {"train_steps", "train_elapsed_sec"}.issubset(history_df.columns):
            train_steps = pd.to_numeric(history_df["train_steps"], errors="coerce")
            train_elapsed = pd.to_numeric(history_df["train_elapsed_sec"], errors="coerce")
            values = (train_steps / train_elapsed.replace(0.0, np.nan)).dropna()
            if not values.empty:
                mean_train_steps_per_sec = float(values.mean())

        mean_train_tokens_per_sec = _series_mean(history_df, "train_tokens_per_sec")
        if mean_train_tokens_per_sec is None and not history_df.empty and {"train_valid_tokens_seen", "train_elapsed_sec"}.issubset(history_df.columns):
            train_tokens = pd.to_numeric(history_df["train_valid_tokens_seen"], errors="coerce")
            train_elapsed = pd.to_numeric(history_df["train_elapsed_sec"], errors="coerce")
            values = (train_tokens / train_elapsed.replace(0.0, np.nan)).dropna()
            if not values.empty:
                mean_train_tokens_per_sec = float(values.mean())

        mean_train_audio_seconds_per_sec = _series_mean(history_df, "train_audio_seconds_per_sec")
        if mean_train_audio_seconds_per_sec is None and not history_df.empty and {"train_audio_seconds_seen", "train_elapsed_sec"}.issubset(history_df.columns):
            train_audio_seconds = pd.to_numeric(history_df["train_audio_seconds_seen"], errors="coerce")
            train_elapsed = pd.to_numeric(history_df["train_elapsed_sec"], errors="coerce")
            values = (train_audio_seconds / train_elapsed.replace(0.0, np.nan)).dropna()
            if not values.empty:
                mean_train_audio_seconds_per_sec = float(values.mean())

        peak_gpu_mem_allocated_mb = _series_max(
            history_df,
            ["train_peak_gpu_mem_allocated_mb", "val_peak_gpu_mem_allocated_mb"],
        )
        peak_gpu_mem_reserved_mb = _series_max(
            history_df,
            ["train_peak_gpu_mem_reserved_mb", "val_peak_gpu_mem_reserved_mb"],
        )

        num_parameters = _safe_float(export_summary.get("num_parameters"))
        if num_parameters is None:
            num_parameters = _safe_float(config.get("num_parameters"))
        checkpoint_num_parameters_loaded = False
        if num_parameters is None:
            checkpoint_num_parameters, checkpoint_num_parameters_loaded = _load_checkpoint_num_parameters(run.run_dir)
            if checkpoint_num_parameters is not None:
                num_parameters = float(checkpoint_num_parameters)

        export_wall_sec_total = _safe_float(export_summary.get("export_wall_sec_total"))
        model_forward_sec_total = _safe_float(export_summary.get("model_forward_sec_total"))
        codec_decode_sec_total = _safe_float(export_summary.get("codec_decode_sec_total"))
        total_audio_sec_generated = _safe_float(export_summary.get("total_audio_sec_generated"))
        clips_per_sec = _safe_float(export_summary.get("clips_per_sec"))
        audio_sec_per_sec = _safe_float(export_summary.get("audio_sec_per_sec"))
        if audio_sec_per_sec is None:
            audio_sec_per_sec = _safe_divide(total_audio_sec_generated, export_wall_sec_total)
        rtf_end_to_end = _safe_float(export_summary.get("rtf_end_to_end"))
        if rtf_end_to_end is None:
            rtf_end_to_end = _safe_divide(export_wall_sec_total, total_audio_sec_generated)
        rtf_model_only = _safe_float(export_summary.get("rtf_model_only"))
        if rtf_model_only is None:
            rtf_model_only = _safe_divide(model_forward_sec_total, total_audio_sec_generated)
        export_peak_gpu_mem_allocated_mb = _safe_float(export_summary.get("peak_gpu_mem_allocated_mb"))
        export_peak_gpu_mem_reserved_mb = _safe_float(export_summary.get("peak_gpu_mem_reserved_mb"))

        row = {
            "model": str(run.name),
            "device_name": str(export_summary.get("device_name") or export_summary.get("device") or config.get("device") or ""),
            "batch_size": _safe_float(export_summary.get("batch_size")),
            "num_examples": _safe_float(export_summary.get("num_examples")),
            "best_checkpoint_epoch": best_checkpoint_epoch,
            "num_parameters": num_parameters,
            "export_wall_sec_total": export_wall_sec_total,
            "model_forward_sec_total": model_forward_sec_total,
            "codec_decode_sec_total": codec_decode_sec_total,
            "total_audio_sec_generated": total_audio_sec_generated,
            "clips_per_sec": clips_per_sec,
            "audio_sec_per_sec": audio_sec_per_sec,
            "rtf_end_to_end": rtf_end_to_end,
            "rtf_model_only": rtf_model_only,
            "export_peak_gpu_mem_allocated_mb": export_peak_gpu_mem_allocated_mb,
            "export_peak_gpu_mem_reserved_mb": export_peak_gpu_mem_reserved_mb,
            "total_train_wall_sec": total_train_wall_sec,
            "total_val_wall_sec": total_val_wall_sec,
            "time_to_best_checkpoint_sec": time_to_best_checkpoint_sec,
            "mean_train_steps_per_sec": mean_train_steps_per_sec,
            "mean_train_tokens_per_sec": mean_train_tokens_per_sec,
            "mean_train_audio_seconds_per_sec": mean_train_audio_seconds_per_sec,
            "peak_gpu_mem_allocated_mb": peak_gpu_mem_allocated_mb,
            "peak_gpu_mem_reserved_mb": peak_gpu_mem_reserved_mb,
        }
        rows.append(row)
        availability.append(
            {
                "model": str(run.name),
                "export_summary_found": bool(export_summary_path.is_file()),
                "history_found": bool(history_path.is_file()),
                "config_found": bool(config_path.is_file()),
                "checkpoint_num_parameters_loaded": bool(checkpoint_num_parameters_loaded),
                "efficiency_fields_found": sorted([str(key) for key, value in row.items() if str(key) != "model" and value is not None]),
            }
        )

    efficiency_df = pd.DataFrame(rows)
    if not efficiency_df.empty:
        efficiency_df = efficiency_df.sort_values("model", kind="stable").reset_index(drop=True)
    return efficiency_df, availability


def aggregate_band_summary(per_clip_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = ["low_flux_cosine", "mid_flux_cosine", "high_flux_cosine", "band_balance_l1"]
    band_summary = per_clip_df.groupby("model", as_index=False)[metric_columns].mean(numeric_only=True)
    return band_summary.sort_values("model", kind="stable").reset_index(drop=True)


def aggregate_band_profile(per_clip_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    pred_cols = [f"pred_{band_name}_ratio" for band_name in BAND_NAMES]
    target_cols = [f"target_{band_name}_ratio" for band_name in BAND_NAMES]
    pred_profile = per_clip_df.groupby("model", as_index=False)[pred_cols].mean(numeric_only=True)
    target_profile = {
        str(band_name): float(per_clip_df[f"target_{band_name}_ratio"].mean())
        for band_name in BAND_NAMES
        if f"target_{band_name}_ratio" in per_clip_df.columns
    }
    return pred_profile, target_profile


def aggregate_style_summary(per_clip_df: pd.DataFrame) -> pd.DataFrame:
    style_summary = (
        per_clip_df.groupby(["model", "style"], as_index=False)[["mel_mae_db", "onset_flux_cosine", "band_balance_l1"]]
        .mean(numeric_only=True)
        .sort_values(["style", "model"], kind="stable")
        .reset_index(drop=True)
    )
    counts = per_clip_df.groupby("style", as_index=False).size().rename(columns={"size": "num_examples"})
    return style_summary.merge(counts, on="style", how="left")


def _format_annot(value: float, metric_name: str) -> str:
    decimals = int(METRIC_SPECS.get(metric_name, {}).get("decimals", 3))
    if pd.isna(value):
        return ""
    return f"{float(value):.{decimals}f}"


def save_table_png(df: pd.DataFrame, path: Path, *, title: str) -> None:
    frame = df.copy()
    fig_height = max(1.8, 0.5 * (len(frame) + 1))
    fig_width = max(8.0, 1.2 * len(frame.columns))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=frame.values,
        colLabels=list(frame.columns),
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    ax.set_title(title, fontsize=12, pad=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_metric_heatmap(df: pd.DataFrame, path: Path, *, metrics: Sequence[str], title: str) -> None:
    metric_columns = [metric for metric in metrics if metric in df.columns]
    if not metric_columns:
        return
    raw = df.set_index("model")[metric_columns].astype(float)
    signed = raw.copy()
    for metric in metric_columns:
        if METRIC_SPECS.get(metric, {}).get("better") == "lower":
            signed[metric] = -signed[metric]
    if len(signed.index) > 1:
        denom = signed.std(axis=0, ddof=0).replace(0.0, 1.0)
        normalized = (signed - signed.mean(axis=0)) / denom
    else:
        normalized = signed * 0.0

    annot = np.empty(raw.shape, dtype=object)
    for row_idx in range(raw.shape[0]):
        for col_idx, metric in enumerate(metric_columns):
            annot[row_idx, col_idx] = _format_annot(float(raw.iloc[row_idx, col_idx]), metric)

    fig_width = max(7.5, 1.35 * len(metric_columns) + 1.5)
    fig_height = max(2.5, 0.6 * len(raw.index) + 1.4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(normalized.to_numpy(dtype=float), aspect="auto", cmap="RdYlGn")
    ax.set_xticks(np.arange(len(metric_columns)))
    ax.set_yticks(np.arange(len(raw.index)))
    ax.set_xticklabels([METRIC_SPECS.get(metric, {}).get("label", metric) for metric in metric_columns], rotation=45, ha="right")
    ax.set_yticklabels(list(raw.index))
    ax.set_title(title, fontsize=12, pad=12)
    for row_idx in range(raw.shape[0]):
        for col_idx in range(raw.shape[1]):
            ax.text(col_idx, row_idx, annot[row_idx, col_idx], ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("direction-aware z-score", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_band_profile_plot(pred_profile: pd.DataFrame, target_profile: Mapping[str, float], path: Path) -> None:
    if pred_profile.empty:
        return
    x = np.arange(len(BAND_NAMES), dtype=np.float32)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    target_values = [float(target_profile.get(str(name), 0.0)) for name in BAND_NAMES]
    ax.plot(x, target_values, marker="o", linewidth=2.5, label="target_mean", color="black")
    for row in pred_profile.itertuples(index=False):
        model_name = str(getattr(row, "model"))
        y = [float(getattr(row, f"pred_{band_name}_ratio")) for band_name in BAND_NAMES]
        ax.plot(x, y, marker="o", linewidth=1.8, alpha=0.9, label=model_name)
    ax.set_xticks(x)
    ax.set_xticklabels([str(name) for name in BAND_NAMES])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("mean spectral energy ratio")
    ax.set_title("Band Balance Profiles")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_group_metric_heatmap(
    df: pd.DataFrame,
    *,
    group_column: str,
    value_column: str,
    path: Path,
    title: str,
    top_k: int,
) -> None:
    if df.empty or value_column not in df.columns or group_column not in df.columns:
        return
    counts = df.groupby(group_column).size().sort_values(ascending=False)
    keep_groups = list(counts.head(int(max(1, int(top_k)))).index)
    subset = df[df[group_column].isin(keep_groups)].copy()
    if subset.empty:
        return
    pivot = subset.pivot_table(index=group_column, columns="model", values=value_column, aggfunc="mean")
    pivot = pivot.loc[keep_groups]
    signed = pivot.copy()
    if METRIC_SPECS.get(value_column, {}).get("better") == "lower":
        signed = -signed
    if len(signed.index) > 1:
        denom = signed.std(axis=0, ddof=0).replace(0.0, 1.0)
        normalized = (signed - signed.mean(axis=0)) / denom
    else:
        normalized = signed * 0.0

    annot = np.empty(pivot.shape, dtype=object)
    for row_idx in range(pivot.shape[0]):
        for col_idx in range(pivot.shape[1]):
            annot[row_idx, col_idx] = _format_annot(float(pivot.iloc[row_idx, col_idx]), value_column)

    fig, ax = plt.subplots(figsize=(max(7.0, 1.2 * pivot.shape[1] + 2.0), max(3.0, 0.45 * pivot.shape[0] + 1.5)))
    im = ax.imshow(normalized.to_numpy(dtype=float), aspect="auto", cmap="RdYlGn")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_xticklabels(list(pivot.columns), rotation=45, ha="right")
    ax.set_yticklabels(list(pivot.index))
    ax.set_title(title)
    for row_idx in range(pivot.shape[0]):
        for col_idx in range(pivot.shape[1]):
            ax.text(col_idx, row_idx, annot[row_idx, col_idx], ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("direction-aware z-score", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _make_link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def materialize_reference_wavs(
    reference_dir: Path,
    *,
    target_cache: TargetAudioCache,
    dataset_indices: Sequence[int],
) -> None:
    reference_dir.mkdir(parents=True, exist_ok=True)
    progress = tqdm(dataset_indices, desc="write_reference_audio", unit="clip", leave=False)
    for dataset_index in progress:
        target_clip = target_cache.get_clip(int(dataset_index))
        row = target_clip.row
        out_path = reference_dir / _clip_file_name(
            int(dataset_index),
            str(row.get("source_id", "")),
            int(row.get("beat_index", -1)),
        )
        if out_path.is_file():
            continue
        sf.write(str(out_path), target_clip.audio.astype(np.float32, copy=False), int(target_clip.sample_rate), subtype="PCM_16")


def materialize_eval_links(
    eval_dir: Path,
    *,
    run: RunPredictions,
    dataset_indices: Sequence[int],
) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)
    progress = tqdm(dataset_indices, desc=f"link_eval_audio[{run.name}]", unit="clip", leave=False)
    for dataset_index in progress:
        row = dict(run.rows_by_dataset_index[int(dataset_index)])
        src = (run.wav_root / str(row.get("wav") or "")).resolve()
        dst = eval_dir / _clip_file_name(
            int(dataset_index),
            str(row.get("source_id", "")),
            int(row.get("beat_index", -1)),
        )
        _make_link_or_copy(src, dst)


def _embedding_cache_path(audio_path: Path, fad_model: str) -> Path:
    audio_file = Path(audio_path)
    return audio_file.parent / "embeddings" / str(fad_model) / audio_file.with_suffix(".npy").name


def _sorted_top_level_files(path: Path) -> List[Path]:
    root = Path(path)
    return sorted([item for item in root.glob("*.*") if item.is_file()], key=lambda item: item.name)


def _expected_embedding_files(audio_dir: Path, fad_model: str) -> List[Path]:
    return [_embedding_cache_path(audio_file, fad_model) for audio_file in _sorted_top_level_files(audio_dir)]


def _file_metadata(path: Path, *, base_dir: Optional[Path] = None) -> Dict[str, Any]:
    file_path = Path(path)
    stat = file_path.stat()
    label = str(file_path.relative_to(base_dir)) if base_dir is not None else file_path.name
    digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
    return {
        "path": label,
        "size": int(stat.st_size),
        "sha256": str(digest),
    }


def _build_file_manifest(files: Sequence[Path], *, base_dir: Optional[Path] = None) -> Dict[str, Any]:
    rows = [_file_metadata(Path(path), base_dir=base_dir) for path in files]
    return {
        "num_files": int(len(rows)),
        "files": rows,
        "hash": _sha256_text(_json_dumps({"files": rows})),
    }


def _load_embedding_array(path: Path) -> np.ndarray:
    arr = np.asarray(np.load(path), dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D embedding array at {path}, found shape {arr.shape}")
    return arr


def _load_embedding_matrix(files: Sequence[Path]) -> np.ndarray:
    arrays = [_load_embedding_array(path) for path in files]
    if not arrays:
        raise ValueError("no embedding files provided")
    return np.concatenate(arrays, axis=0).astype(np.float64, copy=False)


def _cuda_visible_devices_for_device(device: str) -> Optional[str]:
    text = str(device or "").strip().lower()
    if text == "cpu":
        return ""
    if not text.startswith("cuda"):
        return None
    try:
        torch_device = torch.device(text)
    except Exception:
        return None
    if torch_device.index is None:
        return None

    visible = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    if visible:
        entries = [part.strip() for part in visible.split(",") if part.strip()]
        if 0 <= int(torch_device.index) < len(entries):
            return entries[int(torch_device.index)]
    return str(int(torch_device.index))


def _calc_embedding_stats(embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    data = np.asarray(embeddings, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2 or int(data.shape[0]) < 2:
        raise ValueError(f"FAD requires at least two embedding frames, found shape {data.shape}")
    return np.mean(data, axis=0), np.cov(data, rowvar=False)


def _compute_embedding_stats_online(
    embedding_files: Sequence[Path],
    *,
    desc: str = "embedding_stats",
) -> Tuple[np.ndarray, np.ndarray, int]:
    mu: Optional[np.ndarray] = None
    scatter: Optional[np.ndarray] = None
    total = 0
    progress = tqdm(embedding_files, desc=str(desc), unit="file", leave=False)
    for path in progress:
        emb = _load_embedding_array(path)
        count = int(emb.shape[0])
        if count <= 0:
            continue
        local_mu = np.mean(emb, axis=0)
        if count > 1:
            local_scatter = np.cov(emb, rowvar=False) * float(count - 1)
        else:
            local_scatter = np.zeros((int(local_mu.shape[0]), int(local_mu.shape[0])), dtype=np.float64)
        if mu is None:
            mu = np.zeros_like(local_mu, dtype=np.float64)
            scatter = np.zeros_like(local_scatter, dtype=np.float64)
        assert scatter is not None
        delta = local_mu - mu
        denom = int(total + count)
        mu = mu + (float(count) / float(denom)) * delta
        scatter = scatter + local_scatter + np.outer(delta, delta) * (float(total * count) / float(denom))
        total = denom
    if mu is None or scatter is None or total <= 0:
        raise ValueError("no embedding frames were available to compute statistics")
    cov = (scatter / float(total - 1)) if total > 1 else np.zeros_like(scatter)
    return mu.astype(np.float64, copy=False), cov.astype(np.float64, copy=False), int(total)


def _load_or_compute_embedding_stats(audio_dir: Path, fad_model: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    embedding_files = _expected_embedding_files(audio_dir, fad_model)
    if not embedding_files:
        raise FileNotFoundError(f"no audio files found under {audio_dir} to derive {fad_model} embeddings")
    missing = [str(path) for path in embedding_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing cached embeddings for {fad_model}: {missing[:3]}")

    manifest = _build_file_manifest(embedding_files, base_dir=audio_dir)
    stats_dir = Path(audio_dir) / "stats" / str(fad_model)
    mu_path = stats_dir / "mu.npy"
    cov_path = stats_dir / "cov.npy"
    meta_path = stats_dir / "meta.json"
    if mu_path.is_file() and cov_path.is_file() and meta_path.is_file():
        meta = _read_json(meta_path)
        if str(meta.get("embedding_manifest_hash", "")) == str(manifest["hash"]):
            loaded_meta = dict(meta)
            loaded_meta["stats_cache_hit"] = True
            return np.load(mu_path), np.load(cov_path), loaded_meta

    mu, cov, total_frames = _compute_embedding_stats_online(
        embedding_files,
        desc=f"embedding_stats[{Path(audio_dir).name}]",
    )
    stats_dir.mkdir(parents=True, exist_ok=True)
    np.save(mu_path, mu)
    np.save(cov_path, cov)
    meta = {
        "version": FAD_CACHE_VERSION,
        "fad_model": str(fad_model),
        "embedding_manifest_hash": str(manifest["hash"]),
        "num_embedding_files": int(manifest["num_files"]),
        "num_embedding_frames": int(total_frames),
        "mu_file": _file_metadata(mu_path, base_dir=audio_dir),
        "cov_file": _file_metadata(cov_path, base_dir=audio_dir),
        "stats_cache_hit": False,
    }
    _write_json(meta_path, meta)
    return mu, cov, meta


def ensure_fad_embeddings_cached(
    *,
    fad_python: str,
    fad_model: str,
    audio_dirs: Sequence[Path],
    workers: int,
    device: str = "auto",
) -> None:
    missing_dirs: List[Path] = []
    embedding_totals: Dict[Path, int] = {}
    embedding_done_initial: Dict[Path, int] = {}
    scan_progress = tqdm(audio_dirs, desc="scan_fad_embeddings", unit="dir", leave=False)
    for directory in scan_progress:
        audio_files = _sorted_top_level_files(directory)
        if not audio_files:
            raise FileNotFoundError(f"no audio files found under {directory}")
        expected_embeddings = [_embedding_cache_path(audio_file, fad_model) for audio_file in audio_files]
        existing_count = sum(1 for path in expected_embeddings if path.is_file())
        if existing_count < len(expected_embeddings):
            directory_path = Path(directory)
            missing_dirs.append(directory_path)
            embedding_totals[directory_path] = int(len(expected_embeddings))
            embedding_done_initial[directory_path] = int(existing_count)
    if not missing_dirs:
        return
    worker_schedule: List[int] = [max(1, int(workers))]
    if int(worker_schedule[0]) > 1:
        worker_schedule.append(1)

    last_error: Optional[str] = None
    for worker_count in worker_schedule:
        command = [
            str(fad_python),
            "-m",
            "fadtk.embeds",
            "-m",
            str(fad_model),
            "-d",
            *[str(path) for path in missing_dirs],
            "-w",
            str(int(worker_count)),
        ]
        raw_output = ""
        return_code = 0
        process_env = dict(os.environ)
        process_env.setdefault("TORCH_HOME", str((Path(tempfile.gettempdir()) / "torch-hub-cache").resolve()))
        process_env.setdefault("MPLCONFIGDIR", str((Path(tempfile.gettempdir()) / "matplotlib-drum-rendering").resolve()))
        fad_cuda_visible_devices = _cuda_visible_devices_for_device(str(device))
        if fad_cuda_visible_devices is not None:
            process_env["CUDA_VISIBLE_DEVICES"] = fad_cuda_visible_devices
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stdout_handle, tempfile.TemporaryFile(
            mode="w+",
            encoding="utf-8",
        ) as stderr_handle:
            total_embeddings = int(sum(embedding_totals.values()))
            initial_done = int(sum(embedding_done_initial.values()))
            with tqdm(
                total=max(1, int(total_embeddings)),
                desc=f"fadtk.embeds[{worker_count}w]",
                unit="emb",
                leave=False,
                initial=max(0, int(initial_done)),
            ) as progress:
                progress.set_postfix(model=str(fad_model), dirs=int(len(missing_dirs)))
                process = subprocess.Popen(
                    command,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    cwd=str(Path(tempfile.gettempdir()).resolve()),
                    env=process_env,
                    text=True,
                )
                while process.poll() is None:
                    current_done = 0
                    for directory in missing_dirs:
                        emb_dir = Path(directory) / "embeddings" / str(fad_model)
                        finished_here = len(list(emb_dir.glob("*.npy"))) if emb_dir.exists() else 0
                        current_done += min(int(finished_here), int(embedding_totals[directory]))
                    if int(current_done) > int(progress.n):
                        progress.update(int(current_done) - int(progress.n))
                    progress.refresh()
                    time.sleep(0.5)
                return_code = int(process.wait())
                current_done = 0
                for directory in missing_dirs:
                    emb_dir = Path(directory) / "embeddings" / str(fad_model)
                    finished_here = len(list(emb_dir.glob("*.npy"))) if emb_dir.exists() else 0
                    current_done += min(int(finished_here), int(embedding_totals[directory]))
                if int(current_done) > int(progress.n):
                    progress.update(int(current_done) - int(progress.n))
            stdout_handle.seek(0)
            stderr_handle.seek(0)
            raw_output = "\n".join(
                part for part in [stdout_handle.read().strip(), stderr_handle.read().strip()] if part
            )
        if int(return_code) == 0:
            return
        last_error = f"fadtk.embeds failed ({return_code}): {' '.join(command)}\n{raw_output}"
        error_text = raw_output.lower()
        is_cuda_oom = (
            "out of memory" in error_text
            or "cuda error: out of memory" in error_text
            or "cudaerrormemoryallocation" in error_text
        )
        if worker_count == 1 or not is_cuda_oom:
            break

    raise RuntimeError(str(last_error))


def _calc_frechet_distance(mu1: np.ndarray, cov1: np.ndarray, mu2: np.ndarray, cov2: np.ndarray, eps: float = 1.0e-6) -> float:
    mean_1 = np.atleast_1d(np.asarray(mu1, dtype=np.float64))
    mean_2 = np.atleast_1d(np.asarray(mu2, dtype=np.float64))
    sigma_1 = np.atleast_2d(np.asarray(cov1, dtype=np.float64))
    sigma_2 = np.atleast_2d(np.asarray(cov2, dtype=np.float64))
    if mean_1.shape != mean_2.shape:
        raise ValueError(f"mean shape mismatch: {mean_1.shape} vs {mean_2.shape}")
    if sigma_1.shape != sigma_2.shape:
        raise ValueError(f"covariance shape mismatch: {sigma_1.shape} vs {sigma_2.shape}")

    diff = mean_1 - mean_2
    product = sigma_1.dot(sigma_2)
    eigenvalues, eigenvectors = linalg.eig(product)
    covmean = (eigenvectors * scisqrt(eigenvalues)) @ linalg.inv(eigenvectors)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_1.shape[0], dtype=np.float64) * float(eps)
        eigenvalues, eigenvectors = linalg.eig((sigma_1 + offset).dot(sigma_2 + offset))
        covmean = (eigenvectors * scisqrt(eigenvalues)) @ linalg.inv(eigenvectors)
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0.0, atol=1.0e-3):
            raise ValueError(f"Imaginary component in Fréchet computation: {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real
    trace_covmean = float(np.trace(covmean))
    return float(diff.dot(diff) + np.trace(sigma_1) + np.trace(sigma_2) - 2.0 * trace_covmean)


def _run_deterministic_fad_inf(
    *,
    baseline_mu: np.ndarray,
    baseline_cov: np.ndarray,
    eval_embeddings: np.ndarray,
    seed: int,
    steps: int = FAD_INF_STEPS,
    min_n: int = FAD_INF_MIN_N,
) -> Dict[str, Any]:
    embeds = np.asarray(eval_embeddings, dtype=np.float64)
    if embeds.ndim != 2 or int(embeds.shape[0]) < 2:
        raise ValueError(f"expected eval embeddings with shape (n_frames, n_features), found {embeds.shape}")
    rng = np.random.default_rng(int(seed))
    max_n = int(embeds.shape[0])
    start_n = int(max(2, min(int(min_n), max_n)))
    raw_ns = [int(max(2, int(n))) for n in np.linspace(start_n, max_n, int(max(1, int(steps))))]
    ns = list(dict.fromkeys(raw_ns))
    points: List[List[float]] = []
    if len(ns) < 2:
        indices = rng.choice(max_n, size=int(ns[0]), replace=True)
        mu_eval, cov_eval = _calc_embedding_stats(embeds[indices, :])
        score = _calc_frechet_distance(baseline_mu, baseline_cov, mu_eval, cov_eval)
        return {
            "score": float(score),
            "slope": 0.0,
            "r2": 1.0,
            "points": [[int(ns[0]), float(score)]],
        }
    for n in ns:
        indices = rng.choice(max_n, size=int(n), replace=True)
        mu_eval, cov_eval = _calc_embedding_stats(embeds[indices, :])
        score = _calc_frechet_distance(baseline_mu, baseline_cov, mu_eval, cov_eval)
        points.append([int(n), float(score)])
    ys = np.asarray(points, dtype=np.float64)
    xs = 1.0 / np.asarray(ns, dtype=np.float64)
    slope, intercept = np.polyfit(xs, ys[:, 1], 1)
    denom = float(np.sum((ys[:, 1] - np.mean(ys[:, 1])) ** 2))
    r2 = 1.0 if denom <= EPS else float(1.0 - (np.sum((ys[:, 1] - (slope * xs + intercept)) ** 2) / denom))
    return {
        "score": float(intercept),
        "slope": float(slope),
        "r2": float(r2),
        "points": [[int(pair[0]), float(pair[1])] for pair in points],
    }


def _run_deterministic_fad_inf_repeat_job(
    model_name: str,
    repeat_idx: int,
    seed_value: int,
    baseline_mu: np.ndarray,
    baseline_cov: np.ndarray,
    eval_embeddings: np.ndarray,
    steps: int,
    min_n: int,
) -> Dict[str, Any]:
    result = _run_deterministic_fad_inf(
        baseline_mu=baseline_mu,
        baseline_cov=baseline_cov,
        eval_embeddings=eval_embeddings,
        seed=int(seed_value),
        steps=int(steps),
        min_n=int(min_n),
    )
    return {
        "model": str(model_name),
        "repeat_idx": int(repeat_idx),
        "seed": int(seed_value),
        "fad_inf": float(result["score"]),
        "fad_inf_r2": float(result["r2"]),
        "fad_inf_slope": float(result["slope"]),
        "fad_inf_points": result["points"],
    }


def _aggregate_fad_repeat_rows(
    repeat_rows: pd.DataFrame,
    *,
    fad_model: str,
    fad_python: str,
    alpha: float = DEFAULT_INFERENCE_ALPHA,
) -> pd.DataFrame:
    if repeat_rows.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for model, group in repeat_rows.groupby("model", sort=False):
        fad_values = group["fad_inf"].astype(float).to_numpy()
        r2_values = group["fad_inf_r2"].astype(float).to_numpy()
        fad_ci_low, fad_ci_high = _percentile_interval(fad_values, alpha)
        r2_ci_low, r2_ci_high = _percentile_interval(r2_values, alpha)
        rows.append(
            {
                "model": str(model),
                "fad_inf": float(np.mean(fad_values)),
                "fad_inf_sd": _sample_std(fad_values),
                "fad_inf_ci_low": float(fad_ci_low),
                "fad_inf_ci_high": float(fad_ci_high),
                "fad_inf_r2": float(np.mean(r2_values)),
                "fad_inf_r2_sd": _sample_std(r2_values),
                "fad_inf_r2_ci_low": float(r2_ci_low),
                "fad_inf_r2_ci_high": float(r2_ci_high),
                "fad_inf_repeats": int(len(group)),
                "fad_model": str(fad_model),
                "fad_python": str(fad_python),
            }
        )
    return pd.DataFrame(rows)


def _build_fad_cache_payload(
    *,
    model_name: str,
    fad_model: str,
    baseline_stats_meta: Mapping[str, Any],
    eval_manifest: Mapping[str, Any],
    seeds: Sequence[int],
    steps: int,
    min_n: int,
) -> Dict[str, Any]:
    stable_baseline_stats_meta = {str(key): value for key, value in dict(baseline_stats_meta).items() if str(key) != "stats_cache_hit"}
    return {
        "cache_version": FAD_CACHE_VERSION,
        "model": str(model_name),
        "fad_model": str(fad_model),
        "baseline_stats_meta": stable_baseline_stats_meta,
        "eval_embedding_manifest": dict(eval_manifest),
        "seeds": [int(seed) for seed in seeds],
        "steps": int(steps),
        "min_n": int(min_n),
    }


def compute_fad_rows(
    *,
    ablations_root: Path,
    out_root: Path,
    runs: Sequence[RunPredictions],
    dataset_indices: Sequence[int],
    target_cache: TargetAudioCache,
    fad_python: str,
    fad_model: str,
    fad_workers: int,
    fad_inf_workers: int,
    fad_seed: int,
    fad_repeats: int,
    alpha: float,
    device: str = "auto",
) -> FADOutputs:
    fad_root = out_root / "fad_assets"
    reference_dir = fad_root / "reference_audio"
    materialize_reference_wavs(reference_dir, target_cache=target_cache, dataset_indices=dataset_indices)

    eval_dirs: Dict[str, Path] = {}
    for run in runs:
        eval_dir = fad_root / "eval_audio" / str(run.name)
        materialize_eval_links(eval_dir, run=run, dataset_indices=dataset_indices)
        eval_dirs[str(run.name)] = eval_dir

    ensure_fad_embeddings_cached(
        fad_python=str(fad_python),
        fad_model=str(fad_model),
        audio_dirs=[reference_dir, *eval_dirs.values()],
        workers=int(fad_workers),
        device=str(device),
    )

    baseline_mu, baseline_cov, baseline_stats_meta = _load_or_compute_embedding_stats(reference_dir, fad_model)
    cache_root = Path(ablations_root).resolve() / ".cache" / "acoustic_eval_fad_inf" / str(fad_model)
    cache_root.mkdir(parents=True, exist_ok=True)
    seeds = _seed_schedule(int(fad_seed), int(fad_repeats))
    repeats_per_model = max(1, int(len(seeds)))
    total_repeat_jobs = max(1, int(len(runs)) * int(repeats_per_model))

    repeat_records: List[Dict[str, Any]] = []
    cache_hits = 0
    cache_misses = 0
    progress = tqdm(total=total_repeat_jobs, desc="fad_inf", unit="repeat", leave=False)
    for run in runs:
        model_name = str(run.name)
        progress.set_postfix(model=model_name, phase="manifest")
        eval_dir = eval_dirs[model_name]
        eval_embedding_files = _expected_embedding_files(eval_dir, fad_model)
        if not eval_embedding_files:
            raise FileNotFoundError(f"no eval embedding files found under {eval_dir}")
        if any(not path.is_file() for path in eval_embedding_files):
            missing = [str(path) for path in eval_embedding_files if not path.is_file()]
            raise FileNotFoundError(f"missing eval embedding files for {model_name}: {missing[:3]}")
        eval_manifest = _build_file_manifest(eval_embedding_files, base_dir=eval_dir)
        cache_payload = _build_fad_cache_payload(
            model_name=model_name,
            fad_model=fad_model,
            baseline_stats_meta=baseline_stats_meta,
            eval_manifest=eval_manifest,
            seeds=seeds,
            steps=FAD_INF_STEPS,
            min_n=FAD_INF_MIN_N,
        )
        cache_key = _sha256_text(_json_dumps(cache_payload))
        cache_path = cache_root / f"{_sanitize_name(model_name)}__{cache_key}.json"

        cached_payload: Optional[Dict[str, Any]] = None
        cache_hit = False
        if cache_path.is_file():
            try:
                cached_payload = _read_json(cache_path)
                cache_hit = True
            except Exception:
                cached_payload = None
                cache_hit = False

        if cached_payload is None:
            progress.set_postfix(model=model_name, phase="load_embeddings")
            eval_embeddings = _load_embedding_matrix(eval_embedding_files)
            num_frames = int(eval_embeddings.shape[0])
            repeats: List[Dict[str, Any]] = []
            repeat_jobs = [(int(repeat_idx), int(seed_value)) for repeat_idx, seed_value in enumerate(seeds)]
            max_repeat_workers = max(1, min(int(fad_inf_workers), len(repeat_jobs)))
            if max_repeat_workers <= 1:
                for repeat_idx, seed_value in repeat_jobs:
                    progress.set_postfix(model=model_name, phase=f"repeat {repeat_idx + 1}/{len(seeds)}")
                    repeats.append(
                        _run_deterministic_fad_inf_repeat_job(
                            model_name,
                            int(repeat_idx),
                            int(seed_value),
                            baseline_mu,
                            baseline_cov,
                            eval_embeddings,
                            int(FAD_INF_STEPS),
                            int(FAD_INF_MIN_N),
                        )
                    )
                    progress.update(1)
            else:
                progress.set_postfix(model=model_name, phase=f"parallel x{max_repeat_workers}")
                mp_context = multiprocessing.get_context("spawn")
                with concurrent.futures.ProcessPoolExecutor(max_workers=max_repeat_workers, mp_context=mp_context) as executor:
                    future_to_repeat = {
                        executor.submit(
                            _run_deterministic_fad_inf_repeat_job,
                            model_name,
                            int(repeat_idx),
                            int(seed_value),
                            baseline_mu,
                            baseline_cov,
                            eval_embeddings,
                            int(FAD_INF_STEPS),
                            int(FAD_INF_MIN_N),
                        ): (int(repeat_idx), int(seed_value))
                        for repeat_idx, seed_value in repeat_jobs
                    }
                    completed = 0
                    for future in concurrent.futures.as_completed(future_to_repeat):
                        repeats.append(dict(future.result()))
                        completed += 1
                        progress.set_postfix(model=model_name, phase=f"repeat {completed}/{len(seeds)}")
                        progress.update(1)
                repeats.sort(key=lambda row: int(row["repeat_idx"]))
            cached_payload = {
                "cache_version": FAD_CACHE_VERSION,
                "cache_key": cache_key,
                "model": model_name,
                "fad_model": str(fad_model),
                "baseline_stats_meta": dict(baseline_stats_meta),
                "eval_embedding_manifest": dict(eval_manifest),
                "num_eval_embedding_files": int(len(eval_embedding_files)),
                "num_eval_embedding_frames": int(num_frames),
                "seeds": [int(seed_value) for seed_value in seeds],
                "steps": int(FAD_INF_STEPS),
                "min_n": int(FAD_INF_MIN_N),
                "repeat_rows": repeats,
            }
            _write_json(cache_path, cached_payload)
            cache_misses += 1
        else:
            cached_repeat_rows = list(cached_payload.get("repeat_rows", []))
            cached_repeat_count = max(1, len(cached_repeat_rows) or len(seeds))
            progress.set_postfix(model=model_name, phase=f"cache_hit x{cached_repeat_count}")
            progress.update(min(cached_repeat_count, max(0, total_repeat_jobs - int(progress.n))))
            cache_hits += 1

        assert cached_payload is not None
        for row in list(cached_payload.get("repeat_rows", [])):
            repeat_records.append(
                {
                    "model": model_name,
                    "repeat_idx": _safe_int(row.get("repeat_idx"), 0),
                    "seed": _safe_int(row.get("seed"), 0),
                    "fad_inf": float(row.get("fad_inf")),
                    "fad_inf_r2": float(row.get("fad_inf_r2")),
                    "fad_inf_slope": float(row.get("fad_inf_slope", math.nan)),
                    "cache_key": str(cached_payload.get("cache_key", cache_key)),
                    "cache_path": str(cache_path),
                    "cache_hit": bool(cache_hit),
                    "cache_version": str(cached_payload.get("cache_version", FAD_CACHE_VERSION)),
                    "fad_model": str(fad_model),
                    "num_eval_embedding_files": _safe_int(cached_payload.get("num_eval_embedding_files"), len(eval_embedding_files)),
                    "num_eval_embedding_frames": _safe_int(cached_payload.get("num_eval_embedding_frames"), -1),
                    "eval_embedding_manifest_hash": str(eval_manifest["hash"]),
                    "baseline_embedding_manifest_hash": str(baseline_stats_meta.get("embedding_manifest_hash", "")),
                }
            )
    progress.close()

    repeat_df = pd.DataFrame(repeat_records)
    overall_df = _aggregate_fad_repeat_rows(
        repeat_df,
        fad_model=str(fad_model),
        fad_python=str(fad_python),
        alpha=float(alpha),
    )
    return FADOutputs(
        overall_df=overall_df,
        repeat_df=repeat_df,
        cache_stats={
            "cache_hits": int(cache_hits),
            "cache_misses": int(cache_misses),
            "num_models": int(len(runs)),
            "cache_root": str(cache_root),
        },
    )


def _paired_metric_matrix(per_clip_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    pivot = (
        per_clip_df.pivot(index="dataset_index", columns="model", values=metric)
        .sort_index(axis=0, kind="stable")
        .sort_index(axis=1, kind="stable")
    )
    if bool(pivot.isna().any().any()):
        raise ValueError(f"paired metric matrix for {metric} contains missing values")
    return pivot


def _holm_adjust(p_values: Sequence[float]) -> List[float]:
    values = np.asarray(list(p_values), dtype=np.float64).reshape(-1)
    if values.size == 0:
        return []
    order = np.argsort(values)
    sorted_values = values[order]
    adjusted_sorted = np.maximum.accumulate((values.size - np.arange(values.size)) * sorted_values)
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return [float(value) for value in adjusted]


def _paired_sign_flip_pvalue(deltas: np.ndarray, sign_matrix: np.ndarray) -> float:
    diff = np.asarray(deltas, dtype=np.float64).reshape(-1)
    if diff.size == 0 or sign_matrix.size == 0:
        return 1.0
    observed = float(abs(np.mean(diff)))
    if observed <= EPS:
        return 1.0
    permuted = np.mean(sign_matrix.astype(np.float64, copy=False) * diff.reshape(1, -1), axis=1)
    exceed = int(np.count_nonzero(np.abs(permuted) >= observed))
    return float((1 + exceed) / float(1 + permuted.shape[0]))


def compute_paired_inference(
    per_clip_df: pd.DataFrame,
    *,
    seed: int,
    bootstrap_samples: int,
    permutation_samples: int,
    alpha: float,
) -> InferenceOutputs:
    metrics = [metric for metric in PAIRED_METRIC_COLUMNS if metric in per_clip_df.columns]
    if per_clip_df.empty or not metrics:
        return InferenceOutputs(intervals_df=pd.DataFrame(), significance_df=pd.DataFrame())

    dataset_count = int(per_clip_df["dataset_index"].nunique())
    rng_boot = np.random.default_rng(int(seed) + 101)
    rng_perm = np.random.default_rng(int(seed) + 202)
    bootstrap_index = (
        rng_boot.integers(0, dataset_count, size=(int(max(1, bootstrap_samples)), dataset_count), dtype=np.int32)
        if int(bootstrap_samples) > 0
        else np.empty((0, dataset_count), dtype=np.int32)
    )
    sign_matrix = (
        (rng_perm.integers(0, 2, size=(int(max(1, permutation_samples)), dataset_count), dtype=np.int8) * 2) - 1
        if int(permutation_samples) > 0
        else np.empty((0, dataset_count), dtype=np.int8)
    )

    interval_rows: List[Dict[str, Any]] = []
    significance_rows: List[Dict[str, Any]] = []
    for metric in metrics:
        pivot = _paired_metric_matrix(per_clip_df, metric)
        model_names = list(pivot.columns)
        values = pivot.to_numpy(dtype=np.float32, copy=False)
        points = np.mean(values.astype(np.float64, copy=False), axis=0)
        better = str(METRIC_SPECS.get(metric, {}).get("better", "lower"))
        orientation = 1.0 if better == "lower" else -1.0
        best_idx = int(np.argmin(points) if better == "lower" else np.argmax(points))
        best_model = str(model_names[best_idx])

        for model_idx, model_name in enumerate(model_names):
            point = float(points[model_idx])
            if bootstrap_index.size > 0:
                boot = np.mean(values[:, model_idx][bootstrap_index], axis=1, dtype=np.float64)
                ci_low, ci_high = _percentile_interval(boot, alpha)
            else:
                ci_low = point
                ci_high = point
            interval_rows.append(
                {
                    "metric": str(metric),
                    "model": str(model_name),
                    "point_estimate": float(point),
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "better": better,
                    "alpha": float(alpha),
                    "num_examples": int(values.shape[0]),
                }
            )

        raw_p_values: List[float] = []
        pending_rows: List[Dict[str, Any]] = []
        for challenger_idx, challenger_name in enumerate(model_names):
            if int(challenger_idx) == int(best_idx):
                continue
            raw_diff = values[:, challenger_idx].astype(np.float64, copy=False) - values[:, best_idx].astype(np.float64, copy=False)
            oriented_diff = orientation * raw_diff
            if bootstrap_index.size > 0:
                raw_boot = np.mean(raw_diff[bootstrap_index], axis=1, dtype=np.float64)
                oriented_boot = orientation * raw_boot
                raw_ci_low, raw_ci_high = _percentile_interval(raw_boot, alpha)
                oriented_ci_low, oriented_ci_high = _percentile_interval(oriented_boot, alpha)
            else:
                raw_ci_low = float(np.mean(raw_diff))
                raw_ci_high = float(np.mean(raw_diff))
                oriented_ci_low = float(np.mean(oriented_diff))
                oriented_ci_high = float(np.mean(oriented_diff))
            p_value = _paired_sign_flip_pvalue(raw_diff, sign_matrix)
            raw_p_values.append(float(p_value))
            pending_rows.append(
                {
                    "metric": str(metric),
                    "best_model": best_model,
                    "challenger_model": str(challenger_name),
                    "best_point_estimate": float(points[best_idx]),
                    "challenger_point_estimate": float(points[challenger_idx]),
                    "raw_delta": float(np.mean(raw_diff)),
                    "raw_ci_low": float(raw_ci_low),
                    "raw_ci_high": float(raw_ci_high),
                    "oriented_delta": float(np.mean(oriented_diff)),
                    "oriented_ci_low": float(oriented_ci_low),
                    "oriented_ci_high": float(oriented_ci_high),
                    "p_value": float(p_value),
                    "better": better,
                    "alpha": float(alpha),
                    "num_examples": int(values.shape[0]),
                }
            )
        adjusted = _holm_adjust(raw_p_values)
        for row, adjusted_value in zip(pending_rows, adjusted):
            row["p_adj"] = float(adjusted_value)
            row["significant"] = bool(float(adjusted_value) < float(alpha))
            significance_rows.append(row)

    return InferenceOutputs(
        intervals_df=pd.DataFrame(interval_rows),
        significance_df=pd.DataFrame(significance_rows),
    )


def write_inference_summary(path: Path, significance_df: pd.DataFrame) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if significance_df.empty:
        target.write_text("# Inference Summary\n\nNo paired significance comparisons were computed.\n", encoding="utf-8")
        return
    display = significance_df.loc[
        :,
        [
            "metric",
            "best_model",
            "challenger_model",
            "oriented_delta",
            "oriented_ci_low",
            "oriented_ci_high",
            "p_adj",
            "significant",
        ],
    ].copy()
    display["metric"] = display["metric"].map(lambda value: str(METRIC_SPECS.get(str(value), {}).get("label", value)))
    for column in ["oriented_delta", "oriented_ci_low", "oriented_ci_high", "p_adj"]:
        display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    display["significant"] = display["significant"].map(lambda value: "yes" if bool(value) else "no")
    try:
        body = display.to_markdown(index=False)
    except Exception:
        body = "```\n" + display.to_string(index=False) + "\n```"
    target.write_text("# Inference Summary\n\n" + body + "\n", encoding="utf-8")


def write_metric_notes(path: Path) -> None:
    lines = [
        "# Metric Rationale",
        "",
        "This evaluation compares decoded EnCodec 32 kHz predictions against cached 32 kHz target clips.",
        "",
        "Design choices:",
        "",
        "- `fadtk` FAD-inf is the distribution-level metric. This follows recent FAD work that recommends extrapolation toward infinite sample size to reduce bias.",
        "- Recent music-generation evaluation work suggests no single automatic metric is sufficient, so the script combines distributional, spectral, temporal, and dynamic views.",
        "- The paired metrics are an inference from the literature: transient timing is represented with broadband and band-limited spectral flux, timbral balance with low/mid/high spectral ratios plus centroid, and punch/level with RMS and crest-factor deltas.",
        f"- Raw RMS MAE is the absolute difference between clip-level RMS values in dBFS after applying a {RMS_DB_FLOOR:.0f} dB floor over the masked four-beat target region. Peak-Norm RMS MAE is computed after per-clip peak normalization and is the safer reconstruction-shape diagnostic when exported audio has a constant gain offset.",
        "- Clip-level inference uses paired bootstrap confidence intervals and paired sign-flip permutation tests over shared dataset indices.",
        "- FAD∞ uncertainty here reflects deterministic multi-seed Monte Carlo variability of the extrapolation, not a dataset-level significance test.",
        "",
        "References:",
        "",
    ]
    for citation, url in LITERATURE_REFERENCES:
        lines.append(f"- {citation} {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    ablations_root = Path(args.ablations_root).resolve()
    cache_root = Path(args.cache_root).resolve()
    out_root = (Path(args.out_root).resolve() if args.out_root is not None else (ablations_root / "acoustic_eval").resolve())

    if out_root.exists():
        if not bool(args.overwrite):
            raise FileExistsError(f"output root already exists: {out_root}; pass --overwrite to replace it")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(ablations_root, args.models)
    dataset_indices = shared_dataset_indices(runs, int(args.max_items))
    target_cache = TargetAudioCache(cache_root=cache_root)

    per_run_frames: List[pd.DataFrame] = []
    for run in runs:
        frame = evaluate_run(run, dataset_indices=dataset_indices, target_cache=target_cache)
        per_run_frames.append(frame)
    per_clip_df = pd.concat(per_run_frames, axis=0, ignore_index=True)

    fad_df = pd.DataFrame()
    fad_repeat_df = pd.DataFrame()
    fad_cache_stats: Dict[str, Any] = {}
    fad_error: Optional[str] = None
    if not bool(args.skip_fad):
        try:
            fad_outputs = compute_fad_rows(
                ablations_root=ablations_root,
                out_root=out_root,
                runs=runs,
                dataset_indices=dataset_indices,
                target_cache=target_cache,
                fad_python=str(args.fad_python),
                fad_model=str(args.fad_model),
                fad_workers=int(args.fad_workers),
                fad_inf_workers=int(args.fad_inf_workers),
                fad_seed=int(args.fad_seed),
                fad_repeats=int(args.fad_repeats),
                alpha=float(args.inference_alpha),
                device=str(args.device),
            )
            fad_df = fad_outputs.overall_df
            fad_repeat_df = fad_outputs.repeat_df
            fad_cache_stats = dict(fad_outputs.cache_stats)
        except Exception as exc:
            fad_error = str(exc)

    inference_outputs = InferenceOutputs(intervals_df=pd.DataFrame(), significance_df=pd.DataFrame())
    inference_error: Optional[str] = None
    if not bool(args.skip_inference):
        try:
            inference_outputs = compute_paired_inference(
                per_clip_df,
                seed=int(args.fad_seed),
                bootstrap_samples=int(args.bootstrap_samples),
                permutation_samples=int(args.permutation_samples),
                alpha=float(args.inference_alpha),
            )
        except Exception as exc:
            inference_error = str(exc)

    efficiency_df, efficiency_availability = aggregate_efficiency(runs)
    overall_df = aggregate_overall(
        per_clip_df,
        fad_rows=(fad_df if not fad_df.empty else None),
        efficiency_rows=(efficiency_df if not efficiency_df.empty else None),
    )
    band_df = aggregate_band_summary(per_clip_df)
    band_profile_df, target_band_profile = aggregate_band_profile(per_clip_df)
    style_df = aggregate_style_summary(per_clip_df)

    per_clip_path = out_root / "per_clip_metrics.csv"
    overall_path = out_root / "overall_summary.csv"
    band_path = out_root / "band_summary.csv"
    band_profile_path = out_root / "band_profile_summary.csv"
    style_path = out_root / "style_summary.csv"
    efficiency_path = out_root / "efficiency_summary.csv"
    fad_repeat_path = out_root / "fad_repeat_summary.csv"
    inference_intervals_path = out_root / "paired_metric_intervals.csv"
    inference_significance_path = out_root / "paired_metric_significance.csv"
    inference_summary_path = out_root / "inference_summary.md"

    per_clip_df.to_csv(per_clip_path, index=False)
    overall_df.to_csv(overall_path, index=False)
    band_df.to_csv(band_path, index=False)
    band_profile_df.to_csv(band_profile_path, index=False)
    style_df.to_csv(style_path, index=False)
    efficiency_df.to_csv(efficiency_path, index=False)
    if not fad_repeat_df.empty:
        fad_repeat_df.to_csv(fad_repeat_path, index=False)
    if not bool(args.skip_inference):
        inference_outputs.intervals_df.to_csv(inference_intervals_path, index=False)
        inference_outputs.significance_df.to_csv(inference_significance_path, index=False)

    overall_display = build_display_table(overall_df, ["model", "num_examples", *OVERALL_DISPLAY_METRICS])
    band_display = build_display_table(band_df, ["model", *BAND_DISPLAY_METRICS])
    efficiency_display = build_display_table(efficiency_df, EFFICIENCY_DISPLAY_COLUMNS)
    _write_markdown(out_root / "overall_summary.md", "Overall Summary", overall_display)
    _write_markdown(out_root / "band_summary.md", "Band Summary", band_display)
    _write_markdown(out_root / "efficiency_summary.md", "Efficiency Summary", efficiency_display)
    if not bool(args.skip_inference):
        write_inference_summary(inference_summary_path, inference_outputs.significance_df)
    write_metric_notes(out_root / "metric_rationale.md")

    if not bool(args.no_plots):
        figures_root = out_root / "figures"
        plot_jobs: List[Tuple[str, Any]] = [
            (
                "overall_summary_table",
                lambda: save_table_png(
                    overall_display,
                    figures_root / "overall_summary_table.png",
                    title="Overall Acoustic Summary",
                ),
            ),
            (
                "band_summary_table",
                lambda: save_table_png(
                    band_display,
                    figures_root / "band_summary_table.png",
                    title="Band Flux Summary",
                ),
            ),
            (
                "overall_summary_heatmap",
                lambda: save_metric_heatmap(
                    overall_df,
                    figures_root / "overall_summary_heatmap.png",
                    metrics=OVERALL_DISPLAY_METRICS,
                    title="Overall Metrics Heatmap",
                ),
            ),
            (
                "band_summary_heatmap",
                lambda: save_metric_heatmap(
                    band_df,
                    figures_root / "band_summary_heatmap.png",
                    metrics=BAND_DISPLAY_METRICS,
                    title="Band Metrics Heatmap",
                ),
            ),
            (
                "band_balance_profiles",
                lambda: save_band_profile_plot(
                    band_profile_df,
                    target_band_profile,
                    figures_root / "band_balance_profiles.png",
                ),
            ),
            (
                "style_onset_flux_heatmap",
                lambda: save_group_metric_heatmap(
                    per_clip_df,
                    group_column="style",
                    value_column="onset_flux_cosine",
                    path=figures_root / "style_onset_flux_heatmap.png",
                    title="Style Breakdown: Broad Flux Cosine",
                    top_k=int(args.style_plot_top_k),
                ),
            ),
            (
                "style_mel_mae_heatmap",
                lambda: save_group_metric_heatmap(
                    per_clip_df,
                    group_column="style",
                    value_column="mel_mae_db",
                    path=figures_root / "style_mel_mae_heatmap.png",
                    title="Style Breakdown: Mel MAE (dB)",
                    top_k=int(args.style_plot_top_k),
                ),
            ),
        ]
        plot_progress = tqdm(plot_jobs, desc="plots", unit="plot", leave=False)
        for _, plot_job in plot_progress:
            plot_job()

    sample_rates = (
        sorted(int(value) for value in pd.to_numeric(per_clip_df.get("sample_rate"), errors="coerce").dropna().unique())
        if "sample_rate" in per_clip_df
        else [int(SAMPLE_RATE)]
    )
    summary_payload: Dict[str, Any] = {
        "ablations_root": str(ablations_root),
        "cache_root": str(cache_root),
        "out_root": str(out_root),
        "num_runs": int(len(runs)),
        "runs": [str(run.name) for run in runs],
        "num_shared_examples": int(len(dataset_indices)),
        "sample_rate": int(sample_rates[0]) if len(sample_rates) == 1 else None,
        "sample_rates": sample_rates,
        "evaluates_decoded_audio": "cache_native_sample_rate",
        "uses_44k_targets": bool(any(int(value) == 44100 for value in sample_rates)),
        "fad_requested": bool(not args.skip_fad),
        "fad_completed": bool(not fad_df.empty),
        "fad_model": (str(args.fad_model) if not args.skip_fad else None),
        "fad_python": (str(args.fad_python) if not args.skip_fad else None),
        "fad_device": (str(args.device) if not args.skip_fad else None),
        "fad_cuda_visible_devices": (
            _cuda_visible_devices_for_device(str(args.device)) if not args.skip_fad else None
        ),
        "fad_workers": (int(args.fad_workers) if not args.skip_fad else None),
        "fad_inf_workers": (int(args.fad_inf_workers) if not args.skip_fad else None),
        "fad_seed": (int(args.fad_seed) if not args.skip_fad else None),
        "fad_repeats": (int(args.fad_repeats) if not args.skip_fad else None),
        "fad_inf_steps": (int(FAD_INF_STEPS) if not args.skip_fad else None),
        "fad_inf_min_n": (int(FAD_INF_MIN_N) if not args.skip_fad else None),
        "fad_error": fad_error,
        "fad_repeat_summary_csv": (str(fad_repeat_path.relative_to(out_root)) if not fad_repeat_df.empty else None),
        "fad_cache": (dict(fad_cache_stats) if fad_cache_stats else None),
        "skip_inference": bool(args.skip_inference),
        "inference_completed": bool((not args.skip_inference) and (inference_error is None)),
        "inference_error": inference_error,
        "bootstrap_samples": (int(args.bootstrap_samples) if not args.skip_inference else None),
        "permutation_samples": (int(args.permutation_samples) if not args.skip_inference else None),
        "inference_alpha": (float(args.inference_alpha) if not args.skip_inference else None),
        "overall_summary_csv": str(overall_path.relative_to(out_root)),
        "band_summary_csv": str(band_path.relative_to(out_root)),
        "band_profile_summary_csv": str(band_profile_path.relative_to(out_root)),
        "style_summary_csv": str(style_path.relative_to(out_root)),
        "efficiency_summary_csv": str(efficiency_path.relative_to(out_root)),
        "per_clip_metrics_csv": str(per_clip_path.relative_to(out_root)),
        "paired_metric_intervals_csv": (
            str(inference_intervals_path.relative_to(out_root)) if not bool(args.skip_inference) else None
        ),
        "paired_metric_significance_csv": (
            str(inference_significance_path.relative_to(out_root)) if not bool(args.skip_inference) else None
        ),
        "inference_summary_md": (
            str(inference_summary_path.relative_to(out_root)) if not bool(args.skip_inference) else None
        ),
        "efficiency_available": bool(
            any(bool(item.get("efficiency_fields_found")) for item in list(efficiency_availability))
        ),
        "efficiency_runs": list(efficiency_availability),
        "references": [{"citation": citation, "url": url} for citation, url in LITERATURE_REFERENCES],
    }
    _write_json(out_root / "summary.json", summary_payload)

    print(
        json.dumps(
            {
                "event": "acoustic_eval_complete",
                "out_root": str(out_root),
                "num_runs": int(len(runs)),
                "num_shared_examples": int(len(dataset_indices)),
                "fad_completed": bool(not fad_df.empty),
                "fad_error": fad_error,
                "inference_completed": bool((not args.skip_inference) and (inference_error is None)),
                "inference_error": inference_error,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
