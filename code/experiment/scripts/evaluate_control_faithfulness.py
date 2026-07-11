#!/usr/bin/env python3
"""Evaluate generated-onset alignment against the input drum control grid."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT.parent.parent
RUNS_ROOT = PACKAGE_ROOT / "runs"
RESULTS_ROOT = PACKAGE_ROOT / "results"


def _preload_stdlib_inspect() -> None:
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

import torch
import torchaudio

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

from io_utils import write_json
from scripts.baseline_export_common import read_jsonl
from scripts.grid_condition_utils import payload_family_names


GROUPS = ("kick", "snare", "tom", "hihat", "cymbal")
BANDS_HZ = {
    "kick": (30.0, 180.0),
    "snare": (150.0, 2500.0),
    "tom": (70.0, 700.0),
    "hihat": (3000.0, 12000.0),
    "cymbal": (2000.0, 14000.0),
}


def _progress(iterable: Any, *, desc: str) -> Any:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, leave=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=str, default=str(RUNS_ROOT / "mini_cache"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--predictions-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--tolerance-ms", type=float, default=50.0)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--threshold-quantile", type=float, default=0.86)
    parser.add_argument("--threshold-std", type=float, default=0.45)
    parser.add_argument("--refractory-ms", type=float, default=45.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve_out_dir(predictions_dir: Path, explicit: str) -> Path:
    if str(explicit).strip():
        return Path(explicit).expanduser().resolve()
    return (predictions_dir / "control_faithfulness_eval").resolve()


def _prepare_out_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if bool(overwrite):
            shutil.rmtree(path)
        elif any(path.iterdir()):
            raise FileExistsError(f"output directory already exists and is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _family_group(family: str) -> str | None:
    name = str(family).lower()
    if name == "kick":
        return "kick"
    if name == "snare":
        return "snare"
    if name.startswith("tom"):
        return "tom"
    if name == "hihat":
        return "hihat"
    if name in {"crash", "ride", "cymbal"}:
        return "cymbal"
    return None


def _target_events_by_group(payload: Mapping[str, Any]) -> dict[str, list[float]]:
    events = {group: [] for group in GROUPS}
    onsets = torch.as_tensor(payload.get("family_onsets_ft"), dtype=torch.bool)
    grid_times = torch.as_tensor(payload.get("grid_times_sec_t"), dtype=torch.float32)
    if int(onsets.dim()) != 2 or int(grid_times.numel()) <= 0:
        return events
    for family_idx, family in enumerate(payload_family_names(payload)[: int(onsets.shape[0])]):
        group = _family_group(str(family))
        if group is None:
            continue
        frames = torch.nonzero(onsets[int(family_idx)], as_tuple=False).flatten().tolist()
        for frame_idx in frames:
            if 0 <= int(frame_idx) < int(grid_times.numel()):
                events[group].append(float(grid_times[int(frame_idx)].item()))
    for group in events:
        events[group] = sorted(events[group])
    return events


def _load_prediction_audio(path: Path) -> tuple[torch.Tensor, int]:
    audio_ct, sample_rate = torchaudio.load(str(path))
    if int(audio_ct.shape[0]) > 1:
        audio_ct = audio_ct.mean(dim=0, keepdim=True)
    return audio_ct.to(dtype=torch.float32).contiguous(), int(sample_rate)


def _detect_band_onsets(
    audio_ct: torch.Tensor,
    *,
    sample_rate: int,
    group: str,
    n_fft: int,
    hop_length: int,
    threshold_quantile: float,
    threshold_std: float,
    refractory_ms: float,
) -> list[float]:
    audio = torch.as_tensor(audio_ct, dtype=torch.float32)
    if int(audio.dim()) == 2:
        audio = audio.mean(dim=0)
    if int(audio.numel()) < int(n_fft):
        audio = torch.nn.functional.pad(audio, (0, int(n_fft) - int(audio.numel())))
    window = torch.hann_window(int(n_fft), dtype=torch.float32)
    spec = torch.stft(
        audio,
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        win_length=int(n_fft),
        window=window,
        center=True,
        return_complex=True,
    ).abs().pow(2.0)
    freqs = torch.linspace(0.0, float(sample_rate) / 2.0, steps=int(spec.shape[0]))
    lo, hi = BANDS_HZ[str(group)]
    band_mask = freqs.ge(float(lo)) & freqs.le(float(min(hi, float(sample_rate) / 2.0)))
    if not bool(band_mask.any()):
        band_mask = torch.ones_like(freqs, dtype=torch.bool)
    energy = torch.log1p(spec[band_mask].sum(dim=0))
    if int(energy.numel()) <= 2:
        return []
    flux = torch.nn.functional.pad((energy[1:] - energy[:-1]).clamp_min(0.0), (1, 0))
    quantile = float(torch.quantile(flux, float(threshold_quantile)).item())
    threshold = max(quantile, float(flux.mean().item()) + float(threshold_std) * float(flux.std(unbiased=False).item()))
    peaks: list[int] = []
    refractory_frames = max(1, int(round(float(refractory_ms) / 1000.0 * float(sample_rate) / float(hop_length))))
    last = -10_000
    for idx in range(1, int(flux.numel()) - 1):
        value = float(flux[idx].item())
        if value <= threshold:
            continue
        if value < float(flux[idx - 1].item()) or value < float(flux[idx + 1].item()):
            continue
        if int(idx) - int(last) < int(refractory_frames):
            if peaks and value > float(flux[peaks[-1]].item()):
                peaks[-1] = int(idx)
                last = int(idx)
            continue
        peaks.append(int(idx))
        last = int(idx)
    return [float(idx) * float(hop_length) / float(sample_rate) for idx in peaks]


def _match_events(targets: list[float], predictions: list[float], *, tolerance_sec: float) -> dict[str, float | int]:
    used: set[int] = set()
    tp = 0
    total_error = 0.0
    for target in sorted(targets):
        best_idx = -1
        best_dist = float("inf")
        for pred_idx, prediction in enumerate(predictions):
            if pred_idx in used:
                continue
            dist = abs(float(prediction) - float(target))
            if dist <= float(tolerance_sec) and dist < best_dist:
                best_idx = int(pred_idx)
                best_dist = float(dist)
        if best_idx >= 0:
            used.add(best_idx)
            tp += 1
            total_error += float(best_dist)
    fp = max(0, int(len(predictions)) - int(tp))
    fn = max(0, int(len(targets)) - int(tp))
    precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else (1.0 if len(targets) == 0 else 0.0)
    recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-8)
    return {
        "target_count": int(len(targets)),
        "pred_count": int(len(predictions)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_abs_error_sec": float(total_error) / float(tp) if tp > 0 else None,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summarize(per_clip_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, dict[str, Any]] = {}
    for group in GROUPS:
        rows = [row for row in per_clip_rows if row["group"] == group]
        tp = sum(int(row["tp"]) for row in rows)
        fp = sum(int(row["fp"]) for row in rows)
        fn = sum(int(row["fn"]) for row in rows)
        target_count = int(sum(int(row["target_count"]) for row in rows))
        pred_count = int(sum(int(row["pred_count"]) for row in rows))
        precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else (1.0 if target_count == 0 else 0.0)
        recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-8)
        by_group[group] = {
            "target_count": int(target_count),
            "pred_count": int(pred_count),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    supported_groups = [group for group in GROUPS if int(by_group[group]["target_count"]) > 0]
    macro_groups = supported_groups or list(GROUPS)
    macro_f1 = float(sum(float(by_group[group]["f1"]) for group in macro_groups) / float(len(macro_groups)))
    macro_recall = float(sum(float(by_group[group]["recall"]) for group in macro_groups) / float(len(macro_groups)))
    macro_precision = float(sum(float(by_group[group]["precision"]) for group in macro_groups) / float(len(macro_groups)))
    return {
        "metric_name": "proxy_control_faithfulness_onset_alignment",
        "groups": by_group,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "macro_groups": list(macro_groups),
        "note": "Heuristic band-flux onset proxy; use the same settings for all systems.",
    }


def main() -> None:
    args = _parse_args()
    cache_root = Path(args.cache_root).expanduser().resolve()
    split = str(args.split).strip().lower()
    predictions_dir = Path(args.predictions_dir).expanduser().resolve()
    out_dir = _resolve_out_dir(predictions_dir, str(args.out_dir))
    _prepare_out_dir(out_dir, overwrite=bool(args.overwrite))

    prediction_rows = read_jsonl(predictions_dir / "manifest.jsonl")
    split_rows = read_jsonl(cache_root / "manifests" / f"{split}.jsonl")
    if int(args.max_items) > 0:
        prediction_rows = prediction_rows[: int(args.max_items)]

    per_clip_rows: list[dict[str, Any]] = []
    tolerance_sec = float(args.tolerance_ms) / 1000.0
    for row in _progress(prediction_rows, desc=f"control-faithfulness[{split}]"):
        dataset_index = int(row.get("dataset_index", -1))
        if dataset_index < 0 or dataset_index >= len(split_rows):
            raise IndexError(f"dataset_index={dataset_index} is outside split manifest length {len(split_rows)}")
        split_row = split_rows[int(dataset_index)]
        payload = dict(torch.load(cache_root / str(split_row["out_pt"]), map_location="cpu", weights_only=False))
        wav_path = predictions_dir / str(row.get("wav") or "")
        if not wav_path.is_file():
            raise FileNotFoundError(f"prediction wav missing: {wav_path}")
        audio_ct, sample_rate = _load_prediction_audio(wav_path)
        target_events = _target_events_by_group(payload)
        for group in GROUPS:
            predicted_events = _detect_band_onsets(
                audio_ct,
                sample_rate=int(sample_rate),
                group=str(group),
                n_fft=int(args.n_fft),
                hop_length=int(args.hop_length),
                threshold_quantile=float(args.threshold_quantile),
                threshold_std=float(args.threshold_std),
                refractory_ms=float(args.refractory_ms),
            )
            metrics = _match_events(target_events[str(group)], predicted_events, tolerance_sec=float(tolerance_sec))
            per_clip_rows.append(
                {
                    "dataset_index": int(dataset_index),
                    "source_id": str(row.get("source_id", "")),
                    "beat_index": int(row.get("beat_index", 0)),
                    "group": str(group),
                    **metrics,
                }
            )

    summary = _summarize(per_clip_rows)
    summary.update(
        {
            "cache_root": str(cache_root),
            "predictions_dir": str(predictions_dir),
            "split": str(split),
            "num_examples": int(len(prediction_rows)),
            "tolerance_ms": float(args.tolerance_ms),
            "n_fft": int(args.n_fft),
            "hop_length": int(args.hop_length),
            "threshold_quantile": float(args.threshold_quantile),
            "threshold_std": float(args.threshold_std),
            "refractory_ms": float(args.refractory_ms),
        }
    )
    _write_csv(out_dir / "per_clip_control_metrics.csv", per_clip_rows)
    write_json(out_dir / "summary.json", summary)
    print(f"wrote control-faithfulness metrics for {len(prediction_rows)} clips to {out_dir}")


if __name__ == "__main__":
    main()
