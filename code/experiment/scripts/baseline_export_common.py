from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from io_utils import write_json, write_jsonl
from scripts.dac_export_utils import clip_file_name


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(dict(json.loads(text)))
    return rows


def prepare_prediction_dir(out_dir: str | Path, *, overwrite: bool) -> Path:
    out_path = Path(out_dir).expanduser().resolve()
    if out_path.exists():
        if bool(overwrite):
            shutil.rmtree(out_path)
        elif any(out_path.iterdir()):
            raise FileExistsError(f"output directory already exists and is not empty: {out_path}")
    wav_dir = out_path / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    return out_path


def source_id_from_payload(row: Mapping[str, Any], payload: Mapping[str, Any]) -> str:
    return str(payload.get("source_id") or row.get("source_id") or "")


def beat_index_from_payload(row: Mapping[str, Any], payload: Mapping[str, Any]) -> int:
    return int(payload.get("beat_index", row.get("beat_index", 0)))


def manifest_row_for_prediction(
    *,
    dataset_index: int,
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    split: str,
    sample_rate: int,
    num_samples: int,
    duration_sec: float,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source_id = source_id_from_payload(row, payload)
    beat_index = beat_index_from_payload(row, payload)
    wav_rel = Path("wavs") / clip_file_name(int(dataset_index), source_id, int(beat_index))
    out = {
        "dataset_index": int(dataset_index),
        "source_id": str(source_id),
        "source_manifest_index": int(payload.get("source_manifest_index", row.get("source_manifest_index", -1))),
        "beat_index": int(beat_index),
        "split": str(split),
        "sample_rate": int(sample_rate),
        "num_samples": int(num_samples),
        "duration_sec": float(duration_sec),
        "target_num_frames": int(payload.get("target_num_frames", row.get("target_num_frames", 0))),
        "wav": str(wav_rel),
    }
    if extra:
        out.update(dict(extra))
    return out


def runtime_summary(
    *,
    baseline_name: str,
    baseline_family: str,
    baseline_role: str,
    cache_root: str | Path,
    split: str,
    out_dir: str | Path,
    manifest_rows: Iterable[Mapping[str, Any]],
    sample_rate: int,
    started_at: float,
    model_forward_sec: float,
    decode_sec: float = 0.0,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rows = list(manifest_rows)
    audio_sec = float(sum(float(row.get("duration_sec", 0.0) or 0.0) for row in rows))
    wall_sec = float(time.perf_counter() - float(started_at))
    summary: dict[str, Any] = {
        "baseline_name": str(baseline_name),
        "baseline_family": str(baseline_family),
        "baseline_role": str(baseline_role),
        "cache_root": str(Path(cache_root).expanduser().resolve()),
        "out_dir": str(Path(out_dir).expanduser().resolve()),
        "split": str(split),
        "num_examples": int(len(rows)),
        "sample_rate": int(sample_rate),
        "export_wall_sec_total": float(wall_sec),
        "model_forward_sec_total": float(model_forward_sec),
        "codec_decode_sec_total": float(decode_sec),
        "total_audio_sec_generated": float(audio_sec),
        "clips_per_sec": float(len(rows)) / max(float(wall_sec), 1.0e-8),
        "audio_sec_per_sec": float(audio_sec) / max(float(wall_sec), 1.0e-8),
        "rtf_end_to_end": float(wall_sec) / float(audio_sec) if audio_sec > 0.0 else None,
        "rtf_model_only": float(model_forward_sec) / float(audio_sec) if audio_sec > 0.0 else None,
    }
    if metadata:
        summary.update(dict(metadata))
    return summary


def write_prediction_set(out_dir: str | Path, manifest_rows: list[dict[str, Any]], summary: Mapping[str, Any]) -> None:
    out_path = Path(out_dir).expanduser().resolve()
    write_jsonl(out_path / "manifest.jsonl", manifest_rows)
    write_json(out_path / "summary.json", dict(summary))
