#!/usr/bin/env python3
"""Evaluate exported diffusion WAV predictions against decoded target audio."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

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

from data.encodec_utils import load_audio_codec_model, resolve_codec_metadata_from_cache_config, resolve_device
from io_utils import write_json
from model import (
    decode_latent_to_audio,
    mrstft_logmag_l1_per_example,
    resolve_valid_audio_num_samples,
)


def _progress(iterable: Any, *, desc: str) -> Any:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, leave=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate exported diffusion WAV predictions with direct audio L1 and MRSTFT log-magnitude L1.",
    )
    parser.add_argument(
        "--cache-root",
        type=str,
        default=str(RUNS_ROOT / "mini_cache"),
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--predictions-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = str(line).strip()
            if not text:
                continue
            rows.append(dict(json.loads(text)))
    return rows


def _resolve_out_dir(predictions_dir: Path, explicit_out_dir: str) -> Path:
    if str(explicit_out_dir).strip():
        return Path(explicit_out_dir).expanduser().resolve()
    return (predictions_dir / "direct_audio_eval").resolve()


def _peak_normalize_audio(audio_bct: torch.Tensor) -> torch.Tensor:
    audio = torch.as_tensor(audio_bct, dtype=torch.float32)
    if int(audio.dim()) == 2:
        audio = audio.unsqueeze(0)
    if int(audio.dim()) != 3:
        raise ValueError(f"expected audio [B,C,T], got {tuple(audio.shape)}")
    peak = audio.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1.0e-8)
    return (0.95 * audio / peak).contiguous()


def _pad_or_trim_audio(audio_bct: torch.Tensor, target_num_samples: int) -> torch.Tensor:
    target_len = int(max(1, int(target_num_samples)))
    audio = torch.as_tensor(audio_bct, dtype=torch.float32)
    if int(audio.shape[-1]) == target_len:
        return audio.contiguous()
    if int(audio.shape[-1]) > target_len:
        return audio[..., : int(target_len)].contiguous()
    pad = int(target_len) - int(audio.shape[-1])
    return torch.nn.functional.pad(audio, (0, int(pad))).contiguous()


@torch.no_grad()
def evaluate_predictions(
    *,
    cache_root: str | Path,
    split: str,
    predictions_dir: str | Path,
    out_dir: str | Path,
    device: str = "auto",
    max_items: int = 0,
    overwrite: bool = False,
) -> dict[str, Any]:
    cache_root_path = Path(cache_root).expanduser().resolve()
    predictions_dir_path = Path(predictions_dir).expanduser().resolve()
    out_dir_path = Path(out_dir).expanduser().resolve()
    split_name = str(split).strip().lower()

    manifest_path = predictions_dir_path / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"prediction manifest not found: {manifest_path}")
    split_manifest_path = cache_root_path / "manifests" / f"{split_name}.jsonl"
    if not split_manifest_path.is_file():
        raise FileNotFoundError(f"split manifest not found: {split_manifest_path}")

    if out_dir_path.exists():
        if bool(overwrite):
            for child in out_dir_path.iterdir():
                if child.is_dir():
                    import shutil

                    shutil.rmtree(child)
                else:
                    child.unlink()
        elif any(out_dir_path.iterdir()):
            raise FileExistsError(f"output directory already exists and is not empty: {out_dir_path}")
    out_dir_path.mkdir(parents=True, exist_ok=True)

    prediction_rows = _read_jsonl(manifest_path)
    split_rows = _read_jsonl(split_manifest_path)
    if int(max_items) > 0:
        prediction_rows = prediction_rows[: int(max_items)]

    resolved_device = resolve_device(str(device))
    torch_device = torch.device(resolved_device)
    if torch_device.type == "cuda" and torch_device.index is not None:
        torch.cuda.set_device(torch_device)
    codec_metadata = resolve_codec_metadata_from_cache_config(cache_root_path)
    encodec_model, _resolved_codec_device, codec_metadata = load_audio_codec_model(
        device=resolved_device,
        metadata=codec_metadata,
    )
    sample_rate = int(codec_metadata.codec_sample_rate)

    per_clip_rows: list[dict[str, Any]] = []
    audio_l1_values: list[float] = []
    mrstft_values: list[float] = []

    for row in _progress(prediction_rows, desc=f"direct_eval[{split_name}]"):
        dataset_index = int(row.get("dataset_index", -1))
        if dataset_index < 0 or dataset_index >= len(split_rows):
            raise IndexError(
                f"dataset_index={dataset_index} out of range for split={split_name} with {len(split_rows)} rows"
            )
        split_row = dict(split_rows[int(dataset_index)])
        example_path = (cache_root_path / str(split_row["out_pt"])).resolve()
        example_payload = dict(torch.load(example_path, map_location="cpu", weights_only=False))

        pred_wav_path = (predictions_dir_path / str(row.get("wav") or "")).resolve()
        if not pred_wav_path.is_file():
            raise FileNotFoundError(f"prediction wav missing: {pred_wav_path}")

        pred_audio_ct, pred_sample_rate = torchaudio.load(str(pred_wav_path))
        pred_audio_bct = pred_audio_ct.unsqueeze(0).to(device=torch_device, dtype=torch.float32)
        if int(pred_sample_rate) != int(sample_rate):
            pred_audio_bct = torchaudio.functional.resample(
                pred_audio_bct.squeeze(0),
                orig_freq=int(pred_sample_rate),
                new_freq=int(sample_rate),
            ).unsqueeze(0)

        target_latent_payload = example_payload.get("target_sum_td")
        if target_latent_payload is None:
            target_latent_payload = example_payload["target_sum_t128"]
        target_latent = torch.as_tensor(
            target_latent_payload,
            dtype=torch.float32,
            device=torch_device,
        ).unsqueeze(0)
        target_audio_bct = decode_latent_to_audio(target_latent, encodec_model)

        target_num_samples = int(
            resolve_valid_audio_num_samples(
                torch.tensor([float(split_row["duration_sec"])], dtype=torch.float32, device=torch_device),
                sample_rate=int(sample_rate),
                max_num_samples=int(target_audio_bct.shape[-1]),
            )[0].item()
        )

        target_audio_bct = _peak_normalize_audio(_pad_or_trim_audio(target_audio_bct, target_num_samples))
        pred_audio_bct = _peak_normalize_audio(_pad_or_trim_audio(pred_audio_bct, target_num_samples))
        valid_num_samples_b = torch.tensor([int(target_num_samples)], dtype=torch.long, device=torch_device)

        audio_l1 = float((pred_audio_bct - target_audio_bct).abs().mean().item())
        mrstft_logmag_l1 = float(
            mrstft_logmag_l1_per_example(
                pred_audio_bct,
                target_audio_bct,
                valid_num_samples_b,
            )[0].item()
        )

        audio_l1_values.append(float(audio_l1))
        mrstft_values.append(float(mrstft_logmag_l1))
        per_clip_rows.append(
            {
                "dataset_index": int(dataset_index),
                "source_id": str(split_row.get("source_id", row.get("source_id", ""))),
                "source_manifest_index": int(split_row.get("source_manifest_index", row.get("source_manifest_index", -1))),
                "beat_index": int(split_row.get("beat_index", row.get("beat_index", -1))),
                "split": str(split_row.get("split", row.get("split", split_name))),
                "pred_wav": str(pred_wav_path),
                "target_example_pt": str(example_path),
                "sample_rate": int(sample_rate),
                "target_num_samples": int(target_num_samples),
                "audio_l1": float(audio_l1),
                "mrstft_logmag_l1": float(mrstft_logmag_l1),
            }
        )

    per_clip_rows = sorted(per_clip_rows, key=lambda item: int(item["dataset_index"]))
    per_clip_path = out_dir_path / "per_clip_metrics.jsonl"
    per_clip_path.parent.mkdir(parents=True, exist_ok=True)
    per_clip_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in per_clip_rows), encoding="utf-8")
    if per_clip_rows:
        csv_path = out_dir_path / "per_clip_metrics.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_clip_rows[0].keys()))
            writer.writeheader()
            for row in per_clip_rows:
                writer.writerow(row)

    summary = {
        "cache_root": str(cache_root_path),
        "predictions_dir": str(predictions_dir_path),
        "out_dir": str(out_dir_path),
        "split": str(split_name),
        "resolved_device": str(resolved_device),
        "sample_rate": int(sample_rate),
        "num_examples": int(len(per_clip_rows)),
        "audio_l1_mean": (float(sum(audio_l1_values) / len(audio_l1_values)) if audio_l1_values else None),
        "audio_l1_median": (float(torch.tensor(audio_l1_values, dtype=torch.float32).median().item()) if audio_l1_values else None),
        "mrstft_logmag_l1_mean": (float(sum(mrstft_values) / len(mrstft_values)) if mrstft_values else None),
        "mrstft_logmag_l1_median": (float(torch.tensor(mrstft_values, dtype=torch.float32).median().item()) if mrstft_values else None),
        "metric_basis": "export_peak_normalized_audio",
    }
    write_json(out_dir_path / "summary.json", summary)
    return summary


def main() -> None:
    args = _parse_args()
    predictions_dir = Path(args.predictions_dir).expanduser().resolve()
    out_dir = _resolve_out_dir(predictions_dir, str(args.out_dir))
    summary = evaluate_predictions(
        cache_root=args.cache_root,
        split=str(args.split),
        predictions_dir=predictions_dir,
        out_dir=out_dir,
        device=str(args.device),
        max_items=int(args.max_items),
        overwrite=bool(args.overwrite),
    )
    print(
        "direct audio eval complete: "
        f"audio_l1_mean={summary['audio_l1_mean']!r} "
        f"mrstft_logmag_l1_mean={summary['mrstft_logmag_l1_mean']!r}"
    )


if __name__ == "__main__":
    main()
