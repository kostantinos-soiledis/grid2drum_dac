#!/usr/bin/env python3
"""Re-render only symbolic conditioning grids from MIDI at a higher frame rate.

The overlay deliberately contains no audio targets. DiffusionConditioningDataset
loads targets from the canonical cache and replaces only its grid tensors with
the matching overlay entry, keeping the frontend-rate comparison paired.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime_compat import apply_runtime_compat

apply_runtime_compat()

import numpy as np
import torch

try:
    import pretty_midi
except Exception as exc:  # pragma: no cover
    raise RuntimeError("pretty_midi is required to re-render frontend grids") from exc

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

from data.family_state_cache_utils import build_midi_event_cache, render_midi_family_state_grid
from data.grid_rate_downsample import uniform_grid_times


FORMAT = "grid2drum-grid-overlay-v1"
SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, default=None)
    parser.add_argument("--rate-hz", type=float, default=500.0)
    parser.add_argument("--split", choices=("all", *SPLITS), default="all")
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(dict(json.loads(line)))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def tensor_scalar(value: Any, index: int) -> float:
    tensor = torch.as_tensor(value)
    return float(tensor[int(index)].item())


def main() -> None:
    args = parse_args()
    base_cache = args.base_cache.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    rate_hz = float(args.rate_hz)
    if rate_hz <= 0.0:
        raise ValueError("--rate-hz must be positive")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")

    base_config_path = base_cache / "config.json"
    if not base_config_path.is_file():
        raise FileNotFoundError(f"base cache config not found: {base_config_path}")
    base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    source_cache = (
        args.source_cache_root.expanduser().resolve()
        if args.source_cache_root is not None
        else Path(str(base_config.get("source_cache_root") or "")).expanduser().resolve()
    )
    if not (source_cache / "config.json").is_file():
        raise FileNotFoundError(
            f"source cache not found: {source_cache}; pass --source-cache-root explicitly"
        )

    if args.overwrite and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    selected_splits = SPLITS if args.split == "all" else (str(args.split),)
    total_written = 0

    for split in selected_splits:
        rows = load_jsonl(base_cache / "manifests" / f"{split}.jsonl")
        if int(args.max_items) > 0:
            rows = rows[: int(args.max_items)]
        iterator = tqdm(rows, desc=f"grid-overlay[{split}]", unit="example") if tqdm is not None else rows
        output_rows: list[dict[str, Any]] = []
        current_source_path: Path | None = None
        source_payload: dict[str, Any] | None = None
        midi_events: dict[str, np.ndarray] | None = None
        source_anchor_sec = 0.0

        for row in iterator:
            base_example_path = (base_cache / str(row["out_pt"])).resolve()
            base_payload = dict(torch.load(base_example_path, map_location="cpu", weights_only=False))
            source_path = (source_cache / str(base_payload["source_pt_rel"])).resolve()
            if source_path != current_source_path:
                current_source_path = source_path
                source_payload = dict(torch.load(source_path, map_location="cpu", weights_only=False))
                source_meta = dict(source_payload.get("source_meta") or {})
                midi_value = str(source_meta.get("midi_file") or "").strip()
                if not midi_value:
                    raise KeyError(f"source shard has no source_meta.midi_file: {source_path}")
                midi_path = Path(midi_value)
                if not midi_path.is_absolute():
                    midi_path = dataset_root / midi_path
                if not midi_path.is_file():
                    raise FileNotFoundError(f"MIDI file not found: {midi_path}")
                midi_events = build_midi_event_cache(pretty_midi.PrettyMIDI(str(midi_path)))
                source_anchor_sec = float(
                    source_meta.get("audio_start_anchor_sec", source_meta.get("start_sec", 0.0)) or 0.0
                )
            assert source_payload is not None and midi_events is not None

            source_row = int(base_payload["source_row_in_shard"])
            source_start = tensor_scalar(source_payload["beat_start_sec"], source_row)
            source_end = tensor_scalar(source_payload["beat_end_sec"], source_row)
            duration_sec = float(base_payload["duration_sec"])
            num_frames = int(max(1, round(duration_sec * rate_hz)))
            grid_np, ids_np, onsets_np, counts_np, _support, _salience = render_midi_family_state_grid(
                midi_events=midi_events,
                start_sec=float(source_anchor_sec + source_start),
                end_sec=float(source_anchor_sec + source_end),
                num_frames=num_frames,
            )
            overlay = {
                "format": FORMAT,
                "source_id": str(base_payload.get("source_id") or ""),
                "source_manifest_index": int(base_payload["source_manifest_index"]),
                "grid_frame_rate": rate_hz,
                "grid_num_frames": num_frames,
                "grid_ft": torch.from_numpy(np.asarray(grid_np, dtype=np.float32)).contiguous(),
                "grid_ids_ft": torch.from_numpy(np.asarray(ids_np, dtype=np.int16)).contiguous(),
                "family_onsets_ft": torch.from_numpy(np.asarray(onsets_np, dtype=np.bool_)).contiguous(),
                "family_onset_count_ft": torch.from_numpy(np.asarray(counts_np, dtype=np.uint8)).contiguous(),
                "grid_times_sec_t": uniform_grid_times(num_frames, duration_sec),
            }
            output_path = (out_root / str(row["out_pt"])).resolve()
            if args.overwrite or not output_path.is_file():
                output_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
                torch.save(overlay, temp_path)
                temp_path.replace(output_path)
                total_written += 1
            output_rows.append(
                {
                    **row,
                    "grid_frame_rate": rate_hz,
                    "grid_num_frames": num_frames,
                    "source_grid_num_frames": num_frames,
                }
            )
        write_jsonl(out_root / "manifests" / f"{split}.jsonl", output_rows)

    config = {
        "format": FORMAT,
        "base_cache_root": str(base_cache),
        "source_cache_root": str(source_cache),
        "dataset_root": str(dataset_root),
        "grid_frame_rate": rate_hz,
        "splits": list(selected_splits),
    }
    (out_root / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"grid overlay ready: {out_root} ({total_written} files written)")


if __name__ == "__main__":
    main()
