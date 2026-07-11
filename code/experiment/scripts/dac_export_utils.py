from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT.parent.parent
RUNS_ROOT = PACKAGE_ROOT / "runs"
RESULTS_ROOT = PACKAGE_ROOT / "results"


def preload_stdlib_inspect() -> None:
    original_path = list(sys.path)
    repo = str(REPO_ROOT)
    sys.path = [path for path in sys.path if path not in {"", repo}]
    try:
        import inspect  # noqa: F401
        import dataclasses  # noqa: F401
    finally:
        sys.path = original_path


def sanitize_name(text: str) -> str:
    clean = "".join(char if char.isalnum() else "_" for char in str(text))
    while "__" in clean:
        clean = clean.replace("__", "_")
    return clean.strip("_") or "sample"


def clip_file_name(dataset_index: int, source_id: str, beat_index: int) -> str:
    return f"{int(dataset_index):06d}__{sanitize_name(source_id)}__beat_{int(beat_index):04d}.wav"


def samples_per_latent_frame(decoded_num_samples: int, latent_num_frames: int) -> int:
    decoded = int(decoded_num_samples)
    frames = int(latent_num_frames)
    if decoded <= 0:
        raise ValueError(f"decoded_num_samples must be positive, got {decoded_num_samples}")
    if frames <= 0:
        raise ValueError(f"latent_num_frames must be positive, got {latent_num_frames}")
    if decoded % frames != 0:
        raise ValueError(
            f"decoded_num_samples={decoded} is not divisible by latent_num_frames={frames}"
        )
    return int(decoded // frames)


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return str(torch.cuda.get_device_name(device))
    return str(device)


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device=device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def valid_audio_samples(duration_sec: float, sample_rate: int, max_num_samples: int) -> int:
    requested = int(round(float(duration_sec) * float(sample_rate)))
    return int(max(1, min(int(max_num_samples), int(requested))))
