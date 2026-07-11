from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch
import torchaudio


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")


def _normalize_audio(audio_bct: torch.Tensor) -> torch.Tensor:
    audio = torch.as_tensor(audio_bct, dtype=torch.float32).detach().cpu()
    if int(audio.dim()) == 3:
        audio = audio[0]
    elif int(audio.dim()) != 2:
        raise ValueError(f"unexpected audio shape: {tuple(audio.shape)}")
    peak = audio.abs().max().clamp_min(1.0e-8)
    return (0.95 * audio / peak).contiguous()


def save_audio(path: str | Path, audio_bct: torch.Tensor, *, sample_rate: int) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path_obj), _normalize_audio(audio_bct), sample_rate=int(sample_rate))
