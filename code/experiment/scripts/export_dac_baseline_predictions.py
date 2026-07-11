#!/usr/bin/env python3
"""Export non-learned DAC baselines in the diffusion prediction format."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


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

import torch
import torchaudio

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.diffusion_dataset import build_diffusion_dataloader
from data.encodec_utils import (
    decode_codes_to_audio_b1t,
    load_audio_codec_model,
    load_target_pca_basis,
    reconstruct_latent_from_pca,
    resolve_codec_metadata_from_cache_config,
    resolve_device,
    resolve_target_pca_basis_path_from_cache_config,
)
from io_utils import save_audio, write_json, write_jsonl
from model import decode_latent_to_audio
from scripts.dac_export_utils import (
    clip_file_name,
    device_name,
    samples_per_latent_frame,
    valid_audio_samples,
)


BASELINE_MODES = (
    "target_dac_recon",
    "target_pca_recon",
    "source_code_decode",
    "symbolic_nn_train",
    "grid_render",
)


def _progress(iterable: Any, *, desc: str) -> Any:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, leave=False)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(dict(json.loads(text)))
    return rows


def _load_payload(cache_root: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    return dict(torch.load(cache_root / str(row["out_pt"]), map_location="cpu", weights_only=False))


def _feature_from_payload(payload: Mapping[str, Any]) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for key in ("grid16_onset_count", "grid16_onset_vel", "grid16_state_vel"):
        value = payload.get(key)
        if value is not None:
            parts.append(torch.as_tensor(value, dtype=torch.float32).flatten())
    ids = payload.get("grid16_onset_ids")
    if ids is not None:
        ids_t = torch.as_tensor(ids, dtype=torch.float32)
        parts.append(torch.where(ids_t.ge(0), ids_t / 8.0, torch.zeros_like(ids_t)).flatten())
    if not parts:
        grid = torch.as_tensor(payload["grid_ft"], dtype=torch.float32)
        parts.append(torch.cat([grid.mean(dim=1), grid.amax(dim=1)], dim=0))
    parts.append(torch.tensor([float(payload.get("bpm", 0.0)) / 240.0], dtype=torch.float32))
    return torch.cat(parts, dim=0).contiguous()


def _prepare_nn_index(
    *,
    cache_root: Path,
    split: str,
    max_items: int,
) -> tuple[torch.Tensor, list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _read_jsonl(cache_root / "manifests" / f"{split}.jsonl")
    if int(max_items) > 0:
        rows = rows[: int(max_items)]
    payloads: list[dict[str, Any]] = []
    features: list[torch.Tensor] = []
    for row in _progress(rows, desc=f"nn-index[{split}]"):
        payload = _load_payload(cache_root, row)
        payloads.append(payload)
        features.append(_feature_from_payload(payload))
    if not features:
        raise RuntimeError(f"empty nearest-neighbor index for split={split}")
    width = max(int(feat.numel()) for feat in features)
    mat = torch.zeros((len(features), width), dtype=torch.float32)
    for idx, feat in enumerate(features):
        mat[idx, : int(feat.numel())] = feat
    mat = torch.nn.functional.normalize(mat, dim=1)
    return mat.contiguous(), rows, payloads


def _nearest_payload(
    payload: Mapping[str, Any],
    *,
    train_features: torch.Tensor,
    train_payloads: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    feat = _feature_from_payload(payload)
    query = torch.zeros((1, int(train_features.shape[1])), dtype=torch.float32)
    query[0, : int(feat.numel())] = feat[: int(query.shape[1])]
    query = torch.nn.functional.normalize(query, dim=1)
    idx = int(torch.argmax(query @ train_features.T, dim=1).item())
    return train_payloads[idx]


def _load_source_manifest(cache_root: Path) -> tuple[Path | None, list[dict[str, Any]]]:
    cfg = json.loads((cache_root / "config.json").read_text(encoding="utf-8"))
    source_root_text = str(cfg.get("source_cache_root") or "").strip()
    if not source_root_text:
        return None, []
    source_root = Path(source_root_text).expanduser().resolve()
    manifest = source_root / "manifest.jsonl"
    if not manifest.is_file():
        return source_root, []
    return source_root, _read_jsonl(manifest)


def _resolve_audio_path(path_text: str, *, dataset_root: Path | None, source_root: Path | None) -> Path | None:
    text = str(path_text).strip()
    if not text:
        return None
    path = Path(text)
    candidates = [path] if path.is_absolute() else []
    if dataset_root is not None and not path.is_absolute():
        candidates.append(dataset_root / path)
    if source_root is not None and not path.is_absolute():
        candidates.append(source_root / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _event_noise(num_samples: int, *, seed: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    idx = torch.arange(int(num_samples), dtype=dtype)
    phase = idx.mul(12.9898).add(float(seed) * 78.233)
    return torch.frac(torch.sin(phase) * 43758.5453).mul(2.0).sub(1.0)


def _add_event(audio: torch.Tensor, start: int, signal: torch.Tensor, gain: float) -> None:
    if int(start) >= int(audio.numel()) or int(signal.numel()) <= 0:
        return
    start_eff = max(0, int(start))
    offset = max(0, -int(start))
    length = min(int(signal.numel()) - int(offset), int(audio.numel()) - int(start_eff))
    if int(length) <= 0:
        return
    audio[int(start_eff) : int(start_eff) + int(length)] += float(gain) * signal[int(offset) : int(offset) + int(length)]


def _decay_env(num_samples: int, *, sample_rate: int, decay_sec: float, attack_sec: float = 0.002) -> torch.Tensor:
    t = torch.arange(int(num_samples), dtype=torch.float32) / float(sample_rate)
    env = torch.exp(-t / max(1.0e-4, float(decay_sec)))
    attack = torch.clamp(t / max(1.0e-4, float(attack_sec)), 0.0, 1.0)
    return (env * attack).contiguous()


def _velocity_for_event(
    grid_ft: torch.Tensor,
    feature_row_names: Sequence[str],
    family: str,
    family_idx: int,
    frame_idx: int,
) -> float:
    onset_name = f"{family}_onset_vel"
    state_name = f"{family}_state_vel"
    candidates: list[int] = []
    if onset_name in feature_row_names:
        candidates.append(int(feature_row_names.index(onset_name)))
    if state_name in feature_row_names:
        candidates.append(int(feature_row_names.index(state_name)))
    fallback = int(family_idx) * 3 + 1
    if 0 <= fallback < int(grid_ft.shape[0]):
        candidates.append(fallback)
    for row_idx in candidates:
        value = float(grid_ft[int(row_idx), int(frame_idx)].item())
        if value > 0.0:
            return float(max(0.05, min(1.0, value)))
    return 0.75


def _procedural_grid_render(payload: Mapping[str, Any], *, sample_rate: int) -> torch.Tensor:
    duration_sec = float(payload.get("duration_sec", 0.0) or 0.0)
    num_samples = int(max(1, round(float(duration_sec) * float(sample_rate))))
    audio = torch.zeros((num_samples,), dtype=torch.float32)
    class_names = [str(x) for x in list(payload.get("class_names") or [])]
    feature_row_names = [str(x) for x in list(payload.get("feature_row_names") or [])]
    if not class_names:
        class_names = ["kick", "snare", "tom_high", "tom_mid", "tom_floor", "hihat", "crash", "ride"]
    grid_ft = torch.as_tensor(payload.get("grid_ft"), dtype=torch.float32)
    onsets_ct = torch.as_tensor(payload.get("family_onsets_ft"), dtype=torch.bool)
    ids_ct = torch.as_tensor(payload.get("grid_ids_ft"), dtype=torch.long)
    grid_times = torch.as_tensor(payload.get("grid_times_sec_t"), dtype=torch.float32)
    if int(onsets_ct.dim()) != 2 or int(grid_times.numel()) <= 0:
        return audio.view(1, 1, -1)

    for family_idx, family in enumerate(class_names[: int(onsets_ct.shape[0])]):
        frames = torch.nonzero(onsets_ct[int(family_idx)], as_tuple=False).flatten().tolist()
        for event_idx, frame_idx in enumerate(frames):
            time_sec = float(grid_times[int(frame_idx)].item())
            start = int(round(float(time_sec) * float(sample_rate)))
            velocity = _velocity_for_event(grid_ft, feature_row_names, family, int(family_idx), int(frame_idx))
            class_id = int(ids_ct[int(family_idx), int(frame_idx)].item()) if int(ids_ct.dim()) == 2 else 0
            seed = (int(family_idx) + 1) * 1009 + (int(frame_idx) + 1) * 9176 + (int(class_id) + 2) * 37

            if family == "kick":
                length = int(round(0.28 * float(sample_rate)))
                t = torch.arange(length, dtype=torch.float32) / float(sample_rate)
                env = _decay_env(length, sample_rate=int(sample_rate), decay_sec=0.085, attack_sec=0.001)
                freq = 46.0 + 56.0 * torch.exp(-t / 0.045)
                phase = 2.0 * math.pi * torch.cumsum(freq / float(sample_rate), dim=0)
                signal = torch.sin(phase).mul(env)
                click = _event_noise(int(round(0.012 * float(sample_rate))), seed=seed).mul(
                    _decay_env(int(round(0.012 * float(sample_rate))), sample_rate=int(sample_rate), decay_sec=0.004)
                )
                _add_event(audio, start, signal, 1.05 * velocity)
                _add_event(audio, start, click, 0.18 * velocity)
            elif family == "snare":
                length = int(round(0.22 * float(sample_rate)))
                t = torch.arange(length, dtype=torch.float32) / float(sample_rate)
                env = _decay_env(length, sample_rate=int(sample_rate), decay_sec=0.060, attack_sec=0.0015)
                noise = _event_noise(length, seed=seed)
                tone = torch.sin(2.0 * math.pi * 190.0 * t).mul(torch.exp(-t / 0.035))
                signal = (0.82 * noise + 0.18 * tone).mul(env)
                _add_event(audio, start, signal, 0.72 * velocity)
            elif family.startswith("tom"):
                freq = {"tom_high": 170.0, "tom_mid": 125.0, "tom_floor": 88.0}.get(family, 120.0)
                length = int(round(0.32 * float(sample_rate)))
                t = torch.arange(length, dtype=torch.float32) / float(sample_rate)
                env = _decay_env(length, sample_rate=int(sample_rate), decay_sec=0.115, attack_sec=0.001)
                signal = torch.sin(2.0 * math.pi * float(freq) * t).mul(env)
                _add_event(audio, start, signal, 0.78 * velocity)
            elif family == "hihat":
                decay = 0.38 if int(class_id) in {0, 1} else 0.045
                length = int(round((0.52 if decay > 0.1 else 0.10) * float(sample_rate)))
                env = _decay_env(length, sample_rate=int(sample_rate), decay_sec=decay, attack_sec=0.0008)
                noise = _event_noise(length, seed=seed)
                high = noise - torch.nn.functional.pad(noise[:-1], (1, 0))
                _add_event(audio, start, high.mul(env), 0.25 * velocity)
            elif family in {"crash", "ride"}:
                decay = 0.72 if family == "crash" else 0.42
                length = int(round((0.90 if family == "crash" else 0.62) * float(sample_rate)))
                t = torch.arange(length, dtype=torch.float32) / float(sample_rate)
                env = _decay_env(length, sample_rate=int(sample_rate), decay_sec=decay, attack_sec=0.0015)
                noise = _event_noise(length, seed=seed)
                ping = torch.sin(2.0 * math.pi * (520.0 if family == "ride" else 410.0) * t).mul(torch.exp(-t / 0.15))
                signal = (0.72 * noise + 0.28 * ping).mul(env)
                _add_event(audio, start, signal, (0.27 if family == "crash" else 0.20) * velocity)
            else:
                length = int(round(0.12 * float(sample_rate)))
                env = _decay_env(length, sample_rate=int(sample_rate), decay_sec=0.055)
                _add_event(audio, start, _event_noise(length, seed=seed + event_idx).mul(env), 0.35 * velocity)

    peak = float(audio.abs().max().item()) if int(audio.numel()) else 0.0
    if peak > 0.98:
        audio = audio / peak * 0.98
    return audio.view(1, 1, -1).contiguous()


def _render_audio_for_row(
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    source_rows: Sequence[Mapping[str, Any]],
    source_root: Path | None,
    dataset_root: Path | None,
    sample_rate: int,
) -> tuple[torch.Tensor, str]:
    source_idx = int(row.get("source_manifest_index", -1))
    if not (0 <= source_idx < len(source_rows)):
        return _procedural_grid_render(payload, sample_rate=int(sample_rate)), "procedural_grid"
    source_row = dict(source_rows[source_idx])
    audio_path = _resolve_audio_path(
        str(source_row.get("source_rend_wav") or source_row.get("rendered_wav") or ""),
        dataset_root=dataset_root,
        source_root=source_root,
    )
    if audio_path is None:
        return _procedural_grid_render(payload, sample_rate=int(sample_rate)), "procedural_grid"
    audio_ct, sr = torchaudio.load(str(audio_path))
    if int(audio_ct.shape[0]) > 1:
        audio_ct = audio_ct.mean(dim=0, keepdim=True)
    if int(sr) != int(sample_rate):
        audio_ct = torchaudio.functional.resample(audio_ct, orig_freq=int(sr), new_freq=int(sample_rate))
    start = int(round(float(source_row.get("start_sec", row.get("start_sec", 0.0))) * float(sample_rate)))
    end = int(round(float(source_row.get("end_sec", float(row.get("duration_sec", 0.0)))) * float(sample_rate)))
    target_len = int(round(float(row.get("duration_sec", 0.0)) * float(sample_rate)))
    if end <= start:
        end = start + target_len
    segment = audio_ct[:, max(0, start) : max(0, end)].contiguous()
    if int(segment.shape[-1]) < target_len:
        segment = torch.nn.functional.pad(segment, (0, int(target_len) - int(segment.shape[-1])))
    return segment[:, :target_len].unsqueeze(0).contiguous(), "source_render_wav"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=str, default=str(RUNS_ROOT / "mini_cache"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--out-root", type=str, default=str(RUNS_ROOT / "runs_baselines" / "dac_test"))
    parser.add_argument("--modes", nargs="+", default=list(BASELINE_MODES), choices=BASELINE_MODES)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--nn-train-split", type=str, default="train")
    parser.add_argument("--nn-max-train-items", type=int, default=0)
    parser.add_argument("--dataset-root", type=str, default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    cache_root = Path(args.cache_root).expanduser().resolve()
    split = str(args.split).strip().lower()
    out_root = Path(args.out_root).expanduser().resolve()
    if out_root.exists() and bool(args.overwrite):
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    resolved_device = resolve_device(str(args.device))
    device = torch.device(resolved_device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    codec_metadata = resolve_codec_metadata_from_cache_config(cache_root)
    codec_model, _codec_device, codec_metadata = load_audio_codec_model(device=resolved_device, metadata=codec_metadata)
    sample_rate = int(codec_metadata.codec_sample_rate)
    pca_basis_path = resolve_target_pca_basis_path_from_cache_config(cache_root)
    pca_basis = load_target_pca_basis(pca_basis_path, device=device) if pca_basis_path is not None else None

    dataloader = build_diffusion_dataloader(
        cache_root,
        split=split,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        max_items=int(args.max_items),
        pin_memory=False,
    )
    split_rows = _read_jsonl(cache_root / "manifests" / f"{split}.jsonl")
    if int(args.max_items) > 0:
        split_rows = split_rows[: int(args.max_items)]

    train_features = None
    train_payloads: list[dict[str, Any]] = []
    if "symbolic_nn_train" in set(args.modes):
        train_features, _train_rows, train_payloads = _prepare_nn_index(
            cache_root=cache_root,
            split=str(args.nn_train_split).strip().lower(),
            max_items=int(args.nn_max_train_items),
        )
    source_root = None
    source_rows: list[dict[str, Any]] = []
    dataset_root = Path(args.dataset_root).expanduser().resolve() if str(args.dataset_root).strip() else None
    if "grid_render" in set(args.modes):
        source_root, source_rows = _load_source_manifest(cache_root)

    summaries: dict[str, Any] = {}
    for mode in args.modes:
        out_dir = (out_root / str(mode)).resolve()
        if out_dir.exists():
            if bool(args.overwrite):
                shutil.rmtree(out_dir)
            elif any(out_dir.iterdir()):
                raise FileExistsError(f"output directory already exists and is not empty: {out_dir}")
        wav_dir = out_dir / "wavs"
        wav_dir.mkdir(parents=True, exist_ok=True)

        manifest_rows: list[dict[str, Any]] = []
        started = time.perf_counter()
        decode_sec = 0.0
        audio_sec = 0.0
        dataset_index = 0

        if mode == "symbolic_nn_train":
            assert train_features is not None
            for row in _progress(split_rows, desc=f"export[{mode}]"):
                payload = _load_payload(cache_root, row)
                nn_payload = _nearest_payload(payload, train_features=train_features, train_payloads=train_payloads)
                codes = torch.as_tensor(nn_payload["source_codes_ct"], dtype=torch.long, device=device).unsqueeze(0)
                decode_t0 = time.perf_counter()
                decoded = decode_codes_to_audio_b1t(codec_model, codes, device=device, metadata=codec_metadata)
                decode_sec += float(time.perf_counter() - decode_t0)
                target_frames = int(payload["target_num_frames"])
                spf = samples_per_latent_frame(int(decoded.shape[-1]), int(codes.shape[-1]))
                num_samples = min(int(decoded.shape[-1]), int(target_frames) * int(spf))
                audio_i = decoded[:, :, :num_samples].detach().cpu()
                source_id = str(payload.get("source_id") or row.get("source_id") or "")
                beat_index = int(payload.get("beat_index", row.get("beat_index", 0)))
                wav_rel = Path("wavs") / clip_file_name(dataset_index, source_id, beat_index)
                save_audio(out_dir / wav_rel, audio_i, sample_rate=sample_rate)
                duration_sec = float(num_samples) / float(sample_rate)
                audio_sec += duration_sec
                manifest_rows.append({
                    "dataset_index": int(dataset_index),
                    "source_id": source_id,
                    "source_manifest_index": int(payload.get("source_manifest_index", row.get("source_manifest_index", -1))),
                    "beat_index": int(beat_index),
                    "split": split,
                    "sample_rate": int(sample_rate),
                    "num_samples": int(num_samples),
                    "duration_sec": float(duration_sec),
                    "target_num_frames": int(target_frames),
                    "wav": str(wav_rel),
                })
                dataset_index += 1
        elif mode == "grid_render":
            for row in _progress(split_rows, desc=f"export[{mode}]"):
                payload = _load_payload(cache_root, row)
                audio_i, render_kind = _render_audio_for_row(
                    row,
                    payload,
                    source_rows=source_rows,
                    source_root=source_root,
                    dataset_root=dataset_root,
                    sample_rate=int(sample_rate),
                )
                num_samples = int(audio_i.shape[-1])
                source_id = str(payload.get("source_id") or row.get("source_id") or "")
                beat_index = int(payload.get("beat_index", row.get("beat_index", 0)))
                wav_rel = Path("wavs") / clip_file_name(dataset_index, source_id, beat_index)
                save_audio(out_dir / wav_rel, audio_i.cpu(), sample_rate=sample_rate)
                duration_sec = float(num_samples) / float(sample_rate)
                audio_sec += duration_sec
                manifest_rows.append({
                    "dataset_index": int(dataset_index),
                    "source_id": source_id,
                    "source_manifest_index": int(payload.get("source_manifest_index", row.get("source_manifest_index", -1))),
                    "beat_index": int(beat_index),
                    "split": split,
                    "sample_rate": int(sample_rate),
                    "num_samples": int(num_samples),
                    "duration_sec": float(duration_sec),
                    "target_num_frames": int(payload["target_num_frames"]),
                    "render_kind": str(render_kind),
                    "wav": str(wav_rel),
                })
                dataset_index += 1
        else:
            for batch in _progress(dataloader, desc=f"export[{mode}]"):
                target_mask = torch.as_tensor(batch["target_valid_mask_bt"], dtype=torch.bool, device=device)
                target_frames_b = torch.as_tensor(batch["target_num_frames_b"], dtype=torch.long, device=device)
                duration_b = torch.as_tensor(batch["duration_sec"], dtype=torch.float32, device=device)
                if mode == "target_pca_recon":
                    if pca_basis is None:
                        raise FileNotFoundError("target_pca_recon requested but no PCA basis is available")
                    latent = reconstruct_latent_from_pca(
                        torch.as_tensor(batch["target_btd"], dtype=torch.float32, device=device),
                        pca_basis,
                    )
                    latent = latent * target_mask.unsqueeze(-1).to(dtype=latent.dtype)
                    decode_t0 = time.perf_counter()
                    decoded = decode_latent_to_audio(latent, codec_model)
                    decode_sec += float(time.perf_counter() - decode_t0)
                elif mode == "target_dac_recon":
                    latent = torch.as_tensor(batch["target_sum_btd"], dtype=torch.float32, device=device)
                    latent = latent * target_mask.unsqueeze(-1).to(dtype=latent.dtype)
                    decode_t0 = time.perf_counter()
                    decoded = decode_latent_to_audio(latent, codec_model)
                    decode_sec += float(time.perf_counter() - decode_t0)
                elif mode == "source_code_decode":
                    codes = torch.as_tensor(batch["source_codes_bct"], dtype=torch.long, device=device)
                    decode_t0 = time.perf_counter()
                    decoded = decode_codes_to_audio_b1t(codec_model, codes, device=device, metadata=codec_metadata)
                    decode_sec += float(time.perf_counter() - decode_t0)
                else:
                    raise AssertionError(mode)
                spf = samples_per_latent_frame(int(decoded.shape[-1]), int(target_mask.shape[1]))
                batch_size = int(decoded.shape[0])
                for batch_idx in range(batch_size):
                    source_id = str(batch["source_id"][batch_idx])
                    beat_index = int(torch.as_tensor(batch["beat_index_b"][batch_idx]).item())
                    source_manifest_index = int(torch.as_tensor(batch["source_manifest_index_b"][batch_idx]).item())
                    target_frames = int(target_frames_b[batch_idx].item())
                    num_samples = min(
                        int(target_frames) * int(spf),
                        valid_audio_samples(float(duration_b[batch_idx].item()), int(sample_rate), int(decoded.shape[-1])),
                    )
                    wav_rel = Path("wavs") / clip_file_name(dataset_index, source_id, beat_index)
                    audio_i = decoded[int(batch_idx) : int(batch_idx) + 1, :, :num_samples].detach().cpu()
                    save_audio(out_dir / wav_rel, audio_i, sample_rate=sample_rate)
                    duration_sec = float(num_samples) / float(sample_rate)
                    audio_sec += duration_sec
                    manifest_rows.append({
                        "dataset_index": int(dataset_index),
                        "source_id": source_id,
                        "source_manifest_index": int(source_manifest_index),
                        "beat_index": int(beat_index),
                        "split": split,
                        "sample_rate": int(sample_rate),
                        "num_samples": int(num_samples),
                        "duration_sec": float(duration_sec),
                        "target_num_frames": int(target_frames),
                        "wav": str(wav_rel),
                    })
                    dataset_index += 1

        wall_sec = float(time.perf_counter() - started)
        summary = {
            "baseline_mode": str(mode),
            "cache_root": str(cache_root),
            "out_dir": str(out_dir),
            "split": str(split),
            "num_examples": int(len(manifest_rows)),
            "sample_rate": int(sample_rate),
            "resolved_device": str(resolved_device),
            "device_name": device_name(device),
            "export_wall_sec_total": float(wall_sec),
            "model_forward_sec_total": 0.0,
            "codec_decode_sec_total": float(decode_sec),
            "total_audio_sec_generated": float(audio_sec),
            "clips_per_sec": float(len(manifest_rows)) / max(float(wall_sec), 1.0e-8),
            "audio_sec_per_sec": float(audio_sec) / max(float(wall_sec), 1.0e-8),
            "rtf_end_to_end": float(wall_sec) / float(audio_sec) if float(audio_sec) > 0 else None,
            "rtf_model_only": 0.0,
            "num_parameters": 0,
        }
        if device.type == "cuda":
            summary["peak_gpu_mem_allocated_mb"] = float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2))
            summary["peak_gpu_mem_reserved_mb"] = float(torch.cuda.max_memory_reserved(device) / (1024.0 ** 2))
        write_jsonl(out_dir / "manifest.jsonl", manifest_rows)
        write_json(out_dir / "summary.json", summary)
        summaries[str(mode)] = summary
        print(f"exported {len(manifest_rows)} {mode} predictions to {out_dir}")

    write_json(out_root / "summary.json", {"out_root": str(out_root), "modes": summaries})


if __name__ == "__main__":
    main()
