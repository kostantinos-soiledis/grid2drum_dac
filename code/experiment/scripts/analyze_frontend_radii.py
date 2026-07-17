#!/usr/bin/env python3
"""Analyze seconds-grid frontend radii against token-latent correlation."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime_compat import apply_runtime_compat

apply_runtime_compat()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.diffusion_cache_utils import (
    FAMILY_STATE_FEATURE_ROW_NAMES,
    FAMILY_STATE_FAMILY_NAMES,
    FAMILY_STATE_ID_VOCAB_SIZES,
    grid_times_from_fps,
    stack_family_state_grid,
    token_times_from_duration,
)
from data.encodec_utils import (
    extract_codebook_embeddings,
    load_audio_codec_model,
    resolve_codec_metadata_from_cache_config,
    resolve_device,
    summed_frame_latents_from_code_ids,
)
from data.seconds_frontend import (
    expand_sampled_id_windows_onehot,
    sample_grid_id_windows_in_seconds,
    sample_grid_windows_in_seconds,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Redesign seconds-frontend radii from raw feature/latent correlation.")
    parser.add_argument("--source-cache-root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"])
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-items", type=int, default=1024)
    parser.add_argument("--max-frames", type=int, default=80000)
    parser.add_argument("--max-radius", type=int, default=64, help="Maximum grid-step radius to analyze.")
    parser.add_argument(
        "--recommend-thresholds",
        type=str,
        default="0.50,0.75,0.90",
        help="Cumulative correlation-energy thresholds for nonzero recommended radii.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _load_source_manifest(source_cache_root: Path, split: str) -> list[dict[str, Any]]:
    manifest_path = source_cache_root / "manifest.jsonl"
    rows: list[dict[str, Any]] = []
    for line_idx, raw_line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines()):
        text = str(raw_line).strip()
        if not text:
            continue
        row = dict(json.loads(text))
        if str(row.get("split", "")).strip().lower() != str(split):
            continue
        row["_source_manifest_index"] = int(line_idx)
        rows.append(row)
    if not rows:
        raise RuntimeError(f"no rows found for split={split!r} under {source_cache_root}")
    return rows


def _resolve_num_beats(row: dict[str, Any]) -> int:
    if "num_beats" in row:
        return int(row["num_beats"])
    if "beat_index_end" in row and "beat_index" in row:
        return int(row["beat_index_end"]) - int(row["beat_index"]) + 1
    return 4


def _slice_row_tensor(tensor: Any, row_idx: int, valid_len: int | None = None) -> torch.Tensor:
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if not torch.is_tensor(tensor):
        raise TypeError(f"expected tensor-like source payload, got {type(tensor).__name__}")
    row = tensor[int(row_idx)]
    if valid_len is not None:
        row = row[..., : int(valid_len)]
    return row.detach().to(device="cpu").contiguous()


def _load_shard(shard_path: Path) -> dict[str, Any]:
    return dict(torch.load(shard_path, map_location="cpu", weights_only=False))


def _parse_thresholds(text: str) -> list[float]:
    out: list[float] = []
    for part in str(text).split(","):
        piece = str(part).strip()
        if not piece:
            continue
        value = float(piece)
        if not 0.0 < value < 1.0:
            raise ValueError(f"threshold must be between 0 and 1, got {value}")
        out.append(float(value))
    if not out:
        raise ValueError("at least one threshold is required")
    return out


def _prepare_out_dir(out_dir: Path, overwrite: bool) -> None:
    if bool(overwrite) and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def _combine_frontend_position_features(
    *,
    grid_ft: torch.Tensor,
    onset_ids_ft: torch.Tensor,
    grid_times_sec_t: torch.Tensor,
    token_times_sec_t: torch.Tensor,
    class_id_vocab_sizes: list[int],
    max_radius: int,
) -> torch.Tensor:
    grid_bft = grid_ft.unsqueeze(0).to(dtype=torch.float32)
    grid_ids_bct = onset_ids_ft.unsqueeze(0).to(dtype=torch.long)
    grid_times_bt = grid_times_sec_t.unsqueeze(0).to(dtype=torch.float32)
    token_times_bt = token_times_sec_t.unsqueeze(0).to(dtype=torch.float32)
    grid_valid_mask_bt = torch.ones((1, int(grid_ft.shape[-1])), dtype=torch.bool)
    valid_mask_bt = torch.ones((1, int(token_times_sec_t.shape[0])), dtype=torch.bool)
    numeric = sample_grid_windows_in_seconds(
        grid_bft=grid_bft,
        grid_times_sec_bt=grid_times_bt,
        token_times_sec_bt=token_times_bt,
        window_radius=int(max_radius),
        step_seconds=0.0,
        grid_valid_mask_bt=grid_valid_mask_bt,
        valid_mask_bt=valid_mask_bt,
    )[0]
    sampled_ids = sample_grid_id_windows_in_seconds(
        grid_ids_bct=grid_ids_bct,
        grid_times_sec_bt=grid_times_bt,
        token_times_sec_bt=token_times_bt,
        window_radius=int(max_radius),
        step_seconds=0.0,
        grid_valid_mask_bt=grid_valid_mask_bt,
        valid_mask_bt=valid_mask_bt,
    )
    id_onehot = expand_sampled_id_windows_onehot(
        sampled_ids,
        class_id_vocab_sizes=class_id_vocab_sizes,
    )
    if id_onehot is None:
        return numeric.contiguous()
    return torch.cat([numeric, id_onehot[0].to(dtype=numeric.dtype)], dim=1).contiguous()


def _choose_recommended_radii(
    cumulative_energy: np.ndarray,
    *,
    thresholds: list[float],
) -> list[int]:
    radii = [0]
    for threshold in list(thresholds):
        idx = int(np.searchsorted(cumulative_energy, float(threshold), side="left"))
        idx = int(np.clip(idx, 0, int(cumulative_energy.shape[0]) - 1))
        if int(idx) not in set(radii):
            radii.append(int(idx))
    if int(len(radii)) < int(len(thresholds) + 1):
        for idx in range(1, int(cumulative_energy.shape[0])):
            if int(idx) not in set(radii):
                radii.append(int(idx))
            if int(len(radii)) >= int(len(thresholds) + 1):
                break
    return sorted(set(int(x) for x in list(radii)))


def _plot_profile(
    *,
    out_path: Path,
    radius_steps: np.ndarray,
    score_feature_maxabs: np.ndarray,
    score_rms: np.ndarray,
    cumulative_energy: np.ndarray,
    recommended_radii: list[int],
    grid_step_sec: float,
) -> None:
    seconds = radius_steps.astype(np.float32) * np.float32(grid_step_sec)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    axes[0].plot(seconds, score_feature_maxabs, label="mean_f max_d |corr|", lw=1.5)
    axes[0].plot(seconds, score_rms, label="rms corr", lw=1.0, alpha=0.8)
    axes[0].set_title("Offset Correlation by Absolute Radius")
    axes[0].set_xlabel("radius (seconds)")
    axes[0].set_ylabel("score")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(seconds, cumulative_energy, lw=1.5, color="tab:green")
    axes[1].set_title("Cumulative Correlation Energy")
    axes[1].set_xlabel("radius (seconds)")
    axes[1].set_ylabel("fraction")
    axes[1].set_ylim(0.0, 1.01)
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(radius_steps, score_feature_maxabs, lw=1.5, color="tab:blue")
    for radius in list(recommended_radii):
        axes[2].axvline(float(radius), color="tab:red", lw=1.0, alpha=0.7)
        axes[2].text(float(radius), float(score_feature_maxabs.max()) * 0.95, str(radius), color="tab:red", ha="center")
    axes[2].set_title("Recommended Radii (grid steps)")
    axes[2].set_xlabel("radius (grid steps)")
    axes[2].set_ylabel("mean_f max_d |corr|")
    axes[2].grid(True, alpha=0.25)

    fig.savefig(str(out_path), dpi=160)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    source_cache_root = Path(args.source_cache_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    _prepare_out_dir(out_dir, overwrite=bool(args.overwrite))
    thresholds = _parse_thresholds(str(args.recommend_thresholds))
    max_items = int(max(1, int(args.max_items)))
    max_frames = int(max(1, int(args.max_frames)))
    max_radius = int(max(0, int(args.max_radius)))
    rng = np.random.default_rng(int(args.seed))
    py_rng = random.Random(int(args.seed))

    rows = _load_source_manifest(source_cache_root, split=str(args.split).strip().lower())
    py_rng.shuffle(rows)
    rows = rows[: int(max_items)]

    resolved_device = resolve_device(str(args.device))
    codec_metadata = resolve_codec_metadata_from_cache_config(source_cache_root)
    encodec_model, resolved_device, codec_metadata = load_audio_codec_model(
        device=resolved_device,
        metadata=codec_metadata,
    )
    codebook_embed_ckd = extract_codebook_embeddings(encodec_model, device=resolved_device, metadata=codec_metadata)

    class_id_vocab_sizes = list(FAMILY_STATE_ID_VOCAB_SIZES)
    feature_names = list(FAMILY_STATE_FEATURE_ROW_NAMES)
    for family_name, vocab_size in zip(list(FAMILY_STATE_FAMILY_NAMES), list(class_id_vocab_sizes)):
        if int(vocab_size) <= 1:
            continue
        feature_names.extend([f"{family_name}_id{slot}" for slot in range(int(vocab_size))])

    offset_count = int((2 * max_radius) + 1)
    feature_dim = int(len(feature_names))
    target_dim = int(codec_metadata.codec_target_dim)
    sum_x = np.zeros((offset_count, feature_dim), dtype=np.float64)
    sum_x2 = np.zeros((offset_count, feature_dim), dtype=np.float64)
    sum_xy = np.zeros((offset_count, feature_dim, target_dim), dtype=np.float64)
    sum_y = np.zeros((target_dim,), dtype=np.float64)
    sum_y2 = np.zeros((target_dim,), dtype=np.float64)
    total_frames = 0
    grid_step_values: list[float] = []
    used_rows: list[int] = []

    current_shard_path: Path | None = None
    current_shard_payload: dict[str, Any] | None = None

    for row in list(rows):
        if int(total_frames) >= int(max_frames):
            break
        shard_rel = str(row.get("pt") or "").strip()
        shard_path = (source_cache_root / shard_rel).resolve()
        if current_shard_path != shard_path:
            current_shard_path = shard_path
            current_shard_payload = _load_shard(shard_path)
        assert current_shard_payload is not None

        row_in_shard = int(row.get("row_in_shard", -1))
        code_num_frames = int(row.get("code_num_frames", _slice_row_tensor(current_shard_payload["code_num_frames"], row_in_shard).item()))
        if int(code_num_frames) <= 0:
            continue
        grid_num_frames = int(
            row.get(
                "grid_num_frames",
                _slice_row_tensor(
                    current_shard_payload.get("drumgrid_num_frames", current_shard_payload["conditioning_state_vel"].shape[-1]),
                    row_in_shard,
                ).item()
                if torch.is_tensor(current_shard_payload.get("drumgrid_num_frames"))
                else current_shard_payload["conditioning_state_vel"].shape[-1],
            )
        )
        if int(grid_num_frames) <= 1:
            continue
        duration_sec = float(row.get("duration_sec", 0.0))
        grid_frame_rate = float(row.get("grid_frame_rate", row.get("frame_rate", 0.0)) or row.get("frame_rate", 0.0) or 0.0)
        if not float(duration_sec) > 0.0 or not float(grid_frame_rate) > 0.0:
            continue

        codes_ct = _slice_row_tensor(current_shard_payload["codes"], row_in_shard, valid_len=code_num_frames).to(
            device=resolved_device,
            dtype=torch.long,
        )
        target_sum_td = summed_frame_latents_from_code_ids(codes_ct, codebook_embed_ckd).detach().to(
            device="cpu",
            dtype=torch.float32,
        )

        state_vel_ft = _slice_row_tensor(current_shard_payload["conditioning_state_vel"], row_in_shard, valid_len=grid_num_frames)
        onset_vel_ft = _slice_row_tensor(current_shard_payload["conditioning_onset_vel"], row_in_shard, valid_len=grid_num_frames)
        onset_ids_ft = _slice_row_tensor(current_shard_payload["conditioning_onset_ids"], row_in_shard, valid_len=grid_num_frames)
        onset_count_ft = _slice_row_tensor(current_shard_payload["conditioning_family_onset_count"], row_in_shard, valid_len=grid_num_frames)

        grid_ft = torch.from_numpy(
            stack_family_state_grid(
                state_vel_ft=state_vel_ft.numpy(),
                onset_vel_ft=onset_vel_ft.numpy(),
                onset_count_ft=onset_count_ft.numpy(),
            )
        ).to(dtype=torch.float32)
        grid_times_sec_t = torch.from_numpy(grid_times_from_fps(grid_num_frames, grid_frame_rate)).to(dtype=torch.float32)
        token_times_sec_t = torch.from_numpy(token_times_from_duration(code_num_frames, duration_sec)).to(dtype=torch.float32)
        if int(token_times_sec_t.shape[0]) <= 0:
            continue

        raw_windows_tfw = _combine_frontend_position_features(
            grid_ft=grid_ft,
            onset_ids_ft=onset_ids_ft,
            grid_times_sec_t=grid_times_sec_t,
            token_times_sec_t=token_times_sec_t,
            class_id_vocab_sizes=class_id_vocab_sizes,
            max_radius=max_radius,
        )
        if int(raw_windows_tfw.shape[0]) != int(target_sum_td.shape[0]):
            raise RuntimeError(
                f"window/target time mismatch for source row {row.get('_source_manifest_index')}: "
                f"{tuple(raw_windows_tfw.shape)} vs {tuple(target_sum_td.shape)}"
            )

        remaining = int(max_frames) - int(total_frames)
        if int(remaining) <= 0:
            break
        token_count = int(target_sum_td.shape[0])
        if int(token_count) > int(remaining):
            frame_indices = np.sort(rng.choice(int(token_count), size=int(remaining), replace=False))
            frame_idx_t = torch.from_numpy(frame_indices.astype(np.int64))
            raw_windows_tfw = raw_windows_tfw.index_select(0, frame_idx_t)
            target_sum_td = target_sum_td.index_select(0, frame_idx_t)
        x_tfw = raw_windows_tfw.numpy().astype(np.float64, copy=False)
        y_td = target_sum_td.numpy().astype(np.float64, copy=False)

        sum_y += y_td.sum(axis=0)
        sum_y2 += np.square(y_td).sum(axis=0)
        for offset_idx in range(int(offset_count)):
            x_tf = x_tfw[:, :, int(offset_idx)]
            sum_x[int(offset_idx)] += x_tf.sum(axis=0)
            sum_x2[int(offset_idx)] += np.square(x_tf).sum(axis=0)
            sum_xy[int(offset_idx)] += x_tf.T @ y_td

        total_frames += int(y_td.shape[0])
        used_rows.append(int(row["_source_manifest_index"]))
        grid_step_values.append(float(np.median(np.diff(grid_times_sec_t.numpy()))))

    if int(total_frames) <= 0:
        raise RuntimeError("no frames were analyzed")

    mean_y = sum_y / float(total_frames)
    var_y = np.maximum((sum_y2 / float(total_frames)) - np.square(mean_y), 1.0e-12)
    std_y = np.sqrt(var_y)

    corr_feature_maxabs = np.zeros((offset_count,), dtype=np.float64)
    corr_rms = np.zeros((offset_count,), dtype=np.float64)
    top_feature_by_offset: list[str] = []
    top_feature_score_by_offset: list[float] = []

    for offset_idx in range(int(offset_count)):
        mean_x = sum_x[int(offset_idx)] / float(total_frames)
        var_x = np.maximum((sum_x2[int(offset_idx)] / float(total_frames)) - np.square(mean_x), 1.0e-12)
        std_x = np.sqrt(var_x)
        cov = (sum_xy[int(offset_idx)] / float(total_frames)) - (mean_x[:, None] * mean_y[None, :])
        corr = cov / np.maximum(std_x[:, None] * std_y[None, :], 1.0e-12)
        abs_corr = np.abs(corr)
        feature_maxabs = abs_corr.max(axis=1)
        corr_feature_maxabs[int(offset_idx)] = float(feature_maxabs.mean())
        corr_rms[int(offset_idx)] = float(np.sqrt(np.mean(np.square(corr))))
        top_feature_idx = int(np.argmax(feature_maxabs))
        top_feature_by_offset.append(str(feature_names[int(top_feature_idx)]))
        top_feature_score_by_offset.append(float(feature_maxabs[int(top_feature_idx)]))

    center_idx = int(max_radius)
    abs_score_feature_maxabs = np.zeros((max_radius + 1,), dtype=np.float64)
    abs_score_rms = np.zeros((max_radius + 1,), dtype=np.float64)
    for radius in range(int(max_radius) + 1):
        if int(radius) == 0:
            abs_score_feature_maxabs[int(radius)] = float(corr_feature_maxabs[int(center_idx)])
            abs_score_rms[int(radius)] = float(corr_rms[int(center_idx)])
            continue
        left_idx = int(center_idx - radius)
        right_idx = int(center_idx + radius)
        abs_score_feature_maxabs[int(radius)] = float(
            corr_feature_maxabs[int(left_idx)] + corr_feature_maxabs[int(right_idx)]
        )
        abs_score_rms[int(radius)] = float(corr_rms[int(left_idx)] + corr_rms[int(right_idx)])

    cumulative_energy = np.cumsum(abs_score_feature_maxabs)
    cumulative_energy = cumulative_energy / float(max(cumulative_energy[-1], 1.0e-12))
    recommended_radii = _choose_recommended_radii(
        cumulative_energy,
        thresholds=thresholds,
    )
    primary_radius = int(recommended_radii[1] if int(len(recommended_radii)) > 1 else recommended_radii[0])
    median_grid_step_sec = float(np.median(np.asarray(grid_step_values, dtype=np.float64)))

    _plot_profile(
        out_path=out_dir / "correlation_profile.png",
        radius_steps=np.arange(max_radius + 1, dtype=np.int64),
        score_feature_maxabs=abs_score_feature_maxabs.astype(np.float32),
        score_rms=abs_score_rms.astype(np.float32),
        cumulative_energy=cumulative_energy.astype(np.float32),
        recommended_radii=recommended_radii,
        grid_step_sec=median_grid_step_sec,
    )
    _write_json(
        out_dir / "summary.json",
        {
            "source_cache_root": str(source_cache_root),
            "split": str(args.split),
            "max_items": int(max_items),
            "max_frames": int(max_frames),
            "max_radius": int(max_radius),
            "recommend_thresholds": list(thresholds),
            "analyzed_frames": int(total_frames),
            "analyzed_source_manifest_indices": list(used_rows),
            "median_grid_step_sec": float(median_grid_step_sec),
            "recommended_radii": list(recommended_radii),
            "recommended_primary_radius": int(primary_radius),
            "recommended_radii_sec": [float(median_grid_step_sec * float(radius)) for radius in list(recommended_radii)],
            "abs_offset_score_feature_maxabs": abs_score_feature_maxabs.astype(np.float32),
            "abs_offset_score_rms": abs_score_rms.astype(np.float32),
            "abs_offset_cumulative_energy": cumulative_energy.astype(np.float32),
            "top_feature_by_signed_offset": list(top_feature_by_offset),
            "top_feature_score_by_signed_offset": list(top_feature_score_by_offset),
            "feature_names": list(feature_names),
            "artifacts": {
                "correlation_profile_png": "correlation_profile.png",
            },
        },
    )

    radii_text = ",".join(str(int(x)) for x in list(recommended_radii))
    seconds_text = ",".join(f"{median_grid_step_sec * float(x):.3f}" for x in list(recommended_radii))
    print(f"recommended_radii={radii_text}")
    print(f"recommended_primary_radius={primary_radius}")
    print(f"recommended_radii_seconds={seconds_text}")
    print(f"analyzed_frames={int(total_frames)}")
    print(f"summary={out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
