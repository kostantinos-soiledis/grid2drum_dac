#!/usr/bin/env python3
"""Build framewise RVQ targets plus seconds-grid conditioning cache."""

from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
from pathlib import Path
from typing import Any

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

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime_compat import apply_runtime_compat

apply_runtime_compat()

import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

from data.diffusion_cache_utils import (
    FAMILY_STATE_FEATURE_ROW_NAMES,
    FAMILY_STATE_FAMILY_NAMES,
    FAMILY_STATE_ID_VOCAB_SIZES,
    aggregate_grid16_from_conditioning,
    expand_beat_boundaries_to_16th_steps,
    fix_component_signs,
    grid_times_from_fps,
    hash_bytes,
    hash_tensor,
    normalize_segment_beat_boundaries_rel,
    sample_beat_boundaries_from_source,
    stack_family_state_grid,
    token_times_from_duration,
)
from data.encodec_utils import (
    compute_dac_latent_subspace_diagnostics,
    extract_codebook_embeddings,
    load_audio_codec_model,
    resolve_device,
    resolve_codec_metadata_from_cache_config,
    summed_frame_latents_from_code_ids,
)


def _is_legacy_default_codec(metadata: Any) -> bool:
    return (
        str(metadata.codec_family) == "encodec"
        and str(metadata.codec_model_id) == "facebook/encodec_32khz"
        and int(metadata.codec_sample_rate) == 32000
        and int(metadata.codec_num_codebooks) == 4
        and int(metadata.codec_target_dim) == 128
        and float(metadata.encodec_bandwidth or 2.2) == 2.2
    )


def _progress(iterable, *, desc: str, total: int | None = None):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, leave=True, dynamic_ncols=True, unit="example")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build framewise RVQ diffusion cache from an aligned source cache.")
    parser.add_argument("--source-cache-root", type=str, required=True)
    parser.add_argument("--out-root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"])
    parser.add_argument("--kit", dest="kit_filters", action="append", default=None)
    parser.add_argument("--list-kits", action="store_true")
    parser.add_argument("--pca-k", type=int, default=0, help="Optional PCA rank over framewise target sums.")
    parser.add_argument(
        "--target-basis",
        type=str,
        default="pca",
        choices=("pca", "native"),
        help=(
            "pca rotates the fixed DAC output-projection subspace by its training-frame eigenbasis "
            "(variance-ordered, whitened after standardization; default). native keeps the raw orthonormal "
            "DAC subspace coordinates with no rotation, so the target stays correlated. Reconstruction is "
            "identical either way. native requires a DAC codec and --pca-k equal to the subspace rank (72)."
        ),
    )
    parser.add_argument(
        "--geometry-mode",
        type=str,
        default="source",
        choices=("source", "bpm"),
        help=(
            "source keeps source-cache beat durations/frame counts; bpm derives duration, beat boundaries, "
            "grid frame count, and target frame count from row BPM."
        ),
    )
    parser.add_argument(
        "--target-token-rate-hz",
        type=float,
        default=0.0,
        help="Target token rate for --geometry-mode bpm. Defaults to codec sample_rate / hop_length when available.",
    )
    parser.add_argument(
        "--bpm-source",
        type=str,
        default="manifest",
        choices=("manifest", "duration"),
        help=(
            "For --geometry-mode bpm, manifest uses row['bpm']; duration stores an effective BPM from "
            "num_beats/source duration to avoid half/double-time manifest BPM errors."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-items", type=int, default=0, help="Optional example cap for smoke runs.")
    return parser.parse_args()


def _clean_name(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _normalize_name_token(value: Any) -> str:
    return _clean_name(value).casefold()


def _normalize_cli_name_filters(values: list[str] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        for part in str(raw).split(","):
            name = _clean_name(part)
            if not name:
                continue
            key = name.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(name)
    return out


def _available_kit_counts(rows: list[dict[str, Any]]) -> collections.Counter[str]:
    counts: collections.Counter[str] = collections.Counter()
    for row in rows:
        name = _clean_name(row.get("kit_name"))
        if name:
            counts[name] += 1
    return counts


def _print_available_kits(rows: list[dict[str, Any]]) -> None:
    counts = _available_kit_counts(rows)
    if not counts:
        print("[info] no kit_name values found")
        return
    for name, count in sorted(counts.items(), key=lambda item: (str(item[0]).casefold(), str(item[0]))):
        print(f"{count}\t{name}")


def _filter_rows_by_kit_name(rows: list[dict[str, Any]], kit_filters: list[str]) -> list[dict[str, Any]]:
    filters = _normalize_cli_name_filters(kit_filters)
    if not filters:
        return rows
    requested = {_normalize_name_token(name): str(name) for name in filters}
    out = [row for row in rows if _normalize_name_token(row.get("kit_name")) in requested]
    if not out:
        available = sorted(_available_kit_counts(rows).keys(), key=lambda name: name.casefold())
        preview = ", ".join(available[:12])
        raise ValueError(f"no kit_name rows matched filters {filters!r}; available kits include: {preview}")
    matched = {_normalize_name_token(row.get("kit_name")) for row in out}
    missing = [name for key, name in requested.items() if key not in matched]
    if missing:
        raise ValueError(f"requested kit filters were not present in selected rows: {', '.join(missing)}")
    return out


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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")


class _StreamingPcaAccumulator:
    def __init__(
        self,
        *,
        raw_dim: int,
        projection_basis_dq: np.ndarray | None = None,
        basis_mode: str = "pca",
    ) -> None:
        self.raw_dim = int(raw_dim)
        self.basis_mode = str(basis_mode)
        if self.basis_mode not in ("pca", "native"):
            raise ValueError(f"basis_mode must be 'pca' or 'native', got {basis_mode!r}")
        if int(self.raw_dim) <= 0:
            raise ValueError(f"raw_dim must be positive, got {raw_dim}")
        self.projection_basis_dq = None
        if projection_basis_dq is not None:
            basis = np.asarray(projection_basis_dq, dtype=np.float64)
            if int(basis.ndim) != 2:
                raise ValueError(f"projection_basis_dq must be [D,Q], got {tuple(basis.shape)}")
            if int(basis.shape[0]) != int(self.raw_dim):
                raise ValueError(f"projection_basis_dq first dim must be {self.raw_dim}, got {tuple(basis.shape)}")
            self.projection_basis_dq = basis
        stat_dim = int(self.projection_basis_dq.shape[1]) if self.projection_basis_dq is not None else int(self.raw_dim)
        if int(stat_dim) <= 0:
            raise ValueError("PCA statistic dimension must be positive")
        self.stat_dim = int(stat_dim)
        self.count = 0
        self.sum_raw_d = np.zeros((int(self.raw_dim),), dtype=np.float64)
        self.sum_stat_q = np.zeros((int(self.stat_dim),), dtype=np.float64)
        self.cross_stat_qq = np.zeros((int(self.stat_dim), int(self.stat_dim)), dtype=np.float64)

    def update(self, target_sum_td: torch.Tensor) -> None:
        x = target_sum_td.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
        if int(x.ndim) != 2:
            raise ValueError(f"target_sum_td must be [T,D], got shape {tuple(x.shape)}")
        if int(x.shape[1]) != int(self.raw_dim):
            raise ValueError(f"target_sum_td dim mismatch: expected {self.raw_dim}, got {tuple(x.shape)}")
        if int(x.shape[0]) <= 0:
            return
        x64 = np.asarray(x, dtype=np.float64)
        stat = x64 @ self.projection_basis_dq if self.projection_basis_dq is not None else x64
        self.sum_raw_d += x64.sum(axis=0)
        self.sum_stat_q += stat.sum(axis=0)
        self.cross_stat_qq += stat.transpose() @ stat
        self.count += int(x64.shape[0])

    def fit(self, *, pca_k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        if int(self.count) <= 0:
            raise ValueError("cannot fit PCA with zero accumulated frames")
        mean_raw_d = self.sum_raw_d / float(self.count)
        mean_stat_q = self.sum_stat_q / float(self.count)
        cov = self.cross_stat_qq - (float(self.count) * np.outer(mean_stat_q, mean_stat_q))
        cov = cov / float(max(1, int(self.count) - 1))
        cov = 0.5 * (cov + cov.transpose())
        if self.basis_mode == "native":
            if self.projection_basis_dq is None:
                raise ValueError("native basis_mode requires a projection_basis_dq (DAC subspace)")
            # Keep the raw orthonormal DAC subspace coordinates: identity rotation, no variance
            # ordering. eigvals become per-dimension native variances (used only for diagnostics).
            eigvals = np.maximum(np.diag(cov), 0.0)
            eigvecs = np.eye(int(cov.shape[0]), dtype=np.float64)
        else:
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            eigvals = np.maximum(eigvals[order], 0.0)
            eigvecs = eigvecs[:, order]
        k = int(min(max(1, int(pca_k)), int(eigvecs.shape[1])))
        stat_components_kq = eigvecs[:, : int(k)].transpose().astype(np.float32, copy=False)
        if self.projection_basis_dq is not None:
            components = (stat_components_kq @ self.projection_basis_dq.transpose()).astype(np.float32, copy=False)
        else:
            components = stat_components_kq.astype(np.float32, copy=False)
        total_var = float(eigvals.sum())
        explained = (eigvals[: int(k)] / max(total_var, 1.0e-8)).astype(np.float32, copy=False)
        return mean_raw_d.astype(np.float32, copy=False), components, explained, int(k), int(self.count)


def _load_source_manifest(source_cache_root: Path, split: str) -> tuple[list[dict[str, Any]], str, str]:
    manifest_path = source_cache_root / "manifest.jsonl"
    rows: list[dict[str, Any]] = []
    full_lines: list[str] = []
    split_lines: list[str] = []
    for line_idx, raw_line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines()):
        text = str(raw_line).strip()
        if not text:
            continue
        full_lines.append(text)
        row = dict(json.loads(text))
        row["_source_manifest_index"] = int(line_idx)
        if str(row.get("split", "")).strip().lower() != str(split):
            continue
        split_lines.append(text)
        rows.append(row)
    return (
        rows,
        hash_bytes("\n".join(full_lines).encode("utf-8")),
        hash_bytes("\n".join(split_lines).encode("utf-8")),
    )


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


def _as_1d_numpy(tensor_like: Any) -> np.ndarray:
    if isinstance(tensor_like, np.ndarray):
        return np.asarray(tensor_like, dtype=np.float32).reshape(-1)
    if torch.is_tensor(tensor_like):
        return tensor_like.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().reshape(-1)
    return np.asarray(tensor_like, dtype=np.float32).reshape(-1)


def _codec_target_token_rate_hz(metadata: Any, *, override: float = 0.0) -> float:
    if float(override) > 0.0:
        return float(override)
    sample_rate = float(getattr(metadata, "codec_sample_rate", 0.0) or 0.0)
    hop_length = int(getattr(metadata, "codec_hop_length", 0) or 0)
    if sample_rate > 0.0 and int(hop_length) > 0:
        return float(sample_rate) / float(hop_length)
    return float(getattr(metadata, "codec_frame_rate", 0.0) or 0.0)


def _effective_bpm(
    row: dict[str, Any],
    *,
    num_beats: int,
    fallback_duration_sec: float,
    bpm_source: str,
) -> float:
    if str(bpm_source).strip().lower() == "duration" and float(fallback_duration_sec) > 0.0:
        return float(max(1, int(num_beats)) * 60.0) / float(fallback_duration_sec)
    return float(row.get("bpm", 0.0) or 0.0)


def _bpm_duration_sec(bpm: float, *, num_beats: int, fallback_duration_sec: float) -> float:
    if bpm > 1.0e-6:
        return float(max(1, int(num_beats)) * 60.0) / float(bpm)
    return float(fallback_duration_sec)


def _uniform_frame_times_from_duration(num_frames: int, duration_sec: float) -> np.ndarray:
    frames = int(max(0, int(num_frames)))
    duration = float(max(0.0, float(duration_sec)))
    if frames <= 0 or duration <= 0.0:
        return np.zeros((0,), dtype=np.float32)
    idx = np.arange(frames, dtype=np.float32)
    return ((idx + 0.5) * (duration / float(frames))).astype(np.float32, copy=False)


def _slice_or_pad_time(
    tensor: Any,
    row_idx: int,
    *,
    source_valid_len: int,
    target_len: int,
    pad_value: int | float | bool = 0,
    pad_mode: str = "constant",
) -> torch.Tensor:
    source_valid_len = int(max(0, int(source_valid_len)))
    target_len = int(max(0, int(target_len)))
    take_len = int(min(source_valid_len, target_len))
    row = _slice_row_tensor(tensor, int(row_idx), valid_len=int(take_len))
    if int(target_len) <= int(take_len):
        return row.contiguous()
    pad_len = int(target_len) - int(take_len)
    if int(row.dim()) <= 0:
        raise ValueError(f"expected at least 1D time tensor, got {tuple(row.shape)}")
    if str(pad_mode) == "edge" and int(take_len) > 0:
        pad = row[..., int(take_len) - 1 : int(take_len)].expand(*row.shape[:-1], int(pad_len)).clone()
    else:
        pad = torch.full(
            (*row.shape[:-1], int(pad_len)),
            pad_value,
            dtype=row.dtype,
            device=row.device,
        )
    return torch.cat((row, pad), dim=-1).contiguous()


def _load_shard(shard_path: Path) -> dict[str, Any]:
    return dict(torch.load(shard_path, map_location="cpu", weights_only=False))


def _validate_source_payload(payload: dict[str, Any]) -> None:
    required = (
        "codes",
        "code_num_frames",
        "conditioning_state_vel",
        "conditioning_onset_vel",
        "conditioning_onset_ids",
        "conditioning_family_onsets",
        "conditioning_family_onset_count",
        "beat_times_sec",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"source shard is missing required keys: {missing}")


def _prepare_output_dirs(out_root: Path, split: str, overwrite: bool, remove_basis: bool) -> tuple[Path, Path, Path]:
    examples_dir = out_root / "examples" / str(split)
    manifest_path = out_root / "manifests" / f"{str(split)}.jsonl"
    summary_path = out_root / "summaries" / f"{str(split)}.json"
    if bool(overwrite):
        if examples_dir.exists():
            shutil.rmtree(examples_dir)
        if manifest_path.exists():
            manifest_path.unlink()
        if summary_path.exists():
            summary_path.unlink()
        if bool(remove_basis) and (out_root / "pca_basis.pt").exists():
            (out_root / "pca_basis.pt").unlink()
    elif examples_dir.exists() or manifest_path.exists():
        raise FileExistsError(
            f"output for split={split!r} already exists under {out_root}; pass --overwrite to rebuild"
        )
    examples_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    return examples_dir, manifest_path, summary_path


def main() -> None:
    args = _parse_args()
    source_cache_root = Path(args.source_cache_root).resolve()
    out_root = Path(args.out_root).resolve()
    split = str(args.split).strip().lower()
    pca_k = int(max(0, int(args.pca_k)))
    max_items = int(max(0, int(args.max_items)))
    requested_device = resolve_device(str(args.device))

    rows, source_manifest_hash, split_manifest_hash = _load_source_manifest(source_cache_root, split=split)
    if bool(args.list_kits):
        _print_available_kits(rows)
        return
    rows = _filter_rows_by_kit_name(rows, _normalize_cli_name_filters(args.kit_filters))
    if max_items > 0:
        rows = rows[: int(max_items)]
    if not rows:
        raise RuntimeError(f"no manifest rows found for split={split!r} under {source_cache_root}")
    rows = sorted(rows, key=lambda row: (str(row.get("pt") or ""), int(row.get("row_in_shard", -1))))

    basis_path = out_root / "pca_basis.pt"
    remove_basis = split == "train"
    examples_dir, manifest_path, summary_path = _prepare_output_dirs(
        out_root=out_root,
        split=split,
        overwrite=bool(args.overwrite),
        remove_basis=bool(remove_basis),
    )

    basis_payload = None
    if int(pca_k) > 0 and split != "train":
        if not basis_path.is_file():
            raise FileNotFoundError(
                f"PCA basis missing at {basis_path}; build the train split first with --pca-k {pca_k}"
            )
        basis_payload = torch.load(basis_path, map_location="cpu", weights_only=False)
        if int(basis_payload.get("k", -1)) != int(pca_k):
            raise ValueError(
                f"requested --pca-k {pca_k} does not match existing basis k={basis_payload.get('k')}"
            )

    source_codec_metadata = resolve_codec_metadata_from_cache_config(source_cache_root)
    legacy_default_cache_name = "_".join(("cache", "4beats", "t128"))
    if not _is_legacy_default_codec(source_codec_metadata) and str(out_root.name) == legacy_default_cache_name:
        raise ValueError(
            "non-default codecs must not reuse the legacy default output root; "
            "choose a codec-specific diffusion cache path"
        )
    audio_codec_model, resolved_device, source_codec_metadata = load_audio_codec_model(
        device=requested_device,
        metadata=source_codec_metadata,
    )
    codebook_embed_ckd = extract_codebook_embeddings(
        audio_codec_model,
        device=resolved_device,
        metadata=source_codec_metadata,
    )
    codebook_hash = hash_tensor(codebook_embed_ckd)
    raw_target_dim = int(source_codec_metadata.codec_target_dim)
    geometry_mode = str(args.geometry_mode).strip().lower()
    target_token_rate_hz = _codec_target_token_rate_hz(
        source_codec_metadata,
        override=float(args.target_token_rate_hz),
    )
    if bool(geometry_mode == "bpm") and not float(target_token_rate_hz) > 0.0:
        raise ValueError(
            "--geometry-mode bpm requires a positive target token rate; pass --target-token-rate-hz explicitly"
        )
    pca_subspace_basis_t: torch.Tensor | None = None
    pca_subspace_diagnostics: dict[str, Any] | None = None
    if int(pca_k) > 0 and str(source_codec_metadata.codec_family) == "dac":
        pca_subspace_diagnostics = compute_dac_latent_subspace_diagnostics(
            audio_codec_model,
            device=resolved_device,
            metadata=source_codec_metadata,
        )
        pca_subspace_basis_t = (
            torch.as_tensor(pca_subspace_diagnostics["basis_dq"], dtype=torch.float32)
            .detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
        )
    pca_accumulator: _StreamingPcaAccumulator | None = None
    if int(pca_k) > 0 and split == "train":
        pca_accumulator = _StreamingPcaAccumulator(
            raw_dim=int(raw_target_dim),
            projection_basis_dq=(
                pca_subspace_basis_t.detach().cpu().numpy()
                if pca_subspace_basis_t is not None
                else None
            ),
            basis_mode=str(args.target_basis),
        )

    current_shard_path: Path | None = None
    current_shard_payload: dict[str, Any] | None = None
    built_entries: list[dict[str, Any]] = []

    iterator = _progress(rows, desc=f"build_diffusion_cache[{split}]", total=len(rows))
    for row in iterator:
        shard_rel = str(row.get("pt") or "").strip()
        if not shard_rel:
            raise KeyError(f"manifest row is missing pt: {row}")
        shard_path = (source_cache_root / shard_rel).resolve()
        if current_shard_path != shard_path:
            current_shard_path = shard_path
            current_shard_payload = _load_shard(shard_path)
            _validate_source_payload(current_shard_payload)
        assert current_shard_payload is not None

        row_in_shard = int(row.get("row_in_shard", -1))
        source_manifest_index = int(row["_source_manifest_index"])
        num_beats = int(_resolve_num_beats(row))
        beat_index = int(row.get("beat_index", 0))
        beat_index_end = int(row.get("beat_index_end", beat_index + num_beats - 1))
        source_duration_sec = float(row.get("duration_sec", 0.0))
        source_bpm = float(row.get("bpm", 0.0) or 0.0)
        effective_bpm = (
            _effective_bpm(
                row,
                num_beats=int(num_beats),
                fallback_duration_sec=float(source_duration_sec),
                bpm_source=str(args.bpm_source),
            )
            if str(geometry_mode) == "bpm"
            else float(source_bpm)
        )
        duration_sec = (
            _bpm_duration_sec(
                float(effective_bpm),
                num_beats=int(num_beats),
                fallback_duration_sec=float(source_duration_sec),
            )
            if str(geometry_mode) == "bpm"
            else float(source_duration_sec)
        )
        start_sec = float(row.get("start_sec", 0.0))
        end_sec = float(row.get("end_sec", 0.0))

        source_code_num_frames = int(row.get("code_num_frames", _slice_row_tensor(current_shard_payload["code_num_frames"], row_in_shard).item()))
        source_grid_num_frames = int(
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
        grid_frame_rate = float(row.get("grid_frame_rate", row.get("frame_rate", 0.0)) or row.get("frame_rate", 0.0) or 0.0)
        if not float(grid_frame_rate) > 0.0:
            raise ValueError(f"grid_frame_rate must be positive, got {grid_frame_rate} for source row {source_manifest_index}")
        if str(geometry_mode) == "bpm":
            code_num_frames = int(max(1, round(float(duration_sec) * float(target_token_rate_hz))))
            grid_num_frames = int(max(1, round(float(duration_sec) * float(grid_frame_rate))))
        else:
            code_num_frames = int(source_code_num_frames)
            grid_num_frames = int(source_grid_num_frames)
        if int(source_code_num_frames) <= 0:
            raise ValueError(
                f"source code_num_frames must be positive, got {source_code_num_frames} for source row {source_manifest_index}"
            )
        if int(source_grid_num_frames) <= 0:
            raise ValueError(
                f"source grid_num_frames must be positive, got {source_grid_num_frames} for source row {source_manifest_index}"
            )
        if int(code_num_frames) <= 0:
            raise ValueError(f"code_num_frames must be positive, got {code_num_frames} for source row {source_manifest_index}")
        if int(grid_num_frames) <= 0:
            raise ValueError(f"grid_num_frames must be positive, got {grid_num_frames} for source row {source_manifest_index}")

        codes_ct = _slice_or_pad_time(
            current_shard_payload["codes"],
            row_in_shard,
            source_valid_len=int(source_code_num_frames),
            target_len=int(code_num_frames),
            pad_mode="edge",
        ).to(
            device=resolved_device,
            dtype=torch.long,
        )
        target_sum_td = summed_frame_latents_from_code_ids(codes_ct, codebook_embed_ckd).detach().to(
            device="cpu",
            dtype=torch.float32,
        )
        target_sum_pool_d = target_sum_td.sum(dim=0).to(dtype=torch.float32).contiguous()

        state_vel_ft = _slice_or_pad_time(
            current_shard_payload["conditioning_state_vel"],
            row_in_shard,
            source_valid_len=int(source_grid_num_frames),
            target_len=int(grid_num_frames),
            pad_value=0.0,
        ).numpy()
        onset_vel_ft = _slice_or_pad_time(
            current_shard_payload["conditioning_onset_vel"],
            row_in_shard,
            source_valid_len=int(source_grid_num_frames),
            target_len=int(grid_num_frames),
            pad_value=0.0,
        ).numpy()
        onset_ids_ft = _slice_or_pad_time(
            current_shard_payload["conditioning_onset_ids"],
            row_in_shard,
            source_valid_len=int(source_grid_num_frames),
            target_len=int(grid_num_frames),
            pad_value=-1,
        ).numpy()
        family_onsets_ft = _slice_or_pad_time(
            current_shard_payload["conditioning_family_onsets"],
            row_in_shard,
            source_valid_len=int(source_grid_num_frames),
            target_len=int(grid_num_frames),
            pad_value=False,
        ).numpy()
        onset_count_ft = _slice_or_pad_time(
            current_shard_payload["conditioning_family_onset_count"],
            row_in_shard,
            source_valid_len=int(source_grid_num_frames),
            target_len=int(grid_num_frames),
            pad_value=0,
        ).numpy()

        if str(geometry_mode) == "bpm":
            beat_boundaries_rel = np.linspace(
                0.0,
                float(duration_sec),
                num=int(num_beats) + 1,
                endpoint=True,
                dtype=np.float32,
            )
        else:
            beat_boundaries_rel = sample_beat_boundaries_from_source(
                shard=current_shard_payload,
                row=row,
                duration_sec=duration_sec,
            )
            beat_boundaries_rel = normalize_segment_beat_boundaries_rel(
                beat_boundaries_rel,
                expected_num_beats=num_beats,
                duration_sec=duration_sec,
            )
        if int(beat_boundaries_rel.shape[0]) != int(num_beats + 1):
            raise ValueError(
                f"expected {int(num_beats + 1)} beat boundaries, got {tuple(beat_boundaries_rel.shape)} "
                f"for source row {source_manifest_index}"
            )
        grid_ft = stack_family_state_grid(
            state_vel_ft=state_vel_ft,
            onset_vel_ft=onset_vel_ft,
            onset_count_ft=onset_count_ft,
        )
        grid_times_sec_t = (
            _uniform_frame_times_from_duration(grid_num_frames, duration_sec)
            if str(geometry_mode) == "bpm"
            else grid_times_from_fps(grid_num_frames, grid_frame_rate)
        )
        token_times_sec_t = token_times_from_duration(code_num_frames, duration_sec)
        if int(grid_times_sec_t.shape[0]) != int(grid_num_frames):
            raise RuntimeError(
                f"grid_times_sec_t shape mismatch: expected {grid_num_frames}, got {tuple(grid_times_sec_t.shape)}"
            )
        if int(token_times_sec_t.shape[0]) != int(code_num_frames):
            raise RuntimeError(
                f"token_times_sec_t shape mismatch: expected {code_num_frames}, got {tuple(token_times_sec_t.shape)}"
            )
        step_boundaries_sec_rel = expand_beat_boundaries_to_16th_steps(beat_boundaries_rel)
        grid16 = aggregate_grid16_from_conditioning(
            state_vel_ft=state_vel_ft,
            onset_vel_ft=onset_vel_ft,
            onset_ids_ft=onset_ids_ft,
            family_onsets_ft=family_onsets_ft,
            onset_count_ft=onset_count_ft,
            step_boundaries_sec_rel=step_boundaries_sec_rel,
            duration_sec=duration_sec,
        )

        payload: dict[str, Any] = {
            "source_id": str(row.get("source_id") or ""),
            "source_row_id": row.get("source_row_id"),
            "kit_name": row.get("kit_name"),
            "split": str(split),
            "beat_index": int(beat_index),
            "beat_index_end": int(beat_index_end),
            "duration_sec": float(duration_sec),
            "source_duration_sec": float(source_duration_sec),
            "bpm": float(effective_bpm),
            "source_bpm": float(source_bpm),
            "geometry_mode": str(geometry_mode),
            "bpm_source": str(args.bpm_source),
            "target_token_rate_hz": float(target_token_rate_hz),
            "grid_frame_rate": float(grid_frame_rate),
            "grid_num_frames": int(grid_num_frames),
            "source_grid_num_frames": int(source_grid_num_frames),
            "madmom_beat_fps": int(row.get("madmom_beat_fps", -1)),
            "madmom_correct": bool(row.get("madmom_correct", False)),
            "source_manifest_index": int(source_manifest_index),
            "source_pt_rel": str(shard_rel),
            "source_row_in_shard": int(row_in_shard),
            "conditioning_mode": str(
                ((current_shard_payload.get("conditioning_cache_config") or {}).get("mode"))
                or "midi_family_state_onset_ids"
            ),
            "class_names": list(
                ((current_shard_payload.get("conditioning_cache_config") or {}).get("class_names"))
                or list(FAMILY_STATE_FAMILY_NAMES)
            ),
            "class_id_vocab_sizes": list(
                ((current_shard_payload.get("conditioning_cache_config") or {}).get("class_id_vocab_sizes"))
                or list(FAMILY_STATE_ID_VOCAB_SIZES)
            ),
            "feature_row_names": list(
                ((current_shard_payload.get("conditioning_cache_config") or {}).get("feature_row_names"))
                or list(FAMILY_STATE_FEATURE_ROW_NAMES)
            ),
            "codec_metadata": source_codec_metadata.to_dict(),
            "target_layout": "framewise_pca" if int(pca_k) > 0 else "framewise_sum",
            "target_dim": int(pca_k) if int(pca_k) > 0 else int(raw_target_dim),
            "full_target_dim": int(raw_target_dim),
            "target_sum_td": target_sum_td,
            "target_num_frames": int(code_num_frames),
            "source_target_num_frames": int(source_code_num_frames),
            "target_sum_pool_d": target_sum_pool_d,
            "source_codes_ct": codes_ct.detach().to(device="cpu", dtype=torch.int16).contiguous(),
            "grid_ft": torch.from_numpy(np.asarray(grid_ft, dtype=np.float32)).to(dtype=torch.float32),
            "grid_ids_ft": torch.from_numpy(np.asarray(onset_ids_ft, dtype=np.int16)).to(dtype=torch.int16),
            "family_onsets_ft": torch.from_numpy(np.asarray(family_onsets_ft, dtype=np.bool_)).to(dtype=torch.bool),
            "family_onset_count_ft": torch.from_numpy(np.asarray(onset_count_ft, dtype=np.uint8)).to(dtype=torch.uint8),
            "grid_times_sec_t": torch.from_numpy(np.asarray(grid_times_sec_t, dtype=np.float32)).to(dtype=torch.float32),
            "token_times_sec_t": torch.from_numpy(np.asarray(token_times_sec_t, dtype=np.float32)).to(dtype=torch.float32),
            "beat_boundaries_sec_rel": torch.from_numpy(np.asarray(beat_boundaries_rel, dtype=np.float32)).to(dtype=torch.float32),
            "grid16_state_vel": torch.from_numpy(grid16["grid16_state_vel"]).to(dtype=torch.float32),
            "grid16_onset_vel": torch.from_numpy(grid16["grid16_onset_vel"]).to(dtype=torch.float32),
            "grid16_onset_count": torch.from_numpy(grid16["grid16_onset_count"]).to(dtype=torch.uint8),
            "grid16_onset_ids": torch.from_numpy(grid16["grid16_onset_ids"]).to(dtype=torch.int16),
            "step_boundaries_sec_rel": torch.from_numpy(grid16["step_boundaries_sec_rel"]).to(dtype=torch.float32),
        }
        if int(raw_target_dim) == 128:
            payload["target_sum_t128"] = payload["target_sum_td"]
            payload["target_sum_pool_128"] = payload["target_sum_pool_d"]

        if basis_payload is not None:
            mean = torch.as_tensor(basis_payload["mean"], dtype=torch.float32)
            components = torch.as_tensor(basis_payload["components"], dtype=torch.float32)
            payload["target_pc_tk"] = (payload["target_sum_td"] - mean).matmul(components.t()).to(dtype=torch.float32)
            payload["target_pc_pool_k"] = payload["target_pc_tk"].sum(dim=0).to(dtype=torch.float32).contiguous()
            payload["pca_basis_path"] = str(basis_path.relative_to(out_root))
            payload["target_dim"] = int(basis_payload["k"])
            payload["full_target_dim"] = int(payload["target_sum_td"].shape[-1])
            payload["target_layout"] = "framewise_pca"

        out_path = examples_dir / f"{int(source_manifest_index):06d}.pt"
        manifest_row = {
            "source_manifest_index": int(source_manifest_index),
            "source_id": str(row.get("source_id") or ""),
            "source_row_id": row.get("source_row_id"),
            "kit_name": row.get("kit_name"),
            "split": str(split),
            "beat_index": int(beat_index),
            "beat_index_end": int(beat_index_end),
            "duration_sec": float(duration_sec),
            "source_duration_sec": float(source_duration_sec),
            "bpm": float(effective_bpm),
            "source_bpm": float(source_bpm),
            "madmom_beat_fps": int(row.get("madmom_beat_fps", -1)),
            "madmom_correct": bool(row.get("madmom_correct", False)),
            "source_pt_rel": str(shard_rel),
            "source_row_in_shard": int(row_in_shard),
            "out_pt": str(out_path.relative_to(out_root)),
            "pca_k": int(pca_k),
            "target_num_frames": int(code_num_frames),
            "source_target_num_frames": int(source_code_num_frames),
            "target_dim": int(payload["target_dim"]),
            "full_target_dim": int(payload["full_target_dim"]),
            "target_layout": str(payload["target_layout"]),
            "grid_num_frames": int(grid_num_frames),
            "source_grid_num_frames": int(source_grid_num_frames),
            "grid_frame_rate": float(grid_frame_rate),
            "geometry_mode": str(geometry_mode),
            "bpm_source": str(args.bpm_source),
            "target_token_rate_hz": float(target_token_rate_hz),
        }
        if pca_accumulator is not None:
            pca_accumulator.update(payload["target_sum_td"])
        torch.save(payload, out_path)
        built_entries.append(
            {
                "out_path": out_path,
                "manifest_row": manifest_row,
            }
        )

    if int(pca_k) > 0 and split == "train":
        if pca_accumulator is None:
            raise RuntimeError("internal error: missing PCA accumulator for train split")
        mean_np, components_np, explained_np, k, num_train_frames = pca_accumulator.fit(pca_k=int(pca_k))
        if int(k) < int(pca_k):
            # DAC latents live in a low-rank decoder subspace. Preserve the
            # requested target width for architecture-matched experiments by
            # padding null components; reconstruction is unchanged.
            pad_k = int(pca_k) - int(k)
            components_np = np.concatenate(
                (
                    components_np,
                    np.zeros((int(pad_k), int(components_np.shape[1])), dtype=np.float32),
                ),
                axis=0,
            )
            explained_np = np.concatenate(
                (
                    explained_np,
                    np.zeros((int(pad_k),), dtype=np.float32),
                ),
                axis=0,
            )
            k = int(pca_k)
        components_np = fix_component_signs(components_np)
        explained_cum = float(np.cumsum(explained_np)[-1]) if int(explained_np.size) > 0 else 0.0
        basis_payload = {
            "mean": mean_np,
            "components": components_np,
            "explained_variance": explained_np,
            "k": int(k),
            "target_layout": "framewise_pca",
            "target_dim": int(k),
            "full_target_dim": int(raw_target_dim),
            "pca_target": float(explained_cum),
            "metadata": {
                **source_codec_metadata.to_dict(),
                "codebook_hash": codebook_hash,
                "source_manifest_hash": source_manifest_hash,
                "train_split_hash": split_manifest_hash,
                "requested_pca_k": int(pca_k),
                "source_cache_root": str(source_cache_root),
                "num_examples": int(len(built_entries)),
                "num_train_frames": int(num_train_frames),
                "geometry_mode": str(geometry_mode),
                "bpm_source": str(args.bpm_source),
                "target_token_rate_hz": float(target_token_rate_hz),
                "target_layout": "framewise_pca",
                "target_dim": int(k),
                "full_target_dim": int(raw_target_dim),
                "subspace_rank": (
                    int(pca_subspace_basis_t.shape[-1])
                    if pca_subspace_basis_t is not None
                    else int(raw_target_dim)
                ),
                "subspace_rank_tolerance": (
                    float(pca_subspace_diagnostics.get("rank_tolerance"))
                    if pca_subspace_diagnostics is not None
                    else None
                ),
                "subspace_matrix_shape": (
                    list(pca_subspace_diagnostics.get("matrix_shape", []))
                    if pca_subspace_diagnostics is not None
                    else []
                ),
                "subspace_singular_value_max": (
                    float(pca_subspace_diagnostics.get("singular_value_max"))
                    if pca_subspace_diagnostics is not None
                    else None
                ),
                "subspace_singular_value_min_retained": (
                    float(pca_subspace_diagnostics.get("singular_value_min_retained"))
                    if pca_subspace_diagnostics is not None
                    else None
                ),
                "subspace_singular_value_first_discarded": (
                    float(pca_subspace_diagnostics.get("singular_value_first_discarded"))
                    if pca_subspace_diagnostics is not None
                    else None
                ),
            },
        }
        if pca_subspace_basis_t is not None:
            basis_payload["subspace_basis_dq"] = pca_subspace_basis_t.detach().cpu().numpy()
        torch.save(basis_payload, basis_path)
        mean_t = torch.as_tensor(mean_np, dtype=torch.float32)
        components_t = torch.as_tensor(components_np, dtype=torch.float32)
        for entry in _progress(built_entries, desc=f"project_diffusion_cache[{split}]", total=len(built_entries)):
            payload = torch.load(entry["out_path"], map_location="cpu", weights_only=False)
            target_sum_td = torch.as_tensor(payload["target_sum_td"], dtype=torch.float32)
            payload["target_pc_tk"] = (target_sum_td - mean_t).matmul(components_t.t()).to(dtype=torch.float32)
            payload["target_pc_pool_k"] = payload["target_pc_tk"].sum(dim=0).to(dtype=torch.float32).contiguous()
            payload["pca_basis_path"] = str(basis_path.relative_to(out_root))
            payload["target_dim"] = int(k)
            payload["full_target_dim"] = int(target_sum_td.shape[-1])
            payload["target_layout"] = "framewise_pca"
            torch.save(payload, entry["out_path"])
            entry["manifest_row"]["target_dim"] = int(k)
            entry["manifest_row"]["full_target_dim"] = int(target_sum_td.shape[-1])
            entry["manifest_row"]["target_layout"] = "framewise_pca"

    manifest_rows = [entry["manifest_row"] for entry in built_entries]
    _write_jsonl(manifest_path, manifest_rows)
    _write_json(
        out_root / "config.json",
        {
            "source_cache_root": str(source_cache_root),
            "out_root": str(out_root),
            **source_codec_metadata.to_dict(),
            "requested_device": str(args.device),
            "resolved_device": str(resolved_device),
            "pca_k": int(pca_k),
            "target_layout": "framewise_pca" if int(pca_k) > 0 else "framewise_sum",
            "target_dim": int(basis_payload["k"]) if basis_payload is not None else int(raw_target_dim),
            "full_target_dim": int(raw_target_dim),
            "pca_basis_path": str(basis_path.relative_to(out_root)) if basis_path.exists() else "",
            "conditioning_layout": "seconds_family_state_grid",
            "geometry_mode": str(geometry_mode),
            "bpm_source": str(args.bpm_source),
            "target_token_rate_hz": float(target_token_rate_hz),
        },
    )
    _write_json(
        summary_path,
        {
            "split": str(split),
            "num_examples": int(len(built_entries)),
            "source_manifest_hash": source_manifest_hash,
            "split_manifest_hash": split_manifest_hash,
            "codebook_hash": codebook_hash,
            "pca_enabled": bool(int(pca_k) > 0),
            "pca_basis_path": str(basis_path.relative_to(out_root)) if basis_path.exists() else "",
            "target_layout": "framewise_pca" if int(pca_k) > 0 else "framewise_sum",
            "target_dim": int(basis_payload["k"]) if basis_payload is not None else int(raw_target_dim),
            "full_target_dim": int(raw_target_dim),
            "conditioning_layout": "seconds_family_state_grid",
            "geometry_mode": str(geometry_mode),
            "bpm_source": str(args.bpm_source),
            "target_token_rate_hz": float(target_token_rate_hz),
        },
    )


if __name__ == "__main__":
    main()
