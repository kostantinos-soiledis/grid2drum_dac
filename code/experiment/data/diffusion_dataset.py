from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from data.encodec_utils import normalize_target_payload, resolve_target_dim_from_cache_config


@dataclass(frozen=True)
class DiffusionExample:
    example_path: Path
    source_id: str
    source_manifest_index: int
    source_pt_rel: str
    source_row_in_shard: int
    beat_index: int
    beat_index_end: int
    split: str
    kit_name: str
    source_row_id: str
    conditioning_mode: str
    class_names: tuple[str, ...]
    class_id_vocab_sizes: tuple[int, ...]
    feature_row_names: tuple[str, ...]
    grid_ft: torch.Tensor
    grid_ids_ft: torch.Tensor
    family_onsets_ft: torch.Tensor
    family_onset_count_ft: torch.Tensor
    grid_times_sec_t: torch.Tensor
    token_times_sec_t: torch.Tensor
    beat_boundaries_sec_rel: torch.Tensor
    grid_num_frames: int
    grid_frame_rate: float
    bpm: float
    duration_sec: float
    target_sum_td: torch.Tensor
    target_full_sum_td: torch.Tensor
    target_dim: int
    target_full_dim: int
    target_layout: str
    pca_basis_path: str
    target_num_frames: int
    source_codes_ct: torch.Tensor
    target_sum_pool_d: torch.Tensor

    @property
    def target_sum_t128(self) -> torch.Tensor:
        return self.target_sum_td

    @property
    def target_sum_pool_128(self) -> torch.Tensor:
        return self.target_sum_pool_d


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = str(line).strip()
            if not text:
                continue
            rows.append(dict(json.loads(text)))
    return rows


def _require_payload_keys(payload: dict[str, Any], keys: Sequence[str], *, example_path: Path) -> None:
    missing = [str(key) for key in list(keys) if str(key) not in payload]
    if missing:
        raise RuntimeError(
            f"cache example {example_path} is missing required seconds-conditioning fields {missing}. "
            "Rebuild the cache with scripts/build_diffusion_cache.py."
        )


class DiffusionConditioningDataset(Dataset[DiffusionExample]):
    def __init__(
        self,
        cache_root: str | Path,
        *,
        split: str = "train",
        max_items: int = 0,
        conditioning_mode: str = "seconds",
    ) -> None:
        super().__init__()
        self.cache_root = Path(cache_root).resolve()
        self.split = str(split).strip().lower()
        self.conditioning_mode = str(conditioning_mode).strip().lower()
        if self.conditioning_mode not in {"seconds", "seconds_frontend"}:
            raise ValueError(
                f"unsupported conditioning_mode={conditioning_mode!r}; expected 'seconds' or 'seconds_frontend'"
            )
        self.target_dim = int(resolve_target_dim_from_cache_config(self.cache_root))

        manifest_path = self.cache_root / "manifests" / f"{self.split}.jsonl"
        self.rows = _load_jsonl(manifest_path)
        if int(max_items) > 0:
            self.rows = self.rows[: int(max_items)]
        if not self.rows:
            raise RuntimeError(f"no rows found for split={self.split!r} under {self.cache_root}")

    def __len__(self) -> int:
        return int(len(self.rows))

    def __getitem__(self, index: int) -> DiffusionExample:
        return self.example_from_row(self.rows[int(index)])

    def example_from_row(self, row: Mapping[str, Any]) -> DiffusionExample:
        row = dict(row)
        example_path = (self.cache_root / str(row["out_pt"])).resolve()
        payload = dict(torch.load(example_path, map_location="cpu", weights_only=False))
        _require_payload_keys(
            payload,
            (
                "grid_ft",
                "grid_ids_ft",
                "family_onsets_ft",
                "family_onset_count_ft",
                "grid_times_sec_t",
                "token_times_sec_t",
                "beat_boundaries_sec_rel",
                "grid_num_frames",
                "grid_frame_rate",
                "bpm",
                "duration_sec",
                "target_num_frames",
                "source_codes_ct",
            ),
            example_path=example_path,
        )

        target_sum_td, target_sum_pool_d, target_dim = normalize_target_payload(
            payload,
            fallback_target_dim=int(self.target_dim),
        )
        raw_target_sum_td = torch.as_tensor(
            payload.get("target_sum_td", payload.get("target_sum_t128")),
            dtype=torch.float32,
        ).contiguous()
        raw_target_sum_pool_d = torch.as_tensor(
            payload.get("target_sum_pool_d", payload.get("target_sum_pool_128")),
            dtype=torch.float32,
        ).contiguous()
        target_num_frames = int(payload["target_num_frames"])
        if int(target_sum_td.shape[0]) != int(target_num_frames):
            raise RuntimeError(
                f"target_num_frames mismatch: target_sum_td has {target_sum_td.shape[0]} frames vs {target_num_frames}"
            )
        if int(raw_target_sum_td.shape[0]) != int(target_num_frames):
            raise RuntimeError(
                f"raw target_num_frames mismatch: target_sum_td has {raw_target_sum_td.shape[0]} frames vs {target_num_frames}"
            )

        grid_ft = torch.as_tensor(payload["grid_ft"], dtype=torch.float32).contiguous()
        grid_ids_ft = torch.as_tensor(payload["grid_ids_ft"], dtype=torch.long).contiguous()
        family_onsets_ft = torch.as_tensor(payload["family_onsets_ft"], dtype=torch.bool).contiguous()
        family_onset_count_ft = torch.as_tensor(payload["family_onset_count_ft"], dtype=torch.uint8).contiguous()
        grid_times_sec_t = torch.as_tensor(payload["grid_times_sec_t"], dtype=torch.float32).contiguous()
        token_times_sec_t = torch.as_tensor(payload["token_times_sec_t"], dtype=torch.float32).contiguous()
        beat_boundaries_sec_rel = torch.as_tensor(payload["beat_boundaries_sec_rel"], dtype=torch.float32).contiguous()
        source_codes_ct = torch.as_tensor(payload["source_codes_ct"], dtype=torch.int16).contiguous()

        grid_num_frames = int(payload["grid_num_frames"])
        grid_frame_rate = float(payload["grid_frame_rate"])
        bpm = float(payload["bpm"])
        duration_sec = float(payload["duration_sec"])
        if int(grid_ft.dim()) != 2:
            raise RuntimeError(f"expected grid_ft [F,Tg], got {tuple(grid_ft.shape)}")
        if int(grid_ids_ft.dim()) != 2:
            raise RuntimeError(f"expected grid_ids_ft [C,Tg], got {tuple(grid_ids_ft.shape)}")
        if int(family_onsets_ft.dim()) != 2 or int(family_onset_count_ft.dim()) != 2:
            raise RuntimeError(
                f"expected family onset tensors [C,Tg], got {tuple(family_onsets_ft.shape)} / {tuple(family_onset_count_ft.shape)}"
            )
        if int(grid_ft.shape[-1]) != int(grid_num_frames):
            raise RuntimeError(
                f"grid_num_frames mismatch: grid_ft has {grid_ft.shape[-1]} frames vs {grid_num_frames}"
            )
        if int(grid_ids_ft.shape[-1]) != int(grid_num_frames):
            raise RuntimeError(
                f"grid_ids_ft mismatch: grid_ids_ft has {grid_ids_ft.shape[-1]} frames vs {grid_num_frames}"
            )
        if int(grid_times_sec_t.shape[0]) != int(grid_num_frames):
            raise RuntimeError(
                f"grid_times_sec_t mismatch: grid_times_sec_t has {grid_times_sec_t.shape[0]} frames vs {grid_num_frames}"
            )
        if int(token_times_sec_t.shape[0]) != int(target_num_frames):
            raise RuntimeError(
                f"token_times_sec_t mismatch: token_times_sec_t has {token_times_sec_t.shape[0]} frames vs {target_num_frames}"
            )

        class_names = tuple(str(x) for x in list(payload.get("class_names") or []))
        class_id_vocab_sizes = tuple(int(x) for x in list(payload.get("class_id_vocab_sizes") or []))
        feature_row_names = tuple(str(x) for x in list(payload.get("feature_row_names") or []))

        return DiffusionExample(
            example_path=example_path,
            source_id=str(payload.get("source_id") or row.get("source_id") or ""),
            source_manifest_index=int(payload.get("source_manifest_index", row.get("source_manifest_index", -1))),
            source_pt_rel=str(payload.get("source_pt_rel") or row.get("source_pt_rel") or ""),
            source_row_in_shard=int(payload.get("source_row_in_shard", row.get("source_row_in_shard", -1))),
            beat_index=int(payload.get("beat_index", row.get("beat_index", 0))),
            beat_index_end=int(payload.get("beat_index_end", row.get("beat_index_end", 0))),
            split=str(payload.get("split") or row.get("split") or self.split),
            kit_name=str(payload.get("kit_name") or row.get("kit_name") or ""),
            source_row_id=str(payload.get("source_row_id") or row.get("source_row_id") or ""),
            conditioning_mode=str(payload.get("conditioning_mode") or "midi_family_state_onset_ids"),
            class_names=class_names,
            class_id_vocab_sizes=class_id_vocab_sizes,
            feature_row_names=feature_row_names,
            grid_ft=grid_ft,
            grid_ids_ft=grid_ids_ft,
            family_onsets_ft=family_onsets_ft,
            family_onset_count_ft=family_onset_count_ft,
            grid_times_sec_t=grid_times_sec_t,
            token_times_sec_t=token_times_sec_t,
            beat_boundaries_sec_rel=beat_boundaries_sec_rel,
            grid_num_frames=grid_num_frames,
            grid_frame_rate=grid_frame_rate,
            bpm=bpm,
            duration_sec=duration_sec,
            target_sum_td=target_sum_td,
            target_full_sum_td=raw_target_sum_td,
            target_dim=int(target_dim),
            target_full_dim=int(raw_target_sum_td.shape[-1]),
            target_layout=str(payload.get("target_layout") or "framewise_sum"),
            pca_basis_path=str(payload.get("pca_basis_path") or ""),
            target_num_frames=target_num_frames,
            source_codes_ct=source_codes_ct,
            target_sum_pool_d=raw_target_sum_pool_d,
        )


def collate_diffusion_examples(items: Sequence[DiffusionExample]) -> dict[str, Any]:
    if not items:
        raise ValueError("expected at least one DiffusionExample")

    batch_size = int(len(items))
    grid_len = int(max(item.grid_num_frames for item in items))
    grid_dim = int(items[0].grid_ft.shape[0])
    class_dim = int(items[0].grid_ids_ft.shape[0])
    target_len = int(max(item.target_num_frames for item in items))
    codebooks = int(items[0].source_codes_ct.shape[0])
    beat_boundary_len = int(max(item.beat_boundaries_sec_rel.shape[0] for item in items))
    target_dim = int(items[0].target_dim)
    target_full_dim = int(items[0].target_full_dim)
    if any(int(item.target_dim) != int(target_dim) for item in items):
        raise ValueError("all items in a batch must share target_dim")
    if any(int(item.target_full_dim) != int(target_full_dim) for item in items):
        raise ValueError("all items in a batch must share target_full_dim")
    target_layout = str(items[0].target_layout)
    if any(str(item.target_layout) != str(target_layout) for item in items):
        raise ValueError("all items in a batch must share target_layout")
    pca_basis_path = str(items[0].pca_basis_path)
    if any(str(item.pca_basis_path) != str(pca_basis_path) for item in items):
        raise ValueError("all items in a batch must share pca_basis_path")

    grid_bft = torch.zeros((batch_size, grid_dim, grid_len), dtype=torch.float32)
    grid_ids_bct = torch.full((batch_size, class_dim, grid_len), -1, dtype=torch.long)
    family_onsets_bft = torch.zeros((batch_size, class_dim, grid_len), dtype=torch.bool)
    family_onset_count_bft = torch.zeros((batch_size, class_dim, grid_len), dtype=torch.uint8)
    grid_valid_mask_bt = torch.zeros((batch_size, grid_len), dtype=torch.bool)
    grid_times_sec_bt = torch.zeros((batch_size, grid_len), dtype=torch.float32)
    token_times_sec_bt = torch.zeros((batch_size, target_len), dtype=torch.float32)
    beat_boundaries_sec_bk = torch.zeros((batch_size, beat_boundary_len), dtype=torch.float32)
    beat_boundaries_valid_mask_bk = torch.zeros((batch_size, beat_boundary_len), dtype=torch.bool)
    target_btd = torch.zeros((batch_size, target_len, target_dim), dtype=torch.float32)
    target_sum_btd = torch.zeros((batch_size, target_len, target_full_dim), dtype=torch.float32)
    target_valid_mask_bt = torch.zeros((batch_size, target_len), dtype=torch.bool)
    source_codes_bct = torch.full((batch_size, codebooks, target_len), -1, dtype=torch.int16)
    target_sum_pool_bd = torch.zeros((batch_size, target_full_dim), dtype=torch.float32)
    bpm_b = torch.zeros((batch_size,), dtype=torch.float32)
    duration_sec_b = torch.zeros((batch_size,), dtype=torch.float32)
    grid_frame_rate_b = torch.zeros((batch_size,), dtype=torch.float32)

    for row_idx, item in enumerate(items):
        gf = int(item.grid_num_frames)
        tf = int(item.target_num_frames)
        bk = int(item.beat_boundaries_sec_rel.shape[0])
        grid_bft[int(row_idx), :, : int(gf)] = item.grid_ft[:, : int(gf)]
        grid_ids_bct[int(row_idx), :, : int(gf)] = item.grid_ids_ft[:, : int(gf)]
        family_onsets_bft[int(row_idx), :, : int(gf)] = item.family_onsets_ft[:, : int(gf)]
        family_onset_count_bft[int(row_idx), :, : int(gf)] = item.family_onset_count_ft[:, : int(gf)]
        grid_valid_mask_bt[int(row_idx), : int(gf)] = True
        grid_times_sec_bt[int(row_idx), : int(gf)] = item.grid_times_sec_t[: int(gf)]
        token_times_sec_bt[int(row_idx), : int(tf)] = item.token_times_sec_t[: int(tf)]
        beat_boundaries_sec_bk[int(row_idx), : int(bk)] = item.beat_boundaries_sec_rel[: int(bk)]
        beat_boundaries_valid_mask_bk[int(row_idx), : int(bk)] = True
        target_btd[int(row_idx), : int(tf), :] = item.target_sum_td[: int(tf)]
        target_sum_btd[int(row_idx), : int(tf), :] = item.target_full_sum_td[: int(tf)]
        target_valid_mask_bt[int(row_idx), : int(tf)] = True
        source_codes_bct[int(row_idx), :, : int(tf)] = item.source_codes_ct[:, : int(tf)]
        target_sum_pool_bd[int(row_idx), :] = item.target_sum_pool_d
        bpm_b[int(row_idx)] = float(item.bpm)
        duration_sec_b[int(row_idx)] = float(item.duration_sec)
        grid_frame_rate_b[int(row_idx)] = float(item.grid_frame_rate)

    batch = {
        "conditioning_mode": str(items[0].conditioning_mode),
        "class_names": list(items[0].class_names),
        "class_id_vocab_sizes": [int(x) for x in list(items[0].class_id_vocab_sizes)],
        "feature_row_names": list(items[0].feature_row_names),
        "grid": grid_bft.contiguous(),
        "grid_ids": grid_ids_bct.contiguous(),
        "family_onsets_bft": family_onsets_bft.contiguous(),
        "family_onset_count_bft": family_onset_count_bft.contiguous(),
        "grid_valid_mask": grid_valid_mask_bt.contiguous(),
        "grid_times_sec": grid_times_sec_bt.contiguous(),
        "token_times_sec": token_times_sec_bt.contiguous(),
        "beat_boundaries_sec": beat_boundaries_sec_bk.contiguous(),
        "beat_boundaries_valid_mask": beat_boundaries_valid_mask_bk.contiguous(),
        "bpm": bpm_b.contiguous(),
        "duration_sec": duration_sec_b.contiguous(),
        "grid_frame_rate_b": grid_frame_rate_b.contiguous(),
        "grid_num_frames_b": torch.tensor([int(item.grid_num_frames) for item in items], dtype=torch.long),
        "target_layout": str(target_layout),
        "target_dim": int(target_dim),
        "target_full_dim": int(target_full_dim),
        "target_btd": target_btd.contiguous(),
        "target_sum_btd": target_sum_btd.contiguous(),
        "target_valid_mask_bt": target_valid_mask_bt.contiguous(),
        "source_codes_bct": source_codes_bct.contiguous(),
        "target_sum_pool_bd": target_sum_pool_bd.contiguous(),
        "target_pca_basis_path": str(pca_basis_path),
        "target_num_frames_b": torch.tensor([int(item.target_num_frames) for item in items], dtype=torch.long),
        "source_manifest_index_b": torch.tensor([int(item.source_manifest_index) for item in items], dtype=torch.long),
        "source_row_in_shard_b": torch.tensor([int(item.source_row_in_shard) for item in items], dtype=torch.long),
        "beat_index_b": torch.tensor([int(item.beat_index) for item in items], dtype=torch.long),
        "beat_index_end_b": torch.tensor([int(item.beat_index_end) for item in items], dtype=torch.long),
        "source_id": [str(item.source_id) for item in items],
        "source_pt_rel": [str(item.source_pt_rel) for item in items],
        "split": [str(item.split) for item in items],
        "kit_name": [str(item.kit_name) for item in items],
        "source_row_id": [str(item.source_row_id) for item in items],
        "example_path": [str(item.example_path) for item in items],
    }
    return batch


@torch.no_grad()
def estimate_target_normalization(
    dataloader: DataLoader,
    *,
    device: str | torch.device = "cpu",
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    resolved_device = torch.device(device)
    target_sum_d: torch.Tensor | None = None
    target_sq_sum_d: torch.Tensor | None = None
    total_frames = 0

    for batch in dataloader:
        target_btd = torch.as_tensor(batch["target_btd"], dtype=torch.float32, device=resolved_device)
        target_valid_mask_bt = torch.as_tensor(
            batch["target_valid_mask_bt"],
            dtype=torch.bool,
            device=resolved_device,
        )
        if int(target_btd.dim()) != 3:
            raise ValueError(f"expected target_btd [B,T,D], got {tuple(target_btd.shape)}")
        if tuple(target_valid_mask_bt.shape) != tuple(target_btd.shape[:2]):
            raise ValueError(
                "target_valid_mask_bt must match target_btd batch/time dimensions, got "
                f"{tuple(target_valid_mask_bt.shape)} vs {tuple(target_btd.shape[:2])}"
            )
        if target_sum_d is None:
            target_sum_d = torch.zeros((int(target_btd.shape[-1]),), dtype=torch.float32, device=resolved_device)
            target_sq_sum_d = torch.zeros_like(target_sum_d)

        valid_btd = target_btd * target_valid_mask_bt.unsqueeze(-1).to(dtype=target_btd.dtype)
        target_sum_d += valid_btd.sum(dim=(0, 1))
        target_sq_sum_d += valid_btd.square().sum(dim=(0, 1))
        total_frames += int(target_valid_mask_bt.sum().item())

    if target_sum_d is None or target_sq_sum_d is None or int(total_frames) <= 0:
        raise RuntimeError("could not estimate target normalization from an empty dataloader")

    mean_d = target_sum_d / float(total_frames)
    var_d = (target_sq_sum_d / float(total_frames)) - mean_d.square()
    std_d = var_d.clamp_min(float(eps)).sqrt()
    return mean_d.contiguous(), std_d.contiguous()


def build_diffusion_dataloader(
    cache_root: str | Path,
    *,
    split: str = "train",
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 0,
    max_items: int = 0,
    conditioning_mode: str = "seconds",
    pin_memory: bool = False,
    persistent_workers: bool = False,
    multiprocessing_context: str | None = None,
) -> DataLoader:
    dataset = DiffusionConditioningDataset(
        cache_root,
        split=split,
        max_items=max_items,
        conditioning_mode=conditioning_mode,
    )
    loader_kwargs: dict[str, Any] = {}
    if int(num_workers) > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        if multiprocessing_context is not None and str(multiprocessing_context).strip():
            loader_kwargs["multiprocessing_context"] = str(multiprocessing_context).strip()
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        collate_fn=collate_diffusion_examples,
        **loader_kwargs,
    )
