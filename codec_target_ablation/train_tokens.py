#!/usr/bin/env python3
"""Discrete-token arm: the regression arm's MGE and Transformer predicting DAC's nine RVQ codes.

The model is DirectPCASequenceRegressor (code/experiment/standalone_direct_pca_regressor.py) with a
9 x 1024 output head, read as per-codebook logits and trained with cross-entropy against each
excerpt's DAC codes (source_codes_ct). Cache, frontend, width, depth and optimizer follow
runs/final/regression_6layer; best_tokens.pt keeps the epoch with the lowest validation cross-entropy.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1] / "code" / "experiment"
sys.path.insert(0, str(EXPERIMENT_ROOT))

from standalone_direct_pca_regressor import (  # noqa: E402
    DEFAULT_CLASS_LOCAL_DIM,
    DEFAULT_FRONTEND_EMBED_DIM,
    DEFAULT_FRONTEND_OUTPUT_KIND,
    DEFAULT_FRONTEND_PADDING_MODE,
    DEFAULT_FRONTEND_VARIANT,
    DEFAULT_TARGET_TOKEN_RATE_HZ,
    DirectCacheDataset,
    DirectPCASequenceRegressor,
    DirectRegressorConfig,
    append_jsonl,
    batch_to_device,
    build_frontend_cfg_from_batch,
    maybe_read_json,
    read_jsonl,
    resolve_device,
    torch_load,
    write_history_csv,
    write_json,
    write_jsonl,
)

IGNORE_INDEX = -100


class TokenCacheDataset(DirectCacheDataset):
    """Read the token arm's inputs and labels together from one cache file."""

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[int(index)]
        example_path = self.cache_root / str(row["out_pt"])
        payload = torch_load(example_path)
        codes = torch.as_tensor(payload["source_codes_ct"], dtype=torch.long)
        token_times = torch.as_tensor(payload["token_times_sec_t"], dtype=torch.float32)
        if codes.ndim != 2 or codes.shape[1] != token_times.numel():
            raise RuntimeError(f"code/target frame mismatch in {example_path}")
        return {
            "grid": torch.as_tensor(payload["grid_ft"], dtype=torch.float32),
            "grid_ids": torch.as_tensor(payload["grid_ids_ft"], dtype=torch.long),
            "grid_times_sec": torch.as_tensor(payload["grid_times_sec_t"], dtype=torch.float32),
            "token_times_sec": token_times,
            "target_codes_bct": codes,
            "class_names": list(payload["class_names"]),
            "class_id_vocab_sizes": list(payload["class_id_vocab_sizes"]),
            "feature_row_names": list(payload["feature_row_names"]),
        }


def collate_token_examples(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("expected non-empty batch")
    batch_size = len(items)
    grid_len = max(int(item["grid"].shape[-1]) for item in items)
    token_len = max(int(item["token_times_sec"].numel()) for item in items)
    first = items[0]
    grid = torch.zeros((batch_size, first["grid"].shape[0], grid_len), dtype=torch.float32)
    grid_ids = torch.full((batch_size, first["grid_ids"].shape[0], grid_len), -1, dtype=torch.long)
    grid_mask = torch.zeros((batch_size, grid_len), dtype=torch.bool)
    grid_times = torch.zeros((batch_size, grid_len), dtype=torch.float32)
    token_times = torch.zeros((batch_size, token_len), dtype=torch.float32)
    token_mask = torch.zeros((batch_size, token_len), dtype=torch.bool)
    target_codes = torch.full(
        (batch_size, first["target_codes_bct"].shape[0], token_len), IGNORE_INDEX, dtype=torch.long
    )
    for row, item in enumerate(items):
        gf = int(item["grid"].shape[-1])
        tf = int(item["token_times_sec"].numel())
        grid[row, :, :gf] = item["grid"]
        grid_ids[row, :, :gf] = item["grid_ids"]
        grid_mask[row, :gf] = True
        grid_times[row, :gf] = item["grid_times_sec"]
        token_times[row, :tf] = item["token_times_sec"]
        token_mask[row, :tf] = True
        target_codes[row, :, :tf] = item["target_codes_bct"]
    return {
        "grid": grid,
        "grid_ids": grid_ids,
        "grid_valid_mask": grid_mask,
        "grid_times_sec": grid_times,
        "token_times_sec": token_times,
        "target_valid_mask_bt": token_mask,
        "target_codes_bct": target_codes,
        "class_names": first["class_names"],
        "class_id_vocab_sizes": first["class_id_vocab_sizes"],
        "feature_row_names": first["feature_row_names"],
    }


def build_token_loader(cache_root: Path, *, split: str, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool) -> DataLoader:
    return DataLoader(
        TokenCacheDataset(cache_root, split=split),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        collate_fn=collate_token_examples,
        persistent_workers=bool(num_workers > 0),
    )


def code_logits(model: DirectPCASequenceRegressor, batch: Mapping[str, Any], *, num_codebooks: int, codebook_size: int) -> torch.Tensor:
    """Per-codebook logits [B, codebook_size, C, T], the layout F.cross_entropy expects."""
    out_btx = model(
        grid=batch["grid"],
        grid_ids=batch["grid_ids"],
        grid_times_sec=batch["grid_times_sec"],
        token_times_sec=batch["token_times_sec"],
        target_valid_mask_bt=batch["target_valid_mask_bt"],
        grid_valid_mask_bt=batch["grid_valid_mask"],
    )
    batch_size, num_frames, _ = out_btx.shape
    return out_btx.view(batch_size, num_frames, num_codebooks, codebook_size).permute(0, 3, 2, 1)


def run_epoch(
    model: DirectPCASequenceRegressor,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    num_codebooks: int,
    codebook_size: int,
    amp: bool,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    total_ce = torch.zeros((), device=device)
    total_correct = torch.zeros((), device=device, dtype=torch.long)
    total_codes = torch.zeros((), device=device, dtype=torch.long)
    for batch_cpu in loader:
        batch = batch_to_device(batch_cpu, device)
        codes = batch["target_codes_bct"]
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                logits = code_logits(model, batch, num_codebooks=num_codebooks, codebook_size=codebook_size)
                loss = F.cross_entropy(logits, codes, ignore_index=IGNORE_INDEX)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        valid = codes.ne(IGNORE_INDEX)
        num_codes = valid.sum()
        total_ce += loss.detach() * num_codes
        total_correct += (logits.detach().argmax(dim=1).eq(codes) & valid).sum()
        total_codes += num_codes
    denom = total_codes.clamp_min(1)
    return {"ce": (total_ce / denom).item(), "acc": (total_correct / denom).item()}


def save_checkpoint(path: Path, *, model, optimizer, cfg: DirectRegressorConfig, epoch: int, best_val_ce: float, run_config: Mapping[str, Any]) -> None:
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "config": asdict(cfg),
        "num_codebooks": int(run_config["num_codebooks"]),
        "codebook_size": int(run_config["codebook_size"]),
        "best_val_ce": float(best_val_ce),
        "run_config": dict(run_config),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    temp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, temp_path)
    os.replace(temp_path, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-root", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", help="use BF16 autocast on CUDA")
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--frontend-radii", type=str, default="0,22,41,55")
    parser.add_argument("--frontend-primary-radius", type=int, default=22)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", action="store_true", help="continue from --out-dir/last.pt")
    group.add_argument("--overwrite", action="store_true", help="delete --out-dir before training")
    return parser.parse_args()


def frontend_args(args: argparse.Namespace) -> argparse.Namespace:
    """The regressor's frontend flags at their defaults, which runs/final/regression_6layer used."""
    return argparse.Namespace(
        frontend_radii=args.frontend_radii,
        frontend_primary_radius=args.frontend_primary_radius,
        frontend_variant=DEFAULT_FRONTEND_VARIANT,
        frontend_embed_dim=DEFAULT_FRONTEND_EMBED_DIM,
        frontend_output_kind=DEFAULT_FRONTEND_OUTPUT_KIND,
        frontend_padding_mode=DEFAULT_FRONTEND_PADDING_MODE,
        frontend_step_seconds=0.0,
        frontend_chunk_size=0,
        frontend_class_local_fusion=False,
        frontend_class_local_dim=DEFAULT_CLASS_LOCAL_DIM,
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    cache_root = Path(args.cache_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    last_path, best_path = out_dir / "last.pt", out_dir / "best_tokens.pt"
    history_jsonl, history_csv = out_dir / "history.jsonl", out_dir / "history.csv"
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    if args.resume and not last_path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {last_path}")
    if not args.resume and last_path.exists():
        raise RuntimeError(f"{out_dir} already has a run; pass --resume or --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(str(args.device))
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    amp = bool(args.amp and device.type == "cuda")
    if amp and not torch.cuda.is_bf16_supported():
        raise RuntimeError("--amp requires a CUDA GPU with BF16 support")
    pin_memory = device.type == "cuda"
    train_loader = build_token_loader(cache_root, split="train", batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = build_token_loader(cache_root, split="validation", batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)

    cache_config = maybe_read_json(cache_root / "config.json") or {}
    num_codebooks = int(cache_config["codec_num_codebooks"])
    codebook_size = int(cache_config["codec_codebook_size"])
    sample_batch = next(iter(train_loader))
    cfg = DirectRegressorConfig(
        x_dim=num_codebooks * codebook_size,
        frontend_cfg=build_frontend_cfg_from_batch(sample_batch, frontend_args(args)),
        positional_rate_hz=float(cache_config.get("target_token_rate_hz", DEFAULT_TARGET_TOKEN_RATE_HZ)),
        d_model=int(args.d_model),
        num_layers=int(args.num_layers),
        num_heads=int(args.num_heads),
    )
    model = DirectPCASequenceRegressor(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay), fused=device.type == "cuda"
    )

    start_epoch, best_val_ce, history = 0, float("inf"), []
    if args.resume:
        payload = dict(torch_load(last_path))
        if dict(payload["config"]) != asdict(cfg):
            raise ValueError("resume checkpoint configuration does not match the requested configuration")
        previous_precision = payload.get("run_config", {}).get("precision", "float32")
        if previous_precision != ("bfloat16" if amp else "float32"):
            raise ValueError(f"resume precision mismatch: checkpoint={previous_precision}, requested={'bfloat16' if amp else 'float32'}")
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_epoch = int(payload["epoch"]) + 1
        best_val_ce = float(payload["best_val_ce"])
        history = [row for row in read_jsonl(history_jsonl) if int(row["epoch"]) < start_epoch]
    write_jsonl(history_jsonl, history)

    run_config = {
        "cache_root": str(cache_root),
        "out_dir": str(out_dir),
        "device": str(device),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "precision": "bfloat16" if amp else "float32",
        "seed": int(args.seed),
        "num_codebooks": num_codebooks,
        "codebook_size": codebook_size,
        "model_cfg": asdict(cfg),
        "num_parameters": int(sum(int(p.numel()) for p in model.parameters())),
    }
    write_json(out_dir / "run_config.json", run_config)
    print(f"token arm: params={run_config['num_parameters']} start_epoch={start_epoch} device={device}")

    for epoch in range(start_epoch, int(args.epochs)):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, optimizer=optimizer, device=device, num_codebooks=num_codebooks, codebook_size=codebook_size, amp=amp)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, optimizer=None, device=device, num_codebooks=num_codebooks, codebook_size=codebook_size, amp=amp)
        improved = val_metrics["ce"] < best_val_ce
        best_val_ce = min(best_val_ce, val_metrics["ce"])
        row = {
            "epoch": epoch,
            "train_ce": train_metrics["ce"],
            "train_acc": train_metrics["acc"],
            "val_ce": val_metrics["ce"],
            "val_acc": val_metrics["acc"],
            "best_val_ce": best_val_ce,
            "elapsed_sec": time.time() - t0,
            "checkpoint_improved": improved,
        }
        append_jsonl(history_jsonl, row)
        history.append(row)
        write_history_csv(history_csv, history)
        checkpoint = dict(model=model, optimizer=optimizer, cfg=cfg, epoch=epoch, best_val_ce=best_val_ce, run_config=run_config)
        save_checkpoint(last_path, **checkpoint)
        if improved:
            save_checkpoint(best_path, **{**checkpoint, "optimizer": None})
        print(
            f"epoch={epoch:03d} train_ce={row['train_ce']:.4f} val_ce={row['val_ce']:.4f} "
            f"val_acc={row['val_acc']:.4f} best={best_val_ce:.4f} elapsed={row['elapsed_sec']:.0f}s"
        )


if __name__ == "__main__":
    main()
