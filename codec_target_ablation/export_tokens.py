#!/usr/bin/env python3
"""Decode the token arm's argmax codes into split WAVs in the layout run_diffusion_acoustic_eval.py reads.

Mirrors code/experiment/scripts/export_direct_pca_predictions.py; only the model output changes, from
PCA coordinates to per-codebook logits whose argmax codes the frozen DAC decodes (padded frames are -1,
which the decoder treats as empty).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import torch

from train_tokens import code_logits  # importing train_tokens also puts code/experiment on sys.path

from data.audio_codec_utils import decode_codes_to_audio_b1t
from data.encodec_utils import load_audio_codec_model, resolve_codec_metadata_from_cache_config, resolve_device
from io_utils import save_audio, write_json, write_jsonl
from scripts.dac_export_utils import clip_file_name, samples_per_latent_frame
from standalone_direct_pca_regressor import DirectPCASequenceRegressor, DirectRegressorConfig, batch_to_device, build_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--cache-root", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output directory already exists: {out_dir}")
        shutil.rmtree(out_dir)
    (out_dir / "wavs").mkdir(parents=True)

    resolved_device = resolve_device(str(args.device))
    device = torch.device(resolved_device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    checkpoint_path = run_dir / "best_tokens.pt"
    payload = dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False))
    model = DirectPCASequenceRegressor(DirectRegressorConfig(**payload["config"])).to(device).eval()
    model.load_state_dict(payload["model_state_dict"])
    num_codebooks, codebook_size = int(payload["num_codebooks"]), int(payload["codebook_size"])
    amp = payload.get("run_config", {}).get("precision") == "bfloat16" and device.type == "cuda"

    codec_model, _, codec_metadata = load_audio_codec_model(
        device=resolved_device, metadata=resolve_codec_metadata_from_cache_config(cache_root)
    )
    sample_rate = int(codec_metadata.codec_sample_rate)
    loader = build_loader(
        cache_root, split=args.split, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, max_items=args.max_items, pin_memory=False,
    )

    manifest_rows: list[dict[str, Any]] = []
    for batch_cpu in loader:
        batch = batch_to_device(batch_cpu, device)
        mask = batch["target_valid_mask_bt"]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            logits = code_logits(model, batch, num_codebooks=num_codebooks, codebook_size=codebook_size)
        codes = logits.argmax(dim=1).masked_fill(~mask.unsqueeze(1), -1)
        decoded = decode_codes_to_audio_b1t(codec_model, codes, device=device, metadata=codec_metadata)
        spf = samples_per_latent_frame(int(decoded.shape[-1]), int(mask.shape[1]))
        for row in range(int(decoded.shape[0])):
            dataset_index = len(manifest_rows)
            source_id = str(batch["source_id"][row])
            beat_index = int(batch["beat_index_b"][row])
            target_frames = int(batch["target_num_frames_b"][row])
            num_samples = min(int(decoded.shape[-1]), target_frames * spf)
            wav_rel = Path("wavs") / clip_file_name(dataset_index, source_id, beat_index)
            save_audio(out_dir / wav_rel, decoded[row : row + 1, :, :num_samples].cpu(), sample_rate=sample_rate)
            manifest_rows.append({
                "dataset_index": dataset_index,
                "source_id": source_id,
                "source_manifest_index": int(batch["source_manifest_index_b"][row]),
                "beat_index": beat_index,
                "split": str(args.split),
                "conditioning_ablation": "none",
                "sample_rate": sample_rate,
                "num_samples": num_samples,
                "duration_sec": num_samples / sample_rate,
                "target_num_frames": target_frames,
                "wav": str(wav_rel),
            })

    write_jsonl(out_dir / "manifest.jsonl", manifest_rows)
    write_json(out_dir / "summary.json", {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(payload["epoch"]),
        "best_val_ce": float(payload["best_val_ce"]),
        "num_parameters": int(sum(int(p.numel()) for p in model.parameters())),
        "num_examples": len(manifest_rows),
        "split": str(args.split),
        "decoding": "argmax",
    })
    print(f"exported {len(manifest_rows)} token-arm predictions to {out_dir}")


if __name__ == "__main__":
    main()
