#!/usr/bin/env python3
"""Derive a pre-snap DAC target cache from an existing framewise PCA diffusion cache.

Every example keeps its conditioning, codes, `target_sum_td` (so reference audio is
unchanged); only `target_pc_tk` is replaced by DAC `projected_latents` [T, 72], i.e.
the per-stage `in_proj` outputs before codebook snapping. Beat windows are re-encoded
from the source audio using the exact sample ranges stored in the source cache.

This representation is not PCA. It is the native 72-D concatenation of DAC's
nine 8-D projected quantizer inputs. The cache records that representation
directly and decoding snaps each chunk to its codebook. No `target_stats.pt` is
written: the trainer computes normalization for the new targets on first run.

Run parts in parallel, then finalize:
    python build_presnap_cache.py --part 0 --num-parts 3 --device cuda:2 &
    python build_presnap_cache.py --part 1 --num-parts 3 --device cuda:2 &
    python build_presnap_cache.py --part 2 --num-parts 3 --device cuda:2 &
    wait
    python build_presnap_cache.py --finalize
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR))

import torch

from build_diffusion_cache import _slice_or_pad_time
from build_source_cache import _load_audio_mono
from data.audio_codec_utils import PRESNAP_TARGET_LAYOUT, load_audio_codec_model

PACKAGE_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_SRC_CACHE = PACKAGE_ROOT.parent / "pca_diffusion/cache_4beats_dac44q9_pca72_native_bpmgeom_duration_v1"
DEFAULT_OUT = PACKAGE_ROOT.parent / "pca_diffusion/cache_4beats_dac44q9_presnap72_bpmgeom_duration_v1"
DEFAULT_DATASET_ROOT = Path(
    os.environ.get(
        "GROOVE_DATASET_ROOT",
        str(PACKAGE_ROOT.parent.parent.parent / "data/groove-v1.0.0"),
    )
)
SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src-cache", type=Path, default=DEFAULT_SRC_CACHE)
    parser.add_argument("--source-cache-root", type=Path, default=None, help="Defaults to the source cache recorded in the diffusion cache config.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--num-parts", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=0, help="Smoke test: stop after this many examples in this part.")
    parser.add_argument("--finalize", action="store_true", help="Write config, manifests, and summary after all parts finish.")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_part(args: argparse.Namespace) -> None:
    src_config = json.loads((args.src_cache / "config.json").read_text(encoding="utf-8"))
    source_root = Path(args.source_cache_root or src_config["source_cache_root"])
    source_rows = read_manifest(source_root / "manifest.jsonl")

    rows = [row for split in SPLITS for row in read_manifest(args.src_cache / "manifests" / f"{split}.jsonl")]
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_source[str(row["source_pt_rel"])].append(row)
    groups = sorted(by_source.items())[int(args.part) :: int(args.num_parts)]

    codec_model, _device, codec_meta = load_audio_codec_model(
        codec_family="dac",
        codec_model_id=str(src_config.get("codec_model_id", "descript/dac_44khz")),
        device=str(args.device),
        dac_num_quantizers=int(src_config.get("dac_num_quantizers", 9)),
    )
    codec_model.eval()
    num_quantizers = int(codec_meta.dac_num_quantizers)

    stats = {"examples": 0, "code_positions": 0, "code_matches": 0, "exact_examples": 0, "min_match_rate": 1.0}
    started = time.time()
    done = False
    with torch.no_grad():
        for source_pt_rel, group in groups:
            shard = torch.load(source_root / source_pt_rel, map_location="cpu", weights_only=False)
            audio_rel = source_rows[int(group[0]["source_manifest_index"])]["source_audio_file"]
            wav, _sr = _load_audio_mono(args.dataset_root / audio_rel, sample_rate=int(codec_meta.codec_sample_rate))
            for row in group:
                out_path = args.out_root / str(row["out_pt"])
                if out_path.exists():
                    existing = torch.load(out_path, map_location="cpu", weights_only=False)
                    existing_codes = torch.as_tensor(existing["source_codes_ct"])
                    existing_match_rate = float(existing.get("presnap_code_match_rate", 0.0))
                    positions = int(existing_codes.numel())
                    stats["examples"] += 1
                    stats["code_positions"] += positions
                    stats["code_matches"] += int(round(existing_match_rate * positions))
                    stats["exact_examples"] += int(existing_match_rate == 1.0)
                    stats["min_match_rate"] = min(stats["min_match_rate"], existing_match_rate)
                    continue
                payload = torch.load(args.src_cache / str(row["out_pt"]), map_location="cpu", weights_only=False)
                shard_row = int(payload["source_row_in_shard"])
                sample_start = int(shard["beat_sample_start"][shard_row])
                sample_end = int(shard["beat_sample_end"][shard_row])
                source_frames = int(shard["code_num_frames"][shard_row])
                target_frames = int(payload["target_num_frames"])

                window = wav[:, sample_start:sample_end].unsqueeze(0).to(args.device)
                encoded = codec_model.encode(window, n_quantizers=num_quantizers)
                projected_ct = _slice_or_pad_time(
                    encoded.projected_latents.detach().cpu().float(),
                    0,
                    source_valid_len=source_frames,
                    target_len=target_frames,
                    pad_mode="edge",
                )
                codes_ct = _slice_or_pad_time(
                    encoded.audio_codes.detach().cpu().long(),
                    0,
                    source_valid_len=source_frames,
                    target_len=target_frames,
                    pad_mode="edge",
                )
                target_pc_tk = projected_ct.transpose(0, 1).contiguous()
                if tuple(target_pc_tk.shape) != (target_frames, num_quantizers * 8) or not torch.isfinite(target_pc_tk).all():
                    raise RuntimeError(f"bad pre-snap target {tuple(target_pc_tk.shape)} for {row['out_pt']}")

                reference_codes = torch.as_tensor(payload["source_codes_ct"]).long()
                matches = int((codes_ct == reference_codes).sum())
                match_rate = float(matches) / float(reference_codes.numel())

                payload["target_pc_tk"] = target_pc_tk
                payload["target_pc_pool_k"] = target_pc_tk.sum(dim=0)
                payload["presnap_codes_ct"] = codes_ct.to(dtype=torch.int16)
                payload["presnap_code_match_rate"] = match_rate
                payload["target_layout"] = PRESNAP_TARGET_LAYOUT
                payload["target_space"] = PRESNAP_TARGET_LAYOUT
                payload["target_dim"] = int(target_pc_tk.shape[-1])
                payload["pca_basis_path"] = ""
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, out_path)

                stats["examples"] += 1
                stats["code_positions"] += int(reference_codes.numel())
                stats["code_matches"] += matches
                stats["exact_examples"] += int(matches == reference_codes.numel())
                stats["min_match_rate"] = min(stats["min_match_rate"], match_rate)
                if stats["examples"] % 500 == 0:
                    rate = stats["examples"] / max(1e-6, time.time() - started)
                    print(f"part {args.part}: {stats['examples']} examples ({rate:.1f}/s)", flush=True)
                if int(args.max_examples) > 0 and stats["examples"] >= int(args.max_examples):
                    done = True
                    break
            if done:
                break

    stats["elapsed_sec"] = time.time() - started
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / f"presnap_part{args.part}_of{args.num_parts}.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats), flush=True)


def finalize(args: argparse.Namespace) -> None:
    out = args.out_root
    (out / "manifests").mkdir(parents=True, exist_ok=True)
    missing = []
    for split in SPLITS:
        rows = read_manifest(args.src_cache / "manifests" / f"{split}.jsonl")
        missing += [row["out_pt"] for row in rows if not (out / row["out_pt"]).exists()]
        shutil.copy2(args.src_cache / "manifests" / f"{split}.jsonl", out / "manifests" / f"{split}.jsonl")
    if missing:
        raise SystemExit(f"{len(missing)} examples missing, e.g. {missing[:3]}; rerun the parts first")

    parts = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out.glob("presnap_part*_of*.json"))]
    positions = sum(p["code_positions"] for p in parts)
    summary = {
        "target_layout": PRESNAP_TARGET_LAYOUT,
        "latent_space": PRESNAP_TARGET_LAYOUT,
        "source_diffusion_cache": str(args.src_cache),
        "examples": sum(p["examples"] for p in parts),
        "code_match_rate": (sum(p["code_matches"] for p in parts) / positions) if positions else None,
        "exact_match_examples": sum(p["exact_examples"] for p in parts),
        "min_example_match_rate": min((p["min_match_rate"] for p in parts), default=None),
    }
    (out / "presnap_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    config = json.loads((args.src_cache / "config.json").read_text(encoding="utf-8"))
    presnap_dim = int(config.get("dac_num_quantizers", 9)) * 8
    config.update(
        {
            "out_root": str(out),
            "target_layout": PRESNAP_TARGET_LAYOUT,
            "target_dim": presnap_dim,
            "latent_space": PRESNAP_TARGET_LAYOUT,
            "pca_basis_path": "",
            "presnap_source_diffusion_cache": str(args.src_cache),
            "presnap_code_match_rate": summary["code_match_rate"],
        }
    )
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    for stale_path in (out / "target_stats.pt", out / "pca_basis.pt"):
        if stale_path.exists():
            stale_path.unlink()
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    if args.finalize:
        finalize(args)
    else:
        build_part(args)


if __name__ == "__main__":
    main()
