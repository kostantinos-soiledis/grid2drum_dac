#!/usr/bin/env python3
"""The ablations' metrics for the TRIA evaluation (evaluate.sh), scored against the real GMD bars.

The repo's evaluator (code/experiment/scripts/run_diffusion_acoustic_eval.py) scores against the decoded target: the
cached 9-codebook DAC latent of the bar (target_sum_td) through DAC's decoder, the codec ceiling of models that generate
those latents. TRIA generates with the same codec (DAC 44.1 kHz, 9 codebooks), yet the decoded target is not neutral
between the two: it is 3 dB of log-mel MAE away from the real bar, and on 60 test bars our model's clips scored 0.6 dB
better against it than against the real bar, TRIA's 0.6 dB worse. This script keeps the evaluator's metric code and
swaps the reference for the real bar: the GMD recording from the bar's start for its duration, the audio TRIA's bar
prompts are cut from (export_audio.bar_audio). Steps, under --out:

  real_targets/        the real bars in the evaluator's target-cache format, and their loudness (loudness.json)
  systems/<name>/      each system's clips gain-matched to their real bar's integrated loudness (ITU-R BS.1770,
                       pyloudnorm; a gain that would push the peak past 0.999 is reduced to that peak), with the
                       system's manifest and summary. The per-clip metrics are computed on peak-normalized audio and
                       do not change; FAD and KAD then compare what the drums sound like rather than output levels
                       (the real bars sit near -31 LUFS, our model's clips near -24, TRIA's at its pipeline's -20).
  acoustic_eval/       the evaluator's acoustic metrics (log-mel MAE, onset-flux cosine, ...) and FAD-inf
                       (CLAP-LAION-Music, 8 repeats) of every system against the real bars
                       (code/experiment/evaluate_ablations_4beat_acoustic.py)
  direct_audio_eval/   waveform L1 and MR-STFT log-magnitude L1 on peak-normalized audio against the real bars
                       (code/experiment/scripts/evaluate_diffusion_predictions.py's definitions), and onset precision,
                       recall and F1 at +-30 and +-50 ms (librosa onsets of the clip against the real bar's): precision
                       is the share of the clip's onsets that match one of the bar's, recall the share of the bar's
                       onsets that the clip reproduces; per system, and direct_per_clip_metrics.csv over all systems

--direct-only recomputes direct_audio_eval/ and direct_per_clip_metrics.csv from the real_targets/ and systems/ of a
previous run, leaving acoustic_eval/ and its FAD as they are.

    <eval python> evaluate_metrics.py --system grid_250hz=<exports>/grid_250hz --system tria_released_bar=... \\
        --out results/evaluation --gmd-root $GMD_ROOT --device cuda:0
"""
from __future__ import annotations

import argparse
import csv
import dataclasses  # noqa: F401  (stdlib modules first: code/experiment goes on sys.path below)
import inspect  # noqa: F401
import json
import math
import shutil
import sys
from pathlib import Path

import librosa
import mir_eval
import numpy as np
import pyloudnorm
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(REPO / "code" / "experiment"))
import export_audio  # noqa: E402
from model import mrstft_logmag_l1_per_example  # noqa: E402
from scripts import run_diffusion_acoustic_eval as rde  # noqa: E402
from scripts.evaluate_diffusion_predictions import _pad_or_trim_audio, _peak_normalize_audio  # noqa: E402

SR = export_audio.SAMPLE_RATE
PEAK = 0.999
SHARD = 8
ONSET_HOP = 256  # 5.8 ms at 44.1 kHz
ONSET_WINDOWS_MS = (30, 50)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def real_targets(cache_root: Path, gmd_root: Path, split: str, out: Path, max_items: int) -> dict[int, float]:
    """The real bars as the evaluator's target cache (TargetAudioCache reads it); returns their loudness (LUFS)."""
    rows = read_jsonl(cache_root / "manifests" / f"{split}.jsonl")
    rows = rows[:max_items] if max_items else rows
    source_rows = read_jsonl(Path(json.loads((cache_root / "config.json").read_text())["source_cache_root"]) / "manifest.jsonl")
    with (gmd_root / "info.csv").open() as stream:
        gmd = {r["id"]: r for r in csv.DictReader(stream)}
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    meter, loudness, manifest = pyloudnorm.Meter(SR), {}, []
    for first in range(0, len(rows), SHARD):
        clips = []
        for i, row in enumerate(rows[first: first + SHARD], start=first):
            source_row = source_rows[row["source_manifest_index"]]
            audio = export_audio.bar_audio(row, source_row, gmd_root / gmd[row["source_id"]]["audio_filename"])
            n = round(float(row["duration_sec"]) * SR)
            audio = np.pad(audio, (0, n - len(audio)))  # a bar running past the recording's end
            clips.append(audio)
            loudness[i] = meter.integrated_loudness(audio.astype(np.float64))
            manifest.append({**row, "dataset_index": i, "pt": f"shard_{first // SHARD:06d}.pt", "row_in_shard": i - first})
        lengths = torch.tensor([len(a) for a in clips])
        audio = torch.zeros(len(clips), 1, int(lengths.max()))
        for b, a in enumerate(clips):
            audio[b, 0, : len(a)] = torch.from_numpy(a)
        torch.save({"target_audio_32k": audio, "target_audio_32k_sample_rate": SR,  # the evaluator's key names
                    "target_audio_32k_context_ms": rde.EXPECTED_TARGET_AUDIO_CONTEXT_MS,
                    "target_audio_32k_num_samples": lengths, "target_audio_32k_beat_num_samples": lengths,
                    "target_audio_32k_loss_mask": torch.arange(audio.shape[-1])[None] < lengths[:, None],
                    "target_audio_32k_left_context_samples": torch.zeros(len(clips), dtype=torch.long),
                    "target_audio_32k_right_context_samples": torch.zeros(len(clips), dtype=torch.long)},
                   out / f"shard_{first // SHARD:06d}.pt")
    (out / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in manifest))
    (out / "loudness.json").write_text(json.dumps({str(k): v for k, v in loudness.items()}) + "\n")
    (out / "summary.json").write_text(json.dumps({"reference": "real GMD bars (export_audio.bar_audio)",
                                                  "cache_root": cache_root.name, "split": split, "sample_rate": SR,
                                                  "num_examples": len(rows)}, indent=2) + "\n")
    return loudness


def stage(predictions: Path, loudness: dict[int, float], out: Path) -> dict:
    """The system's clips gain-matched to their real bars' loudness (float wavs), with its manifest and summary."""
    if out.exists():
        shutil.rmtree(out)
    (out / "wavs").mkdir(parents=True)
    rows = [r for r in read_jsonl(predictions / "manifest.jsonl") if int(r["dataset_index"]) in loudness]
    gains, limited, unmeasured = [], 0, 0
    for row in rows:
        audio, sr = sf.read(predictions / row["wav"], dtype="float64")
        assert sr == SR, (row["wav"], sr)
        audio = audio.mean(axis=1) if audio.ndim == 2 else audio
        own, real = pyloudnorm.Meter(sr).integrated_loudness(audio), loudness[int(row["dataset_index"])]
        if math.isfinite(own) and math.isfinite(real):
            gain = 10 ** ((real - own) / 20)
            if np.abs(audio).max() * gain > PEAK:
                gain = PEAK / np.abs(audio).max()
                limited += 1
            audio = audio * gain
            gains.append(20 * math.log10(gain))
        else:
            unmeasured += 1  # silence: left as is
        sf.write(out / row["wav"], audio.astype(np.float32), sr, subtype="FLOAT")
    (out / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    match = {"mean_gain_db": float(np.mean(gains)) if gains else None, "peak_limited": limited, "unmeasured": unmeasured}
    summary = json.loads((predictions / "summary.json").read_text()) if (predictions / "summary.json").is_file() else {}
    summary.update(source_predictions_dir=str(predictions), loudness_match=match)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return match


def onsets(audio: np.ndarray) -> np.ndarray:
    """Onset times (s) by librosa's detector (spectral flux of the log-mel spectrogram, librosa's peak picking; level-free),
    the same settings for the real bars and every system's clips, as Break-the-Beat (arXiv 2605.14555) scores rhythm."""
    return librosa.onset.onset_detect(y=np.asarray(audio, dtype=np.float32), sr=SR, hop_length=ONSET_HOP, units="time")


def onset_scores(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    """Onset F1, precision and recall of the clip's onsets against the real bar's (mir_eval, one-to-one matching
    within +-window)."""
    out = {}
    for w in ONSET_WINDOWS_MS:
        f, p, r = mir_eval.onset.f_measure(reference, estimate, window=w / 1000)
        out.update({f"onset_f1_{w}ms": float(f), f"onset_precision_{w}ms": float(p), f"onset_recall_{w}ms": float(r)})
    return out


_REAL_ONSETS: dict[int, np.ndarray] = {}


def direct_metrics(name: str, staged: Path, targets: Path, out: Path, device: str) -> list[dict]:
    """Waveform L1 and MR-STFT log-magnitude L1 of peak-normalized clips, and onset scores, against the real bars."""
    cache = {}
    rows_out = []
    for target in read_jsonl(targets / "manifest.jsonl"):
        shard = cache.setdefault(target["pt"], torch.load(targets / target["pt"], map_location="cpu"))
        n = int(shard["target_audio_32k_num_samples"][target["row_in_shard"]])
        real = shard["target_audio_32k"][target["row_in_shard"], :, :n][None].to(device)
        row = rows_by_index(staged)[int(target["dataset_index"])]
        pred, sr = sf.read(staged / row["wav"], dtype="float32", always_2d=True)
        pred = torch.from_numpy(pred.mean(axis=1))[None, None].to(device)
        pred = _peak_normalize_audio(_pad_or_trim_audio(pred, n))
        real = _peak_normalize_audio(real)
        index = int(target["dataset_index"])
        if index not in _REAL_ONSETS:
            _REAL_ONSETS[index] = onsets(real[0, 0].cpu().numpy())
        pred_onsets = onsets(pred[0, 0].cpu().numpy())
        rows_out.append({"model": name, "dataset_index": int(target["dataset_index"]), "source_id": target["source_id"],
                         "beat_index": target["beat_index"], "target_num_samples": n, "pred_wav": row["wav"],
                         "audio_l1": float((pred - real).abs().mean()),
                         "mrstft_logmag_l1": float(mrstft_logmag_l1_per_example(pred, real, torch.tensor([n], device=device))[0]),
                         "real_onsets": len(_REAL_ONSETS[index]), "pred_onsets": len(pred_onsets),
                         **onset_scores(_REAL_ONSETS[index], pred_onsets)})
    (out / name).mkdir(parents=True, exist_ok=True)
    write_csv(out / name / "per_clip_metrics.csv", rows_out)
    summary = {"reference": "real GMD bars", "num_examples": len(rows_out), "predictions_dir": str(staged),
               "audio_l1_mean": float(np.mean([r["audio_l1"] for r in rows_out])),
               "mrstft_logmag_l1_mean": float(np.mean([r["mrstft_logmag_l1"] for r in rows_out])),
               **{f"{k}_mean": float(np.mean([r[k] for r in rows_out])) for w in ONSET_WINDOWS_MS
                  for k in (f"onset_f1_{w}ms", f"onset_precision_{w}ms", f"onset_recall_{w}ms")},
               "onsets_per_clip_mean": float(np.mean([r["pred_onsets"] for r in rows_out])),
               "real_onsets_per_clip_mean": float(np.mean([r["real_onsets"] for r in rows_out])),
               "onset_detector": f"librosa {librosa.__version__} onset_detect, hop {ONSET_HOP}; mir_eval "
                                 f"{mir_eval.__version__} onset.f_measure",
               "metric_basis": "peak_normalized_audio_vs_real_bars"}
    (out / name / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(name, {k: v for k, v in summary.items() if k.endswith("_mean")}, flush=True)
    return rows_out


_ROWS: dict[Path, dict[int, dict]] = {}


def rows_by_index(staged: Path) -> dict[int, dict]:
    if staged not in _ROWS:
        _ROWS[staged] = {int(r["dataset_index"]): r for r in read_jsonl(staged / "manifest.jsonl")}
    return _ROWS[staged]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", action="append", required=True, help="name=prediction dir (wavs/, manifest.jsonl)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gmd-root", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, default=export_audio.CACHE_ROOT)
    ap.add_argument("--split", default="test")
    ap.add_argument("--fad-model", default="clap-laion-music")
    ap.add_argument("--fad-repeats", type=int, default=8)
    ap.add_argument("--max-items", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--direct-only", action="store_true", help="recompute only the direct metrics of a previous run")
    args = ap.parse_args()
    systems = dict(s.split("=", 1) for s in args.system)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    staged = {name: out / "systems" / name for name in systems}
    if not args.direct_only:
        loudness = real_targets(args.cache_root, args.gmd_root, args.split, out / "real_targets", args.max_items)
        print(f"real bars: {len(loudness)}, median "
              f"{np.median([v for v in loudness.values() if math.isfinite(v)]):.1f} LUFS", flush=True)
        for name, d in systems.items():
            print(name, "loudness match", stage(Path(d).resolve(), loudness, staged[name]), flush=True)

    # before the acoustic evaluator: loading it sets NUMBA_DISABLE_JIT, under which librosa's onset module fails to import
    direct = []
    for name, d in staged.items():
        direct += direct_metrics(name, d, out / "real_targets", out / "direct_audio_eval", args.device)
    write_csv(out / "direct_per_clip_metrics.csv", direct)
    if args.direct_only:
        return

    for name, d in staged.items():
        rde._prepare_prediction_eval_input_root(eval_input_root=out / "eval_input", model_name=name, predictions_dir=d,
                                                overwrite=True)
    rde._run_sibling_acoustic_eval(
        eval_input_root=out / "eval_input", compat_cache_dir=out / "real_targets", acoustic_out_dir=out / "acoustic_eval",
        model_names=list(staged), max_items=args.max_items, device=args.device, skip_fad=False, skip_inference=True,
        no_plots=True, overwrite=True, fad_model=args.fad_model, fad_python=rde._resolve_fad_python("", skip_fad=False),
        fad_workers=1, fad_inf_workers=1, fad_repeats=args.fad_repeats)


if __name__ == "__main__":
    main()
