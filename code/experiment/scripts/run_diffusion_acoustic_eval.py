#!/usr/bin/env python3
"""Run direct and sibling acoustic evaluation for diffusion exports."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

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
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-drum-rendering")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataclasses import dataclass
from unittest import mock

import torch
from data.diffusion_dataset import build_diffusion_dataloader
from data.encodec_utils import load_audio_codec_model, resolve_codec_metadata_from_cache_config, resolve_device
from io_utils import write_json, write_jsonl
from model import (
    audio_valid_mask,
    decode_latent_to_audio,
    resolve_valid_audio_num_samples,
)
from scripts.evaluate_diffusion_predictions import evaluate_predictions
from scripts.export_best_diffusion_predictions import main as export_predictions_main


EXPECTED_TARGET_AUDIO_CONTEXT_MS = 20.0


@dataclass(frozen=True)
class EvalRunSpec:
    train_dir: Path | None
    model_name: str
    predictions_dir: Path | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build diffusion target-audio compatibility cache and run direct + sibling acoustic evaluation.",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default=str(RUNS_ROOT / "model_train"),
        help="Single-run train directory. Ignored when --models or --train-dirs is provided.",
    )
    parser.add_argument(
        "--train-root",
        type=str,
        default="",
        help="Optional root directory containing multiple run folders used by --models.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help="Optional run folder names under --train-root to export/evaluate together.",
    )
    parser.add_argument(
        "--train-dirs",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit run directories to export/evaluate together.",
    )
    parser.add_argument(
        "--train-names",
        type=str,
        nargs="*",
        default=None,
        help="Optional model names for --train-dirs. Defaults to each directory name.",
    )
    parser.add_argument(
        "--prediction-dirs",
        type=str,
        nargs="*",
        default=None,
        help="Optional already-exported prediction directories with manifest.jsonl.",
    )
    parser.add_argument(
        "--prediction-names",
        type=str,
        nargs="*",
        default=None,
        help="Optional model names for --prediction-dirs. Defaults to each directory name.",
    )
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument(
        "--cache-root",
        type=str,
        default=str(RUNS_ROOT / "mini_cache"),
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--predictions-dir", type=str, default="", help="Single-run override for prediction export directory.")
    parser.add_argument("--model-name", type=str, default="", help="Single-run model alias used inside the sibling evaluator.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Single-run: defaults to <train-dir>/<split>_audio_eval. Batch mode: defaults to <repo>/<split>_audio_eval_batch.",
    )
    parser.add_argument(
        "--acoustic-out-dir",
        type=str,
        default="",
        help="Optional output directory for the sibling acoustic evaluator. Defaults to <out-dir>/acoustic_eval.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--num-beats", type=int, default=4)
    parser.add_argument("--beat-crossfade-ms", type=float, default=10.0)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-direct-eval", action="store_true")
    parser.add_argument("--skip-fad", action="store_true")
    parser.add_argument(
        "--fad-model",
        type=str,
        default="encodec-emb",
        help="fadtk embedding model forwarded to the sibling acoustic evaluator.",
    )
    parser.add_argument(
        "--fad-python",
        type=str,
        default="",
        help="Optional Python executable used by the sibling evaluator for `python -m fadtk.embeds`.",
    )
    parser.add_argument(
        "--fad-workers",
        type=int,
        default=4,
        help="Worker count for fadtk embedding extraction.",
    )
    parser.add_argument(
        "--fad-inf-workers",
        type=int,
        default=1,
        help="Worker count for deterministic FAD-inf repeats. Defaults to 1 for stability.",
    )
    parser.add_argument(
        "--fad-repeats",
        type=int,
        default=8,
        help="Deterministic FAD-inf repeat count.",
    )
    parser.add_argument(
        "--skip-inference",
        dest="skip_inference",
        action="store_true",
        help="Skip paired bootstrap/permutation inference in the sibling evaluator (default).",
    )
    parser.add_argument(
        "--with-inference",
        dest="skip_inference",
        action="store_false",
        help="Enable paired bootstrap/permutation inference in the sibling evaluator.",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(skip_inference=True)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = str(line).strip()
            if not text:
                continue
            rows.append(dict(json.loads(text)))
    return rows


def _write_history_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if str(key) not in fieldnames:
                fieldnames.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _resolve_predictions_dir(train_dir: Path, split: str, explicit_predictions_dir: str) -> Path:
    if str(explicit_predictions_dir).strip():
        return Path(explicit_predictions_dir).expanduser().resolve()
    return (train_dir / f"{str(split).strip().lower()}_set_predictions").resolve()


def _resolve_out_dir(train_dir: Path, split: str, explicit_out_dir: str) -> Path:
    if str(explicit_out_dir).strip():
        return Path(explicit_out_dir).expanduser().resolve()
    return (train_dir / f"{str(split).strip().lower()}_audio_eval").resolve()


def _resolve_batch_out_dir(split: str, explicit_out_dir: str) -> Path:
    if str(explicit_out_dir).strip():
        return Path(explicit_out_dir).expanduser().resolve()
    return (REPO_ROOT / f"{str(split).strip().lower()}_audio_eval_batch").resolve()


def _resolve_model_name(train_dir: Path, explicit_model_name: str) -> str:
    text = str(explicit_model_name).strip()
    if text:
        return text
    return str(train_dir.name or "diffusion_run")


def _resolve_train_run_specs(
    *,
    train_dir: str | Path,
    model_name: str = "",
    train_root: str | Path | None = None,
    models: Sequence[str] | None = None,
    train_dirs: Sequence[str | Path] | None = None,
    train_names: Sequence[str] | None = None,
    prediction_dirs: Sequence[str | Path] | None = None,
    prediction_names: Sequence[str] | None = None,
) -> list[EvalRunSpec]:
    explicit_train_dirs = [str(item).strip() for item in list(train_dirs or []) if str(item).strip()]
    explicit_train_names = [str(item).strip() for item in list(train_names or []) if str(item).strip()]
    explicit_models = [str(item).strip() for item in list(models or []) if str(item).strip()]
    explicit_prediction_dirs = [str(item).strip() for item in list(prediction_dirs or []) if str(item).strip()]
    explicit_prediction_names = [str(item).strip() for item in list(prediction_names or []) if str(item).strip()]
    if explicit_train_dirs and explicit_models:
        raise ValueError("pass only one of --train-dirs or --models")
    if explicit_train_names and not explicit_train_dirs:
        raise ValueError("--train-names requires --train-dirs")
    if explicit_train_names and len(explicit_train_names) != len(explicit_train_dirs):
        raise ValueError("--train-names must have the same length as --train-dirs")
    if explicit_prediction_names and len(explicit_prediction_names) != len(explicit_prediction_dirs):
        raise ValueError("--prediction-names must have the same length as --prediction-dirs")

    specs: list[EvalRunSpec] = []
    if explicit_train_dirs:
        for idx, item in enumerate(explicit_train_dirs):
            path = Path(item).expanduser().resolve()
            if not path.is_dir():
                raise FileNotFoundError(f"train dir not found: {path}")
            name = explicit_train_names[idx] if explicit_train_names else str(path.name or "diffusion_run")
            specs.append(EvalRunSpec(train_dir=path, model_name=str(name)))
    elif explicit_models:
        root = Path(train_root).expanduser().resolve() if str(train_root or "").strip() else REPO_ROOT
        if not root.is_dir():
            raise FileNotFoundError(f"train root not found: {root}")
        for item in explicit_models:
            path = (root / str(item)).resolve()
            if not path.is_dir():
                raise FileNotFoundError(f"model folder not found under train root: {path}")
            specs.append(EvalRunSpec(train_dir=path, model_name=str(item)))
    elif not explicit_prediction_dirs:
        path = Path(train_dir).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"train dir not found: {path}")
        specs.append(EvalRunSpec(train_dir=path, model_name=_resolve_model_name(path, model_name)))

    for idx, item in enumerate(explicit_prediction_dirs):
        path = Path(item).expanduser().resolve()
        if not (path / "manifest.jsonl").is_file():
            raise FileNotFoundError(f"prediction manifest not found: {path / 'manifest.jsonl'}")
        name = explicit_prediction_names[idx] if explicit_prediction_names else str(path.name or f"prediction_{idx}")
        specs.append(EvalRunSpec(train_dir=None, model_name=str(name), predictions_dir=path))

    seen_dirs: set[str] = set()
    seen_names: set[str] = set()
    for spec in specs:
        dir_key = str(spec.train_dir or spec.predictions_dir)
        name_key = str(spec.model_name)
        if dir_key in seen_dirs:
            raise ValueError(f"duplicate run requested: {dir_key}")
        if name_key in seen_names:
            raise ValueError(f"duplicate model name requested: {spec.model_name}")
        seen_dirs.add(dir_key)
        seen_names.add(name_key)
    return specs


def _resolve_fad_python(explicit_fad_python: str, *, skip_fad: bool) -> str | None:
    if bool(skip_fad):
        return None
    text = str(explicit_fad_python).strip()
    candidates: list[str] = []
    if text:
        path = Path(text).expanduser()
        candidates.append(str(path.resolve()) if path.is_file() else str(text))
    candidates.append(str(Path(sys.executable).resolve()))
    for name in ("python3", "python"):
        resolved = shutil.which(name)
        if resolved:
            candidates.append(resolved)
    seen: set[str] = set()
    for candidate in candidates:
        candidate_text = str(candidate).strip()
        if not candidate_text or candidate_text in seen:
            continue
        seen.add(candidate_text)
        candidate_path = Path(candidate_text).expanduser()
        if candidate_path.is_file() and not candidate_path.exists():
            continue
        probe_argv = [
            candidate_text,
            "-c",
            (
                "import importlib.util as u, sys; "
                "mods=['torch','fadtk']; "
                "ok=all(bool(u.find_spec(m)) for m in mods); "
                "sys.exit(0 if ok else 1)"
            ),
        ]
        try:
            result = subprocess.run(
                probe_argv,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except Exception:
            continue
        if int(result.returncode) == 0:
            return candidate_text
    raise ModuleNotFoundError(
        "FAD was requested but no usable Python interpreter with both torch and fadtk was found; "
        "pass --fad-python or rerun with --skip-fad"
    )


def _load_sibling_acoustic_eval_module():
    candidate_paths = [
        (REPO_ROOT / "evaluate_ablations_4beat_acoustic.py").resolve(),
    ]
    script_path = next((path for path in candidate_paths if path.is_file()), None)
    if script_path is None:
        raise FileNotFoundError(
            "acoustic eval script not found; looked for "
            + ", ".join(str(path) for path in candidate_paths)
        )
    script_dir = str(script_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    module_name = "evaluate_ablations_4beat_acoustic"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load sibling acoustic eval module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_train_dir_compat_files(train_dir: Path) -> None:
    run_config_path = train_dir / "run_config.json"
    config_path = train_dir / "config.json"
    if not config_path.is_file() and run_config_path.is_file():
        config_path.write_text(run_config_path.read_text(encoding="utf-8"), encoding="utf-8")

    history_jsonl_path = train_dir / "history.jsonl"
    history_csv_path = train_dir / "history.csv"
    if not history_csv_path.is_file() and history_jsonl_path.is_file():
        _write_history_csv(history_csv_path, _read_jsonl(history_jsonl_path))

    checkpoint_path = None
    for candidate_name in ("best_diffusion.pt", "best.pt", "last.pt"):
        candidate_path = train_dir / candidate_name
        if candidate_path.is_file():
            checkpoint_path = candidate_path
            break
    if checkpoint_path is None:
        return

    checkpoint_payload = None
    if not config_path.is_file() or not history_csv_path.is_file():
        checkpoint_payload = dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False))

    if not config_path.is_file() and checkpoint_payload is not None:
        model_state_dict = dict(checkpoint_payload.get("model_state_dict") or {})
        num_parameters = int(
            sum(int(torch.as_tensor(value).numel()) for value in model_state_dict.values())
        )
        write_json(
            config_path,
            {
                "checkpoint_path": str(checkpoint_path),
                "model_cfg": checkpoint_payload.get("config"),
                "frontend_cfg": checkpoint_payload.get("frontend_cfg"),
                "num_steps": checkpoint_payload.get("num_steps"),
                "sample_rate": checkpoint_payload.get("sample_rate"),
                "checkpoint_epoch": checkpoint_payload.get("epoch"),
                "best_val_loss": checkpoint_payload.get("best_val_loss"),
                "best_checkpoint_metric_name": checkpoint_payload.get("best_checkpoint_metric_name"),
                "best_checkpoint_metric_value": checkpoint_payload.get("best_checkpoint_metric_value"),
                "best_checkpoint_epoch": checkpoint_payload.get("best_checkpoint_epoch"),
                "num_parameters": int(num_parameters),
            },
        )

    if not history_csv_path.is_file() and checkpoint_payload is not None:
        history_row = {
            "epoch": int(checkpoint_payload.get("epoch", -1)),
            "val_loss": checkpoint_payload.get("val_loss"),
            "best_val_loss": checkpoint_payload.get("best_val_loss"),
            "best_checkpoint_epoch": int(checkpoint_payload.get("best_checkpoint_epoch", checkpoint_payload.get("epoch", -1))),
            "checkpoint_improved": True,
            "checkpoint_metric_name": checkpoint_payload.get("best_checkpoint_metric_name"),
            "checkpoint_metric_value": checkpoint_payload.get("best_checkpoint_metric_value"),
        }
        _write_history_csv(history_csv_path, [history_row])


def _prepare_eval_input_root(
    *,
    eval_input_root: Path,
    model_name: str,
    train_dir: Path,
    predictions_dir: Path,
    overwrite: bool,
) -> Path:
    run_root = (eval_input_root / model_name).resolve()
    if run_root.exists() and bool(overwrite):
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    for file_name in ("history.csv", "config.json"):
        source_path = train_dir / file_name
        if source_path.is_file():
            target_path = run_root / file_name
            if target_path.exists() or target_path.is_symlink():
                if target_path.is_dir():
                    shutil.rmtree(target_path)
                else:
                    target_path.unlink()
            try:
                target_path.symlink_to(source_path)
            except OSError:
                shutil.copy2(source_path, target_path)

    target_predictions_root = run_root / "test_set_predictions"
    if target_predictions_root.exists() or target_predictions_root.is_symlink():
        if target_predictions_root.is_dir() and not target_predictions_root.is_symlink():
            shutil.rmtree(target_predictions_root)
        else:
            target_predictions_root.unlink()
    try:
        target_predictions_root.symlink_to(predictions_dir, target_is_directory=True)
    except OSError:
        shutil.copytree(predictions_dir, target_predictions_root)
    return run_root


def _prepare_prediction_eval_input_root(
    *,
    eval_input_root: Path,
    model_name: str,
    predictions_dir: Path,
    overwrite: bool,
) -> Path:
    run_root = (eval_input_root / model_name).resolve()
    if run_root.exists() and bool(overwrite):
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    summary_path = predictions_dir / "summary.json"
    config_payload: dict[str, Any] = {
        "prediction_dir": str(predictions_dir),
        "num_parameters": 0,
    }
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            config_payload.update(dict(summary))
        except Exception:
            pass
    write_json(run_root / "config.json", config_payload)
    history_row = {
        "epoch": int(config_payload.get("checkpoint_epoch", -1) or -1),
        "val_loss": config_payload.get("best_val_loss"),
        "best_val_loss": config_payload.get("best_val_loss"),
        "best_checkpoint_epoch": int(config_payload.get("checkpoint_epoch", -1) or -1),
        "checkpoint_improved": True,
    }
    _write_history_csv(run_root / "history.csv", [history_row])

    target_predictions_root = run_root / "test_set_predictions"
    if target_predictions_root.exists() or target_predictions_root.is_symlink():
        if target_predictions_root.is_dir() and not target_predictions_root.is_symlink():
            shutil.rmtree(target_predictions_root)
        else:
            target_predictions_root.unlink()
    try:
        target_predictions_root.symlink_to(predictions_dir, target_is_directory=True)
    except OSError:
        shutil.copytree(predictions_dir, target_predictions_root)
    return run_root


def _export_predictions_for_run(
    *,
    train_dir: Path,
    checkpoint: str,
    cache_root: str | Path,
    split: str,
    predictions_dir: Path,
    device: str,
    batch_size: int,
    num_workers: int,
    max_items: int,
    num_beats: int,
    beat_crossfade_ms: float,
) -> None:
    export_argv = [
        "export_best_diffusion_predictions.py",
        "--train-dir",
        str(train_dir),
        "--cache-root",
        str(Path(cache_root).expanduser().resolve()),
        "--split",
        str(split),
        "--out-dir",
        str(predictions_dir),
        "--device",
        str(device),
        "--batch-size",
        str(int(batch_size)),
        "--num-workers",
        str(int(num_workers)),
        "--max-items",
        str(int(max_items)),
        "--num-beats",
        str(int(num_beats)),
        "--beat-crossfade-ms",
        str(float(beat_crossfade_ms)),
        "--overwrite",
    ]
    if str(checkpoint).strip():
        export_argv.extend(["--checkpoint", str(Path(checkpoint).expanduser().resolve())])
    with mock.patch.object(sys, "argv", list(export_argv)):
        export_predictions_main()


def _run_sibling_acoustic_eval(
    *,
    eval_input_root: Path,
    compat_cache_dir: Path,
    acoustic_out_dir: Path,
    model_names: Sequence[str],
    max_items: int,
    device: str,
    skip_fad: bool,
    skip_inference: bool,
    no_plots: bool,
    overwrite: bool,
    fad_model: str,
    fad_python: str | None,
    fad_workers: int,
    fad_inf_workers: int,
    fad_repeats: int,
) -> dict[str, Any] | None:
    acoustic_module = _load_sibling_acoustic_eval_module()
    acoustic_argv = [
        "evaluate_ablations_4beat_acoustic.py",
        str(eval_input_root),
        "--cache-root",
        str(compat_cache_dir),
        "--out-root",
        str(acoustic_out_dir),
        "--models",
        *[str(name) for name in list(model_names)],
        "--max-items",
        str(int(max_items)),
        "--device",
        str(device),
    ]
    if bool(overwrite):
        acoustic_argv.append("--overwrite")
    if bool(skip_fad):
        acoustic_argv.append("--skip-fad")
    else:
        acoustic_argv.extend(["--fad-model", str(fad_model)])
        if fad_python is not None:
            acoustic_argv.extend(["--fad-python", str(fad_python)])
        acoustic_argv.extend(
            [
                "--fad-workers",
                str(int(fad_workers)),
                "--fad-inf-workers",
                str(int(fad_inf_workers)),
                "--fad-repeats",
                str(int(fad_repeats)),
            ]
        )
    if bool(skip_inference):
        acoustic_argv.append("--skip-inference")
    if bool(no_plots):
        acoustic_argv.append("--no-plots")
    with mock.patch.object(sys, "argv", list(acoustic_argv)):
        acoustic_module.main()

    acoustic_summary_path = acoustic_out_dir / "summary.json"
    if acoustic_summary_path.is_file():
        summary = json.loads(acoustic_summary_path.read_text(encoding="utf-8"))
        if not bool(skip_fad) and not bool(summary.get("fad_completed")):
            raise RuntimeError(f"FAD was requested but did not complete: {summary.get('fad_error')}")
        return summary
    return None


@torch.no_grad()
def build_target_audio_cache(
    *,
    cache_root: str | Path,
    split: str,
    out_dir: str | Path,
    device: str = "auto",
    batch_size: int = 8,
    num_workers: int = 0,
    max_items: int = 0,
    overwrite: bool = False,
) -> dict[str, Any]:
    cache_root_path = Path(cache_root).expanduser().resolve()
    out_dir_path = Path(out_dir).expanduser().resolve()
    split_name = str(split).strip().lower()
    split_manifest_path = cache_root_path / "manifests" / f"{split_name}.jsonl"
    if not split_manifest_path.is_file():
        raise FileNotFoundError(f"split manifest not found: {split_manifest_path}")
    split_rows = _read_jsonl(split_manifest_path)
    if int(max_items) > 0:
        split_rows = split_rows[: int(max_items)]

    if out_dir_path.exists():
        if bool(overwrite):
            shutil.rmtree(out_dir_path)
        elif any(out_dir_path.iterdir()):
            raise FileExistsError(f"output directory already exists and is not empty: {out_dir_path}")
    out_dir_path.mkdir(parents=True, exist_ok=True)

    resolved_device = resolve_device(str(device))
    torch_device = torch.device(resolved_device)
    if torch_device.type == "cuda" and torch_device.index is not None:
        torch.cuda.set_device(torch_device)
    codec_metadata = resolve_codec_metadata_from_cache_config(cache_root_path)
    encodec_model, _resolved_codec_device, codec_metadata = load_audio_codec_model(
        device=resolved_device,
        metadata=codec_metadata,
    )
    sample_rate = int(codec_metadata.codec_sample_rate)
    dataloader = build_diffusion_dataloader(
        cache_root_path,
        split=split_name,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        max_items=int(max_items),
        pin_memory=False,
    )

    manifest_rows: list[dict[str, Any]] = []
    dataset_index_base = 0
    shard_idx = 0
    for batch in dataloader:
        target_btd = torch.as_tensor(
            batch.get("target_sum_btd", batch["target_btd"]),
            dtype=torch.float32,
            device=torch_device,
        )
        target_valid_mask_bt = torch.as_tensor(batch["target_valid_mask_bt"], dtype=torch.bool, device=torch_device)
        duration_sec_b = torch.as_tensor(batch["duration_sec"], dtype=torch.float32, device=torch_device)

        target_latent = target_btd * target_valid_mask_bt.unsqueeze(-1).to(dtype=target_btd.dtype)
        target_audio_bct = decode_latent_to_audio(target_latent, encodec_model)
        valid_num_samples_b = resolve_valid_audio_num_samples(
            duration_sec_b,
            sample_rate=int(sample_rate),
            max_num_samples=int(target_audio_bct.shape[-1]),
        )
        target_loss_mask_bt = audio_valid_mask(
            valid_num_samples_b,
            max_num_samples=int(target_audio_bct.shape[-1]),
        )
        batch_size_eff = int(target_audio_bct.shape[0])

        shard_name = f"shard_{int(shard_idx):06d}.pt"
        torch.save(
            {
                "target_audio_32k": target_audio_bct.detach().cpu().to(dtype=torch.float32).contiguous(),
                "target_audio_32k_sample_rate": int(sample_rate),
                "target_audio_32k_context_ms": float(EXPECTED_TARGET_AUDIO_CONTEXT_MS),
                "target_audio_32k_num_samples": valid_num_samples_b.detach().cpu().to(dtype=torch.long).contiguous(),
                "target_audio_32k_beat_num_samples": valid_num_samples_b.detach().cpu().to(dtype=torch.long).contiguous(),
                "target_audio_32k_loss_mask": target_loss_mask_bt.detach().cpu().to(dtype=torch.bool).contiguous(),
                "target_audio_32k_left_context_samples": torch.zeros((batch_size_eff,), dtype=torch.long),
                "target_audio_32k_right_context_samples": torch.zeros((batch_size_eff,), dtype=torch.long),
            },
            out_dir_path / shard_name,
        )

        for row_in_shard in range(batch_size_eff):
            dataset_index = int(dataset_index_base + row_in_shard)
            source_row = dict(split_rows[int(dataset_index)])
            manifest_rows.append(
                {
                    **source_row,
                    "dataset_index": int(dataset_index),
                    "pt": str(shard_name),
                    "row_in_shard": int(row_in_shard),
                }
            )

        dataset_index_base += batch_size_eff
        shard_idx += 1

    write_jsonl(out_dir_path / "manifest.jsonl", manifest_rows)
    summary = {
        "cache_root": str(cache_root_path),
        "out_dir": str(out_dir_path),
        "split": str(split_name),
        "resolved_device": str(resolved_device),
        "sample_rate": int(sample_rate),
        "num_examples": int(len(manifest_rows)),
        "num_shards": int(shard_idx),
        "context_ms": float(EXPECTED_TARGET_AUDIO_CONTEXT_MS),
    }
    write_json(out_dir_path / "summary.json", summary)
    return summary


def run_diffusion_acoustic_eval(
    *,
    train_dir: str | Path,
    checkpoint: str = "",
    cache_root: str | Path = RUNS_ROOT / "mini_cache",
    split: str = "test",
    predictions_dir: str | Path | None = None,
    model_name: str = "",
    out_dir: str | Path | None = None,
    acoustic_out_dir: str | Path | None = None,
    device: str = "auto",
    batch_size: int = 8,
    num_workers: int = 0,
    max_items: int = 0,
    num_beats: int = 4,
    beat_crossfade_ms: float = 10.0,
    skip_export: bool = False,
    skip_direct_eval: bool = False,
    skip_fad: bool = False,
    skip_inference: bool = False,
    no_plots: bool = False,
    overwrite: bool = False,
    fad_model: str = "encodec-emb",
    fad_python: str | None = None,
    fad_workers: int = 4,
    fad_inf_workers: int = 1,
    fad_repeats: int = 8,
) -> dict[str, Any]:
    train_dir_path = Path(train_dir).expanduser().resolve()
    split_name = str(split).strip().lower()
    predictions_dir_path = _resolve_predictions_dir(train_dir_path, split_name, str(predictions_dir or ""))
    out_dir_path = _resolve_out_dir(train_dir_path, split_name, str(out_dir or ""))
    compat_cache_dir = (out_dir_path / "target_audio_cache").resolve()
    direct_eval_dir = (out_dir_path / "direct_audio_eval").resolve()
    eval_input_root = (out_dir_path / "eval_input").resolve()
    acoustic_out_dir_path = (
        Path(acoustic_out_dir).expanduser().resolve()
        if str(acoustic_out_dir or "").strip()
        else (out_dir_path / "acoustic_eval").resolve()
    )
    resolved_model_name = _resolve_model_name(train_dir_path, str(model_name))
    resolved_fad_python = _resolve_fad_python(str(fad_python or ""), skip_fad=bool(skip_fad))

    if out_dir_path.exists() and bool(overwrite):
        shutil.rmtree(out_dir_path)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    if not bool(skip_export):
        _export_predictions_for_run(
            train_dir=train_dir_path,
            checkpoint=str(checkpoint),
            cache_root=cache_root,
            split=split_name,
            predictions_dir=predictions_dir_path,
            device=str(device),
            batch_size=int(batch_size),
            num_workers=int(num_workers),
            max_items=int(max_items),
            num_beats=int(num_beats),
            beat_crossfade_ms=float(beat_crossfade_ms),
        )
    elif not (predictions_dir_path / "manifest.jsonl").is_file():
        raise FileNotFoundError(
            f"predictions manifest missing and --skip-export was set: {predictions_dir_path / 'manifest.jsonl'}"
        )

    direct_summary = None
    if not bool(skip_direct_eval):
        direct_summary = evaluate_predictions(
            cache_root=cache_root,
            split=split_name,
            predictions_dir=predictions_dir_path,
            out_dir=direct_eval_dir,
            device=device,
            max_items=int(max_items),
            overwrite=True,
        )

    compat_cache_summary = build_target_audio_cache(
        cache_root=cache_root,
        split=split_name,
        out_dir=compat_cache_dir,
        device=device,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        max_items=int(max_items),
        overwrite=True,
    )

    _ensure_train_dir_compat_files(train_dir_path)
    _prepare_eval_input_root(
        eval_input_root=eval_input_root,
        model_name=resolved_model_name,
        train_dir=train_dir_path,
        predictions_dir=predictions_dir_path,
        overwrite=True,
    )

    acoustic_summary = _run_sibling_acoustic_eval(
        eval_input_root=eval_input_root,
        compat_cache_dir=compat_cache_dir,
        acoustic_out_dir=acoustic_out_dir_path,
        model_names=[resolved_model_name],
        max_items=int(max_items),
        device=str(device),
        skip_fad=bool(skip_fad),
        skip_inference=bool(skip_inference),
        no_plots=bool(no_plots),
        overwrite=True,
        fad_model=str(fad_model),
        fad_python=resolved_fad_python,
        fad_workers=int(fad_workers),
        fad_inf_workers=int(fad_inf_workers),
        fad_repeats=int(fad_repeats),
    )
    summary = {
        "train_dir": str(train_dir_path),
        "cache_root": str(Path(cache_root).expanduser().resolve()),
        "split": str(split_name),
        "model_name": str(resolved_model_name),
        "fad_model": str(fad_model),
        "fad_python": resolved_fad_python,
        "predictions_dir": str(predictions_dir_path),
        "direct_eval_dir": str(direct_eval_dir),
        "compat_cache_dir": str(compat_cache_dir),
        "acoustic_eval_dir": str(acoustic_out_dir_path),
        "direct_summary": direct_summary,
        "compat_cache_summary": compat_cache_summary,
        "acoustic_summary": acoustic_summary,
    }
    write_json(out_dir_path / "summary.json", summary)
    return summary


def run_diffusion_acoustic_eval_batch(
    *,
    train_dir: str | Path = RUNS_ROOT / "model_train",
    train_root: str | Path | None = None,
    models: Sequence[str] | None = None,
    train_dirs: Sequence[str | Path] | None = None,
    train_names: Sequence[str] | None = None,
    prediction_dirs: Sequence[str | Path] | None = None,
    prediction_names: Sequence[str] | None = None,
    cache_root: str | Path = RUNS_ROOT / "mini_cache",
    split: str = "test",
    out_dir: str | Path | None = None,
    acoustic_out_dir: str | Path | None = None,
    device: str = "auto",
    batch_size: int = 8,
    num_workers: int = 0,
    max_items: int = 0,
    num_beats: int = 4,
    beat_crossfade_ms: float = 10.0,
    skip_export: bool = False,
    skip_direct_eval: bool = False,
    skip_fad: bool = False,
    skip_inference: bool = False,
    no_plots: bool = False,
    overwrite: bool = False,
    fad_model: str = "encodec-emb",
    fad_python: str | None = None,
    fad_workers: int = 4,
    fad_inf_workers: int = 1,
    fad_repeats: int = 8,
) -> dict[str, Any]:
    split_name = str(split).strip().lower()
    run_specs = _resolve_train_run_specs(
        train_dir=train_dir,
        train_root=train_root,
        models=models,
        train_dirs=train_dirs,
        train_names=train_names,
        prediction_dirs=prediction_dirs,
        prediction_names=prediction_names,
    )
    out_root = _resolve_batch_out_dir(split_name, str(out_dir or ""))
    predictions_root = (out_root / "predictions").resolve()
    direct_eval_root = (out_root / "direct_audio_eval").resolve()
    compat_cache_dir = (out_root / "target_audio_cache").resolve()
    eval_input_root = (out_root / "eval_input").resolve()
    acoustic_out_dir_path = (
        Path(acoustic_out_dir).expanduser().resolve()
        if str(acoustic_out_dir or "").strip()
        else (out_root / "acoustic_eval").resolve()
    )
    resolved_fad_python = _resolve_fad_python(str(fad_python or ""), skip_fad=bool(skip_fad))

    if out_root.exists():
        if bool(overwrite):
            shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    compat_cache_summary = build_target_audio_cache(
        cache_root=cache_root,
        split=split_name,
        out_dir=compat_cache_dir,
        device=device,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        max_items=int(max_items),
        overwrite=True,
    )

    run_summaries: list[dict[str, Any]] = []
    for spec in run_specs:
        predictions_dir = (
            spec.predictions_dir.resolve()
            if spec.predictions_dir is not None
            else (predictions_root / str(spec.model_name)).resolve()
        )
        if spec.train_dir is not None and not bool(skip_export):
            _export_predictions_for_run(
                train_dir=spec.train_dir,
                checkpoint="",
                cache_root=cache_root,
                split=split_name,
                predictions_dir=predictions_dir,
                device=str(device),
                batch_size=int(batch_size),
                num_workers=int(num_workers),
                max_items=int(max_items),
                num_beats=int(num_beats),
                beat_crossfade_ms=float(beat_crossfade_ms),
            )
        elif not (predictions_dir / "manifest.jsonl").is_file():
            raise FileNotFoundError(
                f"predictions manifest missing: {predictions_dir / 'manifest.jsonl'}"
            )

        direct_summary = None
        direct_eval_dir = (direct_eval_root / str(spec.model_name)).resolve()
        if not bool(skip_direct_eval):
            direct_summary = evaluate_predictions(
                cache_root=cache_root,
                split=split_name,
                predictions_dir=predictions_dir,
                out_dir=direct_eval_dir,
                device=device,
                max_items=int(max_items),
                overwrite=True,
            )

        if spec.train_dir is not None:
            _ensure_train_dir_compat_files(spec.train_dir)
            _prepare_eval_input_root(
                eval_input_root=eval_input_root,
                model_name=str(spec.model_name),
                train_dir=spec.train_dir,
                predictions_dir=predictions_dir,
                overwrite=True,
            )
        else:
            _prepare_prediction_eval_input_root(
                eval_input_root=eval_input_root,
                model_name=str(spec.model_name),
                predictions_dir=predictions_dir,
                overwrite=True,
            )
        run_summaries.append(
            {
                "train_dir": None if spec.train_dir is None else str(spec.train_dir),
                "model_name": str(spec.model_name),
                "predictions_dir": str(predictions_dir),
                "direct_eval_dir": str(direct_eval_dir),
                "direct_summary": direct_summary,
            }
        )

    acoustic_summary = _run_sibling_acoustic_eval(
        eval_input_root=eval_input_root,
        compat_cache_dir=compat_cache_dir,
        acoustic_out_dir=acoustic_out_dir_path,
        model_names=[spec.model_name for spec in run_specs],
        max_items=int(max_items),
        device=str(device),
        skip_fad=bool(skip_fad),
        skip_inference=bool(skip_inference),
        no_plots=bool(no_plots),
        overwrite=True,
        fad_model=str(fad_model),
        fad_python=resolved_fad_python,
        fad_workers=int(fad_workers),
        fad_inf_workers=int(fad_inf_workers),
        fad_repeats=int(fad_repeats),
    )
    summary = {
        "train_root": (
            str(Path(train_root).expanduser().resolve())
            if str(train_root or "").strip()
            else None
        ),
        "cache_root": str(Path(cache_root).expanduser().resolve()),
        "split": str(split_name),
        "fad_model": str(fad_model),
        "fad_python": resolved_fad_python,
        "out_dir": str(out_root),
        "compat_cache_dir": str(compat_cache_dir),
        "acoustic_eval_dir": str(acoustic_out_dir_path),
        "compat_cache_summary": compat_cache_summary,
        "acoustic_summary": acoustic_summary,
        "runs": run_summaries,
    }
    write_json(out_root / "summary.json", summary)
    return summary


def main() -> None:
    args = _parse_args()
    explicit_models = [str(item).strip() for item in list(args.models or []) if str(item).strip()]
    explicit_train_dirs = [str(item).strip() for item in list(args.train_dirs or []) if str(item).strip()]
    explicit_train_names = [str(item).strip() for item in list(args.train_names or []) if str(item).strip()]
    explicit_prediction_dirs = [str(item).strip() for item in list(args.prediction_dirs or []) if str(item).strip()]
    explicit_prediction_names = [str(item).strip() for item in list(args.prediction_names or []) if str(item).strip()]
    batch_mode = bool(explicit_models or explicit_train_dirs or explicit_prediction_dirs)
    if bool(batch_mode):
        if str(args.predictions_dir).strip():
            raise ValueError("--predictions-dir is only supported in single-run mode")
        if str(args.checkpoint).strip():
            raise ValueError("--checkpoint is only supported in single-run mode")
        if str(args.model_name).strip():
            raise ValueError("--model-name is only supported in single-run mode")
        summary = run_diffusion_acoustic_eval_batch(
            train_dir=args.train_dir,
            train_root=(str(args.train_root) if str(args.train_root).strip() else None),
            models=explicit_models,
            train_dirs=explicit_train_dirs,
            train_names=explicit_train_names,
            prediction_dirs=explicit_prediction_dirs,
            prediction_names=explicit_prediction_names,
            cache_root=args.cache_root,
            split=str(args.split),
            out_dir=(str(args.out_dir) if str(args.out_dir).strip() else None),
            acoustic_out_dir=(str(args.acoustic_out_dir) if str(args.acoustic_out_dir).strip() else None),
            device=str(args.device),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            max_items=int(args.max_items),
            num_beats=int(args.num_beats),
            beat_crossfade_ms=float(args.beat_crossfade_ms),
            skip_export=bool(args.skip_export),
            skip_direct_eval=bool(args.skip_direct_eval),
            skip_fad=bool(args.skip_fad),
            skip_inference=bool(args.skip_inference),
            no_plots=bool(args.no_plots),
            overwrite=bool(args.overwrite),
            fad_model=str(args.fad_model),
            fad_python=(str(args.fad_python) if str(args.fad_python).strip() else None),
            fad_workers=int(args.fad_workers),
            fad_inf_workers=int(args.fad_inf_workers),
            fad_repeats=int(args.fad_repeats),
        )
        print(
            "batch acoustic eval complete: "
            f"runs={len(summary['runs'])} "
            f"acoustic_eval_dir={summary['acoustic_eval_dir']}"
        )
        return

    summary = run_diffusion_acoustic_eval(
        train_dir=args.train_dir,
        checkpoint=str(args.checkpoint),
        cache_root=args.cache_root,
        split=str(args.split),
        predictions_dir=(str(args.predictions_dir) if str(args.predictions_dir).strip() else None),
        model_name=str(args.model_name),
        out_dir=(str(args.out_dir) if str(args.out_dir).strip() else None),
        acoustic_out_dir=(str(args.acoustic_out_dir) if str(args.acoustic_out_dir).strip() else None),
        device=str(args.device),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        max_items=int(args.max_items),
        num_beats=int(args.num_beats),
        beat_crossfade_ms=float(args.beat_crossfade_ms),
        skip_export=bool(args.skip_export),
        skip_direct_eval=bool(args.skip_direct_eval),
        skip_fad=bool(args.skip_fad),
        skip_inference=bool(args.skip_inference),
        no_plots=bool(args.no_plots),
        overwrite=bool(args.overwrite),
        fad_model=str(args.fad_model),
        fad_python=(str(args.fad_python) if str(args.fad_python).strip() else None),
        fad_workers=int(args.fad_workers),
        fad_inf_workers=int(args.fad_inf_workers),
        fad_repeats=int(args.fad_repeats),
    )
    print(
        "acoustic eval complete: "
        f"predictions_dir={summary['predictions_dir']} "
        f"acoustic_eval_dir={summary['acoustic_eval_dir']}"
    )


if __name__ == "__main__":
    main()
