#!/usr/bin/env python3
"""Run frontend-conditioning ablation exports and paired evals for direct/DAC-CE runs."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from io_utils import write_json, write_jsonl
from scripts.conditioning_ablation import VALID_CONDITIONING_ABLATIONS, normalize_conditioning_ablation


DEFAULT_CACHE_ROOT = RUNS_ROOT / "mini_cache"
DEFAULT_OUT_ROOT = RESULTS_ROOT / "eval" / "frontend_ablation"
DEFAULT_DIRECT_RUNS = (RUNS_ROOT / "runs_direct" / "direct_pca_d1024_l6_seed1234",)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=str, default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--out-root", type=str, default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--ablations",
        type=str,
        default="none,zero,shuffle,phase_shift",
        help=f"Comma-separated modes from: {', '.join(VALID_CONDITIONING_ABLATIONS)}.",
    )
    parser.add_argument("--direct-runs", type=str, nargs="*", default=None)
    parser.add_argument("--dac-ce-runs", type=str, nargs="*", default=None)
    parser.add_argument("--skip-direct", action="store_true")
    parser.add_argument("--skip-dac-ce", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-direct-audio-eval", action="store_true")
    parser.add_argument("--skip-control-faithfulness", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--direct-batch-size", type=int, default=16)
    parser.add_argument("--dac-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument(
        "--eval-max-items",
        type=int,
        default=-1,
        help="Metric max-items. Defaults to --max-items; use 0 for all exported predictions.",
    )
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=1234,
        help="Fixed diffusion sample seed; batch index is added by the exporter.",
    )
    parser.add_argument("--use-bpm-inference-geometry", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return dict(json.load(handle))


def _parse_ablations(text: str) -> list[str]:
    modes: list[str] = []
    for part in str(text).split(","):
        mode = normalize_conditioning_ablation(part)
        if mode not in modes:
            modes.append(mode)
    if not modes:
        raise ValueError("--ablations resolved to an empty mode list")
    return modes


def _default_dac_ce_runs() -> list[Path]:
    root = RUNS_ROOT / "runs_dac_ce"
    if not root.is_dir():
        return []

    def sort_key(path: Path) -> tuple[int, str]:
        digits = "".join(char for char in path.name if char.isdigit())
        return (int(digits) if digits else 10**9, path.name)

    return sorted((path for path in root.iterdir() if path.is_dir() and (path / "best_diffusion.pt").is_file()), key=sort_key)


def _resolve_runs(explicit: list[str] | None, defaults: list[Path] | tuple[Path, ...]) -> list[Path]:
    paths = [Path(item).expanduser().resolve() for item in explicit] if explicit is not None else [path.resolve() for path in defaults]
    existing = [path for path in paths if path.is_dir()]
    missing = [path for path in paths if not path.is_dir()]
    if missing:
        raise FileNotFoundError("missing run directories: " + ", ".join(str(path) for path in missing))
    return existing


def _run_command(cmd: list[str], *, dry_run: bool) -> dict[str, Any]:
    rendered = shlex.join(str(part) for part in cmd)
    print(rendered, flush=True)
    if dry_run:
        return {"command": rendered, "returncode": None, "dry_run": True}
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-drum-rendering")
    completed = subprocess.run([str(part) for part in cmd], cwd=str(REPO_ROOT), env=env, check=True)
    return {"command": rendered, "returncode": int(completed.returncode), "dry_run": False}


def _append_common_eval_args(cmd: list[str], *, args: argparse.Namespace, eval_max_items: int) -> None:
    cmd.extend(["--device", str(args.device)])
    if int(eval_max_items) > 0:
        cmd.extend(["--max-items", str(int(eval_max_items))])
    if bool(args.overwrite):
        cmd.append("--overwrite")


def _run_prediction_eval(
    *,
    predictions_dir: Path,
    out_dir: Path,
    args: argparse.Namespace,
    eval_max_items: int,
    command_rows: list[dict[str, Any]],
) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "evaluate_diffusion_predictions.py"),
        "--cache-root",
        str(Path(args.cache_root).expanduser().resolve()),
        "--split",
        str(args.split),
        "--predictions-dir",
        str(predictions_dir),
        "--out-dir",
        str(out_dir),
    ]
    _append_common_eval_args(cmd, args=args, eval_max_items=eval_max_items)
    command_rows.append(_run_command(cmd, dry_run=bool(args.dry_run)))


def _run_control_eval(
    *,
    predictions_dir: Path,
    out_dir: Path,
    args: argparse.Namespace,
    eval_max_items: int,
    command_rows: list[dict[str, Any]],
) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "evaluate_control_faithfulness.py"),
        "--cache-root",
        str(Path(args.cache_root).expanduser().resolve()),
        "--split",
        str(args.split),
        "--predictions-dir",
        str(predictions_dir),
        "--out-dir",
        str(out_dir),
    ]
    if int(eval_max_items) > 0:
        cmd.extend(["--max-items", str(int(eval_max_items))])
    if bool(args.overwrite):
        cmd.append("--overwrite")
    command_rows.append(_run_command(cmd, dry_run=bool(args.dry_run)))


def _export_direct(
    *,
    run_dir: Path,
    predictions_dir: Path,
    ablation: str,
    args: argparse.Namespace,
    command_rows: list[dict[str, Any]],
) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "export_direct_pca_predictions.py"),
        "--run-dir",
        str(run_dir),
        "--cache-root",
        str(Path(args.cache_root).expanduser().resolve()),
        "--split",
        str(args.split),
        "--out-dir",
        str(predictions_dir),
        "--device",
        str(args.device),
        "--batch-size",
        str(int(args.direct_batch_size)),
        "--num-workers",
        str(int(args.num_workers)),
        "--conditioning-ablation",
        str(ablation),
    ]
    if int(args.max_items) > 0:
        cmd.extend(["--max-items", str(int(args.max_items))])
    if bool(args.overwrite):
        cmd.append("--overwrite")
    command_rows.append(_run_command(cmd, dry_run=bool(args.dry_run)))


def _export_dac_ce(
    *,
    run_dir: Path,
    predictions_dir: Path,
    ablation: str,
    args: argparse.Namespace,
    command_rows: list[dict[str, Any]],
) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "export_best_diffusion_predictions.py"),
        "--train-dir",
        str(run_dir),
        "--cache-root",
        str(Path(args.cache_root).expanduser().resolve()),
        "--split",
        str(args.split),
        "--out-dir",
        str(predictions_dir),
        "--device",
        str(args.device),
        "--batch-size",
        str(int(args.dac_batch_size)),
        "--num-workers",
        str(int(args.num_workers)),
        "--guidance-scale",
        str(float(args.guidance_scale)),
        "--sample-seed",
        str(int(args.sample_seed)),
        "--conditioning-ablation",
        str(ablation),
    ]
    if int(args.max_items) > 0:
        cmd.extend(["--max-items", str(int(args.max_items))])
    if bool(args.use_bpm_inference_geometry):
        cmd.append("--use-bpm-inference-geometry")
    if bool(args.overwrite):
        cmd.append("--overwrite")
    command_rows.append(_run_command(cmd, dry_run=bool(args.dry_run)))


def main() -> None:
    args = _parse_args()
    out_root = Path(args.out_root).expanduser().resolve()
    ablations = _parse_ablations(str(args.ablations))
    direct_runs = [] if bool(args.skip_direct) else _resolve_runs(args.direct_runs, DEFAULT_DIRECT_RUNS)
    dac_ce_runs = [] if bool(args.skip_dac_ce) else _resolve_runs(args.dac_ce_runs, _default_dac_ce_runs())
    if not direct_runs and not dac_ce_runs:
        raise RuntimeError("no runs selected")

    eval_max_items = int(args.max_items) if int(args.eval_max_items) < 0 else int(args.eval_max_items)
    command_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []

    specs: list[tuple[str, Path]] = []
    specs.extend(("direct", path) for path in direct_runs)
    specs.extend(("dac_ce", path) for path in dac_ce_runs)

    for family, run_dir in specs:
        for ablation in ablations:
            run_name = run_dir.name
            predictions_dir = out_root / "predictions" / family / run_name / ablation
            direct_eval_dir = out_root / "direct_audio_eval" / family / run_name / ablation
            control_eval_dir = out_root / "control_faithfulness" / family / run_name / ablation
            if not bool(args.skip_export):
                if family == "direct":
                    _export_direct(
                        run_dir=run_dir,
                        predictions_dir=predictions_dir,
                        ablation=ablation,
                        args=args,
                        command_rows=command_rows,
                    )
                else:
                    _export_dac_ce(
                        run_dir=run_dir,
                        predictions_dir=predictions_dir,
                        ablation=ablation,
                        args=args,
                        command_rows=command_rows,
                    )
            if not bool(args.skip_direct_audio_eval):
                _run_prediction_eval(
                    predictions_dir=predictions_dir,
                    out_dir=direct_eval_dir,
                    args=args,
                    eval_max_items=eval_max_items,
                    command_rows=command_rows,
                )
            if not bool(args.skip_control_faithfulness):
                _run_control_eval(
                    predictions_dir=predictions_dir,
                    out_dir=control_eval_dir,
                    args=args,
                    eval_max_items=eval_max_items,
                    command_rows=command_rows,
                )
            result_rows.append(
                {
                    "family": family,
                    "run_name": run_name,
                    "run_dir": str(run_dir),
                    "ablation": ablation,
                    "predictions_dir": str(predictions_dir),
                    "prediction_summary": _read_json(predictions_dir / "summary.json"),
                    "direct_audio_eval_dir": str(direct_eval_dir),
                    "direct_audio_summary": _read_json(direct_eval_dir / "summary.json"),
                    "control_faithfulness_dir": str(control_eval_dir),
                    "control_faithfulness_summary": _read_json(control_eval_dir / "summary.json"),
                }
            )

    out_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_root / "commands.jsonl", command_rows)
    write_json(
        out_root / "suite_summary.json",
        {
            "cache_root": str(Path(args.cache_root).expanduser().resolve()),
            "split": str(args.split),
            "out_root": str(out_root),
            "ablations": list(ablations),
            "max_items": int(args.max_items),
            "eval_max_items": int(eval_max_items),
            "sample_seed": int(args.sample_seed),
            "num_results": int(len(result_rows)),
            "results": result_rows,
        },
    )
    print(f"wrote frontend ablation suite data to {out_root}")


if __name__ == "__main__":
    main()
