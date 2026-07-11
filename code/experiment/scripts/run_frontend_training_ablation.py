#!/usr/bin/env python3
"""Train and evaluate staged frontend ablation controls."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT.parent.parent
RUNS_ROOT = PACKAGE_ROOT / "runs"
RESULTS_ROOT = PACKAGE_ROOT / "results"
PYTHON = sys.executable

DEFAULT_CACHE_ROOT = RUNS_ROOT / "mini_cache"
DEFAULT_EVAL_ROOT = RESULTS_ROOT / "eval" / "frontend_training_ablation"
DEFAULT_DIRECT_RUN_ROOT = RUNS_ROOT / "runs_direct" / "frontend_ablation"
DEFAULT_DAC25_RUN_ROOT = RUNS_ROOT / "runs_dac_ce" / "frontend_ablation"
DEFAULT_DIRECT_FULL_RUN = RUNS_ROOT / "runs_direct" / "direct_pca_d1024_l6_seed1234"


@dataclass(frozen=True)
class FrontendVariant:
    name: str
    family: str
    train_ablation: str
    export_ablation: str
    frontend_variant: str
    frontend_radii: str
    frontend_primary_radius: int
    concat_multiscale: bool
    use_existing_run: Path | None = None


DIRECT_VARIANTS: tuple[FrontendVariant, ...] = (
    FrontendVariant(
        name="no_conditioning_zero",
        family="direct",
        train_ablation="zero",
        export_ablation="zero",
        frontend_variant="hybrid",
        frontend_radii="0,22,41,55",
        frontend_primary_radius=22,
        concat_multiscale=True,
    ),
    FrontendVariant(
        name="naive_r0_linear",
        family="direct",
        train_ablation="none",
        export_ablation="none",
        frontend_variant="linear",
        frontend_radii="0",
        frontend_primary_radius=0,
        concat_multiscale=False,
    ),
    FrontendVariant(
        name="naive_r22_linear",
        family="direct",
        train_ablation="none",
        export_ablation="none",
        frontend_variant="linear",
        frontend_radii="22",
        frontend_primary_radius=22,
        concat_multiscale=False,
    ),
    FrontendVariant(
        name="our_hybrid_multiscale",
        family="direct",
        train_ablation="none",
        export_ablation="none",
        frontend_variant="hybrid",
        frontend_radii="0,22,41,55",
        frontend_primary_radius=22,
        concat_multiscale=True,
        use_existing_run=DEFAULT_DIRECT_FULL_RUN,
    ),
)

DAC25_VARIANTS: tuple[FrontendVariant, ...] = (
    FrontendVariant(
        name="no_conditioning_zero_25steps",
        family="dac25",
        train_ablation="zero",
        export_ablation="zero",
        frontend_variant="hybrid",
        frontend_radii="0,22,41,55",
        frontend_primary_radius=22,
        concat_multiscale=True,
    ),
    FrontendVariant(
        name="our_hybrid_multiscale_25steps",
        family="dac25",
        train_ablation="none",
        export_ablation="none",
        frontend_variant="hybrid",
        frontend_radii="0,22,41,55",
        frontend_primary_radius=22,
        concat_multiscale=True,
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=str, default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--eval-root", type=str, default=str(DEFAULT_EVAL_ROOT))
    parser.add_argument("--direct-run-root", type=str, default=str(DEFAULT_DIRECT_RUN_ROOT))
    parser.add_argument("--dac25-run-root", type=str, default=str(DEFAULT_DAC25_RUN_ROOT))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--direct-variants",
        type=str,
        default="no_conditioning_zero,naive_r0_linear,naive_r22_linear,our_hybrid_multiscale",
        help="Comma-separated direct variants, or 'none'.",
    )
    parser.add_argument(
        "--dac25-variants",
        type=str,
        default="no_conditioning_zero_25steps,our_hybrid_multiscale_25steps",
        help="Comma-separated DAC-CE 25-step variants, or 'none'.",
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-direct-audio-eval", action="store_true")
    parser.add_argument("--skip-control-faithfulness", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--eval-max-items", type=int, default=-1)
    parser.add_argument("--direct-epochs", type=int, default=75)
    parser.add_argument("--direct-batch-size", type=int, default=4)
    parser.add_argument("--direct-eval-batch-size", type=int, default=4)
    parser.add_argument("--direct-d-model", type=int, default=1024)
    parser.add_argument("--direct-num-layers", type=int, default=6)
    parser.add_argument("--direct-num-heads", type=int, default=8)
    parser.add_argument("--dac25-epochs", type=int, default=75)
    parser.add_argument("--dac25-batch-size", type=int, default=4)
    parser.add_argument("--dac25-eval-batch-size", type=int, default=4)
    parser.add_argument("--dac25-d-model", type=int, default=768)
    parser.add_argument("--dac25-num-layers", type=int, default=6)
    parser.add_argument("--dac25-num-heads", type=int, default=8)
    parser.add_argument("--sample-seed", type=int, default=1234)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _variant_map(variants: tuple[FrontendVariant, ...]) -> dict[str, FrontendVariant]:
    return {variant.name: variant for variant in variants}


def _select_variants(text: str, variants: tuple[FrontendVariant, ...]) -> list[FrontendVariant]:
    text_eff = str(text).strip()
    if not text_eff or text_eff.lower() == "none":
        return []
    known = _variant_map(variants)
    selected: list[FrontendVariant] = []
    for part in text_eff.split(","):
        name = part.strip()
        if not name:
            continue
        if name not in known:
            raise ValueError(f"unknown variant {name!r}; known={sorted(known)}")
        selected.append(known[name])
    return selected


def _run(cmd: list[str], *, dry_run: bool) -> dict[str, Any]:
    rendered = shlex.join(str(part) for part in cmd)
    print(rendered, flush=True)
    if dry_run:
        return {"command": rendered, "returncode": None, "dry_run": True}
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-drum-rendering")
    completed = subprocess.run([str(part) for part in cmd], cwd=str(REPO_ROOT), env=env, check=True)
    return {"command": rendered, "returncode": int(completed.returncode), "dry_run": False}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return dict(json.load(handle))


def _completed_epochs(run_dir: Path) -> int:
    history_path = run_dir / "history.csv"
    if not history_path.is_file():
        return 0
    with history_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    epochs = [int(row["epoch"]) for row in rows if str(row.get("epoch", "")).strip()]
    return (max(epochs) + 1) if epochs else 0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _append_frontend_args(cmd: list[str], variant: FrontendVariant) -> None:
    cmd.extend(
        [
            "--frontend-variant",
            str(variant.frontend_variant),
            "--frontend-radii",
            str(variant.frontend_radii),
            "--frontend-primary-radius",
            str(int(variant.frontend_primary_radius)),
        ]
    )
    if not bool(variant.concat_multiscale):
        cmd.append("--no-concat-multiscale-frontend")


def _train_direct_cmd(
    args: argparse.Namespace,
    variant: FrontendVariant,
    run_dir: Path,
    *,
    resume: bool,
) -> list[str]:
    cmd = [
        PYTHON,
        str(REPO_ROOT / "standalone_direct_pca_regressor.py"),
        "--cache-root",
        str(Path(args.cache_root).expanduser().resolve()),
        "--out-dir",
        str(run_dir),
        "--device",
        str(args.device),
        "--epochs",
        str(int(args.direct_epochs)),
        "--batch-size",
        str(int(args.direct_batch_size)),
        "--eval-batch-size",
        str(int(args.direct_eval_batch_size)),
        "--num-workers",
        str(int(args.num_workers)),
        "--seed",
        "1234",
        "--d-model",
        str(int(args.direct_d_model)),
        "--num-layers",
        str(int(args.direct_num_layers)),
        "--num-heads",
        str(int(args.direct_num_heads)),
        "--conditioning-ablation",
        str(variant.train_ablation),
        "--export-val-predictions",
        "0",
    ]
    _append_frontend_args(cmd, variant)
    if int(args.max_items) > 0:
        cmd.extend(["--max-train-items", str(int(args.max_items)), "--max-val-items", str(int(args.max_items))])
    if bool(resume):
        cmd.append("--resume")
    if bool(args.overwrite):
        cmd.append("--overwrite")
    return cmd


def _train_dac25_cmd(
    args: argparse.Namespace,
    variant: FrontendVariant,
    run_dir: Path,
    *,
    resume: bool,
) -> list[str]:
    cmd = [
        PYTHON,
        str(REPO_ROOT / "train_cli.py"),
        "--cache-root",
        str(Path(args.cache_root).expanduser().resolve()),
        "--out-dir",
        str(run_dir),
        "--device",
        str(args.device),
        "--epochs",
        str(int(args.dac25_epochs)),
        "--batch-size",
        str(int(args.dac25_batch_size)),
        "--eval-batch-size",
        str(int(args.dac25_eval_batch_size)),
        "--num-workers",
        str(int(args.num_workers)),
        "--seed",
        "1234",
        "--d-model",
        str(int(args.dac25_d_model)),
        "--num-layers",
        str(int(args.dac25_num_layers)),
        "--num-heads",
        str(int(args.dac25_num_heads)),
        "--num-steps",
        "25",
        "--eval-plot-steps",
        "6,12,20",
        "--rvq-ce-weight",
        "0.1",
        "--conditioning-ablation",
        str(variant.train_ablation),
        "--use-bpm-training-geometry",
        "--no-pin-memory",
    ]
    _append_frontend_args(cmd, variant)
    if int(args.max_items) > 0:
        cmd.extend(["--max-train-items", str(int(args.max_items)), "--max-val-items", str(int(args.max_items))])
    if bool(resume):
        cmd.append("--resume")
    if bool(args.overwrite):
        cmd.append("--overwrite")
    return cmd


def _export_cmd(args: argparse.Namespace, variant: FrontendVariant, run_dir: Path, predictions_dir: Path) -> list[str]:
    common = [
        "--cache-root",
        str(Path(args.cache_root).expanduser().resolve()),
        "--split",
        str(args.split),
        "--out-dir",
        str(predictions_dir),
        "--device",
        str(args.device),
        "--num-workers",
        str(int(args.num_workers)),
        "--conditioning-ablation",
        str(variant.export_ablation),
    ]
    if variant.family == "direct":
        cmd = [
            PYTHON,
            str(REPO_ROOT / "scripts" / "export_direct_pca_predictions.py"),
            "--run-dir",
            str(run_dir),
            "--batch-size",
            str(int(args.direct_eval_batch_size)),
            *common,
        ]
    else:
        cmd = [
            PYTHON,
            str(REPO_ROOT / "scripts" / "export_best_diffusion_predictions.py"),
            "--train-dir",
            str(run_dir),
            "--batch-size",
            str(int(args.dac25_eval_batch_size)),
            "--guidance-scale",
            str(float(args.guidance_scale)),
            "--sample-seed",
            str(int(args.sample_seed)),
            *common,
        ]
    if int(args.eval_max_items) > 0:
        cmd.extend(["--max-items", str(int(args.eval_max_items))])
    elif int(args.max_items) > 0:
        cmd.extend(["--max-items", str(int(args.max_items))])
    if bool(args.overwrite):
        cmd.append("--overwrite")
    return cmd


def _direct_audio_eval_cmd(args: argparse.Namespace, predictions_dir: Path, out_dir: Path) -> list[str]:
    cmd = [
        PYTHON,
        str(REPO_ROOT / "scripts" / "evaluate_diffusion_predictions.py"),
        "--cache-root",
        str(Path(args.cache_root).expanduser().resolve()),
        "--split",
        str(args.split),
        "--predictions-dir",
        str(predictions_dir),
        "--out-dir",
        str(out_dir),
        "--device",
        str(args.device),
    ]
    max_items = int(args.eval_max_items) if int(args.eval_max_items) >= 0 else int(args.max_items)
    if int(max_items) > 0:
        cmd.extend(["--max-items", str(int(max_items))])
    if bool(args.overwrite):
        cmd.append("--overwrite")
    return cmd


def _control_eval_cmd(args: argparse.Namespace, predictions_dir: Path, out_dir: Path) -> list[str]:
    cmd = [
        PYTHON,
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
    max_items = int(args.eval_max_items) if int(args.eval_max_items) >= 0 else int(args.max_items)
    if int(max_items) > 0:
        cmd.extend(["--max-items", str(int(max_items))])
    if bool(args.overwrite):
        cmd.append("--overwrite")
    return cmd


def main() -> None:
    args = _parse_args()
    eval_root = Path(args.eval_root).expanduser().resolve()
    direct_run_root = Path(args.direct_run_root).expanduser().resolve()
    dac25_run_root = Path(args.dac25_run_root).expanduser().resolve()
    selected = _select_variants(str(args.direct_variants), DIRECT_VARIANTS)
    selected.extend(_select_variants(str(args.dac25_variants), DAC25_VARIANTS))
    if not selected:
        raise RuntimeError("no variants selected")

    commands: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for variant in selected:
        run_dir = (
            variant.use_existing_run.expanduser().resolve()
            if variant.use_existing_run is not None
            else (direct_run_root if variant.family == "direct" else dac25_run_root) / variant.name
        )
        completed_epochs = _completed_epochs(run_dir)
        requested_epochs = int(args.direct_epochs) if variant.family == "direct" else int(args.dac25_epochs)
        training_action = "reused_existing" if variant.use_existing_run is not None else "not_started"
        if variant.use_existing_run is None and not bool(args.skip_train):
            if int(completed_epochs) >= int(requested_epochs) and not bool(args.overwrite):
                training_action = "reused_complete"
                print(
                    f"reusing completed training run: {run_dir} "
                    f"epochs={completed_epochs}/{requested_epochs}",
                    flush=True,
                )
            else:
                resume = bool(int(completed_epochs) > 0 and not bool(args.overwrite))
                training_action = "resumed" if resume else "trained"
                train_cmd = (
                    _train_direct_cmd(args, variant, run_dir, resume=resume)
                    if variant.family == "direct"
                    else _train_dac25_cmd(args, variant, run_dir, resume=resume)
                )
                commands.append(_run(train_cmd, dry_run=bool(args.dry_run)))
        predictions_dir = eval_root / "predictions" / variant.family / variant.name
        direct_eval_dir = eval_root / "direct_audio_eval" / variant.family / variant.name
        control_eval_dir = eval_root / "control_faithfulness" / variant.family / variant.name
        if not bool(args.skip_export):
            commands.append(_run(_export_cmd(args, variant, run_dir, predictions_dir), dry_run=bool(args.dry_run)))
        if not bool(args.skip_direct_audio_eval):
            commands.append(_run(_direct_audio_eval_cmd(args, predictions_dir, direct_eval_dir), dry_run=bool(args.dry_run)))
        if not bool(args.skip_control_faithfulness):
            commands.append(_run(_control_eval_cmd(args, predictions_dir, control_eval_dir), dry_run=bool(args.dry_run)))
        results.append(
            {
                "variant": variant.name,
                "family": variant.family,
                "run_dir": str(run_dir),
                "train_ablation": variant.train_ablation,
                "export_ablation": variant.export_ablation,
                "frontend_variant": variant.frontend_variant,
                "frontend_radii": variant.frontend_radii,
                "frontend_primary_radius": int(variant.frontend_primary_radius),
                "concat_multiscale": bool(variant.concat_multiscale),
                "used_existing_run": variant.use_existing_run is not None,
                "requested_training_epochs": int(requested_epochs),
                "completed_training_epochs_before_run": int(completed_epochs),
                "training_action": str(training_action),
                "prediction_summary": _read_json(predictions_dir / "summary.json"),
                "direct_audio_summary": _read_json(direct_eval_dir / "summary.json"),
                "control_faithfulness_summary": _read_json(control_eval_dir / "summary.json"),
            }
        )

    _write_jsonl(eval_root / "commands.jsonl", commands)
    _write_json(
        eval_root / "suite_summary.json",
        {
            "cache_root": str(Path(args.cache_root).expanduser().resolve()),
            "eval_root": str(eval_root),
            "split": str(args.split),
            "num_variants": int(len(results)),
            "results": results,
        },
    )
    print(f"wrote frontend training ablation suite data to {eval_root}")


if __name__ == "__main__":
    main()
