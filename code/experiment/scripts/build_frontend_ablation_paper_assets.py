#!/usr/bin/env python3
"""Build paper tables, statistics, and figures from completed frontend ablations."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-drum2grid")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INFERENCE_ROOT = REPO_ROOT / "eval" / "frontend_ablation"
DEFAULT_TRAINING_ROOT = REPO_ROOT / "eval" / "frontend_training_ablation"
DEFAULT_OUT_DIR = REPO_ROOT / "paper_results"
DEFAULT_FIGURE_DIR = REPO_ROOT / "figures" / "ablation"

CONDITION_LABELS = {
    "none": "Intact",
    "zero": "Zeroed",
    "shuffle": "Shuffled",
    "phase_shift": "Half-window shift",
}
MODEL_LABELS = {
    ("direct", "direct_pca_d1024_l6_seed1234"): "Direct",
    ("dac_ce", "dac_6steps"): "RVQ-CE, 6 steps",
    ("dac_ce", "dac_12steps"): "RVQ-CE, 12 steps",
    ("dac_ce", "dac_25steps"): "RVQ-CE, 25 steps",
}
TRAINING_LABELS = {
    "no_conditioning_zero": "No conditioning",
    "naive_r0_linear": "Linear, radius 0",
    "naive_r22_linear": "Linear, radius 22",
    "our_hybrid_multiscale": "Hybrid multiscale",
    "no_conditioning_zero_25steps": "No conditioning",
    "our_hybrid_multiscale_25steps": "Hybrid multiscale",
}
TRAINING_REFERENCES = {
    "direct": "our_hybrid_multiscale",
    "dac25": "our_hybrid_multiscale_25steps",
}
CONTROL_GROUPS = ("kick", "snare", "tom", "hihat", "cymbal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-root", type=str, default=str(DEFAULT_INFERENCE_ROOT))
    parser.add_argument("--training-root", type=str, default=str(DEFAULT_TRAINING_ROOT))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--figure-dir", type=str, default=str(DEFAULT_FIGURE_DIR))
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--permutation-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return dict(json.loads(path.read_text(encoding="utf-8")))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _inference_rows(root: Path) -> list[dict[str, Any]]:
    summary = read_json(root / "suite_summary.json")
    rows: list[dict[str, Any]] = []
    for result in summary.get("results", []):
        family = str(result["family"])
        run_name = str(result["run_name"])
        condition = str(result["ablation"])
        audio = dict(result["direct_audio_summary"])
        control = dict(result["control_faithfulness_summary"])
        rows.append(
            {
                "suite": "inference",
                "family": family,
                "model": run_name,
                "display_name": MODEL_LABELS[(family, run_name)],
                "condition": condition,
                "condition_label": CONDITION_LABELS[condition],
                "num_examples": int(audio["num_examples"]),
                "mrstft_logmag_l1": float(audio["mrstft_logmag_l1_mean"]),
                "audio_l1": float(audio["audio_l1_mean"]),
                "proxy_macro_f1": float(control["macro_f1"]),
                "proxy_macro_precision": float(control["macro_precision"]),
                "proxy_macro_recall": float(control["macro_recall"]),
                "audio_csv": str(Path(result["direct_audio_eval_dir"]) / "per_clip_metrics.csv"),
                "control_csv": str(Path(result["control_faithfulness_dir"]) / "per_clip_control_metrics.csv"),
            }
        )
    expected = len(MODEL_LABELS) * len(CONDITION_LABELS)
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} inference rows, found {len(rows)}")
    return rows


def _training_rows(root: Path) -> list[dict[str, Any]]:
    summary = read_json(root / "suite_summary.json")
    rows: list[dict[str, Any]] = []
    for result in summary.get("results", []):
        family = str(result["family"])
        variant = str(result["variant"])
        audio = dict(result["direct_audio_summary"])
        control = dict(result["control_faithfulness_summary"])
        prediction = dict(result["prediction_summary"])
        requested_epochs = int(result["requested_training_epochs"])
        run_dir = Path(str(result["run_dir"]))
        history_path = run_dir / "history.csv"
        history = pd.read_csv(history_path)
        epochs = history["epoch"].astype(int).tolist()
        if epochs[:requested_epochs] != list(range(requested_epochs)):
            raise RuntimeError(
                f"{family}/{variant} does not contain epochs 0--{requested_epochs - 1}"
            )
        used_existing_run = bool(result["used_existing_run"])
        if not used_existing_run and len(epochs) != requested_epochs:
            raise RuntimeError(
                f"{family}/{variant} has {len(epochs)} epochs, expected {requested_epochs}"
            )
        if used_existing_run and int(prediction["checkpoint_epoch"]) >= requested_epochs:
            raise RuntimeError(
                f"{family}/{variant} selected checkpoint lies outside the ablation budget"
            )
        rows.append(
            {
                "suite": "training",
                "family": family,
                "variant": variant,
                "display_name": TRAINING_LABELS[variant],
                "num_examples": int(audio["num_examples"]),
                "mrstft_logmag_l1": float(audio["mrstft_logmag_l1_mean"]),
                "audio_l1": float(audio["audio_l1_mean"]),
                "proxy_macro_f1": float(control["macro_f1"]),
                "proxy_macro_precision": float(control["macro_precision"]),
                "proxy_macro_recall": float(control["macro_recall"]),
                "num_parameters": int(prediction["num_parameters"]),
                "rtf_end_to_end": float(prediction["rtf_end_to_end"]),
                "checkpoint_epoch": int(prediction["checkpoint_epoch"]),
                "best_val_loss": float(prediction["best_val_loss"]),
                "requested_training_epochs": requested_epochs,
                "audio_csv": str(
                    root / "direct_audio_eval" / family / variant / "per_clip_metrics.csv"
                ),
                "control_csv": str(
                    root
                    / "control_faithfulness"
                    / family
                    / variant
                    / "per_clip_control_metrics.csv"
                ),
            }
        )
    if len(rows) != 6:
        raise RuntimeError(f"expected 6 training rows, found {len(rows)}")
    return rows


def _validate_rows(rows: Iterable[Mapping[str, Any]]) -> None:
    for row in rows:
        if int(row["num_examples"]) != 1733:
            raise RuntimeError(f"incomplete row: {row}")
        for key in ("audio_csv", "control_csv"):
            if not Path(str(row[key])).is_file():
                raise FileNotFoundError(str(row[key]))


def _audio_values(row: Mapping[str, Any], metric: str) -> pd.Series:
    frame = pd.read_csv(str(row["audio_csv"]))
    if frame["dataset_index"].duplicated().any():
        raise RuntimeError(f"duplicate dataset indices in {row['audio_csv']}")
    return frame.set_index("dataset_index")[metric].astype(float).sort_index()


def _control_arrays(row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(str(row["control_csv"]))
    indices = np.asarray(sorted(frame["dataset_index"].unique()), dtype=np.int64)
    shape = (len(indices), len(CONTROL_GROUPS))
    tp = np.zeros(shape, dtype=np.float64)
    fp = np.zeros(shape, dtype=np.float64)
    fn = np.zeros(shape, dtype=np.float64)
    index_pos = {int(value): pos for pos, value in enumerate(indices)}
    group_pos = {name: pos for pos, name in enumerate(CONTROL_GROUPS)}
    for record in frame.itertuples(index=False):
        group = str(record.group)
        if group not in group_pos:
            continue
        row_pos = index_pos[int(record.dataset_index)]
        col_pos = group_pos[group]
        tp[row_pos, col_pos] += float(record.tp)
        fp[row_pos, col_pos] += float(record.fp)
        fn[row_pos, col_pos] += float(record.fn)
    return indices, tp, fp, fn


def _macro_f1(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> float:
    denom = (2.0 * tp) + fp + fn
    f1 = np.divide(2.0 * tp, denom, out=np.zeros_like(tp), where=denom > 0.0)
    return float(np.mean(f1))


def _percentile_interval(values: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def _paired_audio_stats(
    reference: Mapping[str, Any],
    comparison: Mapping[str, Any],
    metric: str,
    *,
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
) -> dict[str, Any]:
    reference_values = _audio_values(reference, metric)
    comparison_values = _audio_values(comparison, metric)
    common = reference_values.index.intersection(comparison_values.index)
    if len(common) != 1733:
        raise RuntimeError(f"expected 1733 paired clips, found {len(common)}")
    delta = (
        comparison_values.loc[common].to_numpy(dtype=np.float64)
        - reference_values.loc[common].to_numpy(dtype=np.float64)
    )
    rng = np.random.default_rng(seed)
    bootstrap_index = rng.integers(
        0,
        len(delta),
        size=(bootstrap_samples, len(delta)),
        dtype=np.int32,
    )
    bootstrap_delta = np.mean(delta[bootstrap_index], axis=1)
    ci_low, ci_high = _percentile_interval(bootstrap_delta)
    signs = (rng.integers(0, 2, size=(permutation_samples, len(delta)), dtype=np.int8) * 2) - 1
    permuted = np.mean(delta[None, :] * signs, axis=1)
    observed = abs(float(np.mean(delta)))
    p_value = float((np.count_nonzero(np.abs(permuted) >= observed) + 1) / (len(permuted) + 1))
    return {
        "metric": metric,
        "num_pairs": int(len(delta)),
        "reference_mean": float(reference_values.loc[common].mean()),
        "comparison_mean": float(comparison_values.loc[common].mean()),
        "delta_comparison_minus_reference": float(np.mean(delta)),
        "delta_ci_low": ci_low,
        "delta_ci_high": ci_high,
        "p_value": p_value,
    }


def _paired_control_stats(
    reference: Mapping[str, Any],
    comparison: Mapping[str, Any],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    ref_indices, ref_tp, ref_fp, ref_fn = _control_arrays(reference)
    cmp_indices, cmp_tp, cmp_fp, cmp_fn = _control_arrays(comparison)
    if not np.array_equal(ref_indices, cmp_indices) or len(ref_indices) != 1733:
        raise RuntimeError("control rows do not share all 1733 dataset indices")
    rng = np.random.default_rng(seed)
    deltas = np.zeros((bootstrap_samples,), dtype=np.float64)
    for sample_idx in range(bootstrap_samples):
        selected = rng.integers(0, len(ref_indices), size=len(ref_indices), dtype=np.int32)
        ref_f1 = _macro_f1(
            ref_tp[selected].sum(axis=0),
            ref_fp[selected].sum(axis=0),
            ref_fn[selected].sum(axis=0),
        )
        cmp_f1 = _macro_f1(
            cmp_tp[selected].sum(axis=0),
            cmp_fp[selected].sum(axis=0),
            cmp_fn[selected].sum(axis=0),
        )
        deltas[sample_idx] = cmp_f1 - ref_f1
    ci_low, ci_high = _percentile_interval(deltas)
    return {
        "metric": "proxy_macro_f1",
        "num_pairs": int(len(ref_indices)),
        "reference_mean": _macro_f1(ref_tp.sum(axis=0), ref_fp.sum(axis=0), ref_fn.sum(axis=0)),
        "comparison_mean": _macro_f1(cmp_tp.sum(axis=0), cmp_fp.sum(axis=0), cmp_fn.sum(axis=0)),
        "delta_comparison_minus_reference": float(np.mean(deltas)),
        "delta_ci_low": ci_low,
        "delta_ci_high": ci_high,
        "p_value": math.nan,
    }


def _holm_adjust(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str, str], list[int]] = {}
    for idx, row in enumerate(rows):
        p_value = float(row["p_value"])
        if not math.isfinite(p_value):
            continue
        key = (str(row["suite"]), str(row["reference_id"]), str(row["metric"]))
        groups.setdefault(key, []).append(idx)
    for indices in groups.values():
        ordered = sorted(indices, key=lambda idx: float(rows[idx]["p_value"]))
        running = 0.0
        count = len(ordered)
        for rank, idx in enumerate(ordered):
            adjusted = min(1.0, float(rows[idx]["p_value"]) * float(count - rank))
            running = max(running, adjusted)
            rows[idx]["p_value_holm"] = running
    for row in rows:
        row.setdefault("p_value_holm", math.nan)


def _pairwise_rows(
    inference_rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    def add_comparison(
        suite: str,
        reference: dict[str, Any],
        comparison: dict[str, Any],
        reference_id: str,
        comparison_id: str,
        comparison_label: str,
    ) -> None:
        for metric in ("mrstft_logmag_l1", "audio_l1"):
            stats = _paired_audio_stats(
                reference,
                comparison,
                metric,
                bootstrap_samples=bootstrap_samples,
                permutation_samples=permutation_samples,
                seed=seed + len(output),
            )
            output.append(
                {
                    "suite": suite,
                    "reference_id": reference_id,
                    "comparison_id": comparison_id,
                    "comparison_label": comparison_label,
                    **stats,
                }
            )
        control = _paired_control_stats(
            reference,
            comparison,
            bootstrap_samples=bootstrap_samples,
            seed=seed + len(output),
        )
        output.append(
            {
                "suite": suite,
                "reference_id": reference_id,
                "comparison_id": comparison_id,
                "comparison_label": comparison_label,
                **control,
            }
        )

    inference_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in inference_rows:
        inference_groups.setdefault((row["family"], row["model"]), []).append(row)
    for (family, model), rows in inference_groups.items():
        reference = next(row for row in rows if row["condition"] == "none")
        reference_id = f"{family}:{model}:none"
        for condition in ("zero", "shuffle", "phase_shift"):
            comparison = next(row for row in rows if row["condition"] == condition)
            add_comparison(
                "inference",
                reference,
                comparison,
                reference_id,
                f"{family}:{model}:{condition}",
                CONDITION_LABELS[condition],
            )

    training_groups: dict[str, list[dict[str, Any]]] = {}
    for row in training_rows:
        training_groups.setdefault(row["family"], []).append(row)
    for family, rows in training_groups.items():
        reference_variant = TRAINING_REFERENCES[family]
        reference = next(row for row in rows if row["variant"] == reference_variant)
        for comparison in rows:
            if comparison["variant"] == reference_variant:
                continue
            add_comparison(
                "training",
                reference,
                comparison,
                f"{family}:{reference_variant}",
                f"{family}:{comparison['variant']}",
                str(comparison["display_name"]),
            )
    _holm_adjust(output)
    return output


def _add_relative_columns(rows: list[dict[str, Any]], key_fields: tuple[str, ...], reference_field: str) -> None:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in key_fields), []).append(row)
    for group_rows in grouped.values():
        if reference_field == "condition":
            reference = next(row for row in group_rows if row["condition"] == "none")
        else:
            family = str(group_rows[0]["family"])
            reference = next(
                row for row in group_rows if row["variant"] == TRAINING_REFERENCES[family]
            )
        for row in group_rows:
            for metric in ("mrstft_logmag_l1", "audio_l1", "proxy_macro_f1"):
                reference_value = float(reference[metric])
                row[f"{metric}_delta"] = float(row[metric]) - reference_value
                row[f"{metric}_ratio"] = float(row[metric]) / reference_value


def _format_value(value: float, digits: int = 3, *, bold: bool = False) -> str:
    text = f"{value:.{digits}f}"
    return rf"\textbf{{{text}}}" if bold else text


def _training_table(rows: list[dict[str, Any]]) -> str:
    ordered = [
        next(row for row in rows if row["variant"] == variant)
        for variant in (
            "no_conditioning_zero",
            "naive_r0_linear",
            "naive_r22_linear",
            "our_hybrid_multiscale",
            "no_conditioning_zero_25steps",
            "our_hybrid_multiscale_25steps",
        )
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Model & Training condition / frontend & Params (M) & MRSTFT $\downarrow$ & Audio $L_1$ $\downarrow$ & Proxy F1 $\uparrow$ & RTF $\downarrow$ \\",
        r"\midrule",
    ]
    for family in ("direct", "dac25"):
        family_rows = [row for row in ordered if row["family"] == family]
        best_mrstft = min(float(row["mrstft_logmag_l1"]) for row in family_rows)
        best_audio = min(float(row["audio_l1"]) for row in family_rows)
        best_f1 = max(float(row["proxy_macro_f1"]) for row in family_rows)
        best_rtf = min(float(row["rtf_end_to_end"]) for row in family_rows)
        family_name = "Direct PCA" if family == "direct" else "RVQ-CE diffusion, 25 steps"
        for row_idx, row in enumerate(family_rows):
            lines.append(
                " & ".join(
                    [
                        latex_escape(family_name if row_idx == 0 else ""),
                        latex_escape(row["display_name"]),
                        f"{float(row['num_parameters']) / 1.0e6:.1f}",
                        _format_value(
                            float(row["mrstft_logmag_l1"]),
                            bold=math.isclose(float(row["mrstft_logmag_l1"]), best_mrstft),
                        ),
                        _format_value(
                            float(row["audio_l1"]),
                            bold=math.isclose(float(row["audio_l1"]), best_audio),
                        ),
                        _format_value(
                            float(row["proxy_macro_f1"]),
                            bold=math.isclose(float(row["proxy_macro_f1"]), best_f1),
                        ),
                        _format_value(
                            float(row["rtf_end_to_end"]),
                            bold=math.isclose(float(row["rtf_end_to_end"]), best_rtf),
                        ),
                    ]
                )
                + r" \\"
            )
        if family == "direct":
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Training-time frontend ablations on the same 1,733 test clips. "
                r"All newly trained variants use a 75-epoch budget and validation-loss checkpoint selection. "
                r"The direct hybrid row reuses the main-run checkpoint selected at epoch 14, which lies within "
                r"that budget. Proxy F1 is produced by the fixed heuristic band-flux onset detector and is not "
                r"a perceptual score. Bold marks the best value within each model family.}"
            ),
            r"\label{tab:frontend_training_ablation}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def _inference_table(rows: list[dict[str, Any]]) -> str:
    ordered: list[dict[str, Any]] = []
    for key in MODEL_LABELS:
        model_rows = [row for row in rows if (row["family"], row["model"]) == key]
        ordered.extend(
            next(row for row in model_rows if row["condition"] == condition)
            for condition in ("none", "zero", "shuffle", "phase_shift")
        )
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Model & Test-time condition & MRSTFT $\downarrow$ & Audio $L_1$ $\downarrow$ & Proxy precision $\uparrow$ & Proxy recall $\uparrow$ & Proxy F1 $\uparrow$ \\",
        r"\midrule",
    ]
    previous_model = ""
    for row in ordered:
        model = str(row["display_name"])
        if previous_model and model != previous_model:
            lines.append(r"\midrule")
        lines.append(
            " & ".join(
                [
                    latex_escape(model if model != previous_model else ""),
                    latex_escape(row["condition_label"]),
                    f"{float(row['mrstft_logmag_l1']):.3f}",
                    f"{float(row['audio_l1']):.3f}",
                    f"{float(row['proxy_macro_precision']):.3f}",
                    f"{float(row['proxy_macro_recall']):.3f}",
                    f"{float(row['proxy_macro_f1']):.3f}",
                ]
            )
            + r" \\"
        )
        previous_model = model
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Complete test-time conditioning sensitivity results. Zeroed conditioning removes numeric "
                r"grid values, articulation IDs, and family-onset fields; shuffled conditioning cyclically reassigns "
                r"conditioning examples within each evaluation batch; the phase condition circularly shifts each valid "
                r"symbolic window by half its length. Targets remain unchanged.}"
            ),
            r"\label{tab:frontend_inference_ablation_full}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def _plot_inference(rows: list[dict[str, Any]], figure_dir: Path) -> None:
    condition_order = ("none", "zero", "shuffle", "phase_shift")
    x = np.arange(len(condition_order))
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    markers = ("o", "s", "^", "D")
    metrics = (
        ("mrstft_logmag_l1", "MRSTFT log-magnitude $L_1$", False),
        ("audio_l1", "Waveform $L_1$", True),
        ("proxy_macro_f1", "Proxy macro F1", False),
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.15))
    for model_idx, ((family, model), label) in enumerate(MODEL_LABELS.items()):
        model_rows = [row for row in rows if row["family"] == family and row["model"] == model]
        for axis, (metric, ylabel, log_scale) in zip(axes, metrics):
            values = [
                float(next(row for row in model_rows if row["condition"] == condition)[metric])
                for condition in condition_order
            ]
            axis.plot(
                x,
                values,
                color=colors[model_idx],
                marker=markers[model_idx],
                linewidth=1.7,
                markersize=5.0,
                label=label,
            )
            axis.set_ylabel(ylabel)
            axis.set_xticks(x, [CONDITION_LABELS[item] for item in condition_order], rotation=20)
            axis.grid(axis="y", color="#d9d9d9", linewidth=0.7)
            if log_scale:
                axis.set_yscale("log")
    axes[0].set_title("(a) Spectral error")
    axes[1].set_title("(b) Sample error")
    axes[2].set_title("(c) Onset proxy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "frontend_conditioning_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "frontend_conditioning_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    inference_root = Path(args.inference_root).expanduser().resolve()
    training_root = Path(args.training_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    figure_dir = Path(args.figure_dir).expanduser().resolve()

    inference_rows = _inference_rows(inference_root)
    training_rows = _training_rows(training_root)
    _validate_rows([*inference_rows, *training_rows])
    _add_relative_columns(inference_rows, ("family", "model"), "condition")
    _add_relative_columns(training_rows, ("family",), "variant")
    pairwise = _pairwise_rows(
        inference_rows,
        training_rows,
        bootstrap_samples=int(args.bootstrap_samples),
        permutation_samples=int(args.permutation_samples),
        seed=int(args.seed),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(inference_rows).drop(columns=["audio_csv", "control_csv"]).to_csv(
        out_dir / "frontend_ablation_inference.csv",
        index=False,
    )
    pd.DataFrame(training_rows).drop(columns=["audio_csv", "control_csv"]).to_csv(
        out_dir / "frontend_ablation_training.csv",
        index=False,
    )
    pd.DataFrame(pairwise).to_csv(out_dir / "frontend_ablation_pairwise.csv", index=False)
    (out_dir / "frontend_ablation_training_table.tex").write_text(
        _training_table(training_rows),
        encoding="utf-8",
    )
    (out_dir / "frontend_ablation_inference_table.tex").write_text(
        _inference_table(inference_rows),
        encoding="utf-8",
    )
    _plot_inference(inference_rows, figure_dir)
    write_json(
        out_dir / "frontend_ablation_manifest.json",
        {
            "inference_root": display_path(inference_root),
            "training_root": display_path(training_root),
            "inference_summary": display_path(inference_root / "suite_summary.json"),
            "training_summary": display_path(training_root / "suite_summary.json"),
            "num_inference_rows": len(inference_rows),
            "num_training_rows": len(training_rows),
            "num_pairwise_rows": len(pairwise),
            "bootstrap_samples": int(args.bootstrap_samples),
            "permutation_samples": int(args.permutation_samples),
            "seed": int(args.seed),
            "figure_pdf": display_path(
                figure_dir / "frontend_conditioning_sensitivity.pdf"
            ),
        },
    )
    print(f"wrote frontend ablation paper assets to {out_dir} and {figure_dir}")


if __name__ == "__main__":
    main()
