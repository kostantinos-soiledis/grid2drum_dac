#!/usr/bin/env python3
"""Aggregate the overnight foundation experiments into paper-ready summaries.

Consumes the artifacts produced by ``scripts/run_overnight_foundation.sh``:

* ``<overnight>/seed_stability/acoustic_eval/overall_summary.csv``
    one row per ``<model>_seed<N>``; grouped into mean +/- std over sampling seeds.
* ``<overnight>/control_faithfulness/<system>/{summary.json,per_clip_control_metrics.csv}``
    per-family onset diagnostic; diffusion systems are grouped over their seeds.

Writes (under ``<overnight>``):
* ``seed_stability_summary.csv``         - mean/std of the unified acoustic metrics
* ``control_faithfulness_summary.csv``   - macro + per-family F1 and timing error
* ``overnight_summary.md``               - human-readable digest

Standard library only; safe to re-run.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any

SEED_METRICS = ["mel_mae_db", "onset_flux_cosine", "band_balance_l1", "fad_inf"]
FAMILIES = ["kick", "snare", "tom", "hihat", "cymbal"]
_SEED_RE = re.compile(r"_seed\d+$")


def _base_name(model: str) -> str:
    return _SEED_RE.sub("", str(model))


def _is_seeded(model: str) -> bool:
    return bool(_SEED_RE.search(str(model)))


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None  # drop NaN


def _mean(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _std(values: list[float]) -> float | None:
    return float(statistics.stdev(values)) if len(values) > 1 else (0.0 if values else None)


def _fmt(value: float | None, decimals: int = 4) -> str:
    return "" if value is None else f"{value:.{decimals}f}"


# --------------------------------------------------------------------------------------
# Stage 1 - sampling-seed stability
# --------------------------------------------------------------------------------------
def aggregate_seed_stability(seed_root: Path) -> list[dict[str, Any]]:
    summary_csv = seed_root / "acoustic_eval" / "overall_summary.csv"
    if not summary_csv.is_file():
        print(f"[seed] missing {summary_csv}; skipping")
        return []
    with summary_csv.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        model = str(row.get("model", ""))
        if not _is_seeded(model):
            continue
        grouped.setdefault(_base_name(model), []).append(row)

    out_rows: list[dict[str, Any]] = []
    for base in sorted(grouped):
        seed_rows = grouped[base]
        record: dict[str, Any] = {"model": base, "n_seeds": len(seed_rows)}
        for metric in SEED_METRICS:
            values = [v for v in (_to_float(r.get(metric)) for r in seed_rows) if v is not None]
            record[f"{metric}_mean"] = _mean(values)
            record[f"{metric}_std"] = _std(values)
        out_rows.append(record)
    return out_rows


# --------------------------------------------------------------------------------------
# Stage 2 - control faithfulness
# --------------------------------------------------------------------------------------
def _timing_ms_by_group(per_clip_csv: Path) -> dict[str, float | None]:
    out: dict[str, list[float]] = {fam: [] for fam in FAMILIES}
    if not per_clip_csv.is_file():
        return {fam: None for fam in FAMILIES}
    with per_clip_csv.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            group = str(row.get("group", ""))
            value = _to_float(row.get("mean_abs_error_sec"))
            if group in out and value is not None:
                out[group].append(value * 1000.0)
    return {fam: _mean(vals) for fam, vals in out.items()}


def _read_control_system(system_dir: Path) -> dict[str, Any] | None:
    summary_path = system_dir / "summary.json"
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    groups = summary.get("groups", {}) or {}
    timing = _timing_ms_by_group(system_dir / "per_clip_control_metrics.csv")
    record: dict[str, Any] = {
        "macro_f1": _to_float(summary.get("macro_f1")),
        "macro_precision": _to_float(summary.get("macro_precision")),
        "macro_recall": _to_float(summary.get("macro_recall")),
    }
    family_timings: list[float] = []
    for fam in FAMILIES:
        record[f"{fam}_f1"] = _to_float((groups.get(fam) or {}).get("f1"))
        record[f"{fam}_timing_ms"] = timing.get(fam)
        if timing.get(fam) is not None:
            family_timings.append(float(timing[fam]))
    record["macro_timing_ms"] = _mean(family_timings)
    return record


def aggregate_control(control_root: Path) -> list[dict[str, Any]]:
    if not control_root.is_dir():
        print(f"[control] missing {control_root}; skipping")
        return []
    per_system: dict[str, dict[str, Any]] = {}
    for system_dir in sorted(control_root.iterdir()):
        if not system_dir.is_dir():
            continue
        record = _read_control_system(system_dir)
        if record is not None:
            per_system[system_dir.name] = record

    grouped: dict[str, list[dict[str, Any]]] = {}
    for name, record in per_system.items():
        grouped.setdefault(_base_name(name), []).append(record)

    metric_keys = ["macro_f1", "macro_precision", "macro_recall", "macro_timing_ms"]
    metric_keys += [f"{fam}_f1" for fam in FAMILIES]
    metric_keys += [f"{fam}_timing_ms" for fam in FAMILIES]

    out_rows: list[dict[str, Any]] = []
    for base in sorted(grouped):
        records = grouped[base]
        row: dict[str, Any] = {"model": base, "n_seeds": len(records)}
        for key in metric_keys:
            values = [v for v in (r.get(key) for r in records) if v is not None]
            row[f"{key}_mean"] = _mean(values)
            if key == "macro_f1":
                row["macro_f1_std"] = _std(values)
        out_rows.append(row)
    return out_rows


# --------------------------------------------------------------------------------------
def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        print(f"[write] no rows for {path.name}; skipping")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fieldnames})
    print(f"[write] {path}  ({len(rows)} rows)")


def _write_markdown(path: Path, seed_rows: list[dict[str, Any]], control_rows: list[dict[str, Any]]) -> None:
    lines: list[str] = ["# Overnight foundation results", ""]
    lines.append("## Sampling-seed stability (unified acoustic pipeline)")
    lines.append("")
    if seed_rows:
        header = ["model", "n_seeds"] + [f"{m} (mean+/-std)" for m in SEED_METRICS]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in seed_rows:
            cells = [str(row["model"]), str(row["n_seeds"])]
            for metric in SEED_METRICS:
                cells.append(f"{_fmt(row.get(f'{metric}_mean'))} +/- {_fmt(row.get(f'{metric}_std'))}")
            lines.append("| " + " | ".join(cells) + " |")
    else:
        lines.append("_no seed-stability artifacts found_")
    lines += ["", "## Control faithfulness (fixed onset diagnostic)", ""]
    if control_rows:
        header = ["model", "n", "macro F1", "macro timing (ms)"] + [f"{fam} F1" for fam in FAMILIES]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in control_rows:
            cells = [
                str(row["model"]),
                str(row["n_seeds"]),
                _fmt(row.get("macro_f1_mean"), 3),
                _fmt(row.get("macro_timing_ms_mean"), 1),
            ]
            for fam in FAMILIES:
                cells.append(_fmt(row.get(f"{fam}_f1_mean"), 3))
            lines.append("| " + " | ".join(cells) + " |")
    else:
        lines.append("_no control-faithfulness artifacts found_")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[write] {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overnight-root", type=str, default="paper_results/overnight")
    parser.add_argument("--seeds", type=str, default="", help="Unused; accepted for orchestrator compatibility.")
    args = parser.parse_args()

    root = Path(args.overnight_root).expanduser().resolve()
    seed_rows = aggregate_seed_stability(root / "seed_stability")
    control_rows = aggregate_control(root / "control_faithfulness")

    _write_csv(root / "seed_stability_summary.csv", seed_rows)
    _write_csv(root / "control_faithfulness_summary.csv", control_rows)
    _write_markdown(root / "overnight_summary.md", seed_rows, control_rows)


if __name__ == "__main__":
    main()
