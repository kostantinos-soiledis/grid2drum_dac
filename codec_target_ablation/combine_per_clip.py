#!/usr/bin/env python3
"""Merge this ablation's per-clip metrics with those of the final arms it is tested against.

Both evaluations run run_diffusion_acoustic_eval.py on the same test excerpts and decoded targets, so
rows pair up by dataset_index; the merge refuses to write if an index names different excerpts.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def acoustic_rows(evaluation_dir: Path, models: set[str] | None) -> list[dict[str, str]]:
    rows = read_csv(evaluation_dir / "acoustic_eval" / "per_clip_metrics.csv")
    return [row for row in rows if models is None or row["model"] in models]


def direct_rows(evaluation_dir: Path, models: set[str] | None) -> list[dict[str, str]]:
    rows = []
    for model_dir in sorted((evaluation_dir / "direct_audio_eval").iterdir()):
        if models is None or model_dir.name in models:
            rows += [{"model": model_dir.name, **row} for row in read_csv(model_dir / "per_clip_metrics.csv")]
    return rows


def check_aligned(rows: list[dict[str, str]], name: str) -> None:
    excerpt_by_index: dict[str, tuple[str, str]] = {}
    for row in rows:
        excerpt = (row["source_id"], row["beat_index"])
        if excerpt_by_index.setdefault(row["dataset_index"], excerpt) != excerpt:
            raise RuntimeError(f"{name}: dataset_index {row['dataset_index']} names different excerpts across arms")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evaluation-dir", type=Path, required=True, help="this ablation's evaluation")
    parser.add_argument("--final-evaluation-dir", type=Path, required=True, help="results/final/evaluation")
    parser.add_argument("--final-models", nargs="+", required=True, help="final arms to pair against")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    final_models = set(args.final_models)
    for name, collect in (("acoustic", acoustic_rows), ("direct", direct_rows)):
        rows = collect(args.evaluation_dir, None) + collect(args.final_evaluation_dir, final_models)
        check_aligned(rows, name)
        write_csv(args.out_dir / f"{name}_per_clip.csv", rows)
        print(f"wrote {args.out_dir / f'{name}_per_clip.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
