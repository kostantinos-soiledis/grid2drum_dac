#!/usr/bin/env python3
"""Paired significance tests with recording-level clustering.

Reads a per-clip acoustic-metrics CSV (one row per system x clip, with a
``source_id`` recording column) and, for each requested system pair and metric,
reports the paired mean difference (A - B) with both a naive per-clip analysis
and a recording-clustered analysis:

  * bootstrap 95% CI   (naive: resample clips; clustered: resample source_ids)
  * two-sided sign-flip permutation p-value
        (naive: flip each clip's paired-difference sign independently;
         clustered: flip the sign of all clips in a recording together)

Clips are paired across the two systems by ``dataset_index``. The clustered
analysis is the honest one when many clips come from few recordings; the naive
column is printed only to show how much it under-states the interval.

Training/model-free: consumes existing CSVs, no GPU, no re-inference.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _system_table(
    rows: list[dict[str, str]],
    *,
    model: str,
    key_col: str,
    cluster_col: str,
    metrics: list[str],
) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for row in rows:
        if str(row.get("model", "")) != str(model):
            continue
        key = str(row.get(key_col, "")).strip()
        if not key:
            continue
        entry: dict[str, Any] = {"cluster": str(row.get(cluster_col, "")).strip()}
        ok = True
        for metric in metrics:
            raw = str(row.get(metric, "")).strip()
            try:
                entry[metric] = float(raw)
            except ValueError:
                ok = False
                break
        if ok:
            table[key] = entry
    return table


def _percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    lo = float(np.percentile(samples, 100.0 * alpha / 2.0))
    hi = float(np.percentile(samples, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def _analyze_metric(
    diffs: np.ndarray,
    clusters: np.ndarray,
    *,
    reps: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    n = int(diffs.shape[0])
    observed = float(np.mean(diffs))

    # --- naive: clips are the unit -------------------------------------------
    boot_naive = np.empty(reps, dtype=np.float64)
    for i in range(reps):
        idx = rng.integers(0, n, size=n)
        boot_naive[i] = float(np.mean(diffs[idx]))
    naive_lo, naive_hi = _percentile_ci(boot_naive)

    signs = rng.integers(0, 2, size=(reps, n)) * 2 - 1
    perm_naive = np.abs((signs * diffs[None, :]).mean(axis=1))
    p_naive = float((np.count_nonzero(perm_naive >= abs(observed) - 1e-12) + 1) / (reps + 1))

    # --- clustered: recordings are the unit ----------------------------------
    uniq = list(dict.fromkeys(clusters.tolist()))
    cluster_to_idx = {c: np.where(clusters == c)[0] for c in uniq}
    n_clusters = len(uniq)

    boot_cluster = np.empty(reps, dtype=np.float64)
    for i in range(reps):
        chosen = rng.choice(n_clusters, size=n_clusters, replace=True)
        picked = np.concatenate([cluster_to_idx[uniq[c]] for c in chosen])
        boot_cluster[i] = float(np.mean(diffs[picked]))
    cl_lo, cl_hi = _percentile_ci(boot_cluster)

    # cluster-level sign flip: every clip in a recording shares the flip
    cluster_means = np.array([float(np.mean(diffs[cluster_to_idx[c]])) for c in uniq])
    cluster_sizes = np.array([int(cluster_to_idx[c].shape[0]) for c in uniq], dtype=np.float64)
    total = float(cluster_sizes.sum())
    csign = rng.integers(0, 2, size=(reps, n_clusters)) * 2 - 1
    perm_cluster = np.abs((csign * (cluster_means * cluster_sizes)[None, :]).sum(axis=1) / total)
    p_cluster = float((np.count_nonzero(perm_cluster >= abs(observed) - 1e-12) + 1) / (reps + 1))

    naive_w = naive_hi - naive_lo
    cl_w = cl_hi - cl_lo
    return {
        "mean_diff": observed,
        "n_clips": n,
        "n_clusters": n_clusters,
        "naive_ci95": [naive_lo, naive_hi],
        "clustered_ci95": [cl_lo, cl_hi],
        "ci_width_inflation": (cl_w / naive_w) if naive_w > 0 else math.nan,
        "naive_signflip_p": p_naive,
        "clustered_signflip_p": p_cluster,
        "naive_resolved": not (naive_lo <= 0.0 <= naive_hi),
        "clustered_resolved": not (cl_lo <= 0.0 <= cl_hi),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-clip", required=True, type=str, help="per_clip_metrics.csv path")
    ap.add_argument("--out", required=True, type=str)
    ap.add_argument("--cluster-col", default="source_id", type=str)
    ap.add_argument("--key-col", default="dataset_index", type=str)
    ap.add_argument("--pairs", nargs="+", required=True, help='space-separated "A:B" model-name pairs (diff = A - B)')
    ap.add_argument("--metrics", nargs="+", required=True)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rows = _load_rows(Path(args.per_clip))
    models = sorted({str(r.get("model", "")) for r in rows})
    rng = np.random.default_rng(int(args.seed))

    results: dict[str, Any] = {
        "per_clip_csv": str(args.per_clip),
        "cluster_col": str(args.cluster_col),
        "reps": int(args.reps),
        "seed": int(args.seed),
        "available_models": models,
        "pairs": {},
    }

    for pair in args.pairs:
        if ":" not in pair:
            results["pairs"][pair] = {"error": "pair must be 'A:B'"}
            continue
        a_name, b_name = pair.split(":", 1)
        tab_a = _system_table(rows, model=a_name, key_col=args.key_col, cluster_col=args.cluster_col, metrics=args.metrics)
        tab_b = _system_table(rows, model=b_name, key_col=args.key_col, cluster_col=args.cluster_col, metrics=args.metrics)
        shared = [k for k in tab_a if k in tab_b]
        if not shared:
            results["pairs"][pair] = {
                "error": "no shared clips (check model names)",
                "a_present": len(tab_a),
                "b_present": len(tab_b),
            }
            continue
        clusters = np.array([tab_a[k]["cluster"] for k in shared])
        pair_out: dict[str, Any] = {"n_shared_clips": len(shared), "metrics": {}}
        for metric in args.metrics:
            diffs = np.array([tab_a[k][metric] - tab_b[k][metric] for k in shared], dtype=np.float64)
            pair_out["metrics"][metric] = _analyze_metric(diffs, clusters, reps=int(args.reps), rng=rng)
        results["pairs"][pair] = pair_out

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    # human-readable digest to stdout
    for pair, po in results["pairs"].items():
        print(f"\n=== {pair}  (diff = A - B) ===")
        if "error" in po:
            print("  ERROR:", po["error"], {k: v for k, v in po.items() if k != "error"})
            continue
        print(f"  shared clips: {po['n_shared_clips']}")
        for metric, m in po["metrics"].items():
            print(
                f"  {metric:18s} mean={m['mean_diff']:+.4f} | "
                f"naive CI[{m['naive_ci95'][0]:+.4f},{m['naive_ci95'][1]:+.4f}] p={m['naive_signflip_p']:.4f} "
                f"({'RESOLVED' if m['naive_resolved'] else 'ns'}) | "
                f"CLUSTER CI[{m['clustered_ci95'][0]:+.4f},{m['clustered_ci95'][1]:+.4f}] p={m['clustered_signflip_p']:.4f} "
                f"({'RESOLVED' if m['clustered_resolved'] else 'ns'}) | "
                f"width x{m['ci_width_inflation']:.2f} over {m['n_clusters']} recordings"
            )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
