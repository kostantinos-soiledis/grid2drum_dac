#!/usr/bin/env python3
"""Create small, readable comparison tables from the joint final evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


COMPARISONS = {
    "representation": {
        "question": "Does diffusion work better on native pre-snap DAC latents or post-snap PCA latents?",
        "models": (("presnap_latent", "pre-snap native latent"), ("postsnap_pca", "post-snap PCA")),
    },
    "regression": {
        "question": "Does diffusion improve over deterministic direct regression?",
        "models": (("postsnap_pca", "diffusion"), ("regression_6layer", "direct regression")),
    },
    "capacity": {
        "question": "Is the direct-regression result explained by model capacity?",
        "models": (("regression_6layer", "6-layer regression"), ("regression_8layer_capacity", "8-layer capacity control")),
    },
    "grid_hz": {
        "question": "How sensitive is the system to the conditioning-grid rate?",
        "models": (
            ("grid_90hz", "90 Hz"),
            ("grid_120hz", "120 Hz"),
            ("grid_250hz", "250 Hz"),
            ("grid_500hz", "500 Hz (MIDI re-render)"),
        ),
    },
    "rvq_ce": {
        "title": "RVQ supervision",
        "question": "Does the training-only RVQ cross-entropy term help post-snap PCA diffusion?",
        "models": (("postsnap_pca", "plain diffusion"), ("grid_250hz", "diffusion + RVQ-CE")),
    },
}

COLUMNS = (
    "model", "role", "num_examples", "num_parameters", "mel_mae_db",
    "onset_flux_cosine", "fad_inf", "audio_l1_mean", "mrstft_logmag_l1_mean",
    "rtf_end_to_end",
)

LEGACY_TO_CANONICAL = {
    "postsnap_25steps": "postsnap_pca",
    "presnap_25steps": "presnap_latent",
    "direct_pca_d1024_l6_seed1234": "regression_6layer",
    "direct_pca_d1024_l8_seed1234": "regression_8layer_capacity",
    "grid90hz_25steps": "grid_90hz",
    "grid120hz_25steps": "grid_120hz",
    "grid250hz_25steps": "grid_250hz",
    "grid500hz_25steps": "grid_500hz",
}

RUN_PATHS = {
    "postsnap_pca": "postsnap_pca",
    "presnap_latent": "presnap_latent",
    "regression_6layer": "regression_6layer",
    "regression_8layer_capacity": "regression_8layer_capacity",
    "grid_90hz": "grid_hz/90hz",
    "grid_120hz": "grid_hz/120hz",
    "grid_250hz": "grid_hz/250hz",
    "grid_500hz": "grid_hz/500hz",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def compact(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "—"
    try:
        number = float(text)
    except ValueError:
        return text
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:.2f}M"
    return f"{number:.4g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, default=Path(__file__).resolve().parents[3] / "runs/final")
    args = parser.parse_args()

    evaluation = args.evaluation_dir.expanduser().resolve()
    out = args.out_dir.expanduser().resolve()
    runs = args.runs_dir.expanduser().resolve()
    overall_path = evaluation / "acoustic_eval" / "overall_summary.csv"
    if not overall_path.is_file():
        raise FileNotFoundError(f"missing joint evaluation: {overall_path}")

    by_model: dict[str, dict[str, str]] = {}
    for row in read_csv(overall_path):
        source_model = row["model"]
        model = LEGACY_TO_CANONICAL.get(source_model, source_model)
        by_model[model] = {**row, "model": model, "_source_model": source_model}
        if not str(by_model[model].get("num_parameters") or "").strip() and model in RUN_PATHS:
            run_dir = runs / RUN_PATHS[model]
            for config_name in ("config.json", "run_config.json"):
                config_path = run_dir / config_name
                if not config_path.is_file():
                    continue
                config = json.loads(config_path.read_text(encoding="utf-8"))
                if config.get("num_parameters") is not None:
                    by_model[model]["num_parameters"] = str(config["num_parameters"])
                    break
    direct_rows: list[dict[str, str]] = []
    for model, row in by_model.items():
        source_model = row.get("_source_model", model)
        summary_path = evaluation / "direct_audio_eval" / source_model / "summary.json"
        if summary_path.is_file():
            direct = json.loads(summary_path.read_text(encoding="utf-8"))
            row.update({
                "audio_l1_mean": direct.get("audio_l1_mean", ""),
                "mrstft_logmag_l1_mean": direct.get("mrstft_logmag_l1_mean", ""),
            })
        per_clip = evaluation / "direct_audio_eval" / source_model / "per_clip_metrics.csv"
        if per_clip.is_file():
            direct_rows.extend({"model": model, **clip} for clip in read_csv(per_clip))

    if direct_rows:
        fields: list[str] = []
        for row in direct_rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with (out / "direct_per_clip_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(direct_rows)

    sections: list[str] = [
        "# Final results",
        "",
        "All tables come from one joint evaluation, so every arm uses the same test clips, decoded targets, metric implementation, and FAD configuration.",
        "",
    ]
    if evaluation.name == "grid_hz":
        sections.extend([
            "> **Provisional:** only the retained grid-rate evaluation is available. Its 90/120 Hz arms stopped at 75 epochs while the 250 Hz arm reached 150, and the new MIDI-rerendered 500 Hz arm is not trained yet. The final serial runner completes all four arms and replaces this page with the equal-budget joint evaluation.",
            "",
        ])
    manifest: dict[str, Any] = {"evaluation": str(evaluation), "comparisons": {}}
    for name, spec in COMPARISONS.items():
        rows = []
        for model, role in spec["models"]:
            if model not in by_model:
                continue
            rows.append({**by_model[model], "model": model, "role": role})
        if not rows:
            continue
        write_csv(out / "comparisons" / f"{name}.csv", rows)
        manifest["comparisons"][name] = [row["model"] for row in rows]
        sections.extend([
            f"## {spec.get('title') or name.replace('_', ' ').title()}", "", str(spec["question"]), "",
            "| Arm | Mel MAE ↓ | Onset cosine ↑ | FAD∞ ↓ | Audio L1 ↓ | MR-STFT ↓ | Params |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in rows:
            sections.append(
                "| {role} | {mel} | {onset} | {fad} | {l1} | {stft} | {params} |".format(
                    role=row["role"], mel=compact(row.get("mel_mae_db")),
                    onset=compact(row.get("onset_flux_cosine")), fad=compact(row.get("fad_inf")),
                    l1=compact(row.get("audio_l1_mean")), stft=compact(row.get("mrstft_logmag_l1_mean")),
                    params=compact(row.get("num_parameters")),
                )
            )
        sections.extend([""])

    sections.extend([
        "## Statistical tests", "",
        "Source-recording-clustered bootstrap intervals and sign-flip tests are written to `stats/acoustic.json` and `stats/direct_audio.json` by `run_final_results.sh`.",
        "",
    ])
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("\n".join(sections), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out / 'summary.md'}")


if __name__ == "__main__":
    main()
