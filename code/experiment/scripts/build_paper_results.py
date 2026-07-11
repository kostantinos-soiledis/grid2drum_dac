#!/usr/bin/env python3
"""Aggregate completed run artifacts into publishable CSV/TeX results.

This script is intentionally read-only with respect to model artifacts: it does
not export predictions, run inference, or compute FAD. It gathers whatever has
already been produced by the training/evaluation scripts, normalizes the fields,
flags gaps, and emits standalone LaTeX sections that can be compiled or copied
into the paper.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT.parent.parent
RUNS_ROOT = PACKAGE_ROOT / "runs"
RESULTS_ROOT = PACKAGE_ROOT / "results"
DEFAULT_CACHE_ROOT = RUNS_ROOT / "mini_cache"
DEFAULT_OUT_DIR = RESULTS_ROOT / "paper_results"

CANONICAL_BATCH_EVAL = "full_acoustic_eval"

BATCH_MODEL_ALIASES: dict[str, tuple[str, str]] = {
    "target_dac_recon": ("reconstruction_ceiling", "target_dac_recon"),
    "target_pca_recon": ("reconstruction_ceiling", "target_pca_recon"),
    "grid_render": ("baseline", "grid_render"),
    "source_code_decode": ("baseline", "source_code_decode"),
    "symbolic_nn_train": ("baseline", "symbolic_nn_train"),
    "direct_pca_d1024_l6_seed1234": ("direct_pca", "direct_pca_d1024_l6_seed1234"),
}

METRIC_COLUMNS = (
    "fad_inf",
    "fad_inf_ci_low",
    "fad_inf_ci_high",
    "fad_inf_r2",
    "mel_mae_db",
    "onset_flux_cosine",
    "low_flux_cosine",
    "mid_flux_cosine",
    "high_flux_cosine",
    "band_balance_l1",
    "centroid_mae_hz",
    "rms_mae_db",
    "rms_norm_mae_db",
    "crest_mae_db",
    "mrstft_logmag_l1_mean",
    "audio_l1_mean",
    "rtf_end_to_end",
    "audio_sec_per_sec",
    "total_train_wall_sec",
    "time_to_best_checkpoint_sec",
    "peak_gpu_mem_allocated_mb",
)

CONFIG_COLUMNS = (
    "x_dim",
    "d_model",
    "num_layers",
    "num_heads",
    "frontend_variant",
    "frontend_radii",
    "frontend_embed_dim",
    "positional_encoding",
)

CSV_COLUMNS = (
    "run_id",
    "display_name",
    "family",
    "model",
    "split",
    "num_examples",
    "num_steps",
    "rvq_ce_weight",
    "target_layout",
    "target_dim",
    "num_parameters",
    *CONFIG_COLUMNS,
    "best_checkpoint_epoch",
    *METRIC_COLUMNS,
    "run_dir",
    "eval_dir",
    "source_artifacts",
)

MAIN_TABLE_METRICS = (
    "fad_inf",
    "mel_mae_db",
    "onset_flux_cosine",
    "mrstft_logmag_l1_mean",
    "audio_l1_mean",
    "rtf_end_to_end",
    "num_parameters",
)


@dataclass
class RunRecord:
    run_id: str
    family: str
    model: str
    display_name: str
    data: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)

    def merge(self, values: Mapping[str, Any], *, artifact: str | Path | None = None) -> None:
        for key, value in values.items():
            if value is None:
                continue
            if isinstance(value, str) and value == "":
                continue
            self.data[str(key)] = value
        if artifact is not None:
            text = str(artifact)
            if text not in self.artifacts:
                self.artifacts.append(text)

    def row(self) -> dict[str, Any]:
        out = {key: self.data.get(key, "") for key in CSV_COLUMNS}
        out.update(
            {
                "run_id": self.run_id,
                "display_name": self.display_name,
                "family": self.family,
                "model": self.model,
                "source_artifacts": ";".join(self.artifacts),
            }
        )
        return out


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build normalized CSV/JSON/TeX paper result artifacts from completed runs."
    )
    parser.add_argument("--repo-root", type=str, default=str(RUNS_ROOT))
    parser.add_argument("--cache-root", type=str, default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any table row is missing core publishable metrics.",
    )
    return parser.parse_args(argv)


def read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in columns})


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.12g}"
        return ""
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return value


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        val = float(value)
    else:
        text = str(value).strip()
        if text == "":
            return None
        try:
            val = float(text)
        except ValueError:
            return None
    if not math.isfinite(val):
        return None
    return val


def to_int(value: Any) -> int | None:
    val = to_float(value)
    if val is None:
        return None
    return int(round(val))


def rel(path: str | Path, repo_root: Path) -> str:
    path_obj = Path(path)
    try:
        return str(path_obj.resolve().relative_to(repo_root.resolve()))
    except Exception:
        return str(path)


def run_id_for(family: str, model: str) -> str:
    return f"{family}:{model}"


def family_from_path(path: Path, repo_root: Path) -> str:
    rel_text = rel(path, repo_root)
    if rel_text.startswith("runs_dac_ce/"):
        return "diffusion_pca_rvq_ce"
    if rel_text.startswith("runs_dac/"):
        return "diffusion_pca"
    if rel_text.startswith("runs_direct/"):
        return "direct_pca"
    if rel_text.startswith("runs_baselines/"):
        return "baseline"
    return "unknown"


def canonical_batch_alias(model: str) -> tuple[str, str] | None:
    text = str(model).strip()
    if text in BATCH_MODEL_ALIASES:
        return BATCH_MODEL_ALIASES[text]
    match = re.fullmatch(r"diffusion_pca_(\d+)steps", text)
    if match is not None:
        return "diffusion_pca", f"dac_{int(match.group(1))}steps"
    match = re.fullmatch(r"diffusion_pca_rvq_ce_(\d+)steps", text)
    if match is not None:
        return "diffusion_pca_rvq_ce", f"dac_{int(match.group(1))}steps"
    return None


def display_name(family: str, model: str) -> str:
    if family == "diffusion_pca":
        steps = parse_steps(model)
        return f"Diffusion PCA {steps} steps" if steps is not None else f"Diffusion PCA {model}"
    if family == "diffusion_pca_rvq_ce":
        steps = parse_steps(model)
        return f"Diffusion PCA+RVQ-CE {steps} steps" if steps is not None else f"Diffusion PCA+RVQ-CE {model}"
    if family == "direct_pca":
        return "Direct PCA regressor"
    names = {
        "grid_render": "Symbolic grid render",
        "source_code_decode": "Source-code decode",
        "symbolic_nn_train": "Symbolic NN retrieval",
        "target_dac_recon": "Target DAC reconstruction",
        "target_pca_recon": "Target PCA reconstruction",
    }
    return names.get(model, model.replace("_", " "))


def parse_steps(model: str) -> int | None:
    match = re.search(r"(\d+)steps", str(model))
    return int(match.group(1)) if match else None


def get_record(records: dict[str, RunRecord], family: str, model: str) -> RunRecord:
    key = run_id_for(family, model)
    if key not in records:
        records[key] = RunRecord(
            run_id=key,
            family=family,
            model=model,
            display_name=display_name(family, model),
        )
    return records[key]


def discover_overall_summaries(repo_root: Path) -> list[Path]:
    roots = [repo_root / "runs_dac", repo_root / "runs_dac_ce", repo_root / "runs_direct", repo_root / "runs_baselines"]
    out: list[Path] = []
    for root in roots:
        if root.is_dir():
            out.extend(sorted(root.rglob("overall_summary.csv")))
    return out


def load_overall_summary_path(
    records: dict[str, RunRecord],
    repo_root: Path,
    path: Path,
    *,
    use_batch_aliases: bool = False,
) -> None:
    family = family_from_path(path, repo_root)
    for row in read_csv_rows(path):
        raw_model = str(row.get("model") or path.parent.parent.name)
        if bool(use_batch_aliases):
            alias = canonical_batch_alias(raw_model)
            if alias is None:
                family_eff, model_eff = family, raw_model
            else:
                family_eff, model_eff = alias
        else:
            family_eff, model_eff = family, raw_model
        record = get_record(records, family_eff, model_eff)
        values = normalize_metric_row(row)
        values["eval_dir"] = rel(path.parent, repo_root)
        record.merge(values, artifact=rel(path, repo_root))


def canonical_batch_acoustic_dir(out_dir: Path) -> Path:
    return out_dir / CANONICAL_BATCH_EVAL / "acoustic_eval"


def load_overall_rows(records: dict[str, RunRecord], repo_root: Path, out_dir: Path) -> None:
    for path in discover_overall_summaries(repo_root):
        load_overall_summary_path(records, repo_root, path)
    batch_path = canonical_batch_acoustic_dir(out_dir) / "overall_summary.csv"
    if batch_path.is_file():
        load_overall_summary_path(records, repo_root, batch_path, use_batch_aliases=True)


def normalize_metric_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if key == "model":
            continue
        if key in METRIC_COLUMNS or key in {
            "num_examples",
            "best_checkpoint_epoch",
            "num_parameters",
            "batch_size",
        }:
            fval = to_float(value)
            out[key] = fval if fval is not None else value
        else:
            out[key] = value
    return out


def load_efficiency_summary_path(
    records: dict[str, RunRecord],
    repo_root: Path,
    path: Path,
    *,
    use_batch_aliases: bool = False,
) -> None:
    family = family_from_path(path, repo_root)
    for row in read_csv_rows(path):
        raw_model = str(row.get("model") or "")
        if not raw_model:
            continue
        if bool(use_batch_aliases):
            alias = canonical_batch_alias(raw_model)
            if alias is None:
                family_eff, model_eff = family, raw_model
            else:
                family_eff, model_eff = alias
        else:
            family_eff, model_eff = family, raw_model
        record = get_record(records, family_eff, model_eff)
        values = normalize_metric_row(row)
        values["eval_dir"] = rel(path.parent, repo_root)
        record.merge(values, artifact=rel(path, repo_root))


def load_efficiency_rows(records: dict[str, RunRecord], repo_root: Path, out_dir: Path) -> None:
    for path in discover_overall_summaries(repo_root):
        eff = path.parent / "efficiency_summary.csv"
        if eff.is_file():
            load_efficiency_summary_path(records, repo_root, eff)
    batch_path = canonical_batch_acoustic_dir(out_dir) / "efficiency_summary.csv"
    if batch_path.is_file():
        load_efficiency_summary_path(records, repo_root, batch_path, use_batch_aliases=True)


def load_direct_audio_summary_path(
    records: dict[str, RunRecord],
    repo_root: Path,
    path: Path,
    *,
    use_batch_aliases: bool = False,
) -> None:
    raw_model = path.parent.name
    if bool(use_batch_aliases):
        alias = canonical_batch_alias(raw_model)
        if alias is None:
            return
        family, model = alias
    else:
        family = family_from_path(path, repo_root)
        model = raw_model
        if family == "unknown":
            return
    record = get_record(records, family, model)
    payload = read_json(path)
    keep = {
        key: payload.get(key)
        for key in (
            "audio_l1_mean",
            "audio_l1_median",
            "mrstft_logmag_l1_mean",
            "mrstft_logmag_l1_median",
            "sample_rate",
            "split",
            "num_examples",
        )
    }
    record.merge(keep, artifact=rel(path, repo_root))


def load_direct_audio_eval(records: dict[str, RunRecord], repo_root: Path, out_dir: Path) -> None:
    for path in sorted(repo_root.rglob("direct_audio_eval/*/summary.json")):
        load_direct_audio_summary_path(records, repo_root, path)
    batch_root = out_dir / CANONICAL_BATCH_EVAL / "direct_audio_eval"
    if batch_root.is_dir():
        for path in sorted(batch_root.glob("*/summary.json")):
            load_direct_audio_summary_path(records, repo_root, path, use_batch_aliases=True)


def baseline_family(model: str) -> str:
    if model in {"target_dac_recon", "target_pca_recon"}:
        return "reconstruction_ceiling"
    return "baseline"


def load_baseline_summaries(records: dict[str, RunRecord], repo_root: Path) -> None:
    root = repo_root / "runs_baselines" / "dac_test_v1"
    if not root.is_dir():
        return
    for path in sorted(root.glob("*/summary.json")):
        model = path.parent.name
        family = baseline_family(model)
        record = get_record(records, family, model)
        payload = read_json(path)
        values = {
            key: payload.get(key)
            for key in (
                "audio_sec_per_sec",
                "clips_per_sec",
                "codec_decode_sec_total",
                "device_name",
                "export_wall_sec_total",
                "model_forward_sec_total",
                "num_examples",
                "num_parameters",
                "peak_gpu_mem_allocated_mb",
                "peak_gpu_mem_reserved_mb",
                "resolved_device",
                "rtf_end_to_end",
                "rtf_model_only",
                "sample_rate",
                "split",
                "total_audio_sec_generated",
            )
        }
        values["run_dir"] = rel(path.parent, repo_root)
        record.merge(values, artifact=rel(path, repo_root))


def load_run_configs(records: dict[str, RunRecord], repo_root: Path) -> None:
    for record in records.values():
        run_dir = infer_run_dir(record, repo_root)
        if run_dir is None:
            continue
        config_path = next((p for p in (run_dir / "run_config.json", run_dir / "config.json") if p.is_file()), None)
        if config_path is None:
            continue
        payload = read_json(config_path)
        model_cfg = dict(payload.get("model_cfg") or {})
        frontend_cfg = dict(model_cfg.get("frontend_cfg") or payload.get("frontend_cfg") or {})
        values = {
            "run_dir": rel(run_dir, repo_root),
            "num_steps": payload.get("num_steps", parse_steps(record.model)),
            "rvq_ce_weight": payload.get("rvq_ce_weight"),
            "target_layout": payload.get("target_layout") or payload.get("cache_config", {}).get("target_layout"),
            "target_dim": (
                payload.get("target_dim")
                or payload.get("cache_config", {}).get("target_dim")
                or model_cfg.get("x_dim")
            ),
            "x_dim": model_cfg.get("x_dim"),
            "num_parameters": payload.get("num_parameters"),
            "d_model": model_cfg.get("d_model"),
            "num_layers": model_cfg.get("num_layers"),
            "num_heads": model_cfg.get("num_heads"),
            "frontend_variant": frontend_cfg.get("variant"),
            "frontend_radii": frontend_cfg.get("multiscale_radii") or frontend_cfg.get("window_radius"),
            "frontend_embed_dim": frontend_cfg.get("embed_dim"),
            "positional_encoding": model_cfg.get("positional_encoding"),
        }
        record.merge(values, artifact=rel(config_path, repo_root))


def infer_run_dir(record: RunRecord, repo_root: Path) -> Path | None:
    if record.family == "diffusion_pca":
        return repo_root / "runs_dac" / record.model
    if record.family == "diffusion_pca_rvq_ce":
        return repo_root / "runs_dac_ce" / record.model
    if record.family == "direct_pca":
        return repo_root / "runs_direct" / record.model
    if record.family in {"baseline", "reconstruction_ceiling"}:
        return repo_root / "runs_baselines" / "dac_test_v1" / record.model
    return None


def cache_metadata(cache_root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"cache_root": str(cache_root)}
    config_path = cache_root / "config.json"
    if config_path.is_file():
        out.update(read_json(config_path))
    for split in ("train", "validation", "test"):
        summary_path = cache_root / "summaries" / f"{split}.json"
        if summary_path.is_file():
            summary = read_json(summary_path)
            out[f"{split}_examples"] = summary.get("num_examples")
            out[f"{split}_hash"] = summary.get("split_manifest_hash")
    return out


def git_commit(repo_root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return ""
    return proc.stdout.strip()


def sorted_records(records: Iterable[RunRecord]) -> list[RunRecord]:
    family_order = {
        "reconstruction_ceiling": 0,
        "baseline": 1,
        "direct_pca": 2,
        "diffusion_pca": 3,
        "diffusion_pca_rvq_ce": 4,
        "unknown": 9,
    }
    model_order = {
        "target_dac_recon": 0,
        "target_pca_recon": 1,
        "grid_render": 2,
        "source_code_decode": 3,
        "symbolic_nn_train": 4,
    }

    def key(record: RunRecord) -> tuple[Any, ...]:
        return (
            family_order.get(record.family, 8),
            model_order.get(record.model, 100),
            parse_steps(record.model) if parse_steps(record.model) is not None else 999,
            record.display_name,
        )

    return sorted(records, key=key)


def missing_metrics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    required = ("mel_mae_db", "onset_flux_cosine", "band_balance_l1", "rtf_end_to_end")
    important = ("fad_inf", "mrstft_logmag_l1_mean", "audio_l1_mean")
    out: list[dict[str, Any]] = []
    for row in rows:
        missing = [key for key in required if to_float(row.get(key)) is None]
        soft_missing = [key for key in important if to_float(row.get(key)) is None]
        if missing or soft_missing:
            out.append(
                {
                    "run_id": str(row.get("run_id", "")),
                    "display_name": str(row.get("display_name", "")),
                    "missing_core_metrics": ",".join(missing),
                    "missing_optional_metrics": ",".join(soft_missing),
                }
            )
    return out


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
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def fmt_metric(value: Any, decimals: int = 3, scale: float = 1.0) -> str:
    val = to_float(value)
    if val is None:
        return "--"
    return f"{val * scale:.{decimals}f}"


def fmt_params(value: Any) -> str:
    val = to_float(value)
    if val is None or val <= 0.0:
        return "--"
    return f"{val / 1_000_000.0:.1f}M"


def fmt_percent_delta(delta: float | None) -> str:
    if delta is None or not math.isfinite(float(delta)):
        return "--"
    return f"{100.0 * float(delta):.1f}\\%"


def row_metric(row: Mapping[str, Any] | None, metric: str) -> float | None:
    if row is None:
        return None
    return to_float(row.get(metric))


def best_row(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    higher_is_better: bool = False,
    families: set[str] | None = None,
) -> Mapping[str, Any] | None:
    candidates = [
        row
        for row in rows
        if (families is None or str(row.get("family", "")) in families)
        and to_float(row.get(metric)) is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row_metric(row, metric) or 0.0)) if bool(higher_is_better) else min(
        candidates, key=lambda row: float(row_metric(row, metric) or 0.0)
    )


def relative_improvement(
    baseline: Mapping[str, Any] | None,
    candidate: Mapping[str, Any] | None,
    metric: str,
    *,
    higher_is_better: bool = False,
) -> float | None:
    base = row_metric(baseline, metric)
    cand = row_metric(candidate, metric)
    if base is None or cand is None or abs(float(base)) <= 1.0e-12:
        return None
    if bool(higher_is_better):
        return (float(cand) - float(base)) / abs(float(base))
    return (float(base) - float(cand)) / abs(float(base))


def tex_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"System & Type & Clips & $\mathrm{FAD}_{\infty}$ $\downarrow$ & $\mathrm{FAD}\text{-}R^2$ $\uparrow$ & Mel $\downarrow$ & Flux $\uparrow$ & MRSTFT $\downarrow$ & Audio $L_1$ $\downarrow$ & RTF $\downarrow$ \\",
        r"\midrule",
    ]
    for row in rows:
        if row.get("family") == "unknown":
            continue
        lines.append(
            " & ".join(
                [
                    latex_escape(row.get("display_name", "")),
                    latex_escape(row.get("family", "")),
                    fmt_metric(row.get("num_examples"), 0),
                    fmt_metric(row.get("fad_inf"), 3),
                    fmt_metric(row.get("fad_inf_r2"), 3),
                    fmt_metric(row.get("mel_mae_db"), 2),
                    fmt_metric(row.get("onset_flux_cosine"), 3),
                    fmt_metric(row.get("mrstft_logmag_l1_mean"), 3),
                    fmt_metric(row.get("audio_l1_mean"), 4),
                    fmt_metric(row.get("rtf_end_to_end"), 3),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Automatically aggregated test-set metrics. Missing cells indicate that the corresponding evaluation artifact was not present when the table was generated. $\mathrm{FAD}_{\infty}$ is distributional; $\mathrm{FAD}\text{-}R^2$ is the extrapolation-fit diagnostic; the paired metrics quantify reconstruction and symbolic-control-related structure.}",
            r"\label{tab:auto_main_results}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def result_highlights_tex(rows: Sequence[Mapping[str, Any]], gaps: Sequence[Mapping[str, Any]]) -> str:
    learned_families = {"direct_pca", "diffusion_pca", "diffusion_pca_rvq_ce"}
    diffusion_families = {"diffusion_pca", "diffusion_pca_rvq_ce"}
    direct = next((row for row in rows if row.get("family") == "direct_pca"), None)
    mel_best = best_row(rows, "mel_mae_db", families=learned_families)
    flux_best = best_row(rows, "onset_flux_cosine", higher_is_better=True, families=learned_families)
    mrstft_best = best_row(rows, "mrstft_logmag_l1_mean", families=learned_families)
    audio_best = best_row(rows, "audio_l1_mean", families=learned_families)
    rtf_best_diffusion = best_row(rows, "rtf_end_to_end", families=diffusion_families)
    fad_best_diffusion = best_row(rows, "fad_inf", families=diffusion_families)
    lines = [
        r"\paragraph{Main observations.}",
        "The current artifacts support four controlled observations.",
    ]
    if mel_best is not None and direct is not None:
        lines.append(
            (
                f"First, {latex_escape(mel_best.get('display_name', ''))} gives the lowest learned-system mel error "
                f"({fmt_metric(mel_best.get('mel_mae_db'), 2)} dB), a {fmt_percent_delta(relative_improvement(direct, mel_best, 'mel_mae_db'))} "
                f"relative reduction from the direct PCA regressor ({fmt_metric(direct.get('mel_mae_db'), 2)} dB)."
            )
        )
    if flux_best is not None:
        lines.append(
            (
                f"Second, {latex_escape(flux_best.get('display_name', ''))} gives the strongest onset-flux agreement "
                f"among learned systems ({fmt_metric(flux_best.get('onset_flux_cosine'), 3)}), supporting the auxiliary "
                "RVQ-CE term as codec-structure supervision rather than a discrete-token generator."
            )
        )
    if mrstft_best is not None and direct is not None:
        lines.append(
            (
                f"Third, {latex_escape(mrstft_best.get('display_name', ''))} gives the best learned-system MRSTFT "
                f"({fmt_metric(mrstft_best.get('mrstft_logmag_l1_mean'), 3)}), a {fmt_percent_delta(relative_improvement(direct, mrstft_best, 'mrstft_logmag_l1_mean'))} "
                "relative reduction from direct PCA regression."
            )
        )
    if audio_best is not None and direct is not None:
        lines.append(
            (
                f"Fourth, phase-sensitive waveform $L_1$ is the main counterexample: {latex_escape(audio_best.get('display_name', ''))} "
                f"has the lowest learned-system waveform $L_1$ ({fmt_metric(audio_best.get('audio_l1_mean'), 4)}), "
                "so the diffusion results should be interpreted as spectral/transient improvements rather than universal "
                "samplewise reconstruction dominance."
            )
        )
    if rtf_best_diffusion is not None:
        lines.append(
            (
                "All completed diffusion rows are faster than real time in the stored run; RTF differences should "
                "be interpreted as implementation/runtime measurements rather than architectural consequences of RVQ-CE."
            )
        )
    if fad_best_diffusion is not None:
        lines.append(
            (
                f"The best completed diffusion $\\mathrm{{FAD}}_{{\\infty}}$ row is {latex_escape(fad_best_diffusion.get('display_name', ''))} "
                f"({fmt_metric(fad_best_diffusion.get('fad_inf'), 3)})."
            )
        )
    if gaps:
        lines.append(
            (
                f"{len(gaps)} rows still have missing core or optional metrics, so any claim involving reconstruction ceilings, "
                "non-learned retrieval, or direct-regressor distributional quality must remain limited until those rows are complete."
            )
        )
    return "\n".join(lines) + "\n"


def methodology_tex(metadata: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    train_n = fmt_metric(metadata.get("train_examples"), 0)
    val_n = fmt_metric(metadata.get("validation_examples"), 0)
    test_n = fmt_metric(metadata.get("test_examples"), 0)
    sample_rate = fmt_metric(metadata.get("codec_sample_rate"), 0)
    token_rate = fmt_metric(metadata.get("target_token_rate_hz") or metadata.get("codec_frame_rate"), 4)
    quantizers = fmt_metric(metadata.get("codec_num_codebooks") or metadata.get("dac_num_quantizers"), 0)
    pca_k = fmt_metric(metadata.get("pca_k") or metadata.get("target_dim"), 0)
    full_dim = fmt_metric(metadata.get("full_target_dim") or metadata.get("codec_target_dim"), 0)
    diffusion_rows = [row for row in rows if str(row.get("family", "")).startswith("diffusion")]
    representative = diffusion_rows[0] if diffusion_rows else {}
    d_model = fmt_metric(representative.get("d_model"), 0)
    layers = fmt_metric(representative.get("num_layers"), 0)
    heads = fmt_metric(representative.get("num_heads"), 0)
    x_dim = fmt_metric(representative.get("x_dim") or representative.get("target_dim"), 0)
    frontend_variant = latex_escape(representative.get("frontend_variant") or "hybrid")
    frontend_embed = fmt_metric(representative.get("frontend_embed_dim"), 0)
    frontend_radii = latex_escape(representative.get("frontend_radii") or "0,22,41,55")
    positional = latex_escape(representative.get("positional_encoding") or "seconds")
    plain_steps = sorted(
        {
            int(step)
            for row in rows
            if row.get("family") == "diffusion_pca"
            for step in [to_int(row.get("num_steps") or parse_steps(str(row.get("model", ""))))]
            if step is not None
        }
    )
    rvq_steps = sorted(
        {
            int(step)
            for row in rows
            if row.get("family") == "diffusion_pca_rvq_ce"
            for step in [to_int(row.get("num_steps") or parse_steps(str(row.get("model", ""))))]
            if step is not None
        }
    )
    plain_steps_text = ", ".join(str(step) for step in plain_steps) if plain_steps else "--"
    rvq_steps_text = ", ".join(str(step) for step in rvq_steps) if rvq_steps else "--"
    num_runs = len(rows)
    return "\n".join(
        [
            r"\section{Methodology}",
            r"\subsection{Dataset and codec configuration}",
            (
                f"The diffusion cache contains {train_n} training, {val_n} validation, and {test_n} test windows. "
                f"Audio is represented with DAC at {sample_rate} Hz using {quantizers} quantizers. "
                f"The target token rate is {token_rate} Hz, and the primary target is a {pca_k}-dimensional "
                f"framewise PCA projection of the full {full_dim}-dimensional summed DAC RVQ codebook-embedding trajectory. "
                "The modeled target is the post-quantizer summed embedding, not the raw pre-quantizer encoder output."
            ),
            "",
            r"\subsection{Seconds-aware symbolic front-end}",
            (
                f"The conditioning front-end is the {frontend_variant} seconds-aware local front-end with radii "
                f"\\texttt{{{frontend_radii}}}, {frontend_embed}-dimensional per-scale features, and "
                f"\\texttt{{{positional}}} positional encoding. It samples symbolic drum-grid context at DAC token "
                "times, so symbolic conditioning and codec-latent targets remain aligned under BPM-derived timing."
            ),
            (
                "\\begin{center}\\small\\begin{tabular}{ll}\\toprule Component & Value \\\\ \\midrule "
                "Symbolic grid rate & 250 Hz \\\\ Drum families & 8 \\\\ Numeric channels & 24 \\\\ "
                "Articulation encoding & per-family IDs to one-hot lanes \\\\ "
                "Radii & 0, 22, 41, 55 grid steps \\\\ Radii in ms & 0, 88, 164, 220 ms \\\\ "
                f"Branch output & {frontend_embed} \\\\ Concatenated conditioning dim & 256 \\\\ "
                f"Denoiser width & {d_model} \\\\ \\bottomrule\\end{{tabular}}\\end{{center}}"
            ),
            "Radius 0 denotes the feature sample at the codec-frame time itself; nonzero radii add symmetric local windows around that time.",
            "",
            r"\subsection{PCA-latent diffusion model}",
            (
                f"The completed diffusion runs model {x_dim}-dimensional normalized PCA coefficients with a "
                f"{d_model}-wide Transformer, {layers} layers, and {heads} attention heads. Plain PCA diffusion "
                f"runs are available for denoising step counts {plain_steps_text}; auxiliary RVQ cross-entropy "
                f"runs are available for step counts {rvq_steps_text}. Each stored quantitative row uses one "
                "generated sample per conditioning input; the paired confidence intervals therefore summarize clip "
                "variation, not sampling-seed variation."
            ),
            "",
            r"\subsection{Compared systems}",
            (
                "The aggregation includes reconstruction ceilings, non-learned symbolic/rendering baselines, "
                "a direct PCA regressor, plain PCA diffusion runs, and PCA diffusion runs with auxiliary RVQ "
                "cross-entropy when their artifacts are available. Rows with missing metric artifacts are retained "
                "and flagged in \\texttt{missing\\_metrics.csv} rather than silently dropped."
            ),
            (
                "The symbolic grid renderer is a deterministic procedural renderer from the eight-family grid, "
                "with event times from the 250 Hz seconds grid and velocities from the onset/state lanes. "
                "The symbolic nearest-neighbor baseline retrieves from the training split using normalized dot "
                "products over flattened symbolic features plus BPM, then decodes the retrieved train clip's DAC source codes."
            ),
            (
                "The direct PCA regressor uses the same conditioning frontend and PCA target family but a deterministic "
                "Transformer head. Its width is larger than the diffusion model, so the comparison is conservative with "
                "respect to capacity but not architecture-identical."
            ),
            "",
            r"\subsection{Statistical protocol}",
            (
                "Paired clip-level intervals use 2,000 percentile bootstrap resamples at 95\\% confidence. "
                "Best-versus-rest paired tests use 2,000 two-sided sign-flip permutations over shared dataset indices, "
                "with Holm correction within each metric. FAD uncertainty is reported from repeated extrapolation runs "
                "and is not used as a paired significance test."
            ),
            "",
            r"\subsection{Reproducible result aggregation}",
            (
                "All reported tables are generated by \\texttt{scripts/build\\_paper\\_results.py}. "
                f"The script gathers completed run artifacts without rerunning inference, writes normalized CSV/JSON files, "
                f"and emits the standalone \\LaTeX{{}} tables used in this section. The aggregated artifact contains {num_runs} systems."
            ),
            "",
        ]
    )


def results_tex(rows: Sequence[Mapping[str, Any]], gaps: Sequence[Mapping[str, Any]]) -> str:
    complete = [row for row in rows if row.get("family") != "unknown"]
    text = [
        r"\section{Results and Analysis}",
        (
            "Table~\\ref{tab:auto_main_results} is generated directly from the normalized run manifest. "
            "It should be interpreted as an automatic-metric summary: distributional quality, paired "
            "spectral/onset fidelity, direct reconstruction, and runtime are separate axes."
        ),
        "",
        tex_table(complete),
        result_highlights_tex(complete, gaps),
    ]
    if gaps:
        text.extend(
            [
                r"\paragraph{Completeness.}",
                (
                    f"The current aggregation found {len(gaps)} rows with at least one missing core or optional metric. "
                    "Those gaps are listed in \\texttt{missing\\_metrics.csv}; they should be resolved before making "
                    "final comparative claims."
                ),
                "",
            ]
        )
    return "\n".join(text)


def standalone_tex(methodology: str, results: str) -> str:
    return "\n".join(
        [
            r"\documentclass[11pt]{article}",
            r"\usepackage[margin=1in]{geometry}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage{lmodern}",
            r"\usepackage{microtype}",
            r"\usepackage{amsmath}",
            r"\usepackage{booktabs}",
            r"\usepackage{hyperref}",
            r"\title{Automatically Aggregated Drum Rendering Results}",
            r"\author{}",
            r"\date{}",
            r"\begin{document}",
            r"\maketitle",
            methodology,
            results,
            r"\end{document}",
            "",
        ]
    )


def paper_readiness_md(
    metadata: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    gaps: Sequence[Mapping[str, Any]],
) -> str:
    gap_names = ", ".join(str(row.get("display_name", "")) for row in gaps) if gaps else "none"
    complete_rows = [
        row
        for row in rows
        if all(
            to_float(row.get(key)) is not None
            for key in ("mel_mae_db", "onset_flux_cosine", "band_balance_l1", "rtf_end_to_end")
        )
    ]
    direct_complete = any(row.get("family") == "direct_pca" for row in complete_rows)
    diffusion_complete = any(str(row.get("family", "")).startswith("diffusion") for row in complete_rows)
    lines = [
        "# Paper Readiness",
        "",
        (
            "This report is generated from completed artifacts only. It separates claims that are "
            "currently supported from claims that must be fixed with more evaluation or rerouted to "
            "limitations/future work."
        ),
        "",
        "## Supported With Current Artifacts",
        "",
    ]
    if direct_complete and diffusion_complete:
        lines.append("- Direct PCA regression versus PCA diffusion can be discussed with paired automatic metrics.")
    if any(row.get("family") == "diffusion_pca_rvq_ce" for row in complete_rows):
        lines.append("- RVQ-CE as an auxiliary loss can be discussed as a completed ablation.")
    if any(row.get("family") == "diffusion_pca" for row in complete_rows):
        lines.append("- Denoising-step quality/runtime trade-offs can be discussed for completed diffusion runs.")
    if any(row.get("model") == "grid_render" for row in complete_rows):
        lines.append("- Symbolic grid rendering can be used as a paired-metric baseline, except for missing FAD.")
    if len(lines) <= 6:
        lines.append("- No full analytical claim is supported yet; complete the missing metrics first.")
    lines.extend(
        [
            "",
            "## Must Fix Before Strong Submission",
            "",
            f"- Rows with missing metrics: {gap_names}.",
            "- Reconstruction ceilings need paired acoustic metrics so the paper can quantify the PCA bottleneck rather than only model error.",
            "- Direct PCA regressor needs FAD-infinity if the paper compares distributional quality against diffusion.",
            "- Source-code decode and symbolic nearest-neighbor retrieval need the same metric set as learned models if they remain in the main table.",
            "",
            "## Reroute If Not Fixed",
            "",
            "- Token-space versus token-embedding-space results should be written as motivation or future work unless discrete-token and full-embedding runs are produced.",
            "- The UI should be presented as an analysis/demo interface, not as user-study evidence.",
            "- The conclusion should remain provisional until `missing_metrics.csv` is empty or all missing rows are removed from the main claim set.",
            "",
            "## Artifact Summary",
            "",
            f"- Git commit: `{metadata.get('git_commit', '')}`",
            f"- Aggregated rows: {metadata.get('num_rows', len(rows))}",
            f"- Rows with missing metrics: {metadata.get('num_missing_metric_rows', len(gaps))}",
            f"- Cache: `{metadata.get('cache_root', '')}`",
            "",
        ]
    )
    return "\n".join(lines)


def prediction_dir_for_row(row: Mapping[str, Any], repo_root: Path) -> Path | None:
    family = str(row.get("family") or "")
    model = str(row.get("model") or "")
    candidates: list[Path] = []
    if family == "diffusion_pca":
        candidates.append(repo_root / "runs_dac" / "test_acoustic_eval_fad_all4" / "predictions" / model)
    elif family == "diffusion_pca_rvq_ce":
        candidates.append(repo_root / "runs_dac_ce" / "test_acoustic_eval" / "predictions" / model)

    run_dir_text = str(row.get("run_dir") or "").strip()
    if run_dir_text:
        run_dir = (repo_root / run_dir_text).resolve()
        if family == "direct_pca":
            candidates.append(run_dir / "test_set_predictions")
        candidates.append(run_dir)
    for candidate in candidates:
        if (candidate / "manifest.jsonl").is_file():
            return candidate
    return None


def prediction_name_for_row(row: Mapping[str, Any]) -> str:
    family = str(row.get("family") or "")
    model = str(row.get("model") or "")
    if family == "diffusion_pca":
        steps = parse_steps(model)
        return f"diffusion_pca_{steps}steps" if steps is not None else f"diffusion_pca_{model}"
    if family == "diffusion_pca_rvq_ce":
        steps = parse_steps(model)
        return f"diffusion_pca_rvq_ce_{steps}steps" if steps is not None else f"diffusion_pca_rvq_ce_{model}"
    return model.replace(":", "_")


def suggested_commands(repo_root: Path, rows: Sequence[Mapping[str, Any]], gaps: Sequence[Mapping[str, Any]]) -> str:
    eval_pairs: list[tuple[str, Path]] = []
    seen_dirs: set[Path] = set()
    for row in rows:
        pred_dir = prediction_dir_for_row(row, repo_root)
        if pred_dir is None or pred_dir in seen_dirs:
            continue
        seen_dirs.add(pred_dir)
        eval_pairs.append((prediction_name_for_row(row), pred_dir))
    lines = [
        "# Suggested completion commands",
        "",
        "Run these only after checking GPU availability. They are not executed by build_paper_results.py.",
        "",
    ]
    if eval_pairs:
        dirs = " \\\n    ".join(str(path) for _name, path in eval_pairs)
        names = " ".join(name for name, _path in eval_pairs)
        lines.extend(
            [
                "## Canonical full acoustic evaluation",
                "python scripts/run_diffusion_acoustic_eval.py \\",
                f"  --prediction-dirs {dirs} \\",
                f"  --prediction-names {names} \\",
                "  --cache-root cache_4beats_dac44q9_pca72_native_bpmgeom_duration_v1 \\",
                f"  --out-dir paper_results/{CANONICAL_BATCH_EVAL} \\",
                "  --fad-model clap-laion-music \\",
                "  --fad-workers 1 --fad-inf-workers 1 \\",
                "  --batch-size 16 --num-workers 4 --device auto \\",
                "  --with-inference --no-plots --overwrite",
                "",
                "## Faster fallback if paired inference is too slow",
                "python scripts/run_diffusion_acoustic_eval.py \\",
                f"  --prediction-dirs {dirs} \\",
                f"  --prediction-names {names} \\",
                "  --cache-root cache_4beats_dac44q9_pca72_native_bpmgeom_duration_v1 \\",
                f"  --out-dir paper_results/{CANONICAL_BATCH_EVAL} \\",
                "  --fad-model clap-laion-music \\",
                "  --fad-workers 1 --fad-inf-workers 1 \\",
                "  --batch-size 16 --num-workers 4 --device auto \\",
                "  --no-plots --overwrite",
                "",
            ]
        )
    lines.extend(
        [
            "## Rebuild paper tables after all evaluations finish",
            "python scripts/build_paper_results.py --out-dir paper_results --strict",
            "",
        ]
    )
    return "\n".join(lines)


def build(repo_root: Path, cache_root: Path, out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: dict[str, RunRecord] = {}
    load_overall_rows(records, repo_root, out_dir)
    load_efficiency_rows(records, repo_root, out_dir)
    load_direct_audio_eval(records, repo_root, out_dir)
    load_baseline_summaries(records, repo_root)
    load_run_configs(records, repo_root)

    rows = [record.row() for record in sorted_records(records.values())]
    gaps = missing_metrics(rows)
    metadata = cache_metadata(cache_root)
    batch_summary_path = canonical_batch_acoustic_dir(out_dir) / "summary.json"
    metadata["git_commit"] = git_commit(repo_root)
    metadata["num_rows"] = len(rows)
    metadata["num_missing_metric_rows"] = len(gaps)
    metadata["canonical_batch_eval_dir"] = rel(canonical_batch_acoustic_dir(out_dir), repo_root)
    metadata["canonical_batch_eval_complete"] = batch_summary_path.is_file()
    if batch_summary_path.is_file():
        batch_summary = read_json(batch_summary_path)
        metadata["canonical_batch_fad_completed"] = bool(batch_summary.get("fad_completed"))
        metadata["canonical_batch_inference_completed"] = bool(batch_summary.get("inference_completed"))

    write_csv(out_dir / "run_metrics.csv", rows, CSV_COLUMNS)
    write_json(out_dir / "run_metrics.json", {"metadata": metadata, "rows": rows})
    core_complete_rows = [
        row
        for row in rows
        if all(to_float(row.get(key)) is not None for key in ("mel_mae_db", "onset_flux_cosine", "band_balance_l1", "rtf_end_to_end"))
    ]
    write_csv(out_dir / "core_complete_run_metrics.csv", core_complete_rows, CSV_COLUMNS)
    write_csv(
        out_dir / "missing_metrics.csv",
        gaps,
        ("run_id", "display_name", "missing_core_metrics", "missing_optional_metrics"),
    )
    write_json(out_dir / "manifest.json", metadata)

    methodology = methodology_tex(metadata, rows)
    results = results_tex(rows, gaps)
    highlights = result_highlights_tex(rows, gaps)
    (out_dir / "methodology_section.tex").write_text(methodology, encoding="utf-8")
    (out_dir / "results_section.tex").write_text(results, encoding="utf-8")
    (out_dir / "result_highlights.tex").write_text(highlights, encoding="utf-8")
    (out_dir / "standalone_results.tex").write_text(standalone_tex(methodology, results), encoding="utf-8")
    (out_dir / "paper_readiness.md").write_text(paper_readiness_md(metadata, rows, gaps), encoding="utf-8")
    (out_dir / "completion_commands.md").write_text(suggested_commands(repo_root, rows, gaps), encoding="utf-8")
    return rows, gaps


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    rows, gaps = build(repo_root, cache_root, out_dir)
    print(f"wrote {len(rows)} rows to {out_dir / 'run_metrics.csv'}")
    print(f"wrote core-complete rows to {out_dir / 'core_complete_run_metrics.csv'}")
    print(f"wrote standalone TeX to {out_dir / 'standalone_results.tex'}")
    if gaps:
        print(f"warning: {len(gaps)} rows have missing metrics; see {out_dir / 'missing_metrics.csv'}")
        core_gaps = [row for row in gaps if str(row.get("missing_core_metrics") or "").strip()]
        if bool(args.strict) and core_gaps:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
