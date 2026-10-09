#!/usr/bin/env python3
"""Tables of the TRIA evaluation on GMD (evaluate.sh): results/overall.csv, results/pairs.csv and results/summary.md.

p-values come with Holm's adjustment over the pairs of each metric (pairs.csv: p_holm).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

LABELS = {
    "grid_250hz": "Ours (250 Hz grid, diffusion + RVQ-CE)",
    "tria_released_bar": "TRIA released, bar as prompts",
    "tria_released_midi": "TRIA released, MIDI render + reference",
    "tria_finetuned_bar": "TRIA fine-tuned, bar as prompts",
    "tria_finetuned_midi": "TRIA fine-tuned, MIDI render + reference",
}
# the paper's results columns (results/final/summary.md) with onset F1 next to onset cosine, then KAD
METRICS = [("mel_mae_db", "Mel MAE ↓"), ("onset_flux_cosine", "Onset cosine ↑"), ("onset_f1_30ms_mean", "Onset F1 ±30 ms ↑"),
           ("onset_f1_50ms_mean", "Onset F1 ±50 ms ↑"), ("fad_inf", "FAD∞ ↓"), ("audio_l1_mean", "Audio L1 ↓"),
           ("mrstft_logmag_l1_mean", "MR-STFT ↓")]
KAD_LABEL = "KAD ↓"
TRIA_PARAMETERS = 43_051_008  # TRIA's trainable parameters (its training log; the DAC tokenizer is frozen, as ours)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def holm(p: list[float]) -> list[float]:
    order = sorted(range(len(p)), key=lambda i: p[i])
    out, running = [0.0] * len(p), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        out[i] = running
    return out


def fmt(x, digits=4) -> str:
    return "" if x is None or x == "" else f"{float(x):.{digits}g}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evaluation-dir", type=Path, required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    ev, out = args.evaluation_dir, args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    overall = {row["model"]: row for row in read_csv(ev / "acoustic_eval" / "overall_summary.csv")}
    kad = json.loads((out / "stats" / "kad.json").read_text())
    rows = []
    for name in args.names:
        row = {"model": name, "label": LABELS.get(name, name), **{k: overall[name].get(k, "") for k in
               ("num_examples", "mel_mae_db", "onset_flux_cosine", "fad_inf", "fad_inf_sd")}}
        summary = json.loads((ev / "direct_audio_eval" / name / "summary.json").read_text())
        row.update({k: summary[k] for k in ("audio_l1_mean", "mrstft_logmag_l1_mean", "onset_f1_30ms_mean",
                                            "onset_f1_50ms_mean")})
        staged = json.loads((ev / "systems" / name / "summary.json").read_text())
        row["num_parameters"] = TRIA_PARAMETERS if name.startswith("tria_") else staged.get("num_parameters", "")
        row["loudness_gain_db"] = staged["loudness_match"]["mean_gain_db"]
        k = kad["systems"][name]
        row["kad"], (row["kad_ci_low"], row["kad_ci_high"]) = k["kad"], k["ci95"]
        rows.append(row)
    kad_cols = ["kad", "kad_ci_low", "kad_ci_high"]
    write_csv(out / "overall.csv", rows, ["model", "label", "num_examples", "num_parameters", "mel_mae_db",
                                          "onset_flux_cosine", "onset_f1_30ms_mean", "onset_f1_50ms_mean", "fad_inf",
                                          "fad_inf_sd", "audio_l1_mean", "mrstft_logmag_l1_mean", *kad_cols,
                                          "loudness_gain_db"])

    stats = {}
    for f in ("acoustic", "direct_audio"):
        stats.update({pair: {**stats.get(pair, {}), **v["metrics"]}
                      for pair, v in json.loads((out / "stats" / f"{f}.json").read_text())["pairs"].items()})
    pair_rows = []
    for pair, metrics in stats.items():
        for metric, s in metrics.items():
            pair_rows.append({"pair": pair, "metric": metric, "diff": s["mean_diff"], "ci_low": s["clustered_ci95"][0],
                              "ci_high": s["clustered_ci95"][1], "p": s["clustered_signflip_p"], "test": "sign-flip"})
        k = kad["pairs"][pair]
        pair_rows.append({"pair": pair, "metric": "kad", "diff": k["diff"], "ci_low": k["ci95"][0],
                          "ci_high": k["ci95"][1], "p": k["p_permutation"], "test": "recording permutation"})
    for metric in dict.fromkeys(r["metric"] for r in pair_rows):
        family = [r for r in pair_rows if r["metric"] == metric]
        for r, adjusted in zip(family, holm([float(r["p"]) for r in family])):
            r["p_holm"] = adjusted
    write_csv(out / "pairs.csv", pair_rows, ["pair", "metric", "diff", "ci_low", "ci_high", "p", "p_holm", "test"])

    lines = ["# TRIA on the GMD test split", "",
             f"{rows[0]['num_examples']} clips of {kad['num_recordings']} test recordings, every metric against the real "
             "GMD bars. Each system's clips are gain-matched to their real bar's loudness (ITU-R BS.1770) first, which "
             "only FAD and KAD see: the per-clip metrics are means over clips of peak-normalized audio. Onset F1 matches "
             "the clip's onsets to the real bar's within ±30 or ±50 ms (librosa's onset detector on both, mir_eval), "
             "which a few milliseconds of constant output delay do not change, unlike the onset cosine. FAD∞ is "
             "CLAP-LAION-Music over 8 repeats; KAD (kadtk, ×100) uses PANNs-WGLM embeddings with one kernel for all "
             "systems (bandwidth from the real bars), 95% CI from a bootstrap over recordings.", "",
             "| System | " + " | ".join(l for _, l in METRICS) + f" | Params | {KAD_LABEL} |",
             "| --- |" + " ---: |" * (len(METRICS) + 2)]
    for r in rows:
        cells = [fmt(r[c]) for c, _ in METRICS]
        cells.append(f"{int(r['num_parameters']) / 1e6:.2f}M" if r["num_parameters"] != "" else "")
        cells.append(f"{fmt(r['kad'])} [{fmt(r['kad_ci_low'])}, {fmt(r['kad_ci_high'])}]")
        lines.append(f"| {r['label']} | " + " | ".join(cells) + " |")
    lines += ["", "## Paired comparisons (A − B)", "",
              f"Recording-clustered: 95% CI from a bootstrap over the {kad['num_recordings']} recordings; p from a "
              "recording-level sign-flip test (per-clip metrics) or a recording-level swap of the two systems' clips "
              "(KAD), 5000 resamples each; Holm: adjusted over the pairs of each metric.", "",
              "| A − B | Metric | Difference | 95% CI | p | p (Holm) |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for p in pair_rows:
        lines.append(f"| {p['pair'].replace(':', ' − ')} | {p['metric']} | {fmt(p['diff'])} | "
                     f"[{fmt(p['ci_low'])}, {fmt(p['ci_high'])}] | {fmt(p['p'], 2)} | {fmt(p['p_holm'], 2)} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
