#!/usr/bin/env python3
"""Lay out the paper's ablations on the static listening page.

Run against the docs directory on the anonymous branch. Each example keeps its
EDG plot, recording, and TRIA players; its right column becomes the recording,
our model, and the paper's eight ablations in a two-column grid. Ablation audio
comes from the evaluated test exports, BS.1770 loudness matched to the
recording with peaks below 0.999, and trimmed or zero-padded to the recording's
length, like our model's clip.

    python codec_target_ablation/update_demo.py --docs-root <anonymous-checkout>/docs
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from pathlib import Path

import numpy as np
import pyloudnorm
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
# (audio name, export, label, detail), shown two per row.
ARMS = (
    ("presnap", "runs/final/exports/presnap_latent", "Pre-snap latent", ""),
    ("tokens", "codec_target_ablation/exports/tokens_9x1024", "Discrete tokens", "parallel classifier"),
    ("regression", "runs/final/exports/regression_6layer", "Regression", "6 layers"),
    ("regression8", "runs/final/exports/regression_8layer_capacity", "Regression", "8 layers"),
    ("ours90", "runs/final/exports/grid_90hz", "90 Hz grid", ""),
    ("ours120", "runs/final/exports/grid_120hz", "120 Hz grid", ""),
    ("ours500", "runs/final/exports/grid_500hz", "500 Hz grid", ""),
    ("pca72", "runs/final/exports/postsnap_pca", "Without RVQ-CE", "plain diffusion"),
)
BEGIN = "<!-- ablations:start -->"
END = "<!-- ablations:end -->"
DESCRIPTION = (
    '<p class="abl-description">The 90, 120, and 500 Hz grids keep our model\'s '
    'auxiliary RVQ-CE loss; the other five ablations use the 250 Hz grid without '
    'it, as in the paper.</p>'
)
STYLE = """/* ablations */
.abl { display: grid; gap: 8px; }
.abl-head { font-weight: 700; font-size: 14px; color: var(--ours); }
.abl .cell { border-top-color: var(--abl); }
"""


def read_manifest(directory: Path) -> dict[tuple[str, int], dict]:
    rows = [json.loads(line) for line in (directory / "manifest.jsonl").read_text().splitlines() if line.strip()]
    indexed = {(row["source_id"], int(row["beat_index"])): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"duplicate excerpt identities in {directory}")
    return indexed


def match_loudness(source: Path, reference: Path, destination: Path) -> dict:
    audio, sr = sf.read(source, dtype="float64", always_2d=True)
    target, target_sr = sf.read(reference, dtype="float64", always_2d=True)
    if sr != target_sr:
        raise ValueError(f"sample-rate mismatch: {source}: {sr}, {reference}: {target_sr}")
    audio, target = audio.mean(axis=1), target.mean(axis=1)
    meter = pyloudnorm.Meter(sr)
    own, real = meter.integrated_loudness(audio), meter.integrated_loudness(target)
    gain, limited = 1.0, False
    if math.isfinite(own) and math.isfinite(real):
        gain = 10 ** ((real - own) / 20)
        peak = float(np.abs(audio).max())
        if peak * gain > 0.999:
            gain, limited = 0.999 / peak, True
    fitted = np.pad(audio * gain, (0, max(0, len(target) - len(audio))))[: len(target)]
    sf.write(destination, fitted.astype(np.float32), sr, subtype="PCM_16")
    return {"sample_rate": sr, "export_num_samples": len(audio), "num_samples": len(fitted),
            "gain_db": 20 * math.log10(gain), "peak_limited": limited}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs-root", type=Path, required=True)
    args = parser.parse_args()
    docs = args.docs_root.resolve()
    page = docs / "index.html"
    source = page.read_text()
    manifests = {name: read_manifest(ROOT / directory) for name, directory, _, _ in ARMS}
    # Check every source before writing audio or changing the page.
    articles = re.findall(r'<article class="bar">.*?</article>', source, flags=re.S)
    if not articles:
        raise ValueError("expected the paper listening page with article.bar excerpts")
    examples = []
    for article in articles:
        identity = re.search(r'<span class="meta">([^<]+?) · beat (\d+) ·', article)
        prefix = re.search(r'src="audio/(\d+)_target.wav"', article)
        if identity is None or prefix is None:
            raise ValueError("example is missing its source identity or recording reference")
        key = (html.unescape(identity[1]), int(identity[2]))
        reference = docs / "audio" / f"{prefix[1]}_target.wav"
        if not reference.is_file():
            raise FileNotFoundError(reference)
        for name, directory, _, _ in ARMS:
            row = manifests[name][key]
            if row.get("split") != "test":
                raise ValueError(f"expected test export for {key}: {directory}")
            if not (ROOT / directory / row["wav"]).is_file():
                raise FileNotFoundError(ROOT / directory / row["wav"])
        examples.append((key, prefix[1], reference))

    provenance = []
    for key, prefix, reference in examples:
        item = {"example": prefix, "source_id": key[0], "beat_index": key[1], "audio": {}}
        for name, directory, _, _ in ARMS:
            row = manifests[name][key]
            export = Path(directory) / row["wav"]
            destination = docs / "audio" / f"{prefix}_{name}.wav"
            matching = match_loudness(ROOT / export, reference, destination)
            item["audio"][name] = {"file": f"audio/{destination.name}", "export": str(export), **matching}
        provenance.append(item)

    def lay_out(match: re.Match) -> str:
        pair, inner, tail = match[1], match[2], match[3]
        prefix = re.search(r'src="audio/(\d+)_target.wav"', inner)[1]
        target = re.search(r'<div class="cell target">.*?</audio></div>', inner, flags=re.S)
        ours = re.search(r'<div class="cell ours">.*?</audio></div>', inner, flags=re.S)
        if target is None or ours is None:
            raise ValueError(f"unexpected layout in example {prefix}")
        cells = "".join(
            f'<div class="cell {name}"><span class="lab">{title}{f"<em>{detail}</em>" if detail else ""}</span>'
            f'<audio controls preload="none" aria-label="Example {prefix}: {title}{f", {detail}" if detail else ""}" '
            f'src="audio/{prefix}_{name}.wav"></audio></div>'
            for name, _, title, detail in ARMS
        )
        ablations = f'{BEGIN}<div class="abl"><span class="abl-head">Ablations</span><div class="duo">{cells}</div></div>{END}'
        return f"{pair}{target[0]}\n      {ours[0]}\n      {ablations}{tail}"

    # The right column ends immediately before the TRIA section.
    source, count = re.subn(r'(<div class="pair">)(.*?)(</div>\n  </div>\n  <div class="tria">)', lay_out, source, flags=re.S)
    if count != len(examples):
        raise ValueError(f"laid out {count} of {len(examples)} examples")
    # Earlier versions of this script added a collapsible codec block; remove its leftovers.
    source = re.sub(r"/\* codec-target-controls \*/\n(?:\.codec-controls[^\n]*\n)*", "", source)
    source = re.sub(r'\s*<p class="codec-(?:audit-note|description)".*?</p>', "", source, flags=re.S)
    if "/* ablations */" not in source:
        source = source.replace("</style>", STYLE + "</style>", 1)
    if 'class="abl-description"' in source:
        source = re.sub(r'<p class="abl-description">.*?</p>', lambda _: DESCRIPTION, source, flags=re.S)
    else:
        source = source.replace('  </section>\n  <div class="list">', f'    {DESCRIPTION}\n  </section>\n  <div class="list">', 1)
    page.write_text(source)
    (docs / "ablation_examples.json").write_text(json.dumps({
        "split": "test",
        "normalization": "BS.1770 loudness matched to recording, peak capped at 0.999, fitted to the recording's length",
        "examples": provenance,
    }, indent=2) + "\n")
    print(f"Updated {page}: {len(examples)} examples, {len(examples) * len(ARMS)} ablation players")


if __name__ == "__main__":
    main()
