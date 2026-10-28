#!/usr/bin/env python3
"""Add codec target comparisons to the paper's existing static listening page.

Run against the docs directory on the anonymous branch. Existing example
identities, EDG plots, recording references, and TRIA players are retained.
Audio comes from the evaluated exports and uses the page's BS.1770 loudness
matching, with gain limited to keep peaks below 0.999.

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
ARMS = (
    ("pca72", "runs/final/exports/postsnap_pca", "72-D post-snap", "plain diffusion, 25 steps"),
    ("tokens", "codec_target_ablation/exports/tokens_9x1024", "Discrete tokens", "parallel classifier, argmax"),
)
BEGIN = "<!-- codec-target-controls:start -->"
END = "<!-- codec-target-controls:end -->"
DESCRIPTION = (
    '<p class="codec-description">Expand <strong>Codec target ablation</strong> '
    'for 72-D post-snap diffusion and parallel discrete-token classification. '
    'Both use the 250 Hz EDG without the auxiliary RVQ-CE loss; our main model '
    'above uses that loss.</p>'
)
STYLE = """/* codec-target-controls */
.codec-controls { border-top: 1px solid var(--line); padding-top: 10px; }
.codec-controls summary { cursor: pointer; font-size: 14px; font-weight: 600; color: var(--ours); }
.codec-controls summary:focus-visible { outline: 2px solid var(--ours); outline-offset: 3px; }
.codec-controls .duo { margin-top: 12px; }
.codec-controls .cell { border-top-color: var(--abl); }
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
    sf.write(destination, (audio * gain).astype(np.float32), sr, subtype="PCM_16")
    return {"sample_rate": sr, "num_samples": len(audio), "gain_db": 20 * math.log10(gain), "peak_limited": limited}


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

    def extend_article(match: re.Match) -> str:
        article = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), "", match[0], flags=re.S)
        prefix = re.search(r'src="audio/(\d+)_target.wav"', article)[1]
        cells = [
            f'<div class="cell {name}"><span class="lab">{title}<em>{detail}</em></span>'
            f'<audio controls preload="none" aria-label="Example {prefix}: {title}" '
            f'src="audio/{prefix}_{name}.wav"></audio></div>'
            for name, _, title, detail in ARMS
        ]
        controls = BEGIN + '<details class="codec-controls"><summary>Codec target ablation</summary>'
        controls += '<div class="duo">' + "".join(cells) + '</div></details>' + END
        # The pair ends immediately before the TRIA section; keep both groups intact.
        anchor = '</div>\n  </div>\n  <div class="tria">'
        if article.count(anchor) != 1:
            raise ValueError(f"unexpected layout in example {prefix}")
        return article.replace(anchor, controls + anchor)

    source = re.sub(r'<article class="bar">.*?</article>', extend_article, source, flags=re.S)
    if "/* codec-target-controls */" not in source:
        source = source.replace("</style>", STYLE + "</style>", 1)
    source = re.sub(r'<p class="codec-audit-note".*?</p>', "", source, flags=re.S)
    if 'class="codec-description"' in source:
        source = re.sub(r'<p class="codec-description">.*?</p>', lambda _: DESCRIPTION, source, flags=re.S)
    else:
        source = source.replace('  </section>\n  <div class="list">', f'    {DESCRIPTION}\n  </section>\n  <div class="list">', 1)
    page.write_text(source)
    (docs / "codec_target_examples.json").write_text(json.dumps({
        "split": "test", "normalization": "BS.1770 loudness matched to recording, peak capped at 0.999",
        "examples": provenance,
    }, indent=2) + "\n")
    print(f"Updated {page}: {len(examples)} examples, {len(examples) * len(ARMS)} new audio players")


if __name__ == "__main__":
    main()
