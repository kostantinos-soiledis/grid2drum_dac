#!/usr/bin/env python
"""Strip non-inference state (optimizer) from a diffusion checkpoint.

Keeps model_state_dict and all embedded metadata (target_mean/std,
codec_metadata, frontend_cfg, config, target_pca_basis, ...) so the listener
app loads it unchanged; drops optimizer_state_dict, roughly halving file size
for upload to the demo Space.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

DROP = ["optimizer_state_dict"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = torch.load(a.inp, map_location="cpu", weights_only=False)
    if isinstance(d, dict):
        for k in DROP:
            d.pop(k, None)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(d, out)
    src_mb = Path(a.inp).stat().st_size / 1e6
    dst_mb = out.stat().st_size / 1e6
    print(f"{a.inp} ({src_mb:.0f} MB) -> {a.out} ({dst_mb:.0f} MB)")


if __name__ == "__main__":
    main()
