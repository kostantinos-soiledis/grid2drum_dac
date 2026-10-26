#!/usr/bin/env python3
"""Export postsnap_full1024 with each generated latent projected onto DAC's 72-D post-snap subspace.

Runs code/experiment/scripts/export_best_diffusion_predictions.py unchanged, with the plain export's
arguments and sampling seeds, except that every generated frame z is replaced by mu + (z - mu) U^T U
(mean and components of the post-snap PCA basis) before decoding. Real latents lie on this subspace,
so the projection removes only the part the model generated off it; padded (all-zero) frames are kept.

  python export_projected.py --pca-basis <cache>/pca_basis.pt <export_best_diffusion_predictions args>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1] / "code" / "experiment"
sys.path.insert(0, str(EXPERIMENT_ROOT))

from scripts import export_best_diffusion_predictions as plain_export  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pca-basis", type=Path, required=True)
    args, export_argv = parser.parse_known_args()
    basis = torch.load(args.pca_basis, map_location="cpu", weights_only=False)
    mean = torch.as_tensor(basis["mean"], dtype=torch.float32)
    components = torch.as_tensor(basis["components"], dtype=torch.float32)
    decode = plain_export.decode_latent_to_audio

    def decode_projected(pred_latent: torch.Tensor, codec_model, **kwargs) -> torch.Tensor:
        mu, u = mean.to(pred_latent), components.to(pred_latent)
        projected = mu + ((pred_latent - mu) @ u.T) @ u
        valid = pred_latent.ne(0).any(dim=-1, keepdim=True)
        return decode(torch.where(valid, projected, pred_latent), codec_model, **kwargs)

    plain_export.decode_latent_to_audio = decode_projected
    sys.argv = [sys.argv[0], *export_argv]
    plain_export.main()


if __name__ == "__main__":
    main()
