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


def project_postsnap_latents(
    latent: torch.Tensor, mean: torch.Tensor, components: torch.Tensor,
) -> torch.Tensor:
    """Project unstandardized codec latents, preserving zero-padded frames."""
    mu, u = mean.to(latent), components.to(latent)
    if u.ndim != 2 or mu.ndim != 1 or latent.shape[-1] != mu.numel() or u.shape[1] != mu.numel():
        raise ValueError("projection expects full codec latents and a [K,D] basis; undo normalization first")
    projected = mu + ((latent - mu) @ u.T) @ u
    valid = latent.ne(0).any(dim=-1, keepdim=True)
    return torch.where(valid, projected, latent)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pca-basis", type=Path, required=True)
    args, export_argv = parser.parse_known_args()
    basis = torch.load(args.pca_basis, map_location="cpu", weights_only=False)
    mean = torch.as_tensor(basis["mean"], dtype=torch.float32)
    components = torch.as_tensor(basis["components"], dtype=torch.float32)
    decode = plain_export.decode_latent_to_audio
    write_json = plain_export.write_json

    def decode_projected(pred_latent: torch.Tensor, codec_model, **kwargs) -> torch.Tensor:
        if kwargs.get("target_pca_basis") is not None:
            raise ValueError("output projection requires a full-space checkpoint, not PCA coordinates")
        return decode(project_postsnap_latents(pred_latent, mean, components), codec_model, **kwargs)

    def write_projection_metadata(path, payload):
        if Path(path).name == "summary.json":
            payload = {**payload, "output_projection": {
                "basis_path": str(args.pca_basis), "rank": int(components.shape[0]),
                "after_denormalization": True, "sampling_unchanged": True,
            }}
        return write_json(path, payload)

    plain_export.decode_latent_to_audio = decode_projected
    plain_export.write_json = write_projection_metadata
    sys.argv = [sys.argv[0], *export_argv]
    plain_export.main()


if __name__ == "__main__":
    main()
