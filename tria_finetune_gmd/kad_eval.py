#!/usr/bin/env python3
"""KAD of each system's GMD test clips against the real bars: one number per system, with recording-clustered statistics.

KAD (Chung et al. 2025, "KAD: No More FAD!"), computed with kadtk, the implementation TRIA's paper uses:
  embedding   PANNs-WGLM (Wavegram-LogMel CNN14, 32 kHz; kadtk's "panns-wavegram-logmel"), the embedding the KAD paper
              finds most correlated with human judgments and TRIA's paper calls "PANN". kadtk's own loader and model
              code extract one 2048-d embedding per clip (sox/ffmpeg resampling, cached under <work>/<system>/embeddings/).
  clips       the real bars, and each system's clips as evaluate_metrics.py stages them for every metric: gain-matched
              to their real bar's integrated loudness (ITU-R BS.1770), so KAD compares what the drums sound like rather
              than output levels.
  kernel      Gaussian, bandwidth = the median pairwise distance of the real bars' embeddings (the KAD paper's
              definition; kadtk's code would take it from each evaluated set), so every system is scored with one kernel.
  estimator   kadtk.kad.calc_kernel_audio_distance with that bandwidth: unbiased MMD^2, x100.
Statistics treat recordings as the unit, as clustered_paired_stats.py does for the per-clip metrics:
  CI          B bootstrap draws resample the test recordings with replacement and weight every clip (its real and its
              generated version) by how often its recording was drawn; each draw recomputes KAD with the same kernel and
              the unbiased weighted estimator, which leaves out each clip's pair with itself, so equal weights give
              kadtk's value exactly (checked). 95% percentile CIs per system and per pair difference (A - B).
  p           paired permutation test per pair: under the null that A and B generate alike, swapping A's and B's clips of
              a recording changes nothing, so B random swaps of whole recordings give the null distribution of
              KAD(A) - KAD(B) (the analogue of clustered_paired_stats.py's recording-level sign flip); two-sided,
              p = (1 + #{|null| >= |observed|}) / (B + 1).

    .venv_kad/bin/python kad_eval.py --reference <eval>/acoustic_eval/fad_assets/reference_audio \\
        --system grid_250hz=<eval>/systems/grid_250hz --system tria_finetuned_midi=<eval>/systems/tria_finetuned_midi \\
        --pairs tria_finetuned_midi:grid_250hz ... --out results/stats/kad.json --work <eval>/kad --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

MODEL = "panns-wavegram-logmel"


def model_loader():
    """kadtk's PANNs-WGLM loader (built alone: kadtk's get_all_models() builds every model, and some download their
    weights on construction); loaded by kadtk's embedding workers, one copy per worker."""
    from kadtk.model_loader import PANNsModel

    ml = PANNsModel("wavegram-logmel")
    assert ml.name == MODEL, ml.name
    return ml


def stage(stems: list[str], wav_dir: Path, out: Path) -> list[Path]:
    """Links to the clips under out/, where kadtk writes its resampled audio and embeddings."""
    out.mkdir(parents=True, exist_ok=True)
    for s in stems:
        dst = out / f"{s}.wav"
        if not (dst.exists() or dst.is_symlink()):
            dst.symlink_to((wav_dir / f"{s}.wav").resolve())
    return [out / f"{s}.wav" for s in stems]


def embed(ml, files: list[Path], workers: int) -> torch.Tensor:
    """kadtk's embedding of every file (frames averaged if the model gives several), in order."""
    from kadtk.emb_loader import cache_embedding_files

    cache_embedding_files(files, ml, workers=workers)
    rows = []
    for f in files:
        e = np.load(f.parent / "embeddings" / ml.name / f"{f.stem}.npy").astype(np.float32)
        rows.append(e.reshape(-1, e.shape[-1]).mean(axis=0))
    return torch.from_numpy(np.stack(rows))


class Kad:
    """kadtk's KAD terms against the real bars at a fixed bandwidth, for weighted and permuted recomputation."""

    def __init__(self, ref: torch.Tensor, bandwidth: float, device):
        from kadtk.kad import SCALE_FACTOR

        self.gamma, self.device, self.scale = 1 / (2 * bandwidth ** 2 + 1e-8), device, SCALE_FACTOR
        self.ref = ref.to(device, torch.float32)
        self.k_xx = self.kernel(self.ref, self.ref)

    def kernel(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.gamma * torch.cdist(a.to(self.device), b.to(self.device)).square())

    def weighted(self, gen: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """KAD for clip weights w [B, n] (the same clips weigh in both sets); equal weights give kadtk's value."""
        k_yy, k_xy = self.kernel(gen, gen), self.kernel(self.ref, gen)
        w2, tot = w.square().sum(1), w.sum(1)

        def within(K):  # unbiased: drop each clip's pair with itself
            return (((w @ K) * w).sum(1) - (w.square() * torch.diagonal(K)).sum(1)) / (tot.square() - w2)

        return self.scale * (within(self.k_xx) + within(k_yy) - 2 * ((w @ k_xy) * w).sum(1) / tot.square())

    def swapped_difference(self, a: torch.Tensor, b: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """KAD(A') - KAD(B') where A' takes B's clip wherever s [P, n] is 1 and B' takes A's; the real-bar term cancels."""
        k_aa, k_bb, k_ab = self.kernel(a, a), self.kernel(b, b), self.kernel(a, b)
        c_a, c_b = self.kernel(self.ref, a).sum(0), self.kernel(self.ref, b).sum(0)
        d_a, d_b = torch.diagonal(k_aa), torch.diagonal(k_bb)
        t, n = 1 - s, s.shape[1]

        def quad(u, K, v):
            return ((u @ K) * v).sum(1)

        within_a = quad(t, k_aa, t) + quad(s, k_bb, s) + 2 * quad(t, k_ab, s) - (t * d_a + s * d_b).sum(1)
        within_b = quad(s, k_aa, s) + quad(t, k_bb, t) + 2 * quad(s, k_ab, t) - (s * d_a + t * d_b).sum(1)
        cross = (t @ c_a + s @ c_b) - (t @ c_b + s @ c_a)
        return self.scale * ((within_a - within_b) / (n * (n - 1)) - 2 * cross / n ** 2)


def ci(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=Path, required=True, help="folder of the real bars (<stem>.wav)")
    ap.add_argument("--system", action="append", required=True, help="name=staged dir (wavs/ + manifest.jsonl)")
    ap.add_argument("--pairs", nargs="*", default=[], help='"A:B" (difference A - B)')
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-items", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True, help="staging folder for kadtk's audio and embeddings")
    args = ap.parse_args()

    from kadtk.kad import calc_kernel_audio_distance, median_pairwise_distance

    systems = dict(s.split("=", 1) for s in args.system)
    rows = sorted((json.loads(l) for l in (Path(next(iter(systems.values()))) / "manifest.jsonl").open()),
                  key=lambda r: int(r["dataset_index"]))
    rows = rows[: args.max_items] if args.max_items else rows
    stems = [Path(r["wav"]).stem for r in rows]
    for name, d in systems.items():  # every system covers the same clips under the same names
        other = {Path(json.loads(l)["wav"]).stem for l in (Path(d) / "manifest.jsonl").open()}
        assert set(stems) <= other, f"{name} lacks clips"
    recordings = sorted({r["source_id"] for r in rows})
    clip_recording = torch.tensor([recordings.index(r["source_id"]) for r in rows])
    gen = torch.Generator().manual_seed(args.seed)
    draws = torch.randint(len(recordings), (args.reps, len(recordings)), generator=gen)
    counts = torch.zeros(args.reps, len(recordings)).scatter_add_(1, draws, torch.ones(args.reps, len(recordings)))
    weights = counts[:, clip_recording].to(args.device)  # [B, clips]
    swaps = torch.randint(2, (args.reps, len(recordings)), generator=gen)[:, clip_recording].float().to(args.device)

    ml = model_loader()
    ref = embed(ml, stage(stems, args.reference, args.work / "reference"), args.workers)
    bandwidth = float(median_pairwise_distance(ref))
    kad = Kad(ref, bandwidth, args.device)
    out = {"embedding": MODEL, "bandwidth": bandwidth, "reference": str(args.reference), "num_clips": len(rows),
           "num_recordings": len(recordings), "reps": args.reps, "seed": args.seed, "kadtk_scale": kad.scale,
           "systems": {}, "pairs": {}}
    emb, boot = {}, {}
    for name, d in systems.items():
        emb[name] = embed(ml, stage(stems, Path(d) / "wavs", args.work / name), args.workers)
        value = float(calc_kernel_audio_distance(ref, emb[name], (None, None), args.device, bandwidth=bandwidth))
        full = float(kad.weighted(emb[name], torch.ones(1, len(rows), device=args.device))[0])
        assert math.isclose(full, value, rel_tol=1e-3, abs_tol=1e-4), (name, full, value)
        boot[name] = torch.cat([kad.weighted(emb[name], w) for w in weights.split(250)]).cpu().numpy()
        summary = json.loads((Path(d) / "summary.json").read_text()) if (Path(d) / "summary.json").is_file() else {}
        out["systems"][name] = {"kad": value, "ci95": ci(boot[name]), "loudness_match": summary.get("loudness_match")}
        print(name, f"KAD {value:.4f}", [round(v, 4) for v in out["systems"][name]["ci95"]], flush=True)
        torch.cuda.empty_cache()
    for pair in args.pairs:
        a, b = pair.split(":")
        observed = out["systems"][a]["kad"] - out["systems"][b]["kad"]
        none = torch.zeros(1, len(rows), device=args.device)
        unswapped = float(kad.swapped_difference(emb[a], emb[b], none)[0])
        assert math.isclose(unswapped, observed, rel_tol=1e-3, abs_tol=1e-4), (pair, unswapped, observed)
        null = torch.cat([kad.swapped_difference(emb[a], emb[b], s) for s in swaps.split(250)]).cpu().numpy()
        p = (1 + np.count_nonzero(np.abs(null) >= abs(observed) - 1e-12)) / (len(null) + 1)
        out["pairs"][pair] = {"diff": observed, "ci95": ci(boot[a] - boot[b]), "p_permutation": float(p)}
        print(pair, f"diff {observed:+.4f}", [round(v, 4) for v in out["pairs"][pair]["ci95"]], f"p {p:.4g}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
