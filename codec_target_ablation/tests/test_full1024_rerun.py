"""Regression checks for the failed full-space experiment and rerun setup."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/experiment"))
sys.path.insert(0, str(ROOT / "codec_target_ablation"))

from export_projected import project_postsnap_latents
from model import validate_noise_prediction_dimensions


class FullSpaceRerunTests(unittest.TestCase):
    def test_training_refuses_bad_dimensions_before_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache = root / "cache"
            cache.mkdir()
            (cache / "config.json").write_text(json.dumps({"target_dim": 1024, "target_layout": "framewise_sum"}))
            run = root / "run"
            run.mkdir()
            sentinel = run / "preserve.txt"
            sentinel.write_text("existing run")
            result = subprocess.run([
                sys.executable, str(ROOT / "code/experiment/train_cli.py"),
                "--cache-root", str(cache), "--out-dir", str(run), "--overwrite",
                "--d-model", "768", "--device", "cpu", "--num-workers", "0",
            ], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("direct epsilon prediction requires d_model > x_dim", result.stderr)
            self.assertEqual(sentinel.read_text(), "existing run")

    def test_width_requirement_includes_final_layernorm(self):
        for width in (768, 1024):
            with self.assertRaises(ValueError):
                validate_noise_prediction_dimensions(x_dim=1024, d_model=width)
        validate_noise_prediction_dimensions(x_dim=1024, d_model=1280)
        validate_noise_prediction_dimensions(x_dim=72, d_model=768)

    def test_affine_projection_keeps_clean_coordinates_and_padding(self):
        torch.manual_seed(1234)
        u = torch.linalg.qr(torch.randn(12, 3), mode="reduced")[0].T
        mean = torch.randn(12)
        clean = mean + torch.randn(2, 5, 3) @ u
        noise = torch.randn_like(clean)
        perpendicular = noise - (noise @ u.T) @ u
        generated = clean + perpendicular
        generated[:, -1] = 0
        expected = clean.clone()
        expected[:, -1] = 0
        projected = project_postsnap_latents(generated, mean, u)
        torch.testing.assert_close(projected, expected, rtol=1e-5, atol=2e-6)
        with self.assertRaises(ValueError):
            project_postsnap_latents(torch.randn(1, 5, 3), mean, u)

    def test_rerun_selects_fresh_matched_models_and_rejects_legacy_width(self):
        cmd = ["bash", str(ROOT / "codec_target_ablation/run.sh"), "--dry-run", "--stages", "train"]
        result = subprocess.run(cmd, text=True, capture_output=True, check=True)
        self.assertIn("postsnap_full1024_d1280", result.stdout)
        self.assertIn("postsnap_pca72_d1280", result.stdout)
        self.assertNotIn("--d-model  768", result.stdout)
        bad = subprocess.run(cmd + ["--diffusion-width", "768"], text=True, capture_output=True)
        self.assertNotEqual(bad.returncode, 0)

    def test_legacy_notice_is_idempotent_and_cannot_mark_corrected_results(self):
        with tempfile.TemporaryDirectory() as folder:
            docs = Path(folder)
            page = docs / "index.html"
            page.write_text('<p class="codec-description">Controls</p>1024-D post-snap<em>old</em>1024-D + projection<em>old</em>')
            provenance = docs / "codec_target_examples.json"
            provenance.write_text('{"examples": []}')
            command = [sys.executable, str(ROOT / "codec_target_ablation/update_demo.py"), "--docs-root", str(docs), "--mark-legacy-invalid"]
            subprocess.run(command, check=True, capture_output=True)
            first = page.read_bytes()
            subprocess.run(command, check=True, capture_output=True)
            self.assertEqual(page.read_bytes(), first)
            self.assertEqual(first.count(b"codec-audit-note"), 1)
            self.assertEqual(first.count(b"(invalid run)"), 2)
            provenance.write_text('{"diffusion_width": 1280}')
            result = subprocess.run(command, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(page.read_bytes(), first)


if __name__ == "__main__":
    unittest.main()
