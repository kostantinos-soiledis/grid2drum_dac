"""Standalone anonymous Gradio entrypoint for the interactive drum renderer."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent.parent
RUNS_ROOT = PACKAGE_ROOT / "runs"
os.chdir(PACKAGE_ROOT)
sys.path.insert(0, str(HERE))


def _patch_torchaudio_save() -> None:
    import torch
    import torchaudio
    import soundfile as sf

    if getattr(torchaudio.save, "_anonymous_soundfile_save", False):
        return

    def _subtype_from_torchaudio_args(encoding: str | None, bits_per_sample: int | None) -> str | None:
        if encoding is None and bits_per_sample is None:
            return None
        enc = str(encoding or "PCM_S").upper()
        bits = None if bits_per_sample is None else int(bits_per_sample)
        if enc == "PCM_S" and bits in (16, 24, 32):
            return f"PCM_{bits}"
        if enc == "PCM_U" and bits == 8:
            return "PCM_U8"
        if enc in {"PCM_F", "FLOAT"}:
            if bits == 64:
                return "DOUBLE"
            if bits in (None, 32):
                return "FLOAT"
        if enc in {"ULAW", "ALAW"}:
            return enc
        return None

    def _soundfile_save(
        uri,
        src,
        sample_rate: int,
        channels_first: bool = True,
        format: str | None = None,
        encoding: str | None = None,
        bits_per_sample: int | None = None,
        buffer_size: int = 4096,
        backend: str | None = None,
        compression=None,
        **kwargs,
    ) -> None:
        del buffer_size, backend, compression, kwargs
        audio = torch.as_tensor(src).detach().cpu()
        if audio.dtype in (torch.float16, torch.bfloat16):
            audio = audio.to(torch.float32)
        if int(audio.dim()) == 1:
            audio = audio.unsqueeze(0 if channels_first else 1)
        elif int(audio.dim()) != 2:
            raise ValueError(f"Expected 1D or 2D audio tensor, got shape {tuple(audio.shape)}")
        if channels_first:
            audio = audio.transpose(0, 1)
        data = audio.contiguous().numpy()
        if isinstance(uri, (str, os.PathLike)):
            Path(uri).parent.mkdir(parents=True, exist_ok=True)
        sf.write(
            uri,
            data,
            int(sample_rate),
            format=format,
            subtype=_subtype_from_torchaudio_args(encoding, bits_per_sample),
        )

    _soundfile_save._anonymous_soundfile_save = True
    torchaudio.save = _soundfile_save


def _expected_weight_paths() -> list[Path]:
    manifest_path = RUNS_ROOT / "weights" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [RUNS_ROOT / str(item["path"]) for item in manifest.get("files", [])]


def _validate_local_weights() -> None:
    missing_or_pointer = [path for path in _expected_weight_paths() if not _is_real_weight_file(path)]
    if not missing_or_pointer:
        return
    rel = "\n".join(f"  - {path.relative_to(RUNS_ROOT)}" for path in missing_or_pointer[:12])
    if len(missing_or_pointer) > 12:
        rel += f"\n  - ... {len(missing_or_pointer) - 12} more"
    raise RuntimeError(
        "Required Git LFS model files are missing or were cloned as pointer files:\n"
        f"{rel}\n\n"
        "Check ../../runs/weights/manifest.json and ensure the listed runtime files are present."
    )


def _is_real_weight_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix == ".pt" and path.stat().st_size < 1024:
        head = path.read_bytes()[:128]
        if b"git-lfs" in head:
            return False
    return True


import listen_ui  # noqa: E402

PORT_ENV = os.environ.get("PORT") or os.environ.get("GRADIO_SERVER_PORT")
PORT = int(PORT_ENV) if PORT_ENV else None

argv = [
    "--device", os.environ.get("APP_DEVICE", "cpu"),
    "--cache-root", str(RUNS_ROOT / "mini_cache"),
    "--diffusion-train-dir", str(RUNS_ROOT / "runs_dac" / "dac_25steps"),
    "--sketch-checkpoint", str(RUNS_ROOT / "sketch_expander_dac44_native_v5" / "best_sketch_expander.pt"),
    "--server-name", os.environ.get("SERVER_NAME", "127.0.0.1"),
]
if PORT is not None:
    argv.extend(["--server-port", str(PORT)])

args = listen_ui._parse_args(argv)
app = listen_ui.SketchDiffusionListenApp(args)
demo = listen_ui.build_ui(app)
demo.queue(max_size=8)
launch_kwargs = {
    "server_name": str(args.server_name),
    "show_error": False,
    "enable_monitoring": False,
    "prevent_thread_lock": True,
    "ssr_mode": False,
    "footer_links": [],
}
if PORT is not None:
    launch_kwargs["server_port"] = PORT
try:
    demo.launch(**{key: value for key, value in launch_kwargs.items() if key != "footer_links"})
    while True:
        time.sleep(3600)
except KeyboardInterrupt:
    pass
finally:
    try:
        demo.close()
    except Exception:
        pass
    app.close()
