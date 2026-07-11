"""Standalone Gradio launcher for the packaged drum rendering demo."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent.parent
RUNS_ROOT = PACKAGE_ROOT / "runs"

os.chdir(PACKAGE_ROOT)
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import listen_ui  # noqa: E402

PORT_ENV = os.environ.get("PORT") or os.environ.get("GRADIO_SERVER_PORT")
PORT = int(PORT_ENV) if PORT_ENV else None

args = listen_ui._parse_args(
    [
        "--device",
        os.environ.get("APP_DEVICE", "cpu"),
        "--cache-root",
        str(RUNS_ROOT / "mini_cache"),
        "--diffusion-train-dir",
        str(RUNS_ROOT / "runs_dac" / "dac_25steps"),
        "--sketch-checkpoint",
        str(RUNS_ROOT / "sketch_expander_dac44_native_v5" / "best_sketch_expander.pt"),
        "--server-name",
        os.environ.get("SERVER_NAME", "127.0.0.1"),
    ]
    + ([] if PORT is None else ["--server-port", str(PORT)])
)

app = listen_ui.SketchDiffusionListenApp(args)
demo = listen_ui.build_ui(app)
demo.queue(max_size=8)

launch_kwargs = {
    "server_name": str(args.server_name),
    "prevent_thread_lock": True,
    "show_error": False,
    "enable_monitoring": False,
    "ssr_mode": False,
}
if PORT is not None:
    launch_kwargs["server_port"] = PORT

try:
    main_module = sys.modules.get("__main__")
    if main_module is not None and hasattr(main_module, "__file__"):
        delattr(main_module, "__file__")
    demo.launch(**launch_kwargs)
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
