#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON:-python}"

exec "$PYTHON_BIN" -c '
import os
import sys
import time
from pathlib import Path

here = Path.cwd().resolve()
package = here.parent.parent
runs = package / "runs"
os.chdir(package)
sys.path.insert(0, str(here))

import listen_ui

port_env = os.environ.get("PORT") or os.environ.get("GRADIO_SERVER_PORT")
port = int(port_env) if port_env else None
argv = [
    "--device", os.environ.get("APP_DEVICE", "cpu"),
    "--cache-root", str(runs / "mini_cache"),
    "--diffusion-train-dir", str(runs / "runs_dac" / "dac_25steps"),
    "--sketch-checkpoint", str(runs / "sketch_expander_dac44_native_v5" / "best_sketch_expander.pt"),
    "--server-name", os.environ.get("SERVER_NAME", "127.0.0.1"),
]
if port is not None:
    argv.extend(["--server-port", str(port)])

args = listen_ui._parse_args(argv)
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
if port is not None:
    launch_kwargs["server_port"] = port

try:
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
'
