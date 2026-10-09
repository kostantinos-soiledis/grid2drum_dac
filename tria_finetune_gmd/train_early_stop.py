#!/usr/bin/env python3
"""TRIA's scripts/train.py with early stopping, for the GMD fine-tune (train.sh, README.md).

The trainer runs unchanged; one hook: before each training iteration, if the last PATIENCE validation checks brought no
new best validation token cross-entropy (TRIA's own criterion for `best/`), the run ends. The count comes from the
tracker's validation history, which the checkpoint carries, so it survives a resume. Run from upstream/ (TRIA's paths
are relative to its checkout), with the same arguments as train.py (--args.load <yml>).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

PATIENCE = int(os.environ.get("TRIA_PATIENCE", "5"))  # validation checks without a new best
KEY = "loss/cross_entropy"

upstream = Path.cwd()
sys.path.insert(0, str(upstream / "scripts"))
spec = importlib.util.spec_from_file_location("tria_train", upstream / "scripts" / "train.py")
tria = importlib.util.module_from_spec(spec)
sys.modules["tria_train"] = tria
spec.loader.exec_module(tria)


def stale_checks(history: dict) -> int:
    """Validation checks since the best one."""
    losses = list(history.get("val", {}).get(KEY, []))
    return len(losses) - 1 - losses.index(min(losses)) if losses else 0


class EarlyStop(Exception):
    pass


_train_loop = tria.train_loop


def train_loop(state, batch, accel):
    history = state.tracker.history
    if stale_checks(history) >= PATIENCE:
        raise EarlyStop(history)
    return _train_loop(state, batch, accel)


tria.train_loop = train_loop  # train() wraps the module's train_loop when it starts


if __name__ == "__main__":
    import argbind

    args = argbind.parse_args()
    args["args.debug"] = int(os.getenv("LOCAL_RANK", 0)) == 0
    save_path = Path(args["save_path"])
    try:
        with argbind.scope(args):
            with tria.Accelerator() as accel:
                tria.accel = accel  # train.py's checkpoint() reads the module's accel, set by its __main__
                tria.train(args, accel)
        stopped_early, history = False, None
    except EarlyStop as stop:
        stopped_early, history = True, stop.args[0]
    if history is None:
        extras = __import__("torch").load(save_path / "latest" / "extras.pt", map_location="cpu", weights_only=False)
        history = extras["tracker"]["history"]
    val = history["val"]
    losses = [float(v) for v in val[KEY]]
    if not losses:
        raise SystemExit("no validation check ran")
    best = losses.index(min(losses))
    done = {"stopped_early": stopped_early, "patience": PATIENCE, "last_check_step": int(val["step"][-1]),
            "best_step": int(val["step"][best]), "best_val_cross_entropy": losses[best],
            "checks": [{"step": int(s), "val_cross_entropy": v} for s, v in zip(val["step"], losses)]}
    (save_path / "done.json").write_text(json.dumps(done, indent=1) + "\n")
    print(json.dumps({k: v for k, v in done.items() if k != "checks"}), flush=True)
