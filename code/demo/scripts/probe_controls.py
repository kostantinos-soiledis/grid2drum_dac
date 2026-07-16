"""Probe: does the sketch expander actually *listen* to each control?

For a fixed sketch + seed, sweep one control at a time from 0->1 (others at
default) and count the decoded events by group. If a group's count is flat while
its control rises, the model is ignoring that control.

Symbolic-level only (expander -> decode_event_plan); no diffusion / DAC needed,
so it runs in seconds on CPU. Audio can only reflect a control if the events do.

Usage:
    python scripts/probe_controls.py            # baseline sweep table
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import listen_ui as L  # noqa: E402
from scripts.sketch_diffusion_infer import _load_sketch_expander  # noqa: E402
from data.sketch_dataset import HIHAT_OPEN_CLASS_IDS, SNARE_BACKBEAT_STEPS  # noqa: E402

SEED = 1234
CKPT = Path(L.DEFAULT_SKETCH_CHECKPOINT)  # resolves via the listener default (RUNS_ROOT-aware)

# UI defaults (from build_ui), the "hold everything else here" baseline.
# Advanced ghost knobs default to -1 = "Auto" (follow the main Ghosts knob).
DEFAULTS = dict(
    feel_style="straight",
    feel_amount=0.25,
    ghost_density=0.35,
    kick_ghost_density=-1.0,
    snare_ghost_density=-1.0,
    hihat_density=0.52,
    hihat_openness=0.0,
    fill_density=0.0,
    fill_shape="down",
    crash_density=0.0,
)

TOM_FAMILIES = {"tom_high", "tom_mid", "tom_floor"}


def _load():
    model = _load_sketch_expander(str(CKPT), device=torch.device("cpu"))
    hits = L._hits_table_to_tensor(L._default_hits_table())
    vel = L._derive_velocity_matrix(hits, velocity=0.86, variation=0.20, seed=SEED)
    return model, hits, vel


def _decode(model, hits, vel, overrides, *, pattern_variation=0.0, chunk_idx=0, chunk_count=None):
    cfg = dict(DEFAULTS)
    cfg.update(overrides)
    controls = L._controls_tensor(
        cfg["feel_style"], cfg["feel_amount"], cfg["ghost_density"],
        cfg["kick_ghost_density"], cfg["snare_ghost_density"], cfg["hihat_density"],
        cfg["hihat_openness"], cfg["fill_density"], cfg["fill_shape"],
        control_names=model.cfg.control_names,
    )
    cc = int(chunk_count if chunk_count is not None else (chunk_idx + 1 if pattern_variation > 0 else 1))
    controls = L._vary_controls_for_chunk(
        controls, control_names=model.cfg.control_names, chunk_idx=chunk_idx,
        chunk_count=cc, pattern_variation=pattern_variation, seed=SEED,
    )
    with torch.no_grad():
        outputs = model(hits.unsqueeze(0), vel.unsqueeze(0), controls.unsqueeze(0))
    events = L._decode_event_plan_variant(
        {k: v.detach().cpu() for k, v in outputs.items()},
        sketch_hits=hits, sketch_vel=vel, controls=controls,
        class_names=model.class_names, class_id_vocab_sizes=model.class_id_vocab_sizes,
        control_names=model.cfg.control_names, budget_group_names=model.budget_group_names,
        budget_max_counts=model.budget_max_counts, seed=SEED,
        chunk_idx=chunk_idx, pattern_variation=pattern_variation,
    )
    return L._inject_crash_events(
        events, crash_density=float(cfg.get("crash_density", 0.0)), velocity=0.86,
        chunk_idx=chunk_idx, chunk_count=cc,
    )


def _counts(events):
    def is_ghost(e):  # non-anchor, non-backbeat
        return not bool(e.get("forced", False))
    kick = [e for e in events if e.get("family") == "kick"]
    snare = [e for e in events if e.get("family") == "snare"]
    hihat = [e for e in events if e.get("family") == "hihat"]
    toms = [e for e in events if e.get("family") in TOM_FAMILIES]
    crash = [e for e in events if e.get("family") == "crash"]
    ride = [e for e in events if e.get("family") == "ride"]
    return {
        "total": len(events),
        "kick_ghost": sum(1 for e in kick if is_ghost(e)),
        "snare_ghost": sum(
            1 for e in snare if is_ghost(e) and int(e.get("step", -1)) not in set(SNARE_BACKBEAT_STEPS)
        ),
        "hats": len(hihat),
        "hats_open": sum(1 for e in hihat if int(e.get("class_id", -1)) in set(HIHAT_OPEN_CLASS_IDS)),
        "toms": len(toms),
        "crash": len(crash),
        "ride": len(ride),
        "fill": len(toms) + len(crash) + len(ride),
    }


def _sweep(model, hits, vel, control, values, watch):
    print(f"\n=== sweep {control}  (watching: {watch}) ===")
    header = f"{control:>18} | " + " ".join(f"{w:>10}" for w in watch)
    print(header)
    print("-" * len(header))
    prev = None
    monotone = True
    for v in values:
        c = _counts(_decode(model, hits, vel, {control: v}))
        row = f"{v:>18.2f} | " + " ".join(f"{c[w]:>10d}" for w in watch)
        print(row)
        key = tuple(c[w] for w in watch)
        if prev is not None and key < prev:
            monotone = False
        prev = key
    first = tuple(_counts(_decode(model, hits, vel, {control: values[0]}))[w] for w in watch)
    last = tuple(_counts(_decode(model, hits, vel, {control: values[-1]}))[w] for w in watch)
    delta = "RESPONDS" if last != first else ">>> FLAT (ignored) <<<"
    print(f"  {control}: {first} -> {last}   [{delta}]")
    return last != first


def _sweep_over(model, hits, vel, control, values, watch, *, fixed):
    header = f"{control:>18} | " + " ".join(f"{w:>10}" for w in watch)
    print(header)
    print("-" * len(header))
    for v in values:
        ov = dict(fixed)
        ov[control] = v
        c = _counts(_decode(model, hits, vel, ov))
        print(f"{v:>18.2f} | " + " ".join(f"{c[w]:>10d}" for w in watch))


def main():
    model, hits, vel = _load()
    base = _counts(_decode(model, hits, vel, {}))
    print("baseline event counts:", base)
    vals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # Master ghost knob (advanced knobs on Auto) should now move BOTH kick+snare.
    _sweep(model, hits, vel, "ghost_density", vals, ["kick_ghost", "snare_ghost", "total"])
    # Overrides: pin the master low, move the advanced knob -> its family responds.
    print("\n--- override check: ghost_density fixed at 0.0, advanced knob overrides ---")
    _sweep_over(model, hits, vel, "snare_ghost_density", vals, ["snare_ghost"], fixed={"ghost_density": 0.0})
    _sweep_over(model, hits, vel, "kick_ghost_density", vals, ["kick_ghost"], fixed={"ghost_density": 0.0})
    _sweep(model, hits, vel, "hihat_density", vals, ["hats", "total"])
    _sweep(model, hits, vel, "hihat_openness", vals, ["hats_open", "hats"])
    _sweep(model, hits, vel, "fill_density", vals, ["toms", "crash", "ride", "fill"])
    # crash injection (chunk 0 of a 4-chunk phrase): downbeat crash count/velocity.
    print("\n=== sweep crash_density (chunk 0 of 4) ===")
    _sweep_over(model, hits, vel, "crash_density", vals, ["crash"], fixed={})

    # pattern_variation: does event set differ across chunks as it rises?
    print("\n=== sweep pattern_variation  (event-set difference chunk0 vs chunk3) ===")
    base_events = _decode(model, hits, vel, {"fill_density": 0.5}, pattern_variation=0.0, chunk_idx=0)
    base_keys = {(e["family"], e["step"], e.get("slot", 0)) for e in base_events}
    for pv in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        ev3 = _decode(model, hits, vel, {"fill_density": 0.5}, pattern_variation=pv, chunk_idx=3)
        keys3 = {(e["family"], e["step"], e.get("slot", 0)) for e in ev3}
        diff = len(base_keys ^ keys3)
        print(f"  pattern_variation={pv:.1f}  |chunk0 ^ chunk3| = {diff:>3d}  (n0={len(base_keys)}, n3={len(keys3)})")


if __name__ == "__main__":
    main()
