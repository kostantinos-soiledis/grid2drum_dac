# Anonymous Interactive Drum Rendering Demo

This folder contains the standalone Gradio demo code for the paper. In this
curated layout, model files live in `../../runs` and are discovered by the demo
at runtime.

## Run Smoke

From this directory, install dependencies and run the deterministic CLI demo:

```bash
pip install -r ../../requirements.txt
python scripts/sketch_diffusion_infer.py \
  --sketch-json smoke_sketch.json \
  --sketch-checkpoint ../../runs/sketch_expander_dac44_native_v5/best_sketch_expander.pt \
  --diffusion-train-dir ../../runs/runs_dac/dac_25steps \
  --cache-root ../../runs/mini_cache \
  --device cpu \
  --out-dir /tmp/drumtogrid_smoke_audio \
  --overwrite
```

This writes:

```text
/tmp/drumtogrid_smoke_audio/output.wav
```

`app.py` launches the real interactive Gradio interface. Reviewers can edit the drum grid, velocity, timing feel, density controls, guidance scale, seed, and checkpoint choice, then render conditioning previews and generated audio.

For a ready-to-play walkthrough, see [`TUTORIAL.md`](TUTORIAL.md).

## Packaging

The required model files are expected under `../../runs`. They are not fetched
from an external URL at runtime, and the app does not require a separate
model-fetch step.

If the app reports missing model files, check `../../runs/weights/manifest.json`
against the files present in `../../runs`.

## Repository Layout

- `app.py`: standalone anonymous Gradio entrypoint.
- `demo_entry.py`: importable launcher used by `app.py`.
- `listen_ui.py`: interactive UI and inference orchestration.
- `data/`, `model.py`, `direct_regressor.py`, `sketch_expander.py`: inference support code.
- `../../runs/mini_cache`, `../../runs/sketch_expander_dac44_native_v5`,
  `../../runs/runs_dac`, `../../runs/runs_dac_ce`, and
  `../../runs/runs_direct`: model/cache files required by the app.
- `../experiment`: training, export, evaluation, and paper-result aggregation code.

Groove MIDI Dataset audio/MIDI files are not redistributed. Rebuilding training caches requires obtaining the dataset from its official source.

## Privacy Settings

The app disables Gradio analytics-related environment flags, launches with `show_error=False`, requests `enable_monitoring=False` when supported by the installed Gradio version, and does not log request headers, usernames, or IP addresses.
