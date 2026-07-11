## Run The Demo Smoke

```bash
cd drumtogrid/code/demo
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

This writes `/tmp/drumtogrid_smoke_audio/output.wav`.

The interactive Gradio listener is included under `code/demo/app.py`; it expects
runtime files under `../../runs`; see
`runs/weights/manifest.json` for the required model payloads.
