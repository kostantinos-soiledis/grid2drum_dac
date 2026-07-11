# Ready-To-Play Reviewer Tutorial

This tutorial starts from a fresh clone of the anonymous release repository and
walks through one complete interactive listening session.

## 1. Install And Launch

Clone the repository with Git LFS enabled so the model files are downloaded as
real weights instead of LFS pointer files. Install dependencies, then launch the
interactive Gradio app from the demo directory:

```bash
pip install -r requirements.txt
cd code/demo
python app.py
```

Open the local URL printed by Gradio, normally:

```text
http://127.0.0.1:7860
```

The first launch validates that all required model files are present. If it
reports missing model files, reclone or pull with Git LFS enabled.

To confirm the pipeline without opening a browser, run the deterministic CLI
smoke instead (writes `/tmp/drumtogrid_smoke_audio/output.wav`):

```bash
python scripts/sketch_diffusion_infer.py --sketch-json smoke_sketch.json --sketch-checkpoint ../../runs/sketch_expander_dac44_native_v5/best_sketch_expander.pt --diffusion-train-dir ../../runs/runs_dac/dac_25steps --cache-root ../../runs/mini_cache --device cpu --out-dir /tmp/drumtogrid_smoke_audio --overwrite
```

## 2. Render The Default Pattern

1. Leave the default kick, snare, and hihat grid unchanged.
2. Click `Render Grid`.
3. Inspect `Sketch and Events` to see the sparse user sketch expanded into a
   denser event plan.
4. Inspect `Conditioning Grid` to see the seconds-aligned model conditioning.
5. Click `Generate Audio`.
6. Press play in the `Audio` widget.

For the default 4-beat pattern on CPU, the 25-step diffusion checkpoints usually
take a few seconds. The 6-step checkpoint is faster.

## 3. Try The Main Controls

Use the grid as the hard rhythmic sketch: toggle kick, snare, or hihat cells in
the `Hits` table, then rerender.

The most useful controls for quick listening are:

- `Output beats`: render a longer loop; 8 or 16 beats repeats the sketch with
  variation across chunks.
- `Pattern variation`: increase for more changed fills and placements across
  chunks.
- `Feel style` and `Feel amount`: change timing character.
- `Ghosts`, `Kick ghosts`, and `Snare ghosts`: add soft supporting notes.
- `Hats` and `Hat openness`: change hat density and openness.
- `Fill amount` and `Fill shape`: add tom, ride, and crash-oriented fills.
- `Guidance`: increase if the output should follow the conditioning more
  strongly; high values may sound less natural.
- `Seed`: rerender the same controls with a different stochastic sample.

After any edit, click `Render Grid` for a fast visual check or `Generate Audio`
for listening.

## 4. Compare Checkpoints

The `Diffusion checkpoint` menu contains the shipped trained renderers:

- `plain diffusion 25-step`: baseline PCA diffusion.
- `RVQ-CE diffusion 25-step`: auxiliary-loss model (best quality).
- `direct PCA regressor`: deterministic one-pass baseline.

To compare them, keep the same grid, controls, and seed; switch the checkpoint;
then click `Generate Audio` again. The direct regressor ignores guidance and
seed because it is deterministic.

## 5. Files Produced By A Run

Each generation returns:

- `Audio`: the stitched loop.
- `Chunk WAVs`: per-chunk audio files for longer renders.
- `Events`: the expanded event plan and run metadata.
- `Sketch and Events`: a visual comparison of the user sketch and expanded
  events.
- `Conditioning Grid`: the continuous conditioning passed to the audio model.

Temporary run outputs are written under `.listen_ui_runs/` and are ignored by
Git.
