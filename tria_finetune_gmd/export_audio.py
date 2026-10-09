"""
Baseline: drum grid -> audio with TRIA, an audio-prompted drum generator, on the GMD test split.

Model: "The Rhythm In Anything: Audio-Prompted Drums Generation with Masked
       Language Modeling" (O'Reilly et al., ISMIR 2025)
       https://github.com/interactiveaudiolab/tria (checked out in upstream/)

TRIA takes two audio prompts: a rhythm prompt, which it reduces to a two-band
transient envelope (onset times, loudness and band), and a timbre prompt, of
which it keeps at most 2 s (TIMBRE_SEC) to learn the kit. --prompts picks them:
  bar   both prompts are the clip's own bar from its GMD recording (bar_audio),
        so TRIA hears the target;
  midi  the rhythm prompt is the bar's notes, those our model is conditioned on,
        rendered with a General MIDI soundfont (midi_prompt.py), and the timbre
        prompt is 2 s of the same recording next to the bar, never overlapping
        it (timbre_window): TRIA never hears the bar, as our model never does.

--checkpoint is "released" (the authors' small_musdb_moises_fsl_2b; GMD is not in
its training data) or a fine-tuned model.pt (train.sh). Inference calls upstream
app._inference with mono output and the sampler settings of the checkpoint's
training config (SAMPLING), seed + dataset index per clip.

Environment: tria_finetune_gmd/.venv (README.md). FluidSynth on PATH for --prompts midi.

Usage:
    .venv/bin/python export_audio.py --checkpoint runs/gmd_long/best/model.pt --prompts midi \\
        --out-dir predictions/finetuned_midi --gmd-root $GMD_ROOT --soundfont $SOUNDFONT --device cuda:0
    .venv/bin/python export_audio.py --checkpoint released --prompts bar --out-dir predictions/released_bar --max-items 4

Outputs (--out-dir):
    wavs/<clip>.wav   44.1 kHz mono clips, same names and dataset indices as
                      the other exports
    manifest.jsonl    one row per clip (readable by run_diffusion_acoustic_eval.py)
    summary.json      settings
Clips that already exist in wavs/ are not regenerated unless --overwrite. --shard K/N renders only the clips whose
dataset index is K mod N and writes neither manifest nor summary: run N shards side by side, then once without --shard.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
UPSTREAM = HERE / "upstream"
CACHE_ROOT = REPO / "caches" / "cache_4beats_dac44q9_pca72_native_bpmgeom_duration_v1"

MODEL = "small_musdb_moises_fsl_2b"
SAMPLE_RATE = 44_100

# Sampler settings from the checkpoint's training config
# (upstream/conf/small_musdb_moises_fsl_2b.yml, used by the authors' save_samples):
# top-p 0.85 and no restriction of the vocabulary to the timbre prompt's codes.
# The Gradio app instead defaults to no top-p and restricting all 9 codebooks;
# in a listening test that dropped hits without improving the timbre.
# LOCKED for the paper baseline (decided 2026-10-06): the paper cites the
# checkpoint's own config as the source, so these values must not be tuned.
# The retrieval keys are unused while retrieval is off, but _inference reads them.
SAMPLING = dict(top_p=0.85, top_k=None, temperature=1.0, mask_temperature=10.5)
INFERENCE = dict(
    causal_bias=1.0, cfg_scale=2.0, timbre_vocab_codebooks=0,
    pseudo_stereo_codebook=None, pseudo_stereo_width=1.0,
    rhythm_retrieval=False, timbre_retrieval=False, retrieval_method="greedy",
    retrieval_codebooks=5, retrieval_window=15, retrieval_align="center",
    retrieval_match_threshold=0.9, retrieval_activity_threshold=0.25,
    window_overlap_frames=10, loopable=False, loop_pad_frames=10,
)
AUDIO = dict(loudness_db=-20.0, filter_inputs=False)
SCHEDULE = [8] * 9  # iterations per codebook
TIMBRE_SEC = 2.0  # TRIA keeps at most 2 s of the timbre prompt (a third of its 6 s buffer)
PROMPTS = {
    "bar": "rhythm and timbre: the clip's own bar from its GMD recording",
    "midi": "rhythm: the bar's notes (our model's grid onsets and notes from up to 30 ms before the bar) rendered with "
            "FluidSynth and a General MIDI soundfont, read 5 ms late for the render's onset latency; timbre: 2 s of the "
            "same recording next to the bar, not overlapping it",
}


def read_mono(audio_path, start_sec, num_sec):
    audio, sr = sf.read(audio_path, start=round(start_sec * SAMPLE_RATE), frames=round(num_sec * SAMPLE_RATE),
                        dtype="float32", always_2d=True)
    assert sr == SAMPLE_RATE, audio_path
    return audio.mean(axis=1)


def bar_audio(row, source_row, audio_path):
    """The clip's own bar from its GMD recording, mono."""
    assert (source_row["source_id"], source_row["beat_index"]) == (row["source_id"], row["beat_index"])
    return read_mono(audio_path, source_row["start_sec"], row["duration_sec"])


def timbre_window(row, source_row, audio_path):
    """Start and length (s) of the timbre prompt within the clip's GMD recording.

    TIMBRE_SEC next to the bar but never overlapping it, as TRIA's training
    context never overlaps its target: the TIMBRE_SEC before the bar, else the
    TIMBRE_SEC after it, else the longer of the two sides.
    """
    start = float(source_row["start_sec"])
    end = start + float(row["duration_sec"])
    after = sf.info(str(audio_path)).duration - end
    if start >= TIMBRE_SEC:
        return start - TIMBRE_SEC, TIMBRE_SEC
    if after >= TIMBRE_SEC:
        return end, TIMBRE_SEC
    return (0.0, start) if start >= after else (end, after)


def load_tria(device, checkpoint="released"):
    """The upstream app with MODEL loaded: the released weights, or a fine-tuned model.pt of the same model."""
    sys.path.insert(0, str(UPSTREAM))
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    import app  # upstream Gradio app; importing builds its UI but does not launch it

    config = app.MODEL_ZOO[MODEL]
    weights = UPSTREAM / config["checkpoint"] if checkpoint == "released" else Path(checkpoint).resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    config["checkpoint"] = str(weights)  # upstream path is cwd-relative
    app.DEVICE = torch.device(device)
    print(app.load_model_by_name(MODEL), weights, flush=True)
    return app


def to_signal(audio):
    from audiotools import AudioSignal

    return AudioSignal(torch.as_tensor(audio, dtype=torch.float32)[None, None], sample_rate=SAMPLE_RATE)


def clip_file_name(dataset_index, source_id, beat_index):
    return f"{dataset_index:06d}__{source_id.replace('/', '_')}__beat_{int(beat_index):04d}.wav"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="released", help='"released" or a fine-tuned model.pt')
    parser.add_argument("--prompts", choices=sorted(PROMPTS), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--split", default="test")
    parser.add_argument("--gmd-root", type=Path, default=os.environ.get("GMD_ROOT"),
                        help="GMD's groove/ folder (info.csv, audio, MIDI); default $GMD_ROOT")
    parser.add_argument("--soundfont", type=Path, default=os.environ.get("SOUNDFONT"),
                        help="General MIDI soundfont for --prompts midi (FluidR3_GM.sf2); default $SOUNDFONT")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234, help="clip i uses seed + i")
    parser.add_argument("--max-items", type=int, default=0, help="0 = whole split")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--shard", default="0/1", help="K/N: render the clips with dataset index K mod N only")
    args = parser.parse_args()
    shard, num_shards = map(int, args.shard.split("/"))
    if args.gmd_root is None or (args.prompts == "midi" and args.soundfont is None):
        parser.error("--gmd-root (or $GMD_ROOT) is required, and --soundfont (or $SOUNDFONT) for --prompts midi")

    rows = [json.loads(line) for line in (args.cache_root / "manifests" / f"{args.split}.jsonl").open()]
    if args.max_items > 0:
        rows = rows[: args.max_items]
    source_root = Path(json.loads((args.cache_root / "config.json").read_text())["source_cache_root"])
    source_rows = [json.loads(line) for line in (source_root / "manifest.jsonl").open()]
    with (args.gmd_root / "info.csv").open() as stream:
        gmd = {r["id"]: r for r in csv.DictReader(stream)}

    checkpoint = args.checkpoint if args.checkpoint == "released" else os.path.relpath(Path(args.checkpoint).resolve(), HERE)
    previous = args.out_dir / "summary.json"
    if previous.is_file() and not args.overwrite:
        old = json.loads(previous.read_text())
        if (old.get("prompts"), old.get("checkpoint")) != (PROMPTS[args.prompts], checkpoint):
            raise SystemExit(f"{args.out_dir} holds clips of another checkpoint or prompts; pass --overwrite or another --out-dir")
    if args.prompts == "midi":
        import midi_prompt
        midi_notes = {}
    app = load_tria(args.device, args.checkpoint)
    wav_dir = args.out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for dataset_index, row in enumerate(rows):
        wav = Path("wavs") / clip_file_name(dataset_index, row["source_id"], row["beat_index"])
        seed = args.seed + dataset_index
        source_row = source_rows[row["source_manifest_index"]]
        assert (source_row["source_id"], source_row["beat_index"]) == (row["source_id"], row["beat_index"])
        audio_path = args.gmd_root / gmd[row["source_id"]]["audio_filename"]
        entry = {"dataset_index": dataset_index, "source_id": row["source_id"], "beat_index": row["beat_index"],
                 "split": args.split, "duration_sec": row["duration_sec"], "sample_rate": SAMPLE_RATE, "seed": seed,
                 "wav": str(wav)}
        if args.prompts == "midi":
            if row["source_id"] not in midi_notes:
                midi_notes[row["source_id"]] = midi_prompt.recording_notes(args.gmd_root / gmd[row["source_id"]]["midi_filename"])
            example = torch.load(args.cache_root / row["out_pt"], map_location="cpu", weights_only=False)
            notes, counts = midi_prompt.bar_notes(example, midi_notes[row["source_id"]], float(source_row["start_sec"]))
            start, length = timbre_window(row, source_row, audio_path)
            entry.update(counts, timbre_start_sec=round(start, 4), timbre_sec=round(length, 4))
        if dataset_index % num_shards == shard and (args.overwrite or not (args.out_dir / wav).is_file()):
            if args.prompts == "bar":
                rhythm = timbre = bar_audio(row, source_row, audio_path)
            else:
                num_samples = round(float(row["duration_sec"]) * SAMPLE_RATE)
                rhythm = midi_prompt.render(notes, float(row["duration_sec"]), num_samples, args.soundfont, SAMPLE_RATE)
                timbre = read_mono(audio_path, start, length)
            rhythm_signal = to_signal(rhythm)
            out, _ = app._inference(to_signal(timbre), rhythm_signal, dict(SAMPLING), dict(INFERENCE), dict(AUDIO),
                                    SCHEDULE, [seed])
            audio = out.cpu().audio_data[0, 0].numpy()
            assert out.sample_rate == SAMPLE_RATE and len(audio) == rhythm_signal.signal_length
            sf.write(args.out_dir / wav, audio, SAMPLE_RATE, subtype="FLOAT")
            print(f"[{dataset_index + 1}/{len(rows)}] {wav.name}", flush=True)
        manifest.append(entry)

    if num_shards > 1:
        return
    (args.out_dir / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in manifest))
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=UPSTREAM, capture_output=True, text=True).stdout.strip()
    summary = dict(
        model=MODEL, checkpoint=checkpoint, tria_revision=revision, cache=args.cache_root.name, split=args.split,
        num_clips=len(manifest), seed=args.seed, prompts=PROMPTS[args.prompts], prompts_mode=args.prompts,
        sampling=SAMPLING, inference=INFERENCE, audio=AUDIO, schedule=SCHEDULE,
    )
    if args.prompts == "midi":
        summary.update(soundfont=args.soundfont.name, carry_ms=midi_prompt.CARRY_MS,
                       render_latency_ms=midi_prompt.RENDER_LATENCY_MS, gm_pitch=midi_prompt.GM_PITCH)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
