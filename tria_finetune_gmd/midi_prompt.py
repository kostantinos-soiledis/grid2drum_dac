"""TRIA's MIDI rhythm prompt for a GMD bar: the notes our model is conditioned on, rendered with a General MIDI soundfont.

The notes are the bar's onsets in our model's grid (the 250 Hz cache: time, instrument, articulation id and onset
velocity), which are the bar's GMD MIDI notes, plus the MIDI notes that start up to CARRY_MS before the bar. The bar
start comes from a beat tracker on the audio and often falls just after the downbeat (in 22% of test bars a note starts
within 20 ms before it). The cache then gives that note's onset to the previous bar and carries it into this one only as
instrument state from the first frame; our model hears it there and so does the target audio, so the prompt plays it at
the bar start. Each grid instrument plays one GM drum: rims and edges are the same drum, as in the grid.

The render is read RENDER_LATENCY_MS late: the soundfont's onsets lag the real bars' by that much (FluidSynth starts its
drum samples 1-3 ms after the note-on, and their attacks are softer than GMD's kit), measured as the peak of the onset
envelopes' cross-correlation summed over 300 validation bars (5.3 ms; median of the per-bar lags 5.1 ms). Without it,
the prompt would play every note ~5 ms after the bar's audio does, and TRIA's output would inherit the delay.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code" / "experiment"))
from data.family_state_cache_utils import _FAMILY_STATE_PITCH_TO_EVENT as PITCH_TO_EVENT  # noqa: E402

CARRY_MS = 30.0
RENDER_LATENCY_MS = 5.0
PRE_SEC = 0.05  # rendered before the bar, so notes at its start keep their attack once the latency is cut
NOTE_SEC = 0.1  # rendered note length (drum samples ring out regardless), as the MoisesDB prompts
# grid instrument -> GM pitch per articulation id (PITCH_TO_EVENT's ids): snare head/rim/cross-stick, tom head/rim,
# hi-hat open bow/open edge/closed bow/closed edge/pedal, crash/crash edge, ride bow/edge/bell
GM_PITCH = {"kick": [36], "snare": [38, 38, 37], "tom_high": [48, 48], "tom_mid": [45, 45], "tom_floor": [43, 43],
            "hihat": [46, 46, 42, 42, 44], "crash": [49, 49], "ride": [51, 51, 53]}


def recording_notes(midi_path: Path) -> np.ndarray:
    """A GMD recording's drum notes as rows (start s, pitch, velocity), sorted by start; unmapped pitches dropped."""
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = sorted((n.start, n.pitch, n.velocity) for i in pm.instruments if i.is_drum for n in i.notes
                   if n.pitch in PITCH_TO_EVENT)
    return np.array(notes, dtype=np.float64).reshape(-1, 3)


def bar_notes(example: dict, notes: np.ndarray, bar_start: float) -> tuple[list[tuple[float, str, int, int]], dict]:
    """The bar's notes as (time s, instrument, articulation id, velocity): the cache's onsets plus the MIDI notes that
    start up to CARRY_MS before the bar (at time 0). Also counts: grid onsets, carried-in notes, and the MIDI notes
    inside the bar (a frame of the 250 Hz grid can merge two notes of one instrument into one onset)."""
    names = list(example["class_names"])
    grid, ids, times = example["grid_ft"], example["grid_ids_ft"], example["grid_times_sec_t"]
    out = [(float(times[t]), names[f], int(ids[f, t]), max(1, round(float(grid[3 * f + 1, t]) * 127)))
           for f, t in example["family_onsets_ft"].nonzero().tolist()]
    onsets = len(out)
    duration = float(example["duration_sec"])
    rel = notes[:, 0] - bar_start
    carried = notes[(rel < 0) & (rel >= -CARRY_MS / 1000)]
    out += [(0.0, *PITCH_TO_EVENT[int(p)], int(v)) for _, p, v in carried]
    inside = int(((rel >= 0) & (rel < duration)).sum())
    return sorted(out), {"grid_onsets": onsets, "carried_in": len(carried), "midi_notes_in_bar": inside}


def render(notes: list[tuple[float, str, int, int]], duration: float, num_samples: int, soundfont: Path,
           sample_rate: int) -> np.ndarray:
    """The notes rendered with FluidSynth, mono, from RENDER_LATENCY_MS after the bar start, cut or padded to
    num_samples."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=60.0 * 4 / duration)
    drums = pretty_midi.Instrument(program=0, is_drum=True)
    drums.notes = [pretty_midi.Note(velocity=v, pitch=GM_PITCH[name][i], start=PRE_SEC + t, end=PRE_SEC + t + NOTE_SEC)
                   for t, name, i, v in notes]
    pm.instruments.append(drums)
    with tempfile.TemporaryDirectory() as tmp:
        mid, wav = Path(tmp) / "bar.mid", Path(tmp) / "bar.wav"
        pm.write(str(mid))
        subprocess.run(["fluidsynth", "-ni", "-q", "-F", str(wav), "-r", str(sample_rate), str(soundfont), str(mid)],
                       check=True)
        audio, sr = sf.read(wav, dtype="float32", always_2d=True)
    assert sr == sample_rate
    start = round((PRE_SEC + RENDER_LATENCY_MS / 1000) * sample_rate)
    audio = audio.mean(axis=1)[start: start + num_samples]
    return np.pad(audio, (0, num_samples - len(audio)))
