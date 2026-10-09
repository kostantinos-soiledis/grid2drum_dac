"""
Sets up the fine-tune of TRIA's small_musdb_moises_fsl_2b on GMD (train.sh runs it): manifests, config, start checkpoint.

TRIA's own training script (upstream/scripts/train.py) fine-tunes when it resumes from a checkpoint without its
extras.pt: it loads <save_path>/latest/model.pt and starts a fresh optimizer, scheduler and step counter. So only
model.pt is copied; the shipped extras.pt would resume at step 80,000 with a decayed learning rate.

Data: GMD's own splits, which are our models' (train 846 recordings, 8.45 h; validation 120; the test recordings are
never used). TRIA reads only the drums audio: both of its prompts come from the same excerpt during training. Two
changes from the upstream data settings, both for GMD:
  - TRIA draws a recording uniformly within a manifest, and 64% of GMD's training recordings (mostly fills) are shorter
    than its 6 s excerpt while holding 4.6% of the audio. The training recordings are split into beats and fills,
    weighted by their hours, so excerpts follow the audio rather than the file count.
  - Its salience search wants excerpts above -24 LUFS; GMD's recordings sit around -28 LUFS, so the search would always
    fail and take a random excerpt. The cutoff is -40 LUFS (StemDataset's default), which still skips near silence.
Everything else is the checkpoint's own config (conf/small_musdb_moises_fsl_2b.yml: model, rhythm features, masking,
augmentations, AMP) except the run: batch 8 (12 runs out of memory on a 10 GB GPU in the rhythm prompt's band-pass
augmentation, 24 at once), AdamW lr 1e-5 (released config: 1e-4), at most 80,000 iterations (640,000 excerpts of 6 s),
validation every 2,000 iterations (TRIA's cadence) and audio samples every 8,000. TRIA keeps "latest" and, by
validation token cross-entropy, "best"; train_early_stop.py ends the run after 10 checks without a new best.

    GMD_ROOT=<groove/> .venv/bin/python prepare.py

Writes manifests/{train_beats,train_fills,val}.csv, <name>.yml and runs/<name>/latest/model.pt (machine-specific
absolute paths; not in git).
"""

import argparse
import csv
import os
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / "upstream"
BASE_MODEL = "small_musdb_moises_fsl_2b"
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--name", default="gmd_long", help="run name: runs/<name>, <name>.yml")
parser.add_argument("--num-iters", type=int, default=80000)
parser.add_argument("--val-freq", type=int, default=2000)
parser.add_argument("--sample-freq", type=int, default=8000)
parser.add_argument("--gmd-root", type=Path, default=os.environ.get("GMD_ROOT"), help="GMD's groove/ folder; default $GMD_ROOT")
cli = parser.parse_args()
if cli.gmd_root is None:
    parser.error("--gmd-root (or $GMD_ROOT) is required")
GMD_ROOT = cli.gmd_root
SAVE_PATH = HERE / "runs" / cli.name
RUN = dict(num_iters=cli.num_iters, batch_size=8, val_batch_size=10, num_workers=8, sample_freq=cli.sample_freq,
           val_freq=cli.val_freq, lr=1e-5, loudness_cutoff=-40.0)

rows = [r for r in csv.DictReader((GMD_ROOT / "info.csv").open()) if r["audio_filename"]]
manifests, hours = {}, {}
for name, keep in (("train_beats", lambda r: r["split"] == "train" and r["beat_type"] == "beat"),
                   ("train_fills", lambda r: r["split"] == "train" and r["beat_type"] == "fill"),
                   ("val", lambda r: r["split"] == "validation")):
    chosen = [r for r in rows if keep(r)]
    path = HERE / "manifests" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["drums"])
        writer.writerows([[str(GMD_ROOT / r["audio_filename"])] for r in chosen])
    manifests[name], hours[name] = path, sum(float(r["duration"]) for r in chosen) / 3600
    print(f"{name}: {len(chosen)} recordings, {hours[name]:.2f} h")
total = hours["train_beats"] + hours["train_fills"]
weights = [round(hours["train_beats"] / total, 4), round(hours["train_fills"] / total, 4)]


def setting(text: str, key: str, value) -> str:
    pattern = rf"^{re.escape(key)}: .*$"
    if not re.search(pattern, text, flags=re.M):
        raise KeyError(key)
    return re.sub(pattern, f"{key}: {value}", text, flags=re.M)


config = (UPSTREAM / "conf" / f"{BASE_MODEL}.yml").read_text()
for key, value in (("save_path", SAVE_PATH), ("resume", "true"), ("num_iters", RUN["num_iters"]),
                   ("sample_freq", RUN["sample_freq"]), ("val_freq", RUN["val_freq"]), ("batch_size", RUN["batch_size"]),
                   ("val_batch_size", RUN["val_batch_size"]), ("num_workers", RUN["num_workers"]), ("save_iters", "[]"),
                   ("AdamW.lr", f"{RUN['lr']:.8f}".rstrip("0")), ("StemDataset.loudness_cutoff", RUN["loudness_cutoff"]),
                   ("train/StemDataset.source_weights", weights)):
    config = setting(config, key, value)
config, n = re.subn(r"train/StemDataset\.sources:\n(?:  - .*\n|\s*\n)+",
                    f"train/StemDataset.sources:\n  - {manifests['train_beats']}\n  - {manifests['train_fills']}\n\n", config)
assert n == 1
config, n = re.subn(r"val/StemDataset\.sources:\n(?:  - .*\n)+", f"val/StemDataset.sources:\n  - {manifests['val']}\n", config)
assert n == 1
header = (f"# Fine-tune of {BASE_MODEL} on GMD, written by prepare.py from upstream/conf/{BASE_MODEL}.yml.\n"
          f"# Changed: save_path, resume, num_iters, sample/val_freq, batch sizes, num_workers, save_iters, AdamW.lr,\n"
          f"# StemDataset.loudness_cutoff, train/val sources and weights (beats/fills by hours).\n\n")
(HERE / f"{cli.name}.yml").write_text(header + config)

start = SAVE_PATH / "latest"
start.mkdir(parents=True, exist_ok=True)
if not (start / "model.pt").exists() and not (start / "extras.pt").exists():
    shutil.copyfile(UPSTREAM / "pretrained" / "tria" / BASE_MODEL / "latest" / "model.pt", start / "model.pt")
print(f"train source weights (beats, fills) {weights}; config {HERE / f'{cli.name}.yml'}; start {start / 'model.pt'}")
