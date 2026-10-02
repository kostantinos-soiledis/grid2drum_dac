#!/usr/bin/env python3
"""Rewrite absolute workstation paths in run/result text artifacts as repo-relative paths."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


TEXT_SUFFIXES = {".json", ".jsonl", ".csv", ".md", ".txt"}
SKIPPED_DIRS = {"wavs", "logs", "target_audio_cache", "fad_assets", "eval_input"}
# Interpreter paths (e.g. a conda env under a home directory) become plain `python`.
INTERPRETER_RE = re.compile(r"/[^\s\"',]*/bin/python[0-9.]*")


def replacements_for(repo_root: Path) -> list[tuple[str, str]]:
    spellings = {os.path.abspath(repo_root), str(repo_root.resolve())}
    pairs: list[tuple[str, str]] = []
    # Longest prefixes first: the repo itself, then siblings such as ../pca_diffusion.
    for repo in sorted(spellings, key=len, reverse=True):
        pairs.append((repo + "/", ""))
    for repo in sorted(spellings, key=len, reverse=True):
        pairs.append((os.path.dirname(repo) + "/", "../"))
    return pairs


def iter_text_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIPPED_DIRS]
        for name in filenames:
            path = Path(dirpath) / name
            if path.suffix in TEXT_SUFFIXES:
                yield path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("roots", nargs="+", type=Path, help="directories rewritten in place")
    args = parser.parse_args()

    pairs = replacements_for(args.repo_root)
    changed = 0
    for root in args.roots:
        for path in iter_text_files(root):
            text = path.read_text(encoding="utf-8")
            new_text = text
            for old, new in pairs:
                new_text = new_text.replace(old, new)
            new_text = INTERPRETER_RE.sub("python", new_text)
            if new_text != text:
                path.write_text(new_text, encoding="utf-8")
                changed += 1
    print(f"relativized absolute paths in {changed} file(s)")


if __name__ == "__main__":
    main()
