"""
scripts/build_manifest.py
-------------------------
Write dataset/MANIFEST.sha256: a SHA-256 for every file the published results
depend on (audio, labels, transcripts, alignment cache, TextGrids, config).
`--check` verifies the working tree against it, so drift between machines,
deployments or re-generations is detected instead of silently changing results.

Usage:
    python scripts/build_manifest.py            # (re)write the manifest
    python scripts/build_manifest.py --check    # exit 1 on any mismatch
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
MANIFEST = ROOT / "dataset" / "MANIFEST.sha256"
PATTERNS = [
    "dataset/audio/*.wav",
    "dataset/audio/real_speakers/real*.wav",
    "dataset/labels/**/*.json",
    "dataset/transcripts/*.txt",
    "dataset/alignments/text_*.json",
    "dataset/alignments/real*.json",
    "dataset/textgrids/*.TextGrid",
    "config.yaml",
]


def tracked_files() -> list[Path]:
    files = set()
    for pat in PATTERNS:
        files.update(p for p in ROOT.glob(pat) if p.is_file())
    return sorted(files)


def digest(path: Path) -> str:
    """SHA-256; text files are hashed with LF line endings so a CRLF checkout on
    Windows and an LF checkout on Linux agree."""
    if path.suffix.lower() == ".wav":
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def build() -> dict[str, str]:
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in tracked_files()}


def read_manifest() -> dict[str, str]:
    out = {}
    for line in MANIFEST.read_text().splitlines():
        if line and not line.startswith("#"):
            h, name = line.split("  ", 1)
            out[name] = h
    return out


def check() -> list[str]:
    expected, actual = read_manifest(), build()
    problems = [f"missing: {n}" for n in expected if n not in actual]
    problems += [f"unexpected: {n}" for n in actual if n not in expected]
    problems += [f"changed: {n}" for n in expected if n in actual and actual[n] != expected[n]]
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        problems = check()
        for p in problems[:50]:
            print(p)
        print("manifest OK" if not problems else f"{len(problems)} mismatches")
        return 1 if problems else 0
    entries = build()
    MANIFEST.write_text("# SHA-256 of every file the published results depend on (scripts/build_manifest.py)\n"
                        + "".join(f"{h}  {n}\n" for n, h in entries.items()))
    print(f"wrote {MANIFEST.relative_to(ROOT)} ({len(entries)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
