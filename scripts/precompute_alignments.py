"""
scripts/precompute_alignments.py
--------------------------------
Force-align every dataset clip to its transcript and store the result in
dataset/alignments/{clip}.json (keyed by audio + transcript + aligner version,
so stale entries are ignored automatically).

Usage:
    python scripts/precompute_alignments.py [--workers 3] [--glob "text_06_*"]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

AUDIO_DIR = ROOT / "dataset" / "audio"
TRANSCRIPT_DIR = ROOT / "dataset" / "transcripts"


def transcript_for(wav: Path) -> str:
    """Reader-specific transcript (transcripts/realNN.txt) when one exists, else the bundled text."""
    m = re.match(r"(text_|real)(\d+)", wav.stem)
    if not m:
        raise ValueError(f"Cannot infer text id from {wav.name}")
    tid = int(m.group(2))
    own = TRANSCRIPT_DIR / f"real{tid:02d}.txt"
    if m.group(1) == "real" and own.exists():
        return own.read_text(encoding="utf-8")
    return (TRANSCRIPT_DIR / f"text_{tid:02d}.txt").read_text(encoding="utf-8")


def _work(wav_str: str, threads: int) -> str:
    import torch
    torch.set_num_threads(threads)
    import logging, warnings
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    from pipeline.aligner import align

    wav = Path(wav_str)
    res = align(str(wav), transcript_for(wav))
    n_unc = sum(w.uncertain for w in res.words)
    return f"{wav.name}: {len(res.words)} words, {n_unc} uncertain, {res.model_used}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--glob", default="*.wav")
    args = ap.parse_args()

    wavs = sorted(AUDIO_DIR.glob(args.glob)) + sorted((AUDIO_DIR / "real_speakers").glob(args.glob))
    wavs = [w for w in wavs if re.match(r"(?:text_|real)\d+", w.stem)]
    threads = max(1, (os.cpu_count() or 4) // args.workers)
    print(f"Aligning {len(wavs)} clips with {args.workers} workers x {threads} threads", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_work, str(w), threads): w for w in wavs}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                print(f"[{i}/{len(wavs)}] {f.result()}", flush=True)
            except Exception as exc:
                print(f"[{i}/{len(wavs)}] FAILED {futs[f].name}: {exc}", flush=True)


if __name__ == "__main__":
    main()
