"""
scripts/regenerate_flaws.py
---------------------------
Rebuild every flawed clip from its GOOD baseline with the current (deterministic)
generators, without touching the baselines themselves:

  TTS texts 1-8     dataset/audio/text_NN_good.wav
  Orators 9-11      dataset/audio/text_NN_good.wav
  LibriVox readers  dataset/audio/real_speakers/realNN_good.wav

Word timings come from the transcript-forced alignment of each baseline, and each
baseline's TextGrid (word tier) is rewritten from that alignment. Every clip is
seeded with its clip id (Python and Praat random generators), so running this
script twice produces bit-identical audio and labels.

Usage:
    python scripts/regenerate_flaws.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.aligner import SR, _load_audio_16k, align  # noqa: E402
from pipeline.analysis import load_bundled  # noqa: E402

AUDIO = ROOT / "dataset" / "audio"
REAL = AUDIO / "real_speakers"
LABELS = ROOT / "dataset" / "labels"
TEXTGRIDS = ROOT / "dataset" / "textgrids"
TRANSCRIPTS = ROOT / "dataset" / "transcripts"


def _module(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_word_textgrid(path: Path, words, dur: float) -> None:
    items = [(w.start, w.end, w.word) for w in words]
    lines = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "", "xmin = 0", f"xmax = {dur:.4f}",
             "tiers? <exists>", "size = 1", "item []:", "    item [1]:",
             '        class = "IntervalTier"', '        name = "words"', "        xmin = 0",
             f"        xmax = {dur:.4f}", f"        intervals: size = {len(items)}"]
    for i, (a, b, t) in enumerate(items, 1):
        lines += [f"        intervals [{i}]:", f"            xmin = {a:.4f}",
                  f"            xmax = {b:.4f}", f'            text = "{t}"']
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def baselines(bundled: dict) -> list[dict]:
    gen = _module("03_generate_bad_clips")
    out = []
    for tid in sorted(bundled):
        meta = {**gen.TEXT_META[tid], "text_id": tid}
        out.append({"prefix": f"text_{tid:02d}", "good": AUDIO / f"text_{tid:02d}_good.wav",
                    "transcript": bundled[tid], "meta": meta, "audio_dir": AUDIO, "labels_dir": LABELS})
    real_meta = json.loads((LABELS / "real_speakers.json").read_text())
    for m in real_meta.values():
        tid = int(m["text_id"])
        tx_path = ROOT / m.get("transcript_path", f"dataset/transcripts/text_{tid:02d}.txt")
        out.append({"prefix": f"real{tid:02d}", "good": REAL / f"real{tid:02d}_good.wav",
                    "transcript": tx_path.read_text(encoding="utf-8"),
                    "meta": {"text_id": tid, "speaker_id": m["reader"], "speaker_source": "librivox_human",
                             "split": "real_speaker"},
                    "audio_dir": REAL, "labels_dir": LABELS / "real_speakers"})
    return out


def main() -> None:
    gen = _module("03_generate_bad_clips")
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    bundled, _ = load_bundled()
    for b in baselines(bundled):
        y = _load_audio_16k(str(b["good"]))
        res = align(str(b["good"]), b["transcript"], cfg=cfg)
        write_word_textgrid(TEXTGRIDS / f"{b['prefix']}.TextGrid", res.words, len(y) / SR)
        print(f"=== {b['prefix']}: {len(res.words)} words")
        gen.generate_set(
            clip_prefix=b["prefix"], y=y.astype(np.float32), sr=SR,
            word_ts=[{"word": w.word, "start": w.start, "end": w.end} for w in res.words], phone_ts=[],
            meta=b["meta"], audio_dir=b["audio_dir"], labels_dir=b["labels_dir"], config=cfg,
        )


if __name__ == "__main__":
    main()
