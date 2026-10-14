"""
scripts/prepare_real_speakers.py
--------------------------------
Build contrast sets from the genuine human LibriVox recordings
(dataset/labels/real_speakers.json).

For each recording:
  1. Locate the transcript span inside the full recording (the files contain the
     LibriVox preamble and, for Patrick Henry, the whole speech): Whisper
     recognizes the recording, the transcript tokens are matched to the
     recognized tokens, and the clip is cut at the quietest frame just before
     the first and just after the last matched word.
  2. Save the trimmed GOOD clip as dataset/audio/real_speakers/realNN_good.wav.
  3. Force-align it to the transcript (pipeline.aligner) and write a TextGrid.
  4. Generate the same mild / moderate / egregious flaw variants as the TTS set
     with scripts/03_generate_bad_clips.generate_set (deterministic seeds),
     labels in dataset/labels/real_speakers/.

These clips are NEVER used for threshold calibration of the floors; they are an
evaluation set for (a) same-speaker detection on real human delivery and
(b) cross-speaker comparison against the TTS baseline (see scripts/05).

Usage:
    python scripts/prepare_real_speakers.py
"""

from __future__ import annotations

import difflib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.aligner import (  # noqa: E402
    SR, _get_model, _load_audio_16k, _recognize, frame_db, align, transcript_words,
)

REAL_AUDIO = ROOT / "dataset" / "audio" / "real_speakers"
REAL_LABELS = ROOT / "dataset" / "labels" / "real_speakers"
META_PATH = ROOT / "dataset" / "labels" / "real_speakers.json"
TG_DIR = ROOT / "dataset" / "textgrids"
TRANSCRIPTS = ROOT / "dataset" / "transcripts"


def _load_generator():
    spec = importlib.util.spec_from_file_location("gen", ROOT / "scripts" / "03_generate_bad_clips.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def locate_span(audio: np.ndarray, transcript: str) -> tuple[float, float, float]:
    """(start_s, end_s, fraction of transcript tokens matched) of the transcript inside *audio*."""
    tokens = [t.lower() for t in transcript_words(transcript)]
    rec = _recognize(_get_model("base.en"), audio, prompt=" ".join(tokens[:40]))
    rec_tokens = [r["word"].lower() for r in rec]
    sm = difflib.SequenceMatcher(None, tokens, rec_tokens, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size >= 3]
    if not blocks:
        raise RuntimeError("transcript not found in recording")
    matched = sum(b.size for b in blocks)
    first = rec[blocks[0].b]
    last = rec[blocks[-1].b + blocks[-1].size - 1]
    # Extend to the transcript edges if the first/last tokens were not matched exactly
    db = frame_db(audio)

    def quietest(t0: float, t1: float) -> float:
        i0, i1 = max(int(t0 / 0.01), 0), min(int(t1 / 0.01), len(db) - 1)
        return (i0 + int(np.argmin(db[i0:i1 + 1]))) * 0.01 if i1 > i0 else t0

    start = quietest(first["start"] - 0.8, first["start"] - 0.05)
    end = quietest(last["end"] + 0.05, last["end"] + 0.9)
    return start, end, matched / len(tokens)


def pseudo_phones(words) -> list[dict]:
    """Letter-group subdivisions of each aligned word (used only to place clarity edits)."""
    out = []
    for w in words:
        text = w.word.lower()
        n = max(len(text) // 2, 2)
        step = (w.end - w.start) / n
        for k in range(n):
            ch = text[min(k * 2, len(text) - 1)]
            out.append({"phone": f"{ch}{k + 1}", "start": w.start + k * step, "end": w.start + (k + 1) * step,
                        "word": text})
    return out


def write_textgrid(path: Path, words, phones: list[dict], dur: float) -> None:
    def tier(name, items):
        lines = ["        class = \"IntervalTier\"", f"        name = \"{name}\"", "        xmin = 0",
                 f"        xmax = {dur:.4f}", f"        intervals: size = {len(items)}"]
        for i, (a, b, t) in enumerate(items, 1):
            lines += [f"        intervals [{i}]:", f"            xmin = {a:.4f}",
                      f"            xmax = {b:.4f}", f"            text = \"{t}\""]
        return lines

    lines = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "", "xmin = 0", f"xmax = {dur:.4f}",
             "tiers? <exists>", "size = 2", "item []:", "    item [1]:"]
    lines += tier("words", [(w.start, w.end, w.word) for w in words])
    lines += ["    item [2]:"] + tier("phones", [(p["start"], p["end"], p["phone"]) for p in phones])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    gen = _load_generator()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    meta_all = json.loads(META_PATH.read_text())

    for key, meta in meta_all.items():
        text_id = int(meta["text_id"])
        prefix = f"real{text_id:02d}"
        src = ROOT / Path(meta.get("source_wav", meta["wav_file"]).replace("\\", "/"))
        transcript = (TRANSCRIPTS / f"text_{text_id:02d}.txt").read_text(encoding="utf-8")
        print(f"\n=== {prefix}: {meta['title']} — {meta['reader']} ===")

        audio = _load_audio_16k(str(src))
        start, end, coverage = locate_span(audio, transcript)
        good = audio[int(start * SR): int(end * SR)]
        good_path = REAL_AUDIO / f"{prefix}_good.wav"
        sf.write(str(good_path), good, SR, subtype="PCM_16")
        print(f"  trimmed {start:.2f}-{end:.2f} s of {len(audio) / SR:.1f} s  "
              f"(transcript tokens matched by recognizer: {coverage:.0%}) -> {good_path.name}")

        res = align(str(good_path), transcript, cfg=cfg)
        phones = pseudo_phones(res.words)
        write_textgrid(TG_DIR / f"{prefix}.TextGrid", res.words, phones, len(good) / SR)
        print(f"  forced alignment: {len(res.words)} words, {sum(w.uncertain for w in res.words)} uncertain")

        word_ts = [{"word": w.word, "start": w.start, "end": w.end} for w in res.words]
        gen.generate_set(
            clip_prefix=prefix, y=good.astype(np.float32), sr=SR,
            word_ts=word_ts, phone_ts=phones,
            meta={"text_id": text_id, "speaker_id": meta["reader"], "speaker_source": "librivox_human",
                  "split": "real_speaker"},
            audio_dir=REAL_AUDIO, labels_dir=REAL_LABELS, config=cfg,
        )

        meta.update({
            "source_wav": src.relative_to(ROOT).as_posix(),
            "wav_file": good_path.relative_to(ROOT).as_posix(),
            "trim_s": [round(start, 3), round(end, 3)],
            "duration_s": round(len(good) / SR, 3),
            "transcript_coverage": round(coverage, 3),
        })

    META_PATH.write_text(json.dumps(meta_all, indent=2))
    print("\nUpdated", META_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()
