"""
scripts/prepare_orators.py
--------------------------
Build benchmark texts 9-11 whose GOOD baseline is a real, highly effective
public speaker (US presidential addresses, public domain as works of the US
federal government, 17 U.S.C. § 105), downloaded from Wikimedia Commons.

For each speech:
  1. Download the source recording (once) into .cache/sources/ (gitignored).
  2. Locate the transcript passage in the full recording (Whisper recognition
     matched to the transcript) and cut it at silences -> text_NN_good.wav.
  3. Force-align it to the transcript and write dataset/textgrids/text_NN.TextGrid.
  4. Generate the same 17 flaw variants as the rest of the benchmark
     (scripts/03_generate_bad_clips.generate_set, deterministic seeds).
  5. Record provenance in dataset/labels/orators.json.

Texts 9-11 are HELD OUT: never used for threshold calibration.

Usage:
    python scripts/prepare_orators.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.aligner import SR, _load_audio_16k, align  # noqa: E402

AUDIO = ROOT / "dataset" / "audio"
LABELS = ROOT / "dataset" / "labels"
TEXTGRIDS = ROOT / "dataset" / "textgrids"
TRANSCRIPTS = ROOT / "dataset" / "transcripts"
SOURCES = ROOT / ".cache" / "sources"
USER_AGENT = "CadenceResearch/1.0 (speech delivery analytics benchmark)"

ORATORS = {
    9: {
        "title": "Address at Rice University on the Nation's Space Effort (\"We choose to go to the Moon\")",
        "speaker": "John F. Kennedy", "date": "1962-09-12",
        "commons_file": "File:Jfk rice university we choose to go to the moon.ogg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/5/50/Jfk_rice_university_we_choose_to_go_to_the_moon.ogg",
        "source": "John F. Kennedy Presidential Library & Museum, White House Audio Collection (JFKWHA-127-002)",
        "license": "Public domain (work of the US federal government)",
    },
    10: {
        "title": "First Inaugural Address",
        "speaker": "Ronald Reagan", "date": "1981-01-20",
        "commons_file": "File:Ronald Reagan First Inaugural.ogg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/1/10/Ronald_Reagan_First_Inaugural.ogg",
        "source": "Wikimedia Commons; author recorded as 'assumed to be the White House', public-domain status per americanrhetoric.com",
        "license": "Public domain (tagged on Commons; provenance assumed, not documented by an archive)",
    },
    11: {
        "title": "First Inaugural Address",
        "speaker": "Dwight D. Eisenhower", "date": "1953-01-20",
        "commons_file": "File:Dwight D. Eisenhower's first Inaugural Address, 1953-01-20.ogg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/3/3e/Dwight_D._Eisenhower%27s_first_Inaugural_Address%2C_1953-01-20.ogg",
        "source": "Wikimedia Commons via Internet Archive / Miller Center; author recorded as 'Whitehouse (assumed)'",
        "license": "Public domain (tagged on Commons; provenance assumed, not documented by an archive)",
    },
}


def _module(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fetch(text_id: int, info: dict) -> Path:
    dest = SOURCES / f"text_{text_id:02d}_source.ogg"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(info["url"], headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
            f.write(r.read())
    return dest


def main() -> None:
    gen = _module("03_generate_bad_clips")
    real = _module("prepare_real_speakers")      # locate_span, pseudo_phones, write_textgrid
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    meta_path = LABELS / "orators.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    for tid, info in ORATORS.items():
        prefix = f"text_{tid:02d}"
        transcript = (TRANSCRIPTS / f"{prefix}.txt").read_text(encoding="utf-8")
        print(f"\n=== {prefix}: {info['speaker']} — {info['title']} ({info['date']})")

        src = fetch(tid, info)
        audio = _load_audio_16k(str(src))
        start, end, coverage = real.locate_span(audio, transcript)
        good = audio[int(start * SR): int(end * SR)].astype(np.float32)
        good_path = AUDIO / f"{prefix}_good.wav"
        sf.write(str(good_path), good, SR, subtype="PCM_16")
        print(f"  passage {start:.2f}-{end:.2f} s of {len(audio) / SR:.0f} s "
              f"(recognizer matched {coverage:.0%} of transcript tokens) -> {good_path.name}")

        res = align(str(good_path), transcript, cfg=cfg)
        phones = real.pseudo_phones(res.words)
        real.write_textgrid(TEXTGRIDS / f"{prefix}.TextGrid", res.words, phones, len(good) / SR)
        print(f"  forced alignment: {len(res.words)} words, {sum(w.uncertain for w in res.words)} uncertain")

        gen.generate_set(
            clip_prefix=prefix, y=good, sr=SR,
            word_ts=[{"word": w.word, "start": w.start, "end": w.end} for w in res.words],
            phone_ts=phones,
            meta={"text_id": tid, "speaker_id": info["speaker"], "speaker_source": "orator", "split": "heldout"},
            audio_dir=AUDIO, labels_dir=LABELS, config=cfg,
        )
        meta[prefix] = {**info, "text_id": tid, "trim_s": [round(start, 3), round(end, 3)],
                        "duration_s": round(len(good) / SR, 3), "transcript_coverage": round(coverage, 3),
                        "transcript_path": f"dataset/transcripts/{prefix}.txt",
                        "wav_file": f"dataset/audio/{prefix}_good.wav", "split": "heldout"}

    meta_path.write_text(json.dumps(meta, indent=2))
    print("\nUpdated", meta_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
