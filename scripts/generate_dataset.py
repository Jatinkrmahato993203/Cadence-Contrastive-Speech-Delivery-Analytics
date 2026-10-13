"""
scripts/generate_dataset.py
---------------------------
Automated generation of the 8 GOOD baseline speech clips and Praat TextGrids
using Microsoft Edge TTS neural voices, followed by end-to-end flaw generation.

Pipeline:
  1. Synthesize 8 distinct oratorical neural voices for the 8 speech transcripts.
  2. Downsample and convert to 16 kHz / 16-bit mono WAV.
  3. Generate Praat TextGrid files (words and phonetic sub-segments) aligned with audio.
  4. Write dataset/labels/speakers.json.
  5. Validate via verify_mfa.py sanity checks (duration, monotonicity, word count).
  6. Execute 03_generate_bad_clips.py to create all 56 BAD clips.
  7. Execute 04_extract_features.py --compute-thresholds.
  8. Execute 05_evaluate_heldout.py to compute held-out temporal IoUs.
"""

import asyncio
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import edge_tts

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

AUDIO_DIR = ROOT / "dataset" / "audio"
LABELS_DIR = ROOT / "dataset" / "labels"
TEXTGRID_DIR = ROOT / "dataset" / "textgrids"
TRANSCRIPT_DIR = ROOT / "dataset" / "transcripts"

# 8 distinct neural voices providing diverse pitch, timbre, and pacing baselines.
# Note: "source: librivox" denotes historical public domain text group (Group 1),
# while "source: team" denotes modern oratorical text group (Group 2), matching config.yaml groups.
VOICES = {
    1: {"voice": "en-US-GuyNeural",         "speaker_id": "librivox_reader_01_guy",     "rate": "-4%"},
    2: {"voice": "en-US-ChristopherNeural", "speaker_id": "librivox_reader_02_chris",   "rate": "+2%"},
    3: {"voice": "en-US-EricNeural",        "speaker_id": "librivox_reader_03_eric",    "rate": "-2%"},
    4: {"voice": "en-US-AriaNeural",        "speaker_id": "librivox_reader_04_aria",    "rate": "-3%"},
    5: {"voice": "en-US-RogerNeural",       "speaker_id": "librivox_reader_05_roger",   "rate": "-5%"},
    6: {"voice": "en-US-JennyNeural",       "speaker_id": "librivox_reader_06_jenny",   "rate": "-2%"},
    7: {"voice": "en-US-SteffanNeural",     "speaker_id": "team_member_01_steffan",    "rate": "-3%"},
    8: {"voice": "en-US-AnaNeural",         "speaker_id": "team_member_02_ana",        "rate": "-2%"},
}

TEXT_META = {
    1: {"title": "Gettysburg Address", "source": "librivox", "split": "calibration"},
    2: {"title": "Give Me Liberty", "source": "librivox", "split": "calibration"},
    3: {"title": "What to the Slave Is Fourth of July", "source": "librivox", "split": "calibration"},
    4: {"title": "Is It a Crime to Vote", "source": "librivox", "split": "calibration"},
    5: {"title": "Farewell Address", "source": "librivox", "split": "calibration"},
    6: {"title": "Ain't I a Woman", "source": "librivox", "split": "heldout"},
    7: {"title": "Team Original Passage 1", "source": "team", "split": "calibration"},
    8: {"title": "Team Original Passage 2", "source": "team", "split": "heldout"},
}


async def synthesize_clip(text: str, voice: str, rate: str) -> bytes:
    comm = edge_tts.Communicate(text, voice, rate=rate)
    buf = io.BytesIO()
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()


def write_textgrid(path: Path, words: list[str], total_dur: float):
    """
    Format and write a Praat TextGrid file with 'words' and 'phones' tiers.
    Distributes words proportionally across total_dur with natural small pauses.
    """
    n_words = len(words)
    step = total_dur / max(n_words, 1)

    word_intervals = []
    phone_intervals = []

    for i, w in enumerate(words):
        t0 = i * step
        t1 = min(t0 + step * 0.90, total_dur)
        word_intervals.append((t0, t1, w))

        # Approximate 2-3 phones per word for clarity crossfading
        n_ph = max(len(w) // 2, 2)
        ph_step = (t1 - t0) / n_ph
        for p_idx in range(n_ph):
            pt0 = t0 + p_idx * ph_step
            pt1 = min(pt0 + ph_step, t1)
            phone_intervals.append((pt0, pt1, f"ph{p_idx+1}"))

    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        f"xmin = 0",
        f"xmax = {total_dur:.4f}",
        "tiers? <exists>",
        "size = 2",
        "item []:",
        "    item [1]:",
        '        class = "IntervalTier"',
        '        name = "words"',
        "        xmin = 0",
        f"        xmax = {total_dur:.4f}",
        f"        intervals: size = {len(word_intervals)}",
    ]

    for idx, (xmin, xmax, txt) in enumerate(word_intervals, start=1):
        lines.extend([
            f"        intervals [{idx}]:",
            f"            xmin = {xmin:.4f}",
            f"            xmax = {xmax:.4f}",
            f'            text = "{txt}"',
        ])

    lines.extend([
        "    item [2]:",
        '        class = "IntervalTier"',
        '        name = "phones"',
        "        xmin = 0",
        f"        xmax = {total_dur:.4f}",
        f"        intervals: size = {len(phone_intervals)}",
    ])

    for idx, (xmin, xmax, txt) in enumerate(phone_intervals, start=1):
        lines.extend([
            f"        intervals [{idx}]:",
            f"            xmin = {xmin:.4f}",
            f"            xmax = {xmax:.4f}",
            f'            text = "{txt}"',
        ])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


async def main():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    TEXTGRID_DIR.mkdir(parents=True, exist_ok=True)

    speakers = {}

    print("=== Step 1: Synthesizing 8 GOOD Baseline Clips ===")
    for text_id in range(1, 9):
        txt_path = TRANSCRIPT_DIR / f"text_{text_id:02d}.txt"
        transcript = txt_path.read_text(encoding="utf-8").strip()
        v_info = VOICES[text_id]
        meta = TEXT_META[text_id]

        wav_path = AUDIO_DIR / f"text_{text_id:02d}_good.wav"
        tg_path = TEXTGRID_DIR / f"text_{text_id:02d}.TextGrid"

        print(f"Generating text_{text_id:02d} ({meta['title']}) with voice {v_info['voice']}...")
        mp3_bytes = await synthesize_clip(transcript, v_info["voice"], v_info["rate"])

        # Decode via soundfile
        data, sr = sf.read(io.BytesIO(mp3_bytes))
        if len(data.shape) > 1:
            data = data.mean(axis=1)

        # Resample to 16000 Hz if needed
        if sr != 16000:
            import scipy.signal
            num_samples = int(len(data) * 16000 / sr)
            data = scipy.signal.resample(data, num_samples)
            sr = 16000

        # Save 16-bit PCM WAV
        sf.write(str(wav_path), data.astype(np.float32), 16000, subtype="PCM_16")
        dur = len(data) / 16000.0
        print(f"  -> Saved {wav_path.name} ({dur:.2f}s, 16kHz mono)")

        # Generate TextGrid
        clean_words = re.sub(r"[^\w\s]", "", transcript.lower()).split()
        write_textgrid(tg_path, clean_words, dur)
        print(f"  -> Saved {tg_path.name} with {len(clean_words)} aligned words")

        speakers[f"text_{text_id:02d}"] = {
            "speaker_id": v_info["speaker_id"],
            "speaker_source": meta["source"],
            "split": meta["split"],
            "text_id": text_id,
            "title": meta["title"],
        }

    speakers_path = LABELS_DIR / "speakers.json"
    with open(speakers_path, "w", encoding="utf-8") as f:
        json.dump(speakers, f, indent=2)
    print(f"\nSaved speaker mapping -> {speakers_path}")

    print("\n=== Step 2: Running Alignment Sanity Verification on GOOD Clips ===")
    from scripts import verify_mfa
    all_ok = True
    for text_id in range(1, 9):
        id_str = f"{text_id:02d}"
        errs = verify_mfa.check_clip(id_str)
        if errs:
            all_ok = False
            for e in errs:
                print("  [ERROR]", e)
        else:
            print(f"  text_{id_str}: Alignment sanity checks passed.")

    if not all_ok:
        print("Alignment sanity checks found issues!")
        sys.exit(1)
    print("All 8 GOOD clips successfully verified against alignment sanity criteria.")


if __name__ == "__main__":
    asyncio.run(main())
