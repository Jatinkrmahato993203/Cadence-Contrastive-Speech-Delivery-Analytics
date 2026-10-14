"""
LEGACY — not part of the as-built pipeline. The bundled GOOD baselines were
synthesized by scripts/generate_dataset.py (Edge TTS); the human LibriVox
recordings are prepared by scripts/prepare_real_speakers.py. Kept for anyone
who wants to rebuild the benchmark from their own recordings.
"""
"""
scripts/01_prepare_good_clips.py
--------------------------------
Phase 0/1: Download LibriVox recordings, trim to the ~90s excerpt,
and downsample to 16 kHz / 16-bit mono WAV.

Usage:
    python scripts/01_prepare_good_clips.py

For each text, the script:
  1. Prompts for a local path or URL to the LibriVox MP3/WAV.
  2. Converts to 16 kHz / 16-bit mono WAV using pydub + ffmpeg.
  3. Lets you trim via START_S and END_S env vars or interactive input.
  4. Saves to dataset/audio/text_NN_good.wav.
  5. Writes a speaker mapping entry to dataset/labels/speakers.json.

Run BEFORE scripts/02_run_mfa.sh so MFA has all GOOD clips to align.
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
AUDIO_DIR = ROOT / "dataset" / "audio"
LABELS_DIR = ROOT / "dataset" / "labels"
TRANSCRIPT_DIR = ROOT / "dataset" / "transcripts"

TEXTS = {
    1: {"title": "Gettysburg Address", "source": "librivox", "split": "calibration"},
    2: {"title": "Give Me Liberty", "source": "librivox", "split": "calibration"},
    3: {"title": "What to the Slave Is the Fourth of July", "source": "librivox", "split": "calibration"},
    4: {"title": "Is It a Crime to Vote", "source": "librivox", "split": "calibration"},
    5: {"title": "Farewell Address", "source": "librivox", "split": "calibration"},
    6: {"title": "Ain't I a Woman", "source": "librivox", "split": "heldout"},
    7: {"title": "Team Original Passage 1", "source": "team", "split": "calibration"},
    8: {"title": "Team Original Passage 2", "source": "team", "split": "heldout"},
}


def convert_and_trim(input_path: str, output_path: str, start_s: float, end_s: float):
    """Convert audio to 16kHz / 16-bit mono WAV and trim to [start_s, end_s]."""
    try:
        from pydub import AudioSegment
    except ImportError:
        print("pydub not installed. Run: pip install pydub")
        sys.exit(1)

    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    start_ms = int(start_s * 1000)
    end_ms = int(end_s * 1000) if end_s > 0 else len(audio)
    trimmed = audio[start_ms:end_ms]
    trimmed.export(output_path, format="wav")
    print(f"  Saved {output_path}  ({len(trimmed)/1000:.1f}s)")


def main():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    speakers = {}

    for text_id, meta in TEXTS.items():
        print(f"\n{'='*60}")
        print(f"Text {text_id:02d}: {meta['title']}")
        print(f"  Source: {meta['source']} | Split: {meta['split']}")

        out_path = AUDIO_DIR / f"text_{text_id:02d}_good.wav"
        if out_path.exists():
            print(f"  Already exists: {out_path.name}  (skipping)")
        else:
            src = input("  Path or URL to source audio (MP3/WAV/FLAC): ").strip()
            start_s = float(input("  Trim start (seconds, 0 = beginning): ") or "0")
            end_s = float(input("  Trim end   (seconds, 0 = use full clip): ") or "0")
            convert_and_trim(src, str(out_path), start_s, end_s)

        speaker_id = input(f"  Speaker ID / reader name for text {text_id:02d}: ").strip()
        speakers[f"text_{text_id:02d}"] = {
            "speaker_id": speaker_id,
            "speaker_source": meta["source"],
            "split": meta["split"],
            "text_id": text_id,
            "title": meta["title"],
        }

    speakers_path = LABELS_DIR / "speakers.json"
    with open(speakers_path, "w") as f:
        json.dump(speakers, f, indent=2)
    print(f"\nSaved speaker mapping -> {speakers_path}")
    print("\nPhase 0 GOOD clip preparation complete.")
    print("Next step: run scripts/02_run_mfa.sh")


if __name__ == "__main__":
    main()
