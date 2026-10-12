"""
scripts/verify_mfa.py
---------------------
Automated MFA alignment sanity checks (§2.1 reproducibility requirement).

Per clip checks:
  (a) Total aligned span within 2% of true audio duration
  (b) No negative-duration or overlapping word intervals
  (c) Aligned word count == transcript word count

Run automatically from scripts/02_run_mfa.sh and in GitHub Actions CI.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
TEXTGRID_DIR = ROOT / "dataset" / "textgrids"
AUDIO_DIR = ROOT / "dataset" / "audio"
TRANSCRIPT_DIR = ROOT / "dataset" / "transcripts"

TOLERANCE = 0.02   # 2% duration tolerance


def parse_textgrid(tg_path: Path) -> list[dict]:
    """Minimal TextGrid parser — returns word intervals."""
    text = tg_path.read_text(encoding="utf-8", errors="replace")
    # Find all intervals in the "words" tier
    intervals = []
    in_words_tier = False
    for line in text.splitlines():
        line = line.strip()
        if "words" in line.lower():
            in_words_tier = True
        elif in_words_tier and ("phones" in line.lower() or "item [" in line.lower()):
            in_words_tier = False
        if in_words_tier and line.startswith("xmin"):
            xmin = float(line.split("=")[1].strip())
        elif in_words_tier and line.startswith("xmax"):
            xmax = float(line.split("=")[1].strip())
        elif in_words_tier and line.startswith("text"):
            word = re.search(r'"(.*?)"', line)
            word = word.group(1).strip() if word else ""
            if word and word not in ("", "sp", "sil"):
                intervals.append({"word": word, "start": xmin, "end": xmax})
    return intervals


def get_audio_duration(wav_path: Path) -> float:
    import soundfile as sf
    info = sf.info(str(wav_path))
    return info.duration


def get_transcript_words(txt_path: Path) -> list[str]:
    text = txt_path.read_text().strip().lower()
    return re.sub(r"[^\w\s]", "", text).split()


def check_clip(text_id_str: str) -> list[str]:
    errors = []
    prefix = text_id_str if text_id_str.startswith("text_") else f"text_{text_id_str}"
    tg_path = TEXTGRID_DIR / f"{prefix}.TextGrid"
    wav_path = AUDIO_DIR / f"{prefix}_good.wav"
    txt_path = TRANSCRIPT_DIR / f"{prefix}.txt"

    if not tg_path.exists():
        errors.append(f"{text_id_str}: TextGrid not found at {tg_path}")
        return errors
    if not wav_path.exists():
        errors.append(f"{text_id_str}: Audio not found at {wav_path}")
        return errors
    if not txt_path.exists():
        errors.append(f"{text_id_str}: Transcript not found at {txt_path}")
        return errors

    intervals = parse_textgrid(tg_path)
    audio_dur = get_audio_duration(wav_path)
    transcript_words = get_transcript_words(txt_path)

    # (a) Total aligned span within 2%
    if intervals:
        aligned_span = intervals[-1]["end"] - intervals[0]["start"]
        deviation = abs(aligned_span - audio_dur) / max(audio_dur, 1.0)
        if deviation > TOLERANCE:
            errors.append(
                f"{text_id_str}: Aligned span {aligned_span:.2f}s vs audio {audio_dur:.2f}s "
                f"({deviation*100:.1f}% deviation, tolerance {TOLERANCE*100:.0f}%)"
            )

    # (b) No negative or overlapping intervals
    for i, iv in enumerate(intervals):
        if iv["end"] <= iv["start"]:
            errors.append(f"{text_id_str}: Word '{iv['word']}' has non-positive duration "
                          f"({iv['start']:.3f}s -> {iv['end']:.3f}s)")
        if i > 0 and intervals[i]["start"] < intervals[i-1]["end"] - 0.001:
            errors.append(f"{text_id_str}: Overlapping intervals at word '{iv['word']}'")

    # (c) Word count match
    aligned_words = [iv["word"] for iv in intervals]
    if len(aligned_words) != len(transcript_words):
        errors.append(
            f"{text_id_str}: Aligned {len(aligned_words)} words vs "
            f"transcript {len(transcript_words)} words"
        )

    return errors


def main():
    print("Running MFA alignment sanity checks...")
    all_errors = []
    for i in range(1, 9):
        text_id_str = f"text_{i:02d}"
        errors = check_clip(text_id_str)
        if errors:
            for e in errors:
                print(f"  FAIL: {e}")
            all_errors.extend(errors)
        else:
            print(f"  OK:   {text_id_str}")

    if all_errors:
        print(f"\n{len(all_errors)} check(s) failed. Inspect the TextGrid viewer before proceeding.")
        sys.exit(1)
    else:
        print("\nAll MFA sanity checks passed.")


if __name__ == "__main__":
    main()
