"""
scripts/03_generate_bad_clips.py
---------------------------------
Phase 1: Generate all 56 BAD clips (7 per text x 8 texts) from the GOOD clips.

All 7 flaw mechanisms are parameterized. The MODERATE tier is used for flaw
types 1-5 (per §1.3 of the technical approach). Rows 6 and 7 use EGREGIOUS
and NEAR-PERFECT respectively.

DEPENDENCY: TextGrids must exist for all 8 GOOD clips (via generate_dataset.py).
Pause (row 3) and Clarity (row 5) consume word/phone interval boundaries directly.

Usage:
    python scripts/03_generate_bad_clips.py [--text TEXT_ID] [--flaw FLAW_TYPE]

All outputs committed at 16 kHz / 16-bit mono WAV. Flaw region boundaries
are computed analytically from the known transform and written to
dataset/labels/{clip_id}.json.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
AUDIO_DIR = ROOT / "dataset" / "audio"
LABELS_DIR = ROOT / "dataset" / "labels"
TEXTGRID_DIR = ROOT / "dataset" / "textgrids"
TRANSCRIPT_DIR = ROOT / "dataset" / "transcripts"
CONFIG_PATH = ROOT / "config.yaml"

SR = 16000  # target sample rate

TEXT_META = {
    1: {"speaker_id": "librivox_reader_01", "speaker_source": "librivox", "split": "calibration"},
    2: {"speaker_id": "librivox_reader_02", "speaker_source": "librivox", "split": "calibration"},
    3: {"speaker_id": "librivox_reader_03", "speaker_source": "librivox", "split": "calibration"},
    4: {"speaker_id": "librivox_reader_04", "speaker_source": "librivox", "split": "calibration"},
    5: {"speaker_id": "librivox_reader_05", "speaker_source": "librivox", "split": "calibration"},
    6: {"speaker_id": "librivox_reader_06", "speaker_source": "librivox", "split": "heldout"},
    7: {"speaker_id": "team_member_01",     "speaker_source": "team",     "split": "calibration"},
    8: {"speaker_id": "team_member_02",     "speaker_source": "team",     "split": "heldout"},
}

FLAW_SUFFIX = {
    "pacing_mod":         {"flaw_type": "pacing",   "severity_tier": "moderate"},
    "pitch_mod":          {"flaw_type": "pitch",     "severity_tier": "moderate"},
    "pause_mod":          {"flaw_type": "pause",     "severity_tier": "moderate"},
    "volume_mod":         {"flaw_type": "volume",    "severity_tier": "moderate"},
    "clarity_mod":        {"flaw_type": "clarity",   "severity_tier": "moderate"},
    "combined_egregious": {"flaw_type": "combined",  "severity_tier": "egregious"},
    "nearperfect":        {"flaw_type": "pacing",    "severity_tier": "near_perfect"},
}


# -- Audio I/O helpers ---------------------------------------------------------

def load_audio(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return y, sr


def save_audio(y: np.ndarray, path: Path, sr: int = SR):
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y.astype(np.float32), sr, subtype="PCM_16")


# -- TextGrid reader -----------------------------------------------------------

def load_word_timestamps(text_id: int) -> list[dict]:
    """Load MFA word timestamps from TextGrid. Returns list of {word, start, end}."""
    tg_path = TEXTGRID_DIR / f"text_{text_id:02d}.TextGrid"
    if not tg_path.exists():
        print(f"  WARNING: TextGrid not found for text_{text_id:02d}. "
              f"Run scripts/02_run_mfa.sh first.")
        return []
    text = tg_path.read_text(encoding="utf-8", errors="replace")
    intervals = []
    xmin = xmax = 0.0
    in_words = False
    for line in text.splitlines():
        line = line.strip()
        if "words" in line.lower():
            in_words = True
        if in_words and line.startswith("xmin"):
            xmin = float(line.split("=")[1].strip())
        elif in_words and line.startswith("xmax"):
            xmax = float(line.split("=")[1].strip())
        elif in_words and line.startswith("text"):
            m = re.search(r'"(.*?)"', line)
            word = m.group(1).strip() if m else ""
            if word and word not in ("", "sp", "sil", "<eps>"):
                intervals.append({"word": word, "start": xmin, "end": xmax})
    return intervals


def load_phone_timestamps(text_id: int) -> list[dict]:
    """Load MFA phone timestamps from TextGrid (phones tier)."""
    tg_path = TEXTGRID_DIR / f"text_{text_id:02d}.TextGrid"
    if not tg_path.exists():
        return []
    text = tg_path.read_text(encoding="utf-8", errors="replace")
    intervals = []
    xmin = xmax = 0.0
    in_phones = False
    for line in text.splitlines():
        line = line.strip()
        if "phones" in line.lower():
            in_phones = True
        elif in_phones and "item [" in line.lower():
            in_phones = False
        if in_phones and line.startswith("xmin"):
            xmin = float(line.split("=")[1].strip())
        elif in_phones and line.startswith("xmax"):
            xmax = float(line.split("=")[1].strip())
        elif in_phones and line.startswith("text"):
            m = re.search(r'"(.*?)"', line)
            ph = m.group(1).strip() if m else ""
            if ph and ph not in ("", "sp", "sil", "<eps>"):
                intervals.append({"phone": ph, "start": xmin, "end": xmax})
    return intervals


# -- Flaw 1: Pacing (time-stretch, pitch-preserved) ----------------------------

def apply_pacing(y: np.ndarray, sr: int, rate: float) -> tuple[np.ndarray, list[dict]]:
    """
    Time-stretch y by *rate* (e.g. 0.75 = slower, 1.30 = faster).
    Returns (stretched_audio, flaw_regions).
    """
    import librosa
    stretched = librosa.effects.time_stretch(y, rate=rate)
    duration = len(stretched) / sr
    regions = [{"start_s": 0.0, "end_s": round(duration, 3), "mechanism": f"time_stretch_{rate}x"}]
    return stretched, regions


# -- Flaw 2: Pitch / Monotone (F0 variance compression) ------------------------

def apply_pitch_monotone(
    y: np.ndarray, sr: int, variance_fraction: float
) -> tuple[np.ndarray, list[dict]]:
    """
    Compress F0 variance toward the clip's mean F0 by *variance_fraction*.
    Uses parselmouth Manipulation object for pitch-tier scaling.
    variance_fraction=0.40 -> moderate (retain 40% of original variance).
    """
    try:
        import parselmouth
        from parselmouth.praat import call
    except ImportError:
        print("  parselmouth not installed — skipping pitch flaw")
        return y, []

    snd = parselmouth.Sound(values=y.astype(np.float64), sampling_frequency=float(sr))
    manipulation = call(snd, "To Manipulation", 0.01, 75, 600)
    pitch_tier = call(manipulation, "Extract pitch tier")

    n_points = call(pitch_tier, "Get number of points")
    if n_points == 0:
        return y, []

    # Compute mean F0
    times = [call(pitch_tier, "Get time from index", i + 1) for i in range(n_points)]
    freqs = [call(pitch_tier, "Get value at index", i + 1) for i in range(n_points)]
    mean_f0 = float(np.mean([f for f in freqs if f > 0])) if any(f > 0 for f in freqs) else 150.0

    # Compress variance using Praat's Formula:
    # new_f0 = mean + (self - mean) * variance_fraction
    call(pitch_tier, "Formula", f"{mean_f0} + (self - {mean_f0}) * {variance_fraction}")
    call([pitch_tier, manipulation], "Replace pitch tier")
    result_snd = call(manipulation, "Get resynthesis (overlap-add)")
    result_y = result_snd.values.flatten().astype(np.float32)
    duration = len(result_y) / sr
    regions = [{"start_s": 0.0, "end_s": round(duration, 3),
                "mechanism": f"f0_variance_compress_{variance_fraction}"}]
    return result_y, regions


# -- Flaw 3: Pauses (splice at MFA word boundaries) ----------------------------

def apply_pause(
    y: np.ndarray, sr: int,
    word_timestamps: list[dict],
    count: int, duration_range: tuple[float, float],
) -> tuple[np.ndarray, list[dict]]:
    """
    Insert *count* unplanned pauses at word boundaries from MFA timestamps.
    Returns (modified_audio, flaw_regions).
    """
    if not word_timestamps:
        print("  No word timestamps available — skipping pause flaw (run MFA first)")
        return y, []

    from pydub import AudioSegment, generators

    audio = AudioSegment(
        y.tobytes(), frame_rate=sr, sample_width=2, channels=1
    )
    # Snap float32 -> int16
    y_int16 = (np.clip(y, -1, 1) * 32767).astype(np.int16)
    audio = AudioSegment(y_int16.tobytes(), frame_rate=sr, sample_width=2, channels=1)

    # Pick *count* random word boundaries (not at very start or end)
    eligible = word_timestamps[1:-1]
    if len(eligible) < count:
        count = len(eligible)
    splice_words = random.sample(eligible, count)
    splice_words.sort(key=lambda w: w["start"])

    # Build silence segments
    silences = []
    flaw_regions = []
    cumulative_offset_ms = 0

    for w in splice_words:
        dur_s = random.uniform(*duration_range)
        silence = AudioSegment.silent(duration=int(dur_s * 1000), frame_rate=sr)
        silences.append((w["start"], silence, dur_s))

    # Insert silences (work backwards to keep indices valid)
    result = audio
    flaw_regions = []
    for w_start, silence, dur_s in reversed(silences):
        ms = int(w_start * 1000)
        result = result[:ms] + silence + result[ms:]
        flaw_regions.append({
            "start_s": round(w_start, 3),
            "end_s": round(w_start + dur_s, 3),
            "mechanism": f"silence_splice_{dur_s:.2f}s_at_{w_start:.3f}s",
        })

    result_y = np.array(result.get_array_of_samples()).astype(np.float32) / 32767.0
    return result_y, sorted(flaw_regions, key=lambda r: r["start_s"])


# -- Flaw 4: Volume (gain envelope) --------------------------------------------

def apply_volume(
    y: np.ndarray, sr: int,
    gain_db: float, fraction: float,
) -> tuple[np.ndarray, list[dict]]:
    """
    Apply a gain reduction of *gain_db* dB over *fraction* of the clip.
    Region is chosen to start at ~10% of clip duration.
    """
    total = len(y)
    region_len = int(total * fraction)
    start_idx = int(total * 0.1)
    end_idx = start_idx + region_len
    end_idx = min(end_idx, total)

    gain_linear = 10 ** (gain_db / 20.0)
    result = y.copy()
    result[start_idx:end_idx] = result[start_idx:end_idx] * gain_linear

    start_s = start_idx / sr
    end_s = end_idx / sr
    regions = [{"start_s": round(start_s, 3), "end_s": round(end_s, 3),
                "mechanism": f"gain_{gain_db}dB_over_{fraction*100:.0f}pct"}]
    return result, regions


# -- Flaw 5: Clarity (phone crossfade + STFT smoothing + transient softening) --

def apply_clarity(
    y: np.ndarray, sr: int,
    phone_timestamps: list[dict],
    crossfade_ms_range: tuple[int, int],
    smoothing_bins: int,
    transient_db: float,
) -> tuple[np.ndarray, list[dict]]:
    """
    Reduces articulatory clarity via three layered mechanisms:
    1. Phone-boundary crossfade: merges adjacent phones (acoustic definition of slurring)
    2. STFT magnitude smoothing along frequency axis (blurs formant peaks)
    3. Transient softening on stop-consonant bursts (reduces plosive crispness)

    All operations are driven by MFA phone/word timestamps.
    NOT a flat low-pass filter — models real mumbling, not a bad microphone.
    """
    import librosa
    from scipy.signal import fftconvolve

    if not phone_timestamps:
        print("  No phone timestamps — skipping clarity flaw (run MFA first)")
        return y, []

    result = y.copy()
    crossfade_ms = random.randint(*crossfade_ms_range)
    crossfade_samples = int(crossfade_ms * sr / 1000)

    # 1. Phone-boundary crossfade
    for i in range(1, len(phone_timestamps) - 1):
        boundary_idx = int(phone_timestamps[i]["start"] * sr)
        half = crossfade_samples // 2
        lo = max(0, boundary_idx - half)
        hi = min(len(result), boundary_idx + half)
        if hi > lo:
            fade = np.linspace(0, 1, hi - lo)
            result[lo:hi] = result[lo:hi] * (1 - fade) + result[lo:hi] * fade

    # 2. STFT magnitude smoothing (frequency axis only, phase preserved)
    hop_length = 512
    n_fft = 2048
    S = librosa.stft(result, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(S)
    phase = np.angle(S)

    # Boxcar smoothing along frequency axis
    kernel = np.ones((smoothing_bins, 1)) / smoothing_bins
    smoothed_mag = fftconvolve(mag, kernel, mode="same")
    S_smoothed = smoothed_mag * np.exp(1j * phase)
    result = librosa.istft(S_smoothed, hop_length=hop_length, length=len(y))

    # 3. Transient softening on plosive consonants
    plosives = {"p", "b", "t", "d", "k", "g"}
    gain_linear = 10 ** (transient_db / 20.0)
    for ph in phone_timestamps:
        if any(ph["phone"].lower().startswith(p) for p in plosives):
            idx_start = int(ph["start"] * sr)
            # Soften the first 20ms (transient burst window)
            idx_end = min(idx_start + int(0.02 * sr), len(result))
            result[idx_start:idx_end] *= gain_linear

    duration = len(result) / sr
    regions = [{"start_s": 0.0, "end_s": round(duration, 3),
                "mechanism": f"clarity_xfade{crossfade_ms}ms_smooth{smoothing_bins}bin_transient{transient_db}dB"}]
    return result.astype(np.float32), regions


# -- Flaw 6: Combined egregious -------------------------------------------------

def apply_combined_egregious(
    y: np.ndarray, sr: int,
    word_timestamps: list[dict],
    config: dict,
) -> tuple[np.ndarray, list[dict]]:
    """Chain pacing + pitch + pauses at egregious parameters."""
    params = config.get("flaw_params", {})
    all_regions = []

    # Pacing: egregious (0.65x)
    rate = params.get("pacing", {}).get("egregious", [0.65, 1.5])[0]
    y, r = apply_pacing(y, sr, rate)
    all_regions.extend(r)

    # Pitch: 10% variance
    var = params.get("pitch", {}).get("egregious", 0.10)
    y, r = apply_pitch_monotone(y, sr, var)
    all_regions.extend(r)

    # Pause: 2-3 silences of 1.0-2.0s
    pause_cfg = params.get("pause", {}).get("egregious", {})
    count = pause_cfg.get("count", 3)
    dur_range = pause_cfg.get("duration_range", [1.0, 2.0])
    # Recalculate word timestamps after time-stretch
    if word_timestamps:
        stretched_ts = [
            {"word": w["word"],
             "start": w["start"] / rate,
             "end": w["end"] / rate}
            for w in word_timestamps
        ]
        y, r = apply_pause(y, sr, stretched_ts, count, dur_range)
        all_regions.extend(r)

    return y.astype(np.float32), all_regions


# -- Flaw 7: Near-perfect ------------------------------------------------------

def apply_near_perfect(y: np.ndarray, sr: int, config: dict) -> tuple[np.ndarray, list[dict]]:
    """Apply the mildest pacing perturbation (1.15x)."""
    params = config.get("flaw_params", {})
    rate = params.get("pacing", {}).get("near_perfect", [0.85, 1.15])[1]
    return apply_pacing(y, sr, rate)


# -- Label writer --------------------------------------------------------------

def write_label(
    text_id: int, suffix: str, flaw_regions: list[dict], meta: dict
):
    clip_id = f"text_{text_id:02d}_{suffix}"
    flaw_meta = FLAW_SUFFIX[suffix]
    label = {
        "clip_id": clip_id,
        "source_text_id": text_id,
        "speaker_id": meta["speaker_id"],
        "speaker_source": meta["speaker_source"],
        "flaw_type": flaw_meta["flaw_type"],
        "severity_tier": flaw_meta["severity_tier"],
        "split": meta["split"],
        "flaw_regions": flaw_regions,
        "transcript_path": f"dataset/transcripts/text_{text_id:02d}.txt",
        "audio_path": f"dataset/audio/{clip_id}.wav",
    }
    out = LABELS_DIR / f"{clip_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(label, f, indent=2)


# -- Main ----------------------------------------------------------------------

def process_text(text_id: int, only_flaw: str | None, config: dict):
    import yaml
    meta = TEXT_META[text_id]
    good_path = AUDIO_DIR / f"text_{text_id:02d}_good.wav"
    if not good_path.exists():
        print(f"  SKIP text_{text_id:02d}: GOOD clip not found at {good_path}")
        return

    y, sr = load_audio(good_path)
    word_ts = load_word_timestamps(text_id)
    phone_ts = load_phone_timestamps(text_id)
    params = config.get("flaw_params", {})

    def maybe(suffix: str, fn):
        if only_flaw and suffix != only_flaw:
            return
        out_path = AUDIO_DIR / f"text_{text_id:02d}_{suffix}.wav"
        if out_path.exists():
            print(f"  EXISTS: {out_path.name}")
            return
        print(f"  Generating {suffix}...")
        result_y, regions = fn()
        save_audio(result_y, out_path, sr)
        write_label(text_id, suffix, regions, meta)

    maybe("pacing_mod", lambda: apply_pacing(
        y, sr, params.get("pacing", {}).get("moderate", [0.75, 1.30])[0]))

    maybe("pitch_mod", lambda: apply_pitch_monotone(
        y, sr, params.get("pitch", {}).get("moderate", 0.40)))

    pause_mod_cfg = params.get("pause", {}).get("moderate", {})
    maybe("pause_mod", lambda: apply_pause(
        y, sr, word_ts,
        pause_mod_cfg.get("count", 1),
        pause_mod_cfg.get("duration_range", [0.40, 0.80])))

    vol_cfg = params.get("volume", {}).get("moderate", {})
    maybe("volume_mod", lambda: apply_volume(
        y, sr, vol_cfg.get("db", -10), vol_cfg.get("fraction", 0.20)))

    clar_cfg = params.get("clarity", {}).get("moderate", {})
    maybe("clarity_mod", lambda: apply_clarity(
        y, sr, phone_ts,
        clar_cfg.get("crossfade_ms", [40, 60]),
        clar_cfg.get("smoothing_bins", 7),
        clar_cfg.get("transient_db", -4.0)))

    maybe("combined_egregious", lambda: apply_combined_egregious(y, sr, word_ts, config))
    maybe("nearperfect", lambda: apply_near_perfect(y, sr, config))


def main():
    import yaml

    parser = argparse.ArgumentParser(description="Generate BAD clips from GOOD clips")
    parser.add_argument("--text", type=int, help="Only process this text_id (1-8)")
    parser.add_argument("--flaw", type=str, help="Only generate this flaw suffix")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    text_ids = [args.text] if args.text else list(range(1, 9))
    for tid in text_ids:
        print(f"\n{'='*60}")
        print(f"Processing text_{tid:02d}")
        process_text(tid, args.flaw, config)

    print("\nFlaw generation complete.")
    print("Dataset labels written to dataset/labels/")
    print("Next: python scripts/04_extract_features.py --compute-thresholds")


if __name__ == "__main__":
    main()
