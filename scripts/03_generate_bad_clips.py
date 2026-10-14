"""
scripts/03_generate_bad_clips.py
---------------------------------
Phase 1: Generate BAD clips across a 3-tier severity gradient (mild/near-perfect, moderate, egregious)
with time-bounded flaw regions and exact analytic ground-truth timestamps.

All flaw mechanisms are strictly time-bounded:
  - Pacing: localized time-stretch over a target rhetorical window [t0, t1]
  - Pitch: localized F0 variance compression over [t0, t1] using Praat conditional Formula
  - Pauses: silence splices at genuine acoustic word boundaries from Whisper/TextGrid
  - Volume: localized gain reduction over [t0, t1] with smooth fade transitions
  - Clarity: localized phone crossfade, STFT blur, and plosive softening over [t0, t1]
  - Combined: compound multi-flaw sequence across distinct time windows

All outputs committed at 16 kHz / 16-bit mono WAV. Flaw region boundaries
are computed analytically from the known transforms and written to
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
    # Real presidential addresses (see scripts/prepare_orators.py)
    9: {"speaker_id": "John F. Kennedy",      "speaker_source": "orator",   "split": "heldout"},
    10: {"speaker_id": "Ronald Reagan",       "speaker_source": "orator",   "split": "heldout"},
    11: {"speaker_id": "Dwight D. Eisenhower", "speaker_source": "orator",  "split": "heldout"},
}

FLAW_SUFFIX = {
    # Standard 7 files per text (Moderate anchors + compounds)
    "pacing_mod":         {"flaw_type": "pacing",      "severity_tier": "moderate"},
    "pitch_mod":          {"flaw_type": "pitch",       "severity_tier": "moderate"},
    "pause_mod":          {"flaw_type": "pause",       "severity_tier": "moderate"},
    "volume_mod":         {"flaw_type": "volume",      "severity_tier": "moderate"},
    "clarity_mod":        {"flaw_type": "clarity",     "severity_tier": "moderate"},
    "combined_egregious": {"flaw_type": "combined",    "severity_tier": "egregious"},
    "nearperfect":        {"flaw_type": "nearperfect", "severity_tier": "near_perfect"},

    # Full 3-tier severity gradient (mild, moderate, egregious)
    "pacing_mild":        {"flaw_type": "pacing",      "severity_tier": "near_perfect"},
    "pitch_mild":         {"flaw_type": "pitch",       "severity_tier": "near_perfect"},
    "pause_mild":         {"flaw_type": "pause",       "severity_tier": "near_perfect"},
    "volume_mild":        {"flaw_type": "volume",      "severity_tier": "near_perfect"},
    "clarity_mild":       {"flaw_type": "clarity",     "severity_tier": "near_perfect"},

    "pacing_egregious":   {"flaw_type": "pacing",      "severity_tier": "egregious"},
    "pitch_egregious":    {"flaw_type": "pitch",       "severity_tier": "egregious"},
    "pause_egregious":    {"flaw_type": "pause",       "severity_tier": "egregious"},
    "volume_egregious":   {"flaw_type": "volume",      "severity_tier": "egregious"},
    "clarity_egregious":  {"flaw_type": "clarity",     "severity_tier": "egregious"},
}


# -- Audio I/O helpers ---------------------------------------------------------

def load_audio(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y, sr


def save_audio(y: np.ndarray, path: Path, sr: int = SR):
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y.astype(np.float32), sr, subtype="PCM_16")


# -- TextGrid reader -----------------------------------------------------------

def load_word_timestamps(text_id: int) -> list[dict]:
    """Load acoustic word timestamps from aligned good audio or TextGrid."""
    align_json = ROOT / "dataset" / "alignments" / f"text_{text_id:02d}_good.json"
    if align_json.exists():
        try:
            cached = json.loads(align_json.read_text(encoding="utf-8"))
            return [{"word": w["word"], "start": float(w["start"]), "end": float(w["end"])} for w in cached.get("words", [])]
        except Exception:
            pass
    tg_path = TEXTGRID_DIR / f"text_{text_id:02d}.TextGrid"
    if not tg_path.exists():
        return []
    text = tg_path.read_text(encoding="utf-8", errors="replace")
    intervals = []
    xmin = xmax = 0.0
    in_words = False
    for line in text.splitlines():
        line = line.strip()
        if 'name = "words"' in line:
            in_words = True
        elif in_words and 'item [' in line and 'name = "words"' not in line:
            in_words = False
        if in_words:
            if line.startswith("xmin ="):
                xmin = float(line.split("=")[1].strip())
            elif line.startswith("xmax ="):
                xmax = float(line.split("=")[1].strip())
            elif line.startswith("text ="):
                m = re.search(r'"(.*?)"', line)
                word = m.group(1).strip() if m else ""
                if word and word not in ("", "sp", "sil", "<eps>"):
                    intervals.append({"word": word, "start": xmin, "end": xmax})
    return intervals


def load_phone_timestamps(text_id: int) -> list[dict]:
    """Load acoustic phone timestamps from TextGrid (phones tier)."""
    tg_path = TEXTGRID_DIR / f"text_{text_id:02d}.TextGrid"
    if not tg_path.exists():
        return []
    text = tg_path.read_text(encoding="utf-8", errors="replace")
    intervals = []
    xmin = xmax = 0.0
    in_phones = False
    for line in text.splitlines():
        line = line.strip()
        if 'name = "phones"' in line:
            in_phones = True
        elif in_phones and 'item [' in line and 'name = "phones"' not in line:
            in_phones = False
        if in_phones:
            if line.startswith("xmin ="):
                xmin = float(line.split("=")[1].strip())
            elif line.startswith("xmax ="):
                xmax = float(line.split("=")[1].strip())
            elif line.startswith("text ="):
                m = re.search(r'"(.*?)"', line)
                ph = m.group(1).strip() if m else ""
                if ph and ph not in ("", "sp", "sil", "<eps>"):
                    intervals.append({"phone": ph, "start": xmin, "end": xmax})
    return intervals


# -- Flaw 1: Pacing (time-bounded time-stretch) --------------------------------

def apply_pacing(
    y: np.ndarray, sr: int, rate: float,
    start_frac: float = 0.20, end_frac: float = 0.65,
) -> tuple[np.ndarray, list[dict]]:
    """
    Time-stretch speech by *rate* (PSOLA, pitch preserved) over a bounded window [start_frac, end_frac].
    Smoothly crossfades back into normal speech speed at boundaries.
    """
    import parselmouth
    from parselmouth.praat import call
    total_dur = len(y) / sr
    t0 = total_dur * start_frac
    t1 = total_dur * end_frac
    i0 = int(t0 * sr)
    i1 = int(t1 * sr)

    pre = y[:i0]
    seg = y[i0:i1]
    post = y[i1:]

    # PSOLA (Praat "Lengthen (overlap-add)") keeps the harmonic structure of voiced
    # speech intact. librosa's phase vocoder smeared harmonics ("phasiness"), so a
    # pacing flaw also degraded clarity — an artifact real slow speech does not have.
    snd = parselmouth.Sound(seg.astype(np.float64), sampling_frequency=float(sr))
    stretched_seg = call(snd, "Lengthen (overlap-add)", 75, 600, 1.0 / rate).values.flatten().astype(np.float32)
    xfade_len = min(int(0.015 * sr), len(stretched_seg) // 4)
    if xfade_len > 0 and len(pre) >= xfade_len:
        fade_in = np.linspace(0.0, 1.0, xfade_len)
        stretched_seg[:xfade_len] = stretched_seg[:xfade_len] * fade_in + pre[-xfade_len:] * (1.0 - fade_in)

    result = np.concatenate([pre, stretched_seg, post])
    new_t1 = round(t0 + len(stretched_seg) / sr, 3)
    regions = [{
        "start_s": round(t0, 3),
        "end_s": new_t1,
        "mechanism": f"time_stretch_{rate}x_bounded_{t0:.2f}to{new_t1:.2f}s",
    }]
    return result, regions


# -- Flaw 2: Pitch / Monotone (time-bounded F0 variance compression) -----------

def apply_pitch_monotone(
    y: np.ndarray, sr: int, variance_fraction: float,
    start_frac: float = 0.20, end_frac: float = 0.65,
) -> tuple[np.ndarray, list[dict]]:
    """
    Compress F0 variance toward mean F0 by *variance_fraction* over window [start_frac, end_frac].
    Uses Praat's conditional Formula for exact temporal boundaries.
    """
    try:
        import parselmouth
        from parselmouth.praat import call
    except ImportError:
        return y, []

    total_dur = len(y) / sr
    t0 = total_dur * start_frac
    t1 = total_dur * end_frac

    snd = parselmouth.Sound(values=y.astype(np.float64), sampling_frequency=float(sr))
    manipulation = call(snd, "To Manipulation", 0.01, 75, 600)
    pitch_tier = call(manipulation, "Extract pitch tier")

    n_points = call(pitch_tier, "Get number of points")
    if n_points == 0:
        return y, []

    freqs = [call(pitch_tier, "Get value at index", i + 1) for i in range(n_points)]
    mean_f0 = float(np.mean([f for f in freqs if f > 0])) if any(f > 0 for f in freqs) else 150.0

    call(pitch_tier, "Formula",
         f"if x >= {t0:.4f} and x <= {t1:.4f} then {mean_f0:.2f} + (self - {mean_f0:.2f}) * {variance_fraction:.3f} else self fi")
    call([pitch_tier, manipulation], "Replace pitch tier")
    result_snd = call(manipulation, "Get resynthesis (overlap-add)")
    result_y = result_snd.values.flatten().astype(np.float32)

    regions = [{
        "start_s": round(t0, 3),
        "end_s": round(t1, 3),
        "mechanism": f"f0_variance_compress_{variance_fraction:.2f}_bounded_{t0:.2f}to{t1:.2f}s",
    }]
    return result_y, regions


# -- Flaw 3: Pauses (splice at acoustic word boundaries) ------------------------

def apply_pause(
    y: np.ndarray, sr: int,
    word_timestamps: list[dict],
    count: int, duration_range: tuple[float, float],
) -> tuple[np.ndarray, list[dict]]:
    """
    Insert *count* unplanned pauses at word boundaries from TextGrid timestamps.
    Returns (modified_audio, flaw_regions).
    """
    if not word_timestamps or len(word_timestamps) < 3:
        return y, []

    from pydub import AudioSegment

    y_int16 = (np.clip(y, -1, 1) * 32767).astype(np.int16)
    audio = AudioSegment(y_int16.tobytes(), frame_rate=sr, sample_width=2, channels=1)

    # Pick word boundaries in interior 80% of speech that represent real active speech
    interior = word_timestamps[int(len(word_timestamps) * 0.15) : int(len(word_timestamps) * 0.85)]
    # Filter for words that have positive duration and small preceding gap (active continuous speech)
    eligible = []
    for i in range(1, len(interior)):
        w = interior[i]
        prev = interior[i - 1]
        w_dur = w.get("end", 0.0) - w.get("start", 0.0)
        gap = w.get("start", 0.0) - prev.get("end", 0.0)
        if w_dur >= 0.12 and gap < 0.20:
            eligible.append(w)

    if len(eligible) < count:
        eligible = [w for w in interior if w.get("end", 0.0) - w.get("start", 0.0) >= 0.10]
    if len(eligible) < count:
        eligible = word_timestamps[1:-1]
    count = min(count, len(eligible))

    splice_words = random.sample(eligible, count)
    splice_words.sort(key=lambda w: w["start"])

    silences = []
    for w in splice_words:
        dur_s = random.uniform(*duration_range)
        silence = AudioSegment.silent(duration=int(dur_s * 1000), frame_rate=sr)
        silences.append((w["start"], silence, dur_s))

    result = audio
    # Insert backwards so the *source* positions of earlier splices stay valid
    for w_start, silence, dur_s in reversed(silences):
        ms = int(w_start * 1000)
        result = result[:ms] + silence + result[ms:]

    # Labels live on the OUTPUT timeline: each silence is displaced by every
    # silence inserted before it.
    flaw_regions = []
    offset = 0.0
    for w_start, _, dur_s in silences:
        out_start = w_start + offset
        flaw_regions.append({
            "start_s": round(out_start, 3),
            "end_s": round(out_start + dur_s, 3),
            "mechanism": f"silence_splice_{dur_s:.2f}s_at_src_{w_start:.3f}s",
        })
        offset += dur_s

    result_y = np.array(result.get_array_of_samples()).astype(np.float32) / 32767.0
    return result_y, flaw_regions


def shift_for_insertions(regions: list[dict], pause_regions: list[dict]) -> list[dict]:
    """
    Map regions defined on the pre-splice timeline onto the post-splice timeline.
    *pause_regions* are output-timeline silences produced by apply_pause.
    """
    # Recover each splice's source position and duration.
    splices, offset = [], 0.0
    for p in sorted(pause_regions, key=lambda r: r["start_s"]):
        dur = p["end_s"] - p["start_s"]
        splices.append((p["start_s"] - offset, dur))
        offset += dur

    def shift(t: float, is_end: bool) -> float:
        # Content at or after a splice point moves right; a region ending exactly
        # at a splice point does not absorb the silence.
        return t + sum(d for src, d in splices if (src < t if is_end else src <= t))

    return [
        {**r, "start_s": round(shift(r["start_s"], False), 3), "end_s": round(shift(r["end_s"], True), 3)}
        for r in regions
    ]


# -- Flaw 4: Volume (time-bounded gain reduction) -------------------------------

def apply_volume(
    y: np.ndarray, sr: int,
    gain_db: float, start_frac: float = 0.15, fraction: float = 0.25,
) -> tuple[np.ndarray, list[dict]]:
    """
    Apply gain reduction of *gain_db* dB over a bounded window [start_frac, start_frac + fraction].
    """
    total = len(y)
    region_len = int(total * fraction)
    start_idx = int(total * start_frac)
    end_idx = min(start_idx + region_len, total)

    gain_linear = 10 ** (gain_db / 20.0)
    result = y.copy()

    # Fade in / out at boundaries to avoid clicks
    xfade_len = int(0.02 * sr)
    env = np.ones(end_idx - start_idx, dtype=np.float32) * gain_linear
    if len(env) > 2 * xfade_len:
        env[:xfade_len] = np.linspace(1.0, gain_linear, xfade_len)
        env[-xfade_len:] = np.linspace(gain_linear, 1.0, xfade_len)
    result[start_idx:end_idx] *= env

    start_s = start_idx / sr
    end_s = end_idx / sr
    regions = [{
        "start_s": round(start_s, 3),
        "end_s": round(end_s, 3),
        "mechanism": f"gain_{gain_db}dB_bounded_{start_s:.2f}to{end_s:.2f}s",
    }]
    return result, regions


# -- Flaw 5: Clarity (time-bounded crossfade + STFT blur + plosive softening) ---

def apply_clarity(
    y: np.ndarray, sr: int,
    phone_timestamps: list[dict],
    crossfade_ms_range: tuple[int, int],
    smoothing_bins: int,
    transient_db: float,
    start_frac: float = 0.20,
    end_frac: float = 0.65,
) -> tuple[np.ndarray, list[dict]]:
    """
    Reduces articulatory clarity over the bounded window [start_frac, end_frac]:
    1. Phone-boundary crossfade: blends adjacent phones
    2. STFT magnitude smoothing: blurs formant peaks
    3. Transient softening on stop-consonant bursts
    """
    import librosa
    from scipy.signal import fftconvolve

    total_dur = len(y) / sr
    t0 = total_dur * start_frac
    t1 = total_dur * end_frac
    i0 = int(t0 * sr)
    i1 = int(t1 * sr)

    result = y.copy()
    crossfade_ms = random.randint(*crossfade_ms_range)
    crossfade_samples = int(crossfade_ms * sr / 1000)

    # 1. Phone crossfade within window
    for i in range(1, len(phone_timestamps) - 1):
        ph_st = phone_timestamps[i]["start"]
        if ph_st < t0 or ph_st > t1:
            continue
        boundary_idx = int(ph_st * sr)
        half = crossfade_samples // 2
        lo = max(0, boundary_idx - half)
        hi = min(len(result), boundary_idx + half)
        if hi > lo and boundary_idx > lo and boundary_idx < hi:
            pre_audio = result[lo:boundary_idx]
            post_audio = result[boundary_idx:hi]
            fade_len = hi - lo
            pre_ext = np.interp(np.linspace(0, len(pre_audio) - 1, fade_len), np.arange(len(pre_audio)), pre_audio)
            post_ext = np.interp(np.linspace(0, len(post_audio) - 1, fade_len), np.arange(len(post_audio)), post_audio)
            fade = np.linspace(0.0, 1.0, fade_len)
            result[lo:hi] = pre_ext * (1.0 - fade) + post_ext * fade

    # 2. STFT smoothing on target segment [i0:i1]
    seg = result[i0:i1]
    if len(seg) > 2048:
        n_fft = 2048
        hop_length = 512
        S = librosa.stft(seg, n_fft=n_fft, hop_length=hop_length)
        mag = np.abs(S)
        phase = np.angle(S)
        kernel = np.ones((smoothing_bins, 1)) / smoothing_bins
        smoothed_mag = fftconvolve(mag, kernel, mode="same")
        S_smoothed = smoothed_mag * np.exp(1j * phase)
        seg_smoothed = librosa.istft(S_smoothed, hop_length=hop_length, length=len(seg))
        xfade = int(0.02 * sr)
        if len(seg_smoothed) > 2 * xfade:
            fade_in = np.linspace(0.0, 1.0, xfade)
            seg_smoothed[:xfade] = seg_smoothed[:xfade] * fade_in + seg[:xfade] * (1.0 - fade_in)
            seg_smoothed[-xfade:] = seg_smoothed[-xfade:] * (1.0 - fade_in) + seg[-xfade:] * fade_in
        result[i0:i1] = seg_smoothed

    # 3. Transient softening on stop plosives within window
    plosives = {"p", "b", "t", "d", "k", "g"}
    gain_linear = 10 ** (transient_db / 20.0)
    for ph in phone_timestamps:
        ph_st = ph.get("start", 0.0)
        if ph_st < t0 or ph_st > t1:
            continue
        ph_text = ph.get("phone", "").lower()
        is_plosive = (
            (ph_text in plosives) or
            (any(ph_text.startswith(p) for p in plosives) and not ph_text.startswith("ph"))
        )
        if is_plosive:
            idx_start = int(ph_st * sr)
            idx_end = min(idx_start + int(0.02 * sr), len(result))
            result[idx_start:idx_end] *= gain_linear

    regions = [{
        "start_s": round(t0, 3),
        "end_s": round(t1, 3),
        "mechanism": f"clarity_xfade{crossfade_ms}ms_smooth{smoothing_bins}bin_bounded_{t0:.2f}to{t1:.2f}s",
    }]
    return result.astype(np.float32), regions


# -- Flaw 6: Combined egregious -------------------------------------------------

def apply_combined_egregious(
    y: np.ndarray, sr: int,
    word_timestamps: list[dict],
    phone_timestamps: list[dict],
    config: dict,
) -> tuple[np.ndarray, list[dict]]:
    """Compound flaw chaining pacing + pitch + pauses across distinct time-bounded intervals."""
    params = config.get("flaw_params", {})
    all_regions = []

    # 1. Volume drop over 10% to 25%
    vol_db = params.get("volume", {}).get("egregious", {}).get("db", -18)
    y, r_vol = apply_volume(y, sr, gain_db=vol_db, start_frac=0.10, fraction=0.15)
    all_regions.extend(r_vol)

    # 2. Pitch monotone over 25% to 55%
    pitch_var = params.get("pitch", {}).get("egregious", 0.10)
    y, r_pitch = apply_pitch_monotone(y, sr, pitch_var, start_frac=0.25, end_frac=0.55)
    all_regions.extend(r_pitch)

    # 3. Clarity degradation over 55% to 80%
    clar_cfg = params.get("clarity", {}).get("egregious", {})
    y, r_clar = apply_clarity(
        y, sr, phone_timestamps,
        clar_cfg.get("crossfade_ms", [70, 90]),
        clar_cfg.get("smoothing_bins", 15),
        clar_cfg.get("transient_db", -8.0),
        start_frac=0.55, end_frac=0.80,
    )
    all_regions.extend(r_clar)

    # 4. Awkward pause splices
    pause_cfg = params.get("pause", {}).get("egregious", {})
    count = pause_cfg.get("count", 2)
    dur_range = pause_cfg.get("duration_range", [1.0, 1.8])
    y, r_pause = apply_pause(y, sr, word_timestamps, count=count, duration_range=dur_range)
    # Earlier regions were labelled on the pre-splice timeline
    all_regions = shift_for_insertions(all_regions, r_pause) + r_pause

    return y.astype(np.float32), sorted(all_regions, key=lambda r: r["start_s"])


# -- Flaw 7: Near-perfect ------------------------------------------------------

def apply_near_perfect(y: np.ndarray, sr: int, config: dict) -> tuple[np.ndarray, list[dict]]:
    """Apply mild / near-perfect delivery perturbation over bounded window."""
    params = config.get("flaw_params", {})
    rate = params.get("pacing", {}).get("near_perfect", [0.88, 1.12])[1]
    return apply_pacing(y, sr, rate, start_frac=0.25, end_frac=0.60)


# -- Label writer --------------------------------------------------------------

def write_label(
    clip_id: str, suffix: str, flaw_regions: list[dict], meta: dict,
    transcript_path: str, audio_path: str, labels_dir: Path = LABELS_DIR,
):
    flaw_meta = FLAW_SUFFIX[suffix]
    label = {
        "clip_id": clip_id,
        "source_text_id": meta["text_id"],
        "speaker_id": meta["speaker_id"],
        "speaker_source": meta["speaker_source"],
        "flaw_type": flaw_meta["flaw_type"],
        "severity_tier": flaw_meta["severity_tier"],
        "split": meta["split"],
        "flaw_regions": flaw_regions,
        "label_timeline": "output",
        "transcript_path": transcript_path,
        "audio_path": audio_path,
    }
    out = labels_dir / f"{clip_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(label, f, indent=2)


# -- Main ----------------------------------------------------------------------

def process_text(text_id: int, only_flaw: str | None, config: dict, force: bool = True):
    good_path = AUDIO_DIR / f"text_{text_id:02d}_good.wav"
    if not good_path.exists():
        print(f"  SKIP text_{text_id:02d}: GOOD clip not found at {good_path}")
        return
    y, sr = load_audio(good_path)
    generate_set(
        clip_prefix=f"text_{text_id:02d}",
        y=y, sr=sr,
        word_ts=load_word_timestamps(text_id),
        phone_ts=load_phone_timestamps(text_id),
        meta={**TEXT_META[text_id], "text_id": text_id},
        audio_dir=AUDIO_DIR, labels_dir=LABELS_DIR,
        config=config, only_flaw=only_flaw, force=force,
    )


def generate_set(
    clip_prefix: str, y: np.ndarray, sr: int,
    word_ts: list[dict], phone_ts: list[dict], meta: dict,
    audio_dir: Path, labels_dir: Path, config: dict,
    only_flaw: str | None = None, force: bool = True,
):
    """Generate every flaw variant of one GOOD recording (deterministic per clip)."""
    params = config.get("flaw_params", {})
    transcript_rel = f"dataset/transcripts/text_{meta['text_id']:02d}.txt"

    def generate_and_save(suffix: str, fn):
        if only_flaw and suffix != only_flaw:
            return
        clip_id = f"{clip_prefix}_{suffix}"
        out_path = audio_dir / f"{clip_id}.wav"
        if out_path.exists() and not force:
            print(f"  EXISTS: {out_path.name}")
            return
        print(f"  Generating {suffix}...")
        random.seed(clip_id)           # reproducible splice positions / crossfade lengths
        result_y, regions = fn()
        save_audio(result_y, out_path, sr)
        write_label(clip_id, suffix, regions, meta, transcript_rel,
                    out_path.relative_to(ROOT).as_posix(), labels_dir)

    # 1. Standard Moderate Tier
    generate_and_save("pacing_mod", lambda: apply_pacing(
        y, sr, params.get("pacing", {}).get("moderate", [0.75, 1.30])[0], start_frac=0.20, end_frac=0.65))

    generate_and_save("pitch_mod", lambda: apply_pitch_monotone(
        y, sr, params.get("pitch", {}).get("moderate", 0.40), start_frac=0.20, end_frac=0.65))

    pause_mod_cfg = params.get("pause", {}).get("moderate", {})
    generate_and_save("pause_mod", lambda: apply_pause(
        y, sr, word_ts,
        pause_mod_cfg.get("count", 1),
        pause_mod_cfg.get("duration_range", [0.40, 0.80])))

    vol_cfg = params.get("volume", {}).get("moderate", {})
    generate_and_save("volume_mod", lambda: apply_volume(
        y, sr, vol_cfg.get("db", -10), start_frac=0.15, fraction=vol_cfg.get("fraction", 0.20)))

    clar_cfg = params.get("clarity", {}).get("moderate", {})
    generate_and_save("clarity_mod", lambda: apply_clarity(
        y, sr, phone_ts,
        clar_cfg.get("crossfade_ms", [40, 60]),
        clar_cfg.get("smoothing_bins", 7),
        clar_cfg.get("transient_db", -4.0),
        start_frac=0.20, end_frac=0.65))

    generate_and_save("combined_egregious", lambda: apply_combined_egregious(
        y, sr, word_ts, phone_ts, config))

    generate_and_save("nearperfect", lambda: apply_near_perfect(y, sr, config))

    # 2. Gradient: Mild Tier
    generate_and_save("pacing_mild", lambda: apply_pacing(
        y, sr, params.get("pacing", {}).get("near_perfect", [0.88, 1.12])[1], start_frac=0.25, end_frac=0.60))

    generate_and_save("pitch_mild", lambda: apply_pitch_monotone(
        y, sr, params.get("pitch", {}).get("near_perfect", 0.70), start_frac=0.25, end_frac=0.60))

    pause_mild_cfg = params.get("pause", {}).get("near_perfect", {})
    generate_and_save("pause_mild", lambda: apply_pause(
        y, sr, word_ts,
        pause_mild_cfg.get("count", 1),
        pause_mild_cfg.get("duration_range", [0.20, 0.35])))

    vol_mild_cfg = params.get("volume", {}).get("near_perfect", {})
    generate_and_save("volume_mild", lambda: apply_volume(
        y, sr, vol_mild_cfg.get("db", -6), start_frac=0.20, fraction=vol_mild_cfg.get("fraction", 0.15)))

    clar_mild_cfg = params.get("clarity", {}).get("near_perfect", {})
    generate_and_save("clarity_mild", lambda: apply_clarity(
        y, sr, phone_ts,
        clar_mild_cfg.get("crossfade_ms", [15, 25]),
        clar_mild_cfg.get("smoothing_bins", 3),
        clar_mild_cfg.get("transient_db", -2.0),
        start_frac=0.25, end_frac=0.60))

    # 3. Gradient: Egregious Tier
    generate_and_save("pacing_egregious", lambda: apply_pacing(
        y, sr, params.get("pacing", {}).get("egregious", [0.65, 1.50])[0], start_frac=0.15, end_frac=0.75))

    generate_and_save("pitch_egregious", lambda: apply_pitch_monotone(
        y, sr, params.get("pitch", {}).get("egregious", 0.10), start_frac=0.15, end_frac=0.75))

    pause_egr_cfg = params.get("pause", {}).get("egregious", {})
    generate_and_save("pause_egregious", lambda: apply_pause(
        y, sr, word_ts,
        pause_egr_cfg.get("count", 3),
        pause_egr_cfg.get("duration_range", [1.0, 2.0])))

    vol_egr_cfg = params.get("volume", {}).get("egregious", {})
    generate_and_save("volume_egregious", lambda: apply_volume(
        y, sr, vol_egr_cfg.get("db", -18), start_frac=0.10, fraction=vol_egr_cfg.get("fraction", 0.30)))

    clar_egr_cfg = params.get("clarity", {}).get("egregious", {})
    generate_and_save("clarity_egregious", lambda: apply_clarity(
        y, sr, phone_ts,
        clar_egr_cfg.get("crossfade_ms", [70, 90]),
        clar_egr_cfg.get("smoothing_bins", 15),
        clar_egr_cfg.get("transient_db", -8.0),
        start_frac=0.15, end_frac=0.75))


def main():
    import yaml

    parser = argparse.ArgumentParser(description="Generate BAD clips with severity gradient and time-bounded regions")
    parser.add_argument("--text", type=int, help="Only process this text_id (1-8)")
    parser.add_argument("--flaw", type=str, help="Only generate this flaw suffix")
    parser.add_argument("--no-force", action="store_true", help="Do not overwrite existing files")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    text_ids = [args.text] if args.text else list(range(1, 9))
    force = not args.no_force

    for tid in text_ids:
        print(f"\n{'='*60}")
        print(f"Processing text_{tid:02d} (force={force})")
        process_text(tid, args.flaw, config, force=force)

    print("\nFlaw generation complete.")
    print("Dataset audio regenerated in dataset/audio/ and labels written to dataset/labels/")


if __name__ == "__main__":
    main()
