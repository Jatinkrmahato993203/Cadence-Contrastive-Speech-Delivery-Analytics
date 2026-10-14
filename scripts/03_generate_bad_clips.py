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


# -- Shared helpers --------------------------------------------------------------

SPLICE_MS = 10


def _seed_praat() -> None:
    """
    Praat's overlap-add resynthesis draws random numbers when durations change, so
    two identical calls differ unless its generator is seeded. The seed is taken
    from Python's `random`, which generate_set seeds with the clip id.
    """
    from parselmouth.praat import run
    run(f"random_initializeWithSeedUnsafelyButPredictably ({random.randint(1, 2**31 - 1)})")


def snap_to_gaps(t0: float, t1: float, word_ts: list[dict]) -> tuple[float, float]:
    """Move a window's edges to the nearest gap between words, so no edit starts or ends mid-word."""
    if not word_ts or len(word_ts) < 2:
        return t0, t1
    edges = [word_ts[0]["start"]] + [
        (word_ts[i]["end"] + word_ts[i + 1]["start"]) / 2 for i in range(len(word_ts) - 1)
    ] + [word_ts[-1]["end"]]
    a = min(edges, key=lambda g: abs(g - t0))
    b = min(edges, key=lambda g: abs(g - t1))
    return (a, b) if b > a else (t0, t1)


def _window(y: np.ndarray, sr: int, start_frac: float, end_frac: float, word_ts) -> tuple[int, int]:
    dur = len(y) / sr
    t0, t1 = snap_to_gaps(dur * start_frac, dur * end_frac, word_ts or [])
    return int(t0 * sr), int(t1 * sr)


def splice_edit(y: np.ndarray, sr: int, i0: int, i1: int, edit) -> tuple[np.ndarray, int, int]:
    """
    Apply *edit* (array -> array, length may change) to y[i0:i1] plus a 10 ms margin
    on each side, and splice the result back with 10 ms crossfades. Audio outside
    the window is left bit-identical. Returns (result, out_start, out_end) of the
    edited span on the output timeline.
    """
    n = int(sr * SPLICE_MS / 1000)
    a, b = max(0, i0 - n), min(len(y), i1 + n)
    out = np.asarray(edit(y[a:b].astype(np.float64)), dtype=np.float64)
    na, nb = i0 - a, b - i1
    if len(out) < na + nb + 1:
        return y.copy(), i0, i1
    ramp_a, ramp_b = np.linspace(0, 1, na), np.linspace(1, 0, nb)
    head = y[a:i0] * (1 - ramp_a) + out[:na] * ramp_a
    tail = out[len(out) - nb:] * ramp_b + y[i1:b] * (1 - ramp_b)
    result = np.concatenate([y[:a], head, out[na:len(out) - nb], tail, y[b:]]).astype(np.float32)
    return result, i0, i0 + (len(out) - na - nb)


# -- Flaw 1: Pacing (deterministic PSOLA over a bounded window) ------------------

def apply_pacing(
    y: np.ndarray, sr: int, rate: float,
    start_frac: float = 0.20, end_frac: float = 0.65, word_ts: list[dict] | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """
    Change speaking rate by *rate* (< 1 slower) over a window snapped to word gaps.
    Praat Manipulation + DurationTier + overlap-add resynthesis (PSOLA, pitch kept),
    with Praat's random generator seeded so the result is bit-reproducible.
    """
    import parselmouth
    from parselmouth.praat import call

    def stretch(seg):
        _seed_praat()
        snd = parselmouth.Sound(seg, sampling_frequency=float(sr))
        manip = call(snd, "To Manipulation", 0.01, 60, 600)
        tier = call("Create DurationTier", "stretch", snd.xmin, snd.xmax)
        call(tier, "Add point", (snd.xmin + snd.xmax) / 2, 1.0 / rate)
        call([manip, tier], "Replace duration tier")
        return call(manip, "Get resynthesis (overlap-add)").values.flatten()

    i0, i1 = _window(y, sr, start_frac, end_frac, word_ts)
    result, o0, o1 = splice_edit(y, sr, i0, i1, stretch)
    t0, t1 = o0 / sr, o1 / sr
    return result, [{"start_s": round(t0, 3), "end_s": round(t1, 3),
                     "mechanism": f"time_stretch_{rate}x_bounded_{t0:.2f}to{t1:.2f}s"}]


# -- Flaw 2: Pitch / monotone (F0 variance compression over a bounded window) ----

def apply_pitch_monotone(
    y: np.ndarray, sr: int, variance_fraction: float,
    start_frac: float = 0.20, end_frac: float = 0.65, word_ts: list[dict] | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """
    Compress F0 movement toward the speaker's mean F0 by *variance_fraction* inside
    a window snapped to word gaps (Praat Manipulation, overlap-add resynthesis of the
    window only; the rest of the recording is untouched).
    """
    import parselmouth
    from parselmouth.praat import call

    full = parselmouth.Sound(y.astype(np.float64), sampling_frequency=float(sr))
    pitch = full.to_pitch(time_step=0.01, pitch_floor=60.0, pitch_ceiling=600.0).selected_array["frequency"]
    voiced = pitch[pitch > 0]
    mean_f0 = float(np.mean(voiced)) if len(voiced) else 150.0

    def flatten(seg):
        _seed_praat()
        snd = parselmouth.Sound(seg, sampling_frequency=float(sr))
        manip = call(snd, "To Manipulation", 0.01, 60, 600)
        tier = call(manip, "Extract pitch tier")
        call(tier, "Formula", f"{mean_f0:.3f} + (self - {mean_f0:.3f}) * {variance_fraction:.3f}")
        call([tier, manip], "Replace pitch tier")
        return call(manip, "Get resynthesis (overlap-add)").values.flatten()

    i0, i1 = _window(y, sr, start_frac, end_frac, word_ts)
    result, o0, o1 = splice_edit(y, sr, i0, i1, flatten)
    t0, t1 = o0 / sr, o1 / sr
    return result, [{"start_s": round(t0, 3), "end_s": round(t1, 3),
                     "mechanism": f"f0_variance_compress_{variance_fraction:.2f}_bounded_{t0:.2f}to{t1:.2f}s"}]


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


# -- Flaw 4: Volume (gain envelope) ----------------------------------------------

def apply_volume(
    y: np.ndarray, sr: int,
    gain_db: float, start_frac: float = 0.15, fraction: float = 0.25, word_ts: list[dict] | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Reduce level by *gain_db* over a window snapped to word gaps, with 20 ms fades."""
    i0, i1 = _window(y, sr, start_frac, start_frac + fraction, word_ts)
    g = 10 ** (gain_db / 20.0)
    env = np.full(i1 - i0, g, dtype=np.float32)
    x = int(0.02 * sr)
    if len(env) > 2 * x:
        env[:x] = np.linspace(1.0, g, x)
        env[-x:] = np.linspace(g, 1.0, x)
    result = y.copy()
    result[i0:i1] *= env
    t0, t1 = i0 / sr, i1 / sr
    return result, [{"start_s": round(t0, 3), "end_s": round(t1, 3),
                     "mechanism": f"gain_{gain_db}dB_bounded_{t0:.2f}to{t1:.2f}s"}]


# -- Flaw 5: Clarity (articulatory undershoot) ------------------------------------

def plosive_landmarks(seg: np.ndarray, sr: int) -> list[int]:
    """
    Sample indices of stop-consonant bursts: an abrupt broadband energy rise
    (>= 15 dB within 20 ms) out of a closure (>= 25 dB below the local maximum).
    Acoustic landmarks — no phone transcription needed.
    """
    hop, win = int(0.005 * sr), int(0.010 * sr)
    if len(seg) < 4 * win:
        return []
    seg = np.ascontiguousarray(seg, dtype=np.float64)
    n = (len(seg) - win) // hop + 1
    frames = np.lib.stride_tricks.as_strided(seg, shape=(n, win), strides=(seg.strides[0] * hop, seg.strides[0]))
    db = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)
    loud = np.max(db)
    out, last = [], -10**9
    for i in range(4, n):
        if db[i] - db[i - 4] >= 15.0 and db[i - 4] <= loud - 25.0 and i - last > 20:
            out.append(i * hop)
            last = i
    return out


def articulation_blur(seg: np.ndarray, sr: int, centralize: float, undershoot_ms: float,
                      hf_loss_db: float) -> np.ndarray:
    """
    Model slurred / mumbled articulation by editing ONLY the spectral envelope of each
    frame (formants, consonant spectra) and keeping its fine structure (harmonics):
      * centralization - blend the envelope toward a frequency-smoothed copy
        (weaker formant peaks, the acoustic signature of vowel undershoot);
      * undershoot     - smooth the envelope over time (sluggish articulators,
        smeared transitions between sounds);
      * consonant loss - attenuate energy above 2 kHz (hf_loss_db reached at 6 kHz).
    Voicing and F0 are preserved, so pitch tracking is not disturbed.
    """
    import librosa
    from scipy.ndimage import gaussian_filter1d, uniform_filter1d

    n_fft, hop = 512, 128
    S = librosa.stft(np.asarray(seg, dtype=np.float32), n_fft=n_fft, hop_length=hop)
    mag, phase = np.abs(S) + 1e-9, np.angle(S)
    L = np.log(mag)
    ceps = np.fft.irfft(L, n=n_fft, axis=0)
    lifter = np.zeros(n_fft)
    q = int(0.0016 * sr)                                   # 1.6 ms: envelope vs fine structure
    lifter[:q] = 1.0
    lifter[-q + 1:] = 1.0
    env = np.fft.rfft(ceps * lifter[:, None], axis=0).real
    fine = L - env
    env = (1 - centralize) * env + centralize * gaussian_filter1d(env, sigma=12, axis=0)
    k = max(1, int(round(undershoot_ms / (1000 * hop / sr))))
    if k > 1:
        env = uniform_filter1d(env, size=k, axis=1, mode="nearest")
    f = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    env += (hf_loss_db * np.clip((f - 2000.0) / 4000.0, 0, 1) / 20.0 * np.log(10))[:, None]
    return librosa.istft(np.exp(env + fine) * np.exp(1j * phase), hop_length=hop, length=len(seg))


def apply_clarity(
    y: np.ndarray, sr: int, params: dict,
    start_frac: float = 0.20, end_frac: float = 0.65, word_ts: list[dict] | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """
    Reduce articulatory clarity inside a window snapped to word gaps: spectral-envelope
    centralization + temporal undershoot + high-frequency consonant loss
    (articulation_blur), and softened plosive bursts at acoustic landmarks.
    """
    c, u = float(params.get("centralize", 0.6)), float(params.get("undershoot_ms", 32))
    h, tdb = float(params.get("hf_loss_db", -8.0)), float(params.get("transient_db", -6.0))
    g = 10 ** (tdb / 20.0)

    def degrade(seg):
        out = articulation_blur(seg, sr, c, u, h)
        burst = int(0.015 * sr)
        for b in plosive_landmarks(seg, sr):
            out[b: b + burst] *= g
        return out

    i0, i1 = _window(y, sr, start_frac, end_frac, word_ts)
    result, o0, o1 = splice_edit(y, sr, i0, i1, degrade)
    t0, t1 = o0 / sr, o1 / sr
    return result, [{"start_s": round(t0, 3), "end_s": round(t1, 3),
                     "mechanism": f"clarity_undershoot_c{c:.2f}_u{u:.0f}ms_hf{h:.0f}dB_bounded_{t0:.2f}to{t1:.2f}s"}]


# -- Flaw 6: Combined egregious -------------------------------------------------

def apply_combined_egregious(
    y: np.ndarray, sr: int,
    word_timestamps: list[dict],
    phone_timestamps: list[dict],
    config: dict,
) -> tuple[np.ndarray, list[dict]]:
    """Volume drop, monotone stretch, slurred stretch and long pauses in distinct windows."""
    params = config.get("flaw_params", {})
    w = word_timestamps
    all_regions = []

    y, r = apply_volume(y, sr, params.get("volume", {}).get("egregious", {}).get("db", -18),
                        start_frac=0.10, fraction=0.15, word_ts=w)
    all_regions.extend(r)
    y, r = apply_pitch_monotone(y, sr, params.get("pitch", {}).get("egregious", 0.10),
                                start_frac=0.25, end_frac=0.55, word_ts=w)
    all_regions.extend(r)
    y, r = apply_clarity(y, sr, params.get("clarity", {}).get("egregious", {}),
                         start_frac=0.55, end_frac=0.80, word_ts=w)
    all_regions.extend(r)

    pause_cfg = params.get("pause", {}).get("egregious", {})
    y, r_pause = apply_pause(y, sr, w, count=pause_cfg.get("count", 2),
                             duration_range=pause_cfg.get("duration_range", [1.0, 1.8]))
    # Earlier regions were labelled on the pre-splice timeline
    all_regions = shift_for_insertions(all_regions, r_pause) + r_pause
    return y.astype(np.float32), sorted(all_regions, key=lambda r: r["start_s"])


# -- Flaw 7: Near-perfect ------------------------------------------------------

def apply_near_perfect(y: np.ndarray, sr: int, config: dict,
                       word_ts: list[dict] | None = None) -> tuple[np.ndarray, list[dict]]:
    """
    A barely-noticeable rate change (default 1.07x, half the mild tier's deviation),
    so the gradient has a point between the clean baseline and the mild tier.
    """
    rate = float(config.get("flaw_params", {}).get("nearperfect_rate", 1.07))
    return apply_pacing(y, sr, rate, start_frac=0.25, end_frac=0.60, word_ts=word_ts)


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

    P = lambda f, tier, default: params.get(f, {}).get(tier, default)
    w = word_ts

    # 1. Moderate tier
    generate_and_save("pacing_mod", lambda: apply_pacing(
        y, sr, P("pacing", "moderate", [0.75, 1.30])[0], 0.20, 0.65, w))
    generate_and_save("pitch_mod", lambda: apply_pitch_monotone(
        y, sr, P("pitch", "moderate", 0.40), 0.20, 0.65, w))
    pm = P("pause", "moderate", {})
    generate_and_save("pause_mod", lambda: apply_pause(
        y, sr, w, pm.get("count", 1), pm.get("duration_range", [0.40, 0.80])))
    vm = P("volume", "moderate", {})
    generate_and_save("volume_mod", lambda: apply_volume(
        y, sr, vm.get("db", -10), 0.15, vm.get("fraction", 0.20), w))
    generate_and_save("clarity_mod", lambda: apply_clarity(
        y, sr, P("clarity", "moderate", {}), 0.20, 0.65, w))
    generate_and_save("combined_egregious", lambda: apply_combined_egregious(y, sr, w, phone_ts, config))
    generate_and_save("nearperfect", lambda: apply_near_perfect(y, sr, config, w))

    # 2. Mild tier
    generate_and_save("pacing_mild", lambda: apply_pacing(
        y, sr, P("pacing", "near_perfect", [0.85, 1.15])[1], 0.25, 0.60, w))
    generate_and_save("pitch_mild", lambda: apply_pitch_monotone(
        y, sr, P("pitch", "near_perfect", 0.70), 0.25, 0.60, w))
    pn = P("pause", "near_perfect", {})
    generate_and_save("pause_mild", lambda: apply_pause(
        y, sr, w, pn.get("count", 1), pn.get("duration_range", [0.20, 0.35])))
    vn = P("volume", "near_perfect", {})
    generate_and_save("volume_mild", lambda: apply_volume(
        y, sr, vn.get("db", -6), 0.20, vn.get("fraction", 0.15), w))
    generate_and_save("clarity_mild", lambda: apply_clarity(
        y, sr, P("clarity", "near_perfect", {}), 0.25, 0.60, w))

    # 3. Egregious tier
    generate_and_save("pacing_egregious", lambda: apply_pacing(
        y, sr, P("pacing", "egregious", [0.65, 1.50])[0], 0.15, 0.75, w))
    generate_and_save("pitch_egregious", lambda: apply_pitch_monotone(
        y, sr, P("pitch", "egregious", 0.10), 0.15, 0.75, w))
    pe = P("pause", "egregious", {})
    generate_and_save("pause_egregious", lambda: apply_pause(
        y, sr, w, pe.get("count", 3), pe.get("duration_range", [1.0, 2.0])))
    ve = P("volume", "egregious", {})
    generate_and_save("volume_egregious", lambda: apply_volume(
        y, sr, ve.get("db", -18), 0.10, ve.get("fraction", 0.30), w))
    generate_and_save("clarity_egregious", lambda: apply_clarity(
        y, sr, P("clarity", "egregious", {}), 0.15, 0.75, w))


def main():
    import yaml

    parser = argparse.ArgumentParser(description="Generate BAD clips with severity gradient and time-bounded regions")
    parser.add_argument("--text", type=int, help="Only process this text_id (1-8)")
    parser.add_argument("--flaw", type=str, help="Only generate this flaw suffix")
    parser.add_argument("--no-force", action="store_true", help="Do not overwrite existing files")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    text_ids = [args.text] if args.text else list(range(1, 12))
    force = not args.no_force

    for tid in text_ids:
        print(f"\n{'='*60}")
        print(f"Processing text_{tid:02d} (force={force})")
        process_text(tid, args.flaw, config, force=force)

    print("\nFlaw generation complete.")
    print("Dataset audio regenerated in dataset/audio/ and labels written to dataset/labels/")


if __name__ == "__main__":
    main()
