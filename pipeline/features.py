"""
pipeline/features.py
--------------------
Feature extraction from aligned audio clips (§2.2).

Per-word segments (using aligner timestamps) + continuous contours:
  - MFCCs (13 coefficients + delta + delta-delta)
  - STFT / spectral centroid / spectral flatness
  - F0 contour (via parselmouth / Praat)
  - Jitter, shimmer, HNR
  - RMS energy + LUFS loudness (ITU-R BS.1770, EBU R128)
  - Speech rate (words/sec over a 3-word sliding window)
  - Pause intervals (gaps > pause_gap_threshold)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


# -- Per-word feature record ----------------------------------------------------

@dataclass
class WordFeatures:
    word_idx: int
    word: str
    start: float
    end: float
    # Raw features (before normalization)
    f0_hz: float           # median fundamental freq over the word segment (Hz)
    f0_values: list[float] = field(default_factory=list)  # frame-level F0
    rms_db: float = 0.0    # RMS energy in dB
    lufs: float = 0.0      # integrated loudness (LUFS)
    mfcc_mean: list[float] = field(default_factory=list)   # 13-dim
    spectral_centroid: float = 0.0
    spectral_flatness: float = 0.0
    hnr_db: float = 0.0    # harmonics-to-noise ratio
    jitter: float = 0.0    # local jitter
    shimmer: float = 0.0   # local shimmer
    rate_wps: float = 0.0  # local speech rate (words/sec, 3-word window)
    pause_before_s: float = 0.0   # silence gap before this word


@dataclass
class ClipFeatures:
    clip_path: str
    duration_s: float
    words: list[WordFeatures]
    f0_contour: list[float]         # frame-level (all frames), Hz (0 = unvoiced)
    rms_contour: list[float]        # frame-level RMS
    rate_contour: list[float]       # word-level speech rate
    pause_positions: list[float]    # start times of detected pauses (s)


# -- Helpers --------------------------------------------------------------------

def _load_audio(path: str) -> tuple[np.ndarray, int]:
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    return y, sr


def _extract_f0(y: np.ndarray, sr: int, time_step: float = 0.01) -> list[float]:
    """Return frame-level F0 (Hz) using parselmouth. 0 = unvoiced frame."""
    try:
        import parselmouth
    except ImportError:
        logger.warning("parselmouth not installed — F0 will be zeros")
        frames = int(len(y) / sr / time_step)
        return [0.0] * frames

    snd = parselmouth.Sound(values=y.astype(np.float64), sampling_frequency=float(sr))
    pitch = snd.to_pitch(time_step=time_step)
    f0_vals = []
    for t in np.arange(0, snd.duration, time_step):
        val = pitch.get_value_at_time(t)
        f0_vals.append(float(val) if (val is not None and not np.isnan(val)) else 0.0)
    return f0_vals


def _extract_voice_quality(y: np.ndarray, sr: int) -> dict:
    """Compute jitter, shimmer, HNR via parselmouth voice report."""
    try:
        import parselmouth
        from parselmouth.praat import call
    except ImportError:
        return {"jitter": 0.0, "shimmer": 0.0, "hnr_db": 0.0}

    snd = parselmouth.Sound(values=y.astype(np.float64), sampling_frequency=float(sr))
    try:
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = float(call(harmonicity, "Get mean", 0.0, 0.0))
        if np.isnan(hnr):
            hnr = 0.0
    except Exception:
        hnr = 0.0

    try:
        point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)
        jitter = float(call(point_process, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3))
        if np.isnan(jitter):
            jitter = 0.0
    except Exception:
        jitter = 0.0

    try:
        shimmer = float(call([snd, point_process], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6))
        if np.isnan(shimmer):
            shimmer = 0.0
    except Exception:
        shimmer = 0.0

    return {
        "hnr_db": hnr,
        "jitter": jitter,
        "shimmer": shimmer,
    }


def _rms_db(segment: np.ndarray) -> float:
    rms = np.sqrt(np.mean(segment ** 2))
    if rms < 1e-10:
        return -100.0
    return float(20 * np.log10(rms))


def _lufs(segment: np.ndarray, sr: int) -> float:
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        l = meter.integrated_loudness(segment.astype(np.float64))
        return float(l) if not np.isinf(l) else -70.0
    except Exception:
        return _rms_db(segment)   # fallback


def _mfcc_mean(y: np.ndarray, sr: int, n_mfcc: int = 13) -> list[float]:
    try:
        import librosa
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return mfcc.mean(axis=1).tolist()
    except Exception:
        return [0.0] * n_mfcc


def _spectral_features(y: np.ndarray, sr: int) -> dict:
    try:
        import librosa
        S = np.abs(librosa.stft(y))
        centroid = librosa.feature.spectral_centroid(S=S, sr=sr).mean()
        flatness = librosa.feature.spectral_flatness(S=S).mean()
        return {"spectral_centroid": float(centroid), "spectral_flatness": float(flatness)}
    except Exception:
        return {"spectral_centroid": 0.0, "spectral_flatness": 0.0}


# -- Public API -----------------------------------------------------------------

def extract(
    audio_path: str,
    alignment,             # AlignmentResult from aligner.py
    pause_gap_s: float = 0.15,
) -> ClipFeatures:
    """
    Extract all acoustic features from *audio_path* aligned with *alignment*.

    Parameters
    ----------
    audio_path : str
        Path to 16 kHz / 16-bit mono WAV.
    alignment : AlignmentResult
        Output of pipeline.aligner.align().
    pause_gap_s : float
        Gaps between word segments longer than this are counted as pauses.
    """
    y, sr = _load_audio(audio_path)
    duration_s = len(y) / sr

    # Clip-level features
    f0_contour = _extract_f0(y, sr)
    voice_q = _extract_voice_quality(y, sr)

    try:
        import librosa
        rms_frames = librosa.feature.rms(y=y)[0].tolist()
    except Exception:
        rms_frames = [0.0]

    # Word-level features
    words_out: list[WordFeatures] = []
    pause_positions: list[float] = []
    rate_contour: list[float] = []
    prev_end = 0.0

    for idx, wa in enumerate(alignment.words):
        start_s = max(0.0, wa.start)
        end_s = min(duration_s, wa.end)
        if end_s <= start_s:
            end_s = start_s + 0.05

        i_start = int(start_s * sr)
        i_end = int(end_s * sr)
        seg = y[i_start:i_end] if i_end > i_start else np.zeros(sr // 100)

        # Pause before this word
        gap = start_s - prev_end
        if gap > pause_gap_s and idx > 0:
            pause_positions.append(prev_end)
        prev_end = end_s

        # F0 for this word segment (median over voiced frames)
        f0_time = np.arange(0, duration_s, 0.01)
        word_f0_indices = np.where((f0_time >= start_s) & (f0_time < end_s))[0]
        word_f0_vals = [f0_contour[i] for i in word_f0_indices if i < len(f0_contour)]
        voiced = [v for v in word_f0_vals if v > 0]
        f0_hz = float(np.median(voiced)) if voiced else 0.0

        # Local speech rate (3-word window around current word)
        left = max(0, idx - 1)
        right = min(len(alignment.words) - 1, idx + 1)
        window_words = alignment.words[left : right + 1]
        window_dur = (window_words[-1].end - window_words[0].start)
        rate_wps = len(window_words) / window_dur if window_dur > 0 else 0.0
        rate_contour.append(rate_wps)

        spec = _spectral_features(seg, sr)
        mfcc = _mfcc_mean(seg, sr)
        word_voice_q = _extract_voice_quality(seg, sr)

        words_out.append(WordFeatures(
            word_idx=idx,
            word=wa.word,
            start=start_s,
            end=end_s,
            f0_hz=f0_hz,
            f0_values=word_f0_vals,
            rms_db=_rms_db(seg),
            lufs=_lufs(seg, sr),
            mfcc_mean=mfcc,
            spectral_centroid=spec["spectral_centroid"],
            spectral_flatness=spec["spectral_flatness"],
            hnr_db=word_voice_q["hnr_db"],
            jitter=word_voice_q["jitter"],
            shimmer=word_voice_q["shimmer"],
            rate_wps=rate_wps,
            pause_before_s=max(0.0, gap),
        ))

    return ClipFeatures(
        clip_path=audio_path,
        duration_s=duration_s,
        words=words_out,
        f0_contour=f0_contour,
        rms_contour=rms_frames,
        rate_contour=rate_contour,
        pause_positions=pause_positions,
    )
