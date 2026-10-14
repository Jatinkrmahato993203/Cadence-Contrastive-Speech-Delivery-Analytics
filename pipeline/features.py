"""
pipeline/features.py
--------------------
Acoustic feature extraction over force-aligned words (§2.2).

Every spectral representation is computed ONCE for the whole clip at a 10 ms
hop and then pooled over each word's aligned interval:

  - FFT / STFT magnitude  -> spectral centroid, flatness, contrast, flux,
                             high-frequency energy ratio
  - MFCCs (13)            -> per-word mean vector
  - F0 contour            -> Praat autocorrelation pitch (parselmouth)
  - HNR contour           -> Praat cross-correlation harmonicity
  - RMS energy            -> per-word level in dB
  - Timing                -> word duration, silence before the word
  - Clip-level voice quality: jitter (local), shimmer (local)

Speech rate and pitch variability need a neighbourhood of words, so they are
derived in normalizer.py from these per-word records.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

HOP_S = 0.01
N_FFT = 512          # 32 ms window at 16 kHz
HF_SPLIT_HZ = 3000.0


# -- Records ------------------------------------------------------------------------

@dataclass
class WordFeatures:
    word_idx: int
    word: str
    start: float
    end: float
    f0_hz: float = 0.0                                     # median voiced F0 over the word
    f0_values: list[float] = field(default_factory=list)   # frame-level F0 (0 = unvoiced)
    rms_db: float = -100.0                                 # word RMS level
    duration_s: float = 0.0
    pause_before_s: float = 0.0                            # silence between previous word and this one
    mfcc_mean: list[float] = field(default_factory=list)
    spectral_centroid: float = 0.0                         # Hz
    spectral_flatness: float = 0.0                         # 0-1
    spectral_contrast: float = 0.0                         # dB, mean over bands
    spectral_flux: float = 0.0                             # mean frame-to-frame log-spectral change
    hf_ratio_db: float = 0.0                               # energy above 3 kHz relative to below
    hnr_db: float = 0.0                                    # mean harmonics-to-noise ratio over voiced frames
    uncertain: bool = False                                # alignment confidence flag (from aligner)


@dataclass
class ClipFeatures:
    clip_path: str
    duration_s: float
    words: list[WordFeatures]
    f0_contour: list[float]          # Hz per 10 ms frame (0 = unvoiced)
    rms_contour: list[float]         # dB per 10 ms frame
    jitter: float = 0.0              # clip-level
    shimmer: float = 0.0             # clip-level


# -- Clip-level analyses ----------------------------------------------------------

def speaker_pitch_range(snd) -> tuple[float, float]:
    """
    Speaker-adaptive pitch search range (De Looze & Hirst, 2008): a wide first pass
    (50-700 Hz) estimates the speaker's register, then floor = 0.72 x q35 and
    ceiling = 1.9 x q65. A fixed 60-500 Hz range mis-tracks deep voices (octave
    errors near the floor) and very high ones.
    """
    from parselmouth.praat import call

    first = call(snd, "To Pitch (ac)", HOP_S, 50.0, 15, "no", 0.01, 0.45, 0.01, 0.35, 0.14, 700.0)
    v = first.selected_array["frequency"]
    v = v[v > 0]
    if len(v) < 20:
        return 60.0, 500.0
    q35, q65 = np.percentile(v, [35, 65])
    return float(np.clip(0.72 * q35, 40.0, 200.0)), float(np.clip(1.9 * q65, 250.0, 800.0))


def _praat_contours(y: np.ndarray, sr: int, n_frames: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    """F0 (Hz) and HNR (dB) per 10 ms frame, plus clip jitter/shimmer."""
    f0 = np.zeros(n_frames)
    hnr = np.full(n_frames, np.nan)
    jitter = shimmer = 0.0
    try:
        import parselmouth
        from parselmouth.praat import call
    except ImportError:
        logger.warning("parselmouth not installed — F0/HNR unavailable")
        return f0, hnr, jitter, shimmer

    snd = parselmouth.Sound(values=y.astype(np.float64), sampling_frequency=float(sr))
    times = np.arange(n_frames) * HOP_S

    # Praat's default silence thresholds are relative to the clip PEAK, so a passage
    # spoken (or recorded) 15-20 dB quieter would lose its pitch and HNR entirely.
    # 0.01 of peak (-40 dB) keeps quiet speech; the voicing threshold still rejects noise.
    floor, ceiling = speaker_pitch_range(snd)
    pitch = call(snd, "To Pitch (ac)", HOP_S, floor, 15, "no", 0.01, 0.45, 0.01, 0.35, 0.14, ceiling)
    pt, pv = pitch.xs(), pitch.selected_array["frequency"]
    if len(pt):
        idx = np.clip(np.round((times - pt[0]) / HOP_S).astype(int), 0, len(pv) - 1)
        inside = (times >= pt[0] - HOP_S) & (times <= pt[-1] + HOP_S)
        f0 = np.where(inside, pv[idx], 0.0)

    try:
        harm = call(snd, "To Harmonicity (cc)", HOP_S, 75, 0.01, 1.0)
        hv = harm.values[0]
        ht = np.array([harm.get_time_from_frame_number(i + 1) for i in range(len(hv))])
        if len(ht):
            idx = np.clip(np.round((times - ht[0]) / HOP_S).astype(int), 0, len(hv) - 1)
            vals = hv[idx]
            hnr = np.where(vals > -150, vals, np.nan)      # Praat marks unvoiced as -200
    except Exception:
        pass

    try:
        pp = call(snd, "To PointProcess (periodic, cc)", 75, 500)
        jitter = float(call(pp, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3))
        shimmer = float(call([snd, pp], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6))
        jitter = 0.0 if np.isnan(jitter) else jitter
        shimmer = 0.0 if np.isnan(shimmer) else shimmer
    except Exception:
        pass
    return f0, hnr, jitter, shimmer


def _spectral_frames(y: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    """Frame-level spectral descriptors at a 10 ms hop (one STFT for the whole clip)."""
    import librosa

    hop = int(sr * HOP_S)
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=hop, center=True)) + 1e-10
    P = S ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    logS = np.log(S)

    # High-frequency ratio from the cepstral spectral ENVELOPE (fine structure removed):
    # the envelope carries articulation (formants, consonant spectra), while the raw
    # spectrum also depends on harmonic spacing, i.e. on the speaker's F0 — raising a
    # voice by 5 semitones moved the raw ratio enough to read as a clarity problem.
    ceps = np.fft.irfft(logS, n=N_FFT, axis=0)
    q = int(0.0016 * sr)
    lifter = np.zeros(N_FFT)
    lifter[:q] = 1.0
    lifter[-q + 1:] = 1.0
    env_power = np.exp(2.0 * np.fft.rfft(ceps * lifter[:, None], axis=0).real)
    hf = env_power[freqs >= HF_SPLIT_HZ].sum(axis=0)
    lf = env_power[(freqs >= 80) & (freqs < HF_SPLIT_HZ)].sum(axis=0)
    flux = np.r_[0.0, np.sqrt(np.mean(np.diff(logS, axis=1) ** 2, axis=0))]

    return {
        "rms_db": 20 * np.log10(np.sqrt(P.sum(axis=0) / N_FFT) + 1e-10),
        # Speech level in the 80-2000 Hz band, where vowel energy dominates: a gain change
        # shows in full, while a loss of consonant / high-frequency energy (a clarity
        # problem) does not masquerade as a volume drop.
        "level_db": 10 * np.log10(P[(freqs >= 80) & (freqs < 2000)].sum(axis=0) / N_FFT + 1e-10),
        "centroid": librosa.feature.spectral_centroid(S=S, sr=sr)[0],
        "flatness": librosa.feature.spectral_flatness(S=S)[0],
        "contrast": librosa.feature.spectral_contrast(S=S, sr=sr, fmin=150.0, n_bands=5).mean(axis=0),
        "flux": flux,
        "hf_ratio_db": 10 * np.log10(hf / (lf + 1e-10) + 1e-10),
        "mfcc": librosa.feature.mfcc(S=librosa.power_to_db(
            librosa.feature.melspectrogram(S=P, sr=sr, n_mels=40)), n_mfcc=13),
    }


def _frame_range(start: float, end: float, n: int) -> slice:
    i0 = min(max(int(round(start / HOP_S)), 0), n - 1)
    i1 = min(max(int(round(end / HOP_S)), i0 + 1), n)
    return slice(i0, i1)


def _mean(x: np.ndarray, default: float = 0.0) -> float:
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else default


# -- Public API -----------------------------------------------------------------

def extract(audio_path: str, alignment, pause_gap_s: float = 0.15) -> ClipFeatures:
    """
    Extract per-word acoustic features for *audio_path* using *alignment*
    (pipeline.aligner.AlignmentResult). *pause_gap_s* is kept for API
    compatibility; pause durations are always reported in seconds.
    """
    y, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    duration_s = len(y) / sr

    spec = _spectral_frames(y, sr)
    n = spec["rms_db"].shape[0]
    f0, hnr, jitter, shimmer = _praat_contours(y, sr, n)

    words_out: list[WordFeatures] = []
    prev_end = None
    for idx, wa in enumerate(alignment.words):
        start = max(0.0, min(wa.start, duration_s))
        end = max(start + 0.02, min(wa.end, duration_s))
        sl = _frame_range(start, end, n)

        f0_w = f0[sl]
        voiced = f0_w[f0_w > 0]
        energy = 10 ** (spec["level_db"][sl] / 10)

        words_out.append(WordFeatures(
            word_idx=idx,
            word=wa.word,
            start=round(start, 3),
            end=round(end, 3),
            f0_hz=float(np.median(voiced)) if len(voiced) else 0.0,
            f0_values=[float(v) for v in f0_w],
            rms_db=float(10 * np.log10(np.mean(energy) + 1e-12)),
            duration_s=float(getattr(wa, "articulation_s", 0.0) or round(end - start, 3)),
            pause_before_s=round(max(0.0, start - prev_end), 3) if prev_end is not None else 0.0,
            mfcc_mean=spec["mfcc"][:, sl].mean(axis=1).tolist(),
            spectral_centroid=_mean(spec["centroid"][sl]),
            spectral_flatness=_mean(spec["flatness"][sl]),
            spectral_contrast=_mean(spec["contrast"][sl]),
            spectral_flux=_mean(spec["flux"][sl]),
            hf_ratio_db=_mean(spec["hf_ratio_db"][sl]),
            hnr_db=_mean(hnr[sl]),
            uncertain=bool(getattr(wa, "uncertain", False)),
        ))
        prev_end = end

    return ClipFeatures(
        clip_path=audio_path,
        duration_s=duration_s,
        words=words_out,
        f0_contour=f0.tolist(),
        rms_contour=spec["rms_db"].tolist(),
        jitter=jitter,
        shimmer=shimmer,
    )
