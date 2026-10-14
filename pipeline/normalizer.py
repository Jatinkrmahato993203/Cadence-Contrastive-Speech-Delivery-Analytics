"""
pipeline/normalizer.py
----------------------
Speaker-agnostic normalization (§2.3).

Every clip is normalized against ITS OWN robust statistics, computed over the
whole clip (not a 10-word bootstrap that a flaw near the start would corrupt):

  Pitch level       -> semitones from the clip's median voiced F0
  Pitch variability -> robust spread (IQR/1.349, semitones) of voiced frames over a 3-word window
  Energy            -> dB relative to the clip's 90th-percentile word level
  Tempo             -> words per second of articulation (pauses excluded),
                       and log2 tempo relative to the clip median
  Pauses            -> seconds of silence before each word
  Clarity           -> chosen spectral measure (config: normalization.clarity_feature)
                       minus the clip median

Why self-normalization and not "measure the participant against the baseline
speaker's profile": two different people reading the same text differ in F0
by up to an octave and in loudness by recording setup. Measuring a participant
in the baseline speaker's units penalises their voice instead of their delivery.
After self-normalization, pitch variability and tempo stay comparable across
speakers (a monotone or rushed reading still shows up), while absolute pitch
and recording gain cancel out.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

CLARITY_FEATURES = ("hnr_db", "spectral_contrast", "spectral_flux", "hf_ratio_db", "spectral_flatness")


def _load_cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# -- Speaker profile ------------------------------------------------------------------

@dataclass
class SpeakerProfile:
    """Robust per-speaker statistics. Only pool clips from the SAME speaker."""
    speaker_id: str
    f0_samples: list[float] = field(default_factory=list)    # voiced F0 frames (Hz)
    rms_samples: list[float] = field(default_factory=list)   # word levels (dB)
    rate_samples: list[float] = field(default_factory=list)  # word tempo (words/s)
    clarity_samples: list[float] = field(default_factory=list)
    bootstrapped: bool = False                               # kept for API compatibility

    @property
    def f0_median(self) -> float:
        voiced = [v for v in self.f0_samples if v > 0]
        return float(np.median(voiced)) if voiced else 0.0

    @property
    def rms_ref(self) -> float:
        return float(np.percentile(self.rms_samples, 90)) if self.rms_samples else 0.0

    # Backwards-compatible alias
    @property
    def rms_95pct(self) -> float:
        return self.rms_ref

    @property
    def rate_median(self) -> float:
        return float(np.median(self.rate_samples)) if self.rate_samples else 0.0

    @property
    def clarity_median(self) -> float:
        return float(np.median(self.clarity_samples)) if self.clarity_samples else 0.0


# -- Normalized records ---------------------------------------------------------------

@dataclass
class NormalizedWord:
    word_idx: int
    word: str
    start: float
    end: float
    pitch_st: float = 0.0           # semitones from the clip's median F0 (0 if unvoiced)
    pitch_var_st: float = 0.0       # local pitch variability (semitones)
    energy_db_rel: float = 0.0      # dB relative to the clip's 90th-percentile word level
    rate_wps: float = 0.0           # local articulation tempo (words / s of speech)
    rate_zscore: float = 0.0        # log2 tempo relative to the clip median (name kept for compatibility)
    pause_flag: bool = False        # silence before this word > pause_gap_threshold_s
    pause_duration_s: float = 0.0
    clarity: float = 0.0            # selected clarity measure minus clip median (higher = clearer)
    clarity_raw: Optional[float] = None   # selected clarity measure, un-normalized (sign applied)
    energy_db: Optional[float] = None     # word level in dB, un-normalized
    clarity_mfcc: float = 0.0       # MFCC cosine distance from the clip mean (diagnostic)
    clarity_hnr: float = 0.0        # raw HNR (dB)
    duration_s: float = 0.0
    mfcc_mean: list[float] = field(default_factory=list)
    hnr_db: float = 0.0
    uncertain: bool = False


@dataclass
class NormalizedClip:
    clip_path: str
    speaker_profile: SpeakerProfile
    words: list[NormalizedWord]
    f0_contour_st: list[float]           # per 10 ms frame (NaN when unvoiced)
    rms_contour_db_rel: list[float]      # per 10 ms frame
    rate_contour_zscore: list[float]     # per word (log2 tempo rel. clip median)
    duration_s: float = 0.0
    frame_hop_s: float = 0.01

    @property
    def word_times(self) -> list[float]:
        return [(w.start + w.end) / 2 for w in self.words]


# -- Helpers ---------------------------------------------------------------------------

def f0_to_semitones(f0_hz: float, median_hz: float) -> float:
    if f0_hz <= 0 or median_hz <= 0:
        return 0.0
    return 12.0 * math.log2(f0_hz / median_hz)


def _cosine_distance(a: list[float], b: list[float]) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def _fill_nan(x: np.ndarray) -> np.ndarray:
    """Replace NaNs by the nearest finite value (or 0 if none)."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if not ok.any():
        return np.zeros_like(x)
    idx = np.arange(len(x))
    return np.interp(idx, idx[ok], x[ok])


def articulation_tempo(durations: list[float], half_window: int = 2) -> np.ndarray:
    """Words per second of articulation over a (2*half_window+1)-word window."""
    d = np.maximum(np.asarray(durations, float), 0.03)
    out = np.empty(len(d))
    for i in range(len(d)):
        lo, hi = max(0, i - half_window), min(len(d), i + half_window + 1)
        out[i] = (hi - lo) / d[lo:hi].sum()
    return out


def _profile_from_clip(clip_features, speaker_id: str, clarity_feature: str) -> SpeakerProfile:
    wfs = clip_features.words
    return SpeakerProfile(
        speaker_id=speaker_id,
        f0_samples=[v for v in clip_features.f0_contour if v > 0],
        rms_samples=[w.rms_db for w in wfs],
        rate_samples=list(articulation_tempo([w.duration_s or (w.end - w.start) for w in wfs])),
        clarity_samples=[float(getattr(w, clarity_feature, 0.0)) for w in wfs],
    )


# -- Public API -----------------------------------------------------------------------

def normalize(
    clip_features,
    speaker_id: str,
    speaker_profile: Optional[SpeakerProfile] = None,
    cfg: Optional[dict] = None,
    pause_gap_s: float = 0.15,
) -> tuple[NormalizedClip, SpeakerProfile]:
    """
    Normalize *clip_features* (pipeline.features.ClipFeatures) to speaker-agnostic units.

    *speaker_profile* may carry statistics from earlier recordings of the SAME
    speaker; they are pooled with this clip. Never pass another speaker's profile.
    """
    if cfg is None:
        cfg = _load_cfg()
    n_cfg = cfg.get("normalization", {})
    clarity_feature = n_cfg.get("clarity_feature", "hnr_db")
    clarity_sign = float(n_cfg.get("clarity_sign", 1.0))   # -1 when the raw measure rises as clarity falls
    pause_gap_s = cfg.get("grounding", {}).get("pause_gap_threshold_s", pause_gap_s)

    wfs = clip_features.words
    profile = _profile_from_clip(clip_features, speaker_id, clarity_feature)
    if speaker_profile is not None:
        if speaker_profile.speaker_id != speaker_id:
            logger.warning(
                "Speaker profile '%s' pooled into clip for '%s' — profiles must come from the same speaker",
                speaker_profile.speaker_id, speaker_id,
            )
        profile.f0_samples = speaker_profile.f0_samples + profile.f0_samples
        profile.rms_samples = speaker_profile.rms_samples + profile.rms_samples
        profile.rate_samples = speaker_profile.rate_samples + profile.rate_samples
        profile.clarity_samples = speaker_profile.clarity_samples + profile.clarity_samples

    f0_med = profile.f0_median
    rms_ref = profile.rms_ref
    rate_med = profile.rate_median or 1.0
    clar_med = profile.clarity_median

    durations = [w.duration_s or (w.end - w.start) for w in wfs]
    tempo = articulation_tempo(durations)

    # Local pitch variability over a 3-word window of voiced frames. Interquartile
    # spread (IQR / 1.349), not std: a few octave errors from the F0 tracker
    # (+/-12 st) would otherwise dominate.
    word_st = []
    for w in wfs:
        v = np.asarray(w.f0_values, float)
        v = v[v > 0]
        word_st.append(12.0 * np.log2(v / f0_med) if len(v) and f0_med > 0 else np.array([]))
    pitch_var = np.full(len(wfs), np.nan)
    for i in range(len(wfs)):
        frames = np.concatenate(word_st[max(0, i - 1): i + 2]) if wfs else np.array([])
        if len(frames) >= 5:
            q75, q25 = np.percentile(frames, [75, 25])
            pitch_var[i] = float((q75 - q25) / 1.349)
    pitch_var = _fill_nan(pitch_var)

    all_mfcc = [w.mfcc_mean for w in wfs if w.mfcc_mean]
    mfcc_ref = list(np.mean(all_mfcc, axis=0)) if all_mfcc else []

    norm_words: list[NormalizedWord] = []
    for i, w in enumerate(wfs):
        clar_raw = float(getattr(w, clarity_feature, 0.0))
        if not np.isfinite(clar_raw):
            clar_raw = clar_med
        norm_words.append(NormalizedWord(
            word_idx=w.word_idx,
            word=w.word,
            start=w.start,
            end=w.end,
            pitch_st=f0_to_semitones(w.f0_hz, f0_med),
            pitch_var_st=float(pitch_var[i]),
            energy_db_rel=float(w.rms_db - rms_ref),
            rate_wps=float(tempo[i]),
            rate_zscore=float(np.log2(tempo[i] / rate_med)),
            pause_flag=w.pause_before_s > pause_gap_s,
            pause_duration_s=float(w.pause_before_s),
            clarity=float(clarity_sign * (clar_raw - clar_med)),
            clarity_raw=float(clarity_sign * clar_raw),
            energy_db=float(w.rms_db),
            clarity_mfcc=_cosine_distance(w.mfcc_mean, mfcc_ref) if mfcc_ref else 0.0,
            clarity_hnr=float(w.hnr_db),
            duration_s=float(durations[i]),
            mfcc_mean=w.mfcc_mean,
            hnr_db=float(w.hnr_db),
            uncertain=bool(getattr(w, "uncertain", False)),
        ))

    contour_st = [
        12.0 * math.log2(v / f0_med) if (v > 0 and f0_med > 0) else float("nan")
        for v in clip_features.f0_contour
    ]
    rms_rel = [r - rms_ref for r in clip_features.rms_contour]

    return (
        NormalizedClip(
            clip_path=clip_features.clip_path,
            speaker_profile=profile,
            words=norm_words,
            f0_contour_st=contour_st,
            rms_contour_db_rel=rms_rel,
            rate_contour_zscore=[w.rate_zscore for w in norm_words],
            duration_s=clip_features.duration_s,
        ),
        profile,
    )
