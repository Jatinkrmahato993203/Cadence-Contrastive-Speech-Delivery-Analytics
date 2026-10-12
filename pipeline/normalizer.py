"""
pipeline/normalizer.py
----------------------
Speaker-agnostic normalization (§2.3).

Pitch   -> semitones from speaker's own median F0
Energy  -> dB relative to speaker's own 95th-pct RMS
Rate    -> z-score relative to speaker's own mean words/sec

New-speaker bootstrap (mode-agnostic):
  For any speaker with no prior history (Mode A re-recording OR Mode B upload),
  per-speaker statistics are bootstrapped from the first BOOTSTRAP_WORDS words of
  the current clip and refined online as more words are processed.
  If prior clips exist, accumulated statistics across ALL prior clips are used.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def _load_cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# -- Speaker profile (accumulates across recordings) ----------------------------

@dataclass
class SpeakerProfile:
    speaker_id: str
    f0_samples: list[float] = field(default_factory=list)    # all voiced F0 Hz values
    rms_samples: list[float] = field(default_factory=list)   # all RMS dB values
    rate_samples: list[float] = field(default_factory=list)  # all word-rate values
    bootstrapped: bool = False   # True if derived from single-clip bootstrap

    @property
    def f0_median(self) -> float:
        voiced = [v for v in self.f0_samples if v > 0]
        return float(np.median(voiced)) if voiced else 200.0  # fallback

    @property
    def rms_95pct(self) -> float:
        return float(np.percentile(self.rms_samples, 95)) if self.rms_samples else 0.0

    @property
    def rate_mean(self) -> float:
        return float(np.mean(self.rate_samples)) if self.rate_samples else 3.0

    @property
    def rate_std(self) -> float:
        return float(np.std(self.rate_samples)) if len(self.rate_samples) > 1 else 1.0


# -- Per-word normalized features ----------------------------------------------

@dataclass
class NormalizedWord:
    word_idx: int
    word: str
    start: float
    end: float
    pitch_st: float         # semitones from speaker median F0
    energy_db_rel: float    # dB relative to speaker 95th-pct RMS
    rate_zscore: float      # z-score vs speaker mean rate
    pause_flag: bool        # True if gap before this word > threshold
    pause_duration_s: float
    clarity_mfcc: float     # MFCC cosine distance from clip mean (0-1)
    clarity_hnr: float      # raw HNR in dB (used as delta in grounding)
    mfcc_mean: list[float]  # raw MFCCs (for cosine distance computation)
    hnr_db: float


@dataclass
class NormalizedClip:
    clip_path: str
    speaker_profile: SpeakerProfile
    words: list[NormalizedWord]
    # Clip-level contours in normalized units for plotting
    f0_contour_st: list[float]
    rms_contour_db_rel: list[float]
    rate_contour_zscore: list[float]


# -- Normalization helpers -----------------------------------------------------

def f0_to_semitones(f0_hz: float, median_hz: float) -> float:
    if f0_hz <= 0 or median_hz <= 0:
        return 0.0
    return 12.0 * math.log2(f0_hz / median_hz)


def _cosine_distance(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# -- Bootstrap helpers ---------------------------------------------------------

def _bootstrap_profile(words_features, n: int, speaker_id: str) -> SpeakerProfile:
    """Build a SpeakerProfile from the first n words of a clip."""
    profile = SpeakerProfile(speaker_id=speaker_id, bootstrapped=True)
    for wf in words_features[:n]:
        voiced = [v for v in wf.f0_values if v > 0]
        profile.f0_samples.extend(voiced)
        profile.rms_samples.append(wf.rms_db)
        profile.rate_samples.append(wf.rate_wps)
    return profile


# -- Public API ----------------------------------------------------------------

def normalize(
    clip_features,           # ClipFeatures from features.py
    speaker_id: str,
    speaker_profile: Optional[SpeakerProfile] = None,
    cfg: Optional[dict] = None,
    pause_gap_s: float = 0.15,
) -> tuple[NormalizedClip, SpeakerProfile]:
    """
    Normalize *clip_features* to speaker-agnostic units.

    Parameters
    ----------
    clip_features : ClipFeatures
        Output of pipeline.features.extract().
    speaker_id : str
        Unique speaker identifier.
    speaker_profile : SpeakerProfile | None
        Existing profile if speaker has prior clips; None triggers bootstrap.
    cfg : dict | None
        Config dict; loads from config.yaml if None.

    Returns
    -------
    (NormalizedClip, updated_SpeakerProfile)
    """
    if cfg is None:
        cfg = _load_cfg()

    n_bootstrap = cfg.get("normalization", {}).get("bootstrap_words", 10)
    pause_gap_s = cfg.get("grounding", {}).get("pause_gap_threshold_s", pause_gap_s)

    wfs = clip_features.words

    # Build or update speaker profile
    if speaker_profile is None:
        profile = _bootstrap_profile(wfs, n_bootstrap, speaker_id)
    else:
        profile = speaker_profile
        # Refine online with new clip data
        for wf in wfs:
            voiced = [v for v in wf.f0_values if v > 0]
            profile.f0_samples.extend(voiced)
            profile.rms_samples.append(wf.rms_db)
            profile.rate_samples.append(wf.rate_wps)

    f0_med = profile.f0_median
    rms_95 = profile.rms_95pct
    rate_mu = profile.rate_mean
    rate_sig = profile.rate_std or 1.0

    # Compute clip mean MFCC for cosine distance
    all_mfccs = [wf.mfcc_mean for wf in wfs if wf.mfcc_mean]
    clip_mfcc_mean = list(np.mean(all_mfccs, axis=0)) if all_mfccs else [0.0] * 13

    norm_words: list[NormalizedWord] = []
    for wf in wfs:
        pitch_st = f0_to_semitones(wf.f0_hz, f0_med)
        energy_db_rel = wf.rms_db - rms_95
        rate_z = (wf.rate_wps - rate_mu) / rate_sig
        pause_flag = wf.pause_before_s > pause_gap_s
        clarity_mfcc = _cosine_distance(wf.mfcc_mean, clip_mfcc_mean)
        clarity_hnr = wf.hnr_db

        norm_words.append(NormalizedWord(
            word_idx=wf.word_idx,
            word=wf.word,
            start=wf.start,
            end=wf.end,
            pitch_st=pitch_st,
            energy_db_rel=energy_db_rel,
            rate_zscore=rate_z,
            pause_flag=pause_flag,
            pause_duration_s=wf.pause_before_s,
            clarity_mfcc=clarity_mfcc,
            clarity_hnr=clarity_hnr,
            mfcc_mean=wf.mfcc_mean,
            hnr_db=wf.hnr_db,
        ))

    # Normalized clip-level contours for plotting
    voiced_f0 = [v for v in clip_features.f0_contour if v > 0]
    contour_st = [
        f0_to_semitones(v, f0_med) if v > 0 else 0.0
        for v in clip_features.f0_contour
    ]
    rms_contour_rel = [r - rms_95 for r in clip_features.rms_contour]
    rate_z_contour = [
        (r - rate_mu) / rate_sig for r in clip_features.rate_contour
    ]

    return (
        NormalizedClip(
            clip_path=clip_features.clip_path,
            speaker_profile=profile,
            words=norm_words,
            f0_contour_st=contour_st,
            rms_contour_db_rel=rms_contour_rel,
            rate_contour_zscore=rate_z_contour,
        ),
        profile,
    )


def compute_calibration_stats(
    norm_clips_librivox: list[NormalizedClip],
    norm_clips_team: list[NormalizedClip],
) -> dict:
    """
    Compute sigma_group_librivox, sigma_group_team, sigma_pooled_all for config.yaml.
    Called once after calibration-set normalization.
    """

    def _collect(clips: list[NormalizedClip]) -> dict:
        pitch, energy, rate, mfcc, hnr = [], [], [], [], []
        for nc in clips:
            for w in nc.words:
                pitch.append(w.pitch_st)
                energy.append(w.energy_db_rel)
                rate.append(w.rate_zscore)
                mfcc.append(w.clarity_mfcc)
                hnr.append(w.clarity_hnr)
        return {
            "pitch": float(np.std(pitch)) if pitch else 1.0,
            "energy": float(np.std(energy)) if energy else 1.0,
            "rate": float(np.std(rate)) if rate else 1.0,
            "clarity_mfcc": float(np.std(mfcc)) if mfcc else 0.1,
            "clarity_hnr": float(np.std(hnr)) if hnr else 1.0,
        }

    lv = _collect(norm_clips_librivox)
    tm = _collect(norm_clips_team)
    all_clips = norm_clips_librivox + norm_clips_team
    pooled = _collect(all_clips)

    return {
        "sigma_group_librivox": lv,
        "sigma_group_team": tm,
        "sigma_pooled_all": pooled,
    }
