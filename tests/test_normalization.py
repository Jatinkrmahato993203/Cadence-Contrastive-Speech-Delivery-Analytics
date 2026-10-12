"""
tests/test_normalization.py
----------------------------
Unit tests for pipeline/normalizer.py (§2.3).

Key test: cross-speaker normalization test
  Verifies that a high-pitched speaker's UNFLAWED delivery does NOT get flagged
  as a pitch-deviation outlier relative to a low-pitched speaker's baseline
  once both are expressed in semitone / relative-dB space.
  This makes "speaker-agnostic" a tested property, not just a claim.
"""

import math
import sys
from pathlib import Path
import pytest
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.normalizer import (
    SpeakerProfile,
    NormalizedWord,
    f0_to_semitones,
    normalize,
)


def _make_profile(f0_samples, rms_samples, rate_samples, speaker_id="test"):
    p = SpeakerProfile(speaker_id=speaker_id)
    p.f0_samples = f0_samples
    p.rms_samples = rms_samples
    p.rate_samples = rate_samples
    return p


def _make_clip_features(
    f0_values: list[float],
    rms_values: list[float],
    rate_values: list[float],
    audio_path: str = "dummy.wav",
    duration: float = 60.0,
):
    """Build a minimal ClipFeatures-like object without real audio."""
    from dataclasses import dataclass, field

    @dataclass
    class FakeWordFeature:
        word_idx: int
        word: str
        start: float
        end: float
        f0_hz: float
        f0_values: list
        rms_db: float
        lufs: float = -23.0
        mfcc_mean: list = field(default_factory=lambda: [0.0]*13)
        spectral_centroid: float = 0.0
        spectral_flatness: float = 0.0
        hnr_db: float = 15.0
        jitter: float = 0.01
        shimmer: float = 0.01
        rate_wps: float = 3.0
        pause_before_s: float = 0.0

    @dataclass
    class FakeClipFeatures:
        clip_path: str
        duration_s: float
        words: list
        f0_contour: list
        rms_contour: list
        rate_contour: list
        pause_positions: list = field(default_factory=list)

    step = duration / max(len(f0_values), 1)
    words = [
        FakeWordFeature(
            word_idx=i, word=f"word{i}",
            start=i * step, end=(i + 0.8) * step,
            f0_hz=f0_values[i], f0_values=[f0_values[i]],
            rms_db=rms_values[i], rate_wps=rate_values[i],
        )
        for i in range(len(f0_values))
    ]
    return FakeClipFeatures(
        clip_path=audio_path,
        duration_s=duration,
        words=words,
        f0_contour=f0_values,
        rms_contour=rms_values,
        rate_contour=rate_values,
    )


# -- semitones formula ---------------------------------------------------------

def test_f0_to_semitones_same():
    """Same F0 and median -> 0 semitones."""
    assert f0_to_semitones(200.0, 200.0) == pytest.approx(0.0)


def test_f0_to_semitones_octave_up():
    """Doubling F0 -> +12 semitones."""
    assert f0_to_semitones(400.0, 200.0) == pytest.approx(12.0)


def test_f0_to_semitones_zero_safe():
    """Unvoiced frame (f0=0) -> 0 semitones, no exception."""
    assert f0_to_semitones(0.0, 200.0) == 0.0


# -- Cross-speaker normalization -----------------------------------------------

def test_cross_speaker_clean_clips_not_flagged():
    """
    CRITICAL TEST (§2.3):
    A high-pitched speaker (median F0=250Hz, natural variance ±3 ST) and a
    low-pitched speaker (median F0=100Hz, natural variance ±3 ST) should both
    express their CLEAN words near 0 semitones from their own baseline, not
    from each other's baseline.

    Confirms that speaker-agnostic normalization actually removes between-speaker
    differences in raw Hz, so neither speaker's clean clip is flagged as deviant.
    """
    # High-pitched speaker: F0 values around 250 Hz (+/- small variance)
    n_words = 20
    f0_high = [250.0 + np.random.uniform(-5, 5) for _ in range(n_words)]
    rms_high = [-20.0 + np.random.uniform(-1, 1) for _ in range(n_words)]
    rate_high = [3.0 + np.random.uniform(-0.2, 0.2) for _ in range(n_words)]

    # Low-pitched speaker: F0 values around 100 Hz (+/- small variance)
    f0_low = [100.0 + np.random.uniform(-3, 3) for _ in range(n_words)]
    rms_low = [-25.0 + np.random.uniform(-1, 1) for _ in range(n_words)]
    rate_low = [2.5 + np.random.uniform(-0.2, 0.2) for _ in range(n_words)]

    clip_high = _make_clip_features(f0_high, rms_high, rate_high)
    clip_low = _make_clip_features(f0_low, rms_low, rate_low)

    import yaml
    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    norm_high, _ = normalize(clip_high, "speaker_high", cfg=cfg)
    norm_low, _  = normalize(clip_low,  "speaker_low",  cfg=cfg)

    threshold = cfg.get("grounding", {}).get("deviance_multiplier", 1.5)
    sigma_pitch = cfg.get("sigma_pooled_all", {}).get("pitch", 2.5)

    # All pitch_st values for both speakers should be near 0 (their own baseline)
    for w in norm_high.words:
        assert abs(w.pitch_st) < threshold * sigma_pitch, (
            f"High-pitched speaker word '{w.word}' has pitch_st={w.pitch_st:.2f} "
            f"exceeding threshold={threshold * sigma_pitch:.2f} — cross-speaker leak!"
        )

    for w in norm_low.words:
        assert abs(w.pitch_st) < threshold * sigma_pitch, (
            f"Low-pitched speaker word '{w.word}' has pitch_st={w.pitch_st:.2f} "
            f"exceeding threshold={threshold * sigma_pitch:.2f} — cross-speaker leak!"
        )


def test_bootstrap_from_first_n_words():
    """New speaker profile should be bootstrapped from first 10 words."""
    n_words = 30
    f0_vals = [150.0] * n_words
    rms_vals = [-20.0] * n_words
    rate_vals = [3.0] * n_words

    clip = _make_clip_features(f0_vals, rms_vals, rate_vals)

    import yaml
    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    norm_clip, profile = normalize(clip, "new_speaker", speaker_profile=None, cfg=cfg)

    assert profile.bootstrapped, "New speaker profile should be marked as bootstrapped"
    # Median should be close to the actual F0
    assert abs(profile.f0_median - 150.0) < 10.0, \
        f"Bootstrap median {profile.f0_median:.1f} too far from true 150.0 Hz"


def test_normalization_deterministic():
    """Same input -> same output (reproducibility)."""
    n_words = 15
    f0_vals = [180.0 + i for i in range(n_words)]
    rms_vals = [-22.0] * n_words
    rate_vals = [3.0] * n_words

    clip = _make_clip_features(f0_vals, rms_vals, rate_vals)

    import yaml
    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    norm1, _ = normalize(clip, "spk_det", cfg=cfg)
    norm2, _ = normalize(clip, "spk_det", cfg=cfg)

    for w1, w2 in zip(norm1.words, norm2.words):
        assert w1.pitch_st == pytest.approx(w2.pitch_st, abs=1e-9)
        assert w1.energy_db_rel == pytest.approx(w2.energy_db_rel, abs=1e-9)
