"""
tests/test_normalization.py
---------------------------
Speaker-agnostic normalization (pipeline/normalizer.py).

Key property: two speakers delivering the SAME prosodic pattern — one an
octave higher and 12 dB louder — produce identical normalized pitch, pitch
variability and relative energy. This is what lets a participant be compared
with a baseline read by a different voice.
"""

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.normalizer import articulation_tempo, f0_to_semitones, normalize  # noqa: E402

CFG = {"normalization": {"clarity_feature": "hnr_db"}, "grounding": {"pause_gap_threshold_s": 0.15}}


@dataclass
class FakeWord:
    word_idx: int
    word: str
    start: float
    end: float
    f0_hz: float
    f0_values: list
    rms_db: float
    duration_s: float
    pause_before_s: float = 0.0
    mfcc_mean: list = field(default_factory=lambda: [1.0] * 13)
    hnr_db: float = 15.0


@dataclass
class FakeClip:
    clip_path: str
    duration_s: float
    words: list
    f0_contour: list
    rms_contour: list


def _speaker(f0_scale: float, gain_db: float, n: int = 40) -> FakeClip:
    """Same intonation and loudness pattern, scaled to a different voice / microphone gain."""
    pattern_st = 3.0 * np.sin(np.arange(n) / 3.0)
    level = -10.0 + 4.0 * np.cos(np.arange(n) / 5.0)
    words, f0_all = [], []
    for i in range(n):
        base = 120.0 * f0_scale * 2 ** (pattern_st[i] / 12)
        frames = [base * 2 ** (d / 12) for d in (-0.5, 0.0, 0.5)]
        f0_all += frames
        words.append(FakeWord(i, f"w{i}", i * 0.35, i * 0.35 + 0.3, float(np.median(frames)), frames,
                              float(level[i] + gain_db), 0.3, 0.05))
    return FakeClip("x.wav", n * 0.35, words, f0_all, [w.rms_db for w in words])


def test_f0_to_semitones():
    assert f0_to_semitones(200, 200) == 0.0
    assert f0_to_semitones(400, 200) == pytest.approx(12.0)
    assert f0_to_semitones(0, 200) == 0.0


def test_cross_speaker_invariance():
    low, _ = normalize(_speaker(1.0, 0.0), "low", cfg=CFG)
    high, _ = normalize(_speaker(2.0, 12.0), "high", cfg=CFG)     # octave up, 12 dB louder
    for a, b in zip(low.words, high.words):
        assert a.pitch_st == pytest.approx(b.pitch_st, abs=1e-6)
        assert a.pitch_var_st == pytest.approx(b.pitch_var_st, abs=1e-6)
        assert a.energy_db_rel == pytest.approx(b.energy_db_rel, abs=1e-6)


def test_statistics_use_whole_clip_not_first_words():
    """A flaw at the very start must not become the speaker's reference."""
    clip = _speaker(1.0, 0.0)
    for w in clip.words[:10]:
        w.rms_db -= 20.0
    norm, profile = normalize(clip, "s", cfg=CFG)
    assert profile.rms_ref == pytest.approx(np.percentile([w.rms_db for w in clip.words], 90))
    assert np.mean([w.energy_db_rel for w in norm.words[:10]]) < np.mean([w.energy_db_rel for w in norm.words[10:]]) - 15


def test_articulation_tempo_excludes_pauses():
    assert articulation_tempo([0.25] * 5)[2] == pytest.approx(4.0)
    assert articulation_tempo([0.5] * 5)[2] == pytest.approx(2.0)


def test_normalization_deterministic():
    a, _ = normalize(_speaker(1.3, 3.0), "s", cfg=CFG)
    b, _ = normalize(_speaker(1.3, 3.0), "s", cfg=CFG)
    assert [w.__dict__ for w in a.words] == [w.__dict__ for w in b.words]
    assert all(math.isfinite(w.pitch_var_st) for w in a.words)
