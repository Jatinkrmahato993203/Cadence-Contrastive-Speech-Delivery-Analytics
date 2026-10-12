"""
tests/test_grounding.py
-----------------------
Unit tests for pipeline/grounder.py.

Tests mode detection, delta computation, threshold flagging,
region building, gap merging, and the Mode B population consistency rule.
"""

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.grounder import (
    detect_mode,
    ground,
    _tokenize,
    _jaccard,
    _merge_regions,
    FlawRegion,
    GroundingResult,
)


# -- Mode detection -------------------------------------------------------------

def test_detect_mode_exact_match():
    bundled = {1: "Four score and seven years ago", 2: "We the people of the United States"}
    splits = {1: "calibration", 2: "calibration"}
    mode, tid, split = detect_mode("Four score and seven years ago", bundled, splits)
    assert mode == "A"
    assert tid == 1
    assert split == "calibration"


def test_detect_mode_near_match():
    bundled = {1: "Four score and seven years ago our fathers brought forth"}
    splits = {1: "calibration"}
    # Drop a few words — should still hit >= 0.90 Jaccard if close
    mode, tid, _ = detect_mode(
        "Four score and seven years ago our fathers", bundled, splits, threshold=0.70
    )
    assert mode == "A"
    assert tid == 1


def test_detect_mode_no_match():
    bundled = {1: "Four score and seven years ago our fathers brought forth"}
    splits = {1: "calibration"}
    mode, tid, _ = detect_mode("The quick brown fox jumps over the lazy dog", bundled, splits)
    assert mode == "B"
    assert tid is None


def test_detect_mode_heldout_split():
    bundled = {6: "Well children where there is so much racket"}
    splits = {6: "heldout"}
    mode, tid, split = detect_mode(
        "Well children where there is so much racket there must be something", bundled, splits, threshold=0.70
    )
    assert mode == "A"
    assert split == "heldout"


# -- Gap merging ----------------------------------------------------------------

def test_merge_regions_gap_below_threshold():
    """Two regions with gap < 200ms should be merged."""
    r1 = FlawRegion("rate", 1.0, 3.0, [0,1,2], 0.5, "fast")
    r2 = FlawRegion("rate", 3.1, 5.0, [3,4,5], 0.6, "fast")  # gap = 100ms
    merged = _merge_regions([r1, r2], gap_s=0.2)
    assert len(merged) == 1
    assert merged[0].start_s == pytest.approx(1.0)
    assert merged[0].end_s == pytest.approx(5.0)


def test_merge_regions_gap_above_threshold():
    """Two regions with gap > 200ms should NOT be merged."""
    r1 = FlawRegion("rate", 1.0, 3.0, [0,1,2], 0.5, "fast")
    r2 = FlawRegion("rate", 3.5, 5.0, [3,4,5], 0.6, "fast")  # gap = 500ms
    merged = _merge_regions([r1, r2], gap_s=0.2)
    assert len(merged) == 2


def test_merge_regions_different_flaw_types():
    """Regions of different flaw types should never be merged."""
    r1 = FlawRegion("rate", 1.0, 3.0, [0,1], 0.5, "fast")
    r2 = FlawRegion("pitch", 3.1, 5.0, [2,3], 0.4, "low")
    merged = _merge_regions([r1, r2], gap_s=0.2)
    assert len(merged) == 2


def test_merge_regions_empty():
    assert _merge_regions([], gap_s=0.2) == []


# -- Mode B threshold consistency ----------------------------------------------

def test_mode_b_uses_pooled_sigma():
    """
    Mode B MUST use sigma_pooled_all (not a subgroup sigma) as the threshold.
    This is the critical Mode B population consistency rule (§3.1).
    """
    import yaml
    cfg = yaml.safe_load("""
grounding:
  deviance_multiplier: 1.5
  min_consecutive_deviant: 2
  merge_gap_ms: 200
  pause_gap_threshold_s: 0.15
sigma_pooled_all:
  pitch: 2.5
  energy: 3.0
  rate: 1.0
  clarity_mfcc: 0.15
  clarity_hnr: 2.5
sigma_group_librivox:
  pitch: 1.0
  energy: 1.5
  rate: 0.5
  clarity_mfcc: 0.05
  clarity_hnr: 1.0
""")

    # Build minimal NormalizedClip-like object
    from dataclasses import dataclass, field
    from typing import List

    @dataclass
    class FakeNormWord:
        word_idx: int
        word: str
        start: float
        end: float
        pitch_st: float
        energy_db_rel: float
        rate_zscore: float
        pause_flag: bool
        pause_duration_s: float
        clarity_mfcc: float
        clarity_hnr: float
        mfcc_mean: List[float] = field(default_factory=lambda: [0.0]*13)
        hnr_db: float = 15.0

    @dataclass
    class FakeNormClip:
        clip_path: str
        speaker_profile: object
        words: List
        f0_contour_st: List[float] = field(default_factory=list)
        rms_contour_db_rel: List[float] = field(default_factory=list)
        rate_contour_zscore: List[float] = field(default_factory=list)

    # All words within pooled_all threshold but OUTSIDE subgroup threshold
    # pitch_st = 2.0, pooled sigma=2.5 -> |delta|/sigma = 2.0/2.5 = 0.8 < 1.5 (no flag)
    # but subgroup sigma=1.0 -> |delta|/sigma = 2.0/1.0 = 2.0 > 1.5 (would flag if wrong group used)
    words = [
        FakeNormWord(i, f"w{i}", i*0.5, (i+0.8)*0.5,
                     pitch_st=2.0, energy_db_rel=0.0, rate_zscore=0.0,
                     pause_flag=False, pause_duration_s=0.0,
                     clarity_mfcc=0.0, clarity_hnr=15.0)
        for i in range(5)
    ]
    clip = FakeNormClip("dummy.wav", None, words)

    result = ground(
        clip, mode="B",
        matched_text_id=None,
        matched_split=None,
        cfg=cfg,
        pooled_means={"pitch": 0.0, "energy": 0.0, "rate": 0.0,
                      "clarity_mfcc": 0.0, "clarity_hnr": 0.0},
    )

    # With pooled sigma=2.5 and delta=2.0: 2.0 < 1.5*2.5=3.75 -> should NOT flag
    pitch_flags = result.word_deviant.get("pitch", [])
    assert not any(pitch_flags), (
        "Mode B should use sigma_pooled_all (2.5), not subgroup sigma (1.0). "
        "delta=2.0 should NOT exceed threshold=1.5*2.5=3.75."
    )
