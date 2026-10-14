"""
tests/test_grounding.py
-----------------------
Unit tests for pipeline/grounder.py: mode detection, pairing, the robust
noise estimate, threshold behaviour, pause gating and region merging.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.grounder import (  # noqa: E402
    FlawRegion, _jaccard, _merge_regions, _tokenize, detect_mode, ground, pair_words, robust_noise,
)
from pipeline.normalizer import NormalizedClip, NormalizedWord, SpeakerProfile  # noqa: E402

CFG = {
    "grounding": {
        "k_noise": 3.0, "min_consecutive_deviant": 2, "merge_gap_ms": 300, "smoothing_words": 3,
        "pause_gap_threshold_s": 0.15, "pause_ratio": 2.0,
        "floors": {"pitch": 0.45, "energy": 4.0, "rate": 0.2, "pause": 0.25, "clarity": 1.0},
        "floors_mode_b": {"pitch": 1.0, "energy": 9.0, "rate": 0.6, "pause": 1.0, "clarity": 3.0},
    }
}


def _words(n=40, pitch_var=3.0, energy=-6.0, dur=0.3, pause=0.05, clarity=0.0, names=None):
    out, t = [], 0.0
    for i in range(n):
        t += pause
        out.append(NormalizedWord(i, names[i] if names else f"w{i}", t, t + dur, pitch_var_st=pitch_var,
                                  energy_db_rel=energy, rate_wps=1 / dur, pause_duration_s=pause,
                                  clarity=clarity, duration_s=dur))
        t += dur
    return out


def _clip(words):
    return NormalizedClip("x.wav", SpeakerProfile("s"), words, [], [], [w.rate_zscore for w in words],
                          duration_s=words[-1].end if words else 0.0)


# -- Mode detection ------------------------------------------------------------------

def test_detect_mode_exact_match():
    bundled = {1: "Four score and seven years ago", 2: "We the people of the United States"}
    mode, tid, split = detect_mode("Four score and seven years ago", bundled, {1: "calibration", 2: "calibration"})
    assert (mode, tid, split) == ("A", 1, "calibration")


def test_detect_mode_near_match():
    bundled = {1: "Four score and seven years ago our fathers brought forth"}
    mode, tid, _ = detect_mode("Four score and seven years ago our fathers", bundled, {1: "calibration"},
                               threshold=0.70)
    assert mode == "A" and tid == 1


def test_detect_mode_no_match():
    bundled = {1: "Four score and seven years ago our fathers brought forth"}
    mode, tid, _ = detect_mode("The quick brown fox jumps over the lazy dog", bundled, {1: "calibration"})
    assert mode == "B" and tid is None


def test_detect_mode_heldout_split():
    bundled = {6: "Well children where there is so much racket"}
    mode, _, split = detect_mode("Well children where there is so much racket there must be something",
                                 bundled, {6: "heldout"}, threshold=0.70)
    assert mode == "A" and split == "heldout"


def test_tokenize_and_jaccard():
    assert _tokenize("Ain't I a woman?") == ["ain't", "i", "a", "woman"]
    assert _jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)


def test_pair_words_handles_insertions():
    b = _words(5, names=["a", "b", "c", "d", "e"])
    u = _words(6, names=["a", "b", "x", "c", "d", "e"])
    assert pair_words(u, b) == [0, 1, None, 2, 3, 4]


# -- Robust noise ------------------------------------------------------------------------

def test_robust_noise_is_zero_for_identical_readings():
    assert robust_noise(np.zeros(50)) == 0.0


def test_robust_noise_resists_flaws_covering_most_words():
    rng = np.random.default_rng(0)
    clean = rng.normal(0, 0.05, 40)
    flawed = np.r_[clean, np.full(60, -0.6)]        # 60% of words carry the flaw
    assert robust_noise(flawed) < 0.15
    assert robust_noise(rng.normal(0, 1.0, 200)) == pytest.approx(1.0, rel=0.35)


# -- Grounding behaviour -------------------------------------------------------------

def test_identical_pair_flags_nothing():
    b = _words()
    res = ground(_clip(_words()), "A", 1, "calibration", baseline_norm_clip=_clip(b), cfg=CFG)
    assert res.flaw_regions == []


def test_local_volume_drop_is_grounded_to_its_words():
    b, u = _words(), _words()
    for w in u[10:18]:
        w.energy_db_rel -= 10.0
    res = ground(_clip(u), "A", 1, "calibration", baseline_norm_clip=_clip(b), cfg=CFG)
    regs = [r for r in res.flaw_regions if r.flaw_type == "energy"]
    assert len(regs) == 1 and regs[0].direction == "quiet"
    assert regs[0].start_s == pytest.approx(u[10].start) and regs[0].end_s == pytest.approx(u[17].end)
    assert [r for r in res.flaw_regions if r.flaw_type != "energy"] == []


def test_cross_speaker_variation_raises_threshold():
    """Natural word-to-word differences (std 3 dB) must not be reported; a 20 dB drop still is."""
    rng = np.random.default_rng(1)
    b, u = _words(80), _words(80)
    for w in u:
        w.energy_db_rel += rng.normal(0, 3.0)
    quiet = ground(_clip(u), "A", 1, "calibration", baseline_norm_clip=_clip(b), cfg=CFG)
    assert quiet.thresholds["energy"] > CFG["grounding"]["floors"]["energy"]
    for w in u[30:40]:
        w.energy_db_rel -= 20.0
    loud = ground(_clip(u), "A", 1, "calibration", baseline_norm_clip=_clip(b), cfg=CFG)
    hits = [r for r in loud.flaw_regions if r.flaw_type == "energy"]
    assert any(r.start_s <= u[32].start and r.end_s >= u[37].end for r in hits)


def test_monotone_and_slow_passages_detected_with_direction():
    b, u = _words(), _words()
    for w in u[5:20]:
        w.pitch_var_st = 1.0                 # 3.0 -> 1.0 semitones: flatter
    for w in u[25:35]:
        w.duration_s = 0.45                  # 1.5x longer: slower
    res = ground(_clip(u), "A", 1, "calibration", baseline_norm_clip=_clip(b), cfg=CFG)
    dirs = {(r.flaw_type, r.direction) for r in res.flaw_regions}
    assert ("pitch", "flat") in dirs and ("rate", "slow") in dirs


def test_pause_gate_ignores_pause_stretched_by_slow_pacing():
    b, u = _words(), _words()
    for w in u[6:15]:
        w.duration_s *= 1.44                                         # slowed passage ...
    b[10].pause_duration_s, u[10].pause_duration_s = 0.8, 1.15      # ... stretches its pause 1.44x: pacing only
    b[25].pause_duration_s, u[25].pause_duration_s = 0.05, 0.75     # inserted silence at normal tempo
    res = ground(_clip(u), "A", 1, "calibration", baseline_norm_clip=_clip(b), cfg=CFG)
    pauses = [r for r in res.flaw_regions if r.flaw_type == "pause"]
    assert [p.word_indices for p in pauses] == [[25]]
    assert pauses[0].start_s == pytest.approx(u[24].end) and pauses[0].end_s == pytest.approx(u[25].start)
    assert any(r.flaw_type == "rate" and r.direction == "slow" for r in res.flaw_regions)


def test_pause_lengthened_at_unchanged_tempo_is_flagged():
    """Silence added next to a natural pause (1.05 -> 1.49 s) is a pause flaw if tempo is unchanged."""
    b, u = _words(), _words()
    b[15].pause_duration_s, u[15].pause_duration_s = 1.05, 1.49
    res = ground(_clip(u), "A", 1, "calibration", baseline_norm_clip=_clip(b), cfg=CFG)
    assert [r.word_indices for r in res.flaw_regions if r.flaw_type == "pause"] == [[15]]


def test_mode_b_flags_only_long_pauses_and_local_anomalies():
    u = _words(60)
    u[30].pause_duration_s = 1.6
    for w in u[40:48]:
        w.energy_db_rel -= 15.0
    res = ground(_clip(u), "B", None, None, cfg=CFG)
    kinds = {r.flaw_type for r in res.flaw_regions}
    assert kinds == {"pause", "energy"}


# -- Merging ---------------------------------------------------------------------------

def test_merge_regions_gap_below_threshold():
    r1 = FlawRegion("rate", 1.0, 3.0, [0, 1, 2], 0.5, "fast")
    r2 = FlawRegion("rate", 3.1, 5.0, [3, 4, 5], 0.6, "fast")
    merged = _merge_regions([r1, r2], gap_s=0.2)
    assert len(merged) == 1 and merged[0].start_s == 1.0 and merged[0].end_s == 5.0


def test_merge_regions_gap_above_threshold():
    r1 = FlawRegion("rate", 1.0, 3.0, [0, 1, 2], 0.5, "fast")
    r2 = FlawRegion("rate", 3.5, 5.0, [3, 4, 5], 0.6, "fast")
    assert len(_merge_regions([r1, r2], gap_s=0.2)) == 2


def test_merge_regions_never_merges_types_or_directions():
    a = FlawRegion("rate", 1.0, 3.0, [0, 1], 0.5, "fast")
    assert len(_merge_regions([a, FlawRegion("pitch", 3.1, 5.0, [2, 3], 0.4, "flat")], 0.2)) == 2
    assert len(_merge_regions([a, FlawRegion("rate", 3.1, 5.0, [2, 3], 0.4, "slow")], 0.2)) == 2


def test_merge_regions_empty():
    assert _merge_regions([], gap_s=0.2) == []


def test_edited_copy_uses_floors_and_other_recording_adapts():
    """Edited copy (unedited words identical) -> floors; a different voice -> thresholds widen."""
    rng = np.random.default_rng(3)
    b, u_copy, u_other = _words(60), _words(60), _words(60)
    for wb, wc, wo in zip(b, u_copy, u_other):
        timbre = list(rng.normal(0, 10, 13))
        wb.mfcc_mean, wc.mfcc_mean = timbre, list(timbre)
        wo.mfcc_mean = list(np.asarray(timbre) + rng.normal(0, 15, 13))   # another voice
        wo.energy_db_rel += rng.normal(0, 3.0)                             # natural level variation
    copy = ground(_clip(u_copy), "A", 1, "calibration", baseline_norm_clip=_clip(b), cfg=CFG)
    other = ground(_clip(u_other), "A", 1, "calibration", baseline_norm_clip=_clip(b), cfg=CFG)
    assert copy.same_recording and not other.same_recording
    assert copy.thresholds["energy"] == CFG["grounding"]["floors"]["energy"]
    assert other.thresholds["energy"] > CFG["grounding"]["floors"]["energy"]
