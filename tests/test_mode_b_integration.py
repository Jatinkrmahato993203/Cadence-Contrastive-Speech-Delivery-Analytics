"""
tests/test_mode_b_integration.py
--------------------------------
Integration test for Phase 5 requirement:
Verify Mode B end-to-end on fresh, unseen, non-bundled speeches.
"""

import math
import numpy as np
import pytest
import yaml
from pathlib import Path

from pipeline.grounder import detect_mode, ground
from pipeline.normalizer import normalize
from pipeline.explainer import explain
from pipeline.scorer import score
from pipeline.aligner import align
from pipeline import features as feat_mod

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "config.yaml"

UNSEEN_SPEECHES = [
    {
        "title": "Churchill_We_Shall_Fight",
        "transcript": "We shall fight on the beaches, we shall fight on the landing grounds, we shall fight in the fields and in the streets, we shall fight in the hills; we shall never surrender.",
    },
    {
        "title": "MLK_I_Have_A_Dream",
        "transcript": "I have a dream that one day this nation will rise up and live out the true meaning of its creed: We hold these truths to be self-evident, that all men are created equal.",
    },
    {
        "title": "Jobs_Stanford_Commencement",
        "transcript": "Your time is limited, so don't waste it living someone else's life. Don't be trapped by dogma which is living with the results of other people's thinking.",
    },
]


@pytest.fixture
def cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture
def bundled():
    transcripts, splits = {}, {}
    for i in range(1, 9):
        p = ROOT / "dataset" / "transcripts" / f"text_{i:02d}.txt"
        if p.exists():
            transcripts[i] = p.read_text().strip()
            splits[i] = "calibration" if i not in (6, 8) else "heldout"
    return transcripts, splits


def test_unseen_speeches_trigger_mode_b(bundled):
    transcripts, splits = bundled
    for speech in UNSEEN_SPEECHES:
        mode, matched_id, matched_split = detect_mode(speech["transcript"], transcripts, splits)
        assert mode == "B", f"Unseen speech '{speech['title']}' must trigger Mode B"
        assert matched_id is None, "Mode B must have matched_id=None"
        assert matched_split is None, "Mode B must have matched_split=None"


def test_mode_b_end_to_end_synthetic_audio(cfg, bundled):
    """
    Synthesize mock audio for an unseen speech, align it, extract features,
    self-normalize, ground in Mode B,
    generate causal explanations, and assert rubric scores are valid and non-NaN.
    """
    import soundfile as sf
    import tempfile

    sr = 16000
    dur = 12.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    # Synthetic speech-like signal: 180 Hz carrier modulated with harmonic overtone
    signal = 0.3 * np.sin(2 * np.pi * 180 * t) + 0.15 * np.sin(2 * np.pi * 360 * t)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, signal.astype(np.float32), sr, subtype="PCM_16")
        tmp_path = tmp.name

    try:
        transcript = UNSEEN_SPEECHES[0]["transcript"]
        alignment = align(tmp_path, transcript=transcript, cfg=cfg, use_cache=False)
        from pipeline.aligner import transcript_words
        assert [w.word for w in alignment.words] == transcript_words(transcript)

        # Feature extraction
        clip_feat = feat_mod.extract(tmp_path, alignment)

        # Mode detection
        transcripts, splits = bundled
        mode, matched_id, matched_split = detect_mode(transcript, transcripts, splits)
        assert mode == "B"

        # Self-normalization over the whole clip
        norm_clip, profile = normalize(clip_feat, "unseen_speaker_1", cfg=cfg)
        assert profile.f0_median > 0
        assert profile.rms_ref > -100
        assert len(norm_clip.words) == len(alignment.words)

        # Mode B: deviations from the speaker's own typical delivery
        grounding = ground(norm_clip, mode="B", matched_text_id=None, matched_split=None, cfg=cfg)
        assert grounding.mode == "B"
        assert set(grounding.thresholds) == {"pitch", "energy", "rate", "pause", "clarity"}

        # Causal explanations
        explanations = explain(grounding, norm_clip.words, use_llm=False)
        assert isinstance(explanations, list)

        # Rubric scoring
        rubric = score(grounding, total_duration_s=dur)
        assert 0.0 <= rubric.overall <= 100.0
        assert not math.isnan(rubric.overall)
        for d in rubric.dimensions:
            assert 0.0 <= d.score <= 20.0
            assert not math.isnan(d.score)

    finally:
        Path(tmp_path).unlink(missing_ok=True)
