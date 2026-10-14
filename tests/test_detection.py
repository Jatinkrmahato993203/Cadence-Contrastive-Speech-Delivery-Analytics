"""
tests/test_detection.py
-----------------------
End-to-end detection on REAL dataset audio (not hand-built feature values).

These are the regression tests for the failure modes found in review:
  * an identical copy of the baseline, uploaded under a different file name,
    must produce zero flaw regions and a perfect score;
  * forced alignment must return every transcript word, in order;
  * injected flaws on held-out texts must be found where they were injected;
  * ground-truth pause labels must coincide with the silence in the audio.

They need openai-whisper (alignments are cached in dataset/alignments/).
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("whisper")

from pipeline.aligner import align, transcript_words  # noqa: E402
from pipeline.analysis import analyze, load_bundled, load_config  # noqa: E402
from pipeline.evaluation import clip_metrics  # noqa: E402

AUDIO = ROOT / "dataset" / "audio"
LABELS = ROOT / "dataset" / "labels"
BUNDLED, _ = load_bundled()
CFG = load_config()


def _label(clip_id: str) -> dict:
    return json.loads((LABELS / f"{clip_id}.json").read_text())


def test_real_orator_original_is_clean(tmp_path):
    """An untouched presidential recording, uploaded under another name, is matched and not flagged."""
    copy = tmp_path / "speech.wav"
    shutil.copy(AUDIO / "text_10_good.wav", copy)
    res = analyze(copy, BUNDLED[10], cfg=CFG)
    assert res.mode == "A" and res.matched_id == 10
    assert res.grounding.flaw_regions == [] and res.rubric.overall == pytest.approx(100.0)


def test_identical_copy_scores_perfectly(tmp_path):
    copy = tmp_path / "participant_upload.wav"
    shutil.copy(AUDIO / "text_06_good.wav", copy)
    res = analyze(copy, BUNDLED[6], cfg=CFG)
    assert res.mode == "A" and res.matched_id == 6
    assert res.grounding.flaw_regions == []
    assert res.rubric.overall == pytest.approx(100.0)


def test_forced_alignment_returns_every_transcript_word():
    res = align(str(AUDIO / "text_08_volume_egregious.wav"), BUNDLED[8], cfg=CFG)
    assert [w.word for w in res.words] == transcript_words(BUNDLED[8])
    starts = [w.start for w in res.words]
    assert starts == sorted(starts)


@pytest.mark.parametrize("clip_id,feature,min_iou", [
    ("text_09_volume_mod", "energy", 0.8),     # real orator baseline (JFK)
    ("text_10_pitch_mod", "pitch", 0.6),       # real orator baseline (Reagan)
    ("text_06_volume_mod", "energy", 0.8),
    ("text_08_pitch_mod", "pitch", 0.6),
    ("text_06_pause_egregious", "pause", 0.6),
    ("text_08_pacing_mod", "rate", 0.5),
])
def test_heldout_flaw_is_grounded(clip_id, feature, min_iou):
    """
    IoU over SPEECH time: ground-truth regions are clipped to the span of aligned
    words, since a gain change applied to applause (JFK, text 9) is not a delivery
    flaw a word-level analysis can — or should — report.
    """
    tid = int(clip_id[5:7])
    res = analyze(AUDIO / f"{clip_id}.wav", BUNDLED[tid], cfg=CFG)
    words = res.upload.norm.words
    label = _label(clip_id)
    speech = [(w.start, w.end) for w in words]
    clipped = []
    for r in label["flaw_regions"]:
        for a, b in speech:
            lo, hi = max(a, r["start_s"]), min(b, r["end_s"])
            if hi > lo:
                clipped.append({**r, "start_s": lo, "end_s": hi})
    det = []
    for reg in res.grounding.flaw_regions:
        for a, b in speech:
            lo, hi = max(a, reg.start_s), min(b, reg.end_s)
            if hi > lo:
                det.append({"start_s": lo, "end_s": hi, "flaw_type": reg.flaw_type})
    if feature == "pause":   # pauses ARE the non-speech time: use the plain metric
        m = clip_metrics(res.grounding, label, res.duration_s)
        iou = m["iou_by_feature"][feature]
    else:
        from pipeline.evaluation import temporal_iou, gt_by_feature
        iou = temporal_iou([d for d in det if d["flaw_type"] == feature],
                           gt_by_feature({"flaw_regions": clipped})[feature], res.duration_s)
    assert iou >= min_iou, (clip_id, feature, round(iou, 3))


def test_every_egregious_pause_is_found():
    clip_id = "text_08_pause_egregious"
    res = analyze(AUDIO / f"{clip_id}.wav", BUNDLED[8], cfg=CFG)
    pauses = [r for r in res.grounding.flaw_regions if r.flaw_type == "pause"]
    for gt in _label(clip_id)["flaw_regions"]:
        assert any(min(p.end_s, gt["end_s"]) - max(p.start_s, gt["start_s"]) > 0.5 * (gt["end_s"] - gt["start_s"])
                   for p in pauses), gt


def test_scores_fall_with_severity():
    scores = [analyze(AUDIO / f"text_06_volume_{t}.wav", BUNDLED[6], cfg=CFG).rubric.overall
              for t in ("mild", "mod", "egregious")]
    assert scores[0] >= scores[1] >= scores[2]


def test_pause_labels_match_silence_in_audio():
    for lp in sorted(LABELS.glob("text_*_pause_*.json")):
        if lp.stem.endswith("_features"):
            continue
        label = json.loads(lp.read_text())
        y, sr = sf.read(ROOT / label["audio_path"])
        for r in label["flaw_regions"]:
            seg = y[int(r["start_s"] * sr) + 5: int(r["end_s"] * sr) - 5]
            assert np.mean(np.abs(seg) < 1e-3) > 0.98, (label["clip_id"], r)
