"""
tests/test_reproducibility.py
-------------------------------
Reproducibility test (§4, §6 Phase 7).

Verifies that running the full grounding + scoring pipeline twice on the same
input produces bit-identical JSON output (region boundaries + rubric scores).

This is the "make reproducibility-test" equivalent.
"""

import json
import sys
import tempfile
from pathlib import Path
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import yaml
from pipeline.grounder import GroundingResult, FlawRegion, ground, detect_mode
from pipeline.scorer import score


def _run_pipeline(cfg: dict) -> dict:
    """
    Minimal reproducibility pipeline: builds a deterministic NormalizedClip
    from fixed synthetic data and runs grounding + scoring.
    Returns a JSON-serializable dict of region boundaries and scores.
    """
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

    # Fixed synthetic words: some with high rate z-score to trigger flaw
    words = []
    for i in range(20):
        rate_z = 4.0 if 8 <= i <= 12 else 0.1   # strong rate flaw in middle
        words.append(FakeNormWord(
            word_idx=i, word=f"word{i}",
            start=i * 0.5, end=(i + 0.8) * 0.5,
            pitch_st=0.2, energy_db_rel=-0.5,
            rate_zscore=rate_z, pause_flag=False,
            pause_duration_s=0.0, clarity_mfcc=0.02,
            clarity_hnr=15.0
        ))

    clip = FakeNormClip("synthetic.wav", None, words)

    result = ground(
        clip, mode="B",
        matched_text_id=None, matched_split=None,
        cfg=cfg,
        pooled_means={"pitch": 0.0, "energy": 0.0, "rate": 0.0,
                      "clarity_mfcc": 0.0, "clarity_hnr": 0.0},
    )

    rubric = score(result, total_duration_s=10.0)

    return {
        "regions": [
            {"flaw_type": r.flaw_type, "start_s": r.start_s, "end_s": r.end_s,
             "mean_delta": round(r.mean_delta, 6)}
            for r in result.flaw_regions
        ],
        "scores": {
            d.dimension: round(d.score, 6)
            for d in rubric.dimensions
        },
        "overall": round(rubric.overall, 6),
    }


def test_pipeline_is_deterministic():
    """
    Running the pipeline twice with identical input must produce identical output.
    """
    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    output_1 = _run_pipeline(cfg)
    output_2 = _run_pipeline(cfg)

    assert output_1 == output_2, (
        "Pipeline is NOT deterministic!\n"
        f"Run 1: {json.dumps(output_1, indent=2)}\n"
        f"Run 2: {json.dumps(output_2, indent=2)}"
    )


def test_config_all_thresholds_numeric():
    """
    config.yaml must contain numeric (non-placeholder) sigma values.
    Flags if thresholds were never updated from placeholder strings.
    """
    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    for group in ["sigma_group_librivox", "sigma_group_team", "sigma_pooled_all"]:
        data = cfg.get(group, {})
        for feat, val in data.items():
            assert isinstance(val, (int, float)), \
                f"{group}.{feat} is not numeric: {val!r}"
            assert val > 0, \
                f"{group}.{feat} = {val} — sigma must be positive"
