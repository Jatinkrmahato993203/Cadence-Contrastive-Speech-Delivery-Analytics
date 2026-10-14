"""
tests/test_scoring.py
---------------------
Unit tests for pipeline/scorer.py.

Critical test: test_score_zero_regions
  Asserts that a dimension with zero flagged regions scores exactly 20.0
  and is NEVER NaN (covers the explicit base case in scorer.py that prevents
  IEEE-754 NaN propagation via mean([]) * 0 = NaN * 0 = NaN).
"""

import math
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.scorer import score, DimensionScore, RubricScore
from pipeline.grounder import GroundingResult


def _make_empty_grounding():
    return GroundingResult(
        mode="A",
        matched_text_id=1,
        matched_split="calibration",
        flaw_regions=[],
        word_deviant={},
        uploaded_contours={},
        baseline_contours={},
        normative_corridor={},
    )


def _make_grounding_with_regions(regions):
    return GroundingResult(
        mode="A",
        matched_text_id=1,
        matched_split="calibration",
        flaw_regions=regions,
        word_deviant={},
        uploaded_contours={},
        baseline_contours={},
        normative_corridor={},
    )


# -- Critical: zero-regions NaN safety -----------------------------------------

def test_score_zero_regions():
    """
    A clip with no detected flaw regions must score 20.0 per dimension
    and 100.0 overall. Score must NEVER be NaN.
    """
    grounding = _make_empty_grounding()
    result = score(grounding, total_duration_s=60.0)

    assert not math.isnan(result.overall), "Overall score must not be NaN"
    assert result.overall == pytest.approx(100.0, abs=1e-6), \
        f"Expected 100.0, got {result.overall}"

    for dim in result.dimensions:
        assert not math.isnan(dim.score), f"{dim.dimension} score must not be NaN"
        assert dim.score == pytest.approx(20.0, abs=1e-6), \
            f"Expected {dim.dimension}=20.0, got {dim.score}"
        assert dim.severity_weight == 0.0, \
            f"severity_weight should be 0.0 for zero regions, got {dim.severity_weight}"


def test_score_full_flaw():
    """A fully-flawed clip (100% duration flagged) with max severity should score near 0."""
    from pipeline.grounder import FlawRegion

    regions = [
        FlawRegion(
            flaw_type="rate", start_s=0.0, end_s=60.0,
            word_indices=list(range(30)), mean_delta=1.0,
            direction="fast", threshold=0.2, severity=3.0,
        )
    ]
    grounding = _make_grounding_with_regions(regions)
    result = score(grounding, total_duration_s=60.0)

    pacing_dim = next(d for d in result.dimensions if d.dimension == "Pacing")
    assert pacing_dim.score == pytest.approx(0.0, abs=0.1), \
        f"Fully flawed Pacing should score ~0, got {pacing_dim.score}"
    assert not math.isnan(pacing_dim.score)


def test_score_partial_flaw():
    """Partial flaw (50% duration, moderate severity) should score between 0 and 20."""
    from pipeline.grounder import FlawRegion

    regions = [
        FlawRegion(
            flaw_type="pitch", start_s=15.0, end_s=45.0,
            word_indices=list(range(10, 25)), mean_delta=0.5,
            direction="flat", threshold=0.45, severity=0.5 / 0.45,
        )
    ]
    grounding = _make_grounding_with_regions(regions)
    result = score(grounding, total_duration_s=60.0)

    pitch_dim = next(d for d in result.dimensions if d.dimension == "Pitch")
    assert 0.0 <= pitch_dim.score <= 20.0, \
        f"Partial pitch flaw should score in [0, 20], got {pitch_dim.score}"
    assert not math.isnan(pitch_dim.score)


def test_score_never_nan_with_empty_delta():
    """
    Regression: ensure that a region with mean_delta=0.0 (edge case)
    does not produce NaN in the formula.
    """
    from pipeline.grounder import FlawRegion

    regions = [
        FlawRegion(
            flaw_type="energy", start_s=5.0, end_s=10.0,
            word_indices=[2, 3, 4], mean_delta=0.0,
            direction="quiet",
        )
    ]
    grounding = _make_grounding_with_regions(regions)
    result = score(grounding, total_duration_s=60.0)
    for dim in result.dimensions:
        assert not math.isnan(dim.score), f"{dim.dimension} must not be NaN"


def test_score_clamp():
    """Score should be clamped to [0, 20] even with extreme delta."""
    from pipeline.grounder import FlawRegion

    regions = [
        FlawRegion(
            flaw_type="clarity", start_s=0.0, end_s=60.0,
            word_indices=list(range(50)), mean_delta=100.0,  # extreme
            direction="degraded",
        )
    ]
    grounding = _make_grounding_with_regions(regions)
    result = score(grounding, total_duration_s=60.0)
    for dim in result.dimensions:
        assert 0.0 <= dim.score <= 20.0, \
            f"{dim.dimension} score {dim.score} outside [0, 20]"


def test_overall_is_sum_of_dimensions():
    """Overall score should equal the sum of 5 dimension scores."""
    grounding = _make_empty_grounding()
    result = score(grounding, total_duration_s=60.0)
    computed_sum = sum(d.score for d in result.dimensions)
    assert result.overall == pytest.approx(computed_sum, abs=1e-6)


def test_five_dimensions_always_present():
    """Scoring result must always include exactly 5 dimension scores."""
    grounding = _make_empty_grounding()
    result = score(grounding, total_duration_s=60.0)
    assert len(result.dimensions) == 5
    names = {d.dimension for d in result.dimensions}
    assert names == {"Pacing", "Pitch", "Pauses", "Volume", "Clarity"}


def test_score_decreases_with_severity_and_coverage():
    """More severe or longer flaws must never score higher."""
    from pipeline.grounder import FlawRegion

    def pacing(sev, end):
        g = _make_grounding_with_regions([FlawRegion("rate", 10.0, end, [1, 2, 3], sev * 0.2, "slow",
                                                     threshold=0.2, severity=sev)])
        return next(d for d in score(g, 60.0).dimensions if d.dimension == "Pacing").score

    assert pacing(1.2, 30.0) > pacing(2.0, 30.0) > pacing(3.0, 30.0)
    assert pacing(2.0, 20.0) > pacing(2.0, 40.0)


def test_pause_counts_as_event():
    """A 0.6 s pause must cost points even though it is ~1% of the clip duration."""
    from pipeline.grounder import FlawRegion

    g = _make_grounding_with_regions([FlawRegion("pause", 20.0, 20.6, [5], 0.6, "present",
                                                 threshold=0.25, severity=2.4)])
    pauses = next(d for d in score(g, 60.0).dimensions if d.dimension == "Pauses").score
    assert pauses < 19.5
