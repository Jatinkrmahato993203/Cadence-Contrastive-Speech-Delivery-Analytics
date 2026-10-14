"""
pipeline/scorer.py
------------------
Rubric scoring (§4).

Five dimensions: Pacing, Pitch, Pauses, Volume, Clarity — each 0-20 (100 total).

Formula:
  if dimension has zero flagged regions:
      severity_weight = 0          # EXPLICIT base case — avoids NaN (IEEE-754)
  else:
      severity_weight = mean(|delta| across flagged regions)

  dimension_score = 20 * (1 - clamp(flawed_duration / total_duration, 0, 1) * severity_weight)

Critical: without the explicit base case, numpy.mean([]) returns NaN, and
0 * NaN = NaN under IEEE-754 — silently producing NaN for every clean clip.
This is covered by test_score_zero_regions in tests/test_scoring.py.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

DIMENSION_TO_FLAW = {
    "Pacing": "rate",
    "Pitch": "pitch",
    "Pauses": "pause",
    "Volume": "energy",
    "Clarity": "clarity",
}

FLAW_TO_DIMENSION = {v: k for k, v in DIMENSION_TO_FLAW.items()}


# -- Data structures -----------------------------------------------------------

@dataclass
class DimensionScore:
    dimension: str          # "Pacing" | "Pitch" | "Pauses" | "Volume" | "Clarity"
    flaw_type: str          # internal key
    score: float            # 0.0 – 20.0
    flawed_duration_s: float
    total_duration_s: float
    severity_weight: float
    region_count: int
    max_delta: float


@dataclass
class RubricScore:
    overall: float                          # 0 – 100
    dimensions: list[DimensionScore]
    mode: str                               # "A" | "B"
    matched_text_id: object                 # int | None
    matched_split: object                   # "calibration" | "heldout" | None
    is_nan_safe: bool = True                # always True if scorer is correct


# -- Helpers -------------------------------------------------------------------

def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


PAUSE_MIN_EFFECT_S = 5.0   # a long unplanned silence derails the whole phrase around it
MAX_SEVERITY = 3.0


def _region_weight(region) -> float:
    sev = getattr(region, "severity", 0.0) or 0.0
    if sev <= 0 and getattr(region, "threshold", 0.0) > 0:
        sev = region.mean_delta / region.threshold
    return 1.5 * min(max(sev, 0.0), MAX_SEVERITY) / MAX_SEVERITY


def _score_dimension(
    flaw_type: str,
    dimension: str,
    regions,           # list[FlawRegion] for this flaw type
    total_duration_s: float,
) -> DimensionScore:
    """
    Score a single rubric dimension.
    Returns DimensionScore with score in [0, 20].
    """
    relevant = [r for r in regions if r.flaw_type == flaw_type]

    if not relevant:
        # EXPLICIT base case: severity_weight = 0 to avoid NaN
        return DimensionScore(
            dimension=dimension,
            flaw_type=flaw_type,
            score=20.0,
            flawed_duration_s=0.0,
            total_duration_s=total_duration_s,
            severity_weight=0.0,
            region_count=0,
            max_delta=0.0,
        )

    total = max(total_duration_s, 1e-6)
    flawed_duration_s = sum(r.end_s - r.start_s for r in relevant)
    penalty = 0.0
    for r in relevant:
        eff = r.end_s - r.start_s
        if flaw_type == "pause":
            eff = max(eff, PAUSE_MIN_EFFECT_S)
        penalty += _clamp(eff / total, 0.0, 1.0) * _region_weight(r)
    severity_weight = float(np.mean([_region_weight(r) for r in relevant]))
    all_deltas = [r.mean_delta for r in relevant]
    score = _clamp(20.0 * (1.0 - _clamp(penalty, 0.0, 1.0)), 0.0, 20.0)

    # Sanity check — must never be NaN
    if math.isnan(score):
        logger.error(
            "NaN score detected for %s — falling back to 0.0. "
            "This indicates a bug in the scorer base-case logic.",
            dimension,
        )
        score = 0.0

    return DimensionScore(
        dimension=dimension,
        flaw_type=flaw_type,
        score=score,
        flawed_duration_s=flawed_duration_s,
        total_duration_s=total_duration_s,
        severity_weight=severity_weight,
        region_count=len(relevant),
        max_delta=max(all_deltas),
    )


# -- Public API ----------------------------------------------------------------

def score(grounding_result, total_duration_s: float) -> RubricScore:
    """
    Compute rubric scores from a GroundingResult.

    Parameters
    ----------
    grounding_result : GroundingResult from grounder.py
    total_duration_s : float
        Total clip duration in seconds.

    Returns
    -------
    RubricScore with .overall in [0, 100] and .dimensions list.
    """
    regions = grounding_result.flaw_regions

    dim_scores = [
        _score_dimension(flaw_type, dim, regions, total_duration_s)
        for dim, flaw_type in DIMENSION_TO_FLAW.items()
    ]

    overall = sum(d.score for d in dim_scores)

    # Final NaN guard on overall
    if math.isnan(overall):
        logger.error("NaN overall score detected — setting to 0.0")
        overall = 0.0

    return RubricScore(
        overall=overall,
        dimensions=dim_scores,
        mode=grounding_result.mode,
        matched_text_id=grounding_result.matched_text_id,
        matched_split=grounding_result.matched_split,
    )

