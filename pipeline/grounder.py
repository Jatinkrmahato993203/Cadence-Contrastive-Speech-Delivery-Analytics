"""
pipeline/grounder.py
--------------------
Temporal flaw grounding (§3.0, §3.1).

Two comparison modes:
  Mode A — Paired: upload matches a bundled transcript (>= 90% token overlap).
            Delta computed word-for-word against the matched GOOD baseline.
            Threshold: 1.5 × sigma_group[feature] (librivox or team subgroup).

  Mode B — General: no bundled match.
            Delta computed vs. pooled calibration-GOOD distribution mean.
            Threshold: 1.5 × sigma_pooled_all[feature].
            Reference and threshold drawn from the SAME population (critical).

Output: list of FlawRegion with start_s, end_s, flaw_type, delta magnitude,
        and whether alignment in that span was uncertain.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

# Calibration-set GOOD clips' pooled mean values (populated by compute_calibration_stats)
# Defaults are 0.0 for all — deviation measured from speaker-normalised mean
POOLED_MEAN_DEFAULTS: dict[str, float] = {
    "pitch": 0.0,
    "energy": 0.0,
    "rate": 0.0,
    "clarity_mfcc": 0.0,
    "clarity_hnr": 0.0,
}

FLAW_TYPES = ["pitch", "energy", "rate", "pause", "clarity"]


def _load_cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# -- Data structures -----------------------------------------------------------

@dataclass
class FlawRegion:
    flaw_type: str         # "pitch" | "energy" | "rate" | "pause" | "clarity"
    start_s: float
    end_s: float
    word_indices: list[int]
    mean_delta: float      # mean |delta| across flagged words in this region
    direction: str         # e.g. "high" / "low" / "fast" / "slow" / "present"
    uncertain: bool = False  # True if alignment was uncertain in this span
    # Raw values for explanation templates
    uploaded_mean: float = 0.0
    baseline_mean: float = 0.0


@dataclass
class GroundingResult:
    mode: str                      # "A" | "B"
    matched_text_id: Optional[int]  # text_id for Mode A; None for Mode B
    matched_split: Optional[str]   # "calibration" | "heldout" | None
    flaw_regions: list[FlawRegion]
    # Per-word deviance flags for plotting
    word_deviant: dict[str, list[bool]]   # feature -> [bool per word]
    # Contours for UI overlay (normalised units)
    uploaded_contours: dict[str, list[float]]   # feature -> values
    baseline_contours: dict[str, list[float]]   # feature -> values (Mode A) or corridor
    normative_corridor: dict[str, tuple[float, float]]  # feature -> (mean, half-width) Mode B


# -- Mode detection -------------------------------------------------------------

_PUNCT = re.compile(r"[^\w\s]")


def _tokenize(text: str) -> list[str]:
    return _PUNCT.sub("", text.lower()).split()


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def detect_mode(
    uploaded_transcript: str,
    bundled_transcripts: dict[int, str],   # text_id -> transcript text
    bundled_splits: dict[int, str],        # text_id -> "calibration"|"heldout"
    threshold: float = 0.90,
) -> tuple[str, Optional[int], Optional[str]]:
    """
    Returns (mode, matched_text_id, matched_split).
    mode = "A" if Jaccard overlap >= threshold with any bundled transcript.
    mode = "B" otherwise.
    """
    tokens_up = _tokenize(uploaded_transcript)
    best_score, best_id = 0.0, None
    for tid, ref in bundled_transcripts.items():
        score = _jaccard(tokens_up, _tokenize(ref))
        if score > best_score:
            best_score, best_id = score, tid

    if best_score >= threshold and best_id is not None:
        return "A", best_id, bundled_splits.get(best_id)
    return "B", None, None


# -- Delta computation ---------------------------------------------------------

def _word_deltas_mode_a(
    uploaded_words,     # list[NormalizedWord]
    baseline_words,     # list[NormalizedWord]
) -> dict[str, list[float]]:
    """Word-for-word delta against paired baseline (Mode A)."""
    n = min(len(uploaded_words), len(baseline_words))
    deltas: dict[str, list[float]] = {f: [] for f in FLAW_TYPES}
    for i in range(n):
        u, b = uploaded_words[i], baseline_words[i]
        deltas["pitch"].append(u.pitch_st - b.pitch_st)
        deltas["energy"].append(u.energy_db_rel - b.energy_db_rel)
        deltas["rate"].append(u.rate_zscore - b.rate_zscore)
        # Pause: binary flag difference
        deltas["pause"].append(float(u.pause_flag) - float(b.pause_flag))
        deltas["clarity"].append(u.clarity_mfcc - b.clarity_mfcc)
    return deltas


def _word_deltas_mode_b(
    uploaded_words,     # list[NormalizedWord]
    pooled_means: dict[str, float],
) -> dict[str, list[float]]:
    """Delta vs. pooled calibration-GOOD distribution mean (Mode B)."""
    deltas: dict[str, list[float]] = {f: [] for f in FLAW_TYPES}
    for u in uploaded_words:
        deltas["pitch"].append(u.pitch_st - pooled_means.get("pitch", 0.0))
        deltas["energy"].append(u.energy_db_rel - pooled_means.get("energy", 0.0))
        deltas["rate"].append(u.rate_zscore - pooled_means.get("rate", 0.0))
        deltas["pause"].append(float(u.pause_flag) - pooled_means.get("pause", 0.0))
        deltas["clarity"].append(u.clarity_mfcc - pooled_means.get("clarity_mfcc", 0.0))
    return deltas


# -- Thresholding & region building ---------------------------------------------

def _flag_words(deltas: list[float], sigma: float, multiplier: float) -> list[bool]:
    threshold = multiplier * sigma
    return [abs(d) > threshold for d in deltas]


def _build_regions(
    flags: list[bool],
    words,
    feature: str,
    deltas: list[float],
    min_consecutive: int,
    merge_gap_ms: float,
    uncertain_words: Optional[list[bool]] = None,
) -> list[FlawRegion]:
    """
    Convert per-word boolean flags into FlawRegion objects.
    Applies min_consecutive filter and gap-merge.
    """
    regions: list[FlawRegion] = []
    i = 0
    n = len(flags)
    while i < n:
        if flags[i]:
            j = i
            while j < n and flags[j]:
                j += 1
            # Check minimum consecutive deviant words
            if (j - i) >= min_consecutive:
                span_words = words[i:j]
                span_deltas = deltas[i:j]
                span_uncertain = (
                    any((uncertain_words or [])[i:j]) if uncertain_words else False
                )
                mean_abs_delta = float(np.mean([abs(d) for d in span_deltas]))
                mean_delta_signed = float(np.mean(span_deltas))
                direction = _direction(feature, mean_delta_signed)

                regions.append(FlawRegion(
                    flaw_type=feature,
                    start_s=span_words[0].start,
                    end_s=span_words[-1].end,
                    word_indices=list(range(i, j)),
                    mean_delta=mean_abs_delta,
                    direction=direction,
                    uncertain=span_uncertain,
                    uploaded_mean=float(np.mean([_get_feature(w, feature) for w in span_words])),
                    baseline_mean=float(np.mean([_get_feature(w, feature) for w in span_words]))
                    - mean_delta_signed,
                ))
            i = j
        else:
            i += 1

    # Merge adjacent regions with gap < merge_gap_ms
    return _merge_regions(regions, merge_gap_ms / 1000.0)


def _direction(feature: str, mean_delta: float) -> str:
    if feature == "pitch":
        return "high" if mean_delta > 0 else "low"
    if feature == "energy":
        return "loud" if mean_delta > 0 else "quiet"
    if feature == "rate":
        return "fast" if mean_delta > 0 else "slow"
    if feature == "pause":
        return "present" if mean_delta > 0 else "absent"
    if feature == "clarity":
        return "degraded" if mean_delta > 0 else "improved"
    return "deviant"


def _get_feature(word, feature: str) -> float:
    mapping = {
        "pitch": word.pitch_st,
        "energy": word.energy_db_rel,
        "rate": word.rate_zscore,
        "pause": float(word.pause_flag),
        "clarity": word.clarity_mfcc,
    }
    return mapping.get(feature, 0.0)


def _merge_regions(regions: list[FlawRegion], gap_s: float) -> list[FlawRegion]:
    if not regions:
        return []
    merged = [regions[0]]
    for r in regions[1:]:
        last = merged[-1]
        if r.start_s - last.end_s <= gap_s and r.flaw_type == last.flaw_type:
            # Extend last region
            merged[-1] = FlawRegion(
                flaw_type=last.flaw_type,
                start_s=last.start_s,
                end_s=r.end_s,
                word_indices=last.word_indices + r.word_indices,
                mean_delta=float(np.mean([last.mean_delta, r.mean_delta])),
                direction=last.direction,
                uncertain=last.uncertain or r.uncertain,
                uploaded_mean=float(np.mean([last.uploaded_mean, r.uploaded_mean])),
                baseline_mean=float(np.mean([last.baseline_mean, r.baseline_mean])),
            )
        else:
            merged.append(r)
    return merged


# -- Public API ----------------------------------------------------------------

def ground(
    uploaded_norm_clip,       # NormalizedClip
    mode: str,                # "A" or "B"
    matched_text_id: Optional[int],
    matched_split: Optional[str],
    baseline_norm_clip=None,  # NormalizedClip — required for Mode A
    speaker_source: str = "librivox",   # for sigma group selection
    cfg: Optional[dict] = None,
    pooled_means: Optional[dict] = None,
) -> GroundingResult:
    """
    Run flaw grounding on an uploaded clip.

    Parameters
    ----------
    uploaded_norm_clip : NormalizedClip
    mode : "A" | "B"
    matched_text_id : int | None
    matched_split : "calibration" | "heldout" | None
    baseline_norm_clip : NormalizedClip | None (required for Mode A)
    speaker_source : "librivox" | "team" (selects sigma group for Mode A)
    cfg : pre-loaded config dict
    pooled_means : dict of pooled calibration mean per feature (Mode B)
    """
    if cfg is None:
        cfg = _load_cfg()

    g_cfg = cfg.get("grounding", {})
    multiplier = g_cfg.get("deviance_multiplier", 1.5)
    min_consec = g_cfg.get("min_consecutive_deviant", 2)
    merge_gap_ms = g_cfg.get("merge_gap_ms", 200)

    # Select sigma group
    if mode == "A":
        group_key = f"sigma_group_{speaker_source}"
    else:
        group_key = "sigma_pooled_all"

    sigma_cfg = cfg.get(group_key, {})

    feature_sigma = {
        "pitch": sigma_cfg.get("pitch", 2.5),
        "energy": sigma_cfg.get("energy", 3.0),
        "rate": sigma_cfg.get("rate", 1.0),
        "pause": sigma_cfg.get("rate", 1.0),   # pause uses rate sigma as fallback
        "clarity": sigma_cfg.get("clarity_mfcc", 0.15),
    }

    u_words = uploaded_norm_clip.words
    uncertain_words = [w.uncertain if hasattr(w, 'uncertain') else False for w in u_words]

    # Compute deltas
    if mode == "A":
        if baseline_norm_clip is None:
            raise ValueError("baseline_norm_clip required for Mode A grounding")
        b_words = baseline_norm_clip.words
        deltas = _word_deltas_mode_a(u_words, b_words)
    else:
        means = pooled_means or POOLED_MEAN_DEFAULTS
        deltas = _word_deltas_mode_b(u_words, means)

    # Flag and build regions per feature
    all_regions: list[FlawRegion] = []
    word_deviant: dict[str, list[bool]] = {}

    for feature in FLAW_TYPES:
        feat_deltas = deltas[feature]
        sigma = feature_sigma[feature]
        flags = _flag_words(feat_deltas, sigma, multiplier)
        word_deviant[feature] = flags

        regions = _build_regions(
            flags, u_words, feature, feat_deltas,
            min_consec, merge_gap_ms,
            uncertain_words=uncertain_words,
        )
        all_regions.extend(regions)

    # Build contours dict for UI
    uploaded_contours = {
        "pitch": uploaded_norm_clip.f0_contour_st,
        "energy": uploaded_norm_clip.rms_contour_db_rel,
        "rate": uploaded_norm_clip.rate_contour_zscore,
    }
    baseline_contours = {}
    normative_corridor: dict[str, tuple[float, float]] = {}

    if mode == "A" and baseline_norm_clip:
        baseline_contours = {
            "pitch": baseline_norm_clip.f0_contour_st,
            "energy": baseline_norm_clip.rms_contour_db_rel,
            "rate": baseline_norm_clip.rate_contour_zscore,
        }
    else:
        # Build normative corridor: mean ± threshold half-width
        for feat in ["pitch", "energy", "rate"]:
            mu = (pooled_means or {}).get(feat, 0.0)
            half = multiplier * feature_sigma.get(feat, 1.0)
            normative_corridor[feat] = (mu, half)

    return GroundingResult(
        mode=mode,
        matched_text_id=matched_text_id,
        matched_split=matched_split,
        flaw_regions=all_regions,
        word_deviant=word_deviant,
        uploaded_contours=uploaded_contours,
        baseline_contours=baseline_contours,
        normative_corridor=normative_corridor,
    )
