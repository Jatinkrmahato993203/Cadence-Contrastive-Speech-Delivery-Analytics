"""
pipeline/grounder.py
--------------------
Temporal flaw grounding (§3.0, §3.1).

Mode A — Paired: the uploaded transcript matches a bundled transcript
         (token-sequence similarity >= 0.90). Both clips are force-aligned to
         the same token sequence, so word i of the upload is paired with word i
         of the GOOD baseline, and per-word deltas are computed:

            pitch   : log2(pitch variability_upload / pitch variability_baseline)
            energy  : energy_db_rel_upload - energy_db_rel_baseline            (dB)
            rate    : log2(word duration_baseline / word duration_upload)    (+ = faster)
            pause   : pause_upload - pause_baseline                          (s)
            clarity : clarity_upload - clarity_baseline                     (feature units)

Mode B — General: no paired baseline. Deltas are measured against the clip's
         own robust median (local anomalies in the speaker's own delivery); pauses
         are compared with an absolute "long pause" floor.

Thresholding, per feature f:
    tau_f = max(floor_f, 3 * noise_f)    if the upload is an edited copy of the baseline
                                         recording (paired-word timbre distance below
                                         same_recording_max_dist): a 3-sigma rule on the
                                         alignment jitter of the shared audio
    tau_f = max(floor_f, k_f * noise_f)  otherwise (a different recording): k_f calibrated
                                         on clean human-vs-TTS readings
    noise_f = 20th percentile of |d| / 0.2533   (zero-anchored; 0 = no change)

  * floor_f is the smallest perceptually meaningful effect (calibrated on the
    calibration texts only, see scripts/04_extract_features.py --calibrate).
  * noise_f adapts to how different the two readings naturally are. Its
    breakdown point is 80%: a flaw may cover up to ~4/5 of the words without
    inflating the noise estimate. For a re-recording by
    the same speaker noise_f ~ 0 and the floor applies; for a different
    speaker, natural prosodic variation raises tau_f so ordinary stylistic
    differences are not reported as flaws.

Flagged words are grouped into same-sign runs (grounding.min_run_words: 2 words,
6 for rate — pacing is sustained —, 1 for pauses) and merged across gaps < merge_gap_ms.
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

FLAW_TYPES = ["pitch", "energy", "rate", "pause", "clarity"]

UNITS = {"pitch": "log2 ratio", "energy": "dB", "rate": "log2 ratio", "pause": "s", "clarity": "units"}

DEFAULT_FLOORS = {"pitch": 0.45, "energy": 4.0, "rate": 0.2, "pause": 0.25, "clarity": 1.0}
DEFAULT_FLOORS_B = {"pitch": 1.0, "energy": 9.0, "rate": 0.6, "pause": 1.0, "clarity": 3.0}
SHORTH_COVERAGE = 0.20      # breakdown point 80%: a flaw may cover up to ~4/5 of the words
SHORTH_HALF_WIDTH = 0.2533  # half-width (in sigma) of the central 20% of a normal distribution
# Std of a running median of w i.i.d. Gaussian values relative to the raw std
SMOOTH_NOISE_FACTOR = {1: 1.0, 3: 0.67, 5: 0.54}


def _load_cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# -- Data structures -------------------------------------------------------------

@dataclass
class FlawRegion:
    flaw_type: str         # "pitch" | "energy" | "rate" | "pause" | "clarity"
    start_s: float
    end_s: float
    word_indices: list[int]
    mean_delta: float      # mean |delta| across flagged words (feature units, see UNITS)
    direction: str         # "flat"/"exaggerated", "quiet"/"loud", "slow"/"fast", "present"/"absent", "degraded"/"sharper"
    uncertain: bool = False
    # Interpretable values for explanations (natural units)
    uploaded_mean: float = 0.0
    baseline_mean: float = 0.0
    threshold: float = 0.0       # tau_f used for this feature
    severity: float = 0.0        # mean |delta| / tau_f  (>= 1 by construction)


@dataclass
class GroundingResult:
    mode: str
    matched_text_id: Optional[int]
    matched_split: Optional[str]
    flaw_regions: list[FlawRegion]
    word_deviant: dict[str, list[bool]]
    uploaded_contours: dict[str, list[float]]
    baseline_contours: dict[str, list[float]]
    normative_corridor: dict[str, tuple[float, float]]
    word_deltas: dict[str, list[float]] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    pairs: list[Optional[int]] = field(default_factory=list)   # baseline word index per uploaded word
    recording_match: float = float("nan")   # P20 of paired-word MFCC (c1-c12) distance
    same_recording: bool = False            # upload is an edited copy of the baseline recording


# -- Mode detection ------------------------------------------------------------------

_PUNCT = re.compile(r"[^\w\s']")


def _tokenize(text: str) -> list[str]:
    return _PUNCT.sub(" ", (text or "").lower()).split()


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def _sequence_similarity(a: list[str], b: list[str]) -> float:
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def detect_mode(
    uploaded_transcript: str,
    bundled_transcripts: dict[int, str],
    bundled_splits: dict[int, str],
    threshold: float = 0.90,
) -> tuple[str, Optional[int], Optional[str]]:
    """
    Returns (mode, matched_text_id, matched_split). Mode A when the token
    sequence similarity with a bundled transcript is >= *threshold*.
    """
    tokens_up = _tokenize(uploaded_transcript)
    best_score, best_id = 0.0, None
    for tid, ref in bundled_transcripts.items():
        score = _sequence_similarity(tokens_up, _tokenize(ref))
        if score > best_score:
            best_score, best_id = score, tid
    if best_score >= threshold and best_id is not None:
        return "A", best_id, bundled_splits.get(best_id)
    return "B", None, None


# -- Pairing -----------------------------------------------------------------------------

def pair_words(uploaded_words, baseline_words) -> list[Optional[int]]:
    """Baseline index for each uploaded word (None if the word has no counterpart)."""
    a = [str(w.word).lower() for w in uploaded_words]
    b = [str(w.word).lower() for w in baseline_words]
    if a == b:
        return list(range(len(a)))
    pairs: list[Optional[int]] = [None] * len(a)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            pairs[blk.a + k] = blk.b + k
    return pairs


# -- Delta computation -------------------------------------------------------------------

def recording_match(u_words, b_words, pairs) -> float:
    """
    20th percentile of the timbre distance (MFCC c1-c12, gain-independent) between
    paired words. ~0 when the upload is an edited copy of the baseline recording
    (its unedited words are the same audio); 28+ for any different recording in the
    dataset, including the same text read by another voice.
    """
    dists = [
        float(np.linalg.norm(np.asarray(u_words[i].mfcc_mean[1:]) - np.asarray(b_words[j].mfcc_mean[1:])))
        for i, j in enumerate(pairs)
        if j is not None and len(u_words[i].mfcc_mean) > 1 and len(b_words[j].mfcc_mean) > 1
    ]
    return float(np.percentile(dists, 20)) if dists else float("nan")


def _safe_log2_ratio(u: float, b: float, eps: float) -> float:
    return float(np.log2((max(u, 0.0) + eps) / (max(b, 0.0) + eps)))


def _neighbour_max(values: list[float], i: int) -> float:
    return max(values[max(0, i - 1): i + 2]) if values else 0.0


def pause_delta(u_p: list[float], b_p: list[float], i: int, j: int) -> float:
    """
    Pause delta tolerant to a one-word shift in where the aligner puts a short
    word next to a silence (either side of the pause is acoustically plausible).
    The per-word difference is capped by the difference in TOTAL silence over
    the +/-1-word window: a pause that merely moved one word cancels out, while
    a pause inserted next to a natural pause still counts in full.
    """
    raw = u_p[i] - b_p[j]
    excess = sum(u_p[max(0, i - 1): i + 2]) - sum(b_p[max(0, j - 1): j + 2])
    if raw > 0:
        return max(0.0, min(raw, excess))
    if raw < 0:
        return min(0.0, max(raw, excess))
    return 0.0


def _window_durations(words, half: int) -> np.ndarray:
    """Articulation time (pauses excluded) of each word's +/-half neighbourhood.
    Boundary errors between two words move time from one to the other, so they
    cancel inside the window."""
    d = np.array([w.duration_s or (w.end - w.start) for w in words], float)
    c = np.r_[0.0, np.cumsum(d)]
    idx = np.arange(len(d))
    lo, hi = np.maximum(idx - half, 0), np.minimum(idx + half + 1, len(d))
    return c[hi] - c[lo]


def recenter(d: np.ndarray, tol: float) -> np.ndarray:
    """
    Remove a global offset (gain, voice timbre) from a delta series. When at least
    15% of the words are unchanged (|d| <= tol — e.g. the same recording with an
    edit in it), zero IS the reference; otherwise use the densest cluster. Without
    this, a degradation covering most of the clip becomes the "normal" cluster.
    """
    x = d[np.isfinite(d)]
    if len(x) and np.mean(np.abs(x) <= tol) >= 0.15:
        return d
    return d - robust_center(d)


def _raw(word, raw_attr: str, fallback_attr: str) -> float:
    """Un-normalized value when the record carries it (real pipeline), else the normalized one."""
    v = getattr(word, raw_attr, None)
    return getattr(word, fallback_attr) if v is None else v


def _deltas_mode_a(u_words, b_words, pairs, rate_half: int = 2,
                   floors: Optional[dict] = None) -> dict[str, np.ndarray]:
    n = len(u_words)
    d = {f: np.full(n, np.nan) for f in FLAW_TYPES}
    u_p = [w.pause_duration_s for w in u_words]
    b_p = [w.pause_duration_s for w in b_words]
    u_win, b_win = _window_durations(u_words, rate_half), _window_durations(b_words, rate_half)
    for i, j in enumerate(pairs):
        if j is None:
            continue
        u, b = u_words[i], b_words[j]
        d["pitch"][i] = _safe_log2_ratio(u.pitch_var_st, b.pitch_var_st, eps=0.5)
        # Raw (un-normalized) level and clarity: per-clip normalization subtracts a clip
        # statistic that a flaw covering most of the clip would itself define;
        # recenter() below removes genuine offsets (gain, timbre) robustly instead.
        d["energy"][i] = _raw(u, "energy_db", "energy_db_rel") - _raw(b, "energy_db", "energy_db_rel")
        d["rate"][i] = _safe_log2_ratio(b_win[j], u_win[i], eps=0.05)
        d["pause"][i] = pause_delta(u_p, b_p, i, j)
        d["clarity"][i] = _raw(u, "clarity_raw", "clarity") - _raw(b, "clarity_raw", "clarity")
    # Level and spectral balance carry recording/voice offsets: re-centre on the unflawed words.
    floors = {**DEFAULT_FLOORS, **(floors or {})}
    for f in ("energy", "clarity"):
        d[f] = recenter(d[f], 0.5 * floors[f])
    return d


def _deltas_mode_b(u_words, rate_half: int = 2) -> dict[str, np.ndarray]:
    pv = np.array([w.pitch_var_st for w in u_words], float)
    en = np.array([w.energy_db_rel for w in u_words], float)
    du = _window_durations(u_words, rate_half)
    cl = np.array([w.clarity for w in u_words], float)
    pa = np.array([w.pause_duration_s for w in u_words], float)
    lpv, ldu = np.log2(pv + 0.5), np.log2(du + 0.05)
    return {
        "pitch": lpv - robust_center(lpv),
        "energy": en - robust_center(en),
        "rate": robust_center(ldu) - ldu,
        "pause": pa,
        "clarity": cl - robust_center(cl),
    }


def _rolling_median(x: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or len(x) == 0:
        return x
    h = width // 2
    out = np.empty_like(x)
    for i in range(len(x)):
        win = x[max(0, i - h): i + h + 1]
        win = win[np.isfinite(win)]
        out[i] = np.median(win) if len(win) else np.nan
    return out


def _shorth(d: np.ndarray, coverage: float = SHORTH_COVERAGE) -> Optional[tuple[float, float]]:
    """(lo, hi) of the shortest interval holding *coverage* of the finite values."""
    x = np.sort(d[np.isfinite(d)])
    h = int(np.ceil(coverage * len(x)))
    if len(x) < 8 or h < 2:
        return None
    widths = x[h - 1:] - x[: len(x) - h + 1]
    i = int(np.argmin(widths))
    return float(x[i]), float(x[i + h - 1])


def robust_center(d: np.ndarray) -> float:
    """
    Location of the densest 20% of the deltas — the unflawed words. Used to
    remove a global offset (microphone gain, voice timbre) without letting a
    flaw that covers most of the clip shift the reference.
    """
    s = _shorth(d)
    if s is None:
        x = d[np.isfinite(d)]
        return float(np.median(x)) if len(x) else 0.0
    return (s[0] + s[1]) / 2


def robust_noise(d: np.ndarray, coverage: float = SHORTH_COVERAGE) -> float:
    """
    Noise scale of the deltas, anchored at zero ("no change from the reference"):
    the 20th percentile of |d| divided by its value for a standard normal (0.2533).
    The 20% of words closest to the reference are the unflawed ones even when a
    flaw covers up to ~80% of the clip, and — unlike a free-floating shortest
    interval — a long, uniform flaw cannot be mistaken for the reference cluster.
    Unbiased for Gaussian noise without a finite-sample correction.
    """
    x = np.abs(d[np.isfinite(d)])
    if len(x) < 8:
        return 0.0
    return float(np.quantile(x, coverage) / SHORTH_HALF_WIDTH)


# -- Region building -----------------------------------------------------------------------

def _direction(feature: str, signed: float) -> str:
    pos = signed > 0
    return {
        "pitch": "exaggerated" if pos else "flat",
        "energy": "loud" if pos else "quiet",
        "rate": "fast" if pos else "slow",
        "pause": "present" if pos else "absent",
        "clarity": "sharper" if pos else "degraded",
    }.get(feature, "deviant")


def _natural_value(word, feature: str) -> float:
    return {
        "pitch": word.pitch_var_st,
        "energy": word.energy_db_rel,
        "rate": word.rate_wps,
        "pause": word.pause_duration_s,
        "clarity": word.clarity,
    }[feature]


def _merge_regions(regions: list[FlawRegion], gap_s: float) -> list[FlawRegion]:
    if not regions:
        return []
    regions = sorted(regions, key=lambda r: (r.flaw_type, r.start_s))
    merged = [regions[0]]
    for r in regions[1:]:
        last = merged[-1]
        if (r.flaw_type == last.flaw_type and r.direction == last.direction
                and r.start_s - last.end_s <= gap_s):
            n1, n2 = max(len(last.word_indices), 1), max(len(r.word_indices), 1)
            w = lambda a, b: (a * n1 + b * n2) / (n1 + n2)
            merged[-1] = FlawRegion(
                flaw_type=last.flaw_type,
                start_s=last.start_s,
                end_s=max(last.end_s, r.end_s),
                word_indices=last.word_indices + r.word_indices,
                mean_delta=w(last.mean_delta, r.mean_delta),
                direction=last.direction,
                uncertain=last.uncertain or r.uncertain,
                uploaded_mean=w(last.uploaded_mean, r.uploaded_mean),
                baseline_mean=w(last.baseline_mean, r.baseline_mean),
                threshold=last.threshold,
                severity=w(last.severity, r.severity),
            )
        else:
            merged.append(r)
    return sorted(merged, key=lambda r: r.start_s)


def _pause_ratio_ok(i: int, u_words, b_words, pairs, ratio: float, tempo_changed: bool = True) -> bool:
    """
    When the surrounding tempo also changed (a slowed passage stretches its
    existing pauses), a pause flaw must at least double (or halve) the reference
    silence, so proportional stretching is reported as pacing. With unchanged
    tempo, any silence change above threshold is a pause flaw.
    """
    if b_words is None or pairs[i] is None or not tempo_changed:
        return True
    j = pairs[i]
    u_p = [w.pause_duration_s for w in u_words]
    b_p = [w.pause_duration_s for w in b_words]
    u, b = u_p[i], b_p[j]
    if u >= b:
        return u > ratio * _neighbour_max(b_p, j)
    return b > ratio * _neighbour_max(u_p, i)


def _build_regions(feature, d, tau, u_words, b_words, pairs, min_consecutive, merge_gap_s,
                   pause_ratio: float = 2.0, rate_context: Optional[tuple[np.ndarray, float]] = None,
                   ) -> tuple[list[FlawRegion], list[bool]]:
    """*rate_context* = (rate deltas, rate threshold), used to gate pauses inside tempo changes."""
    n = len(d)
    flags = [bool(np.isfinite(d[i]) and abs(d[i]) > tau) for i in range(n)]
    if feature == "pause":
        def tempo_changed(i: int) -> bool:
            if rate_context is None:
                return True
            rd, rtau = rate_context
            return bool(np.isfinite(rd[i]) and abs(rd[i]) > 0.5 * rtau)
        flags = [f and _pause_ratio_ok(i, u_words, b_words, pairs, pause_ratio, tempo_changed(i))
                 for i, f in enumerate(flags)]
    if isinstance(min_consecutive, dict):
        min_run = min_consecutive.get(feature, 2)
    else:
        min_run = 1 if feature == "pause" else min_consecutive
    regions: list[FlawRegion] = []
    i = 0
    while i < n:
        if not flags[i]:
            i += 1
            continue
        sign = np.sign(d[i])
        j = i
        while j < n and flags[j] and np.sign(d[j]) == sign:
            j += 1
        if j - i >= min_run:
            span = list(range(i, j))
            if feature == "pause":
                # The flaw is the silence itself: previous word end -> this word start
                start_s = u_words[i - 1].end if i > 0 else max(0.0, u_words[i].start - u_words[i].pause_duration_s)
                end_s = u_words[j - 1].start
                if end_s <= start_s:
                    start_s, end_s = u_words[i].start, u_words[j - 1].end
            else:
                start_s, end_s = u_words[i].start, u_words[j - 1].end
            abs_d = np.abs(d[i:j])
            up_vals = [_natural_value(u_words[k], feature) for k in span]
            if b_words is not None:
                b_vals = [_natural_value(b_words[pairs[k]], feature) for k in span if pairs[k] is not None]
            else:
                b_vals = []
            regions.append(FlawRegion(
                flaw_type=feature,
                start_s=round(float(start_s), 3),
                end_s=round(float(end_s), 3),
                word_indices=span,
                mean_delta=float(np.mean(abs_d)),
                direction=_direction(feature, float(np.mean(d[i:j]))),
                uncertain=any(getattr(u_words[k], "uncertain", False) for k in span),
                uploaded_mean=float(np.mean(up_vals)),
                baseline_mean=float(np.mean(b_vals)) if b_vals else float("nan"),
                threshold=float(tau),
                severity=float(np.mean(abs_d) / tau) if tau > 0 else 1.0,
            ))
        i = j
    return _merge_regions(regions, merge_gap_s), flags


UNSMOOTHED = ("pause", "rate")   # pause is an event; rate is already a 5-word window


def prepared_delta(raw: np.ndarray, feature: str, smooth: int) -> tuple[np.ndarray, float]:
    """
    (series to threshold, its noise scale). Noise is measured on the raw deltas
    (smoothing correlates neighbours, which collapses the shortest-interval
    estimate) and scaled to the spread of the smoothed series.
    """
    if feature == "pause":
        # Most word boundaries have no pause in either reading (delta exactly 0);
        # the spread that matters is at boundaries where someone paused.
        return raw, robust_noise(raw[np.abs(raw) > 0.02])
    if feature in UNSMOOTHED:
        return raw, robust_noise(raw)
    return _rolling_median(raw, smooth), robust_noise(raw) * SMOOTH_NOISE_FACTOR.get(smooth, 1.0)


def _rolling_mean(x: np.ndarray, width: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    h = width // 2
    for i in range(len(x)):
        win = x[max(0, i - h): i + h + 1]
        win = win[np.isfinite(win)]
        if len(win) >= max(3, width // 2):
            out[i] = float(np.mean(win))
    return out


LONG_SCALE_FEATURES = ("pitch", "energy", "rate", "clarity")


def long_scale_delta(raw: np.ndarray, width: int) -> tuple[np.ndarray, float]:
    """
    Sustained-deviation series: the per-word deltas averaged over *width* words,
    with the noise scale shrunk accordingly (1/sqrt(width)). Natural differences
    between two readers are mostly word-to-word fluctuation that averages out;
    a flaw sustained over a phrase or longer does not.
    """
    return _rolling_mean(raw, width), robust_noise(raw) / np.sqrt(width)


def k_for(g: dict, feature: str) -> float:
    k = g.get("k_noise", 3.0)
    return float(k.get(feature, 3.0) if isinstance(k, dict) else k)


# -- Public API ---------------------------------------------------------------------------------

def ground(
    uploaded_norm_clip,
    mode: str,
    matched_text_id: Optional[int],
    matched_split: Optional[str],
    baseline_norm_clip=None,
    speaker_source: str = "librivox",     # kept for API compatibility (thresholds are speaker-adaptive)
    cfg: Optional[dict] = None,
    pooled_means: Optional[dict] = None,  # kept for API compatibility
) -> GroundingResult:
    """Locate flaw regions in *uploaded_norm_clip* (Mode A needs *baseline_norm_clip*)."""
    if cfg is None:
        cfg = _load_cfg()
    g = cfg.get("grounding", {})
    # Pacing is a sustained property: a rate flaw must span >= 6 words, otherwise
    # ordinary phrase-final lengthening in human speech reads as "slow".
    min_consec = {**{"pitch": 2, "energy": 2, "rate": 6, "pause": 1, "clarity": 2}, **g.get("min_run_words", {})}
    merge_gap_s = float(g.get("merge_gap_ms", 300)) / 1000.0
    smooth = int(g.get("smoothing_words", 3))
    rate_half = int(g.get("rate_window_words", 5)) // 2

    u_words = uploaded_norm_clip.words
    if mode == "A":
        if baseline_norm_clip is None:
            raise ValueError("baseline_norm_clip required for Mode A grounding")
        b_words = baseline_norm_clip.words
        pairs = pair_words(u_words, b_words)
        floors = {**DEFAULT_FLOORS, **g.get("floors", {})}
        raw = _deltas_mode_a(u_words, b_words, pairs, rate_half, floors)
        match = recording_match(u_words, b_words, pairs)
        # An edited copy of the baseline recording has no between-recording variation,
        # only alignment jitter (3-sigma rule); any other recording gets the calibrated k.
        same_rec = bool(np.isfinite(match) and match < float(g.get("same_recording_max_dist", 20.0)))
    else:
        b_words, pairs = None, [None] * len(u_words)
        raw = _deltas_mode_b(u_words, rate_half)
        floors = {**DEFAULT_FLOORS_B, **g.get("floors_mode_b", {})}
        match, same_rec = float("nan"), False

    all_regions: list[FlawRegion] = []
    prepared: dict[str, np.ndarray] = {}
    word_deviant: dict[str, list[bool]] = {}
    word_deltas: dict[str, list[float]] = {}
    thresholds: dict[str, float] = {}
    for feature in FLAW_TYPES:
        d, noise = prepared_delta(raw[feature], feature, smooth)
        if mode == "B" and feature == "pause":
            tau = floors["pause"]                     # absolute long-pause criterion
        else:
            k = float(g.get("k_same_recording", 3.0)) if same_rec else k_for(g, feature)
            tau = max(floors[feature], k * noise)
        regions, flags = _build_regions(feature, d, tau, u_words, b_words, pairs, min_consec, merge_gap_s,
                                        pause_ratio=float(g.get("pause_ratio", 2.0)),
                                        rate_context=(prepared["rate"], thresholds["rate"]) if feature == "pause" else None)
        prepared[feature] = d
        width = int(g.get("long_window_words", 15))
        k_long = g.get("k_long", {})
        if not same_rec and feature in LONG_SCALE_FEATURES and feature in k_long and len(u_words) >= 2 * width:
            d_long, noise_long = long_scale_delta(raw[feature], width)
            tau_long = max(floors[feature], float(k_long[feature]) * noise_long)
            long_regions, long_flags = _build_regions(
                feature, d_long, tau_long, u_words, b_words, pairs,
                {**min_consec, feature: max(min_consec.get(feature, 2), width // 2)}, merge_gap_s)
            regions = _merge_regions(regions + long_regions, merge_gap_s)
            flags = [a or b for a, b in zip(flags, long_flags)]
        all_regions.extend(regions)
        word_deviant[feature] = flags
        word_deltas[feature] = [float(x) if np.isfinite(x) else 0.0 for x in d]
        thresholds[feature] = float(tau)

    uploaded_contours = {
        "pitch": uploaded_norm_clip.f0_contour_st,
        "energy": uploaded_norm_clip.rms_contour_db_rel,
        "rate": [w.rate_wps for w in u_words],
    }
    baseline_contours: dict[str, list[float]] = {}
    normative_corridor: dict[str, tuple[float, float]] = {}
    if mode == "A":
        baseline_contours = {
            "pitch": baseline_norm_clip.f0_contour_st,
            "energy": baseline_norm_clip.rms_contour_db_rel,
            "rate": [w.rate_wps for w in b_words],
        }
    else:
        rates = np.array([w.rate_wps for w in u_words], float)
        mu = float(np.median(rates)) if len(rates) else 0.0
        normative_corridor = {
            "pitch": (0.0, 6.0),
            "energy": (float(np.median([w.energy_db_rel for w in u_words])) if u_words else 0.0, floors["energy"]),
            "rate": (mu, mu * (2 ** floors["rate"] - 1)),
        }

    return GroundingResult(
        mode=mode,
        matched_text_id=matched_text_id,
        matched_split=matched_split,
        flaw_regions=sorted(all_regions, key=lambda r: r.start_s),
        word_deviant=word_deviant,
        uploaded_contours=uploaded_contours,
        baseline_contours=baseline_contours,
        normative_corridor=normative_corridor,
        word_deltas=word_deltas,
        thresholds=thresholds,
        pairs=pairs,
        recording_match=match,
        same_recording=same_rec,
    )
