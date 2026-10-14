"""
pipeline/evaluation.py
----------------------
Metrics shared by calibration (scripts/04), held-out evaluation (scripts/05)
and the tests.

  * target IoU      — temporal IoU between detected regions of the injected flaw's
                      feature and the ground-truth regions of that feature.
  * false-positive  — seconds flagged (any feature) that do not overlap a
    seconds           ground-truth region OF THE SAME FEATURE. A pacing flaw that
                      also triggers "clarity" counts as false positive.
"""

from __future__ import annotations

import numpy as np

RES = 100  # 10 ms grid

MECHANISM_FEATURE = {
    "time": "rate",      # time_stretch_*
    "f0": "pitch",       # f0_variance_*
    "silence": "pause",  # silence_splice_*
    "gain": "energy",    # gain_*
    "clarity": "clarity",
}
FLAW_TYPE_FEATURE = {"pacing": "rate", "nearperfect": "rate", "pitch": "pitch",
                     "pause": "pause", "volume": "energy", "clarity": "clarity"}


def gt_by_feature(label: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in label.get("flaw_regions", []):
        feat = MECHANISM_FEATURE.get(r.get("mechanism", "").split("_")[0])
        if feat:
            out.setdefault(feat, []).append(r)
    return out


def _mask(regions, n: int) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for r in regions:
        s = r["start_s"] if isinstance(r, dict) else r.start_s
        e = r["end_s"] if isinstance(r, dict) else r.end_s
        m[max(0, int(s * RES)): min(n, int(np.ceil(e * RES)))] = True
    return m


def temporal_iou(detected, ground_truth, duration_s: float) -> float:
    if not detected and not ground_truth:
        return 1.0
    if not detected or not ground_truth:
        return 0.0
    n = int(duration_s * RES) + 2
    d, g = _mask(detected, n), _mask(ground_truth, n)
    union = (d | g).sum()
    return float((d & g).sum() / union) if union else 0.0


def false_positive_seconds(regions, gt: dict[str, list[dict]], duration_s: float) -> float:
    n = int(duration_s * RES) + 2
    fp = 0
    for feat in {r.flaw_type for r in regions}:
        det = _mask([r for r in regions if r.flaw_type == feat], n)
        fp += (det & ~_mask(gt.get(feat, []), n)).sum()
    return fp / RES


def clip_metrics(grounding, label: dict, duration_s: float) -> dict:
    gt = gt_by_feature(label)
    regions = grounding.flaw_regions
    per_feature = {
        feat: temporal_iou([r for r in regions if r.flaw_type == feat], g, duration_s)
        for feat, g in gt.items()
    }
    return {
        "target_iou": float(np.mean(list(per_feature.values()))) if per_feature else None,
        "iou_by_feature": per_feature,
        "fp_seconds": false_positive_seconds(regions, gt, duration_s),
        "fp_fraction": false_positive_seconds(regions, gt, duration_s) / max(duration_s, 1e-6),
    }
