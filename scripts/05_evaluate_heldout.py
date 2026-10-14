"""
scripts/05_evaluate_heldout.py
------------------------------
Phase 5: evaluate flaw grounding on data never used for calibration.

Sets
  A. Held-out TTS texts 6 and 8 — every flaw clip (mild / moderate / egregious,
     combined, near-perfect) paired with its GOOD baseline.
  B. Real human recordings (LibriVox), same-speaker: realNN_<flaw> vs realNN_good.
  C. Real human recordings, cross-speaker: realNN_good and realNN_<flaw> vs the
     TTS GOOD baseline of the same text (the dashboard's real use case: a
     different person reading a bundled speech).
     NOTE: both clean pairs (realNN_good vs TTS) are used to choose k_noise
     (scripts/04), so they are in-sample; the flawed human variants are unseen.

Metrics per clip (pipeline/evaluation.py)
  target IoU   — IoU of detections of the injected flaw's feature vs. ground truth
  FP seconds   — flagged seconds (any feature) outside ground truth of that feature
  score        — rubric score /100

Output: dataset/labels/heldout_results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.analysis import analyze, load_bundled, load_config      # noqa: E402
from pipeline.evaluation import clip_metrics                          # noqa: E402

AUDIO = ROOT / "dataset" / "audio"
LABELS = ROOT / "dataset" / "labels"
REAL_AUDIO = AUDIO / "real_speakers"
REAL_LABELS = LABELS / "real_speakers"
HELDOUT_TEXTS = [6, 8]
TIER_ORDER = {"near_perfect": 0, "moderate": 1, "egregious": 2}


def run(audio: Path, label: dict | None, transcript: str, cfg: dict, baseline: Path | None = None) -> dict:
    res = analyze(audio, transcript, cfg=cfg, baseline_override=baseline)
    label = label or {"flaw_regions": [], "flaw_type": "none", "severity_tier": "clean"}
    m = clip_metrics(res.grounding, label, res.duration_s)
    return {
        "clip_id": audio.stem,
        "flaw_type": label.get("flaw_type"),
        "severity_tier": label.get("severity_tier"),
        "duration_s": round(res.duration_s, 3),
        "score": round(res.rubric.overall, 2),
        "target_iou": None if m["target_iou"] is None else round(m["target_iou"], 4),
        "iou_by_feature": {k: round(v, 4) for k, v in m["iou_by_feature"].items()},
        "fp_seconds": round(m["fp_seconds"], 2),
        "flagged_seconds": round(sum(r.end_s - r.start_s for r in res.grounding.flaw_regions), 2),
        "gt_regions": label.get("flaw_regions", []),
        "detected_regions": [
            {"flaw_type": r.flaw_type, "start_s": r.start_s, "end_s": r.end_s,
             "direction": r.direction, "severity": round(r.severity, 2)}
            for r in res.grounding.flaw_regions
        ],
    }


def summarize(rows: list[dict]) -> dict:
    # "nearperfect" clips are byte-identical to "pacing_mild" (same generator parameters):
    # reported per clip, but excluded from the means so they are not counted twice.
    rows = [r for r in rows if r["flaw_type"] != "nearperfect"]
    flawed = [r for r in rows if r["target_iou"] is not None]
    by_flaw: dict[str, dict] = {}
    for r in flawed:
        key = f"{r['flaw_type']}/{r['severity_tier']}"
        by_flaw.setdefault(key, []).append(r)
    return {
        "n_clips": len(rows),
        "mean_target_iou": round(float(np.mean([r["target_iou"] for r in flawed])), 4) if flawed else None,
        "mean_fp_seconds": round(float(np.mean([r["fp_seconds"] for r in rows])), 2) if rows else None,
        "by_flaw_tier": {
            k: {"n": len(v),
                "mean_iou": round(float(np.mean([x["target_iou"] for x in v])), 4),
                "mean_fp_s": round(float(np.mean([x["fp_seconds"] for x in v])), 2),
                "mean_score": round(float(np.mean([x["score"] for x in v])), 2)}
            for k, v in sorted(by_flaw.items())
        },
    }


def gradient_check(rows: list[dict], prefix: str) -> list[dict]:
    """Score should fall monotonically mild -> moderate -> egregious for each flaw."""
    out = []
    for flaw in ("pacing", "pitch", "pause", "volume", "clarity"):
        tiers = {r["severity_tier"]: r["score"] for r in rows
                 if r["clip_id"].startswith(prefix) and r["flaw_type"] == flaw}
        if len(tiers) == 3:
            s = [tiers["near_perfect"], tiers["moderate"], tiers["egregious"]]
            out.append({"set": prefix, "flaw": flaw, "scores_mild_mod_egr": s,
                        "monotonic": s[0] >= s[1] >= s[2]})
    return out


def main():
    cfg = load_config()
    bundled, _ = load_bundled()
    results: dict = {"held_out_texts": HELDOUT_TEXTS, "config": {
        "floors": cfg["grounding"]["floors"], "k_noise": cfg["grounding"]["k_noise"],
        "clarity_feature": cfg["normalization"]["clarity_feature"]}}

    # A. Held-out TTS texts
    rows_a = []
    for tid in HELDOUT_TEXTS:
        rows_a.append(run(AUDIO / f"text_{tid:02d}_good.wav", None, bundled[tid], cfg))
        for lp in sorted(LABELS.glob(f"text_{tid:02d}_*.json")):
            if lp.stem.endswith("_features"):
                continue
            rows_a.append(run(AUDIO / f"{lp.stem}.wav", json.loads(lp.read_text()), bundled[tid], cfg))
            r = rows_a[-1]
            print(f"A {r['clip_id']:30s} IoU={r['target_iou']}  FP={r['fp_seconds']:5.1f}s  score={r['score']:5.1f}", flush=True)
    results["heldout_tts"] = {"summary": summarize(rows_a), "per_clip": rows_a}
    grad = []
    for tid in HELDOUT_TEXTS:
        grad += gradient_check(rows_a, f"text_{tid:02d}")

    # B / C. Real human recordings
    rows_b, rows_c = [], []
    if REAL_LABELS.exists():
        for good in sorted(REAL_AUDIO.glob("real*_good.wav")):
            prefix = good.stem.replace("_good", "")
            tid = int(prefix[4:])
            own = ROOT / "dataset" / "transcripts" / f"{prefix}.txt"
            tx = own.read_text(encoding="utf-8").strip() if own.exists() else bundled[tid]   # what the reader says
            rows_c.append({**run(good, None, tx, cfg), "used_for_k_calibration": True})
            for lp in sorted(REAL_LABELS.glob(f"{prefix}_*.json")):
                label = json.loads(lp.read_text())
                wav = REAL_AUDIO / f"{lp.stem}.wav"
                rows_b.append(run(wav, label, tx, cfg, baseline=good))
                rows_c.append(run(wav, label, tx, cfg))
                b, c = rows_b[-1], rows_c[-1]
                print(f"B {b['clip_id']:30s} same-speaker IoU={b['target_iou']} FP={b['fp_seconds']:5.1f}s | "
                      f"C cross-speaker IoU={c['target_iou']} FP={c['fp_seconds']:5.1f}s", flush=True)
            grad += gradient_check(rows_b, prefix)
        results["real_same_speaker"] = {"summary": summarize(rows_b), "per_clip": rows_b}
        clean_c = [r for r in rows_c if r["target_iou"] is None]
        results["real_cross_speaker"] = {
            "summary": summarize(rows_c),
            "clean_pairs": [{k: r[k] for k in ("clip_id", "score", "flagged_seconds", "duration_s",
                                                "used_for_k_calibration")} for r in clean_c],
            "per_clip": rows_c,
        }
    results["severity_gradient"] = grad
    results["severity_gradient_monotonic"] = f"{sum(g['monotonic'] for g in grad)}/{len(grad)}"

    (LABELS / "heldout_results.json").write_text(json.dumps(results, indent=2))
    print("\n=== Summary ===")
    for key in ("heldout_tts", "real_same_speaker", "real_cross_speaker"):
        if key in results:
            s = results[key]["summary"]
            print(f"{key:20s} mean target IoU={s['mean_target_iou']}  mean FP={s['mean_fp_seconds']}s  (n={s['n_clips']})")
    if "real_cross_speaker" in results:
        for c in results["real_cross_speaker"]["clean_pairs"]:
            print(f"  clean cross-speaker {c['clip_id']}: score {c['score']}, flagged {c['flagged_seconds']}s "
                  f"of {c['duration_s']}s{' (used for k)' if c['used_for_k_calibration'] else ''}")
    print("severity gradient monotonic:", results["severity_gradient_monotonic"])
    print("Saved dataset/labels/heldout_results.json")


if __name__ == "__main__":
    main()
