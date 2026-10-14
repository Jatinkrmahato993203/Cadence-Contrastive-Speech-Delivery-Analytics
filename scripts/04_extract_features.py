"""
scripts/04_extract_features.py
------------------------------
Phase 2: extract per-word features for every clip and (optionally) calibrate
the grounding thresholds on the CALIBRATION texts only (1-5, 7).

Usage:
    python scripts/04_extract_features.py                 # features JSON per clip
    python scripts/04_extract_features.py --calibrate     # + write thresholds to config.yaml

Calibration uses texts 1-5 and 7 (TTS) plus the LibriVox reading of text 1 (real01)
and its flawed variants; held-out texts 6, 8, 9-11 and real02 are never used:
  1. Clarity measure — for each candidate (HNR, spectral contrast, flux,
     >3 kHz energy ratio, flatness) compute the paired per-word delta on the
     clarity_* clips and the AUC separating words inside the injected region
     from unflawed words. The best measure (and its sign) is kept.
  2. Floors — per feature, choose the floor maximising
        mean target IoU (clips containing that flaw, all severity tiers)
        - LAMBDA_FP * mean false-positive fraction of that feature (all flawed clips).
  3. k_noise — the floors only matter when two readings are acoustically
     near-identical (same speaker). For a different speaker the adaptive term
     k * noise dominates. Per feature, k is the smallest value that keeps BOTH
     clean human-vs-TTS pairs (the two LibriVox readings vs. the TTS baselines of
     texts 1 and 2 — calibration texts) below 2% of their duration flagged.
  Floors and k interact, so steps 2 and 3 alternate until k is stable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import grounder as G                                   # noqa: E402
from pipeline.analysis import load_bundled, prepare, SPLITS          # noqa: E402
from pipeline.evaluation import clip_metrics, gt_by_feature          # noqa: E402
from pipeline.normalizer import CLARITY_FEATURES                     # noqa: E402

AUDIO_DIR = ROOT / "dataset" / "audio"
LABELS_DIR = ROOT / "dataset" / "labels"
CONFIG_PATH = ROOT / "config.yaml"

CALIBRATION_TEXTS = [t for t, s in SPLITS.items() if s == "calibration"]
LAMBDA_FP = 2.0
MAX_CLEAN_FLAGGED_PER_FEATURE = 0.02
K_GRID = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0]


# Human calibration speaker: the LibriVox reading of text 1 (a calibration text).
# Floors fitted on TTS alone came out too tight for real recordings, whose
# alignment jitters more after an edit. real02 and the orators stay held out.
HUMAN_CALIBRATION = ["real01"]
REAL_DIR = ROOT / "dataset" / "audio" / "real_speakers"


def flawed_clips(text_ids, include_human: bool = True) -> list[dict]:
    """Calibration clips: {tid, clip, label, audio, baseline} for TTS texts and the human set."""
    out = []
    for tid in text_ids:
        for lp in sorted(LABELS_DIR.glob(f"text_{tid:02d}_*.json")):
            if lp.stem.endswith("_features"):
                continue
            out.append({"tid": tid, "clip": lp.stem, "label": json.loads(lp.read_text()),
                        "audio": AUDIO_DIR / f"{lp.stem}.wav", "baseline": AUDIO_DIR / f"text_{tid:02d}_good.wav"})
    if include_human:
        for prefix in HUMAN_CALIBRATION:
            tid = int(prefix[4:])
            if tid not in text_ids:
                continue
            for lp in sorted((LABELS_DIR / "real_speakers").glob(f"{prefix}_*.json")):
                out.append({"tid": tid, "clip": lp.stem, "label": json.loads(lp.read_text()),
                            "audio": REAL_DIR / f"{lp.stem}.wav", "baseline": REAL_DIR / f"{prefix}_good.wav"})
    return out


def save_features(clip_id: str, prep) -> None:
    words = [
        {k: (round(v, 4) if isinstance(v, float) else v) for k, v in w.__dict__.items() if k != "mfcc_mean"}
        for w in prep.norm.words
    ]
    (LABELS_DIR / f"{clip_id}_features.json").write_text(json.dumps({
        "clip_id": clip_id,
        "aligner": prep.alignment.model_used,
        "duration_s": round(prep.features.duration_s, 3),
        "jitter": round(prep.features.jitter, 5),
        "shimmer": round(prep.features.shimmer, 5),
        "words": words,
    }, indent=1))


# -- 1. Clarity feature selection ----------------------------------------------------

def _auc(neg: np.ndarray, pos: np.ndarray) -> float:
    """P(pos < neg): probability a flawed word scores lower than an unflawed one."""
    if len(neg) == 0 or len(pos) == 0:
        return 0.5
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1
    r_pos = ranks[: len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2
    return float(1 - u / (len(pos) * len(neg)))


def select_clarity(cfg: dict, bundled: dict) -> tuple[str, float, dict]:
    """
    Choose the clarity measure that is both SENSITIVE and SPECIFIC. Paired per-word
    deltas of words inside injected clarity regions are compared with
      (a) unflawed words of the same clips, and
      (b) words inside the OTHER flaws' regions (pacing, pitch, volume, pause clips).
    Score = min(AUC_a, AUC_b) after orienting the sign on (a). A measure that also
    moves under slower speech or a level change (e.g. spectral flux falls when speech
    is slower) scores low on (b) and is rejected.
    """
    results = {}
    for feat in CLARITY_FEATURES:
        c = json.loads(json.dumps(cfg))
        c.setdefault("normalization", {}).update({"clarity_feature": feat, "clarity_sign": 1.0})
        inside, outside, other = [], [], []
        for item in flawed_clips(CALIBRATION_TEXTS):
            tid, label = item["tid"], item["label"]
            gt = gt_by_feature(label)
            is_clarity = set(gt) == {"clarity"}
            if not is_clarity and "clarity" in gt:
                continue                         # combined clips mix flaws: skip
            base = prepare(item["baseline"], bundled[tid], c)
            up = prepare(item["audio"], bundled[tid], c)
            pairs = G.pair_words(up.norm.words, base.norm.words)
            d = G._rolling_median(G._deltas_mode_a(up.norm.words, base.norm.words, pairs)["clarity"], 3)
            regs = [r for rr in gt.values() for r in rr]
            for w, x in zip(up.norm.words, d):
                if not np.isfinite(x):
                    continue
                mid = (w.start + w.end) / 2
                in_region = any(r["start_s"] <= mid <= r["end_s"] for r in regs)
                if is_clarity:
                    (inside if in_region else outside).append(x)
                elif in_region:
                    other.append(x)
        auc_clean = _auc(np.array(outside), np.array(inside))
        sign = 1.0 if auc_clean >= 0.5 else -1.0
        oriented = lambda v: np.array(v) * sign
        auc_a = _auc(oriented(outside), oriented(inside))
        auc_b = _auc(oriented(other), oriented(inside))
        results[feat] = {"auc_vs_clean": round(auc_a, 4), "auc_vs_other_flaws": round(auc_b, 4),
                         "score": round(min(auc_a, auc_b), 4), "sign": sign}
        print(f"  clarity candidate {feat:18s} AUC vs clean {auc_a:.3f}  vs other flaws {auc_b:.3f}  "
              f"(sign {sign:+.0f})")
    best = max(results, key=lambda f: results[f]["score"])
    return best, results[best]["sign"], results


# -- 2. Floors --------------------------------------------------------------------------

def collect_deltas(cfg: dict, bundled: dict, text_ids) -> list[dict]:
    g = cfg["grounding"]
    sm, rate_half = int(g.get("smoothing_words", 3)), int(g.get("rate_window_words", 5)) // 2
    items = []
    for item in flawed_clips(text_ids):
        tid, clip_id, label = item["tid"], item["clip"], item["label"]
        base = prepare(item["baseline"], bundled[tid], cfg)
        up = prepare(item["audio"], bundled[tid], cfg)
        pairs = G.pair_words(up.norm.words, base.norm.words)
        raw = G._deltas_mode_a(up.norm.words, base.norm.words, pairs, rate_half, g.get("floors"))
        prep = {f: G.prepared_delta(raw[f], f, sm) for f in G.FLAW_TYPES}
        same = G.recording_match(up.norm.words, base.norm.words, pairs) < float(g.get("same_recording_max_dist", 20.0))
        items.append({"clip": clip_id, "tid": tid, "group": "human" if clip_id.startswith("real") else "tts",
                      "label": label, "up": up, "base": base, "pairs": pairs, "same": same,
                      "d": {f: v[0] for f, v in prep.items()}, "noise": {f: v[1] for f, v in prep.items()},
                      "dur": up.features.duration_s})
    return items


def feature_objective(items, feat: str, floor: float, k: float, g: dict, cfg: dict | None = None
                      ) -> tuple[float, float, float]:
    """
    Objective for one feature's floor, computed with the FULL grounding (edge
    refinement, long-scale detector, pause gating) so the calibrated system is the
    evaluated one. TTS and human clips are scored as two groups and weighted
    equally, so the much larger TTS set cannot dictate floors that are too tight
    for real recordings.
    """
    from pipeline.evaluation import temporal_iou, false_positive_seconds
    c = json.loads(json.dumps(cfg))
    c["grounding"]["floors"] = {**c["grounding"]["floors"], feat: float(floor)}
    groups: dict[str, tuple[list, list]] = {}
    for it in items:
        res = G.ground(it["up"].norm, "A", it["tid"], "calibration", baseline_norm_clip=it["base"].norm, cfg=c)
        regs = [r for r in res.flaw_regions if r.flaw_type == feat]
        gt = gt_by_feature(it["label"])
        ious, fps = groups.setdefault(it["group"], ([], []))
        if feat in gt:
            ious.append(temporal_iou(regs, gt[feat], it["dur"]))
        fps.append(false_positive_seconds(regs, gt, it["dur"]) / it["dur"])
    per = [(float(np.mean(i)) if i else 0.0, float(np.mean(f))) for i, f in groups.values()]
    mi, mf = float(np.mean([x[0] for x in per])), float(np.mean([x[1] for x in per]))
    return mi - LAMBDA_FP * mf, mi, mf


def calibrate_floors(items, g: dict, cfg: dict) -> dict:
    floors = {}
    for feat in G.FLAW_TYPES:
        clean = np.concatenate([
            np.abs(it["d"][feat][np.isfinite(it["d"][feat])]) for it in items
            if feat not in gt_by_feature(it["label"])
        ])
        grid = np.unique(np.round(np.quantile(clean, np.linspace(0.80, 0.9995, 25)), 4))
        grid = grid[grid > 0]
        scored = [(feature_objective(items, feat, fl, G.k_for(g, feat), g, cfg), fl) for fl in grid]
        (obj, iou, fp), best = max(scored, key=lambda s: s[0][0])
        floors[feat] = float(best)
        print(f"  floor {feat:8s} = {best:.4f}  (target IoU {iou:.3f}, FP fraction {fp:.3f})")
    return floors


# -- 4. Mode B floors -------------------------------------------------------------------

MODE_B_GRIDS = {
    "pitch": np.round(np.arange(0.4, 2.01, 0.1), 2),
    "energy": np.arange(4.0, 16.1, 1.0),
    "rate": np.round(np.arange(0.2, 1.21, 0.05), 2),
    "pause": np.round(np.arange(0.5, 2.01, 0.1), 2),
    "clarity": np.round(np.arange(2.0, 12.1, 0.5), 2),
}


def calibrate_floors_mode_b(cfg: dict, bundled: dict) -> dict:
    """Mode B floors on the calibration texts (flawed clips and clean baselines, no reference)."""
    from pipeline.evaluation import temporal_iou, false_positive_seconds
    clips = [(AUDIO_DIR / f"text_{t:02d}_good.wav", {"flaw_regions": []}, t) for t in CALIBRATION_TEXTS]
    clips += [(REAL_DIR / f"{p}_good.wav", {"flaw_regions": []}, int(p[4:])) for p in HUMAN_CALIBRATION]
    clips += [(it["audio"], it["label"], it["tid"]) for it in flawed_clips(CALIBRATION_TEXTS)]
    norms = [(prepare(w, bundled[t], cfg), lab) for w, lab, t in clips]
    floors = dict(cfg["grounding"].get("floors_mode_b", {}))
    grids = dict(MODE_B_GRIDS)
    clean = np.concatenate([[w.clarity for w in prep.norm.words] for prep, lab in norms
                            if not lab["flaw_regions"]])
    spread = float(np.percentile(np.abs(clean - np.median(clean)), 75)) or 1.0
    grids["clarity"] = np.round(spread * np.arange(1.0, 8.01, 0.5), 4)   # scale of the chosen measure
    for feat in G.FLAW_TYPES:
        best = None
        for fl in grids[feat]:
            c = json.loads(json.dumps(cfg))
            c["grounding"]["floors_mode_b"] = {**floors, feat: float(fl)}
            ious, fps = [], []
            for prep, lab in norms:
                regs = [r for r in G.ground(prep.norm, "B", None, None, cfg=c).flaw_regions if r.flaw_type == feat]
                gt = gt_by_feature(lab)
                if feat in gt:
                    ious.append(temporal_iou(regs, gt[feat], prep.features.duration_s))
                fps.append(false_positive_seconds(regs, gt, prep.features.duration_s) / prep.features.duration_s)
            obj = (np.mean(ious) if ious else 0.0) - LAMBDA_FP * np.mean(fps)
            if best is None or obj > best[0]:
                best = (obj, float(fl), float(np.mean(ious)) if ious else 0.0, float(np.mean(fps)))
        floors[feat] = best[1]
        print(f"  Mode B floor {feat:8s} = {best[1]:.2f}  (target IoU {best[2]:.3f}, FP fraction {best[3]:.3f})")
    return floors


# -- 0. Same-recording gate -----------------------------------------------------------

def calibrate_recording_gate(cfg: dict, bundled: dict) -> float:
    """Geometric midpoint between edited copies (calibration texts) and a different recording (real01)."""
    edited = []
    for item in flawed_clips(CALIBRATION_TEXTS):
        base = prepare(item["baseline"], bundled[item["tid"]], cfg)
        up = prepare(item["audio"], bundled[item["tid"]], cfg)
        edited.append(G.recording_match(up.norm.words, base.norm.words, G.pair_words(up.norm.words, base.norm.words)))
    base = prepare(AUDIO_DIR / "text_01_good.wav", bundled[1], cfg)
    up = prepare(ROOT / "dataset" / "audio" / "real_speakers" / "real01_good.wav", bundled[1], cfg)
    other = G.recording_match(up.norm.words, base.norm.words, G.pair_words(up.norm.words, base.norm.words))
    hi = max(edited)
    gate = float(np.sqrt(max(hi, 1e-3) * other))
    print(f"  edited copies: max timbre distance {hi:.2f}; different recording: {other:.2f} -> gate {gate:.2f}")
    if other <= hi:
        print("  WARNING: no separation between edited copies and a different recording")
    return round(gate, 2)


# -- 3. k_noise from a clean cross-speaker pair --------------------------------------

REAL_TRANSCRIPT = lambda tid: (ROOT / "dataset" / "transcripts" / f"real{tid:02d}.txt")


def calibrate_k(cfg: dict, bundled: dict) -> dict | None:
    """
    Per feature: smallest k keeping EVERY clean human-vs-TTS pair (both LibriVox
    readings) under MAX_CLEAN_FLAGGED_PER_FEATURE of its duration. With only two
    human recordings, fitting one of them under-estimated the variation of the other
    (a noisier 64 kbps recording), so both are used.
    """
    pairs = []
    for tid in (1, 2):
        real = ROOT / "dataset" / "audio" / "real_speakers" / f"real{tid:02d}_good.wav"
        if not real.exists():
            continue
        tx = REAL_TRANSCRIPT(tid).read_text(encoding="utf-8") if REAL_TRANSCRIPT(tid).exists() else bundled[tid]
        pairs.append((tid, prepare(AUDIO_DIR / f"text_{tid:02d}_good.wav", bundled[tid], cfg), prepare(real, tx, cfg)))
    if not pairs:
        print("  human recordings missing — run scripts/prepare_real_speakers.py; keeping k_noise")
        return None
    chosen = {}
    for feat in G.FLAW_TYPES:
        for k in K_GRID:
            c = json.loads(json.dumps(cfg))
            c["grounding"]["k_noise"] = {**{f: K_GRID[-1] for f in G.FLAW_TYPES}, feat: k}
            fracs = []
            for tid, base, up in pairs:
                res = G.ground(up.norm, "A", tid, "calibration", baseline_norm_clip=base.norm, cfg=c)
                fracs.append(sum(r.end_s - r.start_s for r in res.flaw_regions if r.flaw_type == feat)
                             / up.features.duration_s)
            if max(fracs) <= MAX_CLEAN_FLAGGED_PER_FEATURE:
                break
        chosen[feat] = k
        print(f"  k[{feat:7s}] = {k:.1f}  (clean human-vs-TTS pairs flagged: "
              f"{', '.join(f'{x:.1%}' for x in fracs)})")

    # Long-scale (15-word) detector: smallest multiplier keeping the same budget
    # with both scales active.
    cfg["grounding"]["k_noise"] = chosen
    k_long = {}
    for feat in G.LONG_SCALE_FEATURES:
        for kl in K_GRID:
            c = json.loads(json.dumps(cfg))
            c["grounding"]["k_long"] = {**{f: K_GRID[-1] * 3 for f in G.LONG_SCALE_FEATURES}, feat: kl}
            fracs = []
            for tid, base, up in pairs:
                res = G.ground(up.norm, "A", tid, "calibration", baseline_norm_clip=base.norm, cfg=c)
                fracs.append(sum(r.end_s - r.start_s for r in res.flaw_regions if r.flaw_type == feat)
                             / up.features.duration_s)
            if max(fracs) <= MAX_CLEAN_FLAGGED_PER_FEATURE:
                break
        k_long[feat] = kl
        print(f"  k_long[{feat:7s}] = {kl:.1f}  (both scales on clean pairs: {', '.join(f'{x:.1%}' for x in fracs)})")
    cfg["grounding"]["k_long"] = k_long
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--no-save", action="store_true", help="skip writing *_features.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    bundled, _ = load_bundled()

    if args.calibrate:
        print("1. Clarity measure (calibration texts)")
        feat, sign, aucs = select_clarity(cfg, bundled)
        cfg["normalization"].update({"clarity_feature": feat, "clarity_sign": sign})
        print(f"  -> {feat} (sign {sign:+.0f})")

        print("0. Same-recording gate")
        cfg["grounding"]["same_recording_max_dist"] = calibrate_recording_gate(cfg, bundled)

        # Floors and k interact (tau = max(floor, k * noise)): alternate until k is stable.
        items = collect_deltas(cfg, bundled, CALIBRATION_TEXTS)
        g = cfg["grounding"]
        for rnd in range(1, 4):
            print(f"2.{rnd} Floors (calibration texts, all tiers) with k_noise={g.get('k_noise', 3.0)}")
            g["floors"] = {f: round(v, 4) for f, v in calibrate_floors(items, g, cfg).items()}
            print(f"3.{rnd} k_noise (clean cross-speaker pair, text 1)")
            k = calibrate_k(cfg, bundled)
            if k is None or k == g.get("k_noise"):
                break
            g["k_noise"] = k

        print("4. Mode B floors (calibration texts)")
        g["floors_mode_b"] = calibrate_floors_mode_b(cfg, bundled)

        header = ("# Thresholds under grounding.floors / k_noise / normalization.clarity_feature are\n"
                  "# written by: python scripts/04_extract_features.py --calibrate  (calibration texts only)\n")
        CONFIG_PATH.write_text(header + yaml.safe_dump(cfg, sort_keys=False))
        (LABELS_DIR / "calibration_report.json").write_text(json.dumps({
            "calibration_texts": CALIBRATION_TEXTS, "clarity_auc": aucs,
            "clarity_feature": feat, "clarity_sign": sign,
            "floors": g["floors"], "k_noise": g["k_noise"], "lambda_fp": LAMBDA_FP,
            "same_recording_max_dist": g.get("same_recording_max_dist"),
            "k_long": g.get("k_long"), "floors_mode_b": g.get("floors_mode_b"),
            "k_same_recording": g.get("k_same_recording", 3.0),
            "max_clean_flagged_per_feature": MAX_CLEAN_FLAGGED_PER_FEATURE,
        }, indent=2))
        print("config.yaml updated; report -> dataset/labels/calibration_report.json")

    if not args.no_save:
        n = 0
        for tid in sorted(bundled):
            for wav in sorted(AUDIO_DIR.glob(f"text_{tid:02d}_*.wav")):
                save_features(wav.stem, prepare(wav, bundled[tid], cfg))
                n += 1
        print(f"Saved features for {n} clips.")


if __name__ == "__main__":
    main()
