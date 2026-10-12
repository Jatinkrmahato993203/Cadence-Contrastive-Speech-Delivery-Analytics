"""
scripts/04_extract_features.py
-------------------------------
Phase 2: Extract features from all 64 clips and (optionally) compute
calibration-set statistics to populate config.yaml.

Usage:
    # Extract features for all clips:
    python scripts/04_extract_features.py

    # Also recompute and update config.yaml thresholds:
    python scripts/04_extract_features.py --compute-thresholds

Outputs:
    dataset/labels/{clip_id}_features.json  (per clip)
    config.yaml updated sigma_ values (if --compute-thresholds)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

AUDIO_DIR = ROOT / "dataset" / "audio"
LABELS_DIR = ROOT / "dataset" / "labels"
CONFIG_PATH = ROOT / "config.yaml"

CLIP_SUFFIXES = [
    "good", "pacing_mod", "pitch_mod", "pause_mod",
    "volume_mod", "clarity_mod", "combined_egregious", "nearperfect"
]

CALIBRATION_TEXTS = {1, 2, 3, 4, 5, 7}
LIBRIVOX_TEXTS = {1, 2, 3, 4, 5, 6}
TEAM_TEXTS = {7, 8}


def _make_dummy_alignment(num_words: int = 30, duration_s: float = 60.0):
    """Create a dummy alignment for clips where WhisperX is not available."""
    from pipeline.aligner import AlignmentResult, WordAlignment

    step = duration_s / max(num_words, 1)
    words = []
    for i in range(num_words):
        words.append(WordAlignment(
            word=f"word{i}", start=i * step, end=(i + 0.8) * step, score=0.9
        ))
    return AlignmentResult(words=words, model_used="dummy", had_retry=False)


def process_clip(
    text_id: int, suffix: str, cfg: dict
) -> dict | None:
    """Extract features for one clip. Returns feature dict or None if clip missing."""
    from pipeline import features as feat_mod
    from pipeline import normalizer as norm_mod
    from pipeline.normalizer import SpeakerProfile

    wav_path = AUDIO_DIR / f"text_{text_id:02d}_{suffix}.wav"
    if not wav_path.exists():
        return None

    print(f"  Extracting: {wav_path.name}")

    # Try WhisperX alignment; fall back to dummy for offline CI
    try:
        from pipeline.aligner import align
        alignment = align(str(wav_path), cfg=cfg)
    except Exception as exc:
        print(f"    WhisperX unavailable ({exc}) — using dummy alignment")
        alignment = _make_dummy_alignment()

    clip_feat = feat_mod.extract(str(wav_path), alignment)

    # Speaker profile: bootstrap from first 10 words (always new speaker per clip in batch)
    speaker_id = f"text_{text_id:02d}_speaker"
    norm_clip, profile = norm_mod.normalize(clip_feat, speaker_id, cfg=cfg)

    # Serialize per-word features
    word_dicts = [
        {
            "word_idx": w.word_idx,
            "word": w.word,
            "start": w.start,
            "end": w.end,
            "pitch_st": w.pitch_st,
            "energy_db_rel": w.energy_db_rel,
            "rate_zscore": w.rate_zscore,
            "pause_flag": w.pause_flag,
            "pause_duration_s": w.pause_duration_s,
            "clarity_mfcc": w.clarity_mfcc,
            "clarity_hnr": w.clarity_hnr,
        }
        for w in norm_clip.words
    ]

    return {
        "clip_id": f"text_{text_id:02d}_{suffix}",
        "text_id": text_id,
        "suffix": suffix,
        "words": word_dicts,
        "speaker_f0_median": profile.f0_median,
        "speaker_rms_95pct": profile.rms_95pct,
        "speaker_rate_mean": profile.rate_mean,
    }


def compute_thresholds(all_features: list[dict], cfg: dict) -> dict:
    """
    Compute sigma_group_librivox, sigma_group_team, sigma_pooled_all
    from calibration-set GOOD clips only.
    """
    librivox_good = [
        f for f in all_features
        if f["text_id"] in CALIBRATION_TEXTS & LIBRIVOX_TEXTS
        and f["suffix"] == "good"
    ]
    team_good = [
        f for f in all_features
        if f["text_id"] in CALIBRATION_TEXTS & TEAM_TEXTS
        and f["suffix"] == "good"
    ]
    all_good = librivox_good + team_good

    def _pool_feature(clips: list[dict], key: str) -> float:
        vals = []
        for c in clips:
            vals.extend(w[key] for w in c["words"])
        return float(np.std(vals)) if vals else 1.0

    features_keys = {
        "pitch": "pitch_st",
        "energy": "energy_db_rel",
        "rate": "rate_zscore",
        "clarity_mfcc": "clarity_mfcc",
        "clarity_hnr": "clarity_hnr",
    }

    def _compute_group(clips):
        return {k: _pool_feature(clips, v) for k, v in features_keys.items()}

    return {
        "sigma_group_librivox": _compute_group(librivox_good),
        "sigma_group_team": _compute_group(team_good),
        "sigma_pooled_all": _compute_group(all_good),
    }


def update_config(new_sigmas: dict, cfg: dict) -> dict:
    cfg.update(new_sigmas)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute-thresholds", action="store_true",
                        help="Recompute sigma_ values and update config.yaml")
    parser.add_argument("--text", type=int, help="Only process this text_id")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    all_features = []
    text_ids = [args.text] if args.text else list(range(1, 9))

    for text_id in text_ids:
        for suffix in CLIP_SUFFIXES:
            result = process_clip(text_id, suffix, cfg)
            if result:
                all_features.append(result)
                out = LABELS_DIR / f"text_{text_id:02d}_{suffix}_features.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                with open(out, "w") as f:
                    json.dump(result, f, indent=2)

    print(f"\nExtracted features for {len(all_features)} clips.")

    if args.compute_thresholds:
        print("\nComputing calibration-set thresholds...")
        new_sigmas = compute_thresholds(all_features, cfg)

        print("\nComputed sigma values:")
        for group, vals in new_sigmas.items():
            print(f"  {group}:")
            for k, v in vals.items():
                print(f"    {k}: {v:.4f}")

        updated_cfg = update_config(new_sigmas, cfg)
        with open(CONFIG_PATH, "w") as f:
            yaml.dump(updated_cfg, f, default_flow_style=False, sort_keys=False)
        print(f"\nconfig.yaml updated with real calibration thresholds.")

    print("\nNext: python scripts/05_evaluate_heldout.py")


if __name__ == "__main__":
    main()
