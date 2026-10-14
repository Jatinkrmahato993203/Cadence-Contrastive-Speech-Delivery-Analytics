"""
pipeline/analysis.py
--------------------
Single entry point for the full pipeline, shared by the dashboard, the
calibration / evaluation scripts and the tests, so they cannot drift apart:

    align -> extract features -> normalize -> detect mode -> ground -> explain -> score
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

ROOT = Path(__file__).parent.parent
DATASET = ROOT / "dataset"
SPLITS = {1: "calibration", 2: "calibration", 3: "calibration", 4: "calibration",
          5: "calibration", 6: "heldout", 7: "calibration", 8: "heldout"}


def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def load_bundled() -> tuple[dict[int, str], dict[int, str]]:
    transcripts = {
        tid: (DATASET / "transcripts" / f"text_{tid:02d}.txt").read_text(encoding="utf-8").strip()
        for tid in SPLITS
        if (DATASET / "transcripts" / f"text_{tid:02d}.txt").exists()
    }
    return transcripts, dict(SPLITS)


def baseline_path(text_id: int) -> Path:
    return DATASET / "audio" / f"text_{text_id:02d}_good.wav"


@dataclass
class Prepared:
    alignment: object
    features: object
    norm: object


def _cfg_key(cfg: dict) -> str:
    return yaml.safe_dump({"a": cfg.get("aligner", {}), "n": cfg.get("normalization", {}),
                           "p": cfg.get("grounding", {}).get("pause_gap_threshold_s")}, sort_keys=True)


@lru_cache(maxsize=512)
def _prepare_cached(audio_path: str, transcript: str, mtime: float, cfg_yaml: str) -> Prepared:
    from pipeline.aligner import align
    from pipeline import features as feat_mod
    from pipeline import normalizer as norm_mod

    cfg = yaml.safe_load(cfg_yaml)
    alignment = align(audio_path, transcript=transcript, cfg=cfg)
    feats = feat_mod.extract(audio_path, alignment)
    norm, _ = norm_mod.normalize(feats, Path(audio_path).stem, cfg=cfg)
    return Prepared(alignment, feats, norm)


def prepare(audio_path: str | Path, transcript: str, cfg: dict) -> Prepared:
    p = Path(audio_path)
    return _prepare_cached(str(p), transcript, p.stat().st_mtime, yaml.safe_dump(cfg, sort_keys=True))


@dataclass
class AnalysisResult:
    mode: str
    matched_id: Optional[int]
    matched_split: Optional[str]
    duration_s: float
    upload: Prepared
    baseline: Optional[Prepared]
    grounding: object
    explanations: list
    rubric: object


def analyze(
    audio_path: str | Path,
    transcript: str,
    cfg: Optional[dict] = None,
    use_llm: bool = False,
    force_mode_b: bool = False,
    baseline_override: Optional[str | Path] = None,
) -> AnalysisResult:
    """
    Run the whole pipeline on one recording. *baseline_override* replaces the
    bundled GOOD baseline (e.g. a same-speaker reference recording).
    """
    from pipeline.grounder import detect_mode, ground
    from pipeline.explainer import explain
    from pipeline.scorer import score

    cfg = cfg or load_config()
    bundled, splits = load_bundled()
    mode, matched_id, matched_split = ("B", None, None) if force_mode_b else detect_mode(transcript, bundled, splits)

    upload = prepare(audio_path, transcript, cfg)
    base = None
    if mode == "A":
        bp = Path(baseline_override) if baseline_override else baseline_path(matched_id)
        if bp.exists():
            # A same-speaker reference reads the same words as the upload; the bundled
            # baseline reads the bundled transcript.
            base = prepare(bp, transcript if baseline_override else bundled[matched_id], cfg)
        else:
            mode, matched_id, matched_split = "B", None, None

    grounding = ground(
        upload.norm, mode=mode, matched_text_id=matched_id, matched_split=matched_split,
        baseline_norm_clip=base.norm if base else None, cfg=cfg,
    )
    explanations = explain(grounding, upload.norm.words, use_llm=use_llm,
                           baseline_words=base.norm.words if base else None)
    rubric = score(grounding, total_duration_s=upload.features.duration_s)
    return AnalysisResult(mode, matched_id, matched_split, upload.features.duration_s,
                          upload, base, grounding, explanations, rubric)
