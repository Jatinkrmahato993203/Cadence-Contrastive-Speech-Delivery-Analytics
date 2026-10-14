"""
tests/test_reproducibility.py
-----------------------------
Determinism and configuration sanity (§4, §6 Phase 7).
"""

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.grounder import FLAW_TYPES, ground  # noqa: E402
from pipeline.normalizer import CLARITY_FEATURES, NormalizedClip, NormalizedWord, SpeakerProfile  # noqa: E402
from pipeline.scorer import score  # noqa: E402

CFG = yaml.safe_load((ROOT / "config.yaml").read_text())


def _clip(seed: int) -> NormalizedClip:
    rng = np.random.default_rng(seed)
    words, t = [], 0.0
    for i in range(60):
        dur = 0.25 + 0.1 * rng.random()
        words.append(NormalizedWord(i, f"w{i}", t, t + dur, pitch_var_st=2 + rng.random(),
                                    energy_db_rel=-8 + 4 * rng.random(), rate_wps=1 / dur,
                                    pause_duration_s=0.05 + 0.3 * rng.random() * (i % 7 == 0),
                                    clarity=rng.normal(), duration_s=dur))
        t += dur + 0.05
    return NormalizedClip("x.wav", SpeakerProfile("s"), words, [], [], [], duration_s=t)


def _run() -> dict:
    base, up = _clip(0), _clip(1)
    res = ground(up, "A", 1, "calibration", baseline_norm_clip=base, cfg=CFG)
    rub = score(res, up.duration_s)
    return {"regions": [(r.flaw_type, r.start_s, r.end_s, round(r.mean_delta, 9)) for r in res.flaw_regions],
            "overall": round(rub.overall, 9)}


def test_pipeline_is_deterministic():
    assert json.dumps(_run()) == json.dumps(_run())


def test_config_thresholds_valid():
    g = CFG["grounding"]
    for key in ("floors", "floors_mode_b"):
        assert set(g[key]) == set(FLAW_TYPES), f"{key} must define every flaw type"
        assert all(isinstance(v, (int, float)) and v > 0 for v in g[key].values()), key
    k = g["k_noise"]
    ks = k.values() if isinstance(k, dict) else [k]
    assert all(isinstance(v, (int, float)) and v > 0 for v in ks)
    if isinstance(k, dict):
        assert set(k) == set(FLAW_TYPES)
    assert CFG["normalization"]["clarity_feature"] in CLARITY_FEATURES
    assert CFG["normalization"].get("clarity_sign", 1.0) in (1.0, -1.0)
