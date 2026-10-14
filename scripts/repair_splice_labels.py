"""
scripts/repair_splice_labels.py
-------------------------------
One-off migration for labels written before the pause-label fix in
03_generate_bad_clips.py.

The old apply_pause() labelled every inserted silence at its *source* position,
ignoring the displacement caused by silences inserted earlier in the clip, and
apply_combined_egregious() never shifted its volume/pitch/clarity regions for
the pauses spliced in afterwards. Both affect only clips with pause splices:
  * *_pause_* clips with more than one silence
  * *_combined_egregious clips

The audio is unchanged; this rewrites the labels onto the output timeline and
verifies that every repaired pause window is digital silence in the WAV.
Idempotent: repaired labels carry "label_timeline": "output".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).parent.parent
LABELS_DIR = ROOT / "dataset" / "labels"
AUDIO_DIR = ROOT / "dataset" / "audio"


def _is_pause(r: dict) -> bool:
    return r.get("mechanism", "").startswith("silence_splice")


def repair(label: dict) -> dict:
    regions = label["flaw_regions"]
    pauses = sorted([r for r in regions if _is_pause(r)], key=lambda r: r["start_s"])
    others = [r for r in regions if not _is_pause(r)]

    splices = [(p["start_s"], p["end_s"] - p["start_s"]) for p in pauses]   # source pos, duration

    new_pauses, offset = [], 0.0
    for (src, dur), p in zip(splices, pauses):
        new_pauses.append({**p, "start_s": round(src + offset, 3), "end_s": round(src + offset + dur, 3)})
        offset += dur

    def shift(t: float, is_end: bool) -> float:
        return t + sum(d for src, d in splices if (src < t if is_end else src <= t))

    new_others = [
        {**r, "start_s": round(shift(r["start_s"], False), 3), "end_s": round(shift(r["end_s"], True), 3)}
        for r in others
    ]
    label["flaw_regions"] = sorted(new_others + new_pauses, key=lambda r: r["start_s"])
    label["label_timeline"] = "output"
    return label


def silent_fraction(wav: Path, start: float, end: float) -> float:
    y, sr = sf.read(str(wav), dtype="float32")
    seg = y[int(start * sr) + 5: int(end * sr) - 5]
    return float(np.mean(np.abs(seg) < 1e-3)) if len(seg) else 0.0


def main() -> int:
    changed, bad = 0, []
    for path in sorted(LABELS_DIR.glob("text_*.json")):
        if path.stem.endswith("_features"):
            continue
        label = json.loads(path.read_text())
        n_pause = sum(_is_pause(r) for r in label.get("flaw_regions", []))
        needs = n_pause > 1 or (n_pause >= 1 and len(label["flaw_regions"]) > n_pause)
        if not needs or label.get("label_timeline") == "output":
            continue
        label = repair(label)
        wav = AUDIO_DIR / f"{label['clip_id']}.wav"
        for r in label["flaw_regions"]:
            if _is_pause(r):
                frac = silent_fraction(wav, r["start_s"], r["end_s"])
                if frac < 0.98:
                    bad.append((label["clip_id"], r["start_s"], r["end_s"], round(frac, 3)))
        path.write_text(json.dumps(label, indent=2))
        changed += 1
        print(f"repaired {label['clip_id']}: {[(r['start_s'], r['end_s']) for r in label['flaw_regions']]}")

    print(f"\n{changed} labels repaired.")
    if bad:
        print("Pause windows that are NOT silent after repair:")
        for b in bad:
            print("  ", b)
        return 1
    print("All repaired pause windows verified as digital silence.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
