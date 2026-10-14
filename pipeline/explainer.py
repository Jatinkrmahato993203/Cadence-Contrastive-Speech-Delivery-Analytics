"""
pipeline/explainer.py
---------------------
Causal explanation engine (§3.2).

Each flaw region becomes a structured record and a sentence built ONLY from
numbers measured by the pipeline:

    WHAT   — the feature, direction and size of the deviation in natural units
             (words/s, semitones, dB, seconds), with the paired baseline value
    WHERE  — time span and the words it covers
    WHY    — the acoustic mechanism that produces the measured delta, chosen
             by (feature, direction, severity) — e.g. a tempo drop with pauses
             excluded means words were drawn out, not that silence was added
    EFFECT — the expected effect on the listener

Severity = |delta| / threshold for that feature: < 1.5 mild, < 3 moderate, else severe.

Optional LLM polish (OPENAI_API_KEY + use_llm=True) rewrites the sentence for
fluency only, at temperature 0, with an instruction not to change any numbers;
on any failure the deterministic sentence is used.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

CLARITY_LABELS = {
    "hnr_db": ("harmonics-to-noise ratio", "dB"),
    "spectral_contrast": ("spectral contrast", "dB"),
    "spectral_flux": ("spectral flux (articulatory movement)", ""),
    "hf_ratio_db": ("high-frequency (>3 kHz) energy", "dB"),
    "spectral_flatness": ("spectral flatness", ""),
}


@dataclass
class Explanation:
    flaw_type: str
    region_start_s: float
    region_end_s: float
    region_words: str
    template_text: str
    polished_text: str = ""
    severity_label: str = ""
    evidence: dict = field(default_factory=dict)


def _severity_label(sev: float) -> str:
    return "mild" if sev < 1.5 else ("moderate" if sev < 3.0 else "severe")


def _format_region_words(words, region) -> str:
    idx = region.word_indices
    if not idx:
        return "(unknown span)"
    span = [w.word for w in words[idx[0]: idx[-1] + 1]]
    if region.flaw_type == "pause":
        before = words[idx[0] - 1].word if idx[0] > 0 else "(start)"
        return f"{before} … {span[0]}"
    if len(span) > 6:
        span = span[:3] + ["…"] + span[-2:]
    return " ".join(span)


def _pct(u: float, b: float) -> float:
    return (u - b) / b * 100.0 if b and math.isfinite(b) else float("nan")


def _sentence(region, words, mode: str, clarity_feature: str) -> tuple[str, dict]:
    ft, dirn = region.flaw_type, region.direction
    sev = _severity_label(region.severity)
    ref = "the paired GOOD baseline" if mode == "A" else "this speaker's own typical delivery"
    where = f"{region.start_s:.1f}–{region.end_s:.1f} s, '{_format_region_words(words, region)}'"
    u, b = region.uploaded_mean, region.baseline_mean
    has_b = mode == "A" and math.isfinite(b)
    ev = {"feature": ft, "direction": dirn, "severity": round(region.severity, 2),
          "uploaded": round(u, 3), "baseline": round(b, 3) if has_b else None,
          "delta": round(region.mean_delta, 3), "threshold": round(region.threshold, 3)}

    if ft == "rate":
        factor = 2 ** region.mean_delta                 # duration ratio, >= 1
        cmp = f"{u:.1f} vs {b:.1f} words per second of speech" if has_b else f"{u:.1f} words per second of speech"
        if dirn == "slow":
            what = f"Articulation was {(1 - 1 / factor) * 100:.0f}% slower than {ref} ({cmp})"
            why = (f"each word was held about {factor:.2f}× as long; pauses are excluded from this measure, "
                   "so the words themselves were drawn out rather than silence being added")
            effect = "stressed syllables lose contrast and the passage drags, inviting the audience to drift"
        else:
            what = f"Articulation was {(factor - 1) * 100:.0f}% faster than {ref} ({cmp})"
            why = (f"words were compressed to about {1 / factor:.2f}× their reference duration, shortening vowels "
                   "and the transitions between them")
            effect = "listeners get less time to process each idea and emphasis is flattened"
    elif ft == "pitch":
        ratio = 2 ** region.mean_delta                  # variability ratio, >= 1
        cmp = f"{u:.1f} vs {b:.1f} semitones of local pitch movement" if has_b else f"{u:.1f} semitones of local pitch movement"
        if dirn == "flat":
            what = f"Intonation was {(1 - 1 / ratio) * 100:.0f}% flatter than {ref} ({cmp})"
            why = "the pitch contour was compressed toward the speaker's median, so stressed words no longer rise above their neighbours"
            effect = "the passage sounds monotone and key words lose their emphasis"
        else:
            what = f"Pitch movement was {ratio:.1f}× wider than {ref} ({cmp})"
            why = "pitch swings exceed the reference contour on unstressed as well as stressed words"
            effect = "emphasis becomes indiscriminate and can sound theatrical or anxious"
    elif ft == "energy":
        cmp = f"{u:+.1f} dB vs {b:+.1f} dB relative to each speaker's loud speech" if has_b else f"{u:+.1f} dB relative to loud speech"
        if dirn == "quiet":
            what = f"Volume dropped {region.mean_delta:.1f} dB below {ref} ({cmp})"
            why = "vocal projection fell away in this span while the rest of the delivery kept its level"
            effect = "the passage will sound distant and is easily lost in room noise"
        else:
            what = f"Volume rose {region.mean_delta:.1f} dB above {ref} ({cmp})"
            why = "vocal effort increased sharply relative to the surrounding delivery"
            effect = "the spike sounds forced and can fatigue listeners"
    elif ft == "pause":
        if dirn == "present":
            base_txt = f" where {ref} paused {b:.2f} s" if has_b else ""
            what = f"An unplanned {u:.2f} s silence{base_txt}"
            why = (f"the phrase was interrupted for {region.mean_delta:.2f} s longer than the reference, "
                   "breaking the link between the words on either side")
            effect = "the audience hears hesitation, and the thought loses momentum"
        else:
            what = f"A {b:.2f} s pause in {ref} was cut to {u:.2f} s"
            why = "a breath / clause boundary was run through without a break"
            effect = "phrases blur together and the listener loses the sentence structure"
    else:  # clarity
        label, unit = CLARITY_LABELS.get(clarity_feature, (clarity_feature, ""))
        unit_txt = f" {unit}" if unit else ""
        if dirn == "degraded":
            what = f"Articulatory clarity fell: {label} was {region.mean_delta:.2f}{unit_txt} below {ref}"
            why = ("spectral detail was smeared — formant peaks and consonant transitions were softened, "
                   "the acoustic signature of mumbled or slurred articulation")
            effect = "consonants become hard to distinguish and intelligibility drops"
        else:
            what = f"Articulation was crisper than {ref}: {label} {region.mean_delta:.2f}{unit_txt} higher"
            why = "consonant transitions are sharper than the reference"
            effect = "usually not a problem; flagged for completeness"

    text = f"[{sev}] {what} at {where}. Cause: {why}. Effect: {effect}."
    if region.uncertain:
        text += " (Alignment confidence is low in this span; treat the boundaries as approximate.)"
    return text, ev


def _llm_polish(text: str, evidence: dict) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return text
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        prompt = (
            "Rewrite this speech-coaching explanation for fluency and concision. "
            "Do NOT change, add or drop any number, time, word quote or causal claim. "
            "Return only the rewritten text.\n\n"
            f"Measured evidence (JSON): {json.dumps(evidence)}\n\nText: {text}"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=220,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("LLM polish failed (%s) — using deterministic text", exc)
        return text


def explain(grounding_result, uploaded_words, use_llm: bool = False, baseline_words=None,
            clarity_feature: str | None = None) -> list[Explanation]:
    """One explanation per flaw region, in time order."""
    if clarity_feature is None:
        try:
            from pipeline.analysis import load_config
            clarity_feature = load_config().get("normalization", {}).get("clarity_feature", "hnr_db")
        except Exception:
            clarity_feature = "hnr_db"

    out: list[Explanation] = []
    for region in sorted(grounding_result.flaw_regions, key=lambda r: r.start_s):
        text, ev = _sentence(region, uploaded_words, grounding_result.mode, clarity_feature)
        out.append(Explanation(
            flaw_type=region.flaw_type,
            region_start_s=region.start_s,
            region_end_s=region.end_s,
            region_words=_format_region_words(uploaded_words, region),
            template_text=text,
            polished_text=_llm_polish(text, ev) if use_llm else text,
            severity_label=_severity_label(region.severity),
            evidence=ev,
        ))
    return out
