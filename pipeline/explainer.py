"""
pipeline/explainer.py
---------------------
Causal explanation engine (§3.2).

Deterministic core: template engine maps each (feature, direction, magnitude) triple
to a structured explanation string built ENTIRELY from numeric deltas from the pipeline.
No free-text LLM guessing.

Optional LLM polish: if OPENAI_API_KEY env var is set, passes the structured JSON to
an LLM at temperature=0 for prose smoothing only. If the call fails or is absent,
the template string is shown unchanged.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# -- Explanation record ---------------------------------------------------------

@dataclass
class Explanation:
    flaw_type: str
    region_start_s: float
    region_end_s: float
    region_words: str          # comma-joined word span label
    template_text: str         # deterministic explanation
    polished_text: str = ""    # LLM-polished (if available), else same as template


# -- Templates (one per flaw category) ----------------------------------------

_TEMPLATES = {
    "rate": (
        "Speech rate was {pct_change:.0f}% {direction_label} than {baseline_label} "
        "({uploaded_val:.1f} words/s vs. {baseline_val:.1f} words/s) across '{region_words}'. "
        "This {consequence_rate}."
    ),
    "pitch": (
        "Pitch deviation of {mean_delta:.1f} semitone(s) {direction_label} from "
        "{baseline_label} detected across '{region_words}'. "
        "This {consequence_pitch}."
    ),
    "energy": (
        "Volume was {mean_delta:.1f} dB {direction_label} than {baseline_label} "
        "across '{region_words}'. "
        "This {consequence_energy}."
    ),
    "pause": (
        "{direction_label_cap} pause(s) detected {pause_detail} across '{region_words}'. "
        "This {consequence_pause}."
    ),
    "clarity": (
        "Articulatory clarity degraded by {mean_delta:.2f} (MFCC cosine distance) "
        "relative to {baseline_label} across '{region_words}'. "
        "This is consistent with {consequence_clarity}."
    ),
}

_CONSEQUENCE = {
    "rate_fast": "compresses stressed syllables and reduces listener processing time",
    "rate_slow": "elongates delivery and may reduce audience engagement",
    "pitch_high": "raises vocal tension and may signal anxiety or rushing",
    "pitch_low": "flattens intonation and reduces rhetorical emphasis",
    "energy_loud": "increases forceful emphasis but risks fatiguing the listener",
    "energy_quiet": "reduces projected authority and risks losing the audience",
    "pause_present": "inserts unplanned silence that breaks rhetorical flow",
    "pause_absent": "removes a natural breath point, compressing the phrasing",
    "clarity_degraded": "reduced formant transitions and under-articulated consonants, "
                         "producing a mumbled or slurred quality",
}


def _format_region_words(words, region_flaw) -> str:
    """Return a short label for the region word span."""
    indices = region_flaw.word_indices
    if not indices:
        return "(unknown span)"
    span = words[indices[0] : indices[-1] + 1]
    label_words = [w.word for w in span]
    if len(label_words) > 6:
        label_words = label_words[:3] + ["..."] + label_words[-2:]
    return " ".join(label_words)


def _build_template(region, uploaded_words, mode: str) -> str:
    ft = region.flaw_type
    baseline_label = (
        "the paired GOOD baseline" if mode == "A" else "the expected range for this feature"
    )
    rw = _format_region_words(uploaded_words, region)

    if ft == "rate":
        direction_label = region.direction  # "fast" | "slow"
        pct_change = abs(
            (region.uploaded_mean - region.baseline_mean)
            / max(abs(region.baseline_mean), 0.01)
        ) * 100
        consequence = _CONSEQUENCE.get(f"rate_{direction_label}", "alters pacing")
        return _TEMPLATES["rate"].format(
            pct_change=pct_change,
            direction_label=direction_label,
            baseline_label=baseline_label,
            uploaded_val=region.uploaded_mean,
            baseline_val=region.baseline_mean,
            region_words=rw,
            consequence_rate=consequence,
        )

    elif ft == "pitch":
        direction_label = region.direction  # "high" | "low"
        consequence = _CONSEQUENCE.get(f"pitch_{direction_label}", "shifts intonation")
        return _TEMPLATES["pitch"].format(
            mean_delta=region.mean_delta,
            direction_label=direction_label,
            baseline_label=baseline_label,
            region_words=rw,
            consequence_pitch=consequence,
        )

    elif ft == "energy":
        direction_label = region.direction  # "loud" | "quiet"
        consequence = _CONSEQUENCE.get(f"energy_{direction_label}", "changes loudness")
        return _TEMPLATES["energy"].format(
            mean_delta=region.mean_delta,
            direction_label=direction_label,
            baseline_label=baseline_label,
            region_words=rw,
            consequence_energy=consequence,
        )

    elif ft == "pause":
        direction_label_cap = "Unplanned" if region.direction == "present" else "Missing"
        pause_detail = f"(duration ~{region.mean_delta:.2f}s)" if region.direction == "present" else ""
        consequence = _CONSEQUENCE.get(f"pause_{region.direction}", "disrupts rhythm")
        return _TEMPLATES["pause"].format(
            direction_label_cap=direction_label_cap,
            pause_detail=pause_detail,
            region_words=rw,
            consequence_pause=consequence,
        )

    elif ft == "clarity":
        consequence = _CONSEQUENCE.get("clarity_degraded", "reduced articulation precision")
        return _TEMPLATES["clarity"].format(
            mean_delta=region.mean_delta,
            baseline_label=baseline_label,
            region_words=rw,
            consequence_clarity=consequence,
        )

    return f"Flaw detected in {ft} across '{rw}' (delta={region.mean_delta:.2f})."


def _llm_polish(template_text: str, region_json: dict) -> str:
    """Optional LLM prose polish at temperature=0. Falls back to template on any error."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return template_text

    try:
        from openai import OpenAI  # lazy import

        client = OpenAI(api_key=api_key)
        prompt = (
            "You are editing a speech-analytics explanation for a dashboard. "
            "Rewrite the following sentence for clarity and concision ONLY. "
            "Do NOT change any numbers, timestamps, or factual claims. "
            "Return only the rewritten sentence, no preamble.\n\n"
            f"Original: {template_text}"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("LLM polish failed (%s) — using template unchanged", exc)
        return template_text


# -- Public API -----------------------------------------------------------------

def explain(grounding_result, uploaded_words, use_llm: bool = True) -> list[Explanation]:
    """
    Generate causal explanations for all flaw regions in *grounding_result*.

    Parameters
    ----------
    grounding_result : GroundingResult
    uploaded_words : list[NormalizedWord]
    use_llm : bool
        If True, attempt LLM prose polish. Falls back to template on failure.
    """
    explanations: list[Explanation] = []
    mode = grounding_result.mode

    for region in grounding_result.flaw_regions:
        template = _build_template(region, uploaded_words, mode)
        polished = _llm_polish(template, {}) if use_llm else template

        explanations.append(Explanation(
            flaw_type=region.flaw_type,
            region_start_s=region.start_s,
            region_end_s=region.end_s,
            region_words=_format_region_words(uploaded_words, region),
            template_text=template,
            polished_text=polished,
        ))

    return explanations
