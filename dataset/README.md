# Cadence Dataset — Contrastive Speech Analytics

## Overview
64 labeled audio clips (8 texts × 8 severity gradient points) for training and
evaluating speech-delivery flaw detection. Each text produces:
- **1 GOOD** recording (LibriVox narration or team self-recording)
- **7 BAD** recordings: near-perfect, 5 moderate single-flaw, 1 egregious-combined

All audio: 16 kHz / 16-bit mono WAV. Format follows `{text_id}_{flaw_type}.wav`.

## Calibration / Held-Out Split
| Set | Texts | Clips | Use |
|---|---|---|---|
| Calibration | 1, 2, 3, 4, 5, 7 | 48 | Threshold + normalization statistics |
| Held-out | 6, 8 | 16 | Temporal-IoU accuracy reporting only |

Thresholds and normalization statistics are **never** computed from held-out clips.

## Licensing Citation Table
| text_id | Title | Reader / Source | LibriVox URL | PD Statement Excerpt | Delivery Rationale |
|---|---|---|---|---|---|
| 1 | Lincoln, Gettysburg Address (1863) | LibriVox — to be filled during Phase 0 screening | _URL_ | _Exact PD dedication text copied from catalog page_ | Selected for strongest oratorical stress on "nation", "dedicated", deliberate clause pauses |
| 2 | Patrick Henry, Give Me Liberty (1775) | LibriVox — to be filled during Phase 0 screening | _URL_ | _Exact PD dedication text_ | Selected for rising urgency and emphatic rhetorical questions |
| 3 | Frederick Douglass, Fourth of July (1852) | LibriVox — to be filled during Phase 0 screening | _URL_ | _Exact PD dedication text_ | Selected for emotional range and deliberate pause placement |
| 4 | Susan B. Anthony, Is It a Crime (1873) | LibriVox — to be filled during Phase 0 screening | _URL_ | _Exact PD dedication text_ | Selected for measured, declaratory cadence with stress on legal terms |
| 5 | George Washington, Farewell Address (1796) | LibriVox — to be filled during Phase 0 screening | _URL_ | _Exact PD dedication text_ | Selected for formal oratorical pacing and gravitas |
| 6 | Sojourner Truth, Ain't I a Woman (1851) | LibriVox — to be filled during Phase 0 screening | _URL_ | _Exact PD dedication text_ | Selected for repeated rhetorical refrain and vocal emphasis — HELD OUT |
| 7 | Team original passage #1 | Team self-recording (rehearsed) | N/A — team-owned | Team owns all rights, zero licensing question | Rehearsed for oratorical stress, deliberate pacing, pause-for-emphasis |
| 8 | Team original passage #2 | Team self-recording (rehearsed) | N/A — team-owned | Team owns all rights | Rehearsed oratory anchor — HELD OUT |

> **Note (text 6):** The exact wording of "Ain't I a Woman?" is historically
> contested among transcriptions. This dataset follows the Frances Gage 1863
> transcription, as used by the chosen LibriVox reader. This caveat is disclosed
> here rather than silently accepted.

> **Delivery-style trade-off (texts 1–6):** LibriVox readers default to neutral
> audiobook delivery. Candidates were screened across all available narrations per
> text; the most rhetorically-delivered reading was selected (see Delivery Rationale
> column). Two team-recorded originals (texts 7–8) are performed oratory by construction.

## Flaw Taxonomy and Parameters
| # | Flaw | Severity Used | Tool | Parameters |
|---|---|---|---|---|
| 1 | Pacing | Moderate | librosa.effects.time_stretch | 0.75× or 1.30× |
| 2 | Pitch / Monotone | Moderate | parselmouth pitch-tier | 40% variance compression |
| 3 | Pauses | Moderate | pydub splice at MFA boundaries | 1 unplanned pause 0.4–0.8 s |
| 4 | Volume | Moderate | pydub gain envelope | -10 dB over 20% of clip |
| 5 | Clarity / Articulation | Moderate | phone crossfade + STFT smoothing + transient softening | 40–60 ms crossfade, 7-bin kernel, -4 dB transient |
| 6 | Combined Egregious | Egregious | rows 1+2+3 chained | 0.65×/1.5× + 10% variance + 1.0–2.0 s pause |
| 7 | Near-perfect | Near-perfect | single mildest perturbation | any one flaw at near-perfect tier |

## Dataset JSON Schema (per clip)
```json
{
  "clip_id": "text_01_pacing_mod",
  "source_text_id": 1,
  "speaker_id": "librivox_reader_name",
  "speaker_source": "librivox",
  "flaw_type": "pacing",
  "severity_tier": "moderate",
  "split": "calibration",
  "flaw_regions": [{"start_s": 12.4, "end_s": 38.1, "mechanism": "time_stretch_0.75x"}],
  "transcript_path": "dataset/transcripts/text_01.txt",
  "audio_path": "dataset/audio/text_01_pacing_mod.wav"
}
```

## File Naming Convention
`text_{ID:02d}_{flaw}.wav`

| Flaw suffix | Meaning |
|---|---|
| `good` | GOOD baseline |
| `pacing_mod` | Pacing, moderate |
| `pitch_mod` | Pitch/monotone, moderate |
| `pause_mod` | Pauses, moderate |
| `volume_mod` | Volume, moderate |
| `clarity_mod` | Clarity, moderate |
| `combined_egregious` | Combined/egregious anchor |
| `nearperfect` | Near-perfect anchor |
