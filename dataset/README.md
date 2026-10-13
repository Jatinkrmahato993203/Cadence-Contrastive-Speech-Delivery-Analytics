# Cadence Dataset — Contrastive Speech Analytics

## Overview
64 labeled audio clips (8 texts × 8 severity gradient points) for training and
evaluating speech-delivery flaw detection. Each text produces:
- **1 GOOD** recording (clean neural TTS oratorical baseline)
- **7 BAD** recordings: near-perfect, 5 moderate single-flaw, 1 egregious-combined

All audio: 16 kHz / 16-bit mono WAV. Format follows `{text_id}_{flaw_type}.wav`.

## Calibration / Held-Out Split
| Set | Texts | Clips | Use |
|---|---|---|---|
| Calibration | 1, 2, 3, 4, 5, 7 | 48 | Threshold + normalization statistics |
| Held-out | 6, 8 | 16 | Temporal-IoU accuracy reporting only |

Thresholds and normalization statistics are **never** computed from held-out clips.

## Speech Texts & Voice Configuration Table
| text_id | Title | Voice / Engine | Rate | Text Legal Basis | Voice Persona & Delivery Style |
|---|---|---|---|---|---|
| 1 | Lincoln, Gettysburg Address (1863) | MS Edge: `en-US-GuyNeural` | `-4%` | Public Domain (1863) | Solemn, measured oratorical cadence with deliberate clause pauses |
| 2 | Patrick Henry, Give Me Liberty (1775) | MS Edge: `en-US-ChristopherNeural` | `+2%` | Public Domain (1775) | Urgent, dramatic pacing with heightened vocal emphasis |
| 3 | Frederick Douglass, Fourth of July (1852) | MS Edge: `en-US-EricNeural` | `-2%` | Public Domain (1852) | Broad dynamic volume range, deep resonance on condemnatory clauses |
| 4 | Susan B. Anthony, Is It a Crime (1873) | MS Edge: `en-US-AriaNeural` | `-3%` | Public Domain (1873) | Measured, legalistic delivery with rhythmic syllabic stress |
| 5 | George Washington, Farewell Address (1796) | MS Edge: `en-US-RogerNeural` | `-5%` | Public Domain (1796) | Formal gravity, steady deliberate breathing rhythms |
| 6 | Sojourner Truth, Ain't I a Woman (1851) | MS Edge: `en-US-JennyNeural` | `-2%` | Public Domain (1851 / Gage 1863) | Emphatic rhetorical refrain delivery — HELD OUT |
| 7 | Team original passage #1 | MS Edge: `en-US-SteffanNeural` | `-3%` | CC0 Public Domain Dedication | Modern oratorical prose, rhythmic triads, clause boundaries |
| 8 | Team original passage #2 | MS Edge: `en-US-AnaNeural` | `-2%` | CC0 Public Domain Dedication | Crescendo rhetorical delivery with pitch modulations — HELD OUT |

> **Audio Baseline Architecture:** The underlying speech texts are public domain or author-dedicated under CC0. Clean baseline audio is synthesized using 8 distinct Microsoft Edge TTS neural voices (`edge-tts`) with calibrated speech rates. This provides clean, artifact-free acoustic baselines across distinct vocal profiles, ensuring deterministic reproducibility and enabling controlled programmatic DSP flaw injection without background noise or room reverberation contamination.

> **Note (text 6):** The exact wording of "Ain't I a Woman?" follows the Frances Gage 1863 transcription.

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
