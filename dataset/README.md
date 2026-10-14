# Cadence Dataset — Contrastive Speech Delivery Benchmark

## What is in here

| Set | Baseline ("GOOD") recordings | Flawed variants | Purpose |
|---|---|---|---|
| **TTS benchmark** — `audio/text_NN_*.wav` | 8 texts, synthesized with 8 Microsoft Edge neural voices | 17 per text → 136 | Calibration (texts 1–5, 7) and held-out evaluation (texts 6, 8) |
| **Real orators** — `audio/text_09_*` … `text_11_*` | 3 presidential addresses: JFK "We choose to go to the Moon" (1962), Reagan first inaugural (1981), Eisenhower first inaugural (1953); ~67–75 s passages | 17 per speech → 51 | Held-out evaluation with a genuinely effective speaker as the GOOD baseline |
| **Human recordings** — `audio/real_speakers/realNN_*.wav` | 2 genuine LibriVox readings (Gettysburg Address, Patrick Henry) trimmed to the transcript | 17 per recording → 34 | Evaluation only: same-speaker and cross-speaker (human vs TTS) |

All audio is 16 kHz / 16-bit mono WAV. Every flawed clip has a label JSON with
time-bounded ground-truth regions on the clip's own timeline.

### Honest provenance
* The 8 TTS baselines are **synthetic** (`scripts/generate_dataset.py`, `edge-tts`).
  They are clean and fully controllable, but they are not "highly effective public
  speakers". The `speaker_id` values such as `librivox_reader_01_guy` are legacy
  names — they denote Edge TTS voices (`en-US-GuyNeural`, …), not LibriVox readers.
* Texts 9–11 are recordings of the presidents themselves (`labels/orators.json`, built by
  `scripts/prepare_orators.py`). Public domain as works of the US federal government:
  the JFK recording comes from the JFK Presidential Library (White House Audio
  Collection); the Reagan and Eisenhower files are tagged public domain on Wikimedia
  Commons with the author only *assumed* to be the White House — check before
  redistributing. Source audio is downloaded on demand into `.cache/sources/`.
  Transcripts are the official texts, adjusted to what is actually spoken (JFK says
  "We choose to go to the moon" three times).
* The 2 human recordings are LibriVox volunteer readings (public domain), source URLs
  in `labels/real_speakers.json`. They are read speech by audiobook volunteers, not
  the original orators or champion debaters.
* Texts 1–6 are public-domain speeches; texts 7–8 are team-written passages (CC0).

## Severity gradient (per baseline)

| Flaw | mild (`*_mild`, tier `near_perfect`) | moderate (`*_mod`) | egregious (`*_egregious`) | Mechanism |
|---|---|---|---|---|
| Pacing | 1.15× faster | 0.75× (slower) | 0.65× | Praat PSOLA "Lengthen (overlap-add)" over a bounded window (25–60 / 20–65 / 15–75 % of the clip); replaced librosa's phase vocoder, whose "phasiness" also degraded clarity |
| Pitch | keep 70 % of F0 variance | 35 % (TTS) / 40 % (human) | 10 % | Praat pitch-tier compression toward the mean, bounded window |
| Pauses | 1 × 0.20–0.35 s | 1 × 0.4–0.8 s | 3 × 1.0–2.0 s | digital-silence splice at a word start |
| Volume | −6 dB over 15 % | −10 dB over 20 % | −18 dB over 30 % | gain envelope with short fades |
| Clarity | 15–25 ms x-fade, 3-bin | 40–60 ms, 7-bin | 70–90 ms, 15-bin | phone-boundary crossfade + STFT magnitude smoothing + plosive softening |
| Combined | — | — | volume −18 dB + pitch 10 % + clarity (egregious) + 2 pauses | chained, distinct windows |
| Near-perfect | 1.15× pacing over 25–60 % | | | **identical to `pacing_mild`** (same parameters); kept for compatibility, excluded from summary means |

## Labels

```json
{
  "clip_id": "text_06_pause_egregious",
  "source_text_id": 6,
  "speaker_id": "librivox_reader_06",
  "speaker_source": "librivox",
  "flaw_type": "pause",
  "severity_tier": "egregious",
  "split": "heldout",
  "flaw_regions": [{"start_s": 32.76, "end_s": 34.075, "mechanism": "silence_splice_1.31s_at_src_32.760s"}],
  "label_timeline": "output",
  "transcript_path": "dataset/transcripts/text_06.txt",
  "audio_path": "dataset/audio/text_06_pause_egregious.wav"
}
```

* `flaw_regions` are on the **output** timeline of the flawed clip.
* Labels written before the generator fix placed 2nd/3rd spliced pauses at their
  source position and did not shift the other regions of `combined_egregious` for the
  inserted silences. `scripts/repair_splice_labels.py` rewrote the 16 affected labels;
  every pause window in every label is verified to be digital silence in the audio
  (`tests/test_detection.py::test_pause_labels_match_silence_in_audio`).
* Generation is deterministic per clip (`random.seed(clip_id)`).

## Splits

| Split | Texts | Used for |
|---|---|---|
| Calibration | 1, 2, 3, 4, 5, 7 | choosing the clarity measure and per-feature thresholds (`scripts/04_extract_features.py --calibrate`) |
| Held-out | 6, 8 (TTS) and 9, 10, 11 (real orators) | reporting only (`scripts/05_evaluate_heldout.py`) |
| Human recordings | real01, real02 | reporting only; `real01_good` vs the TTS text-1 baseline was used to set the noise multiplier `k_noise` |

## Other files

| Path | Content |
|---|---|
| `transcripts/text_NN.txt` | ground-truth transcripts |
| `alignments/*.json` | cached transcript-forced word alignments (keyed by audio + transcript + aligner version) |
| `textgrids/text_NN.TextGrid` | word anchors the TTS flaws were generated from (Whisper free-decoding timestamps; the phone tier is a letter-group approximation used only to place clarity edits) |
| `textgrids/realNN.TextGrid` | forced-alignment word tier for the human recordings (same letter-group phone tier) |
| `labels/*_features.json` | per-word normalized features for every TTS clip |
| `labels/calibration_report.json` | chosen clarity measure, floors, k_noise |
| `labels/heldout_results.json` | held-out and human-recording evaluation |

Note (text 6): the wording of "Ain't I a Woman?" follows the Frances Gage 1863 transcription.
