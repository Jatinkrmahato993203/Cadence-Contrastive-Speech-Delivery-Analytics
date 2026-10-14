# Cadence — Contrastive Speech Delivery Analytics
## Technical Report (as built)

Cadence finds delivery flaws in a speech recording, timestamps them, and explains
them in plain language by comparing the recording with a reference reading of the
same text (Mode A) or with the speaker's own typical delivery (Mode B).

This report describes exactly what the code does. Every number in §6 is produced
by `scripts/05_evaluate_heldout.py` and stored in `dataset/labels/heldout_results.json`.

---

### 1. Dataset (Requirement 1)

#### 1.1 Baselines
| Source | Count | Notes |
|---|---|---|
| Microsoft Edge neural TTS (`edge-tts`) | 8 texts | Clean and controllable; **synthetic** — not real orators |
| **Presidential addresses** (public domain, US federal works) | 3 — JFK, Rice University "We choose to go to the Moon" (1962); Reagan, first inaugural (1981); Eisenhower, first inaugural (1953) | Texts 9–11: the real orator IS the GOOD baseline; ~67–75 s passages located in the full recordings and cut in silence by `scripts/prepare_orators.py`; held out |
| LibriVox volunteer readings (public domain) | 2 (Gettysburg Address — Andrew Marty; Patrick Henry — Mark F. Smith) | Genuine human delivery; trimmed to the transcript span by `scripts/prepare_real_speakers.py` (recognition-matched, cut in silence) |

Requirement 1a asks for recordings of highly effective speakers. Texts 9–11 use
three US presidents' own delivery of landmark addresses as the GOOD baseline, with
the full flaw gradient built from them. The JFK recording comes from the JFK
Presidential Library; the Reagan and Eisenhower files are tagged public domain on
Wikimedia Commons with the author only assumed to be the White House. Transcripts are
the official texts adjusted to what is spoken (JFK repeats "We choose to go to the
moon" three times). Texts 1–8 remain TTS.

#### 1.2 Flaw gradient (Requirement 1b)
Every baseline gets 17 variants: mild / moderate / egregious versions of five flaws
(pacing, pitch monotony, pauses, volume, clarity), a combined-egregious clip and a
"near-perfect" clip (identical to `pacing_mild`; excluded from summary means).
Flaws are applied to bounded windows, so most of each clip stays clean. Parameters
are in `dataset/README.md`. Generation is deterministic (`random.seed(clip_id)`).

#### 1.3 Labels (Requirement 1c)
Ground truth comes from the transform itself, on the flawed clip's output timeline.
Review found that multi-pause clips and the combined clip had labels on the wrong
timeline (inserted silences were not offset by earlier insertions). The generator
was fixed and `scripts/repair_splice_labels.py` rewrote the 16 affected labels.
A test asserts that every labelled pause window is digital silence in the audio.

---

### 2. Alignment and features (Requirement 2)

#### 2.1 Transcript-forced alignment (`pipeline/aligner.py`)
Free-decoding word timestamps (Whisper `word_timestamps=True`) drop and mis-hear
words, which breaks word-for-word comparison. Cadence forces the given transcript
onto the audio with Whisper `base.en`:

1. **Anchors** — Whisper transcribes the clip; recognized words are matched to the
   transcript (difflib) to estimate where each token lies.
2. **Windowed evidence** — 30 s windows every 15 s, placed by time. In each window
   the decoder receives the transcript tokens expected within ±3 s; the
   alignment-head cross-attention gives a token × frame evidence matrix
   (same computation as `whisper.timing.find_alignment`).
3. **One global DTW** — window evidence is averaged into a single matrix and one
   monotonic DTW path assigns every transcript token a time.
4. **Silence snapping** — word intervals are trimmed to voiced content and
   boundaries moved onto detected silences. Silence is defined from *local*
   statistics only (35 dB below the loud speech within ±1.5 s, at least 6 dB above
   the noise floor within ±4 s), so identical audio always snaps identically and a
   passage spoken 18 dB quieter is not mistaken for silence. A word that DTW placed
   inside a silence is moved to the nearest voiced run.

Properties verified on the dataset: every transcript word is returned in order;
splicing a pause into a recording leaves the timings before it unchanged (95th
percentile difference 0.00 s); spliced pauses appear as gaps within ~20 ms of the
label. Rejected after measurement: per-chunk forced alignment with anchor-defined
cuts (cut points moved when audio elsewhere changed), shorter audio-defined chunks
(word-to-chunk assignment errors up to 6 s), and a clip-global silence threshold
(a change anywhere in a clip moved word edges everywhere — up to 1.4 s).

Words whose token probability is < 0.25 are marked *uncertain* and shown as grey
bands in the dashboard.

#### 2.2 Features (`pipeline/features.py`)
One STFT (32 ms window, 10 ms hop) and one Praat analysis per clip, pooled over
each aligned word:

| Feature | Method |
|---|---|
| F0 contour | Praat autocorrelation pitch, 60–500 Hz, silence threshold 1% of peak |
| HNR | Praat cross-correlation harmonicity, silence threshold 1% of peak |
| Spectral centroid, flatness, contrast, flux, >3 kHz energy ratio | from the STFT magnitude |
| MFCC (13) | from a 40-band mel spectrogram |
| Energy | frame RMS (dB) |
| Word duration, pause before word | from the alignment |
| Jitter, shimmer | Praat, clip level |

#### 2.3 Normalization (`pipeline/normalizer.py`)
Each clip is normalized against **its own whole-clip** statistics: pitch in
semitones from its median voiced F0; local pitch variability (interquartile spread
of voiced frames over 3 words — robust to octave errors); energy relative to its
90th-percentile word level; tempo as words
per second of articulation (pauses excluded). A participant is never measured in
another speaker's units — an earlier version did so and penalised a human reader
for having a different voice from the TTS baseline. A test asserts that the same
prosodic pattern spoken an octave higher and 12 dB louder normalizes identically.

The clarity measure is chosen on the calibration texts by AUC (flawed vs clean
words): HNR (0.996) narrowly beat spectral contrast (0.995); the >3 kHz energy
ratio reached 0.85; flux and flatness were near chance
(`dataset/labels/calibration_report.json`).

---

### 3. Temporal grounding (Requirement 3a, `pipeline/grounder.py`)

**Mode A** (transcript similarity ≥ 0.90 with a bundled text): both readings are
aligned to the same tokens, so word *i* is compared with word *i*:

| Feature | Per-word delta |
|---|---|
| Pitch | log2(pitch variability ratio) |
| Energy | raw dB difference, re-centred (see below) |
| Rate | log2 ratio of 5-word articulation time (boundary errors cancel inside the window) |
| Pause | silence difference, capped by the difference in total silence over ±1 word (a pause that only moved one word cancels; one inserted beside a natural pause still counts) |
| Clarity | raw HNR difference, re-centred (see below) |

Energy and clarity are compared raw, then a global offset (gain, timbre) is removed:
zero is kept as the reference when ≥ 15% of the words are unchanged, otherwise the
densest 20% of the deltas defines it. Per-clip normalization was tried first and
failed when a degradation covered most of the clip (the clip median became the
degraded level, inverting the detection).

**Threshold:** τ_f = max(floor_f, k · noise_f), with noise_f = 20th percentile of
|Δ| / 0.2533 — anchored at zero (no change), unbiased for Gaussian noise, and
unaffected by a flaw covering up to ~80% of the words. Two cases:

* **Edited copy of the reference recording** — detected from timbre: the 20th
  percentile of paired-word MFCC (c1–c12) distance is ≤ 3.9 for every edited copy
  and ≥ 28 for every different recording; the gate sits at their geometric
  midpoint (12.35). Only alignment jitter separates the two readings, so k = 3.
* **Different recording** (a real user) — k_f per feature is the smallest value that
  keeps both clean human-vs-TTS pairs under 2% of their duration flagged.

Floors are chosen per feature on the calibration texts (all tiers), maximising
target IoU − 2 × false-positive fraction; floors and k are alternated until stable.

Flagged words form same-sign runs (≥ 2 words; ≥ 6 for rate, since pacing is
sustained and human phrase-final lengthening spans a few words; 1 for pauses),
merged across gaps < 300 ms.

**Long-scale detector (different recordings and Mode B).** Natural differences
between two readers are mostly word-to-word fluctuation; a delivery flaw is usually
sustained over a phrase. Each feature's deltas are also averaged over 15 words, with
the noise scale divided by √15, and flagged runs of ≥ 7 words are added. Its
multiplier k_long is calibrated like k (≤ 2% of the clean human-vs-TTS pairs). A pause must at least double the reference silence only where the
surrounding tempo also changed (otherwise a slowed passage would report its
stretched pauses as pause flaws).

**Mode B** (no matching text): deltas from the speaker's own robust centre; pauses
flagged above an absolute length. Its floors are calibrated on the calibration texts
(clean and flawed clips, same objective as Mode A). Without a reference, sustained
volume drops and long pauses are found, but monotone pitch, pacing and clarity
changes are largely indistinguishable from a speaker's natural variation (§6.4b).

### 4. Causal explanations (Requirement 3b, `pipeline/explainer.py`)
One explanation per region, built only from measured values — *what* (feature,
direction, size in natural units against the baseline value), *where* (time span
and words), *why* (acoustic mechanism consistent with the measured delta, e.g. a
tempo drop measured with pauses excluded means drawn-out words, not added
silence), *effect* on the listener — plus a severity (|Δ|/τ: mild < 1.5 ≤
moderate < 3 ≤ severe) and the raw evidence JSON. An optional LLM pass may only
rephrase.

Limitation: the "why" is mechanism-level reasoning chosen from the measured
pattern, not causal inference about the speaker's intent.

### 5. Scoring
Five dimensions × 20. Penalty per region = coverage × 1.5·min(severity, 3)/3;
pauses count as at least 5 s of disruption. A dimension with no regions scores
exactly 20 (explicit NaN-safe base case).

### 6. Results

<!-- RESULTS:BEGIN (generated by scripts/render_results.py) -->
#### 6.1 Summary

| Evaluation set | Clips | Mean target IoU | Mean false-positive seconds / clip |
|---|---|---|---|
| Real orators — JFK, Reagan, Eisenhower — held out (reference = same recording) | 51 | 0.80 | 13.7 |
| TTS held-out texts 6 & 8 (reference = same recording) | 34 | 0.88 | 10.9 |
| Human LibriVox readers, same-speaker reference | 32 | 0.85 | 14.4 |
| Human readers vs the TTS baseline (different voice) | 34 | 0.17 | 5.3 |
| Mode B — no reference recording, all held-out audio | 119 | 0.22 | 19.1 |

#### 6.2 Real orators (texts 9-11, held out) — target IoU (false-positive seconds) per tier

| Flaw | mild | moderate | egregious |
|---|---|---|---|
| pacing | 0.66 (8 s) | 0.83 (24 s) | 0.83 (24 s) |
| pitch | 0.61 (2 s) | 0.94 (4 s) | 0.95 (6 s) |
| pause | 0.57 (1 s) | 0.63 (3 s) | 0.98 (10 s) |
| volume | 0.85 (4 s) | 0.89 (7 s) | 0.97 (20 s) |
| clarity | 0.65 (6 s) | 0.84 (30 s) | 0.91 (55 s) |
| combined | — | — | 0.67 (29 s) |

#### 6.2b Held-out TTS texts — target IoU (false-positive seconds) per tier

| Flaw | mild | moderate | egregious |
|---|---|---|---|
| pacing | 0.89 (10 s) | 0.86 (8 s) | 0.92 (29 s) |
| pitch | 0.92 (3 s) | 0.98 (2 s) | 0.99 (3 s) |
| pause | 0.92 (0 s) | 0.59 (2 s) | 0.86 (4 s) |
| volume | 0.96 (0 s) | 0.98 (1 s) | 0.94 (8 s) |
| clarity | 0.70 (8 s) | 0.97 (18 s) | 0.90 (72 s) |
| combined | — | — | 0.75 (21 s) |

#### 6.3 Human recordings, same-speaker reference

| Flaw | mild | moderate | egregious |
|---|---|---|---|
| pacing | 0.48 (9 s) | 0.98 (17 s) | 0.96 (22 s) |
| pitch | 0.72 (19 s) | 0.93 (7 s) | 0.96 (3 s) |
| pause | 0.45 (1 s) | 0.96 (1 s) | 0.71 (4 s) |
| volume | 0.95 (2 s) | 0.95 (3 s) | 0.99 (16 s) |
| clarity | 0.73 (8 s) | 0.95 (33 s) | 0.99 (63 s) |
| combined | — | — | 0.81 (21 s) |

#### 6.4 Human recordings against the TTS baseline

| Flaw | mild | moderate | egregious |
|---|---|---|---|
| pacing | 0.00 (7 s) | 0.00 (5 s) | 0.00 (2 s) |
| pitch | 0.12 (6 s) | 0.12 (7 s) | 0.00 (7 s) |
| pause | 0.00 (5 s) | 0.00 (3 s) | 0.59 (4 s) |
| volume | 0.05 (5 s) | 0.49 (4 s) | 0.47 (7 s) |
| clarity | 0.00 (4 s) | 0.19 (9 s) | 0.18 (10 s) |
| combined | — | — | 0.57 (3 s) |

Clean human readings compared with the TTS baseline (both used to set k, so in-sample): `real01_good` scores 99.1/100 with 1.0 s of 64 s flagged; `real02_good` scores 98.7/100 with 4.7 s of 114 s flagged.

#### 6.4b Mode B (no reference recording)

| Flaw | mild | moderate | egregious |
|---|---|---|---|
| pacing | 0.03 (7 s) | 0.14 (12 s) | 0.25 (23 s) |
| pitch | 0.00 (10 s) | 0.00 (13 s) | 0.01 (30 s) |
| pause | 0.00 (10 s) | 0.02 (12 s) | 0.57 (9 s) |
| volume | 0.40 (12 s) | 0.74 (13 s) | 0.91 (13 s) |
| clarity | 0.03 (13 s) | 0.01 (26 s) | 0.00 (53 s) |
| combined | — | — | 0.44 (58 s) |

#### 6.5 Severity gradient
Score falls monotonically mild → moderate → egregious in 30/35 cases.

#### 6.6 Calibrated parameters
Clarity measure `hnr_db`; floors `{'pitch': 0.2806, 'energy': 5.5114, 'rate': 0.0981, 'pause': 0.14, 'clarity': 1.2629}`; k (different recording) `{'pitch': 3.5, 'energy': 4.0, 'rate': 1.5, 'pause': 1.5, 'clarity': 4.0}`; k (edited copy) `3.0`; same-recording gate (timbre distance) `12.35`.
<!-- RESULTS:END -->

### 7. Limitations and next steps
**Against the brief**

| Requirement | Status |
|---|---|
| 1a Recordings of highly effective speakers | **Met for texts 9–11** (JFK, Reagan, Eisenhower as GOOD baselines, held out) plus 2 LibriVox readings; texts 1–8 are TTS. Two of the three presidential files have assumed rather than archive-documented public-domain provenance. |
| 1b Gradient egregious → near-perfect | Met for five synthetic flaw types (3 tiers each + combined). Real-world flaws such as filler words, up-talk or hesitations are not modelled. |
| 1c Time-bounded paired labels | Met; labels verified against the audio. |
| 2a Forced alignment | Met (transcript-forced Whisper alignment; every word aligned). |
| 2b Feature extraction | Met (STFT/FFT, MFCC, F0, HNR, energy, tempo, pauses). |
| 3a Temporal grounding | Strong against a reference recording of the same speaker, including real orators (§6.2–6.3). Weak against a different speaker's reading and without any reference (Mode B): only large or sustained flaws are found (§6.4, 6.4b). |
| 3b Causal explanations | Deterministic, measurement-grounded, mechanism-level; not validated with listeners. |
| 4a/4b Upload UI and overlay | Met (time-aligned overlays, delta-vs-threshold charts, flaw cards). |

**Known weaknesses**
* *Cross-speaker comparison* — natural stylistic differences between two readers are
  as large as moderate synthetic flaws, so thresholds must widen; k is fitted on
  only two human readings (in-sample for the clean pairs).
* *Mild pauses* (0.2–0.35 s) are still found only about half the time.
* *Mode B* cannot separate monotone pitch, pacing or clarity changes from a speaker's
  natural variation, and flags 15–25 s of the presidents' own dramatic delivery.
* *Collateral detections* — spectral smoothing (clarity edits) also disturbs pitch
  tracking, and resynthesis shifts alignment slightly, which shows up as
  false-positive seconds on clarity and combined clips.
* *Alignment* — Whisper `base.en` cross-attention has 20 ms resolution; low-confidence
  words are flagged, not corrected. A new upload takes ~30–60 s on CPU.
* `nearperfect` clips duplicate `pacing_mild` (same generator parameters); they are
  excluded from summary means.

**Next steps (in order of impact)**
1. Recordings of the *same* people delivering both well and badly (self-recorded),
   to calibrate cross-speaker and reference-free detection on real variation.
2. A phoneme-level aligner (wav2vec2 / MMS forced alignment) for real phone
   boundaries and tighter word edges.
3. A listener study to validate the flaw regions and the explanations.
4. Disfluency detection (fillers, repetitions), which the benchmark does not cover.

### 8. Reproduction
```bash
python scripts/generate_dataset.py             # TTS baselines (requires network for edge-tts)
python scripts/03_generate_bad_clips.py        # TTS flaw variants
python scripts/prepare_real_speakers.py        # human recordings: trim + flaw variants
python scripts/precompute_alignments.py        # forced alignment cache (~1 h on CPU)
python scripts/04_extract_features.py --calibrate
python scripts/05_evaluate_heldout.py
pytest tests/
streamlit run app/streamlit_app.py
```
