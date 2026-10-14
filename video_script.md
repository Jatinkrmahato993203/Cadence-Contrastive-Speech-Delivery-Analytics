# Cadence — Demo Video Script (~7 minutes)

Every number spoken in this script is read from `dataset/labels/heldout_results.json`
(also shown in the dashboard's evaluation panel). Re-run
`scripts/05_evaluate_heldout.py` and update the numbers here if anything changes.

| # | Segment | Time | On screen |
|---|---|---|---|
| 1 | Problem | 0:00–0:40 | Title, one flawed clip playing |
| 2 | Dataset, honestly | 0:40–1:40 | `dataset/README.md` tables |
| 3 | Alignment & features | 1:40–2:50 | Dashboard ruler + aligner diagram |
| 4 | Mode A demo — human reader | 2:50–4:20 | Preset "Human reader — inserted pauses (text 1)" |
| 5 | Cross-speaker demo | 4:20–5:20 | Preset "Human reader — Gettysburg Address (vs TTS baseline)" |
| 6 | Evaluation | 5:20–6:30 | Evaluation panel |
| 7 | Limits & next steps | 6:30–7:00 | Report §7 |

---

### 1. Problem (0:00–0:40)
> "Speech coaching tools usually give one overall grade. Cadence instead answers
> *where* the delivery went wrong, *what* changed, and *why* — for example: 'at
> 24.1 to 25.5 seconds there is an unplanned 1.4-second silence between *long* and
> *endure*'."

### 2. Dataset, honestly (0:40–1:40)
> "We have thirteen clean baselines. Three are real presidential speeches — Kennedy's
> 'We choose to go to the Moon', and the first inaugurals of Reagan and Eisenhower —
> so the reference really is a highly effective speaker. Two are human LibriVox
> readings, the Gettysburg Address and Patrick Henry. The other eight are Microsoft
> neural TTS voices: fully controllable, but synthetic, and we say so.
> From every baseline we generate seventeen variants: mild, moderate and egregious
> pacing, monotone pitch, pauses, volume and slurred clarity, plus a combined clip
> and a near-perfect one — 221 flawed clips. Only the edit window is changed, snapped
> to gaps between words, and each label gives its exact time span; a test checks
> that every labelled pause is real silence.
> Thresholds are calibrated on six TTS texts only. The presidents, two TTS texts
> and the human readings are held out. Generation is seeded, so rerunning the
> scripts rebuilds every clip byte for byte."

### 3. Alignment & features (1:40–2:50)
> "Free speech recognition drops and mis-hears words, so we force the known
> transcript onto the audio: Whisper's cross-attention gives evidence in
> overlapping 30-second windows, one dynamic-time-warping path assigns every
> word a time, and word edges are snapped to silences. Splicing a pause into a
> recording leaves every timing before it unchanged.
> Per word we measure pitch and pitch movement, loudness, tempo with pauses
> excluded, pause length, and harmonics-to-noise ratio for clarity — each relative
> to the speaker's own voice, so a higher or louder voice is not a flaw."

### 4. Mode A demo — human reader (2:50–4:20)
Select **Human reader — inserted pauses (text 1)**, reference **Same speaker's clean recording**, Analyze.
> "The badge says Mode A, edited copy of the reference recording. Each tab overlays
> the participant and the time-aligned baseline; below it, the per-word delta
> against the threshold. Here are the three inserted pauses, each explained with
> its measured length and the baseline's pause at the same place."
Open one *Evidence* expander to show the raw numbers.

### 5. Cross-speaker demo (4:20–5:20)
Select **Human reader — Gettysburg Address (vs TTS baseline)**, Analyze.
> "Now a different person against the TTS baseline. Two readers never match
> word for word, so Cadence measures how much the two readings differ naturally
> and widens its thresholds accordingly. A clean human reading is not buried in
> false alarms — but this also means only large flaws survive the comparison."

### 6. Evaluation (5:20–6:30)
Scroll to the evaluation panel and read out the three summary lines and the
severity-gradient result exactly as displayed.
> "These texts and recordings were never used to choose thresholds. Detection is
> strong when the reference is the same recording, the score falls as severity
> rises, and the cross-speaker line shows the honest limit."

### 7. Limits & next steps (6:30–7:00)
> "The biggest gap is data: we need recordings of genuinely effective speakers,
> and the same people reading badly, to calibrate cross-speaker comparison. That,
> a stronger aligner, and listener ratings to validate the explanations are next."
