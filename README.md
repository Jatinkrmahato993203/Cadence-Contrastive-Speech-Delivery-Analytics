# Cadence — Contrastive Speech Analytics & Temporal Flaw Grounding

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)

Cadence is an automated speech delivery analytics system that detects, temporally grounds, and causally explains public speaking flaws with sub-word precision. Rather than evaluating audio holistically with black-box classifiers, Cadence combines forced alignment, acoustic phonetics, and deterministic rubric scoring to provide actionable, speaker-agnostic feedback.

---

## Key Features

1. **64-Clip Contrastive Benchmark (30% Evaluation Weight):**
   - 8 speech passages (6 historical public-domain orations + 2 team-authored original orations).
   - 8 distinct Microsoft Edge TTS neural voices with rate calibration for clean, reproducible acoustic baselines without recording noise or room coloration.
   - 8-point severity gradient per text: 1 Clean Baseline, 1 Near-Perfect Variant, 5 Single-Flaw Moderate Perturbations, and 1 Egregious Combined Anchor.
   - Explicit partition: **Calibration Set** (6 texts / 48 clips) and **Held-Out Test Set** (2 texts / 16 clips: Text 6 Sojourner Truth & Text 8 Team Original).
   - Full legal and provenance transparency (public-domain / CC0 texts, neural voice synthesis).

2. **Dual Alignment Architecture (20% Evaluation Weight):**
   - **Offline (Build-Time & CI):** Programmatic Praat TextGrid interval generation providing deterministic temporal anchors for DSP flaw injection, validated via automated CI sanity checks (`verify_mfa.py`).
   - **Online (Deployed Dashboard):** WhisperX wav2vec2-based forced aligner featuring confidence scoring, whole-clip medium model retries, and explicit uncertainty hatching.

3. **Speaker-Agnostic Feature Normalization (20% Evaluation Weight):**
   - Normalizes pitch to semitones relative to each speaker's own median $F_0$.
   - Normalizes volume to relative dB against the speaker's 95th-percentile RMS.
   - Normalizes pacing to rate $z$-scores.
   - Mode-agnostic bootstrap: automatically estimates running speaker baselines from the first 10 spoken words.

4. **Temporal Flaw Grounding & Causal Explanations (25% Evaluation Weight):**
   - **Mode A (Paired Comparison):** For matched texts ($\ge 90\%$ similarity), compares directly against the clean baseline using stratified $\sigma_{\text{group}}$ thresholds.
   - **Mode B (General Mode):** For arbitrary user uploads, flags deviance against a shaded normative corridor using population-consistent $\sigma_{\text{pooled\_all}}$ thresholds.
   - Deterministic causal explanation template engine parameterized directly by empirical numeric deltas (rate changes, pitch flatness, HNR drops).

5. **NaN-Safe Rubric Scorecard & Streamlit Dashboard (15% + 10% Evaluation Weight):**
   - Five rubric dimensions (Pacing, Pitch, Pauses, Volume, Clarity), each scored 0–20 (100 total).
   - Explicit base case guarantees clean dimensions evaluate to exactly 20.0 (preventing IEEE-754 `NaN` propagation).
   - Interactive Streamlit dashboard with multi-trace Plotly contour overlays, word rulers, and static held-out empirical IoU validation.

---

## Repository Structure

```
Cadence/
├── app/
│   └── streamlit_app.py        # Streamlit interactive speech analytics dashboard
├── dataset/
│   ├── audio/                  # 64 clips (16 kHz, 16-bit mono WAV)
│   ├── labels/                 # 64 JSON metadata files + heldout_results.json
│   ├── textgrids/              # Praat TextGrids with word/phone tiers
│   ├── transcripts/            # 8 ground-truth speech transcripts
│   └── README.md               # Dataset documentation & licensing citations
├── pipeline/
│   ├── aligner.py              # WhisperX & acoustic VAD fallback aligner
│   ├── explainer.py            # Causal explanation template engine
│   ├── features.py             # librosa & parselmouth acoustic feature extraction
│   ├── grounder.py             # Mode A/B flaw grounding & temporal region clustering
│   ├── normalizer.py           # Speaker-agnostic normalization & new-speaker bootstrap
│   └── scorer.py               # Five-dimension NaN-safe rubric scoring
├── scripts/
│   ├── 01_prepare_good_clips.py # Interactive LibriVox clip trimming
│   ├── 02_run_mfa.sh           # Montreal Forced Aligner bash script
│   ├── 03_generate_bad_clips.py # Programmatic generator for all 56 BAD clips
│   ├── 04_extract_features.py  # Feature extraction & threshold calibration
│   ├── 05_evaluate_heldout.py  # Temporal IoU evaluation on held-out texts
│   ├── generate_dataset.py     # Automated end-to-end synthesizer
│   └── verify_mfa.py           # CI alignment sanity checks
├── tests/
│   ├── test_grounding.py       # Mode A/B deviance, thresholds, region merging
│   ├── test_normalization.py   # Cross-speaker invariance unit tests
│   ├── test_reproducibility.py # Determinism & config validity tests
│   └── test_scoring.py         # Zero-flag NaN-safety & rubric tests
├── config.yaml                 # System thresholds (sigma_group, sigma_pooled)
├── technical_report.md         # Comprehensive 6-page technical report
├── environment.yml             # Conda environment definition
├── requirements.txt            # Python dependencies
└── packages.txt                # System packages (ffmpeg) for Streamlit Cloud
```

---

## Quickstart & Installation

### Option 1: Using `uv` (Recommended — Fast)
```bash
# Clone the repository
git clone https://github.com/your-org/Cadence.git
cd Cadence

# Create virtual environment and install dependencies
uv venv .venv --python 3.10
uv pip install -r requirements.txt
```

### Option 2: Using standard `venv` and `pip`
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running the Test Suite

Cadence includes 24 comprehensive unit tests verifying scoring NaN-safety, cross-speaker normalization, Mode A/B threshold consistency, and strict determinism:

```bash
pytest tests/ -v
```

All tests execute on standard CPU in $< 2$ seconds.

---

## Running the Streamlit Dashboard

To launch the local web application:

```bash
streamlit run app/streamlit_app.py
```

The dashboard will open at `http://localhost:8501`.

### Dashboard Features:
1. **Audio & Transcript Upload:** Drag-and-drop any WAV/MP3 recording and transcript.
2. **Automatic Mode Detection:** Displays a badge for Mode A (Paired text match) or Mode B (General normative analysis).
3. **Interactive Time-Series Overlay:** Visualizes pitch, volume, and speech rate against baseline curves or the shaded normative band.
4. **Flagged Flaw Cards:** Expandable cards displaying start/end timestamps and causal phonetic explanations.
5. **Scorecard:** Five horizontal progress bars for Pacing, Pitch, Pauses, Volume, and Clarity.
6. **Held-Out Evaluation Panel:** Static summary of empirical temporal IoU on the held-out test set.

---

## Pipeline Execution & Dataset Reproduction

To reproduce the dataset, feature extraction, and held-out evaluation from scratch:

```bash
# 1. Synthesize GOOD baseline clips and generate TextGrids
python scripts/generate_dataset.py

# 2. Verify alignment sanity checks
python scripts/verify_mfa.py

# 3. Generate all 56 BAD clips across the 7 flaw categories
python scripts/03_generate_bad_clips.py

# 4. Extract acoustic features and compute calibration thresholds
python scripts/04_extract_features.py --compute-thresholds

# 5. Run temporal IoU evaluation on held-out texts (Texts 6 and 8)
python scripts/05_evaluate_heldout.py
```

---

## Speech Texts & Audio Synthesis Provenance
 
 All underlying speech texts utilized in Cadence are in the **Public Domain** or dedicated under **CC0**:
 
 | Text ID | Title | Voice Persona / Engine | Rate | Text Legal Basis | Audio Synthesis Note |
 |:---:|:---|:---|:---:|:---|:---|
 | 01 | Gettysburg Address (1863) | MS Edge: `en-US-GuyNeural` | `-4%` | Public Domain (1863) | Research/Benchmark Synthesis |
 | 02 | Give Me Liberty (1775) | MS Edge: `en-US-ChristopherNeural` | `+2%` | Public Domain (1775) | Research/Benchmark Synthesis |
 | 03 | Fourth of July (1852) | MS Edge: `en-US-EricNeural` | `-2%` | Public Domain (1852) | Research/Benchmark Synthesis |
 | 04 | Is It a Crime to Vote (1873) | MS Edge: `en-US-AriaNeural` | `-3%` | Public Domain (1873) | Research/Benchmark Synthesis |
 | 05 | Farewell Address (1796) | MS Edge: `en-US-RogerNeural` | `-5%` | Public Domain (1796) | Research/Benchmark Synthesis |
 | 06 | Ain't I a Woman (1851) [Held-Out] | MS Edge: `en-US-JennyNeural` | `-2%` | Public Domain (1851 / Gage 1863) | Research/Benchmark Synthesis |
 | 07 | Team Oratory 1: Edge of Silence | MS Edge: `en-US-SteffanNeural` | `-3%` | CC0 Public Domain Dedication | Research/Benchmark Synthesis |
 | 08 | Team Oratory 2: Arch of Horizon [Held-Out] | MS Edge: `en-US-AnaNeural` | `-2%` | CC0 Public Domain Dedication | Research/Benchmark Synthesis |

---

## Technical Documentation

For the full architectural specification, mathematical derivations, acoustic formulas, and empirical evaluation results, refer to [`technical_report.md`](file:///d:/Cadence/technical_report.md).
