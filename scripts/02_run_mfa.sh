#!/usr/bin/env bash
# LEGACY — optional Montreal Forced Aligner path. NOT used by the as-built pipeline:
# word timestamps come from pipeline/aligner.py (Whisper transcript-forced alignment),
# see scripts/precompute_alignments.py.
# scripts/02_run_mfa.sh
# ---------------------
# Phase 0: Run Montreal Forced Aligner on all 8 GOOD clips.
# MUST be run BEFORE scripts/03_generate_bad_clips.py, because the pause
# and clarity flaw scripts consume MFA word/phone timestamps directly.
#
# Prerequisites:
#   conda activate cadence
#   mfa model download acoustic english_us_arpa
#   mfa model download dictionary english_us_arpa
#
# Usage:
#   bash scripts/02_run_mfa.sh

set -euo pipefail

CORPUS_DIR="dataset/audio/mfa_corpus"
TEXTGRID_DIR="dataset/textgrids"
MODEL="english_us_arpa"

echo "=== Phase 0: MFA alignment on 8 GOOD clips ==="

# Build MFA corpus structure: each clip needs audio + transcript in same dir,
# transcript named identically to audio but with .txt extension.
mkdir -p "$CORPUS_DIR"
mkdir -p "$TEXTGRID_DIR"

for i in 01 02 03 04 05 06 07 08; do
    WAV="dataset/audio/text_${i}_good.wav"
    TXT="dataset/transcripts/text_${i}.txt"
    if [ -f "$WAV" ] && [ -f "$TXT" ]; then
        cp "$WAV" "${CORPUS_DIR}/text_${i}.wav"
        cp "$TXT"  "${CORPUS_DIR}/text_${i}.txt"
        echo "  Staged text_${i}"
    else
        echo "  WARNING: Missing $WAV or $TXT — skipping text_${i}"
    fi
done

echo ""
echo "Running MFA alignment..."
mfa align \
    "$CORPUS_DIR" \
    "$MODEL" \
    "$MODEL" \
    "$TEXTGRID_DIR" \
    --clean \
    --num_jobs 4

echo ""
echo "MFA alignment complete. TextGrids saved to: $TEXTGRID_DIR"
echo ""
echo "Running sanity checks..."
python scripts/verify_mfa.py
echo ""
echo "All checks passed. Ready for Phase 1 flaw generation."
echo "Next step: python scripts/03_generate_bad_clips.py"

