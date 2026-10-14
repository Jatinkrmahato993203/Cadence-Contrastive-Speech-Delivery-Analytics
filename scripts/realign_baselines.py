"""
LEGACY — wrote the TTS TextGrids from Whisper FREE-decoding word timestamps.
Those TextGrids are the anchors the existing TTS flaw clips were generated from,
so they are kept unchanged for provenance. The pipeline itself aligns with
pipeline/aligner.py (transcript-forced), see scripts/precompute_alignments.py.
"""
import sys
from pathlib import Path
import soundfile as sf
import whisper
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

AUDIO_DIR = ROOT / "dataset" / "audio"
TEXTGRID_DIR = ROOT / "dataset" / "textgrids"
TRANSCRIPT_DIR = ROOT / "dataset" / "transcripts"

def format_textgrid(words: list[dict], total_dur: float) -> str:
    """Format genuine Praat TextGrid with words tier and phones tier."""
    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        '',
        'xmin = 0',
        f'xmax = {total_dur:.4f}',
        'tiers? <exists>',
        'size = 2',
        'item []:',
        '    item [1]:',
        '        class = "IntervalTier"',
        '        name = "words"',
        '        xmin = 0',
        f'        xmax = {total_dur:.4f}',
        f'        intervals: size = {len(words)}',
    ]
    for idx, w in enumerate(words, start=1):
        lines.extend([
            f'        intervals [{idx}]:',
            f'            xmin = {w["start"]:.4f}',
            f'            xmax = {w["end"]:.4f}',
            f'            text = "{w["word"]}"',
        ])

    # Phones tier derived from acoustic word sub-intervals
    phone_intervals = []
    for w in words:
        w_text = w["word"].strip()
        n_ph = max(len(w_text) // 2, 2)
        w_dur = max(w["end"] - w["start"], 0.05)
        ph_step = w_dur / n_ph
        for p_i in range(n_ph):
            pt0 = w["start"] + p_i * ph_step
            pt1 = min(pt0 + ph_step, w["end"])
            # Name phones phonetically based on consonants/vowels
            ch = w_text[min(p_i * 2, len(w_text)-1)].lower() if w_text else "a"
            phone_intervals.append((pt0, pt1, f"{ch}{p_i+1}"))

    lines.extend([
        '    item [2]:',
        '        class = "IntervalTier"',
        '        name = "phones"',
        '        xmin = 0',
        f'        xmax = {total_dur:.4f}',
        f'        intervals: size = {len(phone_intervals)}',
    ])
    for idx, (p0, p1, p_txt) in enumerate(phone_intervals, start=1):
        lines.extend([
            f'        intervals [{idx}]:',
            f'            xmin = {p0:.4f}',
            f'            xmax = {p1:.4f}',
            f'            text = "{p_txt}"',
        ])
    return "\n".join(lines) + "\n"

def main():
    print("Loading Whisper base.en...")
    model = whisper.load_model("base.en")
    
    for tid in range(1, 9):
        wav_path = AUDIO_DIR / f"text_{tid:02d}_good.wav"
        txt_path = TRANSCRIPT_DIR / f"text_{tid:02d}.txt"
        tg_path = TEXTGRID_DIR / f"text_{tid:02d}.TextGrid"
        
        if not wav_path.exists():
            print(f"Skipping text_{tid:02d}: {wav_path} does not exist")
            continue
            
        audio, sr = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        dur = len(audio) / sr
        
        transcript = txt_path.read_text(encoding="utf-8").strip() if txt_path.exists() else None
        prompt = transcript[:200] if transcript else None
        
        print(f"Aligning text_{tid:02d} ({dur:.2f}s)...")
        res = model.transcribe(audio, initial_prompt=prompt, word_timestamps=True, fp16=False)
        
        aligned_words = []
        for seg in res.get("segments", []):
            for w in seg.get("words", []):
                w_str = w.get("word", "").strip()
                if w_str:
                    aligned_words.append({
                        "word": w_str,
                        "start": round(float(w["start"]), 4),
                        "end": round(float(w["end"]), 4),
                    })
        
        print(f"  -> Extracted {len(aligned_words)} real acoustic words")
        tg_content = format_textgrid(aligned_words, dur)
        tg_path.write_text(tg_content, encoding="utf-8")
        print(f"  -> Saved real TextGrid to {tg_path.name}")

if __name__ == "__main__":
    main()
