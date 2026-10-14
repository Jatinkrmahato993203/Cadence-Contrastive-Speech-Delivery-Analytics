"""
pipeline/aligner.py
-------------------
Transcript-forced word alignment built on OpenAI Whisper (§2.1).

Whisper's own word timestamps come from free decoding, so it drops, merges or
mis-hears words and the word list no longer matches the transcript. Cadence
instead forces the *given* transcript onto the audio:

  1. Anchor pass   — Whisper transcribes the clip with word timestamps; the
                     recognized words are matched to the transcript tokens
                     (difflib) to estimate roughly where each token lies.
  2. Evidence      — fixed 30 s windows every 15 s (placed by time, never by
                     content). In each window the decoder is fed the transcript
                     tokens expected within +/-3 s and its alignment-head
                     cross-attention gives token x frame evidence (exactly as
                     whisper.timing.find_alignment computes it).
  3. Global DTW    — window evidence is averaged into one token x frame matrix
                     and a single monotonic DTW path assigns every transcript
                     token a time. No word can be dropped, there are no chunk
                     cut points to get wrong, and identical audio produces
                     identical evidence (a pause spliced into a recording leaves
                     the timings before it unchanged).
  4. Silence snap  — word intervals are trimmed to voiced content and their
                     boundaries moved onto detected silences, so pauses become
                     explicit gaps (applied on load; the cache keeps pre-snap
                     timings).

Words whose forced-alignment token probability is below
``aligner.uncertain_threshold`` are flagged ``uncertain`` and rendered as grey
hatching in the dashboard — never silently dropped.

If Whisper is not installed, an energy-based proportional fallback is used and
every word is marked uncertain, so the UI never presents it as real alignment.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "config.yaml"
DATASET_DIR = ROOT / "dataset"
DATASET_CACHE_DIR = DATASET_DIR / "alignments"
RUNTIME_CACHE_DIR = ROOT / ".cache" / "alignments"

ALIGNER_VERSION = "whisper-global-dtw-v3"   # windowed evidence + one global DTW; cache stores pre-snap timings
SR = 16000
HOP_S = 0.01

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")


def _load_cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def transcript_words(text: str) -> list[str]:
    """Canonical tokenization shared by the aligner and the grounder."""
    return _WORD_RE.findall(text or "")


# -- Data structures ------------------------------------------------------------

@dataclass
class WordAlignment:
    word: str
    start: float          # seconds
    end: float            # seconds
    score: float          # forced-alignment token probability, 0-1
    uncertain: bool = False


@dataclass
class AlignmentResult:
    words: list[WordAlignment]
    model_used: str        # e.g. "whisper-base.en-forced" | "acoustic_fallback"
    had_retry: bool = False
    uncertain_spans: list[tuple[float, float]] = field(default_factory=list)
    transcript: str = ""


# -- Audio helpers ---------------------------------------------------------------

def _load_audio_16k(path: str) -> np.ndarray:
    import soundfile as sf

    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return np.ascontiguousarray(y, dtype=np.float32)


def frame_db(y: np.ndarray, hop_s: float = HOP_S, win_s: float = 0.02) -> np.ndarray:
    """Frame-level RMS in dB (one value per hop)."""
    hop, win = int(SR * hop_s), int(SR * win_s)
    if len(y) < win:
        return np.full(1, -100.0)
    n = (len(y) - win) // hop + 1
    frames = np.lib.stride_tricks.as_strided(
        y, shape=(n, win), strides=(y.strides[0] * hop, y.strides[0])
    )
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    return 20.0 * np.log10(rms)


def silence_threshold_db(db: np.ndarray) -> float:
    """
    Clip-adaptive silence threshold: 35 dB below loud speech, but above the
    recording's noise floor. Digital-zero frames (< -90 dB, e.g. spliced
    silence) are excluded so that inserting silence cannot move the threshold.
    """
    live = db[db > -90.0]
    if len(live) < 20:
        live = db
    return float(max(np.percentile(live, 95) - 35.0, np.percentile(live, 5) + 6.0))


# -- Whisper helpers -------------------------------------------------------------

_MODELS: dict = {}


def _get_model(name: str):
    import whisper

    if name not in _MODELS:
        _MODELS[name] = whisper.load_model(name, device="cpu")
    return _MODELS[name]


def _recognize(model, audio: np.ndarray, prompt: Optional[str]) -> list[dict]:
    """Free-decoding pass; returns canonical tokens with coarse times."""
    res = model.transcribe(
        audio,
        initial_prompt=prompt,
        word_timestamps=True,
        condition_on_previous_text=False,
        fp16=False,
        language="en",
    )
    out = []
    for seg in res.get("segments", []):
        for w in seg.get("words", []):
            for tok in transcript_words(w.get("word", "")):
                out.append({
                    "word": tok,
                    "start": float(w.get("start", 0.0)),
                    "end": float(w.get("end", 0.0)),
                })
    return out


def _anchor(tokens: list[str], recognized: list[dict]) -> dict[int, tuple[float, float]]:
    """Map transcript token index -> (start, end) for tokens Whisper also heard."""
    a = [t.lower() for t in tokens]
    b = [r["word"].lower() for r in recognized]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    anchors: dict[int, tuple[float, float]] = {}
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            r = recognized[blk.b + k]
            anchors[blk.a + k] = (r["start"], r["end"])
    return anchors


def _interp_time(idx: int, anchors: dict[int, tuple[float, float]], n: int, dur: float) -> float:
    """Estimated start time of token *idx*, interpolated between anchors."""
    if idx in anchors:
        return anchors[idx][0]
    keys = sorted(anchors)
    lo = max([k for k in keys if k < idx], default=None)
    hi = min([k for k in keys if k > idx], default=None)
    t_lo, i_lo = (anchors[lo][1], lo) if lo is not None else (0.0, -1)
    t_hi, i_hi = (anchors[hi][0], hi) if hi is not None else (dur, n)
    return t_lo + (t_hi - t_lo) * (idx - i_lo) / max(i_hi - i_lo, 1)


WINDOW_S = 30.0       # Whisper's receptive field
STRIDE_S = 15.0       # windows overlap by half
TOKEN_MARGIN_S = 3.0  # tokens expected within this margin of a window are scored in it


def _window_attention(model, tokenizer, seg: np.ndarray, text_tokens: list[int]):
    """
    Cross-attention alignment evidence of *text_tokens* over *seg* (<= 30 s),
    computed exactly as whisper.timing.find_alignment does: alignment heads,
    softmax over frames, z-scored over tokens, median-filtered, head-averaged.
    Returns (matrix [len(text_tokens)+1, frames@50Hz], token probabilities).
    """
    import torch
    import whisper
    from whisper.audio import HOP_LENGTH
    from whisper.model import disable_sdpa
    from whisper.timing import median_filter

    mel = whisper.log_mel_spectrogram(torch.from_numpy(whisper.pad_or_trim(seg)), n_mels=model.dims.n_mels)
    n_frames = max(len(seg) // HOP_LENGTH // 2, 1)
    tokens = torch.tensor([*tokenizer.sot_sequence, tokenizer.no_timestamps, *text_tokens, tokenizer.eot])
    qks = [None] * model.dims.n_text_layer
    hooks = [blk.cross_attn.register_forward_hook(lambda _, ins, outs, i=i: qks.__setitem__(i, outs[-1][0]))
             for i, blk in enumerate(model.decoder.blocks)]
    try:
        with torch.no_grad(), disable_sdpa():
            logits = model(mel.unsqueeze(0), tokens.unsqueeze(0))[0]
    finally:
        for h in hooks:
            h.remove()
    n_sot = len(tokenizer.sot_sequence)
    probs = logits[n_sot:, : tokenizer.eot].softmax(dim=-1)
    tok_probs = probs[np.arange(len(text_tokens)), text_tokens].numpy()

    w = torch.stack([qks[l][h] for l, h in model.alignment_heads.indices().T])[:, :, :n_frames]
    w = w.softmax(dim=-1)
    std, mean = torch.std_mean(w, dim=-2, keepdim=True, unbiased=False)
    w = median_filter((w - mean) / std, 7)
    return w.mean(dim=0)[n_sot:-1].numpy(), tok_probs


def _global_forced(model, tokenizer, audio: np.ndarray, words: list[str],
                   anchors: dict[int, tuple[float, float]]) -> list[dict]:
    """
    Transcript-forced alignment over the WHOLE recording with one DTW:
      * fixed 30 s windows every 15 s (placed by time, not by content);
      * each window scores the tokens expected near it (from recognition
        anchors, +/- 3 s), so anchor errors below the margin are harmless;
      * window evidence is averaged into one token x frame matrix and a single
        monotonic DTW path assigns every token — no cut points to get wrong,
        and identical audio yields identical evidence.
    """
    from whisper.audio import TOKENS_PER_SECOND
    from whisper.timing import dtw_cpu

    dur = len(audio) / SR
    n_words = len(words)
    word_toks = [tokenizer.encode(" " + w) for w in words]
    text_tokens = [t for wt in word_toks for t in wt]
    tok_word = np.repeat(np.arange(n_words), [len(wt) for wt in word_toks])
    n_tok = len(text_tokens)

    # Expected time of each token (anchored words, interpolated otherwise)
    mids = np.array([(anchors[i][0] + anchors[i][1]) / 2 if i in anchors
                     else _interp_time(i, anchors, n_words, dur) + 0.15 for i in range(n_words)])
    mids = np.maximum.accumulate(mids)
    tok_t = mids[tok_word]

    n_frames = int(np.ceil(dur * TOKENS_PER_SECOND)) + 1
    acc = np.zeros((n_tok + 1, n_frames))
    cnt = np.zeros((n_tok + 1, n_frames))
    prob_sum, prob_cnt = np.zeros(n_tok), np.zeros(n_tok)

    starts = [0.0] if dur <= WINDOW_S else list(np.arange(0.0, dur - WINDOW_S + STRIDE_S, STRIDE_S))
    for ws in starts:
        we = min(ws + WINDOW_S, dur)
        a = int(np.searchsorted(tok_t, ws - TOKEN_MARGIN_S, side="left"))
        b = int(np.searchsorted(tok_t, we + TOKEN_MARGIN_S, side="right"))
        if b <= a:
            continue
        seg = audio[int(ws * SR): int(we * SR)]
        mat, probs = _window_attention(model, tokenizer, seg, text_tokens[a:b])
        f0 = int(round(ws * TOKENS_PER_SECOND))
        nf = min(mat.shape[1], n_frames - f0)
        acc[a:b + 1, f0:f0 + nf] += mat[:, :nf]
        cnt[a:b + 1, f0:f0 + nf] += 1
        prob_sum[a:b] += probs
        prob_cnt[a:b] += 1

    with np.errstate(invalid="ignore"):
        m = acc / cnt
    floor = np.nanmin(m) - 2.0 if np.isfinite(m).any() else -5.0
    m = np.where(cnt > 0, m, floor)                     # no evidence = strongly disfavoured

    text_idx, time_idx = dtw_cpu(np.ascontiguousarray(-m, dtype=np.float64))
    jumps = np.pad(np.diff(text_idx), (1, 0), constant_values=1).astype(bool)
    jump_t = time_idx[jumps] / TOKENS_PER_SECOND
    bounds = np.pad(np.cumsum([len(wt) for wt in word_toks]), (1, 0))
    probs = np.where(prob_cnt > 0, prob_sum / np.maximum(prob_cnt, 1), 0.0)
    out = []
    for i, w in enumerate(words):
        s_i, e_i = bounds[i], bounds[i + 1]
        out.append({
            "word": w,
            "start": float(jump_t[min(s_i, len(jump_t) - 1)]),
            "end": float(jump_t[min(e_i, len(jump_t) - 1)]),
            "score": float(np.mean(probs[s_i:e_i])) if e_i > s_i else 0.0,
        })
    return out


def _proportional(words: list[str], t0: float, t1: float, score: float) -> list[dict]:
    step = (t1 - t0) / max(len(words), 1)
    return [
        {"word": w, "start": t0 + i * step, "end": t0 + (i + 0.85) * step, "score": score}
        for i, w in enumerate(words)
    ]


def _rolling_percentile(x: np.ndarray, half: int, q: float, step: int = 5) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view
    padded = np.pad(x, half, mode="edge")
    win = sliding_window_view(padded, 2 * half + 1)[:, ::step]
    with np.errstate(all="ignore"):
        out = np.nanpercentile(win, q, axis=1)
    return np.where(np.isfinite(out), out, -100.0)


def local_silence_mask(db: np.ndarray, loud_half_s: float = 1.5, floor_half_s: float = 4.0) -> np.ndarray:
    """
    Silence relative to LOCAL statistics only — 35 dB below the loud speech within
    +/-1.5 s, but at least 6 dB above the noise floor within +/-4 s (digital-zero
    frames ignored). Because nothing depends on the rest of the clip, identical
    audio is always snapped identically, and a passage spoken 18 dB quieter is
    not mistaken for silence.
    """
    live = np.where(db > -90.0, db, np.nan)
    local_p95 = _rolling_percentile(live, max(int(loud_half_s / HOP_S), 1), 95)
    local_floor = _rolling_percentile(live, max(int(floor_half_s / HOP_S), 1), 5) + 6.0
    return db < np.maximum(local_p95 - 35.0, local_floor)


def snap_to_silences(words: list[dict], db: np.ndarray, dur: float, min_sil_s: float = 0.05) -> list[dict]:
    """
    Move each inter-word boundary onto the longest silent run between the two
    word midpoints, so inserted or natural pauses become explicit gaps.
    """
    if not words:
        return words
    silent = local_silence_mask(db)
    out = [dict(w) for w in words]

    def longest_run(t0: float, t1: float) -> Optional[tuple[float, float]]:
        i0, i1 = max(int(t0 / HOP_S), 0), min(int(t1 / HOP_S) + 1, len(silent))
        best, cur = None, None
        for i in range(i0, i1):
            if silent[i] and cur is None:
                cur = i
            elif not silent[i] and cur is not None:
                if best is None or i - cur > best[1] - best[0]:
                    best = (cur, i)
                cur = None
        if cur is not None and (best is None or i1 - cur > best[1] - best[0]):
            best = (cur, i1)
        if best and (best[1] - best[0]) * HOP_S >= min_sil_s:
            return best[0] * HOP_S, best[1] * HOP_S
        return None

    # 1. Trim each word to its voiced content. Cross-attention DTW tends to
    #    stretch a word across an adjacent pause; keep the longest voiced run.
    for w in out:
        i0, i1 = max(int(w["start"] / HOP_S), 0), min(int(w["end"] / HOP_S) + 1, len(silent))
        runs, cur, last_voiced = [], None, None
        for i in range(i0, i1):
            if not silent[i]:
                if cur is None:
                    cur = i
                last_voiced = i
            elif cur is not None and i - last_voiced > 15:      # >= 150 ms of silence splits the word
                runs.append((cur, last_voiced + 1))
                cur = None
        if cur is not None:
            runs.append((cur, last_voiced + 1))
        best = max(runs, key=lambda r: r[1] - r[0]) if runs else None
        if best is not None and (best[1] - best[0]) * HOP_S >= 0.03:
            w["start"], w["end"] = best[0] * HOP_S, best[1] * HOP_S
        else:
            # DTW placed the word (almost) entirely inside a silence — typically a
            # short word between two pauses: move it to the nearest voiced run within 1 s.
            centre = int(((w["start"] + w["end"]) / 2) / HOP_S)
            reach = int(1.0 / HOP_S)
            voiced = np.flatnonzero(~silent[max(0, centre - reach): centre + reach + 1])
            if len(voiced):
                k = max(0, centre - reach) + int(voiced[np.argmin(np.abs(voiced + max(0, centre - reach) - centre))])
                a, b = k, k
                while a > 0 and not silent[a - 1]:
                    a -= 1
                while b + 1 < len(silent) and not silent[b + 1]:
                    b += 1
                w["start"], w["end"] = a * HOP_S, (b + 1) * HOP_S

    # 2. Leading / trailing silence
    first_mid = (out[0]["start"] + out[0]["end"]) / 2
    lead = longest_run(0.0, first_mid)
    if lead and lead[0] <= 0.02:
        out[0]["start"] = lead[1]
    last_mid = (out[-1]["start"] + out[-1]["end"]) / 2
    trail = longest_run(last_mid, dur)
    if trail and trail[1] >= dur - 0.02:
        out[-1]["end"] = trail[0]

    # 3. Inter-word boundaries: put each boundary on the longest silence between the words.
    for i in range(len(out) - 1):
        a, b = out[i], out[i + 1]
        mid_a, mid_b = (a["start"] + a["end"]) / 2, (b["start"] + b["end"]) / 2
        run = longest_run(mid_a, mid_b) if mid_b > mid_a else None
        if run:
            a["end"], b["start"] = min(a["end"], run[0]), max(b["start"], run[1])
        if a["end"] > b["start"]:
            m = (a["end"] + b["start"]) / 2
            a["end"] = b["start"] = m

    # Enforce monotonic, positive-duration intervals
    for i, w in enumerate(out):
        if i > 0:
            w["start"] = max(w["start"], out[i - 1]["end"])
        w["end"] = max(w["end"], w["start"] + 0.03)
        w["start"], w["end"] = round(w["start"], 3), round(min(w["end"], dur), 3)
    return out


def _whisper_align(audio: np.ndarray, transcript: Optional[str], model_name: str) -> tuple[list[dict], str]:
    """Forced (pre-snap) word timings. Silence snapping is applied in align()."""
    from whisper.tokenizer import get_tokenizer

    model = _get_model(model_name)
    tokenizer = get_tokenizer(model.is_multilingual, language="en", task="transcribe")
    dur = len(audio) / SR
    db = frame_db(audio)
    tokens = transcript_words(transcript or "")

    prompt = " ".join(tokens[:40]) if tokens else None
    recognized = _recognize(model, audio, prompt)
    if not tokens:
        # No transcript: fall back to the recognized words (free decoding).
        return [dict(r, score=0.7) for r in recognized], f"whisper-{model_name}-recognized"

    anchors = _anchor(tokens, recognized)
    words = _global_forced(model, tokenizer, audio, tokens, anchors)
    return words, f"whisper-{model_name}-forced"


def _acoustic_fallback(audio: np.ndarray, transcript: Optional[str]) -> list[dict]:
    """Energy-VAD proportional placement. Only used when Whisper is unavailable."""
    dur = len(audio) / SR
    tokens = transcript_words(transcript or "") or [f"word_{i + 1}" for i in range(max(int(dur * 2.5), 1))]
    db = frame_db(audio)
    speech = db >= silence_threshold_db(db)
    segs, cur = [], None
    for i, s in enumerate(speech):
        if s and cur is None:
            cur = i
        elif not s and cur is not None:
            if (i - cur) * HOP_S >= 0.1:
                segs.append([cur * HOP_S, i * HOP_S])
            cur = None
    if cur is not None:
        segs.append([cur * HOP_S, dur])
    merged: list[list[float]] = []
    for s, e in segs:
        if merged and s - merged[-1][1] < 0.15:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    if not merged:
        merged = [[0.0, dur]]
    total = sum(e - s for s, e in merged)
    out, k = [], 0
    for j, (s, e) in enumerate(merged):
        n = len(tokens) - k if j == len(merged) - 1 else int(round(len(tokens) * (e - s) / total))
        out.extend(_proportional(tokens[k:k + n], s, e, score=0.0))
        k += n
    return out


# -- Cache ------------------------------------------------------------------------

def _cache_location(audio_path: str, key: str) -> Path:
    p = Path(audio_path).resolve()
    try:
        p.relative_to(DATASET_DIR.resolve())
        return DATASET_CACHE_DIR / f"{p.stem}.json"     # versioned with the dataset
    except ValueError:
        return RUNTIME_CACHE_DIR / f"{key}.json"        # uploads: gitignored runtime cache


def _cache_key(audio: np.ndarray, transcript: Optional[str], model_name: str) -> str:
    h = hashlib.sha1()
    h.update(audio.tobytes())
    h.update(" ".join(transcript_words(transcript or "")).lower().encode())
    h.update(f"{ALIGNER_VERSION}|{model_name}".encode())
    return h.hexdigest()[:20]


# -- Public API -----------------------------------------------------------------

def align(
    audio_path: str,
    transcript: Optional[str] = None,
    device: str = "cpu",
    cfg: Optional[dict] = None,
    use_cache: bool = True,
) -> AlignmentResult:
    """
    Force *transcript* onto *audio_path*. Every transcript token receives exactly
    one WordAlignment, in order (see ``transcript_words``).
    """
    if cfg is None:
        cfg = _load_cfg()
    a_cfg = cfg.get("aligner", {})
    model_name = a_cfg.get("model", "base.en")
    uncertain_threshold = float(a_cfg.get("uncertain_threshold", 0.25))

    audio = _load_audio_16k(audio_path)
    key = _cache_key(audio, transcript, model_name)
    cache_path = _cache_location(audio_path, key)

    raw: Optional[list[dict]] = None
    model_used = ""
    if use_cache and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("key") == key:
                raw, model_used = cached["words"], cached["model_used"]
        except Exception:
            raw = None

    if raw is None:
        try:
            raw, model_used = _whisper_align(audio, transcript, model_name)
        except ImportError:
            logger.warning("openai-whisper not installed — using acoustic fallback (all words uncertain)")
            raw, model_used = _acoustic_fallback(audio, transcript), "acoustic_fallback"
        if use_cache:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    json.dumps({"key": key, "model_used": model_used, "snapped": False, "words": raw}, indent=1),
                    encoding="utf-8",
                )
            except OSError:
                pass

    # Snapping is cheap and deterministic, so it runs on every load: its
    # parameters can change without re-running Whisper.
    raw = snap_to_silences(raw, frame_db(audio), len(audio) / SR)
    words = [
        WordAlignment(
            word=w["word"], start=float(w["start"]), end=float(w["end"]),
            score=float(w["score"]),
            uncertain=float(w["score"]) < uncertain_threshold,
        )
        for w in raw
    ]

    uncertain_spans: list[tuple[float, float]] = []
    span_start: Optional[float] = None
    for i, w in enumerate(words):
        if w.uncertain and span_start is None:
            span_start = w.start
        elif not w.uncertain and span_start is not None:
            uncertain_spans.append((span_start, words[i - 1].end))
            span_start = None
    if span_start is not None:
        uncertain_spans.append((span_start, words[-1].end))

    return AlignmentResult(
        words=words,
        model_used=model_used,
        had_retry=False,
        uncertain_spans=uncertain_spans,
        transcript=transcript or "",
    )
