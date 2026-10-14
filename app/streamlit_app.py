"""
app/streamlit_app.py
---------------------
Cadence — Speech Delivery Analytics Dashboard

Pipeline (pipeline/analysis.py, shared with the evaluation scripts):
  1. Upload audio + transcript (or pick a preset)
  2. Mode detection — A: transcript matches a bundled speech (paired GOOD baseline)
                      B: anything else (deviations from the speaker's own norm)
  3. Transcript-forced word alignment (Whisper cross-attention DTW + silence snapping)
  4. Feature extraction (STFT, MFCC, F0, HNR, energy, tempo, pauses) + self-normalization
  5. Flaw grounding: per-word deltas, thresholds tau = max(floor, k * noise)
  6. Causal explanations (deterministic, from measured numbers)
  7. Rubric scoring (NaN-safe)

Run: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.analysis import analyze, load_bundled, load_config  # noqa: E402

st.set_page_config(page_title="Cadence — Speech Analytics", page_icon="🎙️", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; }
  h2 { font-size: 1.3rem; font-weight: 600; margin-top: 1.4rem; }
  .mode-badge { display:inline-block; padding:3px 12px; border-radius:20px; font-size:0.82rem; font-weight:600; }
  .mode-a { background:#2d4a2d; color:#7ec87e; }
  .mode-b { background:#2d3a4a; color:#7eb8e8; }
  .heldout-badge { background:#4a2d2d; color:#e87e7e; padding:1px 8px; border-radius:10px; }
  .flaw-card { background:#1a1d2e; border-left:4px solid #f38ba8; border-radius:6px;
               padding:0.7rem 1rem; margin:0.5rem 0; font-size:0.9rem; color:#cdd6f4; }
  .info-card { background:#1c1a24; border-left:4px solid #f9e2af; border-radius:8px;
               padding:0.8rem 1.1rem; margin:0.8rem 0; font-size:0.88rem; color:#cdd6f4; }
  .muted { color:#888; font-size:0.78rem; }
</style>
""", unsafe_allow_html=True)

FLAW_COLORS = {
    "pitch": "rgba(243,139,168,0.30)", "energy": "rgba(249,226,175,0.30)",
    "rate": "rgba(137,220,235,0.30)", "pause": "rgba(166,227,161,0.35)",
    "clarity": "rgba(203,166,247,0.30)",
}
FLAW_NAMES = {"pitch": "Pitch / intonation", "energy": "Volume", "rate": "Pacing",
              "pause": "Pauses", "clarity": "Clarity"}
DARK = dict(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117", font=dict(color="#cdd6f4", size=11))
AUDIO = ROOT / "dataset" / "audio"
REAL = AUDIO / "real_speakers"


# --- Data helpers ---------------------------------------------------------------------

@st.cache_data
def heldout_results() -> dict | None:
    p = ROOT / "dataset" / "labels" / "heldout_results.json"
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def real_meta() -> dict:
    p = ROOT / "dataset" / "labels" / "real_speakers.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _to_wav(audio_bytes: bytes) -> str:
    """Decode any supported upload to a 16 kHz mono WAV temp file."""
    import soundfile as sf
    try:
        y, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    except Exception:
        import librosa
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as t:
            t.write(audio_bytes)
        y, sr = librosa.load(t.name, sr=None, mono=True)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != 16000:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, y, 16000, subtype="PCM_16")
    return tmp.name


@st.cache_data(show_spinner=False, max_entries=16)
def run_pipeline(audio_bytes: bytes, transcript: str, baseline: str | None, force_b: bool):
    path = _to_wav(audio_bytes)
    try:
        return analyze(path, transcript, cfg=load_config(), baseline_override=baseline,
                       force_mode_b=force_b), None
    except Exception as exc:  # surface errors in the UI
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        Path(path).unlink(missing_ok=True)


# --- Plot helpers ----------------------------------------------------------------------

def _warp_points(res):
    """Paired word midpoints (baseline time, upload time) used to time-warp the baseline."""
    if res.baseline is None:
        return None
    uw, bw = res.upload.norm.words, res.baseline.norm.words
    pts = sorted({((bw[j].start + bw[j].end) / 2, (uw[i].start + uw[i].end) / 2)
                  for i, j in enumerate(res.grounding.pairs) if j is not None})
    if len(pts) < 2:
        return None
    bx, ux = np.array(pts).T
    keep = np.r_[True, np.diff(bx) > 0]
    return bx[keep], ux[keep]


def _shade(fig, res, feature):
    for r in res.grounding.flaw_regions:
        if r.flaw_type == feature:
            fig.add_vrect(x0=r.start_s, x1=r.end_s, fillcolor=FLAW_COLORS[feature], line_width=0)
    for s, e in res.upload.alignment.uncertain_spans:
        fig.add_vrect(x0=s, x1=e, fillcolor="rgba(150,150,150,0.12)", line_width=0)


def _layout(fig, title, ytitle, dur):
    fig.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0), title=dict(text=title, font=dict(size=13)),
                      xaxis=dict(title="Time in uploaded clip (s)", range=[0, dur], gridcolor="#2a2d3e"),
                      yaxis=dict(title=ytitle, gridcolor="#2a2d3e"),
                      legend=dict(orientation="h", y=1.12, x=1, xanchor="right"), **DARK)
    return fig


def contour_chart(res, feature: str) -> go.Figure:
    up, base, dur = res.upload, res.baseline, res.duration_s
    warp = _warp_points(res)
    fig = go.Figure()

    if feature in ("pitch", "energy"):
        hop = up.norm.frame_hop_s
        yu = np.array(up.norm.f0_contour_st if feature == "pitch" else up.norm.rms_contour_db_rel, float)
        if feature == "energy":
            yu = np.convolve(yu, np.ones(5) / 5, mode="same")
        fig.add_trace(go.Scatter(x=np.arange(len(yu)) * hop, y=yu, name="Participant",
                                 line=dict(color="#89dceb", width=1.4), connectgaps=False))
        if base is not None and warp is not None:
            yb = np.array(base.norm.f0_contour_st if feature == "pitch" else base.norm.rms_contour_db_rel, float)
            if feature == "energy":
                yb = np.convolve(yb, np.ones(5) / 5, mode="same")
            tb = np.interp(np.arange(len(yb)) * hop, *warp)
            fig.add_trace(go.Scatter(x=tb, y=yb, name="GOOD baseline (time-aligned)",
                                     line=dict(color="#a6e3a1", width=1.2, dash="dot"), connectgaps=False))
        ytitle = "semitones re. own median F0" if feature == "pitch" else "dB re. own loud speech"
    elif feature == "rate":
        uw = up.norm.words
        fig.add_trace(go.Scatter(x=[(w.start + w.end) / 2 for w in uw], y=[w.rate_wps for w in uw],
                                 name="Participant", mode="lines+markers", marker=dict(size=3),
                                 line=dict(color="#89dceb", width=1.4)))
        if base is not None:
            bw = base.norm.words
            xs = [(uw[i].start + uw[i].end) / 2 for i, j in enumerate(res.grounding.pairs) if j is not None]
            ys = [bw[j].rate_wps for j in res.grounding.pairs if j is not None]
            fig.add_trace(go.Scatter(x=xs, y=ys, name="GOOD baseline (same words)",
                                     line=dict(color="#a6e3a1", width=1.2, dash="dot")))
        ytitle = "words per second of speech"
    elif feature == "pause":
        uw = up.norm.words
        fig.add_trace(go.Bar(x=[w.start for w in uw], y=[w.pause_duration_s for w in uw], width=0.12,
                             name="Participant pause before word", marker_color="#89dceb"))
        if base is not None:
            bw = base.norm.words
            fig.add_trace(go.Scatter(
                x=[uw[i].start for i, j in enumerate(res.grounding.pairs) if j is not None],
                y=[bw[j].pause_duration_s for j in res.grounding.pairs if j is not None],
                mode="markers", name="GOOD baseline pause", marker=dict(color="#a6e3a1", size=5, symbol="line-ew-open")))
        ytitle = "silence before word (s)"
    else:  # clarity
        uw = up.norm.words
        fig.add_trace(go.Scatter(x=[(w.start + w.end) / 2 for w in uw], y=[w.clarity for w in uw],
                                 name="Participant", line=dict(color="#89dceb", width=1.2)))
        if base is not None:
            bw = base.norm.words
            fig.add_trace(go.Scatter(
                x=[(uw[i].start + uw[i].end) / 2 for i, j in enumerate(res.grounding.pairs) if j is not None],
                y=[bw[j].clarity for j in res.grounding.pairs if j is not None],
                name="GOOD baseline (same words)", line=dict(color="#a6e3a1", width=1.2, dash="dot")))
        ytitle = f"{load_config()['normalization']['clarity_feature']} re. own median"
    _shade(fig, res, feature)
    fig = _layout(fig, FLAW_NAMES[feature], ytitle, dur)
    if feature == "pitch":
        fig.update_yaxes(range=[-12, 12])      # octave errors from the F0 tracker would squash the scale
    elif feature == "energy":
        fig.update_yaxes(range=[-50, 6])
    return fig


def delta_chart(res, feature: str) -> go.Figure:
    uw = res.upload.norm.words
    d = res.grounding.word_deltas.get(feature, [])
    tau = res.grounding.thresholds.get(feature, 0.0)
    x = [(w.start + w.end) / 2 for w in uw]
    flags = res.grounding.word_deviant.get(feature, [False] * len(d))
    fig = go.Figure(go.Scatter(x=x, y=d, mode="lines+markers", name="per-word delta",
                               line=dict(color="#cdd6f4", width=1),
                               marker=dict(size=[7 if f else 3 for f in flags],
                                           color=["#f38ba8" if f else "#89dceb" for f in flags])))
    for s in (tau, -tau):
        fig.add_hline(y=s, line=dict(color="#f9e2af", dash="dash", width=1))
    _shade(fig, res, feature)
    unit = {"pitch": "log2 ratio of pitch variability", "energy": "dB", "rate": "log2 tempo ratio",
            "pause": "seconds", "clarity": "clarity units"}[feature]
    fig = _layout(fig, f"{FLAW_NAMES[feature]} — delta vs threshold τ = ±{tau:.2f}", unit, res.duration_s)
    fig.update_layout(height=200, showlegend=False)
    return fig


def ruler_chart(res) -> go.Figure:
    words = res.upload.alignment.words
    fig = go.Figure(go.Scatter(
        x=[(w.start + w.end) / 2 for w in words], y=[0.5] * len(words), mode="text+markers",
        text=[w.word for w in words], textposition="top center",
        marker=dict(color=["#888" if w.uncertain else "#a0c4ff" for w in words], size=6, symbol="line-ns"),
        textfont=dict(size=9), hovertemplate="%{text}<br>%{x:.2f}s<extra></extra>"))
    for r in res.grounding.flaw_regions:
        fig.add_vrect(x0=r.start_s, x1=r.end_s, fillcolor=FLAW_COLORS[r.flaw_type], line_width=0)
    for s, e in res.upload.alignment.uncertain_spans:
        fig.add_vrect(x0=s, x1=e, fillcolor="rgba(150,150,150,0.15)", line_width=0)
    fig.update_layout(height=120, margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
                      xaxis=dict(range=[0, res.duration_s], title="Time (s)", gridcolor="#2a2d3e"),
                      yaxis=dict(showticklabels=False, range=[-0.5, 1.5]), **DARK)
    return fig


# --- Presets -------------------------------------------------------------------------------

bundled, splits = load_bundled()
meta = real_meta()


def _presets() -> dict:
    p = {"Custom upload": None}
    for key, m in meta.items():
        tid = int(m["text_id"])
        good = REAL / f"real{tid:02d}_good.wav"
        if good.exists():
            p[f"🧑 Human reader — {m['title'].split('(')[0].strip()} (vs TTS baseline)"] = \
                {"path": good, "text_id": tid, "real": key}
            for suf, name in (("pause_egregious", "inserted pauses"), ("pitch_mod", "monotone stretch"),
                              ("pacing_mod", "slowed passage")):
                f = REAL / f"real{tid:02d}_{suf}.wav"
                if f.exists():
                    p[f"🧑 Human reader — {name} (text {tid})"] = {"path": f, "text_id": tid, "real": key,
                                                                    "same_speaker": good}
    for tid, who in ((9, "JFK — Moon speech"), (10, "Reagan — 1981 inaugural"), (11, "Eisenhower — 1953 inaugural")):
        for suf, name in (("good", "original"), ("pause_egregious", "inserted pauses"),
                          ("pitch_mod", "monotone stretch"), ("pacing_mod", "slowed passage")):
            f = AUDIO / f"text_{tid:02d}_{suf}.wav"
            if f.exists():
                p[f"🎤 {who}: {name}"] = {"path": f, "text_id": tid}
    for tid, suf, name in ((6, "pause_mod", "unplanned pause"), (6, "clarity_mod", "slurred passage"),
                           (8, "pacing_mod", "slowed passage"), (8, "volume_mod", "volume drop"),
                           (8, "combined_egregious", "combined flaws")):
        f = AUDIO / f"text_{tid:02d}_{suf}.wav"
        if f.exists():
            p[f"🧪 Held-out text {tid} — {name}"] = {"path": f, "text_id": tid}
    return p


PRESETS = _presets()

with st.sidebar:
    st.markdown("## 🎙️ Cadence")
    choice = st.selectbox("Preset or upload", list(PRESETS))
    preset = PRESETS[choice]
    audio_bytes, transcript_default = None, ""
    if preset:
        audio_bytes = preset["path"].read_bytes()
        # Human recordings use what the reader actually says (may differ from the bundled excerpt)
        reader_tx = meta.get(preset.get("real"), {}).get("transcript_path")
        transcript_default = ((ROOT / reader_tx).read_text(encoding="utf-8").strip() if reader_tx
                              else bundled.get(preset["text_id"], ""))
        st.caption(f"Loaded `{preset['path'].name}`")
    upload = st.file_uploader("Audio (WAV / FLAC / MP3)", type=["wav", "flac", "mp3", "ogg"])
    if upload is not None:
        audio_bytes, transcript_default = upload.read(), ""
    transcript = st.text_area("Transcript (exact words spoken)", value=transcript_default, height=140,
                              key=f"tx_{choice}")
    reference = "Bundled GOOD baseline (TTS)"
    if preset and preset.get("same_speaker"):
        reference = st.radio("Reference recording",
                             ["Same speaker's clean recording", "Bundled GOOD baseline (TTS)"])
    force_b = st.checkbox("Force Mode B (ignore bundled baselines)")
    run = st.button("🚀 Analyze", type="primary", use_container_width=True)
    st.divider()
    st.markdown("**Mode A** — the transcript matches a bundled speech: every word is compared with the "
                "same word in the GOOD baseline.  \n**Mode B** — any other speech: deviations from the "
                "speaker's own typical delivery.")
    show_eval = st.checkbox("Show held-out evaluation", value=True)

st.markdown("# 🎙️ Cadence — Speech Delivery Analytics")
st.caption("Finds, timestamps and explains delivery flaws by comparing a recording with a reference reading.")

if run:
    if not audio_bytes or not transcript.strip():
        st.error("Provide both audio and its transcript.")
    else:
        baseline = str(preset["same_speaker"]) if (preset and preset.get("same_speaker")
                                                   and reference.startswith("Same")) else None
        with st.spinner("Aligning, extracting features and grounding flaws…"):
            res, err = run_pipeline(audio_bytes, transcript.strip(), baseline, force_b)
        st.session_state.update(result=res, error=err, audio=audio_bytes, preset=preset,
                                args=(audio_bytes, transcript.strip(), baseline, force_b))

res, err = st.session_state.get("result"), st.session_state.get("error")
if err:
    st.error(f"Pipeline error: {err}")
elif res is None:
    hr = heldout_results()
    if hr:
        s = hr["heldout_tts"]["summary"]
        c = st.columns(4)
        c[0].metric("Held-out mean target IoU", f"{s['mean_target_iou']:.3f}",
                    help="Texts 6 & 8, all severity tiers; never used for calibration")
        c[1].metric("Mean false-positive seconds / clip", f"{s['mean_fp_seconds']:.1f}")
        c[2].metric("Severity gradient monotonic", hr.get("severity_gradient_monotonic", "–"))
        c[3].metric("Clips evaluated", s["n_clips"])
    st.info("Pick a preset or upload audio + transcript in the sidebar, then click **Analyze**.")
else:
    g = res.grounding
    badge = (f"Mode A — paired with text #{res.matched_id}" if res.mode == "A"
             else "Mode B — compared with the speaker's own typical delivery")
    if res.matched_split == "heldout":
        badge += " <span class='heldout-badge'>HELD-OUT TEXT</span>"
    if res.mode == "A":
        path = ("edited copy of the reference recording → calibrated floors" if g.same_recording
                else "different recording → thresholds widened by the natural variation between readings")
        badge += f" · {path}"
    st.markdown(f"<span class='mode-badge mode-{res.mode.lower()}'>{badge}</span> "
                f"<span class='muted'>aligner: {res.upload.alignment.model_used}</span>", unsafe_allow_html=True)

    p = st.session_state.get("preset") or {}
    if p.get("real"):
        m = meta.get(p["real"], {})
        ref = "the same reader's clean recording" if (p.get("same_speaker") and
              st.session_state["args"][2]) else "the TTS GOOD baseline (a different voice)"
        st.markdown(f"""<div class='info-card'><b>Human recording</b> — {m.get('title')}, read by
{m.get('reader')} for LibriVox (public domain; <a href='{m.get('source_url')}' target='_blank'>source</a>),
trimmed to the transcript ({m.get('trim_s', ['?', '?'])[0]}–{m.get('trim_s', ['?', '?'])[1]} s of the original).
Compared with {ref}. When the reference is a different speaker, thresholds widen automatically to the natural
variation between the two readings, so only deviations beyond ordinary stylistic differences are reported.</div>""",
                    unsafe_allow_html=True)

    st.markdown("## 🔊 Audio & alignment")
    a, b = st.columns([2, 3])
    a.audio(st.session_state["audio"])
    b.plotly_chart(ruler_chart(res), use_container_width=True, config={"displayModeBar": False}, key="ruler")
    if res.upload.alignment.uncertain_spans:
        b.markdown("<span class='muted'>Grey bands: words whose forced-alignment confidence is low.</span>",
                   unsafe_allow_html=True)

    st.markdown("## 📈 Participant vs baseline")
    tabs = st.tabs([FLAW_NAMES[f] for f in ("pitch", "energy", "rate", "pause", "clarity")])
    for tab, feat in zip(tabs, ("pitch", "energy", "rate", "pause", "clarity")):
        with tab:
            st.plotly_chart(contour_chart(res, feat), use_container_width=True, key=f"contour_{feat}")
            st.plotly_chart(delta_chart(res, feat), use_container_width=True, key=f"delta_{feat}")

    st.markdown("## 🔍 Flaw regions and causes")
    if not g.flaw_regions:
        st.success("No delivery deviation exceeded its threshold.", icon="✅")
    for i, (r, e) in enumerate(zip(sorted(g.flaw_regions, key=lambda r: r.start_s), res.explanations)):
        color = {"mild": "#f9e2af", "moderate": "#fab387", "severe": "#f38ba8"}[e.severity_label]
        st.markdown(f"""<div class='flaw-card' style='border-left-color:{color}'>
<b>{FLAW_NAMES[r.flaw_type].upper()}</b> · {r.start_s:.2f}–{r.end_s:.2f} s · {r.direction} ·
severity {r.severity:.1f}× threshold<br>{e.polished_text or e.template_text}</div>""", unsafe_allow_html=True)
        with st.expander("Evidence", expanded=False):
            st.json(e.evidence)

    st.markdown("## 🏆 Rubric")
    rb = res.rubric
    st.markdown(f"### Overall {rb.overall:.1f} / 100")
    cols = st.columns(5)
    for c, dim in zip(cols, rb.dimensions):
        c.metric(dim.dimension, f"{dim.score:.1f} / 20",
                 help=f"{dim.region_count} region(s), {dim.flawed_duration_s:.1f}s flagged")

    if st.button("🔄 Re-run (determinism check)"):
        run_pipeline.clear()
        again, _ = run_pipeline(*st.session_state["args"])
        if again and abs(again.rubric.overall - rb.overall) < 1e-9 and \
                [(x.flaw_type, x.start_s, x.end_s) for x in again.grounding.flaw_regions] == \
                [(x.flaw_type, x.start_s, x.end_s) for x in g.flaw_regions]:
            st.success("Identical regions and scores on re-run.")
        else:
            st.error("Re-run differs — investigate.")

# --- Held-out evaluation panel --------------------------------------------------------------
if show_eval:
    st.divider()
    st.markdown("## 📊 Held-out evaluation")
    hr = heldout_results()
    if not hr:
        st.info("Run `python scripts/05_evaluate_heldout.py` to generate results.")
    else:
        import pandas as pd
        cfg_used = hr.get("config", {})
        st.caption(f"Thresholds calibrated on texts 1–5, 7 only. clarity measure: "
                   f"{cfg_used.get('clarity_feature')}, k_noise: {cfg_used.get('k_noise')}. "
                   "Target IoU = overlap of detections of the injected flaw's type with ground truth; "
                   "FP = seconds flagged outside ground truth of the same type (any type).")
        for key, title in (("heldout_tts", "TTS held-out texts 6 & 8"),
                           ("real_same_speaker", "Human recordings — same-speaker reference"),
                           ("real_cross_speaker", "Human recordings vs TTS baseline (cross-speaker)")):
            if key not in hr:
                continue
            s = hr[key]["summary"]
            st.markdown(f"**{title}** — mean target IoU {s['mean_target_iou']}, "
                        f"mean FP {s['mean_fp_seconds']} s per clip, {s['n_clips']} clips")
            df = pd.DataFrame([{"flaw / tier": k, **v} for k, v in s["by_flaw_tier"].items()])
            if not df.empty:
                st.dataframe(df.set_index("flaw / tier"), use_container_width=True)
        if "real_cross_speaker" in hr:
            st.markdown("**Clean human reading vs TTS baseline** (should be near 100 / few flags)")
            st.dataframe(pd.DataFrame(hr["real_cross_speaker"]["clean_pairs"]), use_container_width=True)
        st.markdown(f"**Severity gradient** (score mild ≥ moderate ≥ egregious): "
                    f"{hr.get('severity_gradient_monotonic')}")
