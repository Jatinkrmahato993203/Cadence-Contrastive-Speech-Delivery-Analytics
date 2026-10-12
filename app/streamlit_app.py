"""
app/streamlit_app.py
---------------------
Cadence — Speech Delivery Analytics Dashboard

Full pipeline:
  1. Upload audio + transcript
  2. Mode detection (A = paired, B = general)
  3. WhisperX alignment (with confidence retry)
  4. Feature extraction + speaker-agnostic normalization
  5. Flaw grounding (Mode A: word-index diff vs GOOD baseline;
                     Mode B: vs pooled normative corridor)
  6. Causal explanation generation
  7. Rubric scoring (NaN-safe)
  8. UI: waveform ruler, time-series overlay, flaw regions, scorecard,
          held-out evaluation panel

Deploy: streamlit run app/streamlit_app.py
        or push to Streamlit Community Cloud (requirements.txt + packages.txt)
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# --- Page config --------------------------------------------------------------
st.set_page_config(
    page_title="Cadence — Speech Analytics",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Global CSS ---------------------------------------------------------------
st.markdown("""
<style>
  .main { background: #0f1117; }
  .block-container { padding-top: 1.5rem; }
  h1 { font-size: 2.1rem; font-weight: 700; letter-spacing: -0.5px; }
  h2 { font-size: 1.3rem; font-weight: 600; color: #a0c4ff; margin-top: 1.5rem; }
  h3 { font-size: 1.05rem; font-weight: 600; color: #cdd6f4; }
  .stMetric { background: #1e2130; border-radius: 10px; padding: 0.5rem 1rem; }
  .mode-badge {
    display: inline-block; padding: 3px 12px; border-radius: 20px;
    font-size: 0.82rem; font-weight: 600; letter-spacing: 0.5px;
  }
  .mode-a { background: #2d4a2d; color: #7ec87e; }
  .mode-b { background: #2d3a4a; color: #7eb8e8; }
  .heldout-badge { background: #4a2d2d; color: #e87e7e; }
  .uncertain-note { font-size: 0.78rem; color: #888; font-style: italic; }
  .flaw-card {
    background: #1a1d2e; border-left: 3px solid #f38ba8;
    border-radius: 6px; padding: 0.7rem 1rem; margin: 0.5rem 0;
    font-size: 0.9rem;
  }
  .score-good { color: #a6e3a1; }
  .score-mid  { color: #f9e2af; }
  .score-bad  { color: #f38ba8; }
</style>
""", unsafe_allow_html=True)

# --- Helpers ------------------------------------------------------------------

@st.cache_resource
def load_config():
    import yaml
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_bundled_transcripts() -> tuple[dict, dict]:
    """Load all 8 bundled transcripts and their split labels."""
    transcripts, splits = {}, {}
    split_map = {1:"calibration",2:"calibration",3:"calibration",4:"calibration",
                 5:"calibration",6:"heldout",7:"calibration",8:"heldout"}
    for i in range(1, 9):
        p = ROOT / "dataset" / "transcripts" / f"text_{i:02d}.txt"
        if p.exists():
            transcripts[i] = p.read_text().strip()
            splits[i] = split_map[i]
    return transcripts, splits


@st.cache_data
def load_heldout_results() -> dict | None:
    p = ROOT / "dataset" / "labels" / "heldout_results.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def load_good_baseline_audio(text_id: int) -> Path | None:
    p = ROOT / "dataset" / "audio" / f"text_{text_id:02d}_good.wav"
    return p if p.exists() else None


@st.cache_data(show_spinner=False)
def run_full_pipeline(audio_bytes: bytes, transcript: str) -> dict:
    """
    Main pipeline: alignment → features → normalize → ground → explain → score.
    Returns a dict of all results for the UI.
    """
    import soundfile as sf
    import yaml

    cfg = load_config()
    bundled, splits = load_bundled_transcripts()

    # Write audio to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        # 1. Alignment
        from pipeline.aligner import align
        try:
            alignment = align(tmp_path, transcript=transcript, cfg=cfg)
        except Exception as e:
            return {"error": f"Alignment failed: {e}"}

        info = sf.info(tmp_path)
        duration_s = info.duration

        # 2. Features
        from pipeline import features as feat_mod
        clip_feat = feat_mod.extract(tmp_path, alignment)

        # 3. Mode detection
        from pipeline.grounder import detect_mode
        mode, matched_id, matched_split = detect_mode(transcript, bundled, splits)

        # 4. Normalize uploaded clip
        from pipeline import normalizer as norm_mod
        norm_clip, profile = norm_mod.normalize(clip_feat, "uploaded_speaker", cfg=cfg)

        # 5. Load + normalize baseline (Mode A)
        baseline_norm = None
        speaker_source = "librivox"
        if mode == "A" and matched_id:
            good_path = load_good_baseline_audio(matched_id)
            if good_path:
                try:
                    good_align = align(str(good_path), cfg=cfg)
                    good_feat = feat_mod.extract(str(good_path), good_align)
                    baseline_norm, _ = norm_mod.normalize(
                        good_feat, f"good_{matched_id}", cfg=cfg
                    )
                except Exception:
                    mode = "B"  # fallback if baseline unavailable

        # 6. Ground
        from pipeline.grounder import ground
        pooled_means = {"pitch":0.0,"energy":0.0,"rate":0.0,
                        "clarity_mfcc":0.0,"clarity_hnr":0.0}
        grounding = ground(
            norm_clip, mode=mode,
            matched_text_id=matched_id,
            matched_split=matched_split,
            baseline_norm_clip=baseline_norm,
            speaker_source=speaker_source,
            cfg=cfg,
            pooled_means=pooled_means,
        )

        # 7. Explain
        from pipeline.explainer import explain
        explanations = explain(grounding, norm_clip.words, use_llm=True)

        # 8. Score
        from pipeline.scorer import score
        rubric = score(grounding, total_duration_s=duration_s)

        return {
            "mode": mode,
            "matched_id": matched_id,
            "matched_split": matched_split,
            "duration_s": duration_s,
            "alignment": alignment,
            "norm_clip": norm_clip,
            "baseline_norm": baseline_norm,
            "grounding": grounding,
            "explanations": explanations,
            "rubric": rubric,
            "error": None,
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# --- Plot helpers --------------------------------------------------------------

FLAW_COLORS = {
    "pitch": "rgba(243,139,168,0.35)",
    "energy": "rgba(249,226,175,0.35)",
    "rate": "rgba(137,220,235,0.35)",
    "pause": "rgba(166,227,161,0.35)",
    "clarity": "rgba(203,166,247,0.35)",
    "combined": "rgba(250,179,135,0.35)",
}

FEATURE_LABELS = {
    "pitch": "Pitch (semitones from speaker median)",
    "energy": "Energy (dB relative to 95th-pct RMS)",
    "rate": "Speech Rate (z-score)",
}


def _build_overlay_chart(result: dict, feature: str) -> go.Figure:
    """Build a Plotly time-series overlay chart for one feature."""
    norm_clip = result["norm_clip"]
    grounding = result["grounding"]
    mode = result["mode"]

    # Build time axis (frame-level or word-level depending on feature)
    if feature == "pitch":
        y_up = norm_clip.f0_contour_st
    elif feature == "energy":
        y_up = norm_clip.rms_contour_db_rel
    else:
        y_up = norm_clip.rate_contour_zscore

    x_up = np.linspace(0, result["duration_s"], len(y_up)).tolist()

    fig = go.Figure()

    # Uploaded clip contour
    fig.add_trace(go.Scatter(
        x=x_up, y=y_up,
        name="Uploaded", line=dict(color="#89dceb", width=1.5),
    ))

    # Baseline or normative corridor
    if mode == "A" and result.get("baseline_norm"):
        bn = result["baseline_norm"]
        if feature == "pitch":
            y_base = bn.f0_contour_st
        elif feature == "energy":
            y_base = bn.rms_contour_db_rel
        else:
            y_base = bn.rate_contour_zscore
        x_base = np.linspace(0, result["duration_s"], len(y_base)).tolist()
        fig.add_trace(go.Scatter(
            x=x_base, y=y_base,
            name="GOOD Baseline", line=dict(color="#a6e3a1", width=1.5, dash="dash"),
        ))
    else:
        # Mode B: normative corridor (shaded band)
        corridor = grounding.normative_corridor.get(feature)
        if corridor:
            mu, half = corridor
            fig.add_hrect(
                y0=mu - half, y1=mu + half,
                fillcolor="rgba(166,227,161,0.12)",
                line_width=0,
                annotation_text="Expected range",
                annotation_position="top left",
                annotation=dict(font_size=10, font_color="#a6e3a1"),
            )

    # Flaw region shading
    for region in grounding.flaw_regions:
        if region.flaw_type in ("pitch", "energy", "rate", "clarity", "combined", "pause"):
            fig.add_vrect(
                x0=region.start_s, x1=region.end_s,
                fillcolor=FLAW_COLORS.get(region.flaw_type, "rgba(200,200,200,0.2)"),
                line_width=0,
            )

    # Uncertain spans (grey hatching)
    for span_start, span_end in result["alignment"].uncertain_spans:
        fig.add_vrect(
            x0=span_start, x1=span_end,
            fillcolor="rgba(150,150,150,0.15)",
            line_width=0,
            annotation_text="? uncertain alignment",
            annotation=dict(font_size=9, font_color="#888"),
        )

    fig.update_layout(
        height=220, margin=dict(l=0, r=0, t=25, b=0),
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        font=dict(color="#cdd6f4", size=11),
        xaxis=dict(title="Time (s)", gridcolor="#2a2d3e", showgrid=True),
        yaxis=dict(title=FEATURE_LABELS.get(feature, feature), gridcolor="#2a2d3e"),
        legend=dict(bgcolor="#1e2130", bordercolor="#2a2d3e", borderwidth=1),
        title=dict(text=FEATURE_LABELS.get(feature, feature).split("(")[0].strip(),
                   font=dict(size=13)),
    )
    return fig


def _score_color_class(score: float, max_score: float = 20.0) -> str:
    pct = score / max_score
    if pct >= 0.75:
        return "score-good"
    if pct >= 0.45:
        return "score-mid"
    return "score-bad"


def _score_bar(score: float, max_score: float = 20.0) -> go.Figure:
    pct = score / max_score
    color = "#a6e3a1" if pct >= 0.75 else "#f9e2af" if pct >= 0.45 else "#f38ba8"
    fig = go.Figure(go.Bar(
        x=[score], y=[""], orientation="h",
        marker=dict(color=color, line=dict(width=0)),
        width=0.5,
    ))
    fig.update_layout(
        height=40, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#1e2130", plot_bgcolor="#1e2130",
        xaxis=dict(range=[0, max_score], showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False),
        showlegend=False,
    )
    return fig


# --- Sidebar ------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🎙️ Cadence")
    st.markdown("**Speech Delivery Analytics**")
    st.divider()
    st.markdown("**Upload**")
    uploaded_audio = st.file_uploader(
        "Audio file (WAV, MP3, FLAC)", type=["wav", "mp3", "flac"],
        key="audio_upload"
    )
    uploaded_transcript = st.text_area(
        "Transcript", height=160, placeholder="Paste the speech transcript here…",
        key="transcript_input"
    )
    run_btn = st.button("🚀 Analyze", use_container_width=True, type="primary")
    st.divider()
    st.markdown("""
**Modes**
- 🎯 **Mode A** — transcript matches a bundled speech (direct comparison)
- 🌐 **Mode B** — arbitrary upload (normative corridor)

**Re-run** button proves determinism.
""")
    st.divider()
    show_heldout = st.checkbox("Show held-out evaluation panel", value=True)


# --- Main panel ---------------------------------------------------------------

st.markdown("# 🎙️ Cadence — Speech Delivery Analytics")
st.markdown(
    "Detects and timestamps delivery flaws in speech using contrastive acoustic analytics."
)

if not run_btn and not st.session_state.get("last_result"):
    # Landing state
    st.info(
        "Upload an audio file and paste the transcript in the sidebar, then click **🚀 Analyze**.",
        icon="ℹ️",
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dataset clips", "64", help="8 texts × 8 severity gradient points")
    with col2:
        st.metric("Flaw categories", "5", help="Pacing, Pitch, Pauses, Volume, Clarity")
    with col3:
        st.metric("Rubric max score", "100", help="Sum of 5 × 20-point dimensions")

elif run_btn:
    if not uploaded_audio or not uploaded_transcript.strip():
        st.error("Please upload an audio file AND paste the transcript before running.")
    else:
        with st.spinner("Running pipeline…"):
            audio_bytes = uploaded_audio.read()
            result = run_full_pipeline(audio_bytes, uploaded_transcript.strip())
            st.session_state["last_result"] = result
            st.session_state["last_transcript"] = uploaded_transcript.strip()
            st.session_state["last_audio"] = audio_bytes

# Render results
result = st.session_state.get("last_result")
if result and not result.get("error"):
    mode = result["mode"]
    matched_id = result["matched_id"]
    matched_split = result["matched_split"]
    alignment = result["alignment"]
    grounding = result["grounding"]
    explanations = result["explanations"]
    rubric = result["rubric"]

    # -- Mode badge ------------------------------------------------------------
    badge_cls = "mode-a" if mode == "A" else "mode-b"
    if mode == "A":
        badge_text = f"Mode A — paired match: Text #{matched_id}"
        if matched_split == "heldout":
            badge_text += " <span class='heldout-badge'>HELD-OUT</span>"
    else:
        badge_text = "Mode B — general analysis (normative corridor)"

    if alignment.had_retry:
        badge_text += f"  ·  ⚠️ WhisperX retry ({alignment.model_used})"

    st.markdown(
        f"<span class='mode-badge {badge_cls}'>{badge_text}</span>",
        unsafe_allow_html=True
    )

    # -- Audio player + word ruler ---------------------------------------------
    st.markdown("## 🔊 Audio")
    audio_col, ruler_col = st.columns([2, 3])
    with audio_col:
        st.audio(st.session_state["last_audio"])

    with ruler_col:
        # Word boundary ruler as Plotly scatter
        words = alignment.words
        if words:
            dur = result["duration_s"]
            word_x = [(w.start + w.end) / 2 for w in words]
            word_text = [w.word for w in words]
            uncertain_mask = [w.uncertain for w in words]

            colors = ["#888" if u else "#a0c4ff" for u in uncertain_mask]

            ruler_fig = go.Figure()
            ruler_fig.add_trace(go.Scatter(
                x=word_x, y=[0.5] * len(words),
                mode="text+markers",
                text=word_text,
                textposition="top center",
                marker=dict(color=colors, size=6, symbol="line-ns"),
                textfont=dict(size=9, color="#cdd6f4"),
                hovertemplate="%{text}<br>%{x:.2f}s<extra></extra>",
            ))

            # Flaw region bands on ruler
            for region in grounding.flaw_regions:
                ruler_fig.add_vrect(
                    x0=region.start_s, x1=region.end_s,
                    fillcolor=FLAW_COLORS.get(region.flaw_type, "rgba(200,200,200,0.2)"),
                    line_width=0,
                )

            # Uncertain spans
            for span_s, span_e in alignment.uncertain_spans:
                ruler_fig.add_vrect(
                    x0=span_s, x1=span_e,
                    fillcolor="rgba(150,150,150,0.15)",
                    line_width=0,
                )
                ruler_fig.add_annotation(
                    x=(span_s + span_e) / 2, y=0.1,
                    text="⚠️", showarrow=False,
                    font=dict(size=10, color="#888"),
                )

            ruler_fig.update_layout(
                height=120, margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                xaxis=dict(range=[0, dur], title="Time (s)",
                           gridcolor="#2a2d3e", showgrid=True,
                           tickfont=dict(color="#888", size=9)),
                yaxis=dict(showticklabels=False, range=[-0.5, 1.5]),
                showlegend=False,
            )
            st.plotly_chart(ruler_fig, use_container_width=True, config={"displayModeBar": False})

            if alignment.uncertain_spans:
                st.markdown(
                    "<span class='uncertain-note'>⚠️ Grey regions = low alignment confidence "
                    "(timestamps may be ±50ms in these spans)</span>",
                    unsafe_allow_html=True
                )

    # -- Time-series overlay ---------------------------------------------------
    st.markdown("## 📈 Acoustic Contours")
    tab_pitch, tab_energy, tab_rate = st.tabs(["🎵 Pitch", "🔊 Volume/Energy", "⚡ Speech Rate"])
    with tab_pitch:
        st.plotly_chart(_build_overlay_chart(result, "pitch"), use_container_width=True)
    with tab_energy:
        st.plotly_chart(_build_overlay_chart(result, "energy"), use_container_width=True)
    with tab_rate:
        st.plotly_chart(_build_overlay_chart(result, "rate"), use_container_width=True)

    # -- Flaw regions + causal explanations ------------------------------------
    st.markdown("## 🔍 Detected Flaw Regions")

    if not grounding.flaw_regions:
        st.success("No significant delivery flaws detected.", icon="✅")
    else:
        exp_map = {e.flaw_type: e for e in explanations}
        for region in grounding.flaw_regions:
            exp = exp_map.get(region.flaw_type)
            exp_text = exp.polished_text or exp.template_text if exp else f"Flaw detected in {region.flaw_type}"
            uncertain_note = " ⚠️ (alignment uncertain in this span)" if region.uncertain else ""
            st.markdown(f"""
<div class='flaw-card'>
  <strong>{region.flaw_type.upper()}</strong> &nbsp;
  {region.start_s:.2f}s – {region.end_s:.2f}s &nbsp;|&nbsp;
  direction: {region.direction} &nbsp;|&nbsp;
  Δ = {region.mean_delta:.2f}{uncertain_note}<br>
  <span style='color:#cdd6f4'>{exp_text}</span>
</div>
""", unsafe_allow_html=True)

    # -- Rubric scorecard ------------------------------------------------------
    st.markdown("## 🏆 Rubric Scorecard")

    overall_color = "#a6e3a1" if rubric.overall >= 75 else "#f9e2af" if rubric.overall >= 45 else "#f38ba8"
    st.markdown(
        f"### Overall Score: <span style='color:{overall_color}; font-size:1.8rem'>"
        f"{rubric.overall:.1f} / 100</span>",
        unsafe_allow_html=True
    )

    score_cols = st.columns(5)
    for i, dim in enumerate(rubric.dimensions):
        with score_cols[i]:
            cls = _score_color_class(dim.score)
            st.markdown(
                f"**{dim.dimension}**<br>"
                f"<span class='{cls}' style='font-size:1.4rem;font-weight:700'>"
                f"{dim.score:.1f}</span><span style='color:#555'>/20</span>",
                unsafe_allow_html=True
            )
            st.plotly_chart(_score_bar(dim.score), use_container_width=True,
                            config={"displayModeBar": False})
            if dim.region_count > 0:
                st.markdown(
                    f"<span style='color:#888;font-size:0.75rem'>"
                    f"{dim.region_count} region(s) · {dim.flawed_duration_s:.1f}s flawed</span>",
                    unsafe_allow_html=True
                )

    # Re-run button (proves determinism)
    if st.button("🔄 Re-run (prove determinism)", key="rerun"):
        rerun_result = run_full_pipeline(
            st.session_state["last_audio"],
            st.session_state["last_transcript"]
        )
        if rerun_result.get("rubric") and result.get("rubric"):
            if abs(rerun_result["rubric"].overall - rubric.overall) < 1e-6:
                st.success("✅ Re-run produced identical scores — pipeline is deterministic.")
            else:
                st.error(
                    f"❌ Scores differ: {rerun_result['rubric'].overall:.4f} vs "
                    f"{rubric.overall:.4f} — investigate!")

elif result and result.get("error"):
    st.error(f"Pipeline error: {result['error']}")

# --- Held-out evaluation panel -----------------------------------------------
if show_heldout:
    st.divider()
    st.markdown("## 📊 Held-Out Evaluation Panel")
    st.markdown(
        "Pre-computed temporal IoU accuracy on the **2 held-out texts** "
        "(texts #6 and #8), **14 BAD clips** with known-injected flaw regions. "
        "These clips were never used to compute any threshold or statistic. "
        "This is the non-circular evidence for the '25% flaw-timestamp accuracy' criterion."
    )

    heldout = load_heldout_results()
    if heldout is None:
        st.info(
            "No held-out results yet. Run `python scripts/05_evaluate_heldout.py` first.",
            icon="ℹ️"
        )
    else:
        summary = heldout.get("summary", {})
        overall = summary.get("overall", {})

        col_over, col_n = st.columns([2, 1])
        with col_over:
            st.metric(
                "Mean temporal IoU (all held-out clips)",
                f"{overall.get('mean_iou', 0):.3f}",
                help="1.0 = perfect overlap with ground-truth injected region"
            )
        with col_n:
            st.metric("Held-out clips evaluated", overall.get("n_clips", 0))

        # Per-flaw-type table
        rows = []
        for flaw, stats in summary.items():
            if flaw == "overall":
                continue
            rows.append({
                "Flaw Type": flaw,
                "Mean IoU": f"{stats['mean_iou']:.3f}",
                "Min IoU": f"{stats['min_iou']:.3f}",
                "Max IoU": f"{stats['max_iou']:.3f}",
                "N clips": stats["n_clips"],
            })

        if rows:
            import pandas as pd
            st.dataframe(pd.DataFrame(rows).set_index("Flaw Type"), use_container_width=True)

        # Bar chart per flaw type
        if rows:
            flaw_names = [r["Flaw Type"] for r in rows]
            iou_means = [float(r["Mean IoU"]) for r in rows]
            colors = ["#a6e3a1" if v >= 0.7 else "#f9e2af" if v >= 0.4 else "#f38ba8"
                      for v in iou_means]
            bar_fig = go.Figure(go.Bar(
                x=flaw_names, y=iou_means,
                marker=dict(color=colors, line=dict(width=0)),
                text=[f"{v:.3f}" for v in iou_means],
                textposition="outside",
            ))
            bar_fig.update_layout(
                height=280, margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                font=dict(color="#cdd6f4", size=11),
                yaxis=dict(range=[0, 1.05], title="Temporal IoU",
                           gridcolor="#2a2d3e"),
                xaxis=dict(gridcolor="#2a2d3e"),
                showlegend=False,
            )
            st.plotly_chart(bar_fig, use_container_width=True)

# --- Footer -------------------------------------------------------------------
st.divider()
st.markdown(
    "<div style='text-align:center;color:#555;font-size:0.78rem'>"
    "Cadence · Contrastive Speech Analytics · "
    "Dataset: 64 clips, 8 PD texts, 7 flaw types · "
    "Pipeline: MFA (offline) + WhisperX (online) · "
    "All thresholds from calibration-set GOOD clips only"
    "</div>",
    unsafe_allow_html=True
)
