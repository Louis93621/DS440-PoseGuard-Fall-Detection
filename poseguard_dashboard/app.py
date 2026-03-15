from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from infer import load_baseline_artifacts, predict_sequence  # noqa: E402
from postprocess import AlertConfig  # noqa: E402
from poseguard_core import KEYPOINTS, KEYPOINT_SLUG, SKELETON_EDGES  # noqa: E402

PALETTE = {
    "canvas": "#F6F3ED",
    "surface": "#FBF9F4",
    "surface_soft": "#F0EBE2",
    "surface_deep": "#E7E0D2",
    "border": "#DDD6C8",
    "text": "#1F1D19",
    "muted": "#6E6A60",
    "accent": "#355C52",
    "accent_soft": "#DDE6E1",
    "accent_line": "#587A71",
    "alert": "#7A5138",
    "alert_soft": "#EBD9CF",
}

st.set_page_config(
    page_title="PoseGuard",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css() -> None:
    css_path = APP_DIR / "assets" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_artifacts_cached(artifacts_path: str):
    return load_baseline_artifacts(Path(artifacts_path).expanduser())


@st.cache_data(show_spinner=False)
def discover_csv_catalog(data_root: str) -> pd.DataFrame:
    root = Path(data_root).expanduser()
    if not root.exists() or not root.is_dir():
        return pd.DataFrame(columns=["label", "scene", "pack", "filename", "path"])

    records: List[Dict[str, str]] = []
    for csv_path in root.rglob("*_keypoints.csv"):
        rel = csv_path.relative_to(root)
        parts = list(rel.parts)
        records.append(
            {
                "label": parts[0] if len(parts) > 0 else "Unknown",
                "scene": parts[1] if len(parts) > 1 else "Unknown",
                "pack": parts[2] if len(parts) > 2 else "",
                "filename": csv_path.name,
                "relative_path": str(rel),
                "path": str(csv_path),
            }
        )

    if not records:
        return pd.DataFrame(columns=["label", "scene", "pack", "filename", "path"])

    df = pd.DataFrame(records)
    return df.sort_values(["label", "scene", "pack", "filename"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_prediction_metadata_summary(artifacts_path: str) -> Dict[str, Any]:
    artifacts = load_artifacts_cached(artifacts_path)
    return {
        "model_name": artifacts.model_name,
        "feature_count": len(artifacts.feature_columns),
        "window": int(artifacts.config.get("window", 30)),
        "stride": int(artifacts.config.get("stride", 10)),
        "conf_threshold": float(artifacts.config.get("conf_threshold", 0.20)),
        "metrics": artifacts.metrics_summary or {},
    }


@st.cache_data(show_spinner=False)
def run_prediction_cached(
    artifacts_path: str,
    csv_path: str,
    window_threshold: float,
    min_positive_run: int,
    merge_gap_windows: int,
    cooldown_windows: int,
    probability_smoothing: int,
):
    artifacts = load_artifacts_cached(artifacts_path)
    alert_config = AlertConfig(
        window_threshold=window_threshold,
        min_positive_run=min_positive_run,
        merge_gap_windows=merge_gap_windows,
        cooldown_windows=cooldown_windows,
        probability_smoothing=probability_smoothing,
    )
    return predict_sequence(Path(csv_path), artifacts, alert_config)


SVG_MAP = {
    "shield": "<path d='M12 3l7 3v6c0 5-3.5 8-7 9-3.5-1-7-4-7-9V6l7-3z'></path>",
    "activity": "<path d='M3 12h4l2.2-5 3.6 10 2.2-5H21'></path>",
    "lock": "<rect x='5' y='10' width='14' height='10' rx='2'></rect><path d='M8 10V7a4 4 0 118 0v3'></path>",
    "eye_off": "<path d='M3 3l18 18'></path><path d='M10.6 10.6a2 2 0 102.8 2.8'></path><path d='M9.9 5.1A10.7 10.7 0 0112 5c5.2 0 8.7 4.4 9.6 5.7a.6.6 0 010 .6c-.5.8-1.7 2.2-3.4 3.5'></path><path d='M6.4 6.4C4.4 7.7 3 9.5 2.4 10.3a.6.6 0 000 .6C3.4 12.5 6.8 17 12 17c1.2 0 2.3-.2 3.3-.5'></path>",
    "wave": "<path d='M3 12c2.5-4 5.5-4 8 0s5.5 4 8 0'></path><path d='M3 17c2.5-4 5.5-4 8 0s5.5 4 8 0'></path>",
    "timeline": "<circle cx='5' cy='12' r='1.5'></circle><circle cx='12' cy='12' r='1.5'></circle><circle cx='19' cy='12' r='1.5'></circle><path d='M6.5 12H10.5'></path><path d='M13.5 12H17.5'></path>",
    "bell": "<path d='M6 16h12l-1.5-2V10a4.5 4.5 0 10-9 0v4L6 16z'></path><path d='M10 18a2 2 0 004 0'></path>",
    "chart": "<path d='M4 19V5'></path><path d='M10 19V9'></path><path d='M16 19V12'></path><path d='M22 19V7'></path>",
    "download": "<path d='M12 4v10'></path><path d='M8.5 10.5L12 14l3.5-3.5'></path><path d='M4 19h16'></path>",
    "file": "<path d='M8 3h6l4 4v14H8z'></path><path d='M14 3v4h4'></path>",
    "spark": "<path d='M12 3l1.7 4.8L18.5 9l-4.8 1.2L12 15l-1.7-4.8L5.5 9l4.8-1.2L12 3z'></path>",
    "camera_off": "<path d='M2 3l20 18'></path><path d='M9 6l1.2-2h3.6L15 6h2a2 2 0 012 2v7'></path><path d='M5 8H4a2 2 0 00-2 2v7a2 2 0 002 2h12'></path><path d='M9.5 9.5a4 4 0 005 5'></path>",
    "layers": "<path d='M12 3l9 5-9 5-9-5 9-5z'></path><path d='M3 12l9 5 9-5'></path><path d='M3 16l9 5 9-5'></path>",
    "info": "<circle cx='12' cy='12' r='9'></circle><path d='M12 10v5'></path><path d='M12 7.5h.01'></path>",
    "settings": "<circle cx='12' cy='12' r='3'></circle><path d='M19.4 15a1.7 1.7 0 00.3 1.8l.1.1a2 2 0 01-2.8 2.8l-.1-.1a1.7 1.7 0 00-1.8-.3 1.7 1.7 0 00-1 1.5V21a2 2 0 01-4 0v-.2a1.7 1.7 0 00-1-1.5 1.7 1.7 0 00-1.8.3l-.1.1a2 2 0 01-2.8-2.8l.1-.1a1.7 1.7 0 00.3-1.8 1.7 1.7 0 00-1.5-1H3a2 2 0 010-4h.2a1.7 1.7 0 001.5-1 1.7 1.7 0 00-.3-1.8l-.1-.1a2 2 0 012.8-2.8l.1.1a1.7 1.7 0 001.8.3h.1a1.7 1.7 0 001-1.5V3a2 2 0 014 0v.2a1.7 1.7 0 001 1.5h.1a1.7 1.7 0 001.8-.3l.1-.1a2 2 0 012.8 2.8l-.1.1a1.7 1.7 0 00-.3 1.8v.1a1.7 1.7 0 001.5 1H21a2 2 0 010 4h-.2a1.7 1.7 0 00-1.5 1z'></path>",
}


def icon_svg(name: str, size: int = 18, stroke: str = "currentColor") -> str:
    paths = SVG_MAP.get(name, SVG_MAP["spark"])
    return (
        f"<svg viewBox='0 0 24 24' width='{size}' height='{size}' fill='none' "
        f"xmlns='http://www.w3.org/2000/svg' stroke='{stroke}' stroke-width='1.8' "
        f"stroke-linecap='round' stroke-linejoin='round'>{paths}</svg>"
    )


def fmt_pct(value: float, digits: int = 1) -> str:
    return f"{100.0 * float(value):.{digits}f}%"


def fmt_float(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def metric_card_html(title: str, value: str, subtitle: str, icon: str, tone: str = "default") -> str:
    return f"""
    <div class='pg-card pg-metric-card pg-tone-{tone}'>
        <div class='pg-card-top'>
            <div class='pg-icon-wrap'>{icon_svg(icon, size=18)}</div>
            <div class='pg-metric-title'>{title}</div>
        </div>
        <div class='pg-metric-value'>{value}</div>
        <div class='pg-metric-subtitle'>{subtitle}</div>
    </div>
    """


def pill_html(text: str, icon: str) -> str:
    return f"<span class='pg-pill'>{icon_svg(icon, size=14)}<span>{text}</span></span>"


def section_heading(title: str, description: str, icon: str) -> None:
    st.markdown(
        f"""
        <div class='pg-section-heading'>
            <div class='pg-icon-wrap pg-icon-wrap-soft'>{icon_svg(icon, size=18)}</div>
            <div>
                <div class='pg-section-title'>{title}</div>
                <div class='pg-section-description'>{description}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_card(title: str, lines: Iterable[str], icon: str = "info") -> None:
    body = "".join(f"<li>{line}</li>" for line in lines)
    st.markdown(
        f"""
        <div class='pg-card pg-info-card'>
            <div class='pg-card-top'>
                <div class='pg-icon-wrap'>{icon_svg(icon, size=18)}</div>
                <div class='pg-card-heading'>{title}</div>
            </div>
            <ul class='pg-list'>{body}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(bundle_summary: Optional[Dict[str, Any]]) -> None:
    summary = bundle_summary or {}
    metrics = summary.get("metrics", {})
    f1 = metrics.get("f1")
    precision = metrics.get("precision")
    recall = metrics.get("recall")
    model_name = summary.get("model_name", "Random Forest")

    quality_pills = [
        pill_html("Private by design", "shield"),
        pill_html("On-device inference", "lock"),
        pill_html("Skeleton-only analytics", "eye_off"),
    ]
    if f1 is not None:
        quality_pills.append(pill_html(f"Baseline F1 {fmt_float(f1, 3)}", "chart"))

    st.markdown(
        f"""
        <div class='pg-hero'>
            <div class='pg-pill-row'>{''.join(quality_pills)}</div>
            <div class='pg-hero-grid'>
                <div>
                    <div class='pg-eyebrow'>POSEGUARD DASHBOARD</div>
                    <h1 class='pg-hero-title'>Elegant fall detection, distilled to its essentials.</h1>
                    <p class='pg-hero-copy'>A premium skeleton-based monitoring interface for privacy-first care. Calm, exact, and designed to feel more like a private clinic than a control room.</p>
                </div>
                <div class='pg-card pg-hero-card'>
                    <div class='pg-card-top'>
                        <div class='pg-icon-wrap'>{icon_svg('spark', size=18)}</div>
                        <div class='pg-card-heading'>Baseline provenance</div>
                    </div>
                    <div class='pg-hero-stat-grid'>
                        <div>
                            <div class='pg-hero-stat-label'>Model</div>
                            <div class='pg-hero-stat-value'>{model_name.upper()}</div>
                        </div>
                        <div>
                            <div class='pg-hero-stat-label'>Feature set</div>
                            <div class='pg-hero-stat-value'>{summary.get('feature_count', '—')}</div>
                        </div>
                        <div>
                            <div class='pg-hero-stat-label'>Precision</div>
                            <div class='pg-hero-stat-value'>{fmt_pct(precision) if precision is not None else '—'}</div>
                        </div>
                        <div>
                            <div class='pg-hero-stat-label'>Recall</div>
                            <div class='pg-hero-stat-value'>{fmt_pct(recall) if recall is not None else '—'}</div>
                        </div>
                    </div>
                    <div class='pg-hero-foot'>Window {summary.get('window', '—')} · Stride {summary.get('stride', '—')} · Confidence ≥ {summary.get('conf_threshold', '—')}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_banner(summary: Dict[str, Any], source_label: str) -> None:
    tone = "alert" if summary.get("alert") else "ok"
    icon = "bell" if summary.get("alert") else "shield"
    st.markdown(
        f"""
        <div class='pg-status-banner pg-tone-{tone}'>
            <div class='pg-status-left'>
                <div class='pg-icon-wrap'>{icon_svg(icon, size=18)}</div>
                <div>
                    <div class='pg-status-title'>{summary['decision']}</div>
                    <div class='pg-status-copy'>Source: {source_label}</div>
                </div>
            </div>
            <div class='pg-status-right'>
                <div class='pg-status-chip'>{summary.get('n_events', 0)} event(s)</div>
                <div class='pg-status-chip'>{summary.get('n_windows', 0)} windows</div>
                <div class='pg-status-chip'>{summary.get('n_frames', 0)} frames</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_grid(summary: Dict[str, Any]) -> None:
    tone = "alert" if summary.get("alert") else "ok"
    cards = [
        metric_card_html(
            "Decision",
            summary.get("decision", "—"),
            "Event-level output after temporal smoothing",
            "bell" if summary.get("alert") else "shield",
            tone=tone,
        ),
        metric_card_html(
            "Event count",
            str(summary.get("n_events", 0)),
            "Merged alert segments",
            "timeline",
        ),
        metric_card_html(
            "Max probability",
            fmt_pct(summary.get("max_window_probability", 0.0)),
            "Peak raw fall probability",
            "activity",
        ),
        metric_card_html(
            "Positive window ratio",
            fmt_pct(summary.get("positive_window_ratio", 0.0)),
            "Share of smoothed windows above threshold",
            "wave",
        ),
    ]
    st.markdown(f"<div class='pg-metrics-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def render_sidebar() -> Dict[str, Any]:
    with st.sidebar:
        st.markdown(
            f"""
            <div class='pg-sidebar-brand'>
                <div class='pg-icon-wrap pg-icon-wrap-soft'>{icon_svg('shield', size=18)}</div>
                <div>
                    <div class='pg-sidebar-brand-title'>PoseGuard</div>
                    <div class='pg-sidebar-brand-copy'>Minimalist clinical intelligence for privacy-first fall monitoring.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("pg-control-form"):
            artifacts_path = st.text_input(
                "Baseline bundle or output directory",
                placeholder="/Users/you/.../baseline_bundle.joblib",
                help="Point this to baseline_bundle.joblib or to your training output folder containing metrics.json.",
            )

            source_mode = st.radio(
                "Sequence source",
                options=["Browse dataset", "Absolute CSV path", "Upload single CSV"],
                horizontal=False,
            )

            data_root = ""
            resolved_csv_path = ""
            upload = None

            if source_mode == "Browse dataset":
                data_root = st.text_input(
                    "Dataset root",
                    placeholder="/Users/you/.../Fall_Dataset",
                    help="The app will discover all *_keypoints.csv files below this directory.",
                )
                catalog = discover_csv_catalog(data_root) if data_root else pd.DataFrame()
                if data_root and catalog.empty:
                    st.caption("No keypoint CSV files found yet. Check the folder path.")
                if not catalog.empty:
                    label_options = sorted(catalog["label"].unique().tolist())
                    label = st.selectbox("Label", label_options)
                    label_df = catalog[catalog["label"] == label]

                    scene_options = sorted(label_df["scene"].unique().tolist())
                    scene = st.selectbox("Scene", scene_options)
                    scene_df = label_df[label_df["scene"] == scene]

                    pack_options = sorted(scene_df["pack"].unique().tolist())
                    pack = st.selectbox("Package", pack_options)
                    pack_df = scene_df[scene_df["pack"] == pack]

                    filenames = pack_df["filename"].tolist()
                    filename = st.selectbox("Sequence", filenames)
                    resolved_csv_path = pack_df.loc[pack_df["filename"] == filename, "path"].iloc[0]
                    st.caption(str(Path(resolved_csv_path).relative_to(Path(data_root).expanduser())))
            elif source_mode == "Absolute CSV path":
                resolved_csv_path = st.text_input(
                    "CSV path",
                    placeholder="/Users/you/.../B_D_0001_keypoints.csv",
                )
            else:
                upload = st.file_uploader(
                    "Upload a keypoints CSV",
                    type=["csv"],
                    accept_multiple_files=False,
                    help="The uploaded CSV stays local to the app session.",
                )

            with st.expander("Alert logic", expanded=True):
                window_threshold = st.slider("Window threshold", 0.05, 0.95, 0.50, 0.01)
                min_positive_run = st.slider("Minimum positive run", 1, 8, 3, 1)
                merge_gap_windows = st.slider("Merge gap (windows)", 0, 5, 1, 1)
                cooldown_windows = st.slider("Cooldown (windows)", 0, 8, 3, 1)
                probability_smoothing = st.slider("Probability smoothing", 1, 9, 3, 1)

            submitted = st.form_submit_button("Run analysis", use_container_width=True, type="primary")

    return {
        "submitted": submitted,
        "artifacts_path": artifacts_path.strip(),
        "source_mode": source_mode,
        "data_root": data_root.strip(),
        "resolved_csv_path": resolved_csv_path.strip(),
        "upload": upload,
        "window_threshold": window_threshold,
        "min_positive_run": min_positive_run,
        "merge_gap_windows": merge_gap_windows,
        "cooldown_windows": cooldown_windows,
        "probability_smoothing": probability_smoothing,
    }


def write_uploaded_csv(upload) -> Path:
    suffix = Path(upload.name).suffix or ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.getbuffer())
        return Path(tmp.name)


def resolve_prediction_source(controls: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    source_mode = controls["source_mode"]
    if source_mode == "Upload single CSV":
        upload = controls["upload"]
        if upload is None:
            return None, None
        temp_path = write_uploaded_csv(upload)
        return str(temp_path), upload.name

    csv_path = controls["resolved_csv_path"]
    if not csv_path:
        return None, None
    return csv_path, Path(csv_path).name


def serialize_prediction_outputs(result) -> Tuple[bytes, bytes, bytes]:
    prediction_json = json.dumps(result.to_serializable(), indent=2).encode("utf-8")
    windows_csv = result.window_results.to_csv(index=False).encode("utf-8")
    frames_csv = result.wide_sequence.to_csv(index=False).encode("utf-8")
    return prediction_json, windows_csv, frames_csv


def make_probability_figure(window_results: pd.DataFrame, summary: Dict[str, Any], events: List[Dict[str, Any]]) -> go.Figure:
    threshold = float(summary.get("alert_config", {}).get("window_threshold", 0.50))
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=window_results["window_index"],
            y=window_results["fall_probability"],
            mode="lines",
            name="Raw probability",
            line=dict(color=PALETTE["accent_line"], width=2),
            hovertemplate="Window %{x}<br>Probability %{y:.3f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=window_results["window_index"],
            y=window_results["smoothed_probability"],
            mode="lines",
            name="Smoothed probability",
            line=dict(color=PALETTE["text"], width=2.5),
            hovertemplate="Window %{x}<br>Smoothed %{y:.3f}<extra></extra>",
        )
    )

    fig.add_hline(
        y=threshold,
        line_width=1,
        line_dash="dot",
        line_color=PALETTE["muted"],
        annotation_text=f"Threshold {threshold:.2f}",
        annotation_position="top left",
        annotation_font_color=PALETTE["muted"],
    )

    for event in events:
        x0 = event.get("start_window_index", 0)
        x1 = event.get("end_window_index", 0)
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor=PALETTE["alert_soft"],
            opacity=0.55,
            line_width=0,
            layer="below",
        )

    fig.update_layout(
        height=360,
        margin=dict(l=14, r=14, t=16, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PALETTE["surface"],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        font=dict(color=PALETTE["text"], family="Inter, ui-sans-serif, system-ui, sans-serif"),
    )
    fig.update_xaxes(
        title_text="Window index",
        gridcolor=PALETTE["border"],
        zeroline=False,
        showline=False,
    )
    fig.update_yaxes(
        title_text="Fall probability",
        range=[0.0, 1.0],
        gridcolor=PALETTE["border"],
        zeroline=False,
    )
    return fig


def make_frame_metric_figure(frame_features: pd.DataFrame, current_frame_number: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame_features["Frame"],
            y=frame_features["collapse"],
            mode="lines",
            name="Collapse",
            line=dict(color=PALETTE["text"], width=2),
            hovertemplate="Frame %{x}<br>Collapse %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame_features["Frame"],
            y=frame_features["torso_horizontalness"],
            mode="lines",
            name="Torso horizontalness",
            line=dict(color=PALETTE["accent_line"], width=2),
            hovertemplate="Frame %{x}<br>Horizontalness %{y:.3f}<extra></extra>",
        )
    )
    fig.add_vline(x=current_frame_number, line_width=1, line_dash="dot", line_color=PALETTE["muted"])
    fig.update_layout(
        height=300,
        margin=dict(l=14, r=14, t=16, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PALETTE["surface"],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        font=dict(color=PALETTE["text"], family="Inter, ui-sans-serif, system-ui, sans-serif"),
    )
    fig.update_xaxes(title_text="Frame", gridcolor=PALETTE["border"], zeroline=False)
    fig.update_yaxes(title_text="Normalized value", gridcolor=PALETTE["border"], zeroline=False)
    return fig


def frame_number_to_row_index(wide_df: pd.DataFrame, frame_number: int) -> int:
    frames = wide_df["Frame"].to_numpy(dtype=int)
    exact = np.where(frames == int(frame_number))[0]
    if exact.size:
        return int(exact[0])
    return int(np.argmin(np.abs(frames - int(frame_number))))


def make_skeleton_figure(wide_df: pd.DataFrame, frame_number: int) -> go.Figure:
    row_idx = frame_number_to_row_index(wide_df, frame_number)
    row = wide_df.iloc[row_idx]

    x_cols = [f"{KEYPOINT_SLUG[kp]}_x" for kp in KEYPOINTS]
    y_cols = [f"{KEYPOINT_SLUG[kp]}_y" for kp in KEYPOINTS]
    x_values = wide_df[x_cols].to_numpy(dtype=float)
    y_values = wide_df[y_cols].to_numpy(dtype=float)

    finite_x = x_values[np.isfinite(x_values)]
    finite_y = y_values[np.isfinite(y_values)]
    if finite_x.size == 0 or finite_y.size == 0:
        x_min, x_max, y_min, y_max = 0.0, 1.0, 0.0, 1.0
    else:
        x_min, x_max = float(np.min(finite_x)), float(np.max(finite_x))
        y_min, y_max = float(np.min(finite_y)), float(np.max(finite_y))

    x_pad = max(12.0, 0.08 * (x_max - x_min + 1e-6))
    y_pad = max(12.0, 0.08 * (y_max - y_min + 1e-6))

    fig = go.Figure()
    for kp_a, kp_b in SKELETON_EDGES:
        ax = row.get(f"{KEYPOINT_SLUG[kp_a]}_x")
        ay = row.get(f"{KEYPOINT_SLUG[kp_a]}_y")
        bx = row.get(f"{KEYPOINT_SLUG[kp_b]}_x")
        by = row.get(f"{KEYPOINT_SLUG[kp_b]}_y")
        if pd.notna(ax) and pd.notna(ay) and pd.notna(bx) and pd.notna(by):
            fig.add_trace(
                go.Scatter(
                    x=[ax, bx],
                    y=[ay, by],
                    mode="lines",
                    line=dict(color=PALETTE["accent_line"], width=3),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    point_x: List[float] = []
    point_y: List[float] = []
    point_text: List[str] = []
    point_conf: List[float] = []
    for kp in KEYPOINTS:
        slug = KEYPOINT_SLUG[kp]
        x = row.get(f"{slug}_x")
        y = row.get(f"{slug}_y")
        conf = float(row.get(f"{slug}_conf", 0.0) or 0.0)
        if pd.notna(x) and pd.notna(y):
            point_x.append(float(x))
            point_y.append(float(y))
            point_text.append(f"{kp}<br>Confidence {conf:.2f}")
            point_conf.append(conf)

    fig.add_trace(
        go.Scatter(
            x=point_x,
            y=point_y,
            mode="markers",
            marker=dict(
                size=10,
                color=point_conf if point_conf else [0.0],
                colorscale=[
                    [0.0, PALETTE["surface_deep"]],
                    [0.5, PALETTE["accent_soft"]],
                    [1.0, PALETTE["accent"]],
                ],
                cmin=0.0,
                cmax=1.0,
                line=dict(color=PALETTE["surface"], width=1.5),
                showscale=False,
            ),
            text=point_text,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )

    fig.update_layout(
        height=560,
        margin=dict(l=8, r=8, t=12, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PALETTE["surface"],
        font=dict(color=PALETTE["text"], family="Inter, ui-sans-serif, system-ui, sans-serif"),
    )
    fig.update_xaxes(
        visible=False,
        range=[x_min - x_pad, x_max + x_pad],
        showgrid=False,
        zeroline=False,
        fixedrange=True,
    )
    fig.update_yaxes(
        visible=False,
        range=[y_max + y_pad, y_min - y_pad],
        showgrid=False,
        zeroline=False,
        fixedrange=True,
        scaleanchor="x",
        scaleratio=1,
    )
    return fig


def event_option_label(event: Dict[str, Any]) -> str:
    return (
        f"Event {event.get('event_id', '—')} · {event.get('severity', 'low').title()} · "
        f"frames {event.get('frame_start', '—')}–{event.get('frame_end', '—')}"
    )


def render_event_cards(events: List[Dict[str, Any]]) -> None:
    if not events:
        st.markdown(
            f"""
            <div class='pg-card pg-empty-card'>
                <div class='pg-card-top'>
                    <div class='pg-icon-wrap'>{icon_svg('shield', size=18)}</div>
                    <div class='pg-card-heading'>No fall events detected</div>
                </div>
                <div class='pg-empty-copy'>The current alert policy did not surface a stable event. The sequence still remains fully inspectable through the timeline and playback views.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    html_cards = []
    for event in events:
        severity = str(event.get("severity", "low"))
        html_cards.append(
            f"""
            <div class='pg-card pg-event-card pg-severity-{severity}'>
                <div class='pg-card-top'>
                    <div class='pg-icon-wrap'>{icon_svg('bell', size=18)}</div>
                    <div class='pg-card-heading'>Event {event.get('event_id', '—')}</div>
                    <div class='pg-severity-chip'>{severity.title()}</div>
                </div>
                <div class='pg-event-grid'>
                    <div><span>Frame range</span><strong>{event.get('frame_start', '—')}–{event.get('frame_end', '—')}</strong></div>
                    <div><span>Window range</span><strong>{event.get('start_window_index', '—')}–{event.get('end_window_index', '—')}</strong></div>
                    <div><span>Max probability</span><strong>{fmt_pct(event.get('max_probability', 0.0))}</strong></div>
                    <div><span>Mean probability</span><strong>{fmt_pct(event.get('mean_probability', 0.0))}</strong></div>
                </div>
            </div>
            """
        )
    st.markdown("".join(html_cards), unsafe_allow_html=True)


def render_empty_state() -> None:
    st.markdown(
        f"""
        <div class='pg-card pg-empty-state'>
            <div class='pg-icon-wrap pg-icon-wrap-soft'>{icon_svg('camera_off', size=22)}</div>
            <div class='pg-empty-title'>Ready when you are.</div>
            <div class='pg-empty-copy'>Choose a baseline bundle, point the app to a keypoint CSV, and run the sequence. The dashboard will render alert logic, skeleton playback, and downloadable outputs in one place.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_results(result, source_label: str, bundle_summary: Dict[str, Any]) -> None:
    summary = result.summary
    render_status_banner(summary, source_label)
    render_metric_grid(summary)

    tabs = st.tabs(["Overview", "Playback", "Events", "Exports"])

    with tabs[0]:
        section_heading(
            "Alert confidence over time",
            "Window-level probabilities, smoothed and consolidated into event-level alerts.",
            "timeline",
        )
        st.plotly_chart(
            make_probability_figure(result.window_results, result.summary, result.events),
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True},
        )

        meta_col, notes_col = st.columns([1.2, 1.0], gap="large")
        with meta_col:
            info_card(
                "Inference summary",
                [
                    f"Model: {summary.get('model_name', '—')}",
                    f"Feature count: {summary.get('feature_count', '—')}",
                    f"Frames processed: {summary.get('n_frames', '—')}",
                    f"Windows processed: {summary.get('n_windows', '—')}",
                    f"Max probability: {fmt_pct(summary.get('max_window_probability', 0.0))}",
                ],
                icon="layers",
            )
        with notes_col:
            baseline_metrics = bundle_summary.get("metrics", {})
            info_card(
                "Baseline quality",
                [
                    f"Baseline F1: {fmt_float(baseline_metrics.get('f1', 0.0), 3) if baseline_metrics else '—'}",
                    f"Baseline precision: {fmt_pct(baseline_metrics.get('precision', 0.0)) if baseline_metrics else '—'}",
                    f"Baseline recall: {fmt_pct(baseline_metrics.get('recall', 0.0)) if baseline_metrics else '—'}",
                    f"Window: {summary.get('window', '—')} · Stride: {summary.get('stride', '—')}",
                    f"Confidence threshold: {summary.get('conf_threshold', '—')}",
                ],
                icon="chart",
            )

    with tabs[1]:
        section_heading(
            "Skeleton playback",
            "Inspect the selected sequence without exposing raw video. Every frame is reconstructed from keypoints only.",
            "eye_off",
        )
        events = result.events
        event_lookup = {"Entire sequence": None}
        for event in events:
            event_lookup[event_option_label(event)] = event
        focus_label = st.selectbox("Focus range", list(event_lookup.keys()))
        focus_event = event_lookup[focus_label]

        available_frames = result.wide_sequence["Frame"].astype(int).tolist()
        if focus_event and focus_event.get("frame_start") is not None and focus_event.get("frame_end") is not None:
            min_frame = int(focus_event["frame_start"])
            max_frame = int(focus_event["frame_end"])
            eligible = [f for f in available_frames if min_frame <= f <= max_frame]
            if eligible:
                available_frames = eligible

        peak_window_idx = int(result.window_results["fall_probability"].astype(float).idxmax())
        peak_frame = int(result.window_results.loc[peak_window_idx, "frame_start"])
        default_frame = peak_frame if peak_frame in available_frames else available_frames[0]
        frame_number = st.select_slider("Frame", options=available_frames, value=default_frame)

        skeleton_col, detail_col = st.columns([1.3, 1.0], gap="large")
        with skeleton_col:
            st.plotly_chart(
                make_skeleton_figure(result.wide_sequence, frame_number),
                use_container_width=True,
                config={"displayModeBar": False, "responsive": True},
            )
        with detail_col:
            st.plotly_chart(
                make_frame_metric_figure(result.frame_features, frame_number),
                use_container_width=True,
                config={"displayModeBar": False, "responsive": True},
            )
            row_idx = frame_number_to_row_index(result.wide_sequence, frame_number)
            frame_row = result.frame_features.iloc[row_idx]
            info_card(
                f"Frame {frame_number}",
                [
                    f"Collapse: {fmt_float(frame_row['collapse'], 3)}",
                    f"Torso horizontalness: {fmt_float(frame_row['torso_horizontalness'], 3)}",
                    f"Mean confidence: {fmt_float(frame_row['mean_conf'], 3)}",
                    f"Confident ratio: {fmt_pct(frame_row['confident_ratio'])}",
                    f"Normalized height: {fmt_float(frame_row['height_n'], 3)}",
                ],
                icon="activity",
            )

    with tabs[2]:
        section_heading(
            "Event log",
            "Stable alert segments, merged and smoothed to reduce chatter.",
            "bell",
        )
        render_event_cards(result.events)
        st.markdown("<div class='pg-table-wrap'>", unsafe_allow_html=True)
        display_cols = [
            "window_index",
            "frame_start",
            "frame_end",
            "fall_probability",
            "smoothed_probability",
            "binary_prediction",
            "predicted_label",
        ]
        display_df = result.window_results[display_cols].copy()
        display_df["fall_probability"] = display_df["fall_probability"].map(lambda x: round(float(x), 4))
        display_df["smoothed_probability"] = display_df["smoothed_probability"].map(lambda x: round(float(x), 4))
        st.dataframe(display_df, use_container_width=True, height=360)
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[3]:
        section_heading(
            "Exports and privacy posture",
            "Package the outputs for your report or demo while keeping the pipeline strictly skeleton-first.",
            "download",
        )
        prediction_json, windows_csv, frames_csv = serialize_prediction_outputs(result)
        download_cols = st.columns(3, gap="large")
        with download_cols[0]:
            st.download_button(
                "Download prediction JSON",
                data=prediction_json,
                file_name=f"{result.sequence_id}_prediction.json",
                mime="application/json",
                use_container_width=True,
            )
        with download_cols[1]:
            st.download_button(
                "Download windows CSV",
                data=windows_csv,
                file_name=f"{result.sequence_id}_windows.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with download_cols[2]:
            st.download_button(
                "Download frames CSV",
                data=frames_csv,
                file_name=f"{result.sequence_id}_frames.csv",
                mime="text/csv",
                use_container_width=True,
            )

        privacy_col, config_col = st.columns([1.1, 1.0], gap="large")
        with privacy_col:
            info_card(
                "Privacy by design",
                [
                    "No raw video is stored by the dashboard.",
                    "Playback is reconstructed entirely from skeletal coordinates.",
                    "Inference remains local to the running machine.",
                    "Exports contain summary JSON and keypoint-derived CSVs only.",
                ],
                icon="lock",
            )
        with config_col:
            alert_cfg = summary.get("alert_config", {})
            info_card(
                "Alert policy",
                [
                    f"Window threshold: {alert_cfg.get('window_threshold', '—')}",
                    f"Minimum positive run: {alert_cfg.get('min_positive_run', '—')}",
                    f"Merge gap: {alert_cfg.get('merge_gap_windows', '—')} window(s)",
                    f"Cooldown: {alert_cfg.get('cooldown_windows', '—')} window(s)",
                    f"Probability smoothing: {alert_cfg.get('probability_smoothing', '—')}",
                ],
                icon="settings",
            )


def render_error(message: str) -> None:
    st.markdown(
        f"""
        <div class='pg-card pg-error-card'>
            <div class='pg-card-top'>
                <div class='pg-icon-wrap'>{icon_svg('info', size=18)}</div>
                <div class='pg-card-heading'>Unable to run the sequence</div>
            </div>
            <div class='pg-empty-copy'>{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


load_css()
controls = render_sidebar()

bundle_summary: Optional[Dict[str, Any]] = None
if controls["artifacts_path"]:
    try:
        bundle_summary = load_prediction_metadata_summary(controls["artifacts_path"])
    except Exception:
        bundle_summary = None

render_hero(bundle_summary)

if controls["submitted"]:
    if not controls["artifacts_path"]:
        st.session_state["pg_error"] = "Please provide a valid baseline bundle path or training output directory."
        st.session_state.pop("pg_result", None)
    else:
        csv_path, source_label = resolve_prediction_source(controls)
        if not csv_path:
            st.session_state["pg_error"] = "Please choose a sequence source before running analysis."
            st.session_state.pop("pg_result", None)
        else:
            try:
                with st.spinner("Running inference and rendering the sequence…"):
                    result = run_prediction_cached(
                        artifacts_path=controls["artifacts_path"],
                        csv_path=csv_path,
                        window_threshold=controls["window_threshold"],
                        min_positive_run=controls["min_positive_run"],
                        merge_gap_windows=controls["merge_gap_windows"],
                        cooldown_windows=controls["cooldown_windows"],
                        probability_smoothing=controls["probability_smoothing"],
                    )
                st.session_state["pg_result"] = result
                st.session_state["pg_source_label"] = source_label or Path(csv_path).name
                st.session_state["pg_bundle_summary"] = load_prediction_metadata_summary(controls["artifacts_path"])
                st.session_state.pop("pg_error", None)
            except Exception as exc:
                st.session_state["pg_error"] = str(exc)
                st.session_state.pop("pg_result", None)

if st.session_state.get("pg_error"):
    render_error(st.session_state["pg_error"])

if st.session_state.get("pg_result") is not None:
    render_results(
        st.session_state["pg_result"],
        st.session_state.get("pg_source_label", "Sequence"),
        st.session_state.get("pg_bundle_summary", bundle_summary or {}),
    )
else:
    render_empty_state()
