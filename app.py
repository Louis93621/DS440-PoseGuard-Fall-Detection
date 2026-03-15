
from __future__ import annotations

import html
import json
import math
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components


# =========================
# Page config
# =========================
st.set_page_config(
    page_title="PoseGuard | Privacy-First Fall Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================
# Constants
# =========================
APP_TITLE = "PoseGuard"
DEFAULT_FPS = 24
DEFAULT_THRESHOLD = 0.62
DEFAULT_WINDOW = 24
DEFAULT_STRIDE = 1

COCO_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

COCO_NAME_EDGES = [
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

MEDIAPIPE_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

MEDIAPIPE_NAME_EDGES = [
    ("nose", "left_eye_inner"),
    ("nose", "right_eye_inner"),
    ("left_eye_inner", "left_eye"),
    ("left_eye", "left_eye_outer"),
    ("right_eye_inner", "right_eye"),
    ("right_eye", "right_eye_outer"),
    ("left_eye_outer", "left_ear"),
    ("right_eye_outer", "right_ear"),
    ("mouth_left", "mouth_right"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_wrist", "left_thumb"),
    ("left_wrist", "left_index"),
    ("left_wrist", "left_pinky"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_wrist", "right_thumb"),
    ("right_wrist", "right_index"),
    ("right_wrist", "right_pinky"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_heel"),
    ("left_heel", "left_foot_index"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_heel"),
    ("right_heel", "right_foot_index"),
]

BASE_COCO_POSE = np.array(
    [
        [0.00, -0.43],  # nose
        [-0.03, -0.46],
        [0.03, -0.46],
        [-0.06, -0.44],
        [0.06, -0.44],
        [-0.12, -0.26],  # shoulders
        [0.12, -0.26],
        [-0.22, -0.08],  # elbows
        [0.22, -0.08],
        [-0.24, 0.12],  # wrists
        [0.24, 0.12],
        [-0.09, 0.02],  # hips
        [0.09, 0.02],
        [-0.08, 0.34],  # knees
        [0.08, 0.34],
        [-0.08, 0.66],  # ankles
        [0.08, 0.66],
    ],
    dtype=float,
)

FEATURE_COLUMNS = [
    "uprightness_min",
    "uprightness_last",
    "torso_tilt_mean",
    "torso_tilt_max",
    "height_ratio_min",
    "height_ratio_last",
    "aspect_ratio_max",
    "aspect_ratio_last",
    "hip_speed_max",
    "hip_speed_mean",
    "hip_drop_delta",
    "head_drop_delta",
    "motion_energy",
    "bbox_area_delta",
]


# =========================
# Data structures
# =========================
@dataclass
class PoseSequence:
    frames: np.ndarray
    keypoint_names: List[str]
    timestamps: np.ndarray
    frame_ids: np.ndarray
    edges: List[Tuple[int, int]]
    source_name: str


@dataclass
class InferenceResult:
    window_features: pd.DataFrame
    frame_probabilities: np.ndarray
    events: pd.DataFrame
    threshold: float
    model_name: str
    model_source: Optional[str]
    summary_label: str
    peak_probability: float
    peak_frame: int
    peak_time: float


# =========================
# Utility helpers
# =========================
def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def smooth_signal(values: np.ndarray, window: int = 3) -> np.ndarray:
    if len(values) == 0:
        return values
    series = pd.Series(values)
    return series.rolling(window, center=True, min_periods=1).mean().to_numpy()


def normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def natural_key(text: str) -> List[object]:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", str(text))]


def safe_nanmean(arr: np.ndarray, default: float = np.nan) -> float:
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0 or np.isnan(arr).all():
        return default
    return float(np.nanmean(arr))


def safe_nanmax(arr: np.ndarray, default: float = np.nan) -> float:
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0 or np.isnan(arr).all():
        return default
    return float(np.nanmax(arr))


def safe_nanmin(arr: np.ndarray, default: float = np.nan) -> float:
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0 or np.isnan(arr).all():
        return default
    return float(np.nanmin(arr))


def format_pct(value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{value:.0%}"


def html_escape(text: str) -> str:
    return html.escape(str(text), quote=True)


def find_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lookup = {normalize_token(c): c for c in df.columns}
    for candidate in candidates:
        norm = normalize_token(candidate)
        if norm in lookup:
            return lookup[norm]
    return None


def softplus(value: float) -> float:
    return math.log1p(math.exp(value))


# =========================
# Theming / animations
# =========================
CUSTOM_CSS = """
<style>
:root {
    --bg-0: #070b14;
    --bg-1: #0a1020;
    --bg-2: #0d1630;
    --surface-0: rgba(14, 22, 43, 0.72);
    --surface-1: rgba(16, 27, 50, 0.92);
    --surface-2: rgba(23, 36, 66, 0.95);
    --line: rgba(148, 163, 184, 0.16);
    --line-strong: rgba(148, 163, 184, 0.28);
    --text-0: #f8fafc;
    --text-1: #dbe7ff;
    --text-2: #95a4c5;
    --blue-0: #8ec5ff;
    --blue-1: #60a5fa;
    --blue-2: #7c93ff;
    --cyan: #7dd3fc;
    --violet: #a78bfa;
    --mint: #93c5fd;
    --green: #34d399;
    --amber: #fbbf24;
    --orange: #fb923c;
    --red: #f87171;
    --shadow-0: 0 24px 80px rgba(2, 6, 23, 0.52);
    --radius-xl: 28px;
    --radius-lg: 24px;
    --radius-md: 20px;
    --content-max: 1260px;
    --ease-smooth: cubic-bezier(.22, 1, .36, 1);
}
html, body, [class*="css"]  {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif;
}
body {
    background:
      radial-gradient(1000px 600px at 15% -10%, rgba(96, 165, 250, 0.18), transparent 50%),
      radial-gradient(900px 580px at 100% 0%, rgba(167, 139, 250, 0.14), transparent 52%),
      linear-gradient(180deg, var(--bg-0), var(--bg-1) 36%, var(--bg-2));
    color: var(--text-0);
}
#MainMenu, footer, header[data-testid="stHeader"], div[data-testid="stToolbar"] {
    visibility: hidden;
    height: 0;
}
main div.block-container {
    padding-top: 1.35rem;
    padding-bottom: 5rem;
    max-width: var(--content-max);
}
section[data-testid="stSidebar"] {
    display: none !important;
}
div[data-testid="stDecoration"] {
    display: none;
}
.stApp {
    background:
      radial-gradient(1100px 700px at 12% -5%, rgba(96, 165, 250, 0.12), transparent 52%),
      radial-gradient(950px 620px at 95% 2%, rgba(167, 139, 250, 0.1), transparent 50%),
      linear-gradient(180deg, var(--bg-0), var(--bg-1) 32%, var(--bg-2));
}
.stTabs [data-baseweb="tab-list"] {
    gap: 0.6rem;
    background: rgba(10, 16, 32, 0.65);
    border: 1px solid var(--line);
    padding: 0.45rem;
    border-radius: 999px;
    width: fit-content;
    backdrop-filter: blur(14px);
}
.stTabs [data-baseweb="tab"] {
    height: auto;
    padding: 0.65rem 1rem;
    border-radius: 999px;
    color: var(--text-2);
    background: transparent;
    transition: all .28s var(--ease-smooth);
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(180deg, rgba(30, 41, 59, .82), rgba(17, 24, 39, .92)) !important;
    border: 1px solid rgba(255,255,255,.08);
    color: var(--text-0) !important;
}
div[data-testid="stMarkdownContainer"] p {
    color: var(--text-1);
}
[data-testid="stFileUploader"] {
    background: linear-gradient(180deg, rgba(13,20,37,.72), rgba(12,18,34,.88));
    border: 1px solid var(--line);
    border-radius: 22px;
    padding: 1rem;
}
[data-testid="stFileUploader"] section {
    background: transparent;
}
div[data-testid="stExpander"] {
    background: linear-gradient(180deg, rgba(13,20,37,.72), rgba(12,18,34,.88));
    border: 1px solid var(--line);
    border-radius: 22px;
    overflow: hidden;
}
div[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(11,18,34,.76), rgba(13,20,37,.95));
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: .8rem 1rem;
}
div[data-testid="stMetricValue"] {
    color: var(--text-0);
}
[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
    border: 1px solid var(--line);
}
[data-testid="stPlotlyChart"] {
    background: linear-gradient(180deg, rgba(11,18,34,.68), rgba(14,22,43,.88));
    border: 1px solid var(--line);
    border-radius: 24px;
    padding: .35rem;
}
.stAlert {
    border-radius: 20px;
}
.streamlit-expanderHeader {
    color: var(--text-0);
}
.hero-shell {
    position: relative;
    border-radius: 36px;
    padding: 1.6rem 1.65rem 1.7rem;
    background:
      linear-gradient(180deg, rgba(10, 16, 32, 0.76), rgba(10, 16, 32, 0.92)),
      radial-gradient(820px 420px at 80% 6%, rgba(96,165,250,.1), transparent 46%);
    border: 1px solid rgba(148, 163, 184, 0.15);
    box-shadow: var(--shadow-0);
    overflow: hidden;
}
.hero-shell::before {
    content: "";
    position: absolute;
    inset: -25% -10% auto auto;
    width: 380px;
    height: 380px;
    border-radius: 999px;
    background: radial-gradient(circle, rgba(142,197,255,.12), transparent 62%);
    filter: blur(12px);
    pointer-events: none;
}
.hero-grid {
    display: grid;
    grid-template-columns: 1.16fr 0.84fr;
    gap: 1.15rem;
    align-items: stretch;
}
@media (max-width: 980px) {
    .hero-grid {
        grid-template-columns: 1fr;
    }
}
.eyebrow {
    display: inline-flex;
    align-items: center;
    gap: .55rem;
    padding: .42rem .78rem;
    border-radius: 999px;
    border: 1px solid rgba(148,163,184,.16);
    background: rgba(15, 23, 42, .58);
    color: var(--text-1);
    font-size: .82rem;
    line-height: 1;
    letter-spacing: .08em;
    text-transform: uppercase;
    backdrop-filter: blur(12px);
}
.eyebrow::before {
    content: "";
    width: .5rem;
    height: .5rem;
    border-radius: 999px;
    background: radial-gradient(circle, var(--cyan), rgba(125,211,252,.2) 70%);
    box-shadow: 0 0 0 6px rgba(125,211,252,.08);
}
.hero-title {
    margin: .95rem 0 .35rem;
    color: var(--text-0);
    line-height: .9;
    font-size: clamp(2.8rem, 7vw, 5rem);
    letter-spacing: -.05em;
    font-weight: 820;
}
.hero-subtitle {
    margin: 0 0 .85rem;
    color: var(--text-1);
    font-weight: 650;
    font-size: clamp(1.1rem, 2vw, 1.42rem);
    letter-spacing: -.02em;
}
.hero-lede {
    max-width: 64ch;
    color: var(--text-2);
    font-size: 1.02rem;
    line-height: 1.75;
    margin-bottom: 1rem;
}
.cta-row {
    display: flex;
    flex-wrap: wrap;
    gap: .85rem;
    margin-top: 1.05rem;
}
.beam-btn {
    --btn-bg: linear-gradient(180deg, rgba(20,30,56,.78), rgba(13,20,37,.96));
    position: relative;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: .55rem;
    padding: .86rem 1.16rem;
    color: var(--text-0);
    text-decoration: none !important;
    border-radius: 999px;
    background: var(--btn-bg);
    border: 1px solid rgba(255,255,255,.08);
    box-shadow: inset 0 1px 0 rgba(255,255,255,.06);
    transition: transform .28s var(--ease-smooth), background .28s ease, border-color .28s ease;
    overflow: hidden;
}
.beam-btn:hover {
    transform: translateY(-2px);
    border-color: rgba(255,255,255,.14);
}
.beam-btn.ghost {
    background: rgba(12, 18, 34, .56);
    color: var(--text-1);
}
.beam-btn::before {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: inherit;
    padding: 1px;
    background:
      conic-gradient(
        from 0deg,
        transparent 0deg,
        transparent 300deg,
        rgba(142,197,255,.16) 330deg,
        rgba(255,255,255,.95) 349deg,
        rgba(96,165,250,.55) 360deg
      );
    -webkit-mask:
      linear-gradient(#000 0 0) content-box,
      linear-gradient(#000 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
    opacity: .45;
    transform: rotate(0deg);
    transition: opacity .28s ease;
    pointer-events: none;
}
.beam-btn:hover::before {
    opacity: 1;
    animation: beamSpin 1.45s linear infinite;
}
.beam-btn::after {
    content: "";
    position: absolute;
    inset: 0;
    background: radial-gradient(240px circle at 50% -10%, rgba(255,255,255,.12), transparent 50%);
    opacity: .35;
    pointer-events: none;
}
@keyframes beamSpin {
    to { transform: rotate(360deg); }
}
.hero-preview {
    position: relative;
    height: 100%;
}
.preview-shell {
    height: 100%;
    padding: 1.05rem;
}
.preview-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: .6rem;
    margin-bottom: 1rem;
}
.pill {
    display: inline-flex;
    align-items: center;
    gap: .45rem;
    padding: .42rem .72rem;
    border-radius: 999px;
    border: 1px solid rgba(148,163,184,.16);
    color: var(--text-1);
    font-size: .84rem;
    background: rgba(9, 14, 27, .38);
}
.pill .dot {
    width: .45rem;
    height: .45rem;
    border-radius: 999px;
    background: var(--green);
    box-shadow: 0 0 0 6px rgba(52,211,153,.08);
}
.preview-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: .8rem;
}
.preview-card {
    border-radius: 22px;
    padding: .9rem;
    background: rgba(10, 16, 32, .62);
    border: 1px solid rgba(148,163,184,.14);
}
.preview-card h4 {
    margin: 0 0 .35rem;
    font-size: .95rem;
    color: var(--text-0);
}
.preview-card p {
    margin: 0;
    font-size: .82rem;
    color: var(--text-2);
    line-height: 1.55;
}
.mini-signal {
    margin-top: .9rem;
    display: grid;
    gap: .48rem;
}
.signal-line {
    height: .72rem;
    border-radius: 999px;
    background: linear-gradient(90deg, rgba(96,165,250,.18), rgba(142,197,255,.55) 38%, rgba(167,139,250,.18));
    background-size: 180% 100%;
    animation: drift 3.2s ease-in-out infinite;
}
.signal-line:nth-child(2) { width: 86%; animation-delay: -.4s; }
.signal-line:nth-child(3) { width: 76%; animation-delay: -.85s; }
@keyframes drift {
    0%, 100% { background-position: 0% 50%; opacity: .58; }
    50% { background-position: 100% 50%; opacity: .92; }
}
.kicker {
    color: var(--text-2);
    font-size: .8rem;
    letter-spacing: .08em;
    text-transform: uppercase;
    margin-bottom: .45rem;
}
.section-copy {
    max-width: 62ch;
    color: var(--text-2);
    line-height: 1.7;
}
.vclip {
    display: inline-flex;
    flex-wrap: wrap;
    gap: .01em;
}
.vclip .char-wrap {
    display: inline-block;
    overflow: hidden;
    vertical-align: top;
}
.vclip .char {
    display: inline-block;
    transform: translateY(-112%);
    clip-path: inset(0 0 100% 0);
    filter: blur(9px);
    opacity: .14;
    will-change: transform, clip-path, filter, opacity;
}
.vclip.is-visible .char {
    animation: charDrop .82s var(--ease-smooth) both;
    animation-delay: calc(var(--i, 0) * 32ms + var(--base-delay, 0ms));
}
@keyframes charDrop {
    0% {
        transform: translateY(-112%);
        clip-path: inset(0 0 100% 0);
        filter: blur(9px);
        opacity: .14;
    }
    100% {
        transform: translateY(0);
        clip-path: inset(0 0 0 0);
        filter: blur(0px);
        opacity: 1;
    }
}
.reveal {
    opacity: .18;
    transform: translateY(16px);
    filter: blur(10px);
}
.reveal.is-visible {
    animation: revealIn .82s var(--ease-smooth) both;
    animation-delay: var(--delay, 0ms);
}
@keyframes revealIn {
    0% {
        opacity: .18;
        transform: translateY(16px);
        filter: blur(10px);
    }
    100% {
        opacity: 1;
        transform: translateY(0);
        filter: blur(0);
    }
}
.blur-rise.is-visible {
    animation-name: blurRise;
}
@keyframes blurRise {
    0% {
        opacity: .18;
        transform: translateY(22px) scale(.985);
        filter: blur(13px);
    }
    100% {
        opacity: 1;
        transform: translateY(0) scale(1);
        filter: blur(0);
    }
}
.fx-card {
    --mx: 50%;
    --my: 50%;
    position: relative;
    border-radius: 28px;
    padding: 1.05rem 1.05rem 1rem;
    background:
      linear-gradient(180deg, rgba(14, 22, 43, 0.76), rgba(11, 18, 34, 0.96));
    border: 1px solid rgba(148, 163, 184, 0.16);
    box-shadow: 0 18px 48px rgba(2, 6, 23, 0.34);
    overflow: hidden;
    isolation: isolate;
}
.fx-card::before {
    content: "";
    position: absolute;
    inset: 0;
    background:
      radial-gradient(260px circle at var(--mx) var(--my), rgba(125,211,252,.11), transparent 40%),
      radial-gradient(340px circle at var(--mx) var(--my), rgba(96,165,250,.08), transparent 54%);
    opacity: 0;
    transition: opacity .22s ease;
    pointer-events: none;
    z-index: 0;
}
.fx-card::after {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: inherit;
    padding: 1px;
    background:
      radial-gradient(180px circle at var(--mx) var(--my), rgba(255,255,255,.88), rgba(142,197,255,.28) 42%, transparent 64%);
    -webkit-mask:
      linear-gradient(#000 0 0) content-box,
      linear-gradient(#000 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
    opacity: 0;
    transition: opacity .22s ease;
    pointer-events: none;
    z-index: 2;
}
.fx-card:hover::before,
.fx-card:hover::after {
    opacity: 1;
}
.fx-card > * {
    position: relative;
    z-index: 1;
}
.info-card h3, .status-card h3, .metric-shell h3 {
    margin: 0;
    color: var(--text-0);
    font-size: 1.04rem;
}
.info-card .icon {
    font-size: 1.18rem;
    line-height: 1;
}
.info-card .topline {
    display: flex;
    align-items: center;
    gap: .65rem;
    margin-bottom: .65rem;
}
.info-card p {
    margin: 0;
    color: var(--text-2);
    line-height: 1.7;
    font-size: .93rem;
}
.micro-tag {
    display: inline-flex;
    width: fit-content;
    margin-top: .9rem;
    padding: .42rem .64rem;
    border-radius: 999px;
    background: rgba(12,18,34,.48);
    border: 1px solid rgba(148,163,184,.14);
    font-size: .78rem;
    color: var(--text-1);
}
.status-card .status-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: .8rem;
    margin-bottom: .75rem;
}
.status-card .status-badge {
    display: inline-flex;
    align-items: center;
    gap: .48rem;
    padding: .5rem .82rem;
    border-radius: 999px;
    border: 1px solid rgba(148,163,184,.12);
    font-weight: 700;
    letter-spacing: .02em;
}
.status-card .status-badge.safe {
    background: rgba(16, 185, 129, .1);
    color: #b7f7d8;
}
.status-card .status-badge.fall {
    background: rgba(248, 113, 113, .12);
    color: #ffd2d2;
}
.status-card .status-badge .dot {
    width: .5rem;
    height: .5rem;
    border-radius: 999px;
}
.status-card .status-badge.safe .dot {
    background: var(--green);
    box-shadow: 0 0 0 6px rgba(52,211,153,.08);
}
.status-card .status-badge.fall .dot {
    background: var(--red);
    box-shadow: 0 0 0 6px rgba(248,113,113,.08);
}
.status-card .frame-pill {
    color: var(--text-1);
    font-size: .82rem;
    padding: .45rem .66rem;
    border-radius: 999px;
    background: rgba(12,18,34,.48);
    border: 1px solid rgba(148,163,184,.12);
}
.status-card .score {
    font-size: clamp(2rem, 5vw, 3.4rem);
    line-height: .95;
    letter-spacing: -.04em;
    margin: .35rem 0 .2rem;
    color: var(--text-0);
    font-weight: 830;
}
.status-card .caption {
    color: var(--text-2);
    margin-bottom: .95rem;
}
.kv-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: .7rem;
}
.kv {
    padding: .78rem .82rem;
    border-radius: 18px;
    background: rgba(12,18,34,.48);
    border: 1px solid rgba(148,163,184,.12);
}
.kv .k {
    display: block;
    color: var(--text-2);
    font-size: .78rem;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-bottom: .3rem;
}
.kv .v {
    display: block;
    color: var(--text-0);
    font-size: 1rem;
    font-weight: 650;
}
.summary-row {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: .85rem;
    margin-top: .45rem;
}
@media (max-width: 900px) {
    .summary-row {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}
.metric-shell {
    padding: .92rem .92rem .85rem;
}
.metric-shell .metric-label {
    font-size: .82rem;
    letter-spacing: .08em;
    color: var(--text-2);
    text-transform: uppercase;
    margin-bottom: .45rem;
}
.metric-shell .metric-value {
    font-size: 1.85rem;
    letter-spacing: -.04em;
    color: var(--text-0);
    font-weight: 800;
    line-height: 1;
}
.metric-shell .metric-foot {
    margin-top: .55rem;
    color: var(--text-2);
    font-size: .86rem;
}
.pipeline {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: .65rem;
}
.pipe-node {
    padding: .72rem .92rem;
    border-radius: 18px;
    background: rgba(12,18,34,.48);
    border: 1px solid rgba(148,163,184,.12);
    color: var(--text-1);
    font-size: .92rem;
}
.pipe-arrow {
    color: var(--text-2);
    font-size: 1.2rem;
    padding: 0 .2rem;
}
.section-spacer {
    height: .8rem;
}
.anchor {
    position: relative;
    top: -110px;
    visibility: hidden;
}
.legend-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: .55rem;
    margin-top: .75rem;
}
.legend-chip {
    display: inline-flex;
    align-items: center;
    gap: .45rem;
    padding: .42rem .66rem;
    border-radius: 999px;
    background: rgba(12,18,34,.48);
    border: 1px solid rgba(148,163,184,.12);
    font-size: .82rem;
    color: var(--text-1);
}
.legend-chip .swatch {
    width: .55rem;
    height: .55rem;
    border-radius: 999px;
}
.note {
    color: var(--text-2);
    font-size: .9rem;
    line-height: 1.7;
}
.small-muted {
    color: var(--text-2);
    font-size: .88rem;
}
hr.soft {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(148,163,184,.16), transparent);
    margin: 1rem 0 1rem;
}
</style>
"""

CUSTOM_JS = """
<script>
(function () {
  const rootWin = window.parent;
  if (!rootWin) return;

  if (!rootWin.__poseguardRevealObserver) {
    rootWin.__poseguardRevealObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-visible");
        rootWin.__poseguardRevealObserver.unobserve(entry.target);
      });
    }, { threshold: 0.18 });
  }

  const bindAll = () => {
    const doc = rootWin.document;
    if (!doc) return;

    doc.querySelectorAll(".reveal:not([data-pg-bound]), .vclip:not([data-pg-bound])").forEach((el) => {
      el.dataset.pgBound = "1";
      rootWin.__poseguardRevealObserver.observe(el);
    });

    doc.querySelectorAll(".fx-card:not([data-fx-bound])").forEach((card) => {
      card.dataset.fxBound = "1";
      const setPointer = (event) => {
        const rect = card.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        card.style.setProperty("--mx", `${x}px`);
        card.style.setProperty("--my", `${y}px`);
      };
      card.addEventListener("mousemove", setPointer);
      card.addEventListener("mouseenter", setPointer);
      card.addEventListener("mouseleave", () => {
        card.style.setProperty("--mx", "50%");
        card.style.setProperty("--my", "50%");
      });
    });
  };

  bindAll();
  setTimeout(bindAll, 120);
  setTimeout(bindAll, 640);

  if (!rootWin.__poseguardMutationWatcher) {
    rootWin.__poseguardMutationWatcher = new MutationObserver(() => bindAll());
    rootWin.__poseguardMutationWatcher.observe(rootWin.document.body, {
      childList: true,
      subtree: true,
    });
  }
})();
</script>
"""


def inject_custom_ui() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    components.html(CUSTOM_JS, height=0, width=0)


# =========================
# HTML components
# =========================
def vertical_clip_text(text: str, tag: str = "h1", classes: str = "") -> str:
    chars = []
    idx = 0
    for ch in text:
        if ch == " ":
            chars.append('<span class="char-wrap"><span class="char" style="--i:{}">&nbsp;</span></span>'.format(idx))
            idx += 1
            continue
        chars.append(
            f'<span class="char-wrap"><span class="char" style="--i:{idx}">{html_escape(ch)}</span></span>'
        )
        idx += 1
    return f'<{tag} class="vclip {classes}">{"".join(chars)}</{tag}>'


def hero_html() -> str:
    return f"""
    <section class="hero-shell">
      <div class="hero-grid">
        <div>
          <div class="eyebrow reveal" style="--delay: 60ms;">Privacy-first fall detection MVP</div>
          {vertical_clip_text(APP_TITLE, tag="h1", classes="hero-title reveal")}
          <div class="hero-subtitle reveal" style="--delay: 160ms;">Skeleton-only inference for a capstone demo that still feels product-grade.</div>
          <p class="hero-lede reveal" style="--delay: 240ms;">
            Upload a pose CSV, reconstruct the stick-figure sequence, and surface a real-time style
            <strong>Safe / Fall Detected</strong> alert — all locally on your Mac, with no raw video shown or stored.
          </p>
          <div class="cta-row reveal" style="--delay: 320ms;">
            <a class="beam-btn" href="#demo-anchor">Open the demo zone</a>
            <a class="beam-btn ghost" href="#privacy-anchor">See privacy controls</a>
          </div>
          <div class="legend-chip-row reveal" style="--delay: 400ms;">
            <span class="legend-chip"><span class="swatch" style="background: var(--green);"></span>Local-only inference</span>
            <span class="legend-chip"><span class="swatch" style="background: var(--cyan);"></span>CSV in, no raw video out</span>
            <span class="legend-chip"><span class="swatch" style="background: var(--violet);"></span>Capstone-ready demo flow</span>
          </div>
        </div>
        <div class="hero-preview">
          <div class="fx-card preview-shell reveal blur-rise" style="--delay: 220ms;">
            <div class="preview-top">
              <span class="pill"><span class="dot"></span>Previewing local session</span>
              <span class="pill">Pose CSV → alert timeline</span>
            </div>
            <div class="preview-grid">
              <div class="preview-card">
                <div class="kicker">Input</div>
                <h4>Pose-only upload</h4>
                <p>Supports wide-format pose CSVs and common x/y keypoint conventions. No camera feed required.</p>
              </div>
              <div class="preview-card">
                <div class="kicker">Output</div>
                <h4>Presentation-grade alerting</h4>
                <p>Stick figure playback, confidence timeline, peak frame detection, and event log in one flow.</p>
              </div>
            </div>
            <div class="mini-signal">
              <div class="signal-line"></div>
              <div class="signal-line"></div>
              <div class="signal-line"></div>
            </div>
          </div>
        </div>
      </div>
    </section>
    """


def feature_card(icon: str, title: str, body: str, tag: str, delay_ms: int = 0) -> str:
    return f"""
    <div class="fx-card info-card reveal blur-rise" style="--delay: {delay_ms}ms;">
      <div class="topline">
        <div class="icon">{icon}</div>
        <h3>{html_escape(title)}</h3>
      </div>
      <p>{body}</p>
      <div class="micro-tag">{html_escape(tag)}</div>
    </div>
    """


def metric_card(label: str, value: str, foot: str = "", delay_ms: int = 0) -> str:
    return f"""
    <div class="fx-card metric-shell reveal" style="--delay: {delay_ms}ms;">
      <div class="metric-label">{html_escape(label)}</div>
      <div class="metric-value">{html_escape(value)}</div>
      <div class="metric-foot">{html_escape(foot)}</div>
    </div>
    """


def status_card_html(
    probability: float,
    threshold: float,
    current_frame: int,
    current_time: float,
    model_name: str,
    source_name: str,
    peak_probability: float,
) -> str:
    is_fall = probability >= threshold
    badge_class = "fall" if is_fall else "safe"
    label = "Fall Detected" if is_fall else "Safe"
    return f"""
    <div class="fx-card status-card reveal is-visible">
      <div class="status-head">
        <div class="status-badge {badge_class}">
          <span class="dot"></span>{label}
        </div>
        <div class="frame-pill">frame {current_frame}</div>
      </div>
      <div class="score">{probability:.0%}</div>
      <div class="caption">Current fall probability @ {current_time:.2f}s</div>
      <div class="kv-grid">
        <div class="kv">
          <span class="k">Decision threshold</span>
          <span class="v">{threshold:.2f}</span>
        </div>
        <div class="kv">
          <span class="k">Model</span>
          <span class="v">{html_escape(model_name)}</span>
        </div>
        <div class="kv">
          <span class="k">Source</span>
          <span class="v">{html_escape(source_name)}</span>
        </div>
        <div class="kv">
          <span class="k">Peak confidence</span>
          <span class="v">{peak_probability:.0%}</span>
        </div>
      </div>
    </div>
    """


# =========================
# Sample data generation
# =========================
def ease_in_out(t: float) -> float:
    return 3 * t**2 - 2 * t**3


def rotate_points(points: np.ndarray, angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=float)
    return points @ rot.T


def transform_pose(
    base_points: np.ndarray,
    tx: float,
    ty: float,
    angle_deg: float,
    scale: float,
    noise_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    points = base_points.copy()
    hip_center = points[[11, 12]].mean(axis=0)
    centered = points - hip_center
    rotated = rotate_points(centered, angle_deg)
    transformed = rotated * scale + np.array([tx, ty], dtype=float)
    noise = rng.normal(scale=noise_scale, size=transformed.shape)
    return transformed + noise


def synthetic_pose_csv(kind: str = "fall", n_frames: int = 120, fps: int = DEFAULT_FPS) -> pd.DataFrame:
    rng = np.random.default_rng(7 if kind == "fall" else 19)
    frames = []
    for frame_idx in range(n_frames):
        t = frame_idx / max(n_frames - 1, 1)

        if kind == "fall":
            if frame_idx < 42:
                angle = 2.5 * np.sin(frame_idx / 10)
                tx = 0.47 + 0.03 * np.sin(frame_idx / 16)
                ty = 0.34 + 0.01 * np.sin(frame_idx / 6)
            elif frame_idx < 70:
                phase = ease_in_out((frame_idx - 42) / 28)
                angle = 78 * phase
                tx = 0.47 + 0.18 * phase
                ty = 0.34 + 0.19 * phase
            else:
                settle = min((frame_idx - 70) / 34, 1.0)
                angle = 82 + 2 * np.sin(frame_idx / 11)
                tx = 0.65 + 0.02 * np.sin(frame_idx / 7)
                ty = 0.53 + 0.015 * np.sin(frame_idx / 9) + 0.015 * settle
            scale = 0.34
            noise = 0.004
        else:
            angle = 2.8 * np.sin(frame_idx / 11)
            tx = 0.49 + 0.06 * np.sin(frame_idx / 18)
            ty = 0.34 + 0.01 * np.sin(frame_idx / 7)
            scale = 0.34
            noise = 0.003

        pose = transform_pose(BASE_COCO_POSE, tx=tx, ty=ty, angle_deg=angle, scale=scale, noise_scale=noise, rng=rng)
        row = {"frame": frame_idx, "timestamp": frame_idx / fps}
        for i, name in enumerate(COCO_NAMES):
            row[f"{name}_x"] = pose[i, 0]
            row[f"{name}_y"] = pose[i, 1]
        frames.append(row)

    df = pd.DataFrame(frames)
    if kind == "fall":
        df["demo_label"] = "fall"
    else:
        df["demo_label"] = "safe"
    return df


# =========================
# Parsing pose CSV
# =========================
def try_parse_long_format(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    frame_col = find_existing_column(df, ["frame", "frame_id", "frameindex", "timestamp", "time", "t"])
    keypoint_col = find_existing_column(df, ["keypoint", "joint", "landmark", "name"])
    x_col = find_existing_column(df, ["x", "x_coord", "xcoordinate"])
    y_col = find_existing_column(df, ["y", "y_coord", "ycoordinate"])

    if frame_col and keypoint_col and x_col and y_col:
        pivot_x = df.pivot_table(index=frame_col, columns=keypoint_col, values=x_col, aggfunc="first")
        pivot_y = df.pivot_table(index=frame_col, columns=keypoint_col, values=y_col, aggfunc="first")
        keypoints = sorted(set(pivot_x.columns).intersection(set(pivot_y.columns)), key=natural_key)
        if not keypoints:
            return None
        wide = pd.DataFrame(index=pivot_x.index)
        wide["frame"] = np.arange(len(wide))
        wide["timestamp"] = wide.index.to_numpy(dtype=float)
        for kp in keypoints:
            wide[f"{kp}_x"] = pivot_x[kp].to_numpy()
            wide[f"{kp}_y"] = pivot_y[kp].to_numpy()
        wide = wide.reset_index(drop=True)
        return wide
    return None


def wide_coordinate_pairs(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    pairs: Dict[str, Dict[str, str]] = {}

    # Strategy 1: columns like nose_x / nose_y or nose.x / nose.y
    for col in df.columns:
        col_str = str(col).strip()
        match = re.match(r"^(.*?)(?:[_\.\-\s]+)(x|y)$", col_str, flags=re.I)
        if match:
            base = match.group(1).strip(" _.-")
            axis = match.group(2).lower()
            if base:
                pairs.setdefault(base, {})[axis] = col

    # Strategy 2: indexed x0 / y0
    x_indexed: Dict[str, str] = {}
    y_indexed: Dict[str, str] = {}
    for col in df.columns:
        col_str = normalize_token(col)
        mx = re.match(r"^x(\d+)$", col_str)
        my = re.match(r"^y(\d+)$", col_str)
        if mx:
            x_indexed[mx.group(1)] = col
        if my:
            y_indexed[my.group(1)] = col
    for idx in sorted(set(x_indexed).intersection(y_indexed), key=lambda x: int(x)):
        pairs.setdefault(f"kp_{idx}", {})["x"] = x_indexed[idx]
        pairs.setdefault(f"kp_{idx}", {})["y"] = y_indexed[idx]

    # Strategy 3: columns like 0_x / 0_y
    numbered: Dict[str, Dict[str, str]] = {}
    for col in df.columns:
        col_str = normalize_token(col)
        match = re.match(r"^(\d+)(x|y)$", col_str)
        if match:
            idx = match.group(1)
            axis = match.group(2)
            numbered.setdefault(f"kp_{idx}", {})[axis] = col
    for name, pair in numbered.items():
        if "x" in pair and "y" in pair:
            pairs[name] = pair

    return {name: pair for name, pair in pairs.items() if "x" in pair and "y" in pair}


def build_fallback_edges(points: np.ndarray) -> List[Tuple[int, int]]:
    n = len(points)
    if n <= 1:
        return []
    valid = np.nan_to_num(points, nan=0.0)
    remaining = set(range(1, n))
    visited = {0}
    edges: List[Tuple[int, int]] = []

    while remaining:
        best_edge = None
        best_dist = float("inf")
        for i in visited:
            for j in remaining:
                dist = float(np.linalg.norm(valid[i] - valid[j]))
                if dist < best_dist:
                    best_dist = dist
                    best_edge = (i, j)
        if best_edge is None:
            break
        i, j = best_edge
        edges.append((i, j))
        visited.add(j)
        remaining.remove(j)
    return edges


def infer_edges(frames: np.ndarray, keypoint_names: List[str]) -> List[Tuple[int, int]]:
    normalized_names = [normalize_token(name) for name in keypoint_names]
    idx_by_name = {normalize_token(name): idx for idx, name in enumerate(keypoint_names)}

    coco_present = any(name in idx_by_name for name in [normalize_token(n) for n in COCO_NAMES])
    if coco_present:
        edges = []
        for a, b in COCO_NAME_EDGES:
            na, nb = normalize_token(a), normalize_token(b)
            if na in idx_by_name and nb in idx_by_name:
                edges.append((idx_by_name[na], idx_by_name[nb]))
        if edges:
            return edges

    mp_present = any(name in idx_by_name for name in [normalize_token(n) for n in MEDIAPIPE_NAMES])
    if mp_present:
        edges = []
        for a, b in MEDIAPIPE_NAME_EDGES:
            na, nb = normalize_token(a), normalize_token(b)
            if na in idx_by_name and nb in idx_by_name:
                edges.append((idx_by_name[na], idx_by_name[nb]))
        if edges:
            return edges

    if len(keypoint_names) == len(COCO_NAMES):
        return [(COCO_NAMES.index(a), COCO_NAMES.index(b)) for a, b in COCO_NAME_EDGES]
    if len(keypoint_names) >= len(MEDIAPIPE_NAMES):
        mp_idx = {name: i for i, name in enumerate(MEDIAPIPE_NAMES)}
        return [(mp_idx[a], mp_idx[b]) for a, b in MEDIAPIPE_NAME_EDGES if mp_idx[a] < len(keypoint_names) and mp_idx[b] < len(keypoint_names)]

    mean_points = np.nanmean(frames[: min(8, len(frames))], axis=0)
    return build_fallback_edges(mean_points)


def parse_pose_dataframe(df: pd.DataFrame, source_name: str = "uploaded_csv") -> PoseSequence:
    long_wide = try_parse_long_format(df)
    if long_wide is not None:
        df = long_wide.copy()

    pairs = wide_coordinate_pairs(df)
    if not pairs:
        raise ValueError(
            "Unable to detect x/y keypoint columns. Expected columns like nose_x / nose_y, "
            "left_shoulder.x / left_shoulder.y, x0 / y0, or a long format with frame, keypoint, x, y."
        )

    ordered_names = sorted(pairs.keys(), key=natural_key)
    frames = []
    for _, row in df.iterrows():
        coords = []
        for name in ordered_names:
            x_val = pd.to_numeric(row[pairs[name]["x"]], errors="coerce")
            y_val = pd.to_numeric(row[pairs[name]["y"]], errors="coerce")
            coords.append([x_val, y_val])
        frames.append(coords)

    frame_array = np.asarray(frames, dtype=float)
    if frame_array.ndim != 3 or frame_array.shape[2] != 2:
        raise ValueError("Pose array parsing failed. Please check the CSV shape and coordinate columns.")

    frame_id_col = find_existing_column(df, ["frame", "frame_id", "frameindex", "index"])
    time_col = find_existing_column(df, ["timestamp", "time", "t", "seconds", "sec", "ms"])

    if time_col:
        timestamps = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
        if np.isnan(timestamps).any():
            timestamps = np.arange(len(df), dtype=float) / DEFAULT_FPS
    else:
        timestamps = np.arange(len(df), dtype=float) / DEFAULT_FPS

    if frame_id_col:
        frame_series = pd.to_numeric(df[frame_id_col], errors="coerce")
        fallback_ids = pd.Series(np.arange(len(df), dtype=int), index=df.index)
        frame_ids = frame_series.where(frame_series.notna(), fallback_ids).to_numpy(dtype=int)
    else:
        frame_ids = np.arange(len(df), dtype=int)

    if len(timestamps) >= 2:
        diffs = np.diff(timestamps)
        if np.nanmedian(diffs) <= 0:
            timestamps = np.arange(len(df), dtype=float) / DEFAULT_FPS
        elif np.nanmedian(diffs) > 10:
            # likely milliseconds
            timestamps = timestamps / 1000.0

    edges = infer_edges(frame_array, ordered_names)
    return PoseSequence(
        frames=frame_array,
        keypoint_names=ordered_names,
        timestamps=timestamps,
        frame_ids=frame_ids,
        edges=edges,
        source_name=source_name,
    )


# =========================
# Feature extraction
# =========================
def detect_joint_groups(keypoint_names: List[str]) -> Dict[str, List[int]]:
    normalized = [normalize_token(name) for name in keypoint_names]

    def get_indices(candidates: Sequence[str]) -> List[int]:
        candidate_set = {normalize_token(c) for c in candidates}
        return [i for i, name in enumerate(normalized) if name in candidate_set]

    return {
        "head": get_indices(
            [
                "nose",
                "head",
                "left_eye",
                "right_eye",
                "left_ear",
                "right_ear",
                "left_eye_inner",
                "right_eye_inner",
                "left_eye_outer",
                "right_eye_outer",
            ]
        ),
        "shoulders": get_indices(["left_shoulder", "right_shoulder", "lshoulder", "rshoulder", "shoulder_left", "shoulder_right"]),
        "hips": get_indices(["left_hip", "right_hip", "lhip", "rhip", "hip_left", "hip_right"]),
        "ankles": get_indices(["left_ankle", "right_ankle", "lankle", "rankle", "left_heel", "right_heel"]),
    }


def proxy_center(frame: np.ndarray, indices: Sequence[int], role: str) -> np.ndarray:
    if indices:
        subset = frame[list(indices)]
        if subset.size and not np.isnan(subset).all():
            return np.nanmean(subset, axis=0)

    valid = frame[~np.isnan(frame).any(axis=1)]
    if len(valid) == 0:
        return np.array([np.nan, np.nan], dtype=float)

    order = np.argsort(valid[:, 1])
    n = len(valid)
    if role == "head":
        subset = valid[order[: max(1, n // 6)]]
    elif role == "shoulders":
        lo = max(1, n // 6)
        hi = max(lo + 1, n // 3)
        subset = valid[order[lo:hi]]
    elif role == "hips":
        lo = max(1, n // 3)
        hi = max(lo + 1, int(n * 0.62))
        subset = valid[order[lo:hi]]
    else:  # ankles
        subset = valid[order[-max(1, n // 6) :]]
    return np.nanmean(subset, axis=0)


def compute_pose_signals(sequence: PoseSequence) -> Dict[str, np.ndarray]:
    frames = sequence.frames
    groups = detect_joint_groups(sequence.keypoint_names)

    head = np.vstack([proxy_center(frame, groups["head"], "head") for frame in frames])
    shoulders = np.vstack([proxy_center(frame, groups["shoulders"], "shoulders") for frame in frames])
    hips = np.vstack([proxy_center(frame, groups["hips"], "hips") for frame in frames])
    ankles = np.vstack([proxy_center(frame, groups["ankles"], "ankles") for frame in frames])

    torso_vec = shoulders - hips
    torso_len = np.linalg.norm(np.nan_to_num(torso_vec, nan=0.0), axis=1)
    uprightness = np.abs(torso_vec[:, 1]) / np.maximum(torso_len, 1e-6)
    torso_tilt_deg = np.degrees(np.arctan2(np.abs(torso_vec[:, 0]), np.abs(torso_vec[:, 1]) + 1e-6))

    body_height = np.abs(ankles[:, 1] - head[:, 1])
    standing_height = np.nanquantile(body_height[~np.isnan(body_height)], 0.82) if np.isfinite(body_height).any() else 1.0
    standing_height = float(max(standing_height, 1e-6))
    height_ratio = body_height / standing_height

    with np.errstate(all="ignore"):
        x_min = np.nanmin(frames[:, :, 0], axis=1)
        x_max = np.nanmax(frames[:, :, 0], axis=1)
        y_min = np.nanmin(frames[:, :, 1], axis=1)
        y_max = np.nanmax(frames[:, :, 1], axis=1)
    bbox_w = x_max - x_min
    bbox_h = y_max - y_min
    bbox_area = (bbox_w * bbox_h) / (standing_height**2 + 1e-6)
    bbox_aspect = bbox_w / np.maximum(bbox_h, 1e-6)

    hip_shift = np.diff(hips, axis=0, prepend=hips[[0]])
    hip_speed = np.linalg.norm(np.nan_to_num(hip_shift, nan=0.0), axis=1) / standing_height
    hip_drop = np.diff(hips[:, 1], prepend=hips[0, 1]) / standing_height
    head_drop = np.diff(head[:, 1], prepend=head[0, 1]) / standing_height
    bbox_area_delta = np.diff(bbox_area, prepend=bbox_area[0])

    return {
        "head": head,
        "shoulders": shoulders,
        "hips": hips,
        "ankles": ankles,
        "uprightness": smooth_signal(uprightness),
        "torso_tilt_deg": smooth_signal(torso_tilt_deg),
        "body_height": smooth_signal(body_height),
        "height_ratio": smooth_signal(height_ratio),
        "bbox_aspect": smooth_signal(bbox_aspect),
        "bbox_area": smooth_signal(bbox_area),
        "hip_speed": smooth_signal(hip_speed),
        "hip_drop": smooth_signal(hip_drop),
        "head_drop": smooth_signal(head_drop),
        "bbox_area_delta": smooth_signal(bbox_area_delta),
        "standing_height": np.full(len(frames), standing_height, dtype=float),
    }


def window_feature_table(sequence: PoseSequence, window_size: int, stride: int) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    signals = compute_pose_signals(sequence)
    n_frames = len(sequence.frames)
    if n_frames < window_size:
        window_size = max(4, n_frames)
    rows = []

    for start in range(0, max(1, n_frames - window_size + 1), stride):
        end = min(start + window_size, n_frames)
        if end - start < 4:
            continue
        seg = slice(start, end)

        uprightness = signals["uprightness"][seg]
        tilt = signals["torso_tilt_deg"][seg]
        height_ratio = signals["height_ratio"][seg]
        aspect = signals["bbox_aspect"][seg]
        hip_speed = signals["hip_speed"][seg]
        hip_drop = signals["hip_drop"][seg]
        head_drop = signals["head_drop"][seg]
        bbox_area_delta = signals["bbox_area_delta"][seg]

        rows.append(
            {
                "window_start": start,
                "window_end": end - 1,
                "window_mid": int((start + end - 1) / 2),
                "uprightness_min": safe_nanmin(uprightness, 0.0),
                "uprightness_last": safe_nanmean(uprightness[-3:], 0.0),
                "torso_tilt_mean": safe_nanmean(tilt, 0.0),
                "torso_tilt_max": safe_nanmax(tilt, 0.0),
                "height_ratio_min": safe_nanmin(height_ratio, 1.0),
                "height_ratio_last": safe_nanmean(height_ratio[-3:], 1.0),
                "aspect_ratio_max": safe_nanmax(aspect, 0.0),
                "aspect_ratio_last": safe_nanmean(aspect[-3:], 0.0),
                "hip_speed_max": safe_nanmax(hip_speed, 0.0),
                "hip_speed_mean": safe_nanmean(hip_speed, 0.0),
                "hip_drop_delta": float(np.nansum(np.clip(hip_drop, 0, None))),
                "head_drop_delta": float(np.nansum(np.clip(head_drop, 0, None))),
                "motion_energy": safe_nanmean(hip_speed, 0.0) + safe_nanmean(np.abs(np.diff(tilt, prepend=tilt[0])), 0.0) / 90.0,
                "bbox_area_delta": float(np.nansum(np.clip(bbox_area_delta, 0, None))),
            }
        )

    feature_df = pd.DataFrame(rows)
    if feature_df.empty:
        fallback = {
            "window_start": 0,
            "window_end": n_frames - 1,
            "window_mid": max(0, n_frames // 2),
        }
        for col in FEATURE_COLUMNS:
            fallback[col] = 0.0
        feature_df = pd.DataFrame([fallback])

    return feature_df, signals


# =========================
# Model loading & inference
# =========================
def load_model_bundle() -> Tuple[Optional[object], Optional[float], Optional[str], str]:
    candidate_paths = [
        Path("models/fall_detector.joblib"),
        Path("models/fall_detector.pkl"),
        Path("models/baseline_model.joblib"),
        Path("models/baseline_model.pkl"),
        Path("artifacts/fall_detector.joblib"),
        Path("artifacts/fall_detector.pkl"),
        Path("artifacts/baseline_model.joblib"),
        Path("artifacts/baseline_model.pkl"),
        Path("baseline_model.pkl"),
    ]

    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            if path.suffix == ".joblib":
                import joblib  # type: ignore

                bundle = joblib.load(path)
            else:
                with path.open("rb") as f:
                    bundle = pickle.load(f)

            threshold = None
            model_name = path.stem
            model_obj = bundle

            if isinstance(bundle, dict):
                threshold = bundle.get("threshold")
                model_name = bundle.get("model_name", model_name)
                model_obj = bundle.get("model") or bundle.get("estimator") or bundle.get("pipeline") or bundle

            return model_obj, threshold, str(path), str(model_name)
        except Exception as exc:  # pragma: no cover - defensive
            return None, None, str(path), f"Load failed: {exc}"

    return None, None, None, "Heuristic detector"


def model_predict_probabilities(model: object, features: pd.DataFrame) -> Optional[np.ndarray]:
    feature_frame = features[FEATURE_COLUMNS].copy()

    if hasattr(model, "feature_names_in_"):
        expected = list(getattr(model, "feature_names_in_"))
        for col in expected:
            if col not in feature_frame.columns:
                feature_frame[col] = 0.0
        feature_frame = feature_frame[expected]

    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(feature_frame)[:, 1]
            return np.asarray(probs, dtype=float)
        if hasattr(model, "decision_function"):
            raw = np.asarray(model.decision_function(feature_frame), dtype=float)
            return np.asarray(sigmoid(raw), dtype=float)
        if hasattr(model, "predict"):
            preds = np.asarray(model.predict(feature_frame), dtype=float)
            return np.clip(preds, 0.0, 1.0)
    except Exception:
        return None
    return None


def heuristic_window_probabilities(features: pd.DataFrame) -> np.ndarray:
    tilt = features["torso_tilt_mean"].to_numpy(dtype=float)
    height = features["height_ratio_min"].to_numpy(dtype=float)
    aspect = features["aspect_ratio_max"].to_numpy(dtype=float)
    hip_speed = features["hip_speed_max"].to_numpy(dtype=float)
    hip_drop = features["hip_drop_delta"].to_numpy(dtype=float)
    head_drop = features["head_drop_delta"].to_numpy(dtype=float)
    upright = features["uprightness_last"].to_numpy(dtype=float)

    s_tilt = np.clip((tilt - 38) / 25, 0, None)
    s_height = np.clip((0.92 - height) / 0.18, 0, None)
    s_aspect = np.clip((aspect - 0.5) / 0.62, 0, None)
    s_speed = np.clip((hip_speed - 0.045) / 0.09, 0, None)
    s_drop = np.clip((hip_drop - 0.06) / 0.12, 0, None)
    s_head = np.clip((head_drop - 0.08) / 0.12, 0, None)
    s_flat = np.clip((0.76 - upright) / 0.28, 0, None)

    score = (
        1.12 * s_tilt
        + 1.02 * s_height
        + 0.72 * s_aspect
        + 0.72 * s_speed
        + 0.58 * s_drop
        + 0.48 * s_head
        + 0.52 * s_flat
        - 1.62
    )
    return np.asarray(sigmoid(score * 1.55), dtype=float)


def spread_window_scores(
    n_frames: int,
    feature_df: pd.DataFrame,
    window_scores: np.ndarray,
) -> np.ndarray:
    accum = np.zeros(n_frames, dtype=float)
    counts = np.zeros(n_frames, dtype=float)
    peak = np.zeros(n_frames, dtype=float)

    for row, score in zip(feature_df.itertuples(index=False), window_scores):
        start = int(getattr(row, "window_start"))
        end = int(getattr(row, "window_end"))
        accum[start : end + 1] += score
        counts[start : end + 1] += 1.0
        peak[start : end + 1] = np.maximum(peak[start : end + 1], score)

    avg = np.divide(accum, np.maximum(counts, 1.0))
    return np.maximum(avg, peak * 0.9)


def detect_events(
    frame_probabilities: np.ndarray,
    timestamps: np.ndarray,
    threshold: float,
    min_duration_frames: int = 6,
) -> pd.DataFrame:
    hits = frame_probabilities >= threshold
    events = []
    start = None

    for idx, hit in enumerate(hits):
        if hit and start is None:
            start = idx
        if not hit and start is not None:
            end = idx - 1
            if end - start + 1 >= min_duration_frames:
                seg = frame_probabilities[start : end + 1]
                peak_rel = int(np.argmax(seg))
                peak_idx = start + peak_rel
                events.append(
                    {
                        "event": f"Fall event {len(events) + 1}",
                        "start_frame": start,
                        "end_frame": end,
                        "peak_frame": peak_idx,
                        "start_time_s": round(float(timestamps[start]), 3),
                        "peak_time_s": round(float(timestamps[peak_idx]), 3),
                        "peak_probability": round(float(frame_probabilities[peak_idx]), 4),
                    }
                )
            start = None

    if start is not None:
        end = len(hits) - 1
        if end - start + 1 >= min_duration_frames:
            seg = frame_probabilities[start : end + 1]
            peak_rel = int(np.argmax(seg))
            peak_idx = start + peak_rel
            events.append(
                {
                    "event": f"Fall event {len(events) + 1}",
                    "start_frame": start,
                    "end_frame": end,
                    "peak_frame": peak_idx,
                    "start_time_s": round(float(timestamps[start]), 3),
                    "peak_time_s": round(float(timestamps[peak_idx]), 3),
                    "peak_probability": round(float(frame_probabilities[peak_idx]), 4),
                }
            )

    return pd.DataFrame(events)


def infer_sequence(
    sequence: PoseSequence,
    threshold: float = DEFAULT_THRESHOLD,
    window_size: int = DEFAULT_WINDOW,
    stride: int = DEFAULT_STRIDE,
) -> InferenceResult:
    features, _signals = window_feature_table(sequence, window_size=window_size, stride=stride)
    model_obj, bundle_threshold, model_path, model_name = load_model_bundle()
    threshold = bundle_threshold if bundle_threshold is not None else threshold

    window_scores = None
    if model_obj is not None:
        window_scores = model_predict_probabilities(model_obj, features)

    if window_scores is None:
        window_scores = heuristic_window_probabilities(features)

    frame_probs = spread_window_scores(len(sequence.frames), features, window_scores)
    frame_probs = np.clip(frame_probs, 0.0, 1.0)
    events = detect_events(frame_probs, sequence.timestamps, threshold=threshold, min_duration_frames=max(4, window_size // 6))

    peak_frame = int(np.argmax(frame_probs))
    peak_prob = float(frame_probs[peak_frame])
    summary_label = "Fall Detected" if peak_prob >= threshold else "Safe"

    return InferenceResult(
        window_features=features.assign(window_probability=window_scores),
        frame_probabilities=frame_probs,
        events=events,
        threshold=threshold,
        model_name=model_name,
        model_source=model_path,
        summary_label=summary_label,
        peak_probability=peak_prob,
        peak_frame=peak_frame,
        peak_time=float(sequence.timestamps[peak_frame]),
    )


# =========================
# Artifact loading
# =========================
def load_metrics_artifact() -> Tuple[Optional[dict], Optional[pd.DataFrame]]:
    metrics_path = Path("artifacts/metrics.json")
    feature_path = Path("artifacts/feature_importance.csv")

    metrics = None
    feature_df = None

    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
        except Exception:
            metrics = None

    if feature_path.exists():
        try:
            feature_df = pd.read_csv(feature_path)
        except Exception:
            feature_df = None

    return metrics, feature_df


# =========================
# Plot builders
# =========================
def figure_bounds(frames: np.ndarray) -> Tuple[float, float, float, float]:
    with np.errstate(all="ignore"):
        x_min = float(np.nanmin(frames[:, :, 0]))
        x_max = float(np.nanmax(frames[:, :, 0]))
        y_min = float(np.nanmin(frames[:, :, 1]))
        y_max = float(np.nanmax(frames[:, :, 1]))
    pad_x = max((x_max - x_min) * 0.18, 0.08)
    pad_y = max((y_max - y_min) * 0.14, 0.08)
    return x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y


def make_skeleton_figure(
    sequence: PoseSequence,
    frame_idx: int,
    current_probability: float,
    threshold: float,
) -> go.Figure:
    coords = sequence.frames[frame_idx]
    x_min, x_max, y_min, y_max = figure_bounds(sequence.frames)
    is_fall = current_probability >= threshold
    line_color = "rgba(248,113,113,.92)" if is_fall else "rgba(142,197,255,.92)"
    marker_color = "#ffe4e6" if is_fall else "#dbeafe"

    line_x: List[float] = []
    line_y: List[float] = []
    for i, j in sequence.edges:
        if i >= len(coords) or j >= len(coords):
            continue
        pi, pj = coords[i], coords[j]
        if np.isnan(pi).any() or np.isnan(pj).any():
            continue
        line_x.extend([float(pi[0]), float(pj[0]), None])
        line_y.extend([float(pi[1]), float(pj[1]), None])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            line=dict(color=line_color, width=4.0),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode="markers",
            marker=dict(
                size=9,
                color=marker_color,
                line=dict(color="rgba(255,255,255,.18)", width=1),
            ),
            hovertemplate="<b>%{text}</b><br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
            text=sequence.keypoint_names,
            showlegend=False,
        )
    )

    fig.add_annotation(
        x=0.01,
        y=0.99,
        xref="paper",
        yref="paper",
        showarrow=False,
        text=f"<b>Frame {frame_idx}</b> &nbsp; • &nbsp; {current_probability:.0%} fall probability",
        font=dict(size=14, color="#e2e8f0"),
        align="left",
        borderpad=8,
        bgcolor="rgba(7,11,20,.55)",
    )

    fig.update_layout(
        height=520,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, range=[x_min, x_max], fixedrange=True),
        yaxis=dict(visible=False, range=[y_max, y_min], fixedrange=True),
    )
    return fig


def make_timeline_figure(
    timestamps: np.ndarray,
    frame_probabilities: np.ndarray,
    threshold: float,
    current_frame: int,
) -> go.Figure:
    ts = timestamps
    if len(ts) != len(frame_probabilities):
        ts = np.arange(len(frame_probabilities), dtype=float) / DEFAULT_FPS

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ts,
            y=frame_probabilities,
            mode="lines",
            line=dict(color="#8ec5ff", width=3.2),
            fill="tozeroy",
            fillcolor="rgba(142,197,255,.16)",
            hovertemplate="t=%{x:.2f}s<br>p(fall)=%{y:.1%}<extra></extra>",
            name="Fall probability",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[ts[current_frame]],
            y=[frame_probabilities[current_frame]],
            mode="markers",
            marker=dict(size=11, color="#fb923c", line=dict(color="rgba(255,255,255,.55)", width=1)),
            hovertemplate="Current frame<br>t=%{x:.2f}s<br>p(fall)=%{y:.1%}<extra></extra>",
            name="Current frame",
        )
    )
    fig.add_hline(y=threshold, line_dash="dot", line_color="rgba(248,113,113,.6)", annotation_text="threshold", annotation_position="top right")
    fig.update_layout(
        height=320,
        margin=dict(l=18, r=18, t=18, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Time (s)", gridcolor="rgba(148,163,184,.12)", zeroline=False),
        yaxis=dict(title="Fall probability", range=[0, 1.02], tickformat=".0%", gridcolor="rgba(148,163,184,.12)", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def make_confusion_matrix_figure(matrix: List[List[float]]) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=["Predicted Safe", "Predicted Fall"],
            y=["Actual Safe", "Actual Fall"],
            colorscale=[[0, "#0f172a"], [0.5, "#60a5fa"], [1, "#8b5cf6"]],
            text=matrix,
            texttemplate="%{text}",
            showscale=False,
        )
    )
    fig.update_layout(
        height=360,
        margin=dict(l=18, r=18, t=18, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def make_feature_importance_figure(feature_df: pd.DataFrame) -> go.Figure:
    working = feature_df.copy()
    cols = {normalize_token(col): col for col in working.columns}
    feature_col = cols.get("feature") or cols.get("name")
    importance_col = cols.get("importance") or cols.get("score") or cols.get("value")
    if not feature_col or not importance_col:
        raise ValueError("feature_importance.csv must include feature/name and importance/score columns.")
    working = working[[feature_col, importance_col]].rename(columns={feature_col: "feature", importance_col: "importance"})
    working["importance"] = pd.to_numeric(working["importance"], errors="coerce")
    working = working.dropna().sort_values("importance", ascending=True).tail(12)
    fig = go.Figure(
        go.Bar(
            x=working["importance"],
            y=working["feature"],
            orientation="h",
            marker=dict(color="#8ec5ff"),
            hovertemplate="%{y}<br>importance=%{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=420,
        margin=dict(l=18, r=18, t=18, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Importance", gridcolor="rgba(148,163,184,.12)"),
        yaxis=dict(title="", gridcolor="rgba(0,0,0,0)"),
    )
    return fig


# =========================
# UI sections
# =========================
def render_intro_section() -> None:
    st.markdown(hero_html(), unsafe_allow_html=True)
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="reveal" style="--delay: 80ms;">
          <div class="kicker">Why this MVP works</div>
          <h2 style="margin: 0 0 .45rem; color: var(--text-0); letter-spacing: -.03em;">A realistic solo-build that still feels polished on presentation day.</h2>
          <p class="section-copy">
            The app is designed around the exact capstone story we aligned on:
            <strong>privacy-preserving fall detection from pose keypoints only</strong>,
            rendered as a skeleton animation with a dynamic alert state and supporting evidence panels.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(3, gap="large")
    cards = [
        feature_card("📤", "Pose CSV upload", "Drop in a precomputed pose CSV or use a bundled sample. The parser accepts common wide-format landmark layouts and pivots long-format pose logs automatically.", "No raw video", 120),
        feature_card("🦴", "Stick-figure playback", "Render the skeleton in motion without exposing identity. This becomes the centerpiece of your demo instead of a notebook cell or static chart.", "Presentation-friendly", 220),
        feature_card("🚨", "Real-time style alerting", "As the sequence advances, the status panel updates between Safe and Fall Detected, supported by a confidence timeline and event table.", "Model-ready", 320),
    ]
    for col, card in zip(cols, cards):
        with col:
            st.markdown(card, unsafe_allow_html=True)


def load_demo_or_upload(uploaded_file, demo_choice: str) -> Tuple[pd.DataFrame, str]:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df, uploaded_file.name
    if demo_choice == "Bundled fall sequence":
        return synthetic_pose_csv("fall"), "sample_fall_pose.csv"
    return synthetic_pose_csv("safe"), "sample_safe_pose.csv"


def render_summary_metrics(inference: InferenceResult, sequence: PoseSequence) -> None:
    event_count = len(inference.events)
    total_seconds = float(sequence.timestamps[-1] - sequence.timestamps[0]) if len(sequence.timestamps) > 1 else len(sequence.frames) / DEFAULT_FPS
    cards = [
        metric_card("Session verdict", inference.summary_label, "Sequence-level label", 60),
        metric_card("Peak confidence", f"{inference.peak_probability:.0%}", f"at {inference.peak_time:.2f}s", 140),
        metric_card("Detected events", str(event_count), "merged threshold crossings", 220),
        metric_card("Clip duration", f"{total_seconds:.1f}s", f"{len(sequence.frames)} frames", 300),
    ]
    st.markdown(f'<div class="summary-row">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_demo_tab() -> None:
    st.markdown('<div id="demo-anchor" class="anchor"></div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="reveal" style="--delay: 40ms;">
          <div class="kicker">Live demo</div>
          <h2 style="margin: 0 0 .4rem; color: var(--text-0); letter-spacing: -.03em;">Upload → reconstruct → alert</h2>
          <p class="section-copy">
            This tab is your presentation flow. Start with a bundled example or upload a real pose CSV, then walk the audience through
            the skeleton playback, threshold crossing, and event evidence.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    controls_col, hint_col = st.columns([1.12, 0.88], gap="large")
    with controls_col:
        uploaded = st.file_uploader("Upload pose CSV", type=["csv"], accept_multiple_files=False, help="Expected pose keypoint coordinates only. No raw video is used anywhere in this app.")
        demo_choice = st.radio(
            "Or use a bundled demo sequence",
            options=["Bundled fall sequence", "Bundled safe sequence"],
            horizontal=True,
        )

    with hint_col:
        st.markdown(
            feature_card(
                "🧠",
                "Inference behavior",
                "If a trained model bundle is found under models/ or artifacts/, the app will use it. Otherwise, it falls back to a deterministic pose heuristic so you can keep building the demo while training runs in parallel.",
                "Plug-and-play model handoff",
                120,
            ),
            unsafe_allow_html=True,
        )

    with st.expander("Advanced options", expanded=False):
        adv_col1, adv_col2, adv_col3 = st.columns(3)
        with adv_col1:
            threshold = st.slider("Decision threshold", 0.30, 0.95, float(DEFAULT_THRESHOLD), 0.01)
        with adv_col2:
            window_size = st.slider("Window size", 8, 64, int(DEFAULT_WINDOW), 2)
        with adv_col3:
            playback_speed = st.select_slider("Playback speed", options=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0], value=1.0)

    df, source_name = load_demo_or_upload(uploaded, demo_choice)

    try:
        sequence = parse_pose_dataframe(df, source_name=source_name)
    except Exception as exc:
        st.error(f"Could not parse this CSV: {exc}")
        st.stop()

    inference = infer_sequence(sequence, threshold=threshold, window_size=window_size, stride=DEFAULT_STRIDE)
    render_summary_metrics(inference, sequence)

    st.markdown('<hr class="soft" />', unsafe_allow_html=True)

    info1, info2, info3 = st.columns(3)
    info1.metric("Frames", f"{len(sequence.frames)}")
    info2.metric("Keypoints", f"{len(sequence.keypoint_names)}")
    info3.metric("Inference backend", inference.model_name)

    current_frame = st.slider(
        "Scrub frame",
        min_value=0,
        max_value=max(0, len(sequence.frames) - 1),
        value=min(max(0, len(sequence.frames) // 3), max(0, len(sequence.frames) - 1)),
        step=1,
    )

    left, right = st.columns([1.32, 0.68], gap="large")
    status_placeholder = right.empty()
    skeleton_placeholder = left.empty()
    timeline_placeholder = left.empty()

    def render_frame(frame_idx: int) -> None:
        frame_prob = float(inference.frame_probabilities[frame_idx])
        skeleton_placeholder.plotly_chart(
            make_skeleton_figure(sequence, frame_idx, frame_prob, inference.threshold),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        timeline_placeholder.plotly_chart(
            make_timeline_figure(sequence.timestamps, inference.frame_probabilities, inference.threshold, frame_idx),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        status_placeholder.markdown(
            status_card_html(
                probability=frame_prob,
                threshold=inference.threshold,
                current_frame=int(sequence.frame_ids[frame_idx]),
                current_time=float(sequence.timestamps[frame_idx]),
                model_name=inference.model_name,
                source_name=source_name,
                peak_probability=inference.peak_probability,
            ),
            unsafe_allow_html=True,
        )

    render_frame(current_frame)

    play_col, note_col = st.columns([0.18, 0.82], gap="large")
    with play_col:
        play = st.button("▶ Play sequence", use_container_width=True)
    with note_col:
        st.markdown(
            """
            <div class="small-muted">
              During a live presentation, press <strong>Play sequence</strong> after scrubbing near the interesting segment.
              The skeleton and alert panel will update frame by frame to simulate streaming inference over a precomputed pose track.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if play:
        delay = 0.08 / float(playback_speed)
        for frame_idx in range(current_frame, len(sequence.frames)):
            render_frame(frame_idx)
            time.sleep(delay)

    lower_left, lower_right = st.columns([0.7, 0.3], gap="large")
    with lower_left:
        st.markdown(
            """
            <div class="reveal" style="--delay: 60ms;">
              <div class="kicker">Detected events</div>
              <p class="note">Threshold crossings are merged into readable incidents so you can explain <em>when</em> the model believes a fall occurred, not just whether it happened somewhere in the clip.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if inference.events.empty:
            st.info("No fall event crossed the current threshold for this clip.")
        else:
            display_events = inference.events.copy()
            display_events["peak_probability"] = display_events["peak_probability"].map(lambda v: f"{v:.0%}")
            st.dataframe(display_events, hide_index=True, use_container_width=True)

    with lower_right:
        st.markdown(
            feature_card(
                "🔒",
                "Privacy note",
                "The visualizer draws only anonymous landmark coordinates. Even in the demo path, the app never renders or persists raw RGB frames.",
                "Privacy by design",
                140,
            ),
            unsafe_allow_html=True,
        )

    with st.expander("Parsed keypoints & raw window features", expanded=False):
        preview_left, preview_right = st.columns(2)
        with preview_left:
            st.write("Detected keypoints")
            st.dataframe(pd.DataFrame({"keypoint": sequence.keypoint_names}), hide_index=True, use_container_width=True)
        with preview_right:
            st.write("Window feature table")
            preview = inference.window_features.copy()
            preview["window_probability"] = preview["window_probability"].map(lambda v: round(float(v), 4))
            st.dataframe(preview, hide_index=True, use_container_width=True)


def render_model_tab() -> None:
    st.markdown(
        """
        <div class="reveal" style="--delay: 40ms;">
          <div class="kicker">Model snapshot</div>
          <h2 style="margin: 0 0 .4rem; color: var(--text-0); letter-spacing: -.03em;">Hook this tab to the artifacts your training pipeline emits.</h2>
          <p class="section-copy">
            When training finishes, drop <code>metrics.json</code> and optionally <code>feature_importance.csv</code> into <code>artifacts/</code>.
            The app will use those files automatically so your dashboard stays synchronized with your baseline results.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics, feature_df = load_metrics_artifact()
    if metrics is None:
        st.info(
            "No metrics artifact found yet. Save `artifacts/metrics.json` to populate this tab. "
            "Supported keys: accuracy, precision, recall, f1, threshold, model_name, window_size, confusion_matrix."
        )
        placeholder_cols = st.columns(4)
        placeholders = [
            ("Accuracy", "—", "awaiting artifact"),
            ("Precision", "—", "awaiting artifact"),
            ("Recall", "—", "awaiting artifact"),
            ("F1", "—", "awaiting artifact"),
        ]
        for col, (label, value, foot) in zip(placeholder_cols, placeholders):
            with col:
                st.markdown(metric_card(label, value, foot), unsafe_allow_html=True)
        return

    top = st.columns(5)
    metric_map = [
        ("Accuracy", metrics.get("accuracy")),
        ("Precision", metrics.get("precision")),
        ("Recall", metrics.get("recall")),
        ("F1", metrics.get("f1")),
        ("Threshold", metrics.get("threshold")),
    ]
    for col, (label, value) in zip(top, metric_map):
        with col:
            display = f"{value:.0%}" if isinstance(value, (int, float)) and label != "Threshold" else (f"{value:.2f}" if isinstance(value, (int, float)) else "—")
            st.markdown(metric_card(label, display, metrics.get("model_name", "baseline")), unsafe_allow_html=True)

    st.markdown('<hr class="soft" />', unsafe_allow_html=True)
    left, right = st.columns([0.72, 0.28], gap="large")

    with left:
        if isinstance(metrics.get("confusion_matrix"), list):
            st.plotly_chart(
                make_confusion_matrix_figure(metrics["confusion_matrix"]),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        else:
            st.info("No confusion_matrix key found in metrics.json.")

    with right:
        st.markdown(
            feature_card(
                "🧪",
                "Suggested metrics schema",
                "Store accuracy, precision, recall, f1, threshold, model_name, window_size, and confusion_matrix in artifacts/metrics.json. This keeps the demo UI decoupled from the training notebook or script.",
                "Simple artifact contract",
                100,
            ),
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="soft" />', unsafe_allow_html=True)
    if feature_df is not None:
        try:
            st.plotly_chart(
                make_feature_importance_figure(feature_df),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        except Exception as exc:
            st.warning(f"Feature importance file found but could not be rendered: {exc}")
    else:
        st.info("No feature importance artifact found yet. Save `artifacts/feature_importance.csv` to render a ranked feature view.")


def render_privacy_tab() -> None:
    st.markdown('<div id="privacy-anchor" class="anchor"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="reveal" style="--delay: 40ms;">
          <div class="kicker">Privacy & system</div>
          <h2 style="margin: 0 0 .4rem; color: var(--text-0); letter-spacing: -.03em;">Turn your constraints into the strongest part of the story.</h2>
          <p class="section-copy">
            This MVP is intentionally framed as a <strong>local, privacy-preserving inference layer</strong>:
            raw video is never shown during the demo, and the UI visualizes only anonymized skeletal coordinates.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="fx-card reveal blur-rise" style="--delay: 120ms;">
          <div class="kicker">System pipeline</div>
          <div class="pipeline">
            <div class="pipe-node">Offline video preprocessing</div>
            <div class="pipe-arrow">→</div>
            <div class="pipe-node">Pose keypoints CSV</div>
            <div class="pipe-arrow">→</div>
            <div class="pipe-node">Local feature extraction</div>
            <div class="pipe-arrow">→</div>
            <div class="pipe-node">Baseline model / heuristic</div>
            <div class="pipe-arrow">→</div>
            <div class="pipe-node">Skeleton-only alert UI</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(3, gap="large")
    blocks = [
        feature_card("🔐", "What the app never does", "It does not render raw video, save RGB frames, call a cloud API, or require a live camera feed to function during the demo.", "Strict privacy boundary", 120),
        feature_card("💻", "What runs locally", "CSV parsing, feature extraction, event scoring, chart generation, and the entire front-end experience all execute on-device inside Streamlit.", "Mac-friendly local stack", 220),
        feature_card("🎓", "Why this is capstone-appropriate", "You still demonstrate the full product loop — data in, model reasoning, interpretable output, and privacy rationale — without overcommitting to production-grade deployment.", "Impressive but finishable", 320),
    ]
    for col, block in zip(cols, blocks):
        with col:
            st.markdown(block, unsafe_allow_html=True)

    st.markdown('<hr class="soft" />', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="reveal" style="--delay: 80ms;">
          <div class="kicker">Recommended repo contract</div>
          <p class="section-copy">
            Keep the UI stable even as the model evolves. A clean handoff looks like:
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.code(
        """.
├── app.py
├── artifacts/
│   ├── metrics.json
│   └── feature_importance.csv
├── models/
│   └── baseline_model.pkl   # or .joblib
└── samples/
    ├── sample_fall_pose.csv
    └── sample_safe_pose.csv
""",
        language="bash",
    )


# =========================
# Main app
# =========================
def main() -> None:
    inject_custom_ui()
    render_intro_section()

    tabs = st.tabs(["Live Demo", "Model Snapshot", "Privacy & System"])
    with tabs[0]:
        render_demo_tab()
    with tabs[1]:
        render_model_tab()
    with tabs[2]:
        render_privacy_tab()


if __name__ == "__main__":
    main()
