#!/usr/bin/env python3
"""
Shared preprocessing and feature engineering utilities for the PoseGuard fall-detection MVP.

These functions are intentionally aligned with the baseline training script so that
training and inference use the same long-format CSV parsing and temporal feature logic.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

KEYPOINTS: List[str] = [
    "Nose",
    "Left Eye",
    "Right Eye",
    "Left Ear",
    "Right Ear",
    "Left Shoulder",
    "Right Shoulder",
    "Left Elbow",
    "Right Elbow",
    "Left Wrist",
    "Right Wrist",
    "Left Hip",
    "Right Hip",
    "Left Knee",
    "Right Knee",
    "Left Ankle",
    "Right Ankle",
]

KEYPOINT_SLUG = {kp: kp.lower().replace(" ", "_") for kp in KEYPOINTS}
TORSO_KEYPOINTS = {"Left Shoulder", "Right Shoulder", "Left Hip", "Right Hip"}
REQUIRED_COLUMNS = ["Frame", "Keypoint", "X", "Y", "Confidence"]

# Useful later for Streamlit skeleton rendering.
SKELETON_EDGES: List[Tuple[str, str]] = [
    ("Nose", "Left Eye"),
    ("Nose", "Right Eye"),
    ("Left Eye", "Left Ear"),
    ("Right Eye", "Right Ear"),
    ("Left Shoulder", "Right Shoulder"),
    ("Left Shoulder", "Left Elbow"),
    ("Left Elbow", "Left Wrist"),
    ("Right Shoulder", "Right Elbow"),
    ("Right Elbow", "Right Wrist"),
    ("Left Shoulder", "Left Hip"),
    ("Right Shoulder", "Right Hip"),
    ("Left Hip", "Right Hip"),
    ("Left Hip", "Left Knee"),
    ("Left Knee", "Left Ankle"),
    ("Right Hip", "Right Knee"),
    ("Right Knee", "Right Ankle"),
]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def normalize_keypoint_name(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"\s+", " ", name)
    return name.title() if name.lower() != "nose" else "Nose"


def validate_long_format(df: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS).difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def split_frame_into_candidates(frame_df: pd.DataFrame) -> List[pd.DataFrame]:
    """
    Some frames contain multiple pose candidates concatenated in row order.
    This function reconstructs per-frame candidate skeletons by starting a new candidate
    whenever a keypoint repeats.
    """
    frame_df = frame_df.reset_index().rename(columns={"index": "row_order"})
    candidates: List[pd.DataFrame] = []
    current_rows: List[pd.Series] = []
    seen: set[str] = set()

    for _, row in frame_df.iterrows():
        kp = normalize_keypoint_name(row["Keypoint"])
        if kp in seen and current_rows:
            candidates.append(pd.DataFrame(current_rows))
            current_rows = []
            seen = set()
        row = row.copy()
        row["Keypoint"] = kp
        current_rows.append(row)
        seen.add(kp)

    if current_rows:
        candidates.append(pd.DataFrame(current_rows))

    cleaned: List[pd.DataFrame] = []
    for cand in candidates:
        cand = (
            cand.sort_values(["Keypoint", "Confidence"], ascending=[True, False])
            .drop_duplicates(subset=["Keypoint"], keep="first")
            .reset_index(drop=True)
        )
        cleaned.append(cand)
    return cleaned


def candidate_score(candidate_df: pd.DataFrame) -> float:
    if candidate_df.empty:
        return -np.inf
    mean_conf = float(candidate_df["Confidence"].mean())
    present = float(candidate_df["Keypoint"].nunique())
    torso = candidate_df[candidate_df["Keypoint"].isin(TORSO_KEYPOINTS)]
    torso_conf = float(torso["Confidence"].mean()) if not torso.empty else 0.0
    return mean_conf + 0.20 * torso_conf + 0.01 * present


def candidate_to_frame_row(frame_id: int, candidate_df: pd.DataFrame) -> Dict[str, float]:
    row: Dict[str, float] = {"Frame": int(frame_id)}
    candidate_df = candidate_df.copy()
    candidate_df["Keypoint"] = candidate_df["Keypoint"].map(normalize_keypoint_name)

    for kp in KEYPOINTS:
        slug = KEYPOINT_SLUG[kp]
        match = candidate_df[candidate_df["Keypoint"] == kp]
        if not match.empty:
            row[f"{slug}_x"] = float(match["X"].iloc[0])
            row[f"{slug}_y"] = float(match["Y"].iloc[0])
            row[f"{slug}_conf"] = float(match["Confidence"].iloc[0])
        else:
            row[f"{slug}_x"] = np.nan
            row[f"{slug}_y"] = np.nan
            row[f"{slug}_conf"] = 0.0
    return row


def long_pose_csv_to_wide_sequence(
    df: pd.DataFrame,
    collect_diag: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert a long-format pose CSV with columns:
        Frame, Keypoint, X, Y, Confidence
    into a wide frame-level table containing one skeleton per frame.
    """
    df = normalize_columns(df)
    validate_long_format(df)
    df = df[REQUIRED_COLUMNS].copy()
    df["Keypoint"] = df["Keypoint"].map(normalize_keypoint_name)
    df["Frame"] = pd.to_numeric(df["Frame"], errors="coerce")
    df["X"] = pd.to_numeric(df["X"], errors="coerce")
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df["Confidence"] = pd.to_numeric(df["Confidence"], errors="coerce")
    df = df.dropna(subset=["Frame", "Keypoint"])
    df = df.sort_values(["Frame"]).reset_index(drop=True)

    wide_rows: List[Dict[str, float]] = []
    diagnostics: List[Dict[str, object]] = []

    for frame_id, frame_df in df.groupby("Frame", sort=True):
        candidates = split_frame_into_candidates(frame_df)
        if not candidates:
            continue

        scored = [(candidate_score(candidate), candidate) for candidate in candidates]
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_candidate = scored[0]

        wide_rows.append(candidate_to_frame_row(int(frame_id), best_candidate))
        if collect_diag:
            diagnostics.append(
                {
                    "Frame": int(frame_id),
                    "n_candidates": int(len(candidates)),
                    "best_candidate_score": float(best_score),
                    "best_mean_conf": float(best_candidate["Confidence"].mean()),
                    "best_unique_keypoints": int(best_candidate["Keypoint"].nunique()),
                }
            )

    if not wide_rows:
        raise ValueError("No valid frames found after parsing candidates.")

    wide_df = pd.DataFrame(wide_rows).sort_values("Frame").reset_index(drop=True)
    diag_df = pd.DataFrame(diagnostics)

    position_cols = [c for c in wide_df.columns if c.endswith("_x") or c.endswith("_y")]
    conf_cols = [c for c in wide_df.columns if c.endswith("_conf")]

    wide_df[position_cols] = wide_df[position_cols].interpolate(limit_direction="both")
    wide_df[conf_cols] = wide_df[conf_cols].fillna(0.0)

    return wide_df, diag_df


def safe_row_nanmean(arr: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(arr, axis=1)


def safe_row_nanmin(arr: np.ndarray) -> np.ndarray:
    out = np.full(arr.shape[0], np.nan, dtype=float)
    for i, row in enumerate(arr):
        finite = row[np.isfinite(row)]
        if finite.size:
            out[i] = float(np.min(finite))
    return out


def safe_row_nanmax(arr: np.ndarray) -> np.ndarray:
    out = np.full(arr.shape[0], np.nan, dtype=float)
    for i, row in enumerate(arr):
        finite = row[np.isfinite(row)]
        if finite.size:
            out[i] = float(np.max(finite))
    return out


def nanmean_pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    stacked = np.vstack([a, b])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(stacked, axis=0)


def build_frame_feature_table(wide_df: pd.DataFrame, conf_threshold: float = 0.20) -> pd.DataFrame:
    x_cols = [f"{KEYPOINT_SLUG[kp]}_x" for kp in KEYPOINTS]
    y_cols = [f"{KEYPOINT_SLUG[kp]}_y" for kp in KEYPOINTS]
    conf_cols = [f"{KEYPOINT_SLUG[kp]}_conf" for kp in KEYPOINTS]

    x = wide_df[x_cols].to_numpy(dtype=float)
    y = wide_df[y_cols].to_numpy(dtype=float)
    conf = wide_df[conf_cols].to_numpy(dtype=float)

    # Suppress unreliable geometry when confidence is low.
    x = np.where(conf >= conf_threshold, x, np.nan)
    y = np.where(conf >= conf_threshold, y, np.nan)

    centroid_x = safe_row_nanmean(x)
    centroid_y = safe_row_nanmean(y)
    min_x = safe_row_nanmin(x)
    max_x = safe_row_nanmax(x)
    min_y = safe_row_nanmin(y)
    max_y = safe_row_nanmax(y)
    width = max_x - min_x
    height = max_y - min_y

    mean_conf = safe_row_nanmean(conf)
    confident_ratio = np.mean(conf >= conf_threshold, axis=1)

    def col(name: str) -> np.ndarray:
        return wide_df[name].to_numpy(dtype=float)

    shoulder_x = nanmean_pair(col("left_shoulder_x"), col("right_shoulder_x"))
    shoulder_y = nanmean_pair(col("left_shoulder_y"), col("right_shoulder_y"))
    hip_x = nanmean_pair(col("left_hip_x"), col("right_hip_x"))
    hip_y = nanmean_pair(col("left_hip_y"), col("right_hip_y"))
    ankle_y = nanmean_pair(col("left_ankle_y"), col("right_ankle_y"))

    torso_dx = hip_x - shoulder_x
    torso_dy = hip_y - shoulder_y
    torso_angle = np.arctan2(torso_dy, torso_dx + 1e-6)
    torso_horizontalness = np.abs(np.cos(torso_angle))
    torso_len = np.sqrt(torso_dx ** 2 + torso_dy ** 2)

    hip_to_ankle = ankle_y - hip_y
    shoulder_to_hip = hip_y - shoulder_y

    valid_scale = np.isfinite(height) & (height > 1e-6)
    scale = float(np.nanmedian(height[valid_scale])) if np.any(valid_scale) else 1.0
    if not np.isfinite(scale) or scale <= 1e-6:
        scale = 1.0

    centroid_x_n = (centroid_x - np.nanmedian(centroid_x)) / scale
    centroid_y_n = (centroid_y - np.nanmedian(centroid_y)) / scale
    hip_y_n = (hip_y - np.nanmedian(hip_y)) / scale
    shoulder_y_n = (shoulder_y - np.nanmedian(shoulder_y)) / scale
    width_n = width / scale
    height_n = height / scale
    torso_len_n = torso_len / scale
    hip_to_ankle_n = hip_to_ankle / scale
    shoulder_to_hip_n = shoulder_to_hip / scale

    max_height = np.nanmax(height_n) if np.any(np.isfinite(height_n)) else 1.0
    collapse = 1.0 - (height_n / (max_height + 1e-6))
    aspect = width_n / (height_n + 1e-6)

    out = pd.DataFrame(
        {
            "Frame": wide_df["Frame"].to_numpy(dtype=int),
            "centroid_x_n": centroid_x_n,
            "centroid_y_n": centroid_y_n,
            "hip_y_n": hip_y_n,
            "shoulder_y_n": shoulder_y_n,
            "width_n": width_n,
            "height_n": height_n,
            "aspect": aspect,
            "mean_conf": mean_conf,
            "confident_ratio": confident_ratio,
            "torso_horizontalness": torso_horizontalness,
            "torso_len_n": torso_len_n,
            "hip_to_ankle_n": hip_to_ankle_n,
            "shoulder_to_hip_n": shoulder_to_hip_n,
            "collapse": collapse,
        }
    )
    return out


def summarize(arr: np.ndarray, prefix: str) -> Dict[str, float]:
    arr = np.asarray(arr, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        finite = np.array([0.0], dtype=float)
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_std": float(np.std(finite)),
        f"{prefix}_min": float(np.min(finite)),
        f"{prefix}_max": float(np.max(finite)),
        f"{prefix}_range": float(np.max(finite) - np.min(finite)),
        f"{prefix}_first": float(finite[0]),
        f"{prefix}_last": float(finite[-1]),
        f"{prefix}_delta": float(finite[-1] - finite[0]),
    }


def diff_or_zero(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.size < 2:
        return np.array([0.0], dtype=float)
    diff = np.diff(arr)
    diff = np.where(np.isfinite(diff), diff, np.nan)
    return diff


def extract_window_features(frame_feats: pd.DataFrame) -> Dict[str, float]:
    feat_cols = [
        "centroid_y_n",
        "hip_y_n",
        "shoulder_y_n",
        "width_n",
        "height_n",
        "aspect",
        "mean_conf",
        "confident_ratio",
        "torso_horizontalness",
        "torso_len_n",
        "hip_to_ankle_n",
        "shoulder_to_hip_n",
        "collapse",
    ]

    features: Dict[str, float] = {}
    for col in feat_cols:
        features.update(summarize(frame_feats[col].to_numpy(dtype=float), col))

    dynamic_cols = ["centroid_y_n", "hip_y_n", "height_n", "torso_horizontalness", "collapse"]
    for col in dynamic_cols:
        features.update(summarize(diff_or_zero(frame_feats[col].to_numpy(dtype=float)), f"{col}_diff"))

    cx = frame_feats["centroid_x_n"].to_numpy(dtype=float)
    cy = frame_feats["centroid_y_n"].to_numpy(dtype=float)
    if len(cx) > 1:
        centroid_speed = np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2)
    else:
        centroid_speed = np.array([0.0], dtype=float)
    features.update(summarize(centroid_speed, "centroid_speed"))

    features["n_frames"] = int(len(frame_feats))
    return features


def iter_sequence_windows(
    frame_feats: pd.DataFrame,
    window: int,
    stride: int,
) -> Iterable[Tuple[int, int, pd.DataFrame]]:
    n = len(frame_feats)
    if n == 0:
        return
    if n <= window:
        yield 0, n, frame_feats
        return
    for start in range(0, n - window + 1, stride):
        end = start + window
        yield start, end, frame_feats.iloc[start:end]


def build_sequence_windows_from_csv(
    csv_path: Path,
    window: int,
    stride: int,
    conf_threshold: float = 0.20,
    collect_diag: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        wide_df           frame-level keypoint table (one selected skeleton per frame)
        frame_feats       engineered frame-level features
        window_features   engineered window-level features
        window_meta       metadata per sliding window
    """
    raw_df = pd.read_csv(csv_path, usecols=REQUIRED_COLUMNS, low_memory=False)
    wide_df, diag_df = long_pose_csv_to_wide_sequence(raw_df, collect_diag=collect_diag)
    frame_feats = build_frame_feature_table(wide_df, conf_threshold=conf_threshold)

    feature_rows: List[Dict[str, float]] = []
    meta_rows: List[Dict[str, int]] = []

    window_index = 0
    for start, end, window_df in iter_sequence_windows(frame_feats, window=window, stride=stride):
        feature_rows.append(extract_window_features(window_df))
        meta_rows.append(
            {
                "window_index": int(window_index),
                "frame_start": int(window_df["Frame"].iloc[0]),
                "frame_end": int(window_df["Frame"].iloc[-1]),
                "n_window_frames": int(len(window_df)),
                "start_row_index": int(start),
                "end_row_index": int(end),
            }
        )
        window_index += 1

    if not feature_rows:
        raise ValueError("No usable windows were produced for this sequence.")

    window_features = pd.DataFrame(feature_rows)
    window_meta = pd.DataFrame(meta_rows)
    return wide_df, frame_feats, window_features, window_meta
