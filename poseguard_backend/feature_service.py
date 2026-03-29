from __future__ import annotations

import warnings
from collections import deque
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Deque, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


# Keep the same 17-keypoint order used by the existing CSV-based pipeline.
KEYPOINTS: List[str] = [
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

TORSO_KEYPOINTS = {"left_shoulder", "right_shoulder", "left_hip", "right_hip"}


@dataclass
class WindowPacket:
    """A feature-engineered sliding window derived from a PoseFrame stream."""

    ts_ms: int
    camera_id: str
    person_id: int
    window_index: int
    start_ts_ms: int
    end_ts_ms: int
    start_frame_id: int
    end_frame_id: int
    n_frames: int
    source_fps: float
    detected_ratio: float
    mean_pose_conf: float
    feature_vector: Dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FeatureBridgeStats:
    pose_frames_seen: int = 0
    windows_built: int = 0
    rows_with_missing_points: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _coerce_pose_frame(pose_frame: Any) -> Dict[str, Any]:
    if is_dataclass(pose_frame):
        return asdict(pose_frame)
    if hasattr(pose_frame, "to_dict"):
        return pose_frame.to_dict()
    if isinstance(pose_frame, dict):
        return pose_frame
    raise TypeError(f"Unsupported pose_frame type: {type(pose_frame)!r}")


def _keypoint_mapping(pose_frame: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    mapping: Dict[str, Dict[str, float]] = {}
    for kp in pose_frame.get("keypoints", []):
        if hasattr(kp, "to_dict"):
            kp = kp.to_dict()
        elif is_dataclass(kp):
            kp = asdict(kp)
        name = str(kp.get("name", "")).strip().lower()
        if not name:
            continue
        mapping[name] = {
            "x": kp.get("x"),
            "y": kp.get("y"),
            "conf": float(kp.get("conf", 0.0) or 0.0),
        }
    return mapping


def pose_frame_to_wide_row(pose_frame: Any) -> Dict[str, float]:
    """Convert one PoseFrame into the same wide row shape used in CSV preprocessing."""
    pf = _coerce_pose_frame(pose_frame)
    keypoints = _keypoint_mapping(pf)

    row: Dict[str, float] = {"Frame": int(pf["frame_id"])}
    for kp in KEYPOINTS:
        item = keypoints.get(kp)
        row[f"{kp}_x"] = float(item["x"]) if item and item["x"] is not None else np.nan
        row[f"{kp}_y"] = float(item["y"]) if item and item["y"] is not None else np.nan
        row[f"{kp}_conf"] = float(item["conf"]) if item else 0.0
    return row


def build_wide_df_from_pose_frames(pose_frames: Sequence[Any]) -> pd.DataFrame:
    if not pose_frames:
        raise ValueError("pose_frames is empty.")

    rows = [pose_frame_to_wide_row(pf) for pf in pose_frames]
    wide_df = pd.DataFrame(rows).sort_values("Frame").reset_index(drop=True)

    position_cols = [c for c in wide_df.columns if c.endswith("_x") or c.endswith("_y")]
    conf_cols = [c for c in wide_df.columns if c.endswith("_conf")]

    wide_df[position_cols] = wide_df[position_cols].interpolate(limit_direction="both")
    wide_df[conf_cols] = wide_df[conf_cols].fillna(0.0)
    return wide_df


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
    """This mirrors the existing CSV-based frame feature logic."""
    x_cols = [f"{kp}_x" for kp in KEYPOINTS]
    y_cols = [f"{kp}_y" for kp in KEYPOINTS]
    conf_cols = [f"{kp}_conf" for kp in KEYPOINTS]

    x = wide_df[x_cols].to_numpy(dtype=float)
    y = wide_df[y_cols].to_numpy(dtype=float)
    conf = wide_df[conf_cols].to_numpy(dtype=float)

    # Hide unreliable geometry below the same confidence threshold used offline.
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

    return pd.DataFrame(
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


def align_feature_columns(window_features: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    aligned = window_features.copy()
    for col in feature_columns:
        if col not in aligned.columns:
            aligned[col] = 0.0
    extra_cols = [c for c in aligned.columns if c not in feature_columns]
    if extra_cols:
        aligned = aligned.drop(columns=extra_cols)
    return aligned[feature_columns].copy()


class SlidingWindowFeatureService:
    """Accumulate PoseFrames and emit model-ready feature windows.

    This is the real-time bridge that replaces CSV files while preserving the
    same downstream feature contract used by the offline Random Forest pipeline.
    """

    def __init__(
        self,
        *,
        window: int = 30,
        stride: int = 10,
        conf_threshold: float = 0.20,
    ) -> None:
        if window <= 0:
            raise ValueError("window must be > 0")
        if stride <= 0:
            raise ValueError("stride must be > 0")

        self.window = int(window)
        self.stride = int(stride)
        self.conf_threshold = float(conf_threshold)
        self._buffer: Deque[Dict[str, Any]] = deque(maxlen=self.window)
        self._frames_since_emit = 0
        self._window_index = 0
        self._stats = FeatureBridgeStats()

    def reset(self) -> None:
        self._buffer.clear()
        self._frames_since_emit = 0
        self._window_index = 0
        self._stats = FeatureBridgeStats()

    def get_status(self) -> dict:
        return {
            "window": self.window,
            "stride": self.stride,
            "conf_threshold": self.conf_threshold,
            "buffer_size": len(self._buffer),
            "frames_since_emit": self._frames_since_emit,
            **self._stats.to_dict(),
        }

    def push(self, pose_frame: Any) -> Optional[WindowPacket]:
        pf = _coerce_pose_frame(pose_frame)
        self._stats.pose_frames_seen += 1
        self._buffer.append(pf)
        self._frames_since_emit += 1

        if len(self._buffer) < self.window:
            return None

        if self._window_index > 0 and self._frames_since_emit < self.stride:
            return None

        frames = list(self._buffer)
        packet = self._build_window_packet(frames)
        self._frames_since_emit = 0
        self._window_index += 1
        self._stats.windows_built += 1
        return packet

    def _build_window_packet(self, pose_frames: Sequence[Dict[str, Any]]) -> WindowPacket:
        wide_df = build_wide_df_from_pose_frames(pose_frames)
        if wide_df.empty:
            raise ValueError("No rows were produced from the PoseFrame window.")

        missing_xy_rows = int(
            wide_df[[c for c in wide_df.columns if c.endswith("_x") or c.endswith("_y")]]
            .isna()
            .all(axis=1)
            .sum()
        )
        self._stats.rows_with_missing_points += missing_xy_rows

        frame_feats = build_frame_feature_table(wide_df, conf_threshold=self.conf_threshold)
        feature_vector = extract_window_features(frame_feats)

        first = pose_frames[0]
        last = pose_frames[-1]
        detected_ratio = float(np.mean([1.0 if pf.get("detected") else 0.0 for pf in pose_frames]))
        mean_pose_conf = float(np.mean([float(pf.get("pose_conf", 0.0) or 0.0) for pf in pose_frames]))
        source_fps = float(last.get("source_fps") or first.get("source_fps") or 0.0)

        return WindowPacket(
            ts_ms=int(last["ts_ms"]),
            camera_id=str(last["camera_id"]),
            person_id=int(last.get("person_id", 0) or 0),
            window_index=int(self._window_index),
            start_ts_ms=int(first["ts_ms"]),
            end_ts_ms=int(last["ts_ms"]),
            start_frame_id=int(first["frame_id"]),
            end_frame_id=int(last["frame_id"]),
            n_frames=int(len(pose_frames)),
            source_fps=source_fps,
            detected_ratio=detected_ratio,
            mean_pose_conf=mean_pose_conf,
            feature_vector=feature_vector,
        )
