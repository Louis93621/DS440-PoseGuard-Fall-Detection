#!/usr/bin/env python3
"""
Train a CPU-friendly baseline fall detector from FallVision-style long-format pose CSV files.

Changes in this version
-----------------------
- Adds tqdm progress bars for long dataset builds.
- Uses a path-based sequence/group ID instead of only file stem to avoid collisions.
- Adds --max_files for quick smoke tests.
- Adds --no_frame_diagnostics to skip the large diagnostics CSV when you want faster runs.
- Reads only the required CSV columns.
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    class _TqdmFallback:
        def __init__(self, iterable=None, total=None, desc=None, unit=None, dynamic_ncols=None):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable) if self.iterable is not None else iter([])

        def update(self, n=1):
            return None

        def set_postfix(self, *args, **kwargs):
            return None

        def write(self, msg: str):
            print(msg)

        def close(self):
            return None

    def tqdm(iterable=None, *args, **kwargs):
        return _TqdmFallback(iterable=iterable, **kwargs)


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
SCENARIOS = {"bed", "chair", "stand"}
REQUIRED_COLUMNS = ["Frame", "Keypoint", "X", "Y", "Confidence"]


@dataclass
class SequenceResult:
    features: List[Dict[str, float]]
    labels: List[int]
    groups: List[str]
    metadata: List[Dict[str, object]]
    frame_diagnostics: List[Dict[str, object]]
    n_files_total: int = 0
    n_files_used: int = 0
    n_files_skipped: int = 0


def find_csv_files(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*.csv") if p.is_file())


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def normalize_keypoint_name(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"\s+", " ", name)
    return name.title() if name.lower() != "nose" else "Nose"


def infer_label(path: Path) -> Optional[int]:
    s = str(path).lower().replace("\\", "/")
    if any(token in s for token in ["/no fall/", "/no_fall/", "/nofall/", "/nonfall/"]):
        return 0
    if "/fall/" in s:
        return 1
    return None


def infer_scenario(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    for part in parts:
        if part in SCENARIOS:
            return part
    return "unknown"


def infer_group_id(path: Path, data_root: Optional[Path] = None) -> str:
    """Use a relative file path so repeated stems across folders do not collide."""
    try:
        if data_root is not None:
            rel = path.resolve().relative_to(data_root.resolve())
        else:
            rel = path
    except Exception:
        rel = path
    return str(rel.with_suffix("")).replace("\\", "/")


def validate_long_format(df: pd.DataFrame) -> None:
    required = set(REQUIRED_COLUMNS)
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def split_frame_into_candidates(frame_df: pd.DataFrame) -> List[pd.DataFrame]:
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


def long_pose_csv_to_wide_sequence(df: pd.DataFrame, collect_diag: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

        scored = [(candidate_score(c), c) for c in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
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


def iter_sequence_windows(frame_feats: pd.DataFrame, window: int, stride: int) -> Iterable[Tuple[int, int, pd.DataFrame]]:
    n = len(frame_feats)
    if n == 0:
        return
    if n <= window:
        yield 0, n, frame_feats
        return
    for start in range(0, n - window + 1, stride):
        end = start + window
        yield start, end, frame_feats.iloc[start:end]


def build_dataset(
    data_root: Path,
    window: int,
    stride: int,
    conf_threshold: float,
    max_files: Optional[int] = None,
    collect_frame_diagnostics: bool = True,
) -> SequenceResult:
    feature_rows: List[Dict[str, float]] = []
    labels: List[int] = []
    groups: List[str] = []
    metadata: List[Dict[str, object]] = []
    diagnostics_rows: List[Dict[str, object]] = []

    csv_files = [p for p in find_csv_files(data_root) if infer_label(p) is not None]
    if max_files is not None:
        csv_files = csv_files[:max_files]

    n_used = 0
    n_skipped = 0

    pbar = tqdm(csv_files, desc="Parsing pose CSVs", unit="file", dynamic_ncols=True)
    for idx, csv_path in enumerate(pbar, start=1):
        label = infer_label(csv_path)
        if label is None:
            continue

        scenario = infer_scenario(csv_path)
        group_id = infer_group_id(csv_path, data_root=data_root)

        try:
            raw_df = pd.read_csv(csv_path, usecols=REQUIRED_COLUMNS, low_memory=False)
            wide_df, diag_df = long_pose_csv_to_wide_sequence(raw_df, collect_diag=collect_frame_diagnostics)
            frame_feats = build_frame_feature_table(wide_df, conf_threshold=conf_threshold)

            if collect_frame_diagnostics and not diag_df.empty:
                for _, diag_row in diag_df.iterrows():
                    diagnostics_rows.append(
                        {
                            "source_file": str(csv_path),
                            "sequence_id": group_id,
                            "scenario": scenario,
                            **{k: (v.item() if hasattr(v, "item") else v) for k, v in diag_row.to_dict().items()},
                        }
                    )

            window_index = 0
            for start, end, window_df in iter_sequence_windows(frame_feats, window=window, stride=stride):
                features = extract_window_features(window_df)
                feature_rows.append(features)
                labels.append(int(label))
                groups.append(group_id)
                metadata.append(
                    {
                        "source_file": str(csv_path),
                        "sequence_id": group_id,
                        "scenario": scenario,
                        "label": int(label),
                        "window_index": int(window_index),
                        "frame_start": int(window_df["Frame"].iloc[0]),
                        "frame_end": int(window_df["Frame"].iloc[-1]),
                        "n_window_frames": int(len(window_df)),
                    }
                )
                window_index += 1
            n_used += 1
        except Exception as exc:
            n_skipped += 1
            tqdm.write(f"[WARN] Skipping {csv_path}: {exc}")

        if idx % 25 == 0 or idx == len(csv_files):
            pbar.set_postfix(used=n_used, skipped=n_skipped, windows=len(feature_rows), refresh=False)

    pbar.close()

    return SequenceResult(
        features=feature_rows,
        labels=labels,
        groups=groups,
        metadata=metadata,
        frame_diagnostics=diagnostics_rows,
        n_files_total=len(csv_files),
        n_files_used=n_used,
        n_files_skipped=n_skipped,
    )


def make_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    if groups.nunique() >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        return train_idx, test_idx

    idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(idx, test_size=test_size, stratify=y, random_state=random_state)
    return train_idx, test_idx


def fit_models(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {
        "logreg": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "rf": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=None,
                        min_samples_leaf=2,
                        n_jobs=-1,
                        class_weight="balanced_subsample",
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    for name, pipe in tqdm(models.items(), total=len(models), desc="Fitting models", unit="model", dynamic_ncols=True):
        pipe.fit(X_train, y_train)
    return models


def evaluate_models(models: Dict[str, Pipeline], X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}
    for name, model in tqdm(models.items(), total=len(models), desc="Evaluating models", unit="model", dynamic_ncols=True):
        pred = model.predict(X_test)
        results[name] = {
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
            "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
        }
    return results


def maybe_export_feature_importance(output_dir: Path, model_name: str, model: Pipeline, feature_names: Sequence[str]) -> None:
    estimator = model.named_steps.get("model")
    if estimator is None or not hasattr(estimator, "feature_importances_"):
        return
    imp = pd.DataFrame(
        {
            "feature": list(feature_names),
            "importance": estimator.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    imp.to_csv(output_dir / f"{model_name}_feature_importance.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True, help="Root folder containing FallVision CSV files")
    parser.add_argument("--output_dir", type=Path, required=True, help="Where to save features, metrics, and model artifacts")
    parser.add_argument("--window", type=int, default=30, help="Window size in frames")
    parser.add_argument("--stride", type=int, default=10, help="Stride size in frames")
    parser.add_argument("--conf_threshold", type=float, default=0.20, help="Minimum keypoint confidence used for geometry")
    parser.add_argument("--max_files", type=int, default=None, help="Optional cap on number of CSV files for quick tests")
    parser.add_argument("--no_frame_diagnostics", action="store_true", help="Skip collecting/saving per-frame diagnostics to reduce runtime and output size")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    seq = build_dataset(
        data_root=args.data_root,
        window=args.window,
        stride=args.stride,
        conf_threshold=args.conf_threshold,
        max_files=args.max_files,
        collect_frame_diagnostics=not args.no_frame_diagnostics,
    )

    X = pd.DataFrame(seq.features)
    y = pd.Series(seq.labels, name="label")
    g = pd.Series(seq.groups, name="sequence_id")

    if X.empty:
        raise SystemExit("No usable training windows found. Check your folder structure and CSV files.")

    train_idx, test_idx = make_split(X, y, g)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    g_train, g_test = g.iloc[train_idx], g.iloc[test_idx]

    models = fit_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)

    best_name = max(results, key=lambda name: results[name]["f1"])

    X.to_csv(args.output_dir / "features.csv", index=False)
    pd.DataFrame(seq.metadata).to_csv(args.output_dir / "window_metadata.csv", index=False)
    if not args.no_frame_diagnostics:
        pd.DataFrame(seq.frame_diagnostics).to_csv(args.output_dir / "frame_diagnostics.csv", index=False)

    for name, model in models.items():
        joblib.dump(model, args.output_dir / f"{name}_model.joblib")
        maybe_export_feature_importance(args.output_dir, name, model, X.columns)

    summary = {
        "n_files_total": int(seq.n_files_total),
        "n_files_used": int(seq.n_files_used),
        "n_files_skipped": int(seq.n_files_skipped),
        "n_windows": int(len(X)),
        "n_train_windows": int(len(X_train)),
        "n_test_windows": int(len(X_test)),
        "n_sequences_total": int(g.nunique()),
        "n_sequences_train": int(g_train.nunique()),
        "n_sequences_test": int(g_test.nunique()),
        "window": int(args.window),
        "stride": int(args.stride),
        "conf_threshold": float(args.conf_threshold),
        "feature_columns": list(X.columns),
        "class_balance": {
            "n_no_fall": int((y == 0).sum()),
            "n_fall": int((y == 1).sum()),
        },
        "results": results,
        "best_model": best_name,
    }

    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Best model: {best_name}")
    print(json.dumps(results[best_name], indent=2))


if __name__ == "__main__":
    main()
