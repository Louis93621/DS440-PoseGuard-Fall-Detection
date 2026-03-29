#!/usr/bin/env python3
"""
Sequence-level inference for the PoseGuard fall-detection MVP.

This script loads a frozen baseline bundle (or a training output directory),
runs one CSV sequence through the exact same preprocessing pipeline used in training,
and then applies temporal alert post-processing to produce stable event-level alerts.

Example:
    python infer.py \
      --artifacts "/Users/you/Desktop/DS 440 Project/Output_v2/baseline_bundle.joblib" \
      --csv_path "/Users/you/Desktop/DS 440 Project/Fall_Dataset/Fall/Bed/f_mask_b_1_keypoints_csv/B_D_0001_keypoints.csv" \
      --save_json "/Users/you/Desktop/DS 440 Project/predictions/B_D_0001_prediction.json" \
      --save_window_csv "/Users/you/Desktop/DS 440 Project/predictions/B_D_0001_windows.csv"
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from poseguard_core import build_sequence_windows_from_csv
from postprocess import AlertConfig, postprocess_window_probabilities


@dataclass
class BaselineArtifacts:
    model_name: str
    model: Any
    feature_columns: List[str]
    config: Dict[str, Any]
    metrics_summary: Dict[str, Any]
    label_map: Dict[int, str]


@dataclass
class PredictionResult:
    csv_path: str
    model_name: str
    sequence_id: str
    wide_sequence: pd.DataFrame
    frame_features: pd.DataFrame
    window_features: pd.DataFrame
    window_results: pd.DataFrame
    events: List[Dict[str, Any]]
    summary: Dict[str, Any]

    def to_serializable(self) -> Dict[str, Any]:
        return {
            "csv_path": self.csv_path,
            "model_name": self.model_name,
            "sequence_id": self.sequence_id,
            "summary": self.summary,
            "events": self.events,
            "window_results": self.window_results.to_dict(orient="records"),
        }


def _normalize_label_map(label_map: Dict[Any, Any]) -> Dict[int, str]:
    normalized: Dict[int, str] = {}
    for key, value in label_map.items():
        normalized[int(key)] = str(value)
    return normalized


def load_baseline_artifacts(artifacts_path: Path) -> BaselineArtifacts:
    artifacts_path = Path(artifacts_path)

    if artifacts_path.is_dir():
        metrics_path = artifacts_path / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"metrics.json not found inside: {artifacts_path}")

        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        model_name = metrics.get("best_model")
        if not model_name:
            raise ValueError("best_model missing in metrics.json.")

        model_path = artifacts_path / f"{model_name}_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        return BaselineArtifacts(
            model_name=model_name,
            model=joblib.load(model_path),
            feature_columns=list(metrics["feature_columns"]),
            config={
                "window": int(metrics.get("window", 30)),
                "stride": int(metrics.get("stride", 10)),
                "conf_threshold": float(metrics.get("conf_threshold", 0.20)),
            },
            metrics_summary=metrics.get("results", {}).get(model_name, {}),
            label_map={0: "no_fall", 1: "fall"},
        )

    bundle = joblib.load(artifacts_path)
    if not isinstance(bundle, dict) or bundle.get("bundle_type") != "poseguard_baseline_bundle":
        raise ValueError(
            "The provided file is not a supported baseline bundle. "
            "Run freeze_baseline.py first or pass the training output directory."
        )

    return BaselineArtifacts(
        model_name=str(bundle["model_name"]),
        model=bundle["model"],
        feature_columns=list(bundle["feature_columns"]),
        config=dict(bundle["config"]),
        metrics_summary=dict(bundle.get("metrics_summary", {})),
        label_map=_normalize_label_map(bundle.get("label_map", {0: "no_fall", 1: "fall"})),
    )


def align_feature_columns(window_features: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    aligned = window_features.copy()
    for col in feature_columns:
        if col not in aligned.columns:
            aligned[col] = np.nan
    extra_cols = [c for c in aligned.columns if c not in feature_columns]
    if extra_cols:
        aligned = aligned.drop(columns=extra_cols)
    return aligned[feature_columns].copy()


def predict_positive_class_probabilities(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if probs.ndim != 2 or probs.shape[1] < 2:
            raise ValueError("predict_proba did not return a 2-class probability matrix.")
        return np.asarray(probs[:, 1], dtype=float)

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-scores))

    preds = np.asarray(model.predict(X), dtype=float)
    return np.clip(preds, 0.0, 1.0)


def predict_sequence(
    csv_path: Path,
    artifacts: BaselineArtifacts,
    alert_config: Optional[AlertConfig] = None,
) -> PredictionResult:
    csv_path = Path(csv_path)
    sequence_id = csv_path.stem

    wide_df, frame_feats, window_features, window_meta = build_sequence_windows_from_csv(
        csv_path=csv_path,
        window=int(artifacts.config["window"]),
        stride=int(artifacts.config["stride"]),
        conf_threshold=float(artifacts.config["conf_threshold"]),
        collect_diag=False,
    )

    X = align_feature_columns(window_features, artifacts.feature_columns)
    fall_probabilities = predict_positive_class_probabilities(artifacts.model, X)

    post = postprocess_window_probabilities(
        window_probabilities=fall_probabilities,
        window_metadata=window_meta,
        config=alert_config or AlertConfig(),
    )

    window_results = window_meta.copy()
    window_results["fall_probability"] = fall_probabilities
    window_results["smoothed_probability"] = post["smoothed_probabilities"]
    window_results["binary_prediction"] = post["binary_window_predictions"]
    window_results["predicted_label"] = window_results["binary_prediction"].map({0: "no_fall", 1: "fall"})

    summary = {
        "decision": post["summary"]["decision"],
        "alert": post["summary"]["alert"],
        "n_events": post["summary"]["n_events"],
        "n_frames": int(len(wide_df)),
        "n_windows": int(len(window_results)),
        "max_window_probability": post["summary"]["max_window_probability"],
        "mean_window_probability": post["summary"]["mean_window_probability"],
        "positive_window_ratio": post["summary"]["positive_window_ratio"],
        "baseline_config": dict(artifacts.config),
        "alert_config": post["summary"]["config"],
        "model_name": artifacts.model_name,
        "feature_count": len(artifacts.feature_columns),
    }

    return PredictionResult(
        csv_path=str(csv_path),
        model_name=artifacts.model_name,
        sequence_id=sequence_id,
        wide_sequence=wide_df,
        frame_features=frame_feats,
        window_features=X,
        window_results=window_results,
        events=post["events"],
        summary=summary,
    )


def save_prediction_outputs(
    result: PredictionResult,
    save_json: Optional[Path] = None,
    save_window_csv: Optional[Path] = None,
    save_frame_csv: Optional[Path] = None,
) -> None:
    if save_json is not None:
        save_json = Path(save_json)
        save_json.parent.mkdir(parents=True, exist_ok=True)
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(result.to_serializable(), f, indent=2)

    if save_window_csv is not None:
        save_window_csv = Path(save_window_csv)
        save_window_csv.parent.mkdir(parents=True, exist_ok=True)
        result.window_results.to_csv(save_window_csv, index=False)

    if save_frame_csv is not None:
        save_frame_csv = Path(save_frame_csv)
        save_frame_csv.parent.mkdir(parents=True, exist_ok=True)
        result.wide_sequence.to_csv(save_frame_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts",
        type=Path,
        required=True,
        help="Path to baseline_bundle.joblib or to a training output directory containing metrics.json.",
    )
    parser.add_argument("--csv_path", type=Path, required=True, help="Path to a single keypoints CSV file.")
    parser.add_argument("--save_json", type=Path, default=None, help="Optional path to save a JSON prediction summary.")
    parser.add_argument("--save_window_csv", type=Path, default=None, help="Optional path to save per-window results.")
    parser.add_argument("--save_frame_csv", type=Path, default=None, help="Optional path to save frame-level wide skeleton data.")
    parser.add_argument("--window_threshold", type=float, default=0.50, help="Probability threshold for positive windows.")
    parser.add_argument("--min_positive_run", type=int, default=3, help="Minimum number of consecutive positive windows for an alert.")
    parser.add_argument("--merge_gap_windows", type=int, default=1, help="Merge events separated by short gaps.")
    parser.add_argument("--cooldown_windows", type=int, default=3, help="Suppress repeated nearby alerts.")
    parser.add_argument("--probability_smoothing", type=int, default=3, help="Moving-average smoothing window over probabilities.")
    args = parser.parse_args()

    artifacts = load_baseline_artifacts(args.artifacts)
    alert_config = AlertConfig(
        window_threshold=args.window_threshold,
        min_positive_run=args.min_positive_run,
        merge_gap_windows=args.merge_gap_windows,
        cooldown_windows=args.cooldown_windows,
        probability_smoothing=args.probability_smoothing,
    )

    result = predict_sequence(csv_path=args.csv_path, artifacts=artifacts, alert_config=alert_config)
    save_prediction_outputs(
        result,
        save_json=args.save_json,
        save_window_csv=args.save_window_csv,
        save_frame_csv=args.save_frame_csv,
    )

    print(json.dumps(result.summary, indent=2))
    if result.events:
        print(json.dumps(result.events, indent=2))
    else:
        print("No fall events detected.")


if __name__ == "__main__":
    main()
