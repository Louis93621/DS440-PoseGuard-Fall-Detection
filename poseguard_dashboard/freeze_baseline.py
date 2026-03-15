#!/usr/bin/env python3
"""
Freeze a trained baseline into a single reusable bundle for inference and demos.

This script packages:
- the trained model pipeline,
- feature column order,
- baseline config (window / stride / conf threshold),
- the best-model metrics summary,
- label mapping metadata.

Example:
    python freeze_baseline.py --output_dir "/Users/you/Desktop/DS 440 Project/Output_v2"
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import joblib


def load_metrics(metrics_path: Path) -> Dict[str, Any]:
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found at: {metrics_path}")
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Training output directory containing metrics.json and *_model.joblib",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Optional model name override. Defaults to metrics.json['best_model'].",
    )
    parser.add_argument(
        "--bundle_name",
        type=str,
        default="baseline_bundle.joblib",
        help="Filename for the frozen bundle.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    metrics = load_metrics(output_dir / "metrics.json")

    model_name = args.model_name or metrics.get("best_model")
    if not model_name:
        raise SystemExit("Could not determine model_name. Pass --model_name explicitly.")

    model_path = output_dir / f"{model_name}_model.joblib"
    if not model_path.exists():
        raise SystemExit(f"Model file not found: {model_path}")

    feature_columns = metrics.get("feature_columns")
    if not feature_columns:
        raise SystemExit("feature_columns not found in metrics.json.")

    model = joblib.load(model_path)

    bundle = {
        "bundle_type": "poseguard_baseline_bundle",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "model": model,
        "feature_columns": list(feature_columns),
        "config": {
            "window": int(metrics.get("window", 30)),
            "stride": int(metrics.get("stride", 10)),
            "conf_threshold": float(metrics.get("conf_threshold", 0.20)),
        },
        "metrics_summary": metrics.get("results", {}).get(model_name, {}),
        "training_summary": {
            "n_files_total": int(metrics.get("n_files_total", 0)),
            "n_files_used": int(metrics.get("n_files_used", 0)),
            "n_files_skipped": int(metrics.get("n_files_skipped", 0)),
            "n_windows": int(metrics.get("n_windows", 0)),
            "n_train_windows": int(metrics.get("n_train_windows", 0)),
            "n_test_windows": int(metrics.get("n_test_windows", 0)),
            "n_sequences_total": int(metrics.get("n_sequences_total", 0)),
            "n_sequences_train": int(metrics.get("n_sequences_train", 0)),
            "n_sequences_test": int(metrics.get("n_sequences_test", 0)),
            "class_balance": metrics.get("class_balance", {}),
        },
        "label_map": {0: "no_fall", 1: "fall"},
    }

    bundle_path = output_dir / args.bundle_name
    manifest_path = output_dir / "baseline_manifest.json"

    joblib.dump(bundle, bundle_path)

    manifest = {k: v for k, v in bundle.items() if k != "model"}
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved bundle: {bundle_path}")
    print(f"Saved manifest: {manifest_path}")
    print(f"Frozen model: {model_name}")


if __name__ == "__main__":
    main()
