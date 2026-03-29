from __future__ import annotations

import math
import queue
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from camera_source import put_latest
from feature_service import SlidingWindowFeatureService, WindowPacket, align_feature_columns


@dataclass
class BaselineBundleArtifacts:
    bundle_path: str
    model_name: str
    model: Any
    feature_columns: List[str]
    config: Dict[str, Any]
    label_map: Dict[int, str]
    metrics_summary: Dict[str, Any]


@dataclass
class InferenceResult:
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
    model_name: str
    feature_count: int
    fall_probability: float
    predicted_label: int
    predicted_label_name: str
    threshold: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class InferenceWorkerStats:
    pose_frames_consumed: int = 0
    windows_built: int = 0
    predictions_emitted: int = 0
    queue_drops: int = 0
    last_probability: Optional[float] = None
    last_predicted_label: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


def load_baseline_bundle(bundle_path: str | Path) -> BaselineBundleArtifacts:
    bundle_path = Path(bundle_path)
    if not bundle_path.exists():
        raise FileNotFoundError(f"baseline bundle not found: {bundle_path}")

    bundle = joblib.load(bundle_path)
    if not isinstance(bundle, dict):
        raise ValueError("baseline bundle must be a dict-like artifact.")
    if bundle.get("bundle_type") != "poseguard_baseline_bundle":
        raise ValueError(f"unexpected bundle_type: {bundle.get('bundle_type')!r}")

    raw_label_map = bundle.get("label_map", {0: "no_fall", 1: "fall"})
    label_map = {int(k): str(v) for k, v in raw_label_map.items()}

    return BaselineBundleArtifacts(
        bundle_path=str(bundle_path),
        model_name=str(bundle.get("model_name", "unknown_model")),
        model=bundle["model"],
        feature_columns=list(bundle["feature_columns"]),
        config=dict(bundle.get("config", {})),
        label_map=label_map,
        metrics_summary=dict(bundle.get("metrics_summary", {})),
    )


class InferenceWorker:
    """Consume PoseFrames, build sliding windows, and emit fall probabilities."""

    def __init__(
        self,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        *,
        baseline_bundle_path: str | Path,
        threshold: float = 0.50,
        window: Optional[int] = None,
        stride: Optional[int] = None,
        conf_threshold: Optional[float] = None,
        poll_timeout: float = 0.2,
    ) -> None:
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.threshold = float(threshold)
        self.poll_timeout = float(poll_timeout)
        self.artifacts = load_baseline_bundle(baseline_bundle_path)

        bundle_cfg = self.artifacts.config
        self.feature_service = SlidingWindowFeatureService(
            window=int(window if window is not None else bundle_cfg.get("window", 30)),
            stride=int(stride if stride is not None else bundle_cfg.get("stride", 10)),
            conf_threshold=float(conf_threshold if conf_threshold is not None else bundle_cfg.get("conf_threshold", 0.20)),
        )

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_error: Optional[str] = None
        self._stats = InferenceWorkerStats()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="InferenceWorker",
            daemon=True,
        )
        self._thread.start()

    def stop(self, join_timeout: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=join_timeout)

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    @property
    def last_error(self) -> Optional[str]:
        with self._lock:
            return self._last_error

    def _set_error(self, message: Optional[str]) -> None:
        with self._lock:
            self._last_error = message

    def get_status(self) -> dict:
        with self._lock:
            return {
                "alive": self.is_alive(),
                "model_name": self.artifacts.model_name,
                "bundle_path": self.artifacts.bundle_path,
                "threshold": self.threshold,
                "last_error": self._last_error,
                "feature_service": self.feature_service.get_status(),
                **self._stats.to_dict(),
            }

    def _predict_probability(self, aligned_features: pd.DataFrame) -> float:
        model = self.artifacts.model

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(aligned_features)
            proba = np.asarray(proba, dtype=float)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return float(proba[0, 1])
            if proba.ndim == 2 and proba.shape[1] == 1:
                return float(proba[0, 0])

        if hasattr(model, "decision_function"):
            score = np.asarray(model.decision_function(aligned_features), dtype=float).reshape(-1)[0]
            return float(1.0 / (1.0 + math.exp(-score)))

        pred = np.asarray(model.predict(aligned_features), dtype=float).reshape(-1)[0]
        return float(pred)

    def _build_result(self, window_packet: WindowPacket, fall_probability: float) -> InferenceResult:
        predicted_label = 1 if fall_probability >= self.threshold else 0
        predicted_label_name = self.artifacts.label_map.get(predicted_label, str(predicted_label))

        return InferenceResult(
            ts_ms=int(window_packet.ts_ms),
            camera_id=window_packet.camera_id,
            person_id=window_packet.person_id,
            window_index=window_packet.window_index,
            start_ts_ms=window_packet.start_ts_ms,
            end_ts_ms=window_packet.end_ts_ms,
            start_frame_id=window_packet.start_frame_id,
            end_frame_id=window_packet.end_frame_id,
            n_frames=window_packet.n_frames,
            source_fps=window_packet.source_fps,
            detected_ratio=window_packet.detected_ratio,
            mean_pose_conf=window_packet.mean_pose_conf,
            model_name=self.artifacts.model_name,
            feature_count=len(self.artifacts.feature_columns),
            fall_probability=float(fall_probability),
            predicted_label=int(predicted_label),
            predicted_label_name=predicted_label_name,
            threshold=self.threshold,
        )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                pose_frame = self.input_queue.get(timeout=self.poll_timeout)
            except queue.Empty:
                continue

            try:
                self._stats.pose_frames_consumed += 1
                window_packet = self.feature_service.push(pose_frame)
                if window_packet is None:
                    continue

                self._stats.windows_built += 1
                feature_df = pd.DataFrame([window_packet.feature_vector])
                aligned = align_feature_columns(feature_df, self.artifacts.feature_columns)
                fall_probability = self._predict_probability(aligned)
                result = self._build_result(window_packet, fall_probability)

                was_full = self.output_queue.full()
                put_latest(self.output_queue, result)
                if was_full:
                    self._stats.queue_drops += 1

                self._stats.predictions_emitted += 1
                self._stats.last_probability = float(result.fall_probability)
                self._stats.last_predicted_label = int(result.predicted_label)
                self._set_error(None)
            except Exception as exc:
                self._set_error(str(exc))
