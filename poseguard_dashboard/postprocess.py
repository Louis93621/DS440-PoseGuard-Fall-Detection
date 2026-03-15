#!/usr/bin/env python3
"""
Temporal alert post-processing for window-level fall probabilities.

This module converts noisy per-window probabilities into stable event-level alerts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class AlertConfig:
    window_threshold: float = 0.50
    min_positive_run: int = 3
    merge_gap_windows: int = 1
    cooldown_windows: int = 3
    probability_smoothing: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _as_numpy(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.ndim != 1:
        raise ValueError("Window probabilities must be a 1D sequence.")
    return arr


def smooth_probabilities(probabilities: Sequence[float], kernel_size: int = 3) -> np.ndarray:
    probs = _as_numpy(probabilities)
    if probs.size == 0:
        return probs
    kernel_size = max(1, int(kernel_size))
    if kernel_size == 1 or probs.size == 1:
        return probs.copy()
    kernel = np.ones(kernel_size, dtype=float) / float(kernel_size)
    return np.convolve(probs, kernel, mode="same")


def find_positive_runs(binary_flags: Sequence[bool]) -> List[Tuple[int, int]]:
    flags = np.asarray(binary_flags, dtype=bool)
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None

    for idx, flag in enumerate(flags):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            runs.append((start, idx))
            start = None

    if start is not None:
        runs.append((start, len(flags)))

    return runs


def filter_short_runs(runs: Sequence[Tuple[int, int]], min_len: int) -> List[Tuple[int, int]]:
    min_len = max(1, int(min_len))
    return [(start, end) for start, end in runs if (end - start) >= min_len]


def merge_runs(runs: Sequence[Tuple[int, int]], max_gap: int) -> List[Tuple[int, int]]:
    if not runs:
        return []
    max_gap = max(0, int(max_gap))
    merged: List[Tuple[int, int]] = [tuple(runs[0])]

    for start, end in runs[1:]:
        last_start, last_end = merged[-1]
        gap = start - last_end
        if gap <= max_gap:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def apply_cooldown(runs: Sequence[Tuple[int, int]], cooldown_windows: int) -> List[Tuple[int, int]]:
    if not runs:
        return []
    cooldown_windows = max(0, int(cooldown_windows))
    cooled: List[Tuple[int, int]] = [tuple(runs[0])]

    for start, end in runs[1:]:
        prev_start, prev_end = cooled[-1]
        if start - prev_end <= cooldown_windows:
            cooled[-1] = (prev_start, max(prev_end, end))
        else:
            cooled.append((start, end))
    return cooled


def probability_to_severity(max_probability: float) -> str:
    if max_probability >= 0.90:
        return "high"
    if max_probability >= 0.75:
        return "medium"
    return "low"


def build_event_records(
    runs: Sequence[Tuple[int, int]],
    raw_probabilities: Sequence[float],
    smoothed_probabilities: Sequence[float],
    window_metadata: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    raw_probs = _as_numpy(raw_probabilities)
    smooth_probs = _as_numpy(smoothed_probabilities)

    if window_metadata is None:
        metadata_records: List[Dict[str, Any]] = [{"window_index": i} for i in range(len(raw_probs))]
    elif isinstance(window_metadata, pd.DataFrame):
        metadata_records = window_metadata.to_dict(orient="records")
    else:
        metadata_records = list(window_metadata)

    events: List[Dict[str, Any]] = []

    for event_id, (start, end) in enumerate(runs, start=1):
        # end is exclusive internally; convert to inclusive for reporting.
        start_idx = int(start)
        end_idx = int(end - 1)
        event_raw = raw_probs[start:end]
        event_smooth = smooth_probs[start:end]

        start_meta = metadata_records[start_idx] if start_idx < len(metadata_records) else {}
        end_meta = metadata_records[end_idx] if end_idx < len(metadata_records) else {}

        max_probability = float(np.max(event_raw)) if len(event_raw) else 0.0
        mean_probability = float(np.mean(event_raw)) if len(event_raw) else 0.0

        events.append(
            {
                "event_id": int(event_id),
                "event_type": "fall",
                "start_window_index": start_idx,
                "end_window_index": end_idx,
                "duration_windows": int(end - start),
                "frame_start": start_meta.get("frame_start"),
                "frame_end": end_meta.get("frame_end"),
                "max_probability": max_probability,
                "mean_probability": mean_probability,
                "max_smoothed_probability": float(np.max(event_smooth)) if len(event_smooth) else 0.0,
                "severity": probability_to_severity(max_probability),
            }
        )
    return events


def postprocess_window_probabilities(
    window_probabilities: Sequence[float],
    window_metadata: Optional[Sequence[Dict[str, Any]]] = None,
    config: Optional[AlertConfig] = None,
) -> Dict[str, Any]:
    config = config or AlertConfig()
    raw_probs = _as_numpy(window_probabilities)

    if raw_probs.size == 0:
        return {
            "summary": {
                "alert": False,
                "n_events": 0,
                "max_window_probability": 0.0,
                "mean_window_probability": 0.0,
                "positive_window_ratio": 0.0,
                "decision": "No Fall Detected",
                "config": config.to_dict(),
            },
            "events": [],
            "window_probabilities": raw_probs.tolist(),
            "smoothed_probabilities": raw_probs.tolist(),
            "binary_window_predictions": [],
        }

    smoothed_probs = smooth_probabilities(raw_probs, kernel_size=config.probability_smoothing)
    binary_flags = smoothed_probs >= float(config.window_threshold)

    runs = find_positive_runs(binary_flags)
    runs = filter_short_runs(runs, min_len=config.min_positive_run)
    runs = merge_runs(runs, max_gap=config.merge_gap_windows)
    runs = apply_cooldown(runs, cooldown_windows=config.cooldown_windows)

    events = build_event_records(runs, raw_probs, smoothed_probs, window_metadata=window_metadata)

    summary = {
        "alert": bool(len(events) > 0),
        "n_events": int(len(events)),
        "max_window_probability": float(np.max(raw_probs)),
        "mean_window_probability": float(np.mean(raw_probs)),
        "positive_window_ratio": float(np.mean(binary_flags)),
        "decision": "Fall Event Detected" if events else "No Fall Detected",
        "config": config.to_dict(),
    }

    return {
        "summary": summary,
        "events": events,
        "window_probabilities": raw_probs.tolist(),
        "smoothed_probabilities": smoothed_probs.tolist(),
        "binary_window_predictions": binary_flags.astype(int).tolist(),
    }
