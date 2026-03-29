from __future__ import annotations

import copy
import json
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ManagedFallEvent:
    event_id: str
    camera_id: str
    person_id: int
    state: str
    ack_status: str
    created_at_ts_ms: int
    first_seen_ts_ms: int
    confirmed_at_ts_ms: Optional[int]
    closed_at_ts_ms: Optional[int]
    first_window_index: int
    last_window_index: int
    start_ts_ms: int
    end_ts_ms: int
    start_frame_id: int
    end_frame_id: int
    peak_probability: float
    last_probability: float
    threshold: float
    positive_windows: int
    negative_windows: int
    notified: bool
    last_updated_ts_ms: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        return payload


@dataclass
class EventTransition:
    transition: str
    state: str
    reason: str
    ts_ms: int
    event: Dict[str, Any]
    should_notify: bool = False
    persisted: bool = False
    cooldown_until_ms: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SQLiteEventStore:
    """Very small SQLite repository for confirmed fall events."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(Path(db_path).expanduser())
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fall_events (
                    event_id TEXT PRIMARY KEY,
                    camera_id TEXT NOT NULL,
                    person_id INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    ack_status TEXT NOT NULL,
                    created_at_ts_ms INTEGER NOT NULL,
                    first_seen_ts_ms INTEGER NOT NULL,
                    confirmed_at_ts_ms INTEGER,
                    closed_at_ts_ms INTEGER,
                    first_window_index INTEGER NOT NULL,
                    last_window_index INTEGER NOT NULL,
                    start_ts_ms INTEGER NOT NULL,
                    end_ts_ms INTEGER NOT NULL,
                    start_frame_id INTEGER NOT NULL,
                    end_frame_id INTEGER NOT NULL,
                    peak_probability REAL NOT NULL,
                    last_probability REAL NOT NULL,
                    threshold REAL NOT NULL,
                    positive_windows INTEGER NOT NULL,
                    negative_windows INTEGER NOT NULL,
                    notified INTEGER NOT NULL DEFAULT 0,
                    last_updated_ts_ms INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def upsert_event(self, event: ManagedFallEvent) -> None:
        payload = event.to_dict()
        metadata_json = json.dumps(payload.get("metadata", {}), ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO fall_events (
                    event_id, camera_id, person_id, state, ack_status,
                    created_at_ts_ms, first_seen_ts_ms, confirmed_at_ts_ms, closed_at_ts_ms,
                    first_window_index, last_window_index, start_ts_ms, end_ts_ms,
                    start_frame_id, end_frame_id, peak_probability, last_probability,
                    threshold, positive_windows, negative_windows, notified,
                    last_updated_ts_ms, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    camera_id=excluded.camera_id,
                    person_id=excluded.person_id,
                    state=excluded.state,
                    ack_status=excluded.ack_status,
                    confirmed_at_ts_ms=excluded.confirmed_at_ts_ms,
                    closed_at_ts_ms=excluded.closed_at_ts_ms,
                    last_window_index=excluded.last_window_index,
                    end_ts_ms=excluded.end_ts_ms,
                    end_frame_id=excluded.end_frame_id,
                    peak_probability=excluded.peak_probability,
                    last_probability=excluded.last_probability,
                    positive_windows=excluded.positive_windows,
                    negative_windows=excluded.negative_windows,
                    notified=excluded.notified,
                    last_updated_ts_ms=excluded.last_updated_ts_ms,
                    metadata_json=excluded.metadata_json
                """,
                (
                    payload["event_id"],
                    payload["camera_id"],
                    payload["person_id"],
                    payload["state"],
                    payload["ack_status"],
                    payload["created_at_ts_ms"],
                    payload["first_seen_ts_ms"],
                    payload["confirmed_at_ts_ms"],
                    payload["closed_at_ts_ms"],
                    payload["first_window_index"],
                    payload["last_window_index"],
                    payload["start_ts_ms"],
                    payload["end_ts_ms"],
                    payload["start_frame_id"],
                    payload["end_frame_id"],
                    payload["peak_probability"],
                    payload["last_probability"],
                    payload["threshold"],
                    payload["positive_windows"],
                    payload["negative_windows"],
                    1 if payload["notified"] else 0,
                    payload["last_updated_ts_ms"],
                    metadata_json,
                ),
            )
            self._conn.commit()

    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM fall_events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def list_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM fall_events ORDER BY created_at_ts_ms DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def update_ack_status(self, event_id: str, ack_status: str, *, ts_ms: Optional[int] = None) -> Optional[Dict[str, Any]]:
        ts_ms = int(ts_ms or time.time() * 1000)
        row = self.get_event(event_id)
        if row is None:
            return None
        metadata = dict(row.get("metadata", {}))
        metadata["ack_updated_at_ts_ms"] = ts_ms
        metadata_json = json.dumps(metadata, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                """
                UPDATE fall_events
                SET ack_status = ?,
                    last_updated_ts_ms = ?,
                    metadata_json = ?
                WHERE event_id = ?
                """,
                (ack_status, ts_ms, metadata_json, event_id),
            )
            self._conn.commit()
        return self.get_event(event_id)

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        payload = dict(row)
        payload["notified"] = bool(payload.get("notified", 0))
        try:
            payload["metadata"] = json.loads(payload.pop("metadata_json") or "{}")
        except Exception:
            payload["metadata"] = {}
            payload.pop("metadata_json", None)
        return payload


class EventManager:
    """Turn noisy window-level inference into stable fall events.

    State machine:
        NORMAL -> SUSPECTED -> VERIFYING -> CONFIRMED

    Cooldown is applied after an event is closed so the same fall sequence does
    not generate dozens of duplicate alerts.
    """

    def __init__(
        self,
        *,
        store: SQLiteEventStore,
        threshold: Optional[float] = None,
        suspected_positive_windows: int = 2,
        verify_positive_windows: int = 3,
        verify_timeout_windows: int = 6,
        reset_negative_windows: int = 2,
        confirmed_clear_negative_windows: int = 3,
        min_confirm_probability: float = 0.65,
        cooldown_ms: int = 15000,
        event_gap_ms: int = 4000,
    ) -> None:
        self.store = store
        self.threshold = threshold
        self.suspected_positive_windows = int(suspected_positive_windows)
        self.verify_positive_windows = int(verify_positive_windows)
        self.verify_timeout_windows = int(verify_timeout_windows)
        self.reset_negative_windows = int(reset_negative_windows)
        self.confirmed_clear_negative_windows = int(confirmed_clear_negative_windows)
        self.min_confirm_probability = float(min_confirm_probability)
        self.cooldown_ms = int(cooldown_ms)
        self.event_gap_ms = int(event_gap_ms)

        self._lock = threading.Lock()
        self._state = "NORMAL"
        self._current_event: Optional[ManagedFallEvent] = None
        self._last_confirmed_event: Optional[Dict[str, Any]] = None
        self._cooldown_until_ms: int = 0
        self._stats = {
            "transitions": 0,
            "confirmed_events": 0,
            "cooldown_suppressions": 0,
            "candidate_resets": 0,
        }

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "state": self._state,
                "cooldown_until_ms": self._cooldown_until_ms,
                "in_cooldown": self._in_cooldown_locked(int(time.time() * 1000)),
                "current_event_id": self._current_event.event_id if self._current_event else None,
                "current_event": self._current_event.to_dict() if self._current_event else None,
                "last_confirmed_event": copy.deepcopy(self._last_confirmed_event),
                **self._stats,
            }

    def snapshot(self) -> Dict[str, Any]:
        return self.get_status()

    def list_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self.store.list_events(limit=limit)

    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self._current_event is not None and self._current_event.event_id == event_id:
                return self._current_event.to_dict()
        return self.store.get_event(event_id)

    def acknowledge_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self._current_event is not None and self._current_event.event_id == event_id:
                self._current_event.ack_status = "ACKNOWLEDGED"
                self._current_event.last_updated_ts_ms = int(time.time() * 1000)
                self.store.upsert_event(self._current_event)
                return self._current_event.to_dict()
        return self.store.update_ack_status(event_id, "ACKNOWLEDGED")

    def dismiss_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self._current_event is not None and self._current_event.event_id == event_id:
                self._current_event.ack_status = "DISMISSED"
                self._current_event.last_updated_ts_ms = int(time.time() * 1000)
                self._current_event.closed_at_ts_ms = self._current_event.last_updated_ts_ms
                self.store.upsert_event(self._current_event)
                payload = self._current_event.to_dict()
                self._last_confirmed_event = copy.deepcopy(payload)
                self._current_event = None
                self._state = "NORMAL"
                self._cooldown_until_ms = payload["closed_at_ts_ms"] + self.cooldown_ms
                return payload
        return self.store.update_ack_status(event_id, "DISMISSED")

    def process_inference(self, inference_result: Dict[str, Any]) -> List[EventTransition]:
        if hasattr(inference_result, "to_dict"):
            inference = inference_result.to_dict()
        else:
            inference = dict(inference_result)

        now_ts_ms = int(inference.get("end_ts_ms") or inference.get("ts_ms") or time.time() * 1000)
        threshold = float(self.threshold if self.threshold is not None else inference.get("threshold", 0.5))
        probability = float(inference.get("fall_probability", 0.0) or 0.0)
        is_positive = probability >= threshold

        with self._lock:
            transitions: List[EventTransition] = []

            if self._current_event is not None:
                gap = now_ts_ms - int(self._current_event.end_ts_ms)
                if gap > self.event_gap_ms:
                    transitions.extend(self._reset_candidate_locked(now_ts_ms, reason="gap_timeout"))

            if self._state == "NORMAL" and self._in_cooldown_locked(now_ts_ms):
                if is_positive:
                    self._stats["cooldown_suppressions"] += 1
                return []

            if self._state == "NORMAL":
                if is_positive:
                    event = self._create_event_locked(inference, threshold)
                    event.state = "SUSPECTED"
                    event.positive_windows = 1
                    self._current_event = event
                    self._state = "SUSPECTED"
                    transitions.append(self._transition_locked("entered_suspected", "candidate_started", event, now_ts_ms))
                return transitions

            if self._current_event is None:
                self._state = "NORMAL"
                return transitions

            self._update_event_locked(self._current_event, inference, probability, is_positive, now_ts_ms)

            if self._state == "SUSPECTED":
                if is_positive:
                    if self._current_event.positive_windows >= self.suspected_positive_windows:
                        self._current_event.state = "VERIFYING"
                        self._state = "VERIFYING"
                        transitions.append(self._transition_locked("entered_verifying", "enough_consecutive_positive_windows", self._current_event, now_ts_ms))
                else:
                    if self._current_event.negative_windows >= self.reset_negative_windows:
                        transitions.extend(self._reset_candidate_locked(now_ts_ms, reason="candidate_negative_reset"))
                return transitions

            if self._state == "VERIFYING":
                total_windows = self._current_event.last_window_index - self._current_event.first_window_index + 1
                if is_positive:
                    should_confirm = (
                        self._current_event.positive_windows >= self.verify_positive_windows
                        and self._current_event.peak_probability >= self.min_confirm_probability
                    )
                    if should_confirm:
                        self._current_event.state = "CONFIRMED"
                        self._state = "CONFIRMED"
                        self._current_event.confirmed_at_ts_ms = now_ts_ms
                        self._current_event.last_updated_ts_ms = now_ts_ms
                        self.store.upsert_event(self._current_event)
                        self._last_confirmed_event = copy.deepcopy(self._current_event.to_dict())
                        self._stats["confirmed_events"] += 1
                        transitions.append(
                            self._transition_locked(
                                "confirmed",
                                "verification_passed",
                                self._current_event,
                                now_ts_ms,
                                should_notify=True,
                                persisted=True,
                            )
                        )
                        return transitions
                if self._current_event.negative_windows >= self.reset_negative_windows or total_windows >= self.verify_timeout_windows:
                    reason = "verification_negative_reset" if self._current_event.negative_windows >= self.reset_negative_windows else "verification_timeout"
                    transitions.extend(self._reset_candidate_locked(now_ts_ms, reason=reason))
                return transitions

            if self._state == "CONFIRMED":
                self.store.upsert_event(self._current_event)
                if not is_positive and self._current_event.negative_windows >= self.confirmed_clear_negative_windows:
                    self._current_event.closed_at_ts_ms = now_ts_ms
                    self._current_event.last_updated_ts_ms = now_ts_ms
                    self.store.upsert_event(self._current_event)
                    payload = copy.deepcopy(self._current_event.to_dict())
                    self._last_confirmed_event = copy.deepcopy(payload)
                    self._cooldown_until_ms = now_ts_ms + self.cooldown_ms
                    transitions.append(
                        EventTransition(
                            transition="closed",
                            state="CONFIRMED",
                            reason="confirmed_event_closed_cooldown_started",
                            ts_ms=now_ts_ms,
                            event=payload,
                            should_notify=False,
                            persisted=True,
                            cooldown_until_ms=self._cooldown_until_ms,
                        )
                    )
                    self._current_event = None
                    self._state = "NORMAL"
                    self._stats["transitions"] += 1
                return transitions

            return transitions

    def _create_event_locked(self, inference: Dict[str, Any], threshold: float) -> ManagedFallEvent:
        ts_ms = int(inference.get("end_ts_ms") or inference.get("ts_ms") or time.time() * 1000)
        event_id = f"evt_{time.strftime('%Y%m%d_%H%M%S', time.localtime(ts_ms / 1000.0))}_{uuid.uuid4().hex[:8]}"
        return ManagedFallEvent(
            event_id=event_id,
            camera_id=str(inference.get("camera_id", "unknown_camera")),
            person_id=int(inference.get("person_id", 0) or 0),
            state="NORMAL",
            ack_status="PENDING",
            created_at_ts_ms=ts_ms,
            first_seen_ts_ms=ts_ms,
            confirmed_at_ts_ms=None,
            closed_at_ts_ms=None,
            first_window_index=int(inference.get("window_index", 0) or 0),
            last_window_index=int(inference.get("window_index", 0) or 0),
            start_ts_ms=int(inference.get("start_ts_ms", ts_ms) or ts_ms),
            end_ts_ms=int(inference.get("end_ts_ms", ts_ms) or ts_ms),
            start_frame_id=int(inference.get("start_frame_id", 0) or 0),
            end_frame_id=int(inference.get("end_frame_id", 0) or 0),
            peak_probability=float(inference.get("fall_probability", 0.0) or 0.0),
            last_probability=float(inference.get("fall_probability", 0.0) or 0.0),
            threshold=float(threshold),
            positive_windows=0,
            negative_windows=0,
            notified=False,
            last_updated_ts_ms=ts_ms,
            metadata={
                "model_name": inference.get("model_name"),
                "source_fps": inference.get("source_fps"),
                "feature_count": inference.get("feature_count"),
            },
        )

    def _update_event_locked(
        self,
        event: ManagedFallEvent,
        inference: Dict[str, Any],
        probability: float,
        is_positive: bool,
        now_ts_ms: int,
    ) -> None:
        event.last_window_index = int(inference.get("window_index", event.last_window_index) or event.last_window_index)
        event.end_ts_ms = int(inference.get("end_ts_ms", event.end_ts_ms) or event.end_ts_ms)
        event.end_frame_id = int(inference.get("end_frame_id", event.end_frame_id) or event.end_frame_id)
        event.last_probability = float(probability)
        event.peak_probability = max(float(event.peak_probability), float(probability))
        event.last_updated_ts_ms = now_ts_ms
        if is_positive:
            event.positive_windows += 1
            event.negative_windows = 0
        else:
            event.negative_windows += 1

    def _transition_locked(
        self,
        transition: str,
        reason: str,
        event: ManagedFallEvent,
        ts_ms: int,
        *,
        should_notify: bool = False,
        persisted: bool = False,
    ) -> EventTransition:
        if should_notify:
            event.notified = True
            event.last_updated_ts_ms = ts_ms
            self.store.upsert_event(event)
        self._stats["transitions"] += 1
        return EventTransition(
            transition=transition,
            state=event.state,
            reason=reason,
            ts_ms=int(ts_ms),
            event=copy.deepcopy(event.to_dict()),
            should_notify=should_notify,
            persisted=persisted or should_notify,
            cooldown_until_ms=self._cooldown_until_ms if self._cooldown_until_ms > ts_ms else None,
        )

    def _reset_candidate_locked(self, ts_ms: int, *, reason: str) -> List[EventTransition]:
        transitions: List[EventTransition] = []
        if self._current_event is not None:
            payload = copy.deepcopy(self._current_event.to_dict())
            transitions.append(
                EventTransition(
                    transition="reset",
                    state=self._current_event.state,
                    reason=reason,
                    ts_ms=int(ts_ms),
                    event=payload,
                    should_notify=False,
                    persisted=False,
                    cooldown_until_ms=self._cooldown_until_ms if self._cooldown_until_ms > ts_ms else None,
                )
            )
            self._stats["transitions"] += 1
        self._current_event = None
        self._state = "NORMAL"
        self._stats["candidate_resets"] += 1
        return transitions

    def _in_cooldown_locked(self, now_ts_ms: int) -> bool:
        return now_ts_ms < self._cooldown_until_ms
