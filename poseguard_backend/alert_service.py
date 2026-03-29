from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from camera_source import put_latest


@dataclass
class AlertEnvelope:
    ts_ms: int
    event: Dict[str, Any]
    transition: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AlertService:
    """Mock notifier for development.

    Current behavior:
    - prints a highly visible alert block to the terminal
    - optionally appends JSON lines to a local log file
    """

    def __init__(
        self,
        *,
        log_path: Optional[str | Path] = None,
        queue_size: int = 32,
        poll_timeout: float = 0.2,
    ) -> None:
        self.log_path = str(Path(log_path).expanduser()) if log_path else None
        if self.log_path:
            Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)

        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._poll_timeout = float(poll_timeout)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_error: Optional[str] = None
        self._stats = {
            "alerts_enqueued": 0,
            "alerts_sent": 0,
            "queue_drops": 0,
            "last_alert_event_id": None,
            "last_alert_ts_ms": None,
        }

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="AlertService",
            daemon=True,
        )
        self._thread.start()

    def stop(self, join_timeout: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=join_timeout)

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "alive": self.is_alive(),
                "log_path": self.log_path,
                "last_error": self._last_error,
                **self._stats,
            }

    def enqueue_event(self, event_payload: Dict[str, Any], transition_payload: Dict[str, Any]) -> None:
        envelope = AlertEnvelope(
            ts_ms=int(time.time() * 1000),
            event=dict(event_payload),
            transition=dict(transition_payload),
        )
        was_full = self._queue.full()
        put_latest(self._queue, envelope)
        with self._lock:
            self._stats["alerts_enqueued"] += 1
            if was_full:
                self._stats["queue_drops"] += 1

    def _set_error(self, message: Optional[str]) -> None:
        with self._lock:
            self._last_error = message

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                envelope: AlertEnvelope = self._queue.get(timeout=self._poll_timeout)
            except queue.Empty:
                continue

            try:
                payload = envelope.to_dict()
                event = payload["event"]
                event_id = event.get("event_id")
                pretty = json.dumps(payload, ensure_ascii=False, indent=2)
                banner = (
                    "\n" + "=" * 72 + "\n"
                    "POSEGUARD MOCK FALL ALERT\n"
                    f"Event ID: {event_id}\n"
                    f"Camera: {event.get('camera_id')} | State: {event.get('state')}\n"
                    f"Peak Probability: {event.get('peak_probability')}\n"
                    + "-" * 72 + "\n"
                    + pretty
                    + "\n" + "=" * 72 + "\n"
                )
                print(banner, flush=True)

                if self.log_path:
                    with open(self.log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

                with self._lock:
                    self._stats["alerts_sent"] += 1
                    self._stats["last_alert_event_id"] = event_id
                    self._stats["last_alert_ts_ms"] = payload["ts_ms"]
                self._set_error(None)
            except Exception as exc:  # pragma: no cover
                self._set_error(str(exc))
