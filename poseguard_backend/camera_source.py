from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class FramePacket:
    """A single frame emitted by the simulated live camera source."""

    ts_ms: int
    camera_id: str
    frame_id: int
    raw_frame_index: int
    frame_bgr: np.ndarray
    width: int
    height: int
    source_fps: float
    video_path: str

    def to_meta_dict(self) -> dict:
        return {
            "ts_ms": self.ts_ms,
            "camera_id": self.camera_id,
            "frame_id": self.frame_id,
            "raw_frame_index": self.raw_frame_index,
            "width": self.width,
            "height": self.height,
            "source_fps": self.source_fps,
            "video_path": self.video_path,
        }


def put_latest(q: queue.Queue, item: object) -> None:
    """Put the newest item into a bounded queue.

    When the queue is full, drop the oldest item first so the pipeline stays
    low-latency instead of accumulating backlog.
    """
    try:
        q.put_nowait(item)
        return
    except queue.Full:
        pass

    try:
        q.get_nowait()
    except queue.Empty:
        pass

    try:
        q.put_nowait(item)
    except queue.Full:
        # Another producer may have filled the queue again. Drop the item.
        pass


class MP4CameraSource:
    """Read an MP4 file and emit frames at a fixed target FPS.

    This class simulates a live camera source while keeping the rest of the
    backend pipeline identical to a future real-camera implementation.
    """

    def __init__(
        self,
        video_path: str,
        output_queue: queue.Queue,
        *,
        target_fps: float = 12.0,
        camera_id: str = "living_room_cam_1",
        loop: bool = True,
        queue_size_hint: Optional[int] = None,
        resize_to: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.video_path = str(video_path)
        self.output_queue = output_queue
        self.target_fps = float(target_fps)
        self.camera_id = camera_id
        self.loop = loop
        self.resize_to = resize_to
        self.queue_size_hint = queue_size_hint

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_error: Optional[str] = None
        self._connected = False
        self._stats = {
            "frames_emitted": 0,
            "frames_read": 0,
            "frames_skipped": 0,
            "queue_drops": 0,
            "started_at_ms": None,
            "last_emit_ts_ms": None,
            "video_opened": False,
            "native_fps": None,
            "frame_step": 1,
        }

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"MP4CameraSource[{self.camera_id}]",
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
    def connected(self) -> bool:
        with self._lock:
            return self._connected

    @property
    def last_error(self) -> Optional[str]:
        with self._lock:
            return self._last_error

    def get_status(self) -> dict:
        with self._lock:
            return {
                "camera_id": self.camera_id,
                "video_path": self.video_path,
                "connected": self._connected,
                "alive": self.is_alive(),
                "target_fps": self.target_fps,
                "resize_to": self.resize_to,
                "last_error": self._last_error,
                **self._stats,
            }

    def _set_error(self, message: Optional[str]) -> None:
        with self._lock:
            self._last_error = message

    def _set_connected(self, value: bool) -> None:
        with self._lock:
            self._connected = value

    def _open_capture(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video file: {self.video_path}")
        return cap

    def _run(self) -> None:
        video_file = Path(self.video_path)
        if not video_file.exists():
            self._set_error(f"Video file not found: {video_file}")
            self._set_connected(False)
            return

        self._stats["started_at_ms"] = int(time.time() * 1000)
        emitted_frame_id = 0
        raw_frame_index = 0

        try:
            cap = self._open_capture()
        except Exception as exc:  # pragma: no cover - simple startup failure path
            self._set_error(str(exc))
            self._set_connected(False)
            return

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if native_fps <= 0:
            native_fps = self.target_fps
        frame_step = max(1, int(round(native_fps / self.target_fps))) if native_fps > self.target_fps else 1
        sleep_interval = 1.0 / max(self.target_fps, 1.0)
        next_emit_time = time.perf_counter()

        with self._lock:
            self._stats["video_opened"] = True
            self._stats["native_fps"] = native_fps
            self._stats["frame_step"] = frame_step

        self._set_connected(True)
        self._set_error(None)

        try:
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    if self.loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        raw_frame_index = 0
                        continue
                    break

                raw_frame_index += 1
                self._stats["frames_read"] += 1

                if frame_step > 1 and raw_frame_index % frame_step != 0:
                    self._stats["frames_skipped"] += 1
                    continue

                if self.resize_to is not None:
                    frame = cv2.resize(frame, self.resize_to, interpolation=cv2.INTER_AREA)

                now = time.perf_counter()
                if now < next_emit_time:
                    time.sleep(next_emit_time - now)
                next_emit_time = max(next_emit_time + sleep_interval, time.perf_counter())

                height, width = frame.shape[:2]
                packet = FramePacket(
                    ts_ms=int(time.time() * 1000),
                    camera_id=self.camera_id,
                    frame_id=emitted_frame_id,
                    raw_frame_index=raw_frame_index,
                    frame_bgr=frame,
                    width=width,
                    height=height,
                    source_fps=self.target_fps,
                    video_path=str(video_file),
                )

                was_full = self.output_queue.full()
                put_latest(self.output_queue, packet)
                if was_full:
                    self._stats["queue_drops"] += 1

                self._stats["frames_emitted"] += 1
                self._stats["last_emit_ts_ms"] = packet.ts_ms
                emitted_frame_id += 1
        except Exception as exc:  # pragma: no cover - runtime failure path
            self._set_error(str(exc))
        finally:
            cap.release()
            self._set_connected(False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simulate a live camera stream from an MP4 file.")
    parser.add_argument("video_path", type=str, help="Path to an MP4 file.")
    parser.add_argument("--target-fps", type=float, default=12.0, help="Frames per second to emit.")
    parser.add_argument("--queue-size", type=int, default=16, help="Max size of the output queue.")
    args = parser.parse_args()

    q: queue.Queue = queue.Queue(maxsize=args.queue_size)
    source = MP4CameraSource(args.video_path, q, target_fps=args.target_fps)
    source.start()

    try:
        while True:
            packet: FramePacket = q.get(timeout=1.0)
            print(packet.to_meta_dict())
    except KeyboardInterrupt:
        pass
    finally:
        source.stop()
