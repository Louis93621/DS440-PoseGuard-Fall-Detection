from __future__ import annotations

import queue
import threading
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import cv2

from camera_source import FramePacket, put_latest

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - import checked at runtime on the user's machine
    mp = None


COCO_17_ORDER = [
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

# MediaPipe Pose (33 landmarks) -> COCO 17 mapping.
MP_TO_COCO_INDEX = {
    "nose": 0,
    "left_eye": 2,
    "right_eye": 5,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


@dataclass
class Keypoint:
    name: str
    x: Optional[float]
    y: Optional[float]
    conf: float


@dataclass
class PoseFrame:
    ts_ms: int
    camera_id: str
    frame_id: int
    person_id: int
    bbox: Optional[List[float]]
    keypoints: List[Keypoint]
    pose_conf: float
    source_fps: float
    frame_width: int
    frame_height: int
    model_name: str
    detected: bool
    quality: str
    raw_frame_index: int
    video_path: str

    def to_dict(self) -> dict:
        return asdict(self)


class BasePoseExtractor(ABC):
    """Interface for pluggable pose extractors.

    Future model swaps, including a YOLOv10-based implementation, only need to
    implement this interface.
    """

    model_name = "base_pose_extractor"

    @abstractmethod
    def extract(self, frame_packet: FramePacket) -> PoseFrame:
        raise NotImplementedError

    def close(self) -> None:
        return None


class MediaPipePoseExtractor(BasePoseExtractor):
    """Stable CPU-friendly pose extraction using MediaPipe Pose.

    This is the simplest way to stand up a local prototype now, while keeping
    the interface replaceable later.
    """

    model_name = "mediapipe_pose"

    def __init__(
        self,
        *,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        visibility_threshold: float = 0.15,
    ) -> None:
        if mp is None:
            raise ImportError(
                "mediapipe is not installed. Install it with `pip install mediapipe`."
            )
        self.visibility_threshold = float(visibility_threshold)
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self) -> None:
        self._pose.close()

    def extract(self, frame_packet: FramePacket) -> PoseFrame:
        rgb = cv2.cvtColor(frame_packet.frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)

        if not result.pose_landmarks:
            return PoseFrame(
                ts_ms=frame_packet.ts_ms,
                camera_id=frame_packet.camera_id,
                frame_id=frame_packet.frame_id,
                person_id=0,
                bbox=None,
                keypoints=[Keypoint(name=name, x=None, y=None, conf=0.0) for name in COCO_17_ORDER],
                pose_conf=0.0,
                source_fps=frame_packet.source_fps,
                frame_width=frame_packet.width,
                frame_height=frame_packet.height,
                model_name=self.model_name,
                detected=False,
                quality="no_person",
                raw_frame_index=frame_packet.raw_frame_index,
                video_path=frame_packet.video_path,
            )

        landmarks = result.pose_landmarks.landmark
        keypoints: List[Keypoint] = []
        valid_xy: List[tuple] = []
        confs: List[float] = []

        for keypoint_name in COCO_17_ORDER:
            mp_idx = MP_TO_COCO_INDEX[keypoint_name]
            lm = landmarks[mp_idx]
            conf = float(getattr(lm, "visibility", 1.0))
            x_px = float(max(0.0, min(1.0, lm.x)) * frame_packet.width)
            y_px = float(max(0.0, min(1.0, lm.y)) * frame_packet.height)
            keypoints.append(Keypoint(name=keypoint_name, x=x_px, y=y_px, conf=conf))
            confs.append(conf)
            if conf >= self.visibility_threshold:
                valid_xy.append((x_px, y_px))

        if valid_xy:
            xs = [xy[0] for xy in valid_xy]
            ys = [xy[1] for xy in valid_xy]
            bbox = [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
        else:
            bbox = None

        pose_conf = float(sum(confs) / max(len(confs), 1))
        quality = "ok" if pose_conf >= 0.5 else "low_conf"

        return PoseFrame(
            ts_ms=frame_packet.ts_ms,
            camera_id=frame_packet.camera_id,
            frame_id=frame_packet.frame_id,
            person_id=0,
            bbox=bbox,
            keypoints=keypoints,
            pose_conf=pose_conf,
            source_fps=frame_packet.source_fps,
            frame_width=frame_packet.width,
            frame_height=frame_packet.height,
            model_name=self.model_name,
            detected=True,
            quality=quality,
            raw_frame_index=frame_packet.raw_frame_index,
            video_path=frame_packet.video_path,
        )


class PoseExtractorWorker:
    """Continuously convert incoming frames into PoseFrame payloads."""

    def __init__(
        self,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        *,
        extractor: Optional[BasePoseExtractor] = None,
        poll_timeout: float = 0.2,
    ) -> None:
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.extractor = extractor or MediaPipePoseExtractor()
        self.poll_timeout = poll_timeout

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_error: Optional[str] = None
        self._stats = {
            "frames_consumed": 0,
            "pose_frames_emitted": 0,
            "detections": 0,
            "no_person_frames": 0,
            "queue_drops": 0,
        }

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="PoseExtractorWorker",
            daemon=True,
        )
        self._thread.start()

    def stop(self, join_timeout: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=join_timeout)
        self.extractor.close()

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    @property
    def last_error(self) -> Optional[str]:
        with self._lock:
            return self._last_error

    def get_status(self) -> dict:
        with self._lock:
            return {
                "alive": self.is_alive(),
                "model_name": getattr(self.extractor, "model_name", "unknown"),
                "last_error": self._last_error,
                **self._stats,
            }

    def _set_error(self, message: Optional[str]) -> None:
        with self._lock:
            self._last_error = message

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame_packet: FramePacket = self.input_queue.get(timeout=self.poll_timeout)
            except queue.Empty:
                continue

            try:
                pose_frame = self.extractor.extract(frame_packet)
                was_full = self.output_queue.full()
                put_latest(self.output_queue, pose_frame)
                if was_full:
                    self._stats["queue_drops"] += 1

                self._stats["frames_consumed"] += 1
                self._stats["pose_frames_emitted"] += 1
                if pose_frame.detected:
                    self._stats["detections"] += 1
                else:
                    self._stats["no_person_frames"] += 1
            except Exception as exc:  # pragma: no cover - runtime failure path
                self._set_error(str(exc))


if __name__ == "__main__":
    import argparse
    import time

    from camera_source import MP4CameraSource

    parser = argparse.ArgumentParser(description="Run the simulated camera + pose extractor pipeline.")
    parser.add_argument("video_path", type=str, help="Path to an MP4 file.")
    parser.add_argument("--target-fps", type=float, default=12.0)
    args = parser.parse_args()

    frame_q: queue.Queue = queue.Queue(maxsize=16)
    pose_q: queue.Queue = queue.Queue(maxsize=32)

    camera = MP4CameraSource(args.video_path, frame_q, target_fps=args.target_fps)
    worker = PoseExtractorWorker(frame_q, pose_q)
    camera.start()
    worker.start()

    try:
        while True:
            pose_frame: PoseFrame = pose_q.get(timeout=1.0)
            print(pose_frame.to_dict())
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        worker.stop()
