from __future__ import annotations

import argparse
import queue
from pathlib import Path

import uvicorn

from alert_service import AlertService
from backend_api import BackendState, create_app
from camera_source import MP4CameraSource
from event_manager import EventManager, SQLiteEventStore
from inference_worker import InferenceWorker
from pose_extractor import MediaPipePoseExtractor, PoseExtractorWorker


def build_app(
    video_path: str,
    *,
    target_fps: float = 12.0,
    baseline_bundle_path: str | None = None,
    inference_threshold: float = 0.50,
    events_db_path: str = "./data/poseguard_events.db",
    alert_log_path: str = "./data/poseguard_alerts.jsonl",
    cooldown_ms: int = 15000,
    min_confirm_probability: float = 0.65,
    reviewer_dashboard_dir: str | None = None,
):
    frame_queue: queue.Queue = queue.Queue(maxsize=24)
    pose_queue: queue.Queue = queue.Queue(maxsize=48)
    inference_input_queue: queue.Queue = queue.Queue(maxsize=48)
    inference_output_queue: queue.Queue = queue.Queue(maxsize=48)

    camera = MP4CameraSource(
        video_path=video_path,
        output_queue=frame_queue,
        target_fps=target_fps,
        camera_id="living_room_cam_1",
        loop=True,
    )

    pose_worker = PoseExtractorWorker(
        input_queue=frame_queue,
        output_queue=pose_queue,
        extractor=MediaPipePoseExtractor(),
    )

    inference_worker = None
    event_store = None
    event_manager = None
    alert_service = None

    if baseline_bundle_path is not None:
        inference_worker = InferenceWorker(
            input_queue=inference_input_queue,
            output_queue=inference_output_queue,
            baseline_bundle_path=baseline_bundle_path,
            threshold=inference_threshold,
        )
        event_store = SQLiteEventStore(events_db_path)
        event_manager = EventManager(
            store=event_store,
            threshold=inference_threshold,
            cooldown_ms=cooldown_ms,
            min_confirm_probability=min_confirm_probability,
        )
        alert_service = AlertService(log_path=alert_log_path)

    state = BackendState(
        camera_source=camera,
        pose_worker=pose_worker,
        inference_worker=inference_worker,
        event_manager=event_manager,
        alert_service=alert_service,
        input_frame_queue=frame_queue,
        output_pose_queue=pose_queue,
        inference_input_queue=inference_input_queue,
        inference_output_queue=inference_output_queue if inference_worker else None,
    )

    app = create_app(
        state,
        pose_queue,
        inference_input_queue=inference_input_queue if inference_worker else None,
        inference_output_queue=inference_output_queue if inference_worker else None,
        event_manager=event_manager,
        alert_service=alert_service,
        reviewer_dashboard_dir=reviewer_dashboard_dir,
    )
    return app, camera, pose_worker, inference_worker, event_store, event_manager, alert_service


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the PoseGuard backend with Phase 5 reviewer dashboard support.")
    parser.add_argument("video_path", type=str, help="Path to the MP4 file used to simulate a live camera.")
    parser.add_argument(
        "--baseline-bundle",
        type=str,
        default=None,
        help="Path to baseline_bundle.joblib. If omitted, only pose streaming runs.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--target-fps", type=float, default=12.0)
    parser.add_argument("--threshold", type=float, default=0.50, help="Fall probability threshold.")
    parser.add_argument("--events-db", type=str, default="./data/poseguard_events.db", help="SQLite file for persisted fall events.")
    parser.add_argument("--alert-log", type=str, default="./data/poseguard_alerts.jsonl", help="JSONL log file for the mock notifier.")
    parser.add_argument("--cooldown-ms", type=int, default=15000, help="Cooldown after a confirmed event closes.")
    parser.add_argument("--min-confirm-probability", type=float, default=0.65, help="Minimum peak probability needed to confirm an event.")
    parser.add_argument(
        "--reviewer-dashboard-dir",
        type=str,
        default=None,
        help="Optional path to the reviewer dashboard static files. Defaults to ./reviewer_dashboard next to main.py.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    video_path = str(Path(args.video_path).expanduser())
    baseline_bundle = str(Path(args.baseline_bundle).expanduser()) if args.baseline_bundle else None
    events_db = str(Path(args.events_db).expanduser())
    alert_log = str(Path(args.alert_log).expanduser())
    reviewer_dir = (
        str(Path(args.reviewer_dashboard_dir).expanduser())
        if args.reviewer_dashboard_dir
        else str(base_dir / "reviewer_dashboard")
    )

    app, camera, pose_worker, inference_worker, event_store, event_manager, alert_service = build_app(
        video_path,
        target_fps=args.target_fps,
        baseline_bundle_path=baseline_bundle,
        inference_threshold=args.threshold,
        events_db_path=events_db,
        alert_log_path=alert_log,
        cooldown_ms=args.cooldown_ms,
        min_confirm_probability=args.min_confirm_probability,
        reviewer_dashboard_dir=reviewer_dir,
    )

    camera.start()
    pose_worker.start()
    if inference_worker is not None:
        inference_worker.start()
    if alert_service is not None:
        alert_service.start()

    try:
        uvicorn.run(app, host=args.host, port=args.port, reload=False)
    finally:
        camera.stop()
        pose_worker.stop()
        if inference_worker is not None:
            inference_worker.stop()
        if alert_service is not None:
            alert_service.stop()
        if event_store is not None:
            event_store.close()
