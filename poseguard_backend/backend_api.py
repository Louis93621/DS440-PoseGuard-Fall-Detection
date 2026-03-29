from __future__ import annotations

import asyncio
import queue
import threading
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Set

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from alert_service import AlertService
from camera_source import put_latest
from event_manager import EventManager


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)

    async def broadcast(self, payload: Dict[str, Any]) -> None:
        dead_connections = []
        async with self._lock:
            for websocket in list(self._connections):
                try:
                    await websocket.send_json(payload)
                except Exception:
                    dead_connections.append(websocket)
            for websocket in dead_connections:
                self._connections.discard(websocket)

    async def count(self) -> int:
        async with self._lock:
            return len(self._connections)


class BackendState:
    def __init__(
        self,
        *,
        camera_source: Optional[Any] = None,
        pose_worker: Optional[Any] = None,
        inference_worker: Optional[Any] = None,
        event_manager: Optional[EventManager] = None,
        alert_service: Optional[AlertService] = None,
        input_frame_queue: Optional[queue.Queue] = None,
        output_pose_queue: Optional[queue.Queue] = None,
        inference_input_queue: Optional[queue.Queue] = None,
        inference_output_queue: Optional[queue.Queue] = None,
    ) -> None:
        self.camera_source = camera_source
        self.pose_worker = pose_worker
        self.inference_worker = inference_worker
        self.event_manager = event_manager
        self.alert_service = alert_service
        self.input_frame_queue = input_frame_queue
        self.output_pose_queue = output_pose_queue
        self.inference_input_queue = inference_input_queue
        self.inference_output_queue = inference_output_queue

        self._lock = threading.Lock()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.latest_pose: Optional[Dict[str, Any]] = None
        self.latest_inference: Optional[Dict[str, Any]] = None
        self.latest_event: Optional[Dict[str, Any]] = None
        self.last_pose_received_at_ms: Optional[int] = None
        self.last_inference_received_at_ms: Optional[int] = None
        self.last_event_received_at_ms: Optional[int] = None
        self.last_error: Optional[str] = None

    def _coerce_payload(self, payload: Any) -> Dict[str, Any]:
        if is_dataclass(payload):
            return asdict(payload)
        if hasattr(payload, "to_dict"):
            return payload.to_dict()
        if isinstance(payload, dict):
            return payload
        raise TypeError(f"Unsupported payload type: {type(payload)!r}")

    def update_pose(self, pose_frame: Any) -> Dict[str, Any]:
        pose_payload = self._coerce_payload(pose_frame)
        with self._lock:
            self.latest_pose = pose_payload
            self.last_pose_received_at_ms = int(time.time() * 1000)
        return pose_payload

    def update_inference(self, inference_result: Any) -> Dict[str, Any]:
        inference_payload = self._coerce_payload(inference_result)
        with self._lock:
            self.latest_inference = inference_payload
            self.last_inference_received_at_ms = int(time.time() * 1000)
        return inference_payload

    def update_event(self, event_payload: Any) -> Dict[str, Any]:
        event_payload = self._coerce_payload(event_payload)
        with self._lock:
            self.latest_event = event_payload
            self.last_event_received_at_ms = int(time.time() * 1000)
        return event_payload

    def set_error(self, message: Optional[str]) -> None:
        with self._lock:
            self.last_error = message

    def ready(self) -> bool:
        with self._lock:
            last_pose_received_at_ms = self.last_pose_received_at_ms

        recent_pose = (
            last_pose_received_at_ms is not None
            and (int(time.time() * 1000) - last_pose_received_at_ms) <= 5000
        )
        camera_ok = bool(self.camera_source and self.camera_source.connected and self.camera_source.is_alive())
        pose_ok = bool(self.pose_worker and self.pose_worker.is_alive())
        inference_ok = True if self.inference_worker is None else bool(self.inference_worker.is_alive())
        alert_ok = True if self.alert_service is None else bool(self.alert_service.is_alive())
        return recent_pose and camera_ok and pose_ok and inference_ok and alert_ok

    def inference_ready(self) -> bool:
        with self._lock:
            last_inference_received_at_ms = self.last_inference_received_at_ms
        if self.inference_worker is None:
            return False
        return (
            last_inference_received_at_ms is not None
            and (int(time.time() * 1000) - last_inference_received_at_ms) <= 10000
            and self.inference_worker.is_alive()
        )

    def event_ready(self) -> bool:
        return bool(self.event_manager is not None)

    def _worker_snapshot(self) -> Dict[str, Any]:
        camera_status = self.camera_source.get_status() if self.camera_source else {}
        pose_status = self.pose_worker.get_status() if self.pose_worker else {}
        inference_status = self.inference_worker.get_status() if self.inference_worker else {}
        event_status = self.event_manager.get_status() if self.event_manager else {}
        alert_status = self.alert_service.get_status() if self.alert_service else {}

        return {
            "camera": camera_status,
            "pose_worker": pose_status,
            "inference_worker": inference_status,
            "event_manager": event_status,
            "alert_service": alert_status,
            "frame_queue_size": self.input_frame_queue.qsize() if self.input_frame_queue is not None else None,
            "pose_queue_size": self.output_pose_queue.qsize() if self.output_pose_queue is not None else None,
            "inference_input_queue_size": self.inference_input_queue.qsize() if self.inference_input_queue is not None else None,
            "inference_output_queue_size": self.inference_output_queue.qsize() if self.inference_output_queue is not None else None,
        }

    def live_payload(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "started_at": self.started_at,
            "ready": self.ready(),
            "inference_ready": self.inference_ready(),
            "event_ready": self.event_ready(),
            "last_error": self.last_error,
            **self._worker_snapshot(),
        }

    def ready_payload(self) -> Dict[str, Any]:
        with self._lock:
            latest_pose = self.latest_pose
            latest_inference = self.latest_inference
            latest_event = self.latest_event
            last_pose_received_at_ms = self.last_pose_received_at_ms
            last_inference_received_at_ms = self.last_inference_received_at_ms
            last_event_received_at_ms = self.last_event_received_at_ms

        return {
            "ready": self.ready(),
            "inference_ready": self.inference_ready(),
            "event_ready": self.event_ready(),
            "started_at": self.started_at,
            "last_pose_received_at_ms": last_pose_received_at_ms,
            "last_inference_received_at_ms": last_inference_received_at_ms,
            "last_event_received_at_ms": last_event_received_at_ms,
            "latest_pose_frame_id": latest_pose.get("frame_id") if latest_pose else None,
            "latest_window_index": latest_inference.get("window_index") if latest_inference else None,
            "latest_event_id": latest_event.get("event_id") if latest_event else None,
            "last_error": self.last_error,
            **self._worker_snapshot(),
        }

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            latest_pose = self.latest_pose
            latest_inference = self.latest_inference
            latest_event = self.latest_event
            last_pose_received_at_ms = self.last_pose_received_at_ms
            last_inference_received_at_ms = self.last_inference_received_at_ms
            last_event_received_at_ms = self.last_event_received_at_ms

        return {
            "started_at": self.started_at,
            "ready": self.ready(),
            "inference_ready": self.inference_ready(),
            "event_ready": self.event_ready(),
            "last_pose_received_at_ms": last_pose_received_at_ms,
            "last_inference_received_at_ms": last_inference_received_at_ms,
            "last_event_received_at_ms": last_event_received_at_ms,
            "last_error": self.last_error,
            "latest_pose": latest_pose,
            "latest_inference": latest_inference,
            "latest_event": latest_event,
            **self._worker_snapshot(),
        }


async def _pose_queue_bridge(app: FastAPI) -> None:
    state: BackendState = app.state.backend_state
    pose_output_queue: queue.Queue = app.state.pose_output_queue
    inference_input_queue: Optional[queue.Queue] = getattr(app.state, "inference_input_queue", None)
    ws_manager: ConnectionManager = app.state.ws_manager

    last_heartbeat_at = 0.0
    while True:
        try:
            pose_frame = await asyncio.to_thread(pose_output_queue.get, True, 0.5)
            payload = state.update_pose(pose_frame)
            if inference_input_queue is not None:
                put_latest(inference_input_queue, pose_frame)

            await ws_manager.broadcast(
                {
                    "type": "pose_frame",
                    "ts_ms": int(time.time() * 1000),
                    "data": payload,
                }
            )
        except queue.Empty:
            now = time.time()
            if now - last_heartbeat_at >= 1.0:
                last_heartbeat_at = now
                await ws_manager.broadcast(
                    {
                        "type": "heartbeat",
                        "ts_ms": int(now * 1000),
                        "data": state.snapshot(),
                    }
                )
            await asyncio.sleep(0.05)
        except asyncio.CancelledError:  # pragma: no cover
            raise
        except Exception as exc:  # pragma: no cover
            state.set_error(str(exc))
            await ws_manager.broadcast(
                {
                    "type": "error",
                    "ts_ms": int(time.time() * 1000),
                    "data": {"message": str(exc), "stage": "pose_bridge"},
                }
            )
            await asyncio.sleep(0.2)


async def _inference_queue_bridge(app: FastAPI) -> None:
    state: BackendState = app.state.backend_state
    inference_output_queue: Optional[queue.Queue] = getattr(app.state, "inference_output_queue", None)
    event_manager: Optional[EventManager] = getattr(app.state, "event_manager", None)
    alert_service: Optional[AlertService] = getattr(app.state, "alert_service", None)
    ws_manager: ConnectionManager = app.state.ws_manager

    if inference_output_queue is None:
        return

    while True:
        try:
            inference_result = await asyncio.to_thread(inference_output_queue.get, True, 0.5)
            payload = state.update_inference(inference_result)
            await ws_manager.broadcast(
                {
                    "type": "inference_result",
                    "ts_ms": int(time.time() * 1000),
                    "data": payload,
                }
            )

            if event_manager is not None:
                transitions = event_manager.process_inference(payload)
                for transition in transitions:
                    transition_payload = transition.to_dict() if hasattr(transition, "to_dict") else dict(transition)
                    event_payload = state.update_event(transition_payload["event"])
                    if transition_payload.get("should_notify") and alert_service is not None:
                        alert_service.enqueue_event(event_payload, transition_payload)
                    await ws_manager.broadcast(
                        {
                            "type": "fall_event",
                            "ts_ms": int(time.time() * 1000),
                            "data": transition_payload,
                        }
                    )
        except queue.Empty:
            await asyncio.sleep(0.05)
        except asyncio.CancelledError:  # pragma: no cover
            raise
        except Exception as exc:  # pragma: no cover
            state.set_error(str(exc))
            await ws_manager.broadcast(
                {
                    "type": "error",
                    "ts_ms": int(time.time() * 1000),
                    "data": {"message": str(exc), "stage": "inference_bridge"},
                }
            )
            await asyncio.sleep(0.2)


def create_app(
    state: BackendState,
    pose_output_queue: queue.Queue,
    *,
    inference_input_queue: Optional[queue.Queue] = None,
    inference_output_queue: Optional[queue.Queue] = None,
    event_manager: Optional[EventManager] = None,
    alert_service: Optional[AlertService] = None,
    reviewer_dashboard_dir: Optional[str] = None,
) -> FastAPI:
    app = FastAPI(
        title="PoseGuard Backend API",
        version="0.5.0",
        description="Edge-first real-time backend with reviewer dashboard for privacy-preserving fall detection.",
    )
    app.state.backend_state = state
    app.state.pose_output_queue = pose_output_queue
    app.state.inference_input_queue = inference_input_queue
    app.state.inference_output_queue = inference_output_queue
    app.state.event_manager = event_manager
    app.state.alert_service = alert_service
    app.state.ws_manager = ConnectionManager()
    app.state.pose_bridge_task = None
    app.state.inference_bridge_task = None

    reviewer_path = Path(reviewer_dashboard_dir).expanduser().resolve() if reviewer_dashboard_dir else None
    if reviewer_path and reviewer_path.exists():
        app.mount("/reviewer", StaticFiles(directory=str(reviewer_path), html=True), name="reviewer")

    @app.on_event("startup")
    async def on_startup() -> None:
        app.state.pose_bridge_task = asyncio.create_task(_pose_queue_bridge(app))
        if inference_output_queue is not None:
            app.state.inference_bridge_task = asyncio.create_task(_inference_queue_bridge(app))

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        for task_name in ["pose_bridge_task", "inference_bridge_task"]:
            task = getattr(app.state, task_name, None)
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    @app.get("/")
    async def root() -> Dict[str, Any]:
        return {
            "service": "PoseGuard Backend API",
            "status": "ok",
            "docs": "/docs",
            "websocket": "/ws/status",
            "reviewer_dashboard": "/reviewer/" if reviewer_path and reviewer_path.exists() else None,
        }

    @app.get("/dashboard")
    async def reviewer_redirect() -> RedirectResponse:
        if not reviewer_path or not reviewer_path.exists():
            raise HTTPException(status_code=404, detail="reviewer dashboard is not mounted")
        return RedirectResponse(url="/reviewer/")

    @app.get("/health/live")
    async def health_live() -> Dict[str, Any]:
        return state.live_payload()

    @app.get("/health/ready")
    async def health_ready() -> JSONResponse:
        payload = state.ready_payload()
        status_code = 200 if payload["ready"] else 503
        return JSONResponse(content=payload, status_code=status_code)

    @app.get("/api/v1/status")
    async def get_status() -> Dict[str, Any]:
        return state.snapshot()

    @app.get("/api/v1/events")
    async def list_events(limit: int = 100) -> Dict[str, Any]:
        if event_manager is None:
            raise HTTPException(status_code=503, detail="event manager is not configured")
        limit = max(1, min(int(limit), 500))
        items = event_manager.list_events(limit=limit)
        return {"count": len(items), "items": items}

    @app.get("/api/v1/events/{event_id}")
    async def get_event(event_id: str) -> Dict[str, Any]:
        if event_manager is None:
            raise HTTPException(status_code=503, detail="event manager is not configured")
        event = event_manager.get_event(event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="event not found")
        return event

    @app.post("/api/v1/events/{event_id}/ack")
    async def acknowledge_event(event_id: str) -> Dict[str, Any]:
        if event_manager is None:
            raise HTTPException(status_code=503, detail="event manager is not configured")
        event = event_manager.acknowledge_event(event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="event not found")
        state.update_event(event)
        await app.state.ws_manager.broadcast(
            {
                "type": "fall_event",
                "ts_ms": int(time.time() * 1000),
                "data": {
                    "transition": "acknowledged",
                    "state": event.get("state", "CONFIRMED"),
                    "reason": "user_acknowledged_event",
                    "ts_ms": int(time.time() * 1000),
                    "event": event,
                    "should_notify": False,
                    "persisted": True,
                    "cooldown_until_ms": state.event_manager.get_status().get("cooldown_until_ms") if state.event_manager else None,
                },
            }
        )
        return event

    @app.post("/api/v1/events/{event_id}/dismiss")
    async def dismiss_event(event_id: str) -> Dict[str, Any]:
        if event_manager is None:
            raise HTTPException(status_code=503, detail="event manager is not configured")
        event = event_manager.dismiss_event(event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="event not found")
        state.update_event(event)
        await app.state.ws_manager.broadcast(
            {
                "type": "fall_event",
                "ts_ms": int(time.time() * 1000),
                "data": {
                    "transition": "dismissed",
                    "state": event.get("state", "CONFIRMED"),
                    "reason": "user_dismissed_event",
                    "ts_ms": int(time.time() * 1000),
                    "event": event,
                    "should_notify": False,
                    "persisted": True,
                    "cooldown_until_ms": state.event_manager.get_status().get("cooldown_until_ms") if state.event_manager else None,
                },
            }
        )
        return event

    @app.websocket("/ws/status")
    async def websocket_status(websocket: WebSocket) -> None:
        manager: ConnectionManager = app.state.ws_manager
        await manager.connect(websocket)
        try:
            await websocket.send_json(
                {
                    "type": "hello",
                    "ts_ms": int(time.time() * 1000),
                    "data": state.snapshot(),
                }
            )
            while True:
                try:
                    message = await asyncio.wait_for(websocket.receive_text(), timeout=15.0)
                    if message.strip().lower() == "ping":
                        await websocket.send_json(
                            {
                                "type": "pong",
                                "ts_ms": int(time.time() * 1000),
                                "data": {
                                    "ready": state.ready(),
                                    "inference_ready": state.inference_ready(),
                                    "event_ready": state.event_ready(),
                                },
                            }
                        )
                except asyncio.TimeoutError:
                    await websocket.send_json(
                        {
                            "type": "heartbeat",
                            "ts_ms": int(time.time() * 1000),
                            "data": state.snapshot(),
                        }
                    )
        except WebSocketDisconnect:
            await manager.disconnect(websocket)
        except Exception:
            await manager.disconnect(websocket)

    return app
