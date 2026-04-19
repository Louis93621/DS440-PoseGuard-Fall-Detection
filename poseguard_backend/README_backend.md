# poseguard_backend

This folder contains the **real-time backend** for PoseGuard.

## Purpose
The backend simulates a live fall detection service using:
- MP4 input as a live stream source
- OpenCV frame capture with FPS throttling
- MediaPipe pose extraction
- sliding-window inference
- event-state management
- SQLite persistence
- mock alerts
- FastAPI APIs and WebSocket broadcasting

## Key files
- `main.py` — bootstraps the backend
- `camera_source.py` — capture service and bounded queue input
- `pose_extractor.py` — pose extraction into `PoseFrame`
- `feature_service.py` — converts live pose windows into model-ready features
- `inference_worker.py` — loads `baseline_bundle.joblib` and predicts fall probability
- `event_manager.py` — state machine (`NORMAL -> SUSPECTED -> VERIFYING -> CONFIRMED`) and cooldown
- `alert_service.py` — terminal + JSON mock notifier
- `backend_api.py` — REST APIs, WebSocket, status bridge, reviewer dashboard mount
- `reviewer_dashboard/` — static frontend served by FastAPI

## Run

### Fall
```bash
python main.py "../sample_data/Fall/Bed/B_D_0001.mp4"   --baseline-bundle "../Output/baseline_bundle.joblib"   --target-fps 12   --threshold 0.50   --events-db "./data/poseguard_events.db"   --alert-log "./data/poseguard_alerts.jsonl"
```
### No Fall
```bash
python main.py "../sample_data/No Fall/Bed/B_N_60.mp4"   --baseline-bundle "../Output/baseline_bundle.joblib"   --target-fps 12   --threshold 0.50   --events-db "./data/poseguard_events.db"   --alert-log "./data/poseguard_alerts.jsonl"
```

## Review URLs
- Docs: `http://127.0.0.1:8000/docs`
- Reviewer dashboard: `http://127.0.0.1:8000/reviewer/`
- Status: `http://127.0.0.1:8000/api/v1/status`
- Events: `http://127.0.0.1:8000/api/v1/events`
