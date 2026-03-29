# PoseGuard Phase 5 — Reviewer Dashboard

This package adds a lightweight reviewer-facing dashboard to the Phase 4 backend.

## Included
- `reviewer_dashboard/index.html`
- `reviewer_dashboard/styles.css`
- `reviewer_dashboard/app.js`
- updated `backend_api.py`
- updated `main.py`

## What the dashboard does
- Connects directly to `ws://127.0.0.1:8000/ws/status`
- Draws the latest skeleton on an HTML canvas from `pose_frame`
- Displays live fall probability and current event-manager state
- Fetches persisted confirmed events from `/api/v1/events`
- Sends ACK requests to `/api/v1/events/{event_id}/ack`

## How to wire it in
Replace your Phase 4 `backend_api.py` and `main.py` with these updated versions, then place the `reviewer_dashboard/` folder next to `main.py`.

Expected layout:

```text
poseguard_backend_phase4/
  main.py
  backend_api.py
  camera_source.py
  pose_extractor.py
  inference_worker.py
  feature_service.py
  event_manager.py
  alert_service.py
  reviewer_dashboard/
    index.html
    styles.css
    app.js
```

## Run

```bash
python main.py \
  "/absolute/path/to/sample.mp4" \
  --baseline-bundle "/absolute/path/to/baseline_bundle.joblib" \
  --target-fps 12 \
  --threshold 0.50 \
  --events-db "./data/poseguard_events.db" \
  --alert-log "./data/poseguard_alerts.jsonl"
```

## Open
- API docs: `http://127.0.0.1:8000/docs`
- reviewer dashboard: `http://127.0.0.1:8000/reviewer/`
- shortcut redirect: `http://127.0.0.1:8000/dashboard`

## Notes
- Use a fresh SQLite file and a fresh alert log when you want isolated test runs.
- For no-fall tests, keep the dashboard open and confirm that:
  - probability remains low overall,
  - the event manager stays in `NORMAL`,
  - `/api/v1/events` stays empty for that run.
