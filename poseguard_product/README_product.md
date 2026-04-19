# poseguard_product

This folder contains the **product-facing Streamlit UI** for PoseGuard.

## Purpose
The Streamlit app presents PoseGuard from a user / caregiver perspective.

It supports two views:
- **Product Console** — original local CSV-based workflow
- **Live Backend** — REST-connected view of the FastAPI backend

## Key files
- `app.py` — main Streamlit entry point
- `pages/2_Live_Backend.py` — backend-connected page
- `backend_client.py` — REST client for FastAPI integration
- `live_backend_panel.py` — UI rendering helpers for live backend state and events
- `infer.py` / `postprocess.py` — local/offline inference pipeline
- `poseguard_core.py` — shared schema and keypoint constants
- `assets/style.css` — custom product styling

## Run
```bash
streamlit run app.py
```

## Notes
If you want the **Product Console** page to auto-detect the baseline bundle, place `baseline_bundle.joblib` inside this folder or use the custom path input.

The **Live Backend** page expects the FastAPI backend to be running on `http://127.0.0.1:8000` unless you override the base URL in the page controls.
