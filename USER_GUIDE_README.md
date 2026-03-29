# PoseGuard

**PoseGuard** is a privacy-preserving, edge-first fall detection system designed for older adults living independently.  
Instead of storing or transmitting raw video, PoseGuard converts motion into **skeleton keypoints** and performs fall inference locally. This architecture reduces privacy risk while still supporting real-time fall monitoring, event confirmation, and caregiver-facing alert review.

---

## Project Overview

PoseGuard was built as a capstone project to answer a practical question:

> How can a fall detection system be useful in a home environment **without turning the home into a surveillance system**?

The core design choice is simple:

- **Do not store raw video by default**
- **Perform inference locally on-device**
- **Use skeleton coordinates instead of identifiable imagery**
- **Separate technical monitoring from the caregiver-facing product UI**

This gives PoseGuard two major advantages:

### Privacy-preserving by design
Raw camera frames are used only as transient input inside the local processing pipeline. The system performs inference on **pose keypoints** and exposes **events, probabilities, and state transitions**, not stored video footage.

### Edge-first system behavior
The real-time backend runs locally and does not depend on a cloud inference service. This reduces latency, improves resilience, and makes the system easier to deploy in privacy-sensitive home settings.

---

## System Architecture

PoseGuard is organized into three major subsystems.

### 1) Offline Training
The training pipeline prepares the baseline fall detector from skeleton keypoint CSV files.

- Script: `train_fallvision_baseline_tqdm.py`
- Model family: **Random Forest** baseline
- Output artifacts stored in `Output/`
- Main deployable artifact: `baseline_bundle.joblib`

This stage is used to train and freeze the model once. The bundled artifact is then reused by the real-time backend and the local product console.

### 2) Real-time Backend
The backend is implemented in **FastAPI** and behaves like an edge event-processing system.

Core responsibilities:
- Read an MP4 file as a **simulated live stream**
- Throttle output FPS and push frames through a **bounded queue**
- Extract pose landmarks with **MediaPipe**
- Build **sliding windows** over the live pose stream
- Reuse the same feature logic as the offline CSV-based pipeline
- Run **Random Forest inference** from `baseline_bundle.joblib`
- Manage event lifecycle with an **Event State Machine**
- Persist confirmed events to **SQLite**
- Emit alerts through a **mock alert service**
- Expose REST APIs and WebSocket updates
- Serve a **Reviewer Dashboard** for technical validation

### 3) Product UI
The product-facing interface is built with **Streamlit**.

It serves a different audience from the reviewer dashboard:
- end users
- family members
- caregivers
- instructors reviewing the product flow

The Streamlit app supports:
- local/offline product console workflows
- backend-connected live status view
- confirmed fall event list
- ACK action for acknowledging alerts through the FastAPI API

### Dual-view design
PoseGuard intentionally provides **two synchronized perspectives**:

#### Reviewer Dashboard
A lightweight HTML/JavaScript dashboard served by FastAPI for:
- live skeleton view
- fall probability tracking
- backend state visibility
- confirmed event review

#### Product UI
A calmer, caregiver-style Streamlit interface for:
- system status
- confirmed events
- acknowledgement workflow
- product-style presentation

This separation makes the project stronger both technically and commercially:
- the **reviewer view** proves system integrity
- the **product view** demonstrates usability

---

## Quick Start

### Prerequisites
- Python **3.9+** (3.10 recommended)
- macOS / Linux / Windows with a working Python environment
- A local clone or copy of the PoseGuard project files

### Recommended environment setup
Create one clean virtual environment for the project.

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate  # Windows
```

Install the backend dependencies:

```bash
pip install -r poseguard_backend/requirements.txt
```

Install the product UI dependencies:

```bash
pip install -r poseguard_product/requirements.txt
```

> If your environment currently uses NumPy 2.x and you encounter compatibility warnings from older packages such as `numexpr` or `bottleneck`, use a clean environment and pin `numpy<2` for the demo run.

---

## Quick Start: Run the Full System

PoseGuard is easiest to demo using **two terminals**.

### Terminal A — Start the FastAPI backend

```bash
cd poseguard_backend

python main.py   "../sample_data/Fall/Bed/B_D_0191.mp4"   --baseline-bundle "../Output/baseline_bundle.joblib"   --target-fps 12   --threshold 0.50   --events-db "./data/poseguard_events.db"   --alert-log "./data/poseguard_alerts.jsonl"
```

Useful backend URLs:

- FastAPI docs: `http://127.0.0.1:8000/docs`
- Reviewer dashboard: `http://127.0.0.1:8000/reviewer/`
- Status endpoint: `http://127.0.0.1:8000/api/v1/status`
- Events endpoint: `http://127.0.0.1:8000/api/v1/events`

### Terminal B — Start the Streamlit product UI

```bash
cd poseguard_product
streamlit run app.py
```

Then open the Streamlit URL shown in the terminal, usually:

- `http://localhost:8501`

### Demo notes

#### Fall test
A fall demo clip is already available:

- `sample_data/Fall/Bed/B_D_0191.mp4`

#### No-fall test
A no-fall validation clip is also included:

- `sample_data/No Fall/Bed/B_N_57.mp4`

You can swap the backend input path to validate false-positive suppression.

---

## Product Console vs Live Backend Page

The Streamlit app supports **two different workflows**.

### Product Console
This page supports the original local CSV-based product workflow:
- load a baseline bundle
- choose a keypoint CSV
- run local inference
- review events and exports

**Important:** because `baseline_bundle.joblib` is stored in `Output/`, you should either:
1. enter the custom path manually in the app, or
2. copy `baseline_bundle.joblib` into `poseguard_product/` if you want auto-discovery.

### Live Backend
This page talks directly to the FastAPI backend through REST APIs:
- reads `/api/v1/status`
- reads `/api/v1/events`
- sends ACK requests to `/api/v1/events/{event_id}/ack`

This page is recommended for the final integrated demo.

---

## Directory Structure

```text
PoseGuard/
├── Fall_Dataset/
├── Output/
│   ├── baseline_bundle.joblib
│   ├── baseline_manifest.json
│   ├── features.csv
│   ├── logreg_model.joblib
│   ├── metrics.json
│   ├── rf_feature_importance.csv
│   ├── rf_model.joblib
│   └── window_metadata.csv
├── poseguard_backend/
│   ├── alert_service.py
│   ├── backend_api.py
│   ├── camera_source.py
│   ├── data/
│   ├── event_manager.py
│   ├── feature_service.py
│   ├── inference_worker.py
│   ├── main.py
│   ├── pose_extractor.py
│   ├── reviewer_dashboard/
│   │   ├── app.js
│   │   ├── index.html
│   │   └── styles.css
│   └── requirements.txt
├── poseguard_product/
│   ├── app.py
│   ├── assets/
│   │   └── style.css
│   ├── backend_client.py
│   ├── infer.py
│   ├── live_backend_panel.py
│   ├── pages/
│   │   └── 2_Live_Backend.py
│   ├── poseguard_core.py
│   ├── postprocess.py
│   └── requirements.txt
├── predictions/
├── sample_data/
│   ├── Fall/
│   └── No Fall/
└── train_fallvision_baseline_tqdm.py
```

### Important folders and files

#### `train_fallvision_baseline_tqdm.py`
Offline training entry point. Trains the baseline Random Forest model from skeleton CSV sequences.

#### `Output/`
Stores frozen model artifacts and training outputs.
Most important file:
- `baseline_bundle.joblib` — deployable model bundle used for inference

#### `poseguard_backend/`
Real-time backend services.
Key files:
- `main.py` — startup entry point
- `camera_source.py` — MP4-as-live-stream capture service
- `pose_extractor.py` — MediaPipe pose extraction into `PoseFrame`
- `feature_service.py` — sliding-window feature bridge
- `inference_worker.py` — live Random Forest inference
- `event_manager.py` — state machine and cooldown logic
- `alert_service.py` — mock alert logging
- `backend_api.py` — FastAPI routes, status, events, WebSocket, static dashboard
- `reviewer_dashboard/` — browser-based technical monitoring UI

#### `poseguard_product/`
Product-facing UI built in Streamlit.
Key files:
- `app.py` — main product console
- `pages/2_Live_Backend.py` — backend-connected live page
- `backend_client.py` — REST bridge to FastAPI
- `live_backend_panel.py` — UI components for live backend monitoring
- `infer.py` / `postprocess.py` — original local analysis path
- `assets/style.css` — PoseGuard visual theme

#### `sample_data/`
Small demo inputs used to validate the system without a physical camera.
Includes:
- fall example clip(s)
- no-fall example clip(s)
- sample keypoint CSV files

#### `predictions/`
Stores generated local inference outputs such as:
- `*_prediction.json`
- `*_windows.csv`
- `*_frames.csv`

#### `Fall_Dataset/`
The larger dataset workspace used during model development and experimentation.

---

## API Summary

Core backend endpoints:

- `GET /health/live`
- `GET /health/ready`
- `GET /api/v1/status`
- `GET /api/v1/events`
- `POST /api/v1/events/{event_id}/ack`
- `POST /api/v1/events/{event_id}/dismiss`
- `WS /ws/status`

---

## Business Value

PoseGuard is more than a model demo.

It demonstrates a realistic product direction for privacy-sensitive in-home monitoring:

- **Privacy-preserving**: no raw video storage by default
- **Edge-first**: local processing and reduced latency
- **Explainable event flow**: probability → event state → confirmation → ACK
- **Dual-view design**: technical reviewer dashboard + caregiver-facing product UI
- **Commercially relevant**: suitable for independent living, home monitoring, and assisted-care scenarios where privacy and trust are critical