# PoseGuard Streamlit Dashboard

A premium, minimalist Streamlit dashboard for the PoseGuard capstone MVP.

## What this dashboard does

- Loads your `baseline_bundle.joblib` or baseline output folder.
- Runs inference directly on a single keypoint CSV.
- Applies temporal post-processing to stabilize alerts.
- Shows a skeleton-only playback view with no raw video.
- Visualizes window-level probabilities and event-level alert segments.
- Exports `prediction.json`, `windows.csv`, and `frames.csv` from the UI.

## Folder contents

- `app.py` — main Streamlit application
- `infer.py` — inference entry point used by the dashboard
- `postprocess.py` — alert smoothing and event construction
- `poseguard_core.py` — pose parsing and feature engineering utilities
- `freeze_baseline.py` — helper to package your best model if needed
- `assets/style.css` — premium UI styling
- `.streamlit/config.toml` — cohesive theme defaults

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Inputs you need

You need one of the following:

1. `baseline_bundle.joblib`
2. or a training output directory containing:
   - `metrics.json`
   - `rf_model.joblib` (or whichever model is marked as `best_model`)

For the sequence source, use either:
- a dataset root so the app can browse all `*_keypoints.csv` files,
- a direct CSV path,
- or a single uploaded CSV.

## Recommended demo flow

1. Open the app.
2. Paste the path to `baseline_bundle.joblib`.
3. Choose **Browse dataset**.
4. Point **Dataset root** to your `Fall_Dataset` directory.
5. Select a sequence and click **Run analysis**.
6. Use the tabs to inspect the timeline, skeleton playback, and event log.
7. Download the outputs from the **Exports** tab for your report.

## Notes

- The dashboard never requires raw video.
- Playback is reconstructed from skeletal coordinates only.
- All inference runs locally in the Streamlit session.
- The palette and spacing are tuned for a clean, premium, low-noise presentation.
