# PoseGuard Product-Hardened Dashboard

This version hardens the Streamlit dashboard for class demo and testing.

## What's new
- step-by-step sidebar workflow
- demo mode for a quick showcase sequence
- system check / preflight validation before inference
- calmer advanced settings with collapsed alert controls
- diagnostics tab for runtime and reproducibility
- user-facing error recovery messages
- session report export
- persistent privacy note

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Expected files
Keep these files in the same folder:
- `app.py`
- `infer.py`
- `postprocess.py`
- `poseguard_core.py`
- `assets/style.css`

Then point the app to:
- `baseline_bundle.joblib`, or
- a training output directory containing `metrics.json`

## Best class demo flow
1. Load your `baseline_bundle.joblib`
2. Choose **Browse dataset**
3. Turn on **Demo mode**
4. Run **System check**
5. Click **Run analysis**
