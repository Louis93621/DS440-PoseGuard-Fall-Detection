# sample_data

This folder contains **small demo inputs** used to validate PoseGuard without a physical camera.

## Purpose
These files let the backend simulate real-time operation and help validate:
- fall detection
- no-fall suppression
- reviewer dashboard behavior
- end-to-end UI integration

## Typical files
- `Fall/.../*.mp4` — fall demo clips
- `No Fall/.../*.mp4` — no-fall validation clips
- sample keypoint CSV files for offline/local testing

## Recommended demo clips
- Fall example: `Fall/Bed/B_D_0191.mp4`
- No-fall example: `No Fall/Bed/B_N_57.mp4`
