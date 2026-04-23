# Face Tracking Lab

**Lynn University × Universitetit të Tiranës (Informatica) — April 2026**

A joint cybersecurity demonstration lab exploring real-time face detection,
PID-based tracking simulation, and ethical/legal analysis of biometric AI systems.

---

## Prerequisites

- **Python 3.11 or 3.12 only.**  
  Python 3.13+ is **not supported** — MediaPipe wheels are not yet available for it.
- A webcam (built-in or USB). Scripts try index `0` then index `1` automatically.

---

## Setup

All runnable code lives in the `face_lab/` directory.

```bash
# 1. Enter the project directory
cd face_lab

# 2. Create and activate a virtual environment
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Project Structure

```
face_lab/
├── webcam_test.py       # Session 1 — webcam stream with live FPS and HH:MM:SS overlay
├── face_detect.py       # Session 1 — Haar + MediaPipe face detection & benchmarking
├── pid_tracker.py       # Session 2 — MockTello PID tracker + CSV logging
├── analysis_plot.py     # Session 2 — 3-panel time-series plot from rc_log.csv
├── requirements.txt     # pip dependencies
├── models/              # Auto-downloaded MediaPipe .tflite model (first run only)
├── outputs/             # tracking_analysis.png is saved here automatically
└── docs/                # Per-member notes, checklists, and ethics templates
```

---

## Running the Scripts

All commands below should be run from inside the `face_lab/` directory with the virtual environment activated.

### 1 — Webcam Test

```bash
python webcam_test.py
```

Verify: green FPS counter and gold `HH:MM:SS` timestamp appear in the top-left corner.  
Press **Q** to quit.

---

### 2 — Face Detection & Benchmark

```bash
# Run both backends for 30 seconds each
python face_detect.py --backend both --camera-index 0 --benchmark-seconds 30

# Haar only
python face_detect.py --backend haar --benchmark-seconds 30

# MediaPipe only (downloads ~5 MB model on first run)
python face_detect.py --backend mediapipe --benchmark-seconds 30
```

---

### 3 — PID Tracker

```bash
python pid_tracker.py --camera-index 0 --output rc_log.csv --kp 0.4 --kd 0.1
```

Expected output:

```
[SIM] Connected.
[SIM] Takeoff.
[t= 0.03s] LR:  +0 FB: +15 UD:  +0 YAW: +12
...
```

Press **Q** to stop. `rc_log.csv` is written to the current directory.

---

### 4 — Analysis Plot

```bash
python analysis_plot.py --input rc_log.csv --output outputs/tracking_analysis.png
```

Opens a 3-panel matplotlib figure and saves it to `outputs/tracking_analysis.png`.

---

## Webcam Troubleshooting

| Issue | Fix |
|---|---|
| Black window or immediate crash | Try `--camera-index 1` |
| Low FPS on Windows | Scripts already use `cv2.CAP_DSHOW` automatically |
| `/dev/video*` permission denied (Linux) | `sudo usermod -aG video $USER` then log out/in |
| MediaPipe model download fails | Place `blaze_face_short_range.tflite` manually in `face_lab/models/` |

---

## Auto-Generated Files

| File | Created by |
|---|---|
| `rc_log.csv` | `pid_tracker.py` — overwritten each session |
| `outputs/tracking_analysis.png` | `analysis_plot.py` |
| `models/blaze_face_short_range.tflite` | `face_detect.py` / `pid_tracker.py` (first MediaPipe run) |

---

## AI Use Disclosure

> AI Use: We used GitHub Copilot to assist with code scaffolding and documentation.  
> All analysis, PID tuning experiments, and ethics conclusions are our own.
