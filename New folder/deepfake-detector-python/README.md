# Deepfake Lip-Sync Detector — Python GUI

A desktop application that detects deepfakes by analyzing audio-visual synchronization
using **MediaPipe FaceMesh**, **OpenCV**, and **Librosa**.

---

## Requirements

- Python 3.9 or higher
- pip

---

## Setup

### Step 1 — Create a virtual environment (recommended)

```bash
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on macOS / Linux:
source venv/bin/activate
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> First run may take a few minutes as MediaPipe downloads its face model automatically.

### Step 3 — Run the app

```bash
python main.py
```

---

## How to Use

1. Launch the app with `python main.py`
2. Click **Browse** and select a video file (MP4, AVI, MOV, MKV, WebM)
3. Click **▶ Analyze Video**
4. Wait for analysis (a 30-second video takes ~15–30 seconds)
5. Review the verdict, confidence score, timeline chart, and findings

---

## Files

```
deepfake-detector-python/
├── main.py            ← App entry point, UI layout
├── detector.py        ← Core detection engine (MediaPipe + librosa)
├── ui_components.py   ← Verdict card, timeline chart, findings list
├── requirements.txt   ← Python dependencies
└── README.md          ← This file
```

---

## Detection Algorithms

| Algorithm | What it checks | Deepfake signal |
|---|---|---|
| **Pearson Correlation** | Overall lip-audio sync quality | r < 0.3 |
| **Cross-Correlation** | Timing offset between signals | > 100 ms lag |
| **Motion Jitter** | Smoothness of lip movement | Too smooth (< 0.02) |
| **Energy Mismatch** | Audio loud, lips barely moving | > 25% of frames |
| **Over-Sync Check** | Suspiciously mechanical sync | Max corr > 0.9 |

A confidence score of **40% or above** triggers a deepfake verdict.

---

## Troubleshooting

**`No module named 'mediapipe'`** — Run `pip install mediapipe`

**`No module named 'customtkinter'`** — Run `pip install customtkinter`

**`moviepy` errors** — Run `pip install moviepy` and ensure `ffmpeg` is installed:
- Windows: `winget install ffmpeg` or download from https://ffmpeg.org
- macOS: `brew install ffmpeg`
- Linux: `sudo apt install ffmpeg`

**Video has no audio track** — The detector requires a video with an audio track to perform sync analysis.

**Slow analysis** — Reduce `MAX_FRAMES` in `detector.py` (default: 200) for faster results.

---

## Upgrade Ideas

- Add batch processing (analyze multiple videos at once)
- Export results to PDF report
- Add a video preview panel with live landmark overlay
- Train a CNN on lip-sync features for higher accuracy
