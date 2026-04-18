"""
detector.py — Core deepfake detection engine

Libraries used:
  - mediapipe  : FaceMesh for 468 facial landmarks (precise lip tracking)
  - opencv-python : Video frame extraction
  - librosa    : Audio energy analysis
  - numpy      : Signal processing & statistics
  - moviepy    : Extract audio track from video
"""

import cv2
import numpy as np
import librosa
import tempfile
import os
import sys
import types
from moviepy import VideoFileClip

# Import face_mesh properly
# Stub mediapipe.tasks to avoid pulling in TensorFlow (not needed for solutions).
if "mediapipe.tasks.python" not in sys.modules:
    tasks_pkg = types.ModuleType("mediapipe.tasks")
    tasks_pkg.__path__ = []
    tasks_py = types.ModuleType("mediapipe.tasks.python")
    tasks_py.__path__ = []
    sys.modules.setdefault("mediapipe.tasks", tasks_pkg)
    sys.modules.setdefault("mediapipe.tasks.python", tasks_py)

try:
    from mediapipe.python.solutions import face_mesh
except ImportError:
    from mediapipe.solutions import face_mesh


# ── MediaPipe lip landmark indices ──────────────────────────
# Upper lip top center → lower lip bottom center (vertical aperture)
LIP_UPPER_TOP    = 13
LIP_LOWER_BOTTOM = 14
LIP_LEFT         = 61
LIP_RIGHT        = 291

# Full lip outline for drawing
LIP_OUTLINE = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61
]


class DeepfakeDetector:
    """
    Analyzes a video file for deepfake artifacts by measuring
    audio-visual synchronization quality.
    """

    def __init__(self):
        self.face_mesh = face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.FPS_TARGET = 15       # frames per second to analyze
        self.MAX_FRAMES = 200      # cap to keep analysis fast

    # ── Main pipeline ────────────────────────────────────────
    def analyze(self, video_path, progress_callback=None, status_callback=None):
        def prog(v, msg=""):
            if progress_callback: progress_callback(v, msg)
        def stat(msg, color=None):
            if status_callback: status_callback(msg, color)

        # Step 1: Extract audio
        stat("Extracting audio track…")
        audio_data, duration = self._extract_audio_energy(video_path, prog)
        prog(0.3, f"Audio extracted — {len(audio_data)} frames at {self.FPS_TARGET} fps")

        # Step 2: Extract lip landmarks
        stat("Running MediaPipe FaceMesh on video frames…")
        lip_data, frame_times = self._extract_lip_aperture(video_path, len(audio_data), prog)
        prog(0.75, f"Lip tracking done — {len(lip_data)} frames analyzed")

        # Step 3: Score
        stat("Computing synchronization metrics…")
        result = self._compute_score(lip_data, audio_data)
        prog(1.0, "Done")

        return result

    # ── Step 1: Audio energy ─────────────────────────────────
    def _extract_audio_energy(self, video_path, prog):
        """
        Extracts the audio track from the video, computes RMS energy
        per frame window at FPS_TARGET fps, and normalizes to [0, 1].
        """
        # Extract audio to temp wav
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            clip = VideoFileClip(video_path)
            duration = clip.duration
            if clip.audio is None:
                raise ValueError("Video has no audio track. Cannot perform sync analysis.")
            clip.audio.write_audiofile(tmp.name, logger=None)
            clip.close()

            # Load with librosa
            y, sr = librosa.load(tmp.name, sr=None, mono=True)
            hop = int(sr / self.FPS_TARGET)
            rms = librosa.feature.rms(y=y, frame_length=hop*2, hop_length=hop)[0]

            # Normalize
            max_val = rms.max() or 1.0
            normalized = (rms / max_val).tolist()
            return normalized, duration
        finally:
            os.unlink(tmp.name)

    # ── Step 2: Lip aperture via MediaPipe ───────────────────
    def _extract_lip_aperture(self, video_path, target_frames, prog):
        """
        Seeks through video at FPS_TARGET, runs MediaPipe FaceMesh on
        each frame, and computes the normalized lip aperture (vertical
        distance between upper and lower lip landmarks).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps

        num_samples = min(target_frames, self.MAX_FRAMES)
        sample_times = np.linspace(0, duration * 0.98, num_samples)

        apertures = []
        frame_times = []

        for idx, t in enumerate(sample_times):
            frame_num = int(t * video_fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                apertures.append(0.0)
                frame_times.append(t)
                continue

            aperture = self._measure_lip_aperture(frame)
            apertures.append(aperture)
            frame_times.append(t)

            if idx % 10 == 0:
                p = 0.3 + (idx / num_samples) * 0.45
                prog(p, f"Analyzing frame {idx+1}/{num_samples}…")

        cap.release()

        # Normalize
        arr = np.array(apertures, dtype=float)
        max_val = arr.max() or 1.0
        normalized = (arr / max_val).tolist()
        return normalized, frame_times

    def _measure_lip_aperture(self, frame_bgr):
        """
        Runs MediaPipe FaceMesh on a single BGR frame and returns
        the normalized vertical distance between upper and lower lips.
        Falls back to pixel-based estimation if no face detected.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        h, w = frame_bgr.shape[:2]

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            upper_y = lm[LIP_UPPER_TOP].y * h
            lower_y = lm[LIP_LOWER_BOTTOM].y * h
            left_x  = lm[LIP_LEFT].x * w
            right_x  = lm[LIP_RIGHT].x * w
            lip_width = abs(right_x - left_x) or 1
            aperture = abs(lower_y - upper_y) / lip_width
            return float(aperture)
        else:
            # Fallback: dark pixel density in estimated lip region
            return self._pixel_fallback(frame_bgr)

    def _pixel_fallback(self, frame):
        """Pixel brightness fallback when no face detected."""
        h, w = frame.shape[:2]
        y1, y2 = int(h * 0.55), int(h * 0.82)
        x1, x2 = int(w * 0.25), int(w * 0.75)
        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        avg = float(gray.mean())
        threshold = avg * 0.6
        dark = float(np.sum(gray < threshold))
        return min(1.0, dark / gray.size * 6)

    # ── Step 3: Scoring ──────────────────────────────────────
    def _compute_score(self, lip_data, audio_data):
        n = min(len(lip_data), len(audio_data))
        lip   = np.array(lip_data[:n])
        audio = np.array(audio_data[:n])

        # Metrics
        corr      = self._pearson(lip, audio)
        lag_frames, max_corr = self._cross_correlation(lip, audio, max_lag=8)
        lag_ms    = abs(lag_frames) * (1000 / self.FPS_TARGET)
        jitter    = self._motion_jitter(lip)
        mismatch  = self._energy_mismatch(lip, audio)

        findings = []
        score    = 0

        # ① Correlation
        if corr < 0.3:
            score += 30
            findings.append({"label": "Low audio-lip correlation",
                              "detail": f"Pearson r = {corr:.2f} (real speech typically > 0.4)",
                              "severity": "high"})
        elif corr < 0.5:
            score += 15
            findings.append({"label": "Moderate sync mismatch",
                              "detail": f"Pearson r = {corr:.2f}",
                              "severity": "medium"})
        else:
            findings.append({"label": "Good audio-lip correlation",
                              "detail": f"Pearson r = {corr:.2f}",
                              "severity": "ok"})

        # ② Temporal lag
        if lag_ms > 150:
            score += 35
            findings.append({"label": "Significant temporal lag",
                              "detail": f"{lag_ms:.0f} ms offset detected (threshold: 100 ms)",
                              "severity": "high"})
        elif lag_ms > 80:
            score += 15
            findings.append({"label": "Minor temporal offset",
                              "detail": f"{lag_ms:.0f} ms offset",
                              "severity": "medium"})
        else:
            findings.append({"label": "Temporal alignment normal",
                              "detail": f"{lag_ms:.0f} ms offset",
                              "severity": "ok"})

        # ③ Motion jitter
        if jitter < 0.02 and n > 20:
            score += 20
            findings.append({"label": "Unnaturally smooth lip motion",
                              "detail": f"Jitter index = {jitter:.4f} (too low — real speech has micro-tremors)",
                              "severity": "high"})
        else:
            findings.append({"label": "Natural lip motion variability",
                              "detail": f"Jitter index = {jitter:.4f}",
                              "severity": "ok"})

        # ④ Energy mismatch
        if mismatch > 0.25:
            score += 20
            findings.append({"label": "Audio-visual energy mismatch",
                              "detail": f"{mismatch*100:.0f}% of frames: high audio but low lip movement",
                              "severity": "high"})
        else:
            findings.append({"label": "Energy patterns consistent",
                              "detail": f"{mismatch*100:.0f}% mismatch frames",
                              "severity": "ok"})

        # ⑤ Over-synchronization
        if max_corr > 0.9:
            score += 10
            findings.append({"label": "Over-synchronization detected",
                              "detail": "Max cross-correlation unusually high — may indicate mechanical/generated sync",
                              "severity": "medium"})

        confidence = min(int(score), 100)
        return {
            "confidence":    confidence,
            "is_deepfake":   confidence >= 40,
            "findings":      findings,
            "pearson_r":     round(corr, 3),
            "lag_ms":        round(lag_ms, 1),
            "jitter":        round(jitter, 4),
            "mismatch_pct":  round(mismatch * 100, 1),
            "frames":        n,
            "avg_aperture":  round(float(lip.mean()), 3),
            "sync_score":    int(corr * 100),
            "lip_data":      lip.tolist(),
            "audio_data":    audio.tolist(),
        }

    # ── Signal processing helpers ────────────────────────────
    @staticmethod
    def _pearson(a, b):
        if len(a) < 2: return 0.0
        ma, mb = a.mean(), b.mean()
        num = ((a - ma) * (b - mb)).sum()
        den = np.sqrt(((a - ma)**2).sum() * ((b - mb)**2).sum())
        return float(num / den) if den else 0.0

    @staticmethod
    def _cross_correlation(a, b, max_lag=8):
        n = len(a)
        best_corr, best_lag = -np.inf, 0
        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                s = float(np.dot(a[lag:], b[:n-lag])) / (n - lag)
            else:
                s = float(np.dot(a[:n+lag], b[-lag:])) / (n + lag)
            if s > best_corr:
                best_corr, best_lag = s, lag
        return best_lag, best_corr

    @staticmethod
    def _motion_jitter(lip):
        if len(lip) < 3: return 0.0
        d1 = np.diff(lip)
        d2 = np.diff(d1)
        return float(np.abs(d2).mean())

    @staticmethod
    def _energy_mismatch(lip, audio):
        mask = (audio > 0.5) & (lip < 0.15)
        return float(mask.sum() / len(lip))
