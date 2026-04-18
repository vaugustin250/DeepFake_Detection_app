"""
core/av_sync_detector.py
=========================
Layer 3: Audio-Visual Synchronisation Detection
-------------------------------------------------
Paper:  HOLA — Enhancing Audio-visual Deepfake Detection (Wu et al., MM 2025)

MediaPipe compatibility:
  ≥ 0.10.x → mp.tasks.vision.FaceLandmarker (new Tasks API)
  ≤ 0.9.x  → mp.solutions.face_mesh         (old API)
  Fallback  → lip aperture returns 0.0       (AV score uses audio-only path)

Audio extraction (FIX — was completely missing in original):
  Uses ffmpeg subprocess (primary) or moviepy (fallback) to pull
  16kHz mono audio from the video before analysis begins.
"""

import cv2
import numpy as np
import subprocess
import os
import tempfile
import urllib.request
from pathlib import Path
from collections import deque
from typing import Dict, Any, Optional, List, Tuple

LIP_TOP    = 13    # inner lip top
LIP_BOTTOM = 14    # inner lip bottom

# Model bundle for new MediaPipe Tasks API
_MP_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_MP_LANDMARKER_PATH = (
    Path(__file__).parent.parent / "models" / "face_landmarker.task"
)


class AVSyncDetector:

    HISTORY_LEN  = 60
    SAMPLE_RATE  = 16000
    HOP_MS       = 40
    CHUNK_FRAMES = 16000 // 25

    def __init__(self):
        self._mp_mesh_new  = None   # new Tasks API (≥0.10)
        self._mp_mesh_old  = None   # old solutions API (<0.10)
        self._mp_available = False

        self._lip_history:   deque = deque(maxlen=self.HISTORY_LEN)
        self._av_offsets:    deque = deque(maxlen=30)
        self._score_history: deque = deque(maxlen=30)

        self._w2v_model      = None
        self._w2v_proc       = None
        self.wav2vec2_loaded = False

        self._audio_buffer: deque = deque(maxlen=50)

        self._load_mediapipe()

    # ── Registry Integration ───────────────────────────────────────────────────
    def inject_registry(self, registry):
        if registry.wav2vec2_ok:
            self._w2v_model      = registry.wav2vec2_model
            self._w2v_proc       = registry.wav2vec2_proc
            self.wav2vec2_loaded = True

    # ── MediaPipe Init ─────────────────────────────────────────────────────────
    def _load_mediapipe(self):
        """
        Priority:
          1. New Tasks API  (mediapipe ≥ 0.10) — mp.tasks.vision.FaceLandmarker
          2. Old API        (mediapipe < 0.10)  — mp.solutions.face_mesh
          3. Fallback: lip_aperture returns 0.0 and AV score uses audio only
        """
        try:
            import mediapipe as mp

            # New Tasks API
            if hasattr(mp, "tasks") and hasattr(mp.tasks, "vision") \
                    and hasattr(mp.tasks.vision, "FaceLandmarker"):
                result = self._init_mp_tasks_landmarker(mp)
                if result:
                    self._mp_available = True
                    return

            # Old solutions API
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                self._mp_mesh_old = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
                self._mp_available = True
                return

            print("[AVSyncDetector] No supported MediaPipe API found — "
                  "lip aperture disabled, AV score uses audio-only mode.")

        except Exception as e:
            print(f"[AVSyncDetector] MediaPipe init error: {e} — "
                  "lip aperture disabled.")

    def _init_mp_tasks_landmarker(self, mp) -> bool:
        """Download and initialise FaceLandmarker (new API)."""
        try:
            model_path = _MP_LANDMARKER_PATH
            if not model_path.exists():
                model_path.parent.mkdir(parents=True, exist_ok=True)
                print("[AVSyncDetector] Downloading MediaPipe face landmarker "
                      "bundle (~5 MB)...")
                urllib.request.urlretrieve(_MP_LANDMARKER_URL, str(model_path))
                print("[AVSyncDetector] Downloaded OK.")

            BaseOptions      = mp.tasks.BaseOptions
            FaceLandmarker   = mp.tasks.vision.FaceLandmarker
            FaceLandmarkOpts = mp.tasks.vision.FaceLandmarkerOptions
            RunningMode      = mp.tasks.vision.RunningMode

            opts = FaceLandmarkOpts(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False)

            self._mp_mesh_new = FaceLandmarker.create_from_options(opts)
            return True

        except Exception as e:
            print(f"[AVSyncDetector] FaceLandmarker init failed: {e}")
            return False

    # ── Audio Extraction ───────────────────────────────────────────────────────
    def extract_audio_from_video(self, video_path: str
                                  ) -> Optional[np.ndarray]:
        """
        Extracts 16kHz mono audio from video using ffmpeg (primary)
        or moviepy (fallback). Returns float32 numpy array or None.
        """
        if not video_path or not os.path.exists(video_path):
            return None
        audio = self._extract_ffmpeg(video_path)
        if audio is not None:
            return audio
        audio = self._extract_moviepy(video_path)
        if audio is not None:
            return audio
        print("[AVSyncDetector] No audio track — AV sync uses lip-naturalness only.")
        return None

    def _extract_ffmpeg(self, path: str) -> Optional[np.ndarray]:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                tmp = tf.name
            cmd = ["ffmpeg", "-y", "-i", path,
                   "-ac", "1", "-ar", str(self.SAMPLE_RATE),
                   "-f", "wav", "-loglevel", "error", tmp]
            r = subprocess.run(cmd, timeout=60, capture_output=True)
            if r.returncode != 0 or not os.path.exists(tmp):
                return None
            import soundfile as sf
            audio, sr = sf.read(tmp)
            os.unlink(tmp)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != self.SAMPLE_RATE:
                audio = self._resample(audio, sr, self.SAMPLE_RATE)
            return audio.astype(np.float32)
        except FileNotFoundError:
            return None
        except Exception as e:
            print(f"[AVSyncDetector] ffmpeg error: {e}")
            return None

    def _extract_moviepy(self, path: str) -> Optional[np.ndarray]:
        try:
            from moviepy.editor import VideoFileClip
            clip = VideoFileClip(path)
            if clip.audio is None:
                clip.close(); return None
            arr = clip.audio.to_soundarray(fps=self.SAMPLE_RATE)
            clip.close()
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            return arr.astype(np.float32)
        except ImportError:
            return None
        except Exception as e:
            print(f"[AVSyncDetector] moviepy error: {e}")
            return None

    def _resample(self, audio, orig_sr, target_sr):
        try:
            import scipy.signal
            return scipy.signal.resample(audio, int(len(audio) * target_sr / orig_sr))
        except Exception:
            factor = orig_sr / target_sr
            idx = np.clip(np.arange(0, len(audio), factor).astype(int), 0, len(audio) - 1)
            return audio[idx]

    def get_audio_chunks(self, audio: Optional[np.ndarray],
                          fps: float = 25.0,
                          n_frames: int = 0) -> List[Optional[np.ndarray]]:
        if audio is None:
            return [None] * n_frames
        spf = int(self.SAMPLE_RATE / fps)
        return [
            audio[i * spf: (i + 1) * spf] if i * spf < len(audio) else None
            for i in range(n_frames)
        ]

    # ── Main Analysis ─────────────────────────────────────────────────────────
    def analyse(self, bgr_frame: np.ndarray,
                audio_chunk: Optional[np.ndarray] = None) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        lip_aperture = self._get_lip_aperture(bgr_frame)
        self._lip_history.append(lip_aperture)
        results["lip_aperture"] = round(lip_aperture, 4)

        av_offset_ms = 0.0
        if audio_chunk is not None and len(audio_chunk) > 0:
            self._audio_buffer.append(audio_chunk)
            if self.wav2vec2_loaded:
                av_offset_ms = self._wav2vec2_av_offset(audio_chunk)
            else:
                av_offset_ms = self._librosa_av_offset(audio_chunk)
        else:
            # No audio = bias toward DEEPFAKE (missing sync = suspicious)
            av_offset_ms = 150.0

        results["av_offset_ms"] = round(av_offset_ms, 1)
        av_score = self._compute_av_score(av_offset_ms)
        self._score_history.append(av_score)
        results["av_score"] = round(av_score, 2)
        return results

    # ── Lip Aperture ──────────────────────────────────────────────────────────
    def _get_lip_aperture(self, bgr: np.ndarray) -> float:
        if self._mp_mesh_new is not None:
            return self._lip_aperture_new(bgr)
        if self._mp_mesh_old is not None:
            return self._lip_aperture_old(bgr)
        return 0.0

    def _lip_aperture_new(self, bgr: np.ndarray) -> float:
        """MediaPipe Tasks API (≥ 0.10)."""
        try:
            import mediapipe as mp
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._mp_mesh_new.detect(mp_image)
            if not result.face_landmarks:
                return 0.0
            lm = result.face_landmarks[0]
            h, w = bgr.shape[:2]
            top_y      = lm[LIP_TOP].y    * h
            bottom_y   = lm[LIP_BOTTOM].y * h
            chin_y     = lm[152].y        * h
            forehead_y = lm[10].y         * h
            face_h     = max(abs(chin_y - forehead_y), 1)
            return round(abs(bottom_y - top_y) / face_h, 4)
        except Exception as e:
            print(f"[AVSyncDetector] new API lip aperture error: {e}")
            self._mp_mesh_new = None  # disable and fall through
            return 0.0

    def _lip_aperture_old(self, bgr: np.ndarray) -> float:
        """MediaPipe solutions API (< 0.10)."""
        try:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = self._mp_mesh_old.process(rgb)
            rgb.flags.writeable = True
            if not result.multi_face_landmarks:
                return 0.0
            lm = result.multi_face_landmarks[0].landmark
            h, w = bgr.shape[:2]
            top_y      = lm[LIP_TOP].y    * h
            bottom_y   = lm[LIP_BOTTOM].y * h
            chin_y     = lm[152].y        * h
            forehead_y = lm[10].y         * h
            face_h     = max(abs(chin_y - forehead_y), 1)
            return round(abs(bottom_y - top_y) / face_h, 4)
        except Exception:
            return 0.0

    # ── Wav2Vec2 AV Offset ────────────────────────────────────────────────────
    def _wav2vec2_av_offset(self, audio_chunk: np.ndarray) -> float:
        try:
            import torch
            audio = audio_chunk.astype(np.float32)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            mx = np.abs(audio).max()
            if mx > 0: audio /= mx

            inputs = self._w2v_proc(
                audio, sampling_rate=self.SAMPLE_RATE,
                return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = self._w2v_model(**inputs)

            hidden = outputs.last_hidden_state.squeeze(0).numpy()
            audio_energy = np.linalg.norm(hidden, axis=1)

            lip_arr = np.array(list(self._lip_history), dtype=np.float32)
            if len(lip_arr) < 4:
                return 0.0

            n = min(len(audio_energy), len(lip_arr), 50)
            ae = np.interp(np.linspace(0, 1, n),
                           np.linspace(0, 1, len(audio_energy)), audio_energy)
            la = np.interp(np.linspace(0, 1, n),
                           np.linspace(0, 1, len(lip_arr)), lip_arr)
            ae = (ae - ae.mean()) / (ae.std() + 1e-8)
            la = (la - la.mean()) / (la.std() + 1e-8)
            xcorr = np.correlate(ae, la, mode="full")
            lag = int(np.argmax(xcorr)) - (n - 1)
            return float(abs(lag) * self.HOP_MS)
        except Exception as e:
            print(f"[AVSyncDetector] Wav2Vec2 error: {e}")
            return self._librosa_av_offset(audio_chunk)

    # ── Librosa Fallback ───────────────────────────────────────────────────────
    def _librosa_av_offset(self, audio_chunk: np.ndarray) -> float:
        try:
            from scipy.signal import find_peaks
            audio = audio_chunk.astype(np.float32)
            if audio.ndim > 1: audio = audio.mean(axis=1)
            envelope = np.abs(audio)
            if envelope.max() == 0: return 0.0
            a_peaks, _ = find_peaks(envelope, height=0.3 * envelope.max(), distance=20)
            lip_arr = np.array(list(self._lip_history))
            if len(lip_arr) < 4 or lip_arr.max() == 0: return 0.0
            l_peaks, _ = find_peaks(lip_arr, height=0.3 * lip_arr.max(), distance=3)
            if len(a_peaks) == 0 or len(l_peaks) == 0: return 0.0
            offsets = []
            for ap in a_peaks[:5]:
                ap_ms = (ap / self.SAMPLE_RATE) * 1000
                l_ms  = l_peaks * self.HOP_MS
                offsets.append(abs(ap_ms - l_ms[np.argmin(np.abs(l_ms - ap_ms))]))
            return float(np.mean(offsets)) if offsets else 0.0
        except Exception:
            return self._lip_regularity_offset()

    def _lip_regularity_offset(self) -> float:
        if len(self._lip_history) < 8: return 0.0
        arr = np.array(list(self._lip_history)[-16:])
        if arr.std() < 1e-5: return 150.0
        jerk_std = float(np.std(np.diff(np.diff(arr))))
        if len(arr) > 4:
            acf  = np.correlate(arr - arr.mean(), arr - arr.mean(), mode='full')
            acfn = acf / (acf.max() + 1e-8)
            mid  = len(acfn) // 2
            per  = float(np.max(np.abs(acfn[mid + 2: mid + 15])))
        else:
            per = 0.0
        if per > 0.8 and jerk_std < 0.005: return 120.0
        if per > 0.6: return 70.0
        return float(jerk_std * 20.0)

    # ── AV Score ──────────────────────────────────────────────────────────────
    def _compute_av_score(self, av_offset_ms: float) -> float:
        lip_nat = self._lip_naturalness_score()
        if av_offset_ms < 50:
            av_comp = 1.0
        elif av_offset_ms > 120:
            av_comp = 0.0
        else:
            av_comp = 1.0 - (av_offset_ms - 50) / 70.0

        has_av = len(self._av_offsets) > 0 or len(self._audio_buffer) > 0
        score  = (av_comp * 0.65 + lip_nat * 0.35) * 100 if has_av else lip_nat * 100
        return round(max(0.0, min(100.0, score)), 2)

    def _lip_naturalness_score(self) -> float:
        if len(self._lip_history) < 10: return 0.5
        arr = np.array(list(self._lip_history))
        std = float(arr.std())
        if std < 1e-4: return 0.3
        nat = std / 0.01 * 0.6 if std < 0.01 else (0.5 if std > 0.20
              else 0.7 + 0.3 * (std / 0.15))
        try:
            recent = arr[-20:]
            if len(recent) > 6:
                acf  = np.correlate(recent - recent.mean(),
                                    recent - recent.mean(), 'full')
                acfn = acf / (np.abs(acf).max() + 1e-8)
                mid  = len(acfn) // 2
                if float(np.max(np.abs(acfn[mid + 2: mid + 10]))) > 0.85:
                    nat *= 0.6
        except Exception:
            pass
        return round(min(1.0, nat), 4)