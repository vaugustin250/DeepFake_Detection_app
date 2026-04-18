"""
core/av_sync_detector.py
=========================
Layer 3: Audio-Visual Synchronisation Detection
-------------------------------------------------
Paper:  HOLA — Enhancing Audio-visual Deepfake Detection
        (Wu et al., MM 2025)

Models used (loaded via ModelRegistry):
  • facebook/wav2vec2-base-960h  — Speech audio encoder
    Extracts phoneme-level hidden-state embeddings from raw audio.
    Cross-correlated against lip-aperture (MediaPipe) to find A/V offset.

CRITICAL FIX from original:
  The original code never extracted audio from the video file — audio_chunk
  was always None, so Wav2Vec2 NEVER ran. Layer 3 was silently falling back
  to lip-naturalness heuristics only.

  FIX: Added extract_audio_from_video() which uses ffmpeg (subprocess) or
  soundfile+moviepy to pull the audio track before analysis. The pipeline now
  calls this once per video and passes chunks to analyse().

Usage from pipeline / tabs:
  detector = AVSyncDetector()
  audio = detector.extract_audio_from_video(video_path)  # returns np.ndarray or None
  for frame in frames:
      chunk = audio[i*chunk_size:(i+1)*chunk_size] if audio is not None else None
      result = detector.analyse(frame, audio_chunk=chunk)
"""

import cv2
import numpy as np
import subprocess
import os
import tempfile
from collections import deque
from typing import Dict, Any, Optional, List, Tuple
import time

LIP_TOP    = 13   # MediaPipe inner lip top landmark
LIP_BOTTOM = 14   # MediaPipe inner lip bottom landmark


class AVSyncDetector:

    HISTORY_LEN  = 60
    SAMPLE_RATE  = 16000      # Wav2Vec2 expected sample rate
    HOP_MS       = 40         # ~1 video frame at 25fps
    CHUNK_FRAMES = 16000 // 25  # audio samples per video frame at 25fps

    def __init__(self):
        self._mp_mesh        = None
        self._lip_history:   deque = deque(maxlen=self.HISTORY_LEN)
        self._av_offsets:    deque = deque(maxlen=30)
        self._score_history: deque = deque(maxlen=30)

        # Wav2Vec2 (injected from registry)
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

    # ── MediaPipe Init ────────────────────────────────────────────────────────
    def _load_mediapipe(self):
        try:
            import mediapipe as mp
            self._mp_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5)
        except Exception as e:
            print(f"[AVSyncDetector] MediaPipe not available: {e}")

    # ── AUDIO EXTRACTION (FIX: this was missing in original) ─────────────────
    def extract_audio_from_video(self, video_path: str) -> Optional[np.ndarray]:
        """
        Extracts audio from a video file as a float32 mono numpy array at
        self.SAMPLE_RATE (16 kHz) for Wav2Vec2.

        Priority:
          1. ffmpeg subprocess (fastest, most reliable)
          2. moviepy (if installed)
          3. None (no audio track or extraction failed)

        Returns None if the video has no audio or extraction fails.
        """
        if not video_path or not os.path.exists(video_path):
            return None

        # Method 1: ffmpeg
        audio = self._extract_audio_ffmpeg(video_path)
        if audio is not None:
            return audio

        # Method 2: moviepy
        audio = self._extract_audio_moviepy(video_path)
        if audio is not None:
            return audio

        print("[AVSyncDetector] No audio track found or extraction failed — "
              "AV sync will use lip-naturalness fallback.")
        return None

    def _extract_audio_ffmpeg(self, video_path: str) -> Optional[np.ndarray]:
        """Uses ffmpeg CLI to extract 16kHz mono PCM audio."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                tmp_wav = tf.name
            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-ac", "1",                        # mono
                "-ar", str(self.SAMPLE_RATE),      # 16kHz
                "-f", "wav",
                "-loglevel", "error",
                tmp_wav
            ]
            result = subprocess.run(cmd, timeout=60,
                                    capture_output=True, text=True)
            if result.returncode != 0 or not os.path.exists(tmp_wav):
                return None
            import soundfile as sf
            audio, sr = sf.read(tmp_wav)
            os.unlink(tmp_wav)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            # Resample if needed
            if sr != self.SAMPLE_RATE:
                audio = self._resample(audio, sr, self.SAMPLE_RATE)
            return audio.astype(np.float32)
        except FileNotFoundError:
            return None  # ffmpeg not installed
        except Exception as e:
            print(f"[AVSyncDetector] ffmpeg error: {e}")
            return None

    def _extract_audio_moviepy(self, video_path: str) -> Optional[np.ndarray]:
        """Uses moviepy as fallback for audio extraction."""
        try:
            from moviepy.editor import VideoFileClip
            clip = VideoFileClip(video_path)
            if clip.audio is None:
                clip.close()
                return None
            audio_arr = clip.audio.to_soundarray(fps=self.SAMPLE_RATE)
            clip.close()
            if audio_arr.ndim > 1:
                audio_arr = audio_arr.mean(axis=1)
            return audio_arr.astype(np.float32)
        except ImportError:
            return None
        except Exception as e:
            print(f"[AVSyncDetector] moviepy error: {e}")
            return None

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple linear resampling."""
        try:
            import scipy.signal
            ratio = target_sr / orig_sr
            new_len = int(len(audio) * ratio)
            return scipy.signal.resample(audio, new_len)
        except Exception:
            # Crude fallback
            factor = orig_sr / target_sr
            indices = np.arange(0, len(audio), factor).astype(int)
            indices = np.clip(indices, 0, len(audio) - 1)
            return audio[indices]

    # ── Split audio into per-frame chunks ─────────────────────────────────────
    def get_audio_chunks(self, audio: np.ndarray,
                          fps: float = 25.0,
                          n_frames: int = 0) -> List[Optional[np.ndarray]]:
        """
        Splits full audio array into per-frame chunks aligned to video frames.
        Returns a list of numpy arrays (one per frame) or None if no audio.
        """
        if audio is None:
            return [None] * n_frames
        samples_per_frame = int(self.SAMPLE_RATE / fps)
        chunks = []
        for i in range(n_frames):
            start = i * samples_per_frame
            end   = start + samples_per_frame
            chunk = audio[start:end] if start < len(audio) else None
            chunks.append(chunk)
        return chunks

    # ── Main Analysis ─────────────────────────────────────────────────────────
    def analyse(self, bgr_frame: np.ndarray,
                audio_chunk: Optional[np.ndarray] = None) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        # Lip aperture via MediaPipe
        lip_aperture = self._get_lip_aperture(bgr_frame)
        self._lip_history.append(lip_aperture)
        results["lip_aperture"] = round(lip_aperture, 4)

        # Audio processing (now actually runs when extract_audio_from_video was called)
        av_offset_ms = 0.0
        if audio_chunk is not None and len(audio_chunk) > 0:
            self._audio_buffer.append(audio_chunk)
            if self.wav2vec2_loaded:
                av_offset_ms = self._wav2vec2_av_offset(audio_chunk)
            else:
                av_offset_ms = self._librosa_av_offset(audio_chunk)

        results["av_offset_ms"] = round(av_offset_ms, 1)

        # AV score
        av_score = self._compute_av_score(av_offset_ms)
        self._score_history.append(av_score)
        results["av_score"] = round(av_score, 2)
        return results

    # ── Wav2Vec2 AV Offset ────────────────────────────────────────────────────
    def _wav2vec2_av_offset(self, audio_chunk: np.ndarray) -> float:
        """
        HOLA approach using Wav2Vec2:
          1. Encode audio chunk → frame-level hidden states (768-dim)
          2. Compute L2 norm of hidden states → audio energy signal
          3. Cross-correlate with lip aperture history
          4. Peak lag → A/V offset in ms
          5. |offset| > 80ms → deepfake sync error
        """
        try:
            import torch

            audio = audio_chunk.astype(np.float32)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            max_val = np.abs(audio).max()
            if max_val > 0:
                audio /= max_val

            inputs = self._w2v_proc(
                audio,
                sampling_rate=self.SAMPLE_RATE,
                return_tensors="pt",
                padding=True)

            with torch.no_grad():
                outputs = self._w2v_model(**inputs)

            # (1, T, 768) → energy per frame
            hidden = outputs.last_hidden_state.squeeze(0).numpy()
            audio_energy = np.linalg.norm(hidden, axis=1)

            lip_arr = np.array(list(self._lip_history), dtype=np.float32)
            if len(lip_arr) < 4:
                return 0.0

            target_len = min(len(audio_energy), len(lip_arr), 50)
            ae = np.interp(np.linspace(0, 1, target_len),
                           np.linspace(0, 1, len(audio_energy)), audio_energy)
            la = np.interp(np.linspace(0, 1, target_len),
                           np.linspace(0, 1, len(lip_arr)), lip_arr)

            ae = (ae - ae.mean()) / (ae.std() + 1e-8)
            la = (la - la.mean()) / (la.std() + 1e-8)

            xcorr = np.correlate(ae, la, mode="full")
            lag_samples = int(np.argmax(xcorr)) - (target_len - 1)
            offset_ms = abs(lag_samples) * self.HOP_MS
            return float(offset_ms)

        except Exception as e:
            print(f"[AVSyncDetector] Wav2Vec2 error: {e}")
            return self._librosa_av_offset(audio_chunk)

    # ── Librosa AV Offset (fallback) ──────────────────────────────────────────
    def _librosa_av_offset(self, audio_chunk: np.ndarray) -> float:
        try:
            from scipy.signal import find_peaks

            audio = audio_chunk.astype(np.float32)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            envelope = np.abs(audio)
            if envelope.max() == 0:
                return 0.0

            audio_peaks, _ = find_peaks(envelope,
                                         height=0.3 * envelope.max(),
                                         distance=20)
            lip_arr = np.array(list(self._lip_history))
            if len(lip_arr) < 4 or lip_arr.max() == 0:
                return 0.0

            lip_peaks, _ = find_peaks(lip_arr,
                                       height=0.3 * lip_arr.max(),
                                       distance=3)
            if len(audio_peaks) == 0 or len(lip_peaks) == 0:
                return 0.0

            offsets = []
            for ap in audio_peaks[:5]:
                ap_time_ms   = (ap / self.SAMPLE_RATE) * 1000
                lip_times_ms = lip_peaks * self.HOP_MS
                nearest      = lip_times_ms[np.argmin(np.abs(lip_times_ms - ap_time_ms))]
                offsets.append(abs(ap_time_ms - nearest))

            return float(np.mean(offsets)) if offsets else 0.0

        except Exception:
            return self._lip_regularity_offset()

    # ── Lip Regularity Fallback ────────────────────────────────────────────────
    def _lip_regularity_offset(self) -> float:
        """Last resort: infer sync issues from lip motion unnaturalness."""
        if len(self._lip_history) < 8:
            return 0.0
        arr = np.array(list(self._lip_history)[-16:])
        if arr.std() < 1e-5:
            return 150.0
        jerk = np.diff(np.diff(arr))
        jerk_std = float(np.std(jerk))
        if len(arr) > 4:
            acf  = np.correlate(arr - arr.mean(), arr - arr.mean(), mode='full')
            acfn = acf / (acf.max() + 1e-8)
            mid  = len(acfn) // 2
            periodicity = float(np.max(np.abs(acfn[mid + 2: mid + 15])))
        else:
            periodicity = 0.0
        if periodicity > 0.8 and jerk_std < 0.005:
            return 120.0
        elif periodicity > 0.6:
            return 70.0
        return float(jerk_std * 20.0)

    # ── AV Score ──────────────────────────────────────────────────────────────
    def _compute_av_score(self, av_offset_ms: float) -> float:
        lip_naturalness = self._lip_naturalness_score()

        if av_offset_ms < 50:
            av_component = 1.0
        elif av_offset_ms > 120:
            av_component = 0.0
        else:
            av_component = 1.0 - (av_offset_ms - 50) / 70.0

        has_av = len(self._av_offsets) > 0 or len(self._audio_buffer) > 0
        if has_av:
            score = (av_component * 0.65 + lip_naturalness * 0.35) * 100
        else:
            score = lip_naturalness * 100

        return round(max(0.0, min(100.0, score)), 2)

    def _lip_naturalness_score(self) -> float:
        if len(self._lip_history) < 10:
            return 0.5
        arr = np.array(list(self._lip_history))
        std = float(arr.std())
        if std < 1e-4:
            return 0.3
        if std < 0.01:
            naturalness = std / 0.01 * 0.6
        elif std > 0.20:
            naturalness = 0.5
        else:
            naturalness = 0.7 + 0.3 * (std / 0.15)
        try:
            recent = arr[-20:]
            if len(recent) > 6:
                acf  = np.correlate(recent - recent.mean(),
                                    recent - recent.mean(), 'full')
                acfn = acf / (np.abs(acf).max() + 1e-8)
                mid  = len(acfn) // 2
                max_per = float(np.max(np.abs(acfn[mid + 2: mid + 10])))
                if max_per > 0.85:
                    naturalness *= 0.6
        except Exception:
            pass
        return round(min(1.0, naturalness), 4)

    # ── Lip Aperture via MediaPipe ────────────────────────────────────────────
    def _get_lip_aperture(self, bgr: np.ndarray) -> float:
        if self._mp_mesh is None:
            return 0.0
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._mp_mesh.process(rgb)
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
