"""
core/pipeline.py  — ForensicStream AI  v4
==========================================
Orchestrates the 3-layer deepfake detection pipeline.

KEY FIXES:
  1. face_bbox now passed from spatial → temporal layer
  2. Trust score thresholds tightened: >=65=AUTHENTIC, <45=DEEPFAKE
     (old: >=60/>=40 — too easy to score as authentic with broken fallback math)
  3. analyse_video() properly extracts audio before frame loop
  4. Debug mode prints per-frame signal breakdown
"""

import numpy as np
import cv2
from typing import Callable, Optional, Dict, Any, List, Tuple

from spatial_detector import SpatialDetector
from temporal_detector import TemporalDetector
from av_sync_detector import AVSyncDetector

DEBUG = False   # set True to print per-frame signal breakdown to console


class DeepfakePipeline:

    WEIGHTS = {"spatial": 0.30, "temporal": 0.40, "av": 0.30}

    def __init__(self):
        self._spatial  = SpatialDetector()
        self._temporal = TemporalDetector()
        self._av       = AVSyncDetector()
        self._registry = None
        self._models_loaded = False

    def load_models(self, progress_cb: Optional[Callable] = None):
        def cb(msg, level="info"):
            if progress_cb: progress_cb(msg, level)
        try:
            from models.model_loader import ModelRegistry
            self._registry = ModelRegistry()
            self._registry.load_all(progress_cb=cb)
        except Exception as e:
            cb(f"ModelRegistry error: {e}", "warn")
            self._registry = None

        if self._registry:
            self._spatial.inject_registry(self._registry)
            self._temporal.inject_registry(self._registry)
            self._av.inject_registry(self._registry)
        else:
            self._spatial.load(progress_cb=cb)

        self._models_loaded = True
        cb("Pipeline ready.", "success")

    def model_status_text(self) -> str:
        if self._registry:
            return self._registry.status_text()
        s = "✓" if self._spatial.is_loaded    else "○"
        t = "✓" if self._temporal.xclip_loaded else "○"
        a = "✓" if self._av.wav2vec2_loaded    else "○"
        return (f"Spatial:   {s} ViT {'Loaded' if self._spatial.is_loaded else 'Not loaded (math fallback)'}\n"
                f"X-CLIP:    {t} {'Loaded' if self._temporal.xclip_loaded else 'Not loaded'}\n"
                f"Wav2Vec2:  {a} {'Loaded' if self._av.wav2vec2_loaded else 'Not loaded'}")

    def analyse_video(self, video_path: str,
                       max_frames: int = 32,
                       progress_cb=None) -> Tuple[List[Dict], List[np.ndarray]]:
        cap   = cv2.VideoCapture(video_path)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step  = max(1, total // max_frames)
        frames = []
        i = 0
        while len(frames) < max_frames:
            ret, f = cap.read()
            if not ret: break
            if i % step == 0: frames.append(f)
            i += 1
        cap.release()

        audio  = self._av.extract_audio_from_video(video_path)
        chunks = self._av.get_audio_chunks(audio, fps=fps, n_frames=len(frames))

        results = []
        for idx, (frame, chunk) in enumerate(zip(frames, chunks)):
            r = self.analyse_frame(frame, audio_chunk=chunk)
            results.append(r)
            if progress_cb: progress_cb(idx+1, len(frames))
        return results, frames

    def analyse_frame(self, frame: np.ndarray,
                       audio_chunk: Optional[np.ndarray] = None) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        spatial_r  = self._spatial.analyse(frame)
        results.update(spatial_r)

        # Pass face_bbox to temporal so it can focus signals on face region
        face_bbox  = results.get("face_bbox")
        temporal_r = self._temporal.analyse(frame, face_bbox=face_bbox)
        results.update(temporal_r)

        av_r = self._av.analyse(frame, audio_chunk=audio_chunk)
        results.update(av_r)

        s1 = results.get("spatial_score",  50.0)
        s2 = results.get("temporal_score", 50.0)
        s3 = results.get("av_score",       50.0)
        trust = s1*self.WEIGHTS["spatial"] + s2*self.WEIGHTS["temporal"] + s3*self.WEIGHTS["av"]
        results["trust_score"] = round(trust, 2)

        if DEBUG:
            print(f"[Pipeline] spatial={s1:.1f}  temporal={s2:.1f}  av={s3:.1f}  "
                  f"trust={trust:.1f}  "
                  f"dct={results.get('dct_score',0):.1f}  "
                  f"blend={results.get('blend_score',0):.1f}  "
                  f"decouple={results.get('motion_consistency',0):.1f}")
        return results
