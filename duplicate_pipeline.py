"""
core/pipeline.py
================
Orchestrates the 3-layer deepfake detection pipeline.

Model loading flow:
  1. Run download_models.py  → downloads to ./models/
  2. ModelRegistry.load_all() → loads from disk into memory
  3. inject_registry() → shares loaded models with all three detectors
  4. No model is loaded twice, no duplicate RAM usage

Layer weights (from paper accuracy contributions):
  Spatial  : 0.40  (EfficientNet B0 paper)
  Temporal : 0.35  (AltFreezing paper)
  AV Sync  : 0.25  (HOLA paper)
"""
import numpy as np
import cv2
import time
from typing import Callable, Optional, Dict, Any

from file.spatial_detector import SpatialDetector
from duplicate_temporal_detector import TemporalDetector
from duplicate_av_sync_detector import AVSyncDetector


class DeepfakePipeline:

    WEIGHTS = {"spatial": 0.40, "temporal": 0.35, "av": 0.25}

    def __init__(self):
        self._spatial   = SpatialDetector()
        self._temporal  = TemporalDetector()
        self._av        = AVSyncDetector()
        self._registry  = None
        self._models_loaded = False

    # ── Model Loading ──────────────────────────────────────────────────────────
    def load_models(self, progress_cb: Optional[Callable] = None):
        """
        Loads all models via ModelRegistry.
        Prefers local ./models/ directory (populated by download_models.py).
        Falls back to HuggingFace download if local files not found.
        """
        def cb(msg, level="info"):
            if progress_cb: progress_cb(msg, level)

        try:
            from models.model_loader import ModelRegistry
            self._registry = ModelRegistry()
            self._registry.load_all(progress_cb=cb)
        except Exception as e:
            cb(f"ModelRegistry error: {e} — falling back to direct load.", "warn")
            self._registry = None

        # Inject shared model objects into each detector
        if self._registry is not None:
            self._spatial.inject_registry(self._registry)
            self._temporal.inject_registry(self._registry)
            self._av.inject_registry(self._registry)
            cb("All models injected into detection layers.", "success")
        else:
            # Direct fallback loading
            cb("Loading models directly...", "info")
            self._spatial.load(progress_cb=cb)
            cb("Temporal layer: optical flow (always available).", "success")
            cb("AV layer: MediaPipe peak matching (always available).", "success")

        self._models_loaded = True
        cb("Pipeline ready.", "success")

    def model_status_text(self) -> str:
        if self._registry:
            return self._registry.status_text()
        s = "✓" if self._spatial.is_loaded  else "○"
        t = "✓" if self._temporal.xclip_loaded else "○"
        a = "✓" if self._av.wav2vec2_loaded  else "○"
        return (f"Spatial:   {s} {'Loaded' if self._spatial.is_loaded else 'Not loaded'}\n"
                f"X-CLIP:    {t} {'Loaded' if self._temporal.xclip_loaded else 'Not loaded'}\n"
                f"Wav2Vec2:  {a} {'Loaded' if self._av.wav2vec2_loaded else 'Not loaded'}")

    # ── Main Analysis ──────────────────────────────────────────────────────────
    def analyse_frame(self, frame: np.ndarray,
                       audio_chunk: Optional[np.ndarray] = None) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        spatial_r  = self._spatial.analyse(frame)
        results.update(spatial_r)

        temporal_r = self._temporal.analyse(frame)
        results.update(temporal_r)

        av_r = self._av.analyse(frame, audio_chunk=audio_chunk)
        results.update(av_r)

        s1 = results.get("spatial_score",  50.0)
        s2 = results.get("temporal_score", 50.0)
        s3 = results.get("av_score",       50.0)

        trust = (s1 * self.WEIGHTS["spatial"] +
                 s2 * self.WEIGHTS["temporal"] +
                 s3 * self.WEIGHTS["av"])
        results["trust_score"] = round(trust, 2)
        return results
