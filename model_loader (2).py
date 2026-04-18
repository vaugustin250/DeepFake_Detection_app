"""
models/model_loader.py
=======================
Central model loader for ForensicStream AI.

After running download_models.py, this module:
  1. Reads models/config.json to find local model paths
  2. Loads each model from disk (no re-download)
  3. Provides a single shared registry so all detector layers
     share the same loaded model objects (no duplicate RAM usage)

Usage:
    from models.model_loader import ModelRegistry
    registry = ModelRegistry()          # reads config.json
    registry.load_all(progress_cb=cb)   # loads from disk

    spatial_pipe = registry.spatial_pipeline
    wav2vec_model, wav2vec_proc = registry.wav2vec2
    xclip_model, xclip_proc = registry.xclip
"""

import json
import os
from pathlib import Path
from typing import Optional, Callable, Tuple, Any

MODELS_DIR   = Path(__file__).parent
CONFIG_PATH  = MODELS_DIR / "config.json"


class ModelRegistry:
    """
    Singleton-style registry that holds all loaded model objects.
    Shared across all detector layers to avoid loading the same
    weights multiple times.
    """

    def __init__(self):
        self._config          = self._read_config()
        self.spatial_pipeline = None    # HF pipeline("image-classification")
        self.wav2vec2_model   = None    # Wav2Vec2Model
        self.wav2vec2_proc    = None    # Wav2Vec2Processor
        self.xclip_model      = None    # XCLIPModel
        self.xclip_proc       = None    # XCLIPProcessor

        self.spatial_ok  = False
        self.wav2vec2_ok = False
        self.xclip_ok    = False

    # ── Config ─────────────────────────────────────────────────────────────────
    def _read_config(self) -> dict:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH) as f:
                return json.load(f)
        return {}

    def _model_path(self, key: str) -> Optional[str]:
        """Returns local model path if it exists and was successfully downloaded."""
        if not self._config.get(f"{key}_ok", False):
            return None
        path = self._config.get(f"{key}_model")
        if path and Path(path).exists() and (Path(path) / "DOWNLOAD_COMPLETE").exists():
            return path
        return None

    # ── Load All ───────────────────────────────────────────────────────────────
    def load_all(self, progress_cb: Optional[Callable] = None):
        """
        Loads all three layer models from local disk.
        Call after download_models.py has been run.
        """
        def cb(msg, level="info"):
            if progress_cb:
                progress_cb(msg, level)
            else:
                print(f"[ModelLoader] {msg}")

        cb("Loading models from local disk...", "info")

        self._load_spatial(cb)
        self._load_xclip(cb)
        self._load_wav2vec2(cb)

        loaded = sum([self.spatial_ok, self.xclip_ok, self.wav2vec2_ok])
        cb(f"Model loading complete — {loaded}/3 layers fully loaded.", "success")

    # ── Layer 1: Spatial Classifier ────────────────────────────────────────────
    def _load_spatial(self, cb):
        cb("Loading Layer 1 — Spatial classifier (ViT)...", "info")
        try:
            from transformers import pipeline as hf_pipeline

            path = self._model_path("spatial")
            model_src = path if path else "prithivMLmods/Deep-Fake-Detector-v2-Model"

            if path:
                cb(f"  Loading from local disk: {path}", "muted")
            else:
                cb("  Local model not found — downloading from HuggingFace...", "warn")

            self.spatial_pipeline = hf_pipeline(
                "image-classification",
                model=model_src,
                device=-1)  # CPU; change to device=0 for GPU

            self.spatial_ok = True
            cb("✓ Spatial classifier loaded.", "success")

        except Exception as e:
            cb(f"✗ Spatial load failed: {e}", "warn")
            cb("  Layer 1 will use warp-math fallback.", "muted")
            self.spatial_ok = False

    # ── Layer 2: X-CLIP Temporal Encoder ──────────────────────────────────────
    def _load_xclip(self, cb):
        cb("Loading Layer 2 — X-CLIP temporal encoder...", "info")
        try:
            from transformers import XCLIPProcessor, XCLIPModel

            path = self._model_path("xclip")
            model_src = path if path else "microsoft/xclip-base-patch32"

            if path:
                cb(f"  Loading from local disk: {path}", "muted")
            else:
                cb("  Local model not found — downloading from HuggingFace...", "warn")

            self.xclip_proc  = XCLIPProcessor.from_pretrained(model_src)
            self.xclip_model = XCLIPModel.from_pretrained(model_src)
            self.xclip_model.eval()

            self.xclip_ok = True
            cb("✓ X-CLIP temporal encoder loaded.", "success")

        except Exception as e:
            cb(f"✗ X-CLIP load failed: {e}", "warn")
            cb("  Layer 2 will use optical flow fallback.", "muted")
            self.xclip_ok = False

    # ── Layer 3: Wav2Vec2 Audio Encoder ───────────────────────────────────────
    def _load_wav2vec2(self, cb):
        cb("Loading Layer 3 — Wav2Vec2 audio encoder...", "info")
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2Model

            path = self._model_path("wav2vec2")
            model_src = path if path else "facebook/wav2vec2-base-960h"

            if path:
                cb(f"  Loading from local disk: {path}", "muted")
            else:
                cb("  Local model not found — downloading from HuggingFace...", "warn")

            self.wav2vec2_proc  = Wav2Vec2Processor.from_pretrained(model_src)
            self.wav2vec2_model = Wav2Vec2Model.from_pretrained(model_src)
            self.wav2vec2_model.eval()

            self.wav2vec2_ok = True
            cb("✓ Wav2Vec2 audio encoder loaded.", "success")

        except Exception as e:
            cb(f"✗ Wav2Vec2 load failed: {e}", "warn")
            cb("  Layer 3 will use peak-matching fallback.", "muted")
            self.wav2vec2_ok = False

    # ── Status ─────────────────────────────────────────────────────────────────
    def status_text(self) -> str:
        def _s(ok): return "✓ Loaded" if ok else "○ Not loaded"
        return (f"Spatial:   {_s(self.spatial_ok)}\n"
                f"X-CLIP:    {_s(self.xclip_ok)}\n"
                f"Wav2Vec2:  {_s(self.wav2vec2_ok)}")

    @property
    def wav2vec2(self) -> Tuple[Any, Any]:
        return self.wav2vec2_model, self.wav2vec2_proc

    @property
    def xclip(self) -> Tuple[Any, Any]:
        return self.xclip_model, self.xclip_proc
