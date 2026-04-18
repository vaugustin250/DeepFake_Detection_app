"""
core/temporal_detector.py  — ForensicStream AI  v4
=====================================================
Layer 2: Temporal Artifact Detection  (weight 35%)

WHAT WAS WRONG:
  flow_var <= 3.5 → flow_real = 1.0  (full real score)
  BUT smooth low-variance flow is EXACTLY what deepfakes produce.
  A person talking naturally has smooth face motion → always scored as real.
  The logic was backwards.

REPLACEMENT:
  1. X-CLIP zero-shot (when loaded)          [45%]
  2. Face-vs-background motion decoupling    [35% / 50% no model]
     Real: face+bg move together (camera). Deepfake: face rendered independently.
  3. Face region temporal flicker            [25% / 35%]
     GAN generates each frame independently → subtle per-frame colour flicker.
  4. Eye blink naturalness                   [15% / bonus]
     Some deepfakes have no blinks or robotically periodic blinks.
"""

import cv2
import numpy as np
from collections import deque
from typing import Dict, Any, Optional, Tuple

XCLIP_REAL = ["a real authentic face video",
               "genuine human face footage",
               "real person speaking naturally"]
XCLIP_FAKE = ["a deepfake AI generated face video",
               "fake synthetic face footage",
               "manipulated AI face swapped video"]


class TemporalDetector:

    XCLIP_CLIP = 8
    WINDOW     = 20

    def __init__(self):
        self._bgr_buf:     deque = deque(maxlen=self.WINDOW)
        self._gray_buf:    deque = deque(maxlen=self.WINDOW)
        self._clip_buf:    deque = deque(maxlen=self.XCLIP_CLIP)
        self._face_colors: deque = deque(maxlen=self.WINDOW)
        self._eye_bright:  deque = deque(maxlen=self.WINDOW)

        self._xclip_model = None
        self._xclip_proc  = None
        self.xclip_loaded = False
        self._last_xclip  = 50.0

    def inject_registry(self, registry):
        if registry.xclip_ok:
            self._xclip_model = registry.xclip_model
            self._xclip_proc  = registry.xclip_proc
            self.xclip_loaded = True

    # ─── Main ───────────────────────────────────────────────────────────────────
    def analyse(self, bgr: np.ndarray,
                face_bbox: Optional[Tuple] = None) -> Dict[str, Any]:
        from PIL import Image
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        self._bgr_buf.append(bgr); self._gray_buf.append(gray)
        rgb224 = cv2.cvtColor(cv2.resize(bgr,(224,224)), cv2.COLOR_BGR2RGB)
        self._clip_buf.append(Image.fromarray(rgb224))

        if face_bbox is not None:
            x,y,w,h = face_bbox
            face = bgr[max(0,y):min(bgr.shape[0],y+h),
                       max(0,x):min(bgr.shape[1],x+w)]
            if face.size > 0:
                self._face_colors.append(face.mean(axis=(0,1)).astype(float))
            ey1=max(0,y+int(h*0.24)); ey2=min(bgr.shape[0],y+int(h*0.46))
            ex1=max(0,x+int(w*0.05)); ex2=min(bgr.shape[1],x+int(w*0.95))
            eye = bgr[ey1:ey2,ex1:ex2]
            if eye.size > 0:
                self._eye_bright.append(
                    float(cv2.cvtColor(eye,cv2.COLOR_BGR2GRAY).mean()))

        if len(self._bgr_buf) < 3:
            return {"temporal_score":45.0,"flow_variance":0.0,
                    "frame_entropy":0.0,"motion_consistency":45.0,"xclip_score":45.0}

        xclip    = self._xclip_classify()
        decouple = self._face_bg_decoupling(face_bbox)
        flicker  = self._face_flicker()
        blink    = self._blink_naturalness()

        if self.xclip_loaded:
            t = (xclip*0.45 + decouple*0.30 + flicker*0.20 + blink*0.05) * 100
        else:
            t = (decouple*0.50 + flicker*0.35 + blink*0.15) * 100

        t = float(np.clip(t, 0, 100))
        return {
            "temporal_score":     round(t, 2),
            "xclip_score":        round(xclip*100, 2),
            "flow_variance":      round(1.0-decouple, 4),
            "frame_entropy":      round(1.0-flicker, 4),
            "motion_consistency": round(decouple*100, 2),
        }

    # ─── Signal 1: X-CLIP ───────────────────────────────────────────────────────
    def _xclip_classify(self) -> float:
        if not self.xclip_loaded or len(self._clip_buf) < self.XCLIP_CLIP:
            return self._last_xclip / 100.0
        try:
            import torch
            frames = list(self._clip_buf)
            all_texts = XCLIP_REAL + XCLIP_FAKE
            inputs = self._xclip_proc(text=all_texts, videos=[frames],
                                       return_tensors="pt", padding=True)
            with torch.no_grad():
                out = self._xclip_model(**inputs)
            probs = torch.softmax(out.logits_per_video, dim=-1).squeeze(0)
            score = float(probs[:len(XCLIP_REAL)].sum() /
                          probs.sum().clamp(min=1e-6))
            self._last_xclip = score * 100.0
            return score
        except Exception as e:
            print(f"[Temporal] X-CLIP error: {e}")
            return self._last_xclip / 100.0

    # ─── Signal 2: Face vs background motion decoupling ─────────────────────────
    def _face_bg_decoupling(self, face_bbox: Optional[Tuple]) -> float:
        """
        Real video: face and background share camera motion (correlated).
        Deepfake:   face rendered independently → motion ratio abnormal.
        Returns 0–1 (1 = natural correlation = likely real).
        """
        if len(self._gray_buf) < 2 or face_bbox is None: return 0.60
        try:
            prev=self._gray_buf[-2]; curr=self._gray_buf[-1]
            if prev.shape != curr.shape: return 0.60
            h, w = curr.shape
            x,y,fw,fh = face_bbox
            fy1=max(0,y); fy2=min(h,y+fh); fx1=max(0,x); fx2=min(w,x+fw)

            face_diff = float(cv2.absdiff(prev[fy1:fy2,fx1:fx2],
                                           curr[fy1:fy2,fx1:fx2]).mean())
            bp=prev.copy(); bc=curr.copy()
            bp[fy1:fy2,fx1:fx2]=128; bc[fy1:fy2,fx1:fx2]=128
            bg_diff = float(cv2.absdiff(bp,bc).mean())

            if bg_diff < 0.1: return 0.65   # static scene
            ratio = face_diff / (bg_diff + 1e-6)

            # Real: ratio 0.4–2.0 (face moves similarly to background)
            # Deepfake flicker: ratio > 2.5 (face moves independently)
            # Deepfake frozen:  ratio < 0.3 (face completely static vs moving bg)
            if 0.4 <= ratio <= 2.0:
                return 0.70 + 0.25*(1.0 - abs(ratio-1.0)/1.6)
            elif ratio > 2.0:
                return float(np.clip(1.0-(ratio-2.0)/3.0, 0.10, 0.70))
            else:
                return float(np.clip(ratio/0.4*0.5, 0.10, 0.60))
        except Exception as e:
            print(f"[Temporal] decoupling error: {e}"); return 0.60

    # ─── Signal 3: Face region flickering ──────────────────────────────────────
    def _face_flicker(self) -> float:
        """
        GAN generates each frame independently → subtle per-frame colour
        flicker in face region.
        Real face: brightness std across 20 frames < 3.
        Deepfake:  brightness std > 6.
        Returns 0–1 (1 = stable = likely real).
        """
        if len(self._face_colors) < 5: return 0.65
        try:
            colors = np.array(list(self._face_colors))
            bright_std  = float(colors.mean(axis=1).std())
            channel_std = float(colors.std(axis=0).max())

            bright_fake = float(np.clip((bright_std  - 2.5)/6.0, 0.0, 1.0))
            chan_fake    = float(np.clip((channel_std - 4.0)/8.0, 0.0, 1.0))
            fake_prob    = bright_fake*0.6 + chan_fake*0.4
            return round(1.0 - fake_prob, 4)
        except Exception as e:
            print(f"[Temporal] flicker error: {e}"); return 0.65

    # ─── Signal 4: Eye blink naturalness ───────────────────────────────────────
    def _blink_naturalness(self) -> float:
        """
        Some deepfakes have no blinks (GAN trained on open-eye data)
        or unnaturally static eye regions.
        Returns 0–1 (1 = natural blink pattern = likely real).
        """
        if len(self._eye_bright) < 8: return 0.65
        try:
            arr  = np.array(list(self._eye_bright))
            mean = arr.mean()
            dips = int(np.sum(arr < mean*0.88))
            n    = len(arr)
            no_blink_fake = 0.45 if (dips == 0 and n >= 12) else 0.0
            too_many_fake = 0.35 if dips > n*0.3 else 0.0
            static_fake   = 0.40 if arr.std() < 1.2 else 0.0
            fake_prob = max(no_blink_fake, too_many_fake, static_fake)
            return round(1.0 - fake_prob, 4)
        except Exception as e:
            print(f"[Temporal] blink error: {e}"); return 0.65
