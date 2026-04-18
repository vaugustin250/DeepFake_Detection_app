"""
screen_capture.py
Lightweight screen capture utility for the Live Detection tab.
"""

from __future__ import annotations

from typing import Optional

import time
import numpy as np
import cv2


class ScreenCapture:
    """Simple screen capture wrapper using Pillow ImageGrab."""

    def __init__(self, target_fps: float = 25.0):
        self.active = False
        self._target_fps = max(1.0, float(target_fps))
        self._last_ts = 0.0

    def start(self) -> None:
        self.active = True
        self._last_ts = 0.0

    def stop(self) -> None:
        self.active = False

    def read(self) -> Optional[np.ndarray]:
        if not self.active:
            return None

        # Basic frame pacing to avoid excessive CPU usage.
        now = time.time()
        if self._last_ts:
            min_dt = 1.0 / self._target_fps
            if now - self._last_ts < min_dt:
                return None
        self._last_ts = now

        try:
            from PIL import ImageGrab

            try:
                img = ImageGrab.grab(all_screens=True)
            except TypeError:
                img = ImageGrab.grab()

            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            return frame
        except Exception:
            return None
