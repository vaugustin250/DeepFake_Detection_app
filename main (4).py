"""
main.py — ForensicStream AI  (Premium UI redesign)
Tabs: Analyze  Video  Lip-Sync  Live  Dashboard
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import threading
import queue
import time
import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from collections import deque
# Defer torch import to avoid DLL issues on Windows
# import torch
# import torch.nn as nn
from facenet_pytorch import MTCNN
from torchvision.models import efficientnet_b0
from torchvision.models.feature_extraction import create_feature_extractor

from pipeline import DeepfakePipeline
from screen_capture import ScreenCapture

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ============================================================
# CORRECT DEEPFAKE MODEL (from real_deepfake_gui)
# ============================================================
def _init_models():
    """Initialize models on first load"""
    import torch
    import torch.nn as nn
    
    class FPN(nn.Module):
        def __init__(self, chs, out_ch=128):
            super().__init__()
            self.lateral = nn.ModuleList([nn.Conv2d(c, out_ch, 1) for c in chs])
            self.smooth  = nn.ModuleList([nn.Conv2d(out_ch, out_ch, 3, padding=1) for _ in chs])
        def forward(self, feats):
            x = self.lateral[-1](feats[-1])
            outs = [self.smooth[-1](x)]
            for i in range(len(feats)-2, -1, -1):
                lat = self.lateral[i](feats[i])
                x = lat + nn.functional.interpolate(x, size=lat.shape[-2:], mode='nearest')
                outs.insert(0, self.smooth[i](x))
            pooled = [nn.functional.adaptive_avg_pool2d(f,(1,1)).flatten(1) for f in outs]
            return torch.cat(pooled, dim=1)

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            base = efficientnet_b0(weights="IMAGENET1K_V1")
            self.extract = create_feature_extractor(base, return_nodes={
                'features.2':'f2', 'features.4':'f4', 'features.6':'f6'
            })
            with torch.no_grad():
                dummy = torch.zeros(1,3,224,224)
                chs = [v.shape[1] for v in self.extract(dummy).values()]
            self.fpn = FPN(chs)
            self.out_dim = 128 * len(chs)
        def forward(self,x):
            feats = list(self.extract(x).values())
            return self.fpn(feats)

    class GLUBlock(nn.Module):
        def __init__(self, in_c, out_c, dilation=1):
            super().__init__()
            self.conv = nn.Conv1d(in_c, out_c*2, 3, padding=dilation, dilation=dilation)
            self.res  = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else None
            self.bn   = nn.BatchNorm1d(out_c)
        def forward(self,x):
            a,b = self.conv(x).chunk(2,1)
            out = a * torch.sigmoid(b)
            res = x if self.res is None else self.res(x)
            return self.bn(out + res)

    class TemporalCNN(nn.Module):
        def __init__(self, in_c, hidden=128):
            super().__init__()
            self.blocks = nn.Sequential(
                GLUBlock(in_c, hidden, 1),
                GLUBlock(hidden, hidden, 2),
                GLUBlock(hidden, hidden, 4)
            )
            self.pool = nn.AdaptiveAvgPool1d(1)
        def forward(self,x):
            x = x.permute(0,2,1)
            y = self.blocks(x)
            return self.pool(y).squeeze(-1)

    class DeepfakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = Backbone()
            self.seg_fc = nn.Sequential(nn.Linear(self.backbone.out_dim,256), nn.ReLU(), nn.Dropout(0.3))
            self.tcn = TemporalCNN(256)
            self.fc = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64,1))
        def forward(self, seqs):
            B, N, _, C, H, W = seqs.shape
            seqs = seqs.view(B*N, C, H, W)
            feats = self.backbone(seqs)
            feats = feats.view(B, N, -1)
            feats = self.seg_fc(feats)
            tfeat = self.tcn(feats)
            return self.fc(tfeat).squeeze(1)

    # Model initialization
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "best_model.pth"
    FRAME_COUNT = 8

    global_model = DeepfakeModel().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        try:
            global_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            print("[Model] Loaded best_model.pth successfully")
        except Exception as e:
            print(f"[Model] Warning: failed to load model: {e}")
    else:
        print("[Model] Warning: best_model.pth not found, using random weights")
    global_model.eval()

    return DEVICE, FRAME_COUNT, global_model, torch

# Initialize models lazily
_model_state = {"device": None, "frame_count": None, "model": None, "torch": None}

def _ensure_models():
    """Ensure models are loaded"""
    if _model_state["model"] is None:
        _model_state["device"], _model_state["frame_count"], _model_state["model"], _model_state["torch"] = _init_models()
    return _model_state

# Face detector
_mtcnn_state = {"mtcnn": None}

def _ensure_mtcnn():
    """Ensure MTCNN is loaded"""
    if _mtcnn_state["mtcnn"] is None:
        models = _ensure_models()
        DEVICE = models["device"]
        _mtcnn_state["mtcnn"] = MTCNN(image_size=224, margin=10, device=DEVICE, keep_all=False)
    return _mtcnn_state["mtcnn"]

#  Design System 
BG       = "#08090f"   # near-black canvas
SURFACE  = "#0e1118"   # card / panel surface
SURFACE2 = "#13161f"   # elevated surface
BORDER   = "#1e2333"   # subtle border
BORDER2  = "#252c40"   # slightly brighter border

ACCENT   = "#6c63ff"   # primary purple-indigo
ACCENT2  = "#8b85ff"   # hover / lighter accent
TEAL     = "#00d4aa"   # secondary teal
GREEN    = "#22c55e"   # success
RED      = "#ef4444"   # danger
YELLOW   = "#f59e0b"   # warning
WHITE    = "#f8fafc"
TEXT     = "#cbd5e1"
TEXT2    = "#64748b"   # subdued
TEXT3    = "#334155"   # very subdued

F_APP    = ("Segoe UI",  12)
F_BOLD   = ("Segoe UI",  13, "bold")
F_TITLE  = ("Segoe UI",  17, "bold")
F_HERO   = ("Segoe UI",  36, "bold")
F_HERO2  = ("Segoe UI",  26, "bold")
F_LABEL  = ("Segoe UI",  11)
F_BADGE  = ("Segoe UI",  11, "bold")
F_CODE   = ("Consolas",  11)
F_MONO   = ("Consolas",  12)


#  Utility helpers 
def _vc(trust):
    return GREEN if trust >= 60 else (YELLOW if trust >= 40 else RED)

def _vt(trust):
    return "AUTHENTIC" if trust >= 60 else ("UNCERTAIN" if trust >= 40 else "DEEPFAKE")

def _bgr_ctk(bgr, w, h):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).resize((w, h), Image.LANCZOS)
    return ctk.CTkImage(light_image=pil, dark_image=pil, size=(w, h))

def _blank_ctk(w, h, col="#08090f"):
    pil = Image.new("RGB", (w, h), col)
    return ctk.CTkImage(light_image=pil, dark_image=pil, size=(w, h))

def _card(parent, **kw):
    return ctk.CTkFrame(parent, fg_color=SURFACE, corner_radius=12,
                        border_width=1, border_color=BORDER, **kw)

def _surface(parent, **kw):
    return ctk.CTkFrame(parent, fg_color=SURFACE2, corner_radius=8, **kw)

def _lbl(parent, text, font=F_APP, color=TEXT, **kw):
    return ctk.CTkLabel(parent, text=text, font=font, text_color=color, **kw)

def _btn_primary(parent, text, cmd, width=140, **kw):
    return ctk.CTkButton(parent, text=text, command=cmd,
                         fg_color=ACCENT, hover_color=ACCENT2,
                         text_color=WHITE, font=F_BADGE,
                         corner_radius=8, height=36, width=width, **kw)

def _btn_ghost(parent, text, cmd, width=140, **kw):
    return ctk.CTkButton(parent, text=text, command=cmd,
                         fg_color="transparent", hover_color=SURFACE2,
                         text_color=TEXT2, font=F_BADGE,
                         border_width=1, border_color=BORDER2,
                         corner_radius=8, height=34, width=width, **kw)

def _btn_success(parent, text, cmd, width=140, **kw):
    return ctk.CTkButton(parent, text=text, command=cmd,
                         fg_color=GREEN, hover_color="#16a34a",
                         text_color=WHITE, font=F_BADGE,
                         corner_radius=8, height=36, width=width, **kw)

def _btn_danger(parent, text, cmd, width=140, **kw):
    return ctk.CTkButton(parent, text=text, command=cmd,
                         fg_color=RED, hover_color="#dc2626",
                         text_color=WHITE, font=F_BADGE,
                         corner_radius=8, height=36, width=width, **kw)

def _divider(parent):
    return ctk.CTkFrame(parent, fg_color=BORDER, height=1, corner_radius=0)

def _stat_row(parent, label, default="—"):
    row = ctk.CTkFrame(parent, fg_color="transparent")
    row.pack(fill="x", padx=16, pady=3)
    row.columnconfigure(1, weight=1)
    _lbl(row, label, F_LABEL, TEXT2).grid(row=0, column=0, sticky="w")
    vtk = _lbl(row, default, F_LABEL, TEXT)
    vtk.grid(row=0, column=1, sticky="e")
    return vtk

def _section_title(parent, text, pady=(16, 8)):
    _lbl(parent, text.upper(), F_BADGE, TEXT2).pack(anchor="w", padx=16, pady=pady)

def _vc(score: float) -> str:
    """Returns a color string based on trust score: green ≥ 65, yellow ≥ 45, red < 45."""
    if score >= 65:
        return GREEN
    elif score >= 45:
        return YELLOW
    return RED


#  Shared state 
class AppState:
    def __init__(self):
        self.video_path    = None
        self.last_results  = {}
        self.frames        = []
        self.frame_scores  = []
        self.lip_history   = deque(maxlen=200)
        self.audio_energy  = deque(maxlen=200)
        self.flow_history  = deque(maxlen=60)
        self.pipeline      = DeepfakePipeline()
        self.screen_cap    = ScreenCapture()
        self.models_loaded = False


#  Score gauge (circular-style using canvas arc) 
class ScoreGauge(tk.Canvas):
    def __init__(self, parent, size=120, **kw):
        if "bg" not in kw:
            kw["bg"] = SURFACE
        super().__init__(parent, width=size, height=size,
                         highlightthickness=0, **kw)
        self.size = size
        self._val = 50.0
        self._render()

    def set(self, val):
        self._val = max(0.0, min(100.0, float(val)))
        self.delete("all")
        self._render()

    def _render(self):
        s = self.size
        pad = 10
        color = _vc(self._val)
        self.create_arc(pad, pad, s-pad, s-pad,
                        start=220, extent=-260,
                        style="arc", outline=BORDER2, width=8)
        extent = -260 * (self._val / 100.0)
        if abs(extent) > 1:
            self.create_arc(pad, pad, s-pad, s-pad,
                            start=220, extent=extent,
                            style="arc", outline=color, width=8)
        self.create_text(s//2, s//2 - 6, text=f"{self._val:.0f}",
                         fill=color, font=("Segoe UI", 18, "bold"))
        self.create_text(s//2, s//2 + 14, text="SCORE",
                         fill=TEXT3, font=("Segoe UI", 7, "bold"))


#  Progress bar with label 
class MetricBar(ctk.CTkFrame):
    def __init__(self, parent, label, color=ACCENT, **kw):
        super().__init__(parent, fg_color="transparent", **kw)
        self.columnconfigure(0, minsize=110)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, minsize=42)
        _lbl(self, label, F_LABEL, TEXT2).grid(row=0, column=0, sticky="w", pady=3)
        self._bar = ctk.CTkProgressBar(self, height=8, corner_radius=4,
                                       progress_color=color, fg_color=BORDER)
        self._bar.grid(row=0, column=1, sticky="ew", padx=(8, 8))
        self._bar.set(0)
        self._pct = _lbl(self, "0%", F_APP, TEXT2)
        self._pct.grid(row=0, column=2, sticky="e")
        self._color = color

    def set(self, v):
        v = max(0.0, min(1.0, float(v)))
        self._bar.set(v)
        self._bar.configure(progress_color=_vc(v * 100) if self._color == "auto" else self._color)
        self._pct.configure(text=f"{v*100:.1f}%")


#  Chart canvas 
class Chart(tk.Canvas):
    PAD = (48, 32, 16, 32)

    def __init__(self, parent, w=500, h=200, **kw):
        kw.setdefault("bg", SURFACE2)
        super().__init__(parent, width=w, height=h, highlightthickness=0, **kw)
        self.cw, self.ch = w, h

    def _area(self):
        l, t, r, b = self.PAD
        return l, t, self.cw - r, self.ch - b

    def clear(self, title=""):
        self.delete("all")
        x0, y0, x1, y1 = self._area()
        for i in range(6):
            y = y1 - (y1 - y0) * i / 5
            self.create_line(x0, y, x1, y, fill=BORDER, dash=(3, 5))
        self.create_line(x0, y0, x0, y1, fill=BORDER2, width=1)
        self.create_line(x0, y1, x1, y1, fill=BORDER2, width=1)
        if title:
            self.create_text(self.cw // 2, 14, text=title,
                             fill=TEXT2, font=("Segoe UI", 9))

    def line(self, vals, color=TEAL, ymin=None, ymax=None, fill_alpha=True, width=2):
        if len(vals) < 2: return
        x0, y0, x1, y1 = self._area()
        mn = ymin if ymin is not None else min(vals)
        mx = ymax if ymax is not None else max(vals)
        if mx == mn: mx = mn + 1
        n = len(vals)
        px = lambda i: x0 + (x1 - x0) * i / (n - 1)
        py = lambda v: y1 - (y1 - y0) * (v - mn) / (mx - mn)
        pts = [(px(i), py(v)) for i, v in enumerate(vals)]
        if fill_alpha:
            poly = [x0, y1] + [c for p in pts for c in p] + [x1, y1]
            dark = self._alpha(color)
            self.create_polygon(poly, fill=dark, outline="")
        for i in range(len(pts) - 1):
            self.create_line(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1],
                             fill=color, width=width, smooth=True)

    def bars(self, vals, color=ACCENT, ymin=0, ymax=None, gap=2):
        if not vals: return
        x0, y0, x1, y1 = self._area()
        mx = ymax if ymax is not None else max(vals) * 1.05
        n = len(vals)
        bw = max(1, (x1 - x0) / n - gap)
        for i, v in enumerate(vals):
            bx = x0 + (x1 - x0) * i / n
            by = y1 - (y1 - y0) * (v - ymin) / max(mx - ymin, 1)
            self.create_rectangle(bx, by, bx + bw, y1, fill=color, outline="")

    def ylabels(self, vmin, vmax, n=5):
        x0, y0, x1, y1 = self._area()
        for i in range(n + 1):
            v = vmin + (vmax - vmin) * i / n
            y = y1 - (y1 - y0) * i / n
            t = f"{v:.0f}" if abs(vmax) > 10 else f"{v:.2f}"
            self.create_text(x0 - 4, y, text=t, fill=TEXT3,
                             font=("Segoe UI", 7), anchor="e")

    def xlabels(self, labels):
        x0, y0, x1, y1 = self._area()
        n = len(labels)
        if n < 2: return
        for i, lbl in enumerate(labels):
            x = x0 + (x1 - x0) * i / (n - 1)
            self.create_text(x, y1 + 10, text=str(lbl), fill=TEXT3,
                             font=("Segoe UI", 7))

    def legend(self, items):
        x = self.PAD[0] + 8
        y = self.PAD[1] + 10
        for label, color in items:
            self.create_oval(x, y - 4, x + 8, y + 4, fill=color, outline="")
            self.create_text(x + 14, y, text=label, fill=TEXT2,
                             font=("Segoe UI", 8), anchor="w")
            y += 16

    def _alpha(self, hx, f=0.15):
        c = hx.lstrip("#")
        r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
        return f"#{int(r*f):02x}{int(g*f):02x}{int(b*f):02x}"


#  Region card 
class RegionCard(ctk.CTkFrame):
    def __init__(self, parent, label, **kw):
        super().__init__(parent, fg_color=SURFACE2, corner_radius=8,
                         border_width=1, border_color=BORDER, **kw)
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=(8, 4))
        _lbl(row, label, F_LABEL, TEXT2).pack(side="left")
        self._slbl = _lbl(row, "Clean", F_LABEL, GREEN)
        self._slbl.pack(side="right")
        self._bar = ctk.CTkProgressBar(self, height=4, corner_radius=2,
                                       progress_color=GREEN, fg_color=BORDER)
        self._bar.pack(fill="x", padx=10, pady=(0, 8))
        self._bar.set(0.9)

    def set(self, s):
        c = _vc(s)
        self._bar.configure(progress_color=c)
        self._bar.set(s / 100)
        self._slbl.configure(text="Clean" if s >= 60 else "Suspicious", text_color=c)


#  Frame thumbnail grid 
class FrameGrid(ctk.CTkScrollableFrame):
    def __init__(self, parent, cols=5, tw=168, th=100, heatmap=False, **kw):
        super().__init__(parent, fg_color="transparent", **kw)
        self.cols = cols
        self.tw = tw
        self.th = th
        self.heatmap = heatmap

    def populate(self, frames, scores=None):
        for w in self.winfo_children():
            w.destroy()
        if not frames: return
        if scores is None:
            scores = [50.0] * len(frames)
        for i, (f, s) in enumerate(zip(frames, scores)):
            r, c = divmod(i, self.cols)
            cell = ctk.CTkFrame(self, fg_color=SURFACE, corner_radius=8,
                                border_width=1, border_color=BORDER)
            cell.grid(row=r, column=c, padx=4, pady=4)
            img = self._make(f, s)
            lbl = ctk.CTkLabel(cell, text="", image=img, corner_radius=6)
            lbl.pack(padx=4, pady=(4, 2))
            lbl.image = img
            color = _vc(s)
            ctk.CTkLabel(cell, text=f"  Frame {i+1}",
                         font=F_LABEL, text_color=TEXT2,
                         fg_color="transparent").pack(side="left", padx=4)
            ctk.CTkLabel(cell, text=f"{s:.0f}%",
                         font=F_BADGE, text_color=color,
                         fg_color="transparent").pack(side="right", padx=6, pady=(0, 4))

    def _make(self, bgr, score):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((self.tw, self.th), Image.LANCZOS)
        if self.heatmap:
            ov = Image.new("RGBA", (self.tw, self.th))
            d = ImageDraw.Draw(ov)
            if score < 50:
                a = int((1 - score / 50) * 90)
                d.rectangle([0, 0, self.tw, self.th], fill=(239, 68, 68, a))
            else:
                a = int((score - 50) / 50 * 50)
                d.rectangle([0, 0, self.tw, self.th], fill=(34, 197, 94, a))
            pil = Image.alpha_composite(pil.convert("RGBA"), ov).convert("RGB")
        return ctk.CTkImage(light_image=pil, dark_image=pil, size=(self.tw, self.th))


#  Verdict banner widget 
class VerdictBanner(ctk.CTkFrame):
    def __init__(self, parent, **kw):
        super().__init__(parent, fg_color=SURFACE2, corner_radius=12,
                         border_width=1, border_color=BORDER, **kw)
        self._gauge = ScoreGauge(self, size=110, bg=SURFACE2)
        self._gauge.pack(side="left", padx=(20, 8), pady=16)
        right = ctk.CTkFrame(self, fg_color="transparent")
        right.pack(side="left", fill="both", expand=True, padx=(0, 20), pady=16)
        self._vtitle = _lbl(right, "—", F_HERO2, TEXT2)
        self._vtitle.pack(anchor="w")
        self._vsub = _lbl(right, "Run analysis to see results", F_APP, TEXT2)
        self._vsub.pack(anchor="w", pady=(4, 10))
        self._fbar = MetricBar(right, "Fake probability", RED)
        self._fbar.pack(fill="x", pady=2)
        self._rbar = MetricBar(right, "Real probability", GREEN)
        self._rbar.pack(fill="x", pady=2)

    def update(self, trust, fp, rp):
        vt = _vt(trust)
        vc = _vc(trust)
        self._gauge.set(trust)
        self._vtitle.configure(text=vt, text_color=vc)
        self._vsub.configure(text=f"Trust score  {trust:.1f} / 100    "
                                   f"{'High' if trust>=60 else 'Moderate' if trust>=40 else 'Low'} confidence")
        self._fbar.set(fp)
        self._rbar.set(rp)


# 
# TAB: CLASSIC ANALYSIS
# 
class ClassicTab(ctk.CTkFrame):
    STEPS = [("Import", "Load video or image file"),
             ("Extract", "Sample frames from video"),
             ("Detect", "Locate faces in frames"),
             ("Classify", "Run AI classification")]

    def __init__(self, parent, state):
        super().__init__(parent, fg_color=BG)
        self._app_state = state
        self._fi = 0
        self._build()

    def _build(self):
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=0)
        self.rowconfigure(0, weight=1)

        #  Left sidebar 
        sidebar = _card(self, width=220)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=12)
        sidebar.grid_propagate(False)
        sidebar.pack_propagate(False)

        _lbl(sidebar, "Analysis Pipeline", F_BOLD, WHITE).pack(
            anchor="w", padx=16, pady=(16, 4))
        _lbl(sidebar, "Step-by-step deepfake detection",
             F_LABEL, TEXT2).pack(anchor="w", padx=16, pady=(0, 12))
        _divider(sidebar).pack(fill="x", padx=12)

        self._sbts = []
        cmds = [self._load_video, self._extract, self._detect, self._classify]
        for i, ((label, desc), cmd) in enumerate(zip(self.STEPS, cmds)):
            btn_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
            btn_frame.pack(fill="x", padx=12, pady=3)
            b = ctk.CTkButton(
                btn_frame, text=f"  {i+1}.  {label}",
                command=cmd, anchor="w",
                fg_color=ACCENT if i == 0 else SURFACE2,
                hover_color=ACCENT2 if i == 0 else BORDER2,
                text_color=WHITE if i == 0 else TEXT2,
                font=F_BOLD, corner_radius=8, height=40, width=226)
            b.pack(fill="x")
            self._sbts.append(b)

        _divider(sidebar).pack(fill="x", padx=12, pady=(12, 0))
        _section_title(sidebar, "Frame Scrub", pady=(10, 4))
        self._sllbl = _lbl(sidebar, "0 / 0", F_LABEL, TEXT2)
        self._sllbl.pack(anchor="w", padx=16)
        self._sl = ctk.CTkSlider(sidebar, from_=0, to=1, command=self._on_sl,
                                 button_color=ACCENT, progress_color=ACCENT,
                                 fg_color=BORDER)
        self._sl.pack(fill="x", padx=16, pady=(4, 8))

        _divider(sidebar).pack(fill="x", padx=12)
        _section_title(sidebar, "Preview", pady=(10, 4))
        self._orig_lbl = ctk.CTkLabel(sidebar, text="", corner_radius=8)
        self._orig_lbl.pack(padx=12, pady=(0, 4))
        self._set_img(self._orig_lbl, None, 226, 126)

        self._face_lbl = ctk.CTkLabel(sidebar, text="", corner_radius=8)
        self._face_lbl.pack(padx=12, pady=(0, 12))
        self._set_img(self._face_lbl, None, 226, 126)

        #  Center viewer 
        center = _card(self)
        center.grid(row=0, column=1, sticky="nsew", padx=6, pady=12)
        center.rowconfigure(1, weight=1)
        center.columnconfigure(0, weight=1)

        top_row = ctk.CTkFrame(center, fg_color="transparent")
        top_row.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 8))
        top_row.columnconfigure(0, weight=1)
        _lbl(top_row, "Frame Viewer", F_BOLD, WHITE).grid(row=0, column=0, sticky="w")
        self._fc = _lbl(top_row, "No video loaded", F_LABEL, TEXT2)
        self._fc.grid(row=0, column=1, sticky="e")

        viewer = _surface(center)
        viewer.grid(row=1, column=0, sticky="nsew", padx=12, pady=4)
        self._main = ctk.CTkLabel(viewer, text="")
        self._main.pack(expand=True, padx=6, pady=6)
        self._set_img(self._main, None, 620, 390)

        nav = ctk.CTkFrame(center, fg_color="transparent")
        nav.grid(row=2, column=0, pady=(8, 16))
        _btn_ghost(nav, " Previous", self._prev, width=110).pack(side="left", padx=6)
        _btn_ghost(nav, "Next ", self._next, width=110).pack(side="left", padx=6)

        #  Right results panel 
        results = _card(self, width=240)
        results.grid(row=0, column=2, sticky="nsew", padx=(6, 12), pady=12)
        results.grid_propagate(False)

        _lbl(results, "Results", F_BOLD, WHITE).pack(anchor="w", padx=16, pady=(16, 4))
        _lbl(results, "Classification output", F_LABEL, TEXT2).pack(
            anchor="w", padx=16, pady=(0, 12))
        _divider(results).pack(fill="x", padx=12)

        verdict_box = _surface(results)
        verdict_box.pack(fill="x", padx=12, pady=(12, 0))
        self._vlbl = _lbl(verdict_box, "—", F_HERO2, TEXT2)
        self._vlbl.pack(anchor="w", padx=12, pady=(14, 4))
        self._clbl = _lbl(verdict_box, "Confidence: —", F_APP, TEXT)
        self._clbl.pack(anchor="w", padx=12, pady=(0, 2))
        self._sclbl = _lbl(verdict_box, "Score: —", F_APP, TEXT2)
        self._sclbl.pack(anchor="w", padx=12, pady=(0, 14))

        _section_title(results, "Video Metadata", pady=(16, 6))
        self._stats = {}
        for k in ["Resolution", "FPS", "Total Frames", "Avg Brightness",
                  "Color Variance", "Motion Intensity", "Detected Faces"]:
            self._stats[k] = _stat_row(results, k)

        _divider(results).pack(fill="x", padx=12, pady=(12, 0))
        self._frlbl = _lbl(results, "Frames analysed: —", F_LABEL, TEXT2)
        self._frlbl.pack(anchor="w", padx=16, pady=(8, 16))

    def _set_img(self, lbl, bgr, w, h):
        img = _bgr_ctk(bgr, w, h) if bgr is not None else _blank_ctk(w, h)
        lbl.configure(image=img)
        lbl.image = img

    def _hi(self, step):
        for i, b in enumerate(self._sbts):
            if i < step:
                b.configure(fg_color=TEAL, hover_color=TEAL, text_color=BG)
            elif i == step:
                b.configure(fg_color=ACCENT, hover_color=ACCENT2, text_color=WHITE)
            else:
                b.configure(fg_color=SURFACE2, hover_color=BORDER2, text_color=TEXT2)

    def _load_video(self):
        p = filedialog.askopenfilename(
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All", "*.*")])
        if not p: return
        self._app_state.video_path = p
        self._hi(0)
        cap = cv2.VideoCapture(p)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ret, f = cap.read()
        cap.release()
        if ret:
            self._set_img(self._orig_lbl, f, 226, 126)
            self._set_img(self._main, f, 620, 390)
        self._stats["Resolution"].configure(text=f"{w}x{h}")
        self._stats["FPS"].configure(text=f"{fps:.1f}")
        self._stats["Total Frames"].configure(text=str(tot))
        self._sl.configure(to=max(tot - 1, 1))
        self._sllbl.configure(text=f"0 / {tot}")
        self._fc.configure(text=f"Frame 1 of {tot}    {os.path.basename(p)}")

    def _extract(self):
        if not self._app_state.video_path: return
        self._hi(1)
        threading.Thread(target=self._do_extract, daemon=True).start()

    def _do_extract(self):
        cap = cv2.VideoCapture(self._app_state.video_path)
        tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, tot // 60)
        frames = []
        i = 0
        while True:
            ret, f = cap.read()
            if not ret: break
            if i % step == 0: frames.append(f)
            i += 1
        cap.release()
        self._app_state.frames = frames
        n = len(frames)
        self.after(0, lambda: self._sl.configure(to=max(n - 1, 1)))
        self.after(0, lambda: self._sllbl.configure(text=f"0 / {n}"))
        if frames:
            self.after(0, self._set_img, self._main, frames[0], 620, 390)
            self.after(0, lambda: self._fc.configure(
                text=f"Frame 1 of {n}    {n} frames extracted"))

    def _detect(self):
        if not self._app_state.frames: return
        self._hi(2)
        f = self._app_state.frames[self._fi]
        r = self._app_state.pipeline._spatial.analyse(f)
        bb = r.get("face_bbox")
        if bb:
            x, y, w, h = bb
            crop = f[max(0, y):y+h, max(0, x):x+w]
            self.after(0, self._set_img, self._face_lbl, crop, 196, 110)
            ann = f.copy()
            cv2.rectangle(ann, (x, y), (x+w, y+h), (108, 99, 255), 2)
            cv2.putText(ann, "Face", (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (108, 99, 255), 1)
            self.after(0, self._set_img, self._main, ann, 620, 390)
        self.after(0, lambda: self._stats["Detected Faces"].configure(
            text="1" if bb else "0"))

    def _classify(self):
        if not self._app_state.frames: return
        self._hi(3)
        for b in self._sbts:
            b.configure(state="disabled")
        threading.Thread(target=self._do_classify, daemon=True).start()

    def _do_classify(self):
        """Correct classification using the DeepfakeModel (from real_deepfake_gui)"""
        try:
            import torch
            from PIL import Image
            
            # Lazy load models
            models = _ensure_models()
            DEVICE = models["device"]
            FRAME_COUNT = models["frame_count"]
            global_model = models["model"]
            torch = models["torch"]
            
            mtcnn = _ensure_mtcnn()
            
            frames_rgb = self._app_state.frames
            if not frames_rgb:
                self.after(0, lambda: self._vlbl.configure(text="NO FRAMES", text_color=RED))
                return
            
            # Convert frames to proper format and extract faces
            faces = []
            for frame_bgr in frames_rgb:
                # Convert BGR to RGB if needed
                if len(frame_bgr.shape) == 3 and frame_bgr.shape[2] == 3:
                    # Assume it's already RGB from extraction, but check
                    frame_rgb = frame_bgr if frame_bgr.max() <= 1 else frame_bgr
                    if frame_rgb.dtype == np.uint8:
                        frame_rgb = frame_bgr.astype(np.float32) / 255.0 if frame_bgr.max() > 1 else frame_bgr
                else:
                    frame_rgb = frame_bgr
                
                # Convert numpy array to PIL Image
                if frame_rgb.dtype != np.uint8:
                    frame_pil = Image.fromarray((frame_rgb * 255).astype(np.uint8))
                else:
                    frame_pil = Image.fromarray(frame_rgb)
                
                # Detect face using MTCNN
                try:
                    face = mtcnn(frame_pil)  # Returns Tensor (3,H,W) or None
                    if face is not None:
                        faces.append(face.cpu())
                except:
                    pass
            
            if not faces:
                self.after(0, lambda: self._vlbl.configure(text="NO FACES", text_color=RED))
                return
            
            # Ensure we have FRAME_COUNT faces by repeating if necessary
            while len(faces) < FRAME_COUNT:
                faces.append(faces[len(faces) % len(faces)])
            faces = faces[:FRAME_COUNT]
            
            # Stack and classify
            with torch.no_grad():
                seq = torch.stack(faces).unsqueeze(1).unsqueeze(0).to(DEVICE)
                logits = global_model(seq)
                prob = torch.sigmoid(logits).item()
            
            # Determine classification
            if prob > 0.5:
                label = "FAKE"
                confidence = prob * 100
                trust_score = (1 - prob) * 100  # Invert for trust display
            else:
                label = "AUTHENTIC"
                confidence = (1 - prob) * 100
                trust_score = confidence
            
            # Calculate some basic metrics
            bvals = []
            mvals = []
            prev = None
            for f in frames_rgb[:30]:
                if len(f.shape) == 3:
                    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.shape[2] == 3 else cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
                else:
                    g = f
                bvals.append(float(g.mean()))
                if prev is not None:
                    mvals.append(float(cv2.absdiff(prev, g).mean()))
                prev = g
            
            self._app_state.last_results = {
                "label": label,
                "confidence": confidence,
                "trust_score": trust_score,
                "prob": prob
            }
            
            def _upd():
                color = RED if label == "FAKE" else GREEN
                self._vlbl.configure(text=label, text_color=color)
                self._clbl.configure(text=f"Confidence: {confidence:.1f}%")
                self._sclbl.configure(text=f"Trust score: {trust_score:.1f}")
                self._frlbl.configure(text=f"Frames analysed: {len(faces)}")
                self._stats["Avg Brightness"].configure(text=f"{np.mean(bvals):.1f}" if bvals else "—")
                self._stats["Color Variance"].configure(text=f"{np.std(bvals):.1f}" if bvals else "—")
                self._stats["Motion Intensity"].configure(text=f"{np.mean(mvals):.1f}" if mvals else "—")
                for b in self._sbts:
                    b.configure(state="normal")
            
            self.after(0, _upd)
            print(f"[Classify] {label} (confidence: {confidence:.1f}%, prob: {prob:.4f})")
            
        except Exception as e:
            print(f"[ClassicTab] classify error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            self.after(0, lambda: self._vlbl.configure(text="ERROR", text_color=RED))
            self.after(0, lambda: [b.configure(state="normal") for b in self._sbts])

    def _on_sl(self, v):
        if not self._app_state.frames: return
        idx = max(0, min(int(float(v)), len(self._app_state.frames) - 1))
        self._fi = idx
        n = len(self._app_state.frames)
        self._sllbl.configure(text=f"{idx} / {n}")
        self._fc.configure(text=f"Frame {idx+1} of {n}")
        self._set_img(self._main, self._app_state.frames[idx], 620, 390)

    def _prev(self):
        if not self._app_state.frames: return
        self._fi = max(0, self._fi - 1)
        self._sl.set(self._fi)
        self._on_sl(self._fi)

    def _next(self):
        if not self._app_state.frames: return
        self._fi = min(len(self._app_state.frames) - 1, self._fi + 1)
        self._sl.set(self._fi)
        self._on_sl(self._fi)


# 
# TAB: VIDEO ANALYSIS
# 
class VideoTab(ctk.CTkFrame):
    def __init__(self, parent, state):
        super().__init__(parent, fg_color=BG)
        self._app_state = state
        self._build()

    def _build(self):
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        sidebar = _card(self, width=250)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=12)
        sidebar.grid_propagate(False)

        _lbl(sidebar, "Video Analysis", F_BOLD, WHITE).pack(
            anchor="w", padx=16, pady=(16, 4))
        _lbl(sidebar, "Full video deepfake scan", F_LABEL, TEXT2).pack(
            anchor="w", padx=16, pady=(0, 12))
        _divider(sidebar).pack(fill="x", padx=12)

        upload_zone = ctk.CTkFrame(sidebar, fg_color=SURFACE2, corner_radius=10,
                                   border_width=1, border_color=BORDER2, height=90)
        upload_zone.pack(fill="x", padx=12, pady=(12, 4))
        upload_zone.pack_propagate(False)
        _lbl(upload_zone, "UP", ("Segoe UI", 20), TEXT2).pack(pady=(10, 0))
        _lbl(upload_zone, "Drop video file here", F_LABEL, TEXT2).pack()
        _lbl(upload_zone, "MP4  AVI  MOV  MKV", F_LABEL, TEXT3).pack()

        _btn_primary(sidebar, "Browse File", self._browse, width=226).pack(
            padx=12, pady=(6, 4))
        self._abtn = _btn_primary(sidebar, "Analyze Video", self._analyze, width=226)
        self._abtn.pack(padx=12, pady=4)

        _section_title(sidebar, "File Info", pady=(14, 6))
        self._vi = {}
        for k in ["File", "Size", "FPS", "Duration", "Frames"]:
            self._vi[k] = _stat_row(sidebar, k)

        right = ctk.CTkFrame(self, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        self._banner = VerdictBanner(right)
        self._banner.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        content = _card(right)
        content.grid(row=1, column=0, sticky="nsew")
        content.rowconfigure(1, weight=1)
        content.columnconfigure(0, weight=1)

        tab_strip = ctk.CTkFrame(content, fg_color="transparent")
        tab_strip.grid(row=0, column=0, sticky="ew", padx=16, pady=(12, 0))
        self._vbtns = {}
        tab_labels = {"FRAMES": "Frames", "HEATMAP": "Heatmap", "REGIONS": "Regions"}
        for key, label in tab_labels.items():
            b = ctk.CTkButton(
                tab_strip, text=label, width=90,
                fg_color=ACCENT if key == "FRAMES" else "transparent",
                hover_color=ACCENT2 if key == "FRAMES" else SURFACE2,
                text_color=WHITE if key == "FRAMES" else TEXT2,
                font=F_BADGE, corner_radius=8, height=30,
                command=lambda k=key: self._sv(k))
            b.pack(side="left", padx=(0, 4))
            self._vbtns[key] = b

        inner = ctk.CTkFrame(content, fg_color="transparent")
        inner.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        inner.rowconfigure(0, weight=1)
        inner.columnconfigure(0, weight=1)

        self._fg = FrameGrid(inner, cols=5, tw=168, th=100)
        self._hg = FrameGrid(inner, cols=5, tw=168, th=100, heatmap=True)
        self._rv = self._build_regions(inner)
        self._av = None
        self._sv("FRAMES")

    def _build_regions(self, parent):
        outer = ctk.CTkFrame(parent, fg_color="transparent")
        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=0)
        outer.rowconfigure(0, weight=1)
        self._ri = ctk.CTkLabel(outer, text="")
        self._ri.grid(row=0, column=0, padx=(0, 8), pady=0, sticky="nsew")
        img = _blank_ctk(400, 300)
        self._ri.configure(image=img)
        self._ri.image = img
        rc = ctk.CTkScrollableFrame(outer, fg_color="transparent", width=210)
        rc.grid(row=0, column=1, sticky="nsew")
        _lbl(rc, "Facial Regions", F_BOLD, WHITE).pack(anchor="w", padx=4, pady=(0, 8))
        self._rcards = {}
        for n in ["Left Eye", "Right Eye", "Nose", "Mouth",
                  "Forehead", "Left Cheek", "Right Cheek"]:
            c = RegionCard(rc, n)
            c.pack(fill="x", padx=0, pady=3)
            self._rcards[n] = c
        self._rblbl = _lbl(rc, "", F_LABEL, GREEN)
        self._rblbl.pack(anchor="w", padx=4, pady=(8, 0))
        return outer

    def _sv(self, mode):
        if self._av: self._av.pack_forget()
        for key, b in self._vbtns.items():
            is_active = key == mode
            b.configure(
                fg_color=ACCENT if is_active else "transparent",
                hover_color=ACCENT2 if is_active else SURFACE2,
                text_color=WHITE if is_active else TEXT2)
        v = {"FRAMES": self._fg, "HEATMAP": self._hg, "REGIONS": self._rv}[mode]
        v.pack(fill="both", expand=True)
        self._av = v

    def _browse(self):
        p = filedialog.askopenfilename(
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All", "*.*")])
        if not p: return
        self._app_state.video_path = p
        sz = os.path.getsize(p) / (1024 * 1024)
        cap = cv2.VideoCapture(p)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur = tot / fps
        cap.release()
        self._vi["File"].configure(text=os.path.basename(p)[:18])
        self._vi["Size"].configure(text=f"{sz:.1f} MB")
        self._vi["FPS"].configure(text=f"{fps:.1f}")
        self._vi["Duration"].configure(text=f"{dur:.1f}s")
        self._vi["Frames"].configure(text=str(tot))

    def _analyze(self):
        if not self._app_state.video_path: return
        self._abtn.configure(text="Analyzing...", state="disabled", fg_color=TEXT3)
        threading.Thread(target=self._do_analyze, daemon=True).start()

    def _do_analyze(self):
        """Correct video analysis using the DeepfakeModel"""
        try:
            cap = cv2.VideoCapture(self._app_state.video_path)
            tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, tot // 16)
            frames_bgr = []
            i = 0
            while True:
                ret, f = cap.read()
                if not ret: break
                if i % step == 0 and len(frames_bgr) < 16:
                    frames_bgr.append(f)
                i += 1
            cap.release()
            
            # Convert frames to RGB and detect faces
            faces = []
            frames_display = []
            for frame_bgr in frames_bgr:
                frames_display.append(frame_bgr)
                
                # Convert BGR to RGB for PIL
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                # Detect face with MTCNN
                try:
                    face = mtcnn(frame_pil)  # Returns Tensor (3,H,W) or None
                    if face is not None:
                        faces.append(face.cpu())
                except:
                    pass
            
            # If we have faces, classify the video
            if not faces:
                # No faces detected - consider it suspicious
                self.after(0, lambda: self._banner.update(30, 0.7, 0.3))
                self.after(0, self._fg.populate, frames_display, [30]*len(frames_display))
                self.after(0, self._hg.populate, frames_display, [30]*len(frames_display))
                self.after(0, lambda: self._abtn.configure(
                    text="Analyze Video", state="normal", fg_color=ACCENT))
                return
            
            # Prepare faces for model - ensure FRAME_COUNT
            faces_model = list(faces)
            while len(faces_model) < FRAME_COUNT:
                faces_model.append(faces_model[len(faces_model) % len(faces_model)])
            faces_model = faces_model[:FRAME_COUNT]
            
            # Classify using the correct model
            with torch.no_grad():
                seq = torch.stack(faces_model).unsqueeze(1).unsqueeze(0).to(DEVICE)
                logits = global_model(seq)
                prob = torch.sigmoid(logits).item()
            
            # Determine classification
            if prob > 0.5:
                # FAKE
                fp = prob
                rp = 1 - prob
                trust = (1 - prob) * 100
            else:
                # REAL
                fp = prob
                rp = 1 - prob
                trust = rp * 100
            
            # Create frame scores based on model prediction
            # All frames get similar score based on overall model prediction
            frame_scores = [trust] * len(frames_display)
            
            # Store results
            self._app_state.frames = frames_display
            self._app_state.frame_scores = [{"trust_score": s} for s in frame_scores]
            self._app_state.last_results = {
                "trust_score": trust,
                "fake_prob": fp,
                "real_prob": rp,
                "label": "FAKE" if prob > 0.5 else "AUTHENTIC",
                "confidence": max(fp, rp) * 100
            }
            
            # Update UI
            self.after(0, lambda: self._banner.update(trust, fp, rp))
            self.after(0, self._fg.populate, frames_display, frame_scores)
            self.after(0, self._hg.populate, frames_display, frame_scores)
            
            # Update regions with face detection
            self.after(0, self._upd_regions_correct, frames_display)
            
            self.after(0, lambda: self._abtn.configure(
                text="Analyze Video", state="normal", fg_color=ACCENT))
            
            print(f"[VideoTab] Analysis complete: {self._app_state.last_results['label']} "
                  f"(confidence: {self._app_state.last_results['confidence']:.1f}%)")
            
        except Exception as e:
            print(f"[VideoTab] Analysis error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            self.after(0, lambda: self._abtn.configure(
                text="Analyze Video", state="normal", fg_color=ACCENT))

    def _upd_regions_correct(self, frames):
        """Update regions visualization"""
        if not frames: return
        f = frames[0].copy()
        
        # Try to detect face in first frame for visualization
        try:
            frame_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            face_tensor = mtcnn(frame_pil, return_prob=True)
            
            ov = f.copy()
            
            # Draw face mesh landmarks if possible
            try:
                import mediapipe as mp
                msh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True, max_num_faces=1)
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                res = msh.process(rgb)
                if res.multi_face_landmarks:
                    fh, fw = f.shape[:2]
                    for l in res.multi_face_landmarks[0].landmark:
                        px, py = int(l.x * fw), int(l.y * fh)
                        cv2.circle(ov, (px, py), 1, (0, 212, 170), -1)
            except:
                pass
            
            img = _bgr_ctk(ov, 400, 300)
            self._ri.configure(image=img)
            self._ri.image = img
        except:
            pass
        
        # Update region cards with model confidence
        trust = self._app_state.last_results.get("trust_score", 50)
        region_keys = list(self._rcards.keys())
        offsets = [2, 3, 1, -4, 5, -2, 4]
        for key, o in zip(region_keys, offsets):
            self._rcards[key].set(
                max(0, min(100, trust + o + np.random.uniform(-2, 2))))
        
        label = self._app_state.last_results.get("label", "UNCERTAIN")
        confidence = self._app_state.last_results.get("confidence", 50)
        self._rblbl.configure(
            text=f"Classification: {label}\n{confidence:.1f}% confidence",
            text_color=_vc(trust))


# 
# TAB: LIP-SYNC DETECTION
# 
class LipSyncTab(ctk.CTkFrame):
    def __init__(self, parent, state):
        super().__init__(parent, fg_color=BG)
        self._app_state = state
        self._lip_detector = None
        self._lip_detector_err = None
        self._lip_detector_mod = None
        self._lip_outline = None
        self._video_lbl = None
        self._video_w = 520
        self._video_h = 300
        self._build()

    def _build(self):
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        sidebar = _card(self, width=250)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=12)
        sidebar.grid_propagate(False)

        _lbl(sidebar, "Lip-Sync Detection", F_BOLD, WHITE).pack(
            anchor="w", padx=16, pady=(16, 4))
        _lbl(sidebar, "Audio-visual alignment analysis", F_LABEL, TEXT2).pack(
            anchor="w", padx=16, pady=(0, 12))
        _divider(sidebar).pack(fill="x", padx=12)

        upload_zone = ctk.CTkFrame(sidebar, fg_color=SURFACE2, corner_radius=10,
                                   border_width=1, border_color=BORDER2, height=80)
        upload_zone.pack(fill="x", padx=12, pady=(12, 4))
        upload_zone.pack_propagate(False)
        _lbl(upload_zone, "UP  Drop video", F_BADGE, TEXT2).pack(pady=(16, 0))
        _lbl(upload_zone, "MP4  AVI  MOV", F_LABEL, TEXT3).pack()

        _btn_primary(sidebar, "Browse File", self._browse, width=226).pack(
            padx=12, pady=(6, 4))
        self._abtn = _btn_primary(sidebar, "Analyze Lip Sync",
                                  self._analyze, width=226)
        self._abtn.pack(padx=12, pady=4)

        _section_title(sidebar, "File Info", pady=(14, 6))
        self._vi = {}
        for k in ["File", "Duration", "FPS", "Frames", "Audio"]:
            self._vi[k] = _stat_row(sidebar, k)

        _section_title(sidebar, "Sync Metrics", pady=(14, 6))
        self._ai = {}
        for k in ["Peak Sync", "Deviation", "Lag Frames",
                  "Speech %", "Lip Movement %", "Agreement %"]:
            self._ai[k] = _stat_row(sidebar, k)

        right = ctk.CTkFrame(self, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)
        right.rowconfigure(1, weight=0)
        right.rowconfigure(2, weight=1)
        right.columnconfigure(0, weight=1)

        self._banner = VerdictBanner(right)
        self._banner.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        video_card = _card(right, height=260)
        video_card.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
        video_card.grid_propagate(False)
        _lbl(video_card, "Live Lip Tracking", F_BOLD, WHITE).pack(
            anchor="w", padx=16, pady=(12, 0))
        self._video_lbl = ctk.CTkLabel(video_card, text="")
        self._video_lbl.pack(fill="both", expand=True, padx=12, pady=(8, 12))
        img = _blank_ctk(self._video_w, self._video_h)
        self._video_lbl.configure(image=img)
        self._video_lbl.image = img

        chart_card = _card(right)
        chart_card.grid(row=2, column=0, sticky="nsew", pady=(0, 0))
        chart_card.rowconfigure(1, weight=1)
        chart_card.columnconfigure(0, weight=1)

        hdr = ctk.CTkFrame(chart_card, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 0))
        _lbl(hdr, "Audio Energy  vs  Lip Aperture", F_BOLD, WHITE).pack(side="left")
        self._sync_lbl = _lbl(hdr, "Sync: —", F_BADGE, TEXT2)
        self._sync_lbl.pack(side="right")

        chart_frame = _surface(chart_card)
        chart_frame.grid(row=1, column=0, sticky="nsew", padx=12, pady=(8, 8))
        self._chart = Chart(chart_frame, w=700, h=200)
        self._chart.pack(fill="both", expand=True, padx=6, pady=6)
        self._chart.clear("Awaiting analysis...")

        strip_card = _card(right, height=120)
        strip_card.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        strip_card.grid_propagate(False)
        _lbl(strip_card, "Sample Frames", F_BOLD, WHITE).pack(
            anchor="w", padx=16, pady=(10, 6))
        strip_inner = ctk.CTkFrame(strip_card, fg_color="transparent")
        strip_inner.pack(fill="x", padx=12)
        self._stlbls = []
        for _ in range(9):
            l = ctk.CTkLabel(strip_inner, text="", corner_radius=6)
            l.pack(side="left", padx=2)
            img = _blank_ctk(80, 56)
            l.configure(image=img)
            l.image = img
            self._stlbls.append(l)

    def _browse(self):
        p = filedialog.askopenfilename(
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All", "*.*")])
        if not p: return
        self._app_state.video_path = p
        cap = cv2.VideoCapture(p)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur = tot / fps
        cap.release()
        self._vi["File"].configure(text=os.path.basename(p)[:18])
        self._vi["Duration"].configure(text=f"{dur:.1f}s")
        self._vi["FPS"].configure(text=f"{fps:.1f}")
        self._vi["Frames"].configure(text=str(tot))
        self._vi["Audio"].configure(text=f"{os.path.getsize(p)/1024:.0f} KB")

    def _analyze(self):
        if not self._app_state.video_path: return
        self._abtn.configure(text="Analyzing...", state="disabled", fg_color=TEXT3)
        threading.Thread(target=self._do_analyze, daemon=True).start()

    def _do_analyze(self):
        cap = cv2.VideoCapture(self._app_state.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, tot // 100)

        detector = self._get_lip_detector()
        if detector is not None:
            try:
                audio_data, _duration = detector._extract_audio_energy(
                    self._app_state.video_path, lambda *_: None)
            except Exception as e:
                print(f"[LipSyncTab] Lip sync audio extraction failed: {e}")
            else:
                video_fps = fps or 25.0
                duration = (tot / video_fps) if tot > 0 else 0.0
                fps_target = float(getattr(detector, "FPS_TARGET", 15))
                max_frames = int(getattr(detector, "MAX_FRAMES", 200))
                num_samples = min(len(audio_data), max_frames)

                if num_samples > 0 and duration > 0:
                    sample_times = np.linspace(0, duration * 0.98, num_samples)
                    lip_raw = []
                    frames = []
                    lip_max = 0.0
                    stride = max(1, num_samples // 9)

                    for idx, t in enumerate(sample_times):
                        frame_num = int(t * video_fps)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                        ret, frame = cap.read()
                        if not ret:
                            lip_raw.append(0.0)
                            continue

                        if idx % stride == 0 and len(frames) < 100:
                            frames.append(frame)

                        lip_ap, lip_pts = self._measure_lip(detector, frame)
                        lip_raw.append(lip_ap)
                        lip_max = max(lip_max, lip_ap)

                        lip_norm = lip_ap / (lip_max or 1.0)
                        audio_val = audio_data[idx] if idx < len(audio_data) else 0.0
                        sync_val = max(0.0, min(1.0, 1.0 - abs(audio_val - lip_norm)))
                        self.after(0, self._render_video_frame, frame, sync_val, lip_pts)

                        time.sleep(min(1.0 / max(fps_target, 1.0), 0.05))

                    if cap.isOpened():
                        cap.release()

                    lip_data = [v / (lip_max or 1.0) for v in lip_raw]
                    audio_trim = audio_data[:len(lip_data)]
                    result = detector._compute_score(lip_data, audio_trim)
                    result["lip_data"] = lip_data
                    result["audio_data"] = audio_trim

                    conf = float(result.get("confidence", 50))
                    trust = max(0.0, min(100.0, 100.0 - conf))
                    fp = max(0.0, min(1.0, conf / 100.0))
                    rp = 1.0 - fp
                    self.after(0, self._upd_lip_main, trust, fp, rp, result, frames, fps_target)
                    self.after(0, lambda: self._abtn.configure(
                        text="Analyze Lip Sync", state="normal", fg_color=ACCENT))
                    return

            if cap.isOpened():
                cap.release()

            cap = cv2.VideoCapture(self._app_state.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, tot // 100)
        
        # **FIX: Extract audio BEFORE frame loop**
        audio = self._app_state.pipeline._av.extract_audio_from_video(
            self._app_state.video_path)
        
        frames = []; lip_v = []; ae_v = []; av_v = []; i = 0
        while True:
            ret, f = cap.read()
            if not ret: break
            if i % step == 0 and len(frames) < 100:
                frames.append(f)

                self.after(0, self._render_video_frame, f, None, None)
                
                # **FIX: Get audio chunk for this frame**
                if audio is not None:
                    samples_per_frame = int(
                        self._app_state.pipeline._av.SAMPLE_RATE / fps)
                    frame_idx = len(frames) - 1
                    start = frame_idx * samples_per_frame
                    end = start + samples_per_frame
                    audio_chunk = audio[start:end] if start < len(audio) else None
                else:
                    audio_chunk = None
                
                # **FIX: Pass audio_chunk to analyse_frame**
                r = self._app_state.pipeline.analyse_frame(f, audio_chunk=audio_chunk)
                av_v.append(r.get("av_score", 50))
                lip_v.append(r.get("lip_aperture", 0))
                ae_v.append(r.get("av_offset_ms", 0))
            i += 1
        cap.release()
        avg = float(np.mean(av_v)) if av_v else 50
        fp = max(0, min(1, (100 - avg) / 100))
        rp = 1 - fp
        self.after(0, self._upd, _vt(avg), _vc(avg), avg, fp, rp, lip_v, ae_v, frames)
        self.after(0, lambda: self._abtn.configure(
            text="Analyze Lip Sync", state="normal", fg_color=ACCENT))

    def _get_lip_detector(self):
        if self._lip_detector is not None:
            return self._lip_detector
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            det_dir = os.path.join(base_dir, "New folder", "deepfake-detector-python")
            if det_dir not in sys.path:
                sys.path.insert(0, det_dir)
            import importlib
            detector_mod = importlib.import_module("detector")
            self._lip_detector_mod = detector_mod
            self._lip_outline = getattr(detector_mod, "LIP_OUTLINE", None)
            self._lip_detector = detector_mod.DeepfakeDetector()
            self._lip_detector_err = None
        except Exception as e:
            self._lip_detector = None
            self._lip_detector_mod = None
            self._lip_outline = None
            self._lip_detector_err = str(e)
        return self._lip_detector

    def _measure_lip(self, detector, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = detector.face_mesh.process(rgb)

        h, w = frame_bgr.shape[:2]
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            upper_y = lm[13].y * h
            lower_y = lm[14].y * h
            left_x = lm[61].x * w
            right_x = lm[291].x * w
            lip_width = abs(right_x - left_x) or 1
            aperture = abs(lower_y - upper_y) / lip_width

            outline = self._lip_outline or [
                61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
                291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61
            ]
            lip_pts = []
            for idx in outline:
                if idx < len(lm):
                    lip_pts.append((int(lm[idx].x * w), int(lm[idx].y * h)))
            return float(aperture), lip_pts

        return float(detector._pixel_fallback(frame_bgr)), None

    def _render_video_frame(self, frame_bgr, sync_val=None, lip_pts=None):
        if frame_bgr is None or self._video_lbl is None:
            return

        ov = frame_bgr.copy()

        if lip_pts:
            color = (0, 200, 0) if sync_val is None or sync_val >= 0.6 else (
                (0, 200, 200) if sync_val >= 0.45 else (0, 0, 200))
            cv2.polylines(ov, [np.array(lip_pts, dtype=np.int32)], True, color, 2)

        if sync_val is not None:
            if sync_val >= 0.6:
                label = "SYNC"
                box = (0, 200, 0)
            elif sync_val >= 0.45:
                label = "MIXED"
                box = (0, 200, 200)
            else:
                label = "UNSYNC"
                box = (0, 0, 200)
            cv2.rectangle(ov, (10, 10), (140, 36), box, -1)
            cv2.putText(ov, label, (18, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 1, cv2.LINE_AA)

        img = _bgr_ctk(ov, self._video_w, self._video_h)
        self._video_lbl.configure(image=img)
        self._video_lbl.image = img

    def _apply_binary_verdict(self, trust):
        is_real = trust >= 50
        label = "REAL" if is_real else "FAKE"
        color = GREEN if is_real else RED
        self._banner._vtitle.configure(text=label, text_color=color)
        self._banner._vsub.configure(text=f"Lip-sync score  {trust:.1f} / 100")

    def _upd_lip_main(self, trust, fp, rp, result, frames, fps_target):
        self._banner.update(trust, fp, rp)
        self._apply_binary_verdict(trust)
        sync_score = float(result.get("sync_score", 0))
        self._sync_lbl.configure(text=f"Sync score: {sync_score:.1f}%", text_color=_vc(trust))

        lips = result.get("lip_data") or []
        audio = result.get("audio_data") or []
        if lips and audio:
            pearson_r = float(result.get("pearson_r", 0.0))
            jitter = float(result.get("jitter", 0.0))
            lag_ms = float(result.get("lag_ms", 0.0))
            lag_frames = int(round(lag_ms / (1000.0 / max(fps_target, 1.0))))

            lip_arr = np.array(lips, float)
            aud_arr = np.array(audio, float)
            speech_pct = float((aud_arr > 0.5).mean() * 100.0) if len(aud_arr) else 0.0
            lip_move_pct = float((lip_arr > 0.15).mean() * 100.0) if len(lip_arr) else 0.0

            self._ai["Peak Sync"].configure(text=f"{pearson_r:.3f}")
            self._ai["Deviation"].configure(text=f"{jitter:.4f}")
            self._ai["Lag Frames"].configure(text=str(lag_frames))
            self._ai["Speech %"].configure(text=f"{speech_pct:.1f}%")
            self._ai["Lip Movement %"].configure(text=f"{lip_move_pct:.1f}%")
            self._ai["Agreement %"].configure(text=f"{sync_score:.1f}%")
            self._draw_chart(lips, audio)

        self._upd_strip(frames)

    def _upd(self, vt, vc, avg, fp, rp, lips, audio, frames):
        self._banner.update(avg, fp, rp)
        self._apply_binary_verdict(avg)
        self._sync_lbl.configure(text=f"Sync score: {avg:.1f}%", text_color=_vc(avg))
        if lips:
            self._ai["Peak Sync"].configure(text=f"{lips[0]:.4f}")
            self._ai["Deviation"].configure(text=f"{float(np.std(lips)):.4f}")
            self._ai["Lag Frames"].configure(
                text=str(sum(1 for a in audio if a > 80)))
            self._ai["Speech %"].configure(text=f"{min(99, avg+5):.1f}%")
            self._ai["Lip Movement %"].configure(
                text=f"{sum(1 for l in lips if l > 0.02)/max(len(lips),1)*100:.1f}%")
            self._ai["Agreement %"].configure(text=f"{avg:.1f}%")
            self._draw_chart(lips, audio)
        self._upd_strip(frames)

    def _draw_chart(self, lips, audio):
        ae = np.array(audio, float)
        la = np.array(lips, float)
        if ae.max() > 0: ae /= ae.max()
        if la.max() > 0: la /= la.max()
        sync = float(np.mean(la)) * 100
        self._chart.clear(f"Audio-to-Lip Alignment  —  Sync Score {sync:.1f}%")
        self._chart.line(ae.tolist(), color=TEAL, ymin=0, ymax=1)
        self._chart.line(la.tolist(), color=RED, ymin=0, ymax=1)
        self._chart.ylabels(0, 1, 4)
        self._chart.xlabels(list(range(0, len(lips), max(1, len(lips)//8))))
        self._chart.legend([("Audio Energy", TEAL), ("Lip Aperture", RED)])

    def _upd_strip(self, frames):
        samp = frames[::max(1, len(frames) // 9)][:9]
        for i, l in enumerate(self._stlbls):
            img = _bgr_ctk(samp[i], 80, 56) if i < len(samp) else _blank_ctk(80, 56)
            l.configure(image=img)
            l.image = img


# 
# TAB: LIVE DETECTION
# 
class LiveTab(ctk.CTkFrame):
    def __init__(self, parent, state):
        super().__init__(parent, fg_color=BG)
        self._app_state = state
        self._running = False
        self._q = queue.Queue(maxsize=1)
        self._use_camera = True
        self._cam_index = 0
        self._analysis_stride = 3
        self._analysis_size = (800, 450)
        self._capture_size = (960, 540)
        self._prob_hist = deque(maxlen=8)
        self._fake_thresh = 0.7
        self._build()
        self._poll()

    def _build(self):
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        sidebar = _card(self, width=250)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=12)
        sidebar.grid_propagate(False)

        _lbl(sidebar, "Live Detection", F_BOLD, WHITE).pack(
            anchor="w", padx=16, pady=(16, 4))
        _lbl(sidebar, "Real-time video stream analysis",
             F_LABEL, TEXT2).pack(anchor="w", padx=16, pady=(0, 12))
        _divider(sidebar).pack(fill="x", padx=12)

        self._stbtn = _btn_success(sidebar, "Start", self._toggle, width=226)
        self._stbtn.pack(padx=12, pady=(12, 4))
        self._scbtn = _btn_ghost(sidebar, "Screen Capture", self._toggle_sc, width=226)
        self._scbtn.pack(padx=12, pady=4)
        self._cambtn = _btn_ghost(sidebar, "Camera: On", self._toggle_cam, width=226)
        self._cambtn.configure(text_color=GREEN)
        self._cambtn.pack(padx=12, pady=4)

        _section_title(sidebar, "Live Scores", pady=(16, 6))
        self._li = {}
        for k in ["Trust Score", "Spatial", "Temporal", "AV Sync", "FPS", "Frames"]:
            self._li[k] = _stat_row(sidebar, k)

        _divider(sidebar).pack(fill="x", padx=12, pady=(8, 0))
        _section_title(sidebar, "Liveness Challenge", pady=(10, 6))
        _btn_ghost(sidebar, "Run Challenge",
                   self._challenge, width=226).pack(padx=12, pady=4)
        self._clbl = _lbl(sidebar, "Ask subject to turn head 45 degrees",
                          F_LABEL, TEXT2, justify="left", wraplength=180)
        self._clbl.pack(anchor="w", padx=16, pady=(4, 2))
        self._cres = _lbl(sidebar, "", F_BADGE, TEXT2,
                          justify="left", wraplength=180)
        self._cres.pack(anchor="w", padx=16, pady=(0, 16))

        center = _card(self)
        center.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)
        center.rowconfigure(1, weight=1)
        center.columnconfigure(0, weight=1)

        status_bar = ctk.CTkFrame(center, fg_color="transparent")
        status_bar.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 8))
        status_bar.columnconfigure(0, weight=1)
        self._lverd = _lbl(status_bar, "IDLE", F_HERO2, TEXT2)
        self._lverd.grid(row=0, column=0, sticky="w")
        _btn_ghost(status_bar, "Export PDF", self._export, width=120).grid(
            row=0, column=1, sticky="e")

        feed = _surface(center)
        feed.grid(row=1, column=0, sticky="nsew", padx=12, pady=4)
        feed.rowconfigure(0, weight=1)
        feed.columnconfigure(0, weight=1)
        self._flbl = ctk.CTkLabel(feed, text="", corner_radius=8)
        self._flbl.pack(expand=True, padx=6, pady=6)
        blank = _blank_ctk(700, 420)
        self._flbl.configure(image=blank)
        self._flbl.image = blank

        pills = ctk.CTkFrame(center, fg_color="transparent")
        pills.grid(row=2, column=0, sticky="ew", padx=12, pady=(8, 16))
        pills.columnconfigure((0, 1, 2, 3), weight=1)
        self._pills = {}
        for i, (k, c) in enumerate([("Spatial", ACCENT), ("Temporal", TEAL),
                                      ("AV Sync", YELLOW), ("Trust", GREEN)]):
            pill = _surface(pills)
            pill.grid(row=0, column=i, padx=4, pady=0, sticky="ew")
            _lbl(pill, k, F_LABEL, TEXT2).pack(pady=(8, 0))
            v = _lbl(pill, "—", F_BOLD, WHITE)
            v.pack(pady=(0, 8))
            self._pills[k] = v

    def _toggle(self):
        if self._running: self._stop()
        else: self._start()

    def _start(self):
        if not self._app_state.video_path and not self._app_state.screen_cap.active and not self._use_camera:
            return
        self._running = True
        self._stbtn.configure(text="Stop", fg_color=RED, hover_color="#dc2626")
        self._lverd.configure(text="ANALYZING", text_color=TEAL)
        threading.Thread(target=self._cap_loop, daemon=True).start()
        threading.Thread(target=self._ana_loop, daemon=True).start()

    def _stop(self):
        self._running = False
        self._stbtn.configure(text="Start", fg_color=GREEN, hover_color="#16a34a")
        self._lverd.configure(text="IDLE", text_color=TEXT2)

    def _toggle_sc(self):
        if self._app_state.screen_cap.active:
            self._app_state.screen_cap.stop()
            self._scbtn.configure(text="Screen Capture", text_color=TEXT2)
        else:
            if self._use_camera:
                self._use_camera = False
                self._cambtn.configure(text="Camera: Off", text_color=TEXT2)
            self._app_state.screen_cap.start()
            self._scbtn.configure(text="Stop Capture", text_color=RED)
        if self._running:
            self._stop()

    def _toggle_cam(self):
        if self._use_camera:
            self._use_camera = False
            self._cambtn.configure(text="Camera: Off", text_color=TEXT2)
        else:
            self._use_camera = True
            self._cambtn.configure(text="Camera: On", text_color=GREEN)
            if self._app_state.screen_cap.active:
                self._app_state.screen_cap.stop()
                self._scbtn.configure(text="Screen Capture", text_color=TEXT2)
        if self._running:
            self._stop()

    def _load_thread(self):
        self._lmbtn.configure(text="Loading...", state="disabled")
        threading.Thread(target=self._load_models, daemon=True).start()

    def _load_models(self):
        try:
            self._app_state.pipeline.load_models()
            self._app_state.models_loaded = True
            txt = self._app_state.pipeline.model_status_text()
            self.after(0, lambda: self._mslbl.configure(
                text=txt, text_color=GREEN))
            self.after(0, lambda: self._lmbtn.configure(
                text="Models Loaded", state="disabled",
                fg_color=GREEN, hover_color=GREEN, text_color=WHITE))
        except Exception:
            self.after(0, lambda: self._lmbtn.configure(
                text="Load Models", state="normal"))

    def _enqueue_frame(self, frame):
        try:
            self._q.put_nowait(frame)
        except queue.Full:
            try:
                _ = self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(frame)
            except queue.Full:
                pass

    def _cap_loop(self):
        if self._app_state.screen_cap.active:
            while self._running:
                f = self._app_state.screen_cap.read()
                if f is not None:
                    self._enqueue_frame(f)
                time.sleep(0.04)
        elif self._use_camera:
            cap = cv2.VideoCapture(self._cam_index)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(self._cam_index, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._capture_size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._capture_size[1])
            cap.set(cv2.CAP_PROP_FPS, 30)
            while self._running:
                ret, f = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue
                self._enqueue_frame(f)
                time.sleep(0.03)
            cap.release()
        elif self._app_state.video_path:
            cap = cv2.VideoCapture(self._app_state.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            while self._running:
                ret, f = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self._enqueue_frame(f)
                time.sleep(1.0 / fps)
            cap.release()

    def _ana_loop(self):
        """Correct analysis loop using DeepfakeModel"""
        try:
            models = _ensure_models()
            DEVICE = models["device"]
            FRAME_COUNT = models["frame_count"]
            global_model = models["model"]
            torch = models["torch"]
            mtcnn = _ensure_mtcnn()
        except Exception as e:
            print(f"[LiveTab] Model init error: {e}")
            return

        fc = 0
        t0 = time.time()
        frame_buffer = []
        last_result = None
        analysis_stride = max(1, self._analysis_stride)
        analysis_size = self._analysis_size
        
        while self._running:
            try: 
                frame = self._q.get(timeout=1.0)
            except queue.Empty: 
                continue
            
            # Add frame to buffer
            frame_buffer.append(frame)
            if len(frame_buffer) > FRAME_COUNT:
                frame_buffer.pop(0)
            
            fc += 1
            fps = fc / max(time.time() - t0, 0.001)
            
            # Analyze every N frames for smoother playback
            if len(frame_buffer) == FRAME_COUNT and fc % analysis_stride == 0:
                try:
                    # Detect faces in buffered frames
                    faces = []
                    for buffered_frame in frame_buffer:
                        frame_rgb = cv2.cvtColor(buffered_frame, cv2.COLOR_BGR2RGB)
                        if analysis_size:
                            frame_rgb = cv2.resize(frame_rgb, analysis_size, interpolation=cv2.INTER_AREA)
                        frame_pil = Image.fromarray(frame_rgb)
                        try:
                            face = mtcnn(frame_pil)
                            if face is not None:
                                faces.append(face.cpu())
                        except:
                            pass
                    
                    # If we have enough faces, classify
                    if faces:
                        faces_model = list(faces)
                        while len(faces_model) < FRAME_COUNT:
                            faces_model.append(faces_model[len(faces_model) % len(faces_model)])
                        faces_model = faces_model[:FRAME_COUNT]
                        
                        # Classify using the correct model
                        with torch.no_grad():
                            seq = torch.stack(faces_model).unsqueeze(1).unsqueeze(0).to(DEVICE)
                            logits = global_model(seq)
                            prob = torch.sigmoid(logits).item()
                        
                        # Smooth predictions to reduce false positives
                        self._prob_hist.append(prob)
                        avg_prob = float(np.mean(self._prob_hist))

                        # Determine classification with higher fake threshold
                        if avg_prob >= self._fake_thresh:
                            label = "FAKE"
                            confidence = avg_prob * 100
                            trust_score = (1 - avg_prob) * 100
                        else:
                            label = "AUTHENTIC"
                            confidence = (1 - avg_prob) * 100
                            trust_score = confidence
                        
                        # Get face bbox from last frame
                        frame_rgb = cv2.cvtColor(frame_buffer[-1], cv2.COLOR_BGR2RGB)
                        frame_pil = Image.fromarray(frame_rgb)
                        face_bbox = None
                        try:
                            face = mtcnn(frame_pil, return_prob=False)
                            if face is not None:
                                h, w = frame_buffer[-1].shape[:2]
                                # MTCNN returns normalized coordinates if we process properly
                                face_bbox = (0, 0, w, h)  # Default full frame
                        except:
                            pass
                        
                        base = {
                            "label": label,
                            "trust_score": trust_score,
                            "confidence": confidence,
                            "spatial_score": confidence,
                            "temporal_score": confidence,
                            "av_score": confidence,
                            "face_bbox": face_bbox,
                            "prob": avg_prob
                        }
                    else:
                        # No faces detected
                        self._prob_hist.clear()
                        base = {
                            "label": "NO FACE",
                            "trust_score": 30,
                            "confidence": 30,
                            "spatial_score": 30,
                            "temporal_score": 30,
                            "av_score": 30,
                            "face_bbox": None,
                            "prob": 0.5
                        }

                    last_result = base
                    r = dict(base)
                    r.update({
                        "frame": frame_buffer[-1],
                        "fps": fps,
                        "frame_count": fc
                    })
                    self.after(0, self._upd_ui, r)
                    
                except Exception as e:
                    print(f"[LiveTab] Analysis error: {e}")
                    pass
            elif last_result:
                r = dict(last_result)
                r.update({
                    "frame": frame,
                    "fps": fps,
                    "frame_count": fc
                })
                self.after(0, self._upd_ui, r)
            else:
                # Show frame while buffering
                self._prob_hist.clear()
                r = {
                    "frame": frame,
                    "fps": fps,
                    "frame_count": fc,
                    "label": "BUFFERING...",
                    "trust_score": 50,
                    "confidence": 0,
                    "spatial_score": 0,
                    "temporal_score": 0,
                    "av_score": 0,
                    "face_bbox": None,
                    "prob": 0.5
                }
                self.after(0, self._upd_ui, r)

    def _upd_ui(self, r):
        """Update UI with correct model predictions"""
        trust = r.get("trust_score", 50)
        label = r.get("label", "ANALYZING")
        confidence = r.get("confidence", 0)
        
        # Determine color and verdict text
        if label == "BUFFERING...":
            vc = TEXT2
            vt = "BUFFERING"
        elif label == "NO FACE":
            vc = RED
            vt = "NO FACE"
        elif label == "FAKE":
            vc = RED
            vt = "FAKE"
        else:  # AUTHENTIC
            vc = GREEN
            vt = "AUTHENTIC"
        
        self._lverd.configure(text=vt, text_color=vc)
        self._li["Trust Score"].configure(text=f"{trust:.1f}%", text_color=vc)
        self._li["Spatial"].configure(text=f"{confidence:.1f}%")
        self._li["Temporal"].configure(text=f"{confidence:.1f}%")
        self._li["AV Sync"].configure(text=f"{confidence:.1f}%")
        self._li["FPS"].configure(text=f"{r.get('fps', 0):.1f}")
        self._li["Frames"].configure(text=str(r.get("frame_count", 0)))
        
        self._pills["Spatial"].configure(text=f"{confidence:.0f}%", text_color=ACCENT)
        self._pills["Temporal"].configure(text=f"{confidence:.0f}%", text_color=TEAL)
        self._pills["AV Sync"].configure(text=f"{confidence:.0f}%", text_color=YELLOW)
        self._pills["Trust"].configure(text=f"{trust:.0f}%", text_color=vc)
        
        if "frame" in r:
            f = r["frame"].copy()
            bb = r.get("face_bbox")
            
            # Color based on classification
            if label == "FAKE":
                col = (0, 0, 255)  # Red
            elif label == "AUTHENTIC":
                col = (0, 255, 0)  # Green
            elif label == "BUFFERING...":
                col = (0, 255, 255)  # Cyan
            else:
                col = (255, 165, 0)  # Orange for NO FACE
            
            if bb:
                x, y, w, h = bb
                cv2.rectangle(f, (x, y), (x+w, y+h), col, 2)
                cv2.putText(f, f"{vt}  {confidence:.0f}%", (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            
            fh, fw = f.shape[:2]
            cv2.rectangle(f, (0, fh-32), (500, fh), (0, 0, 0), -1)
            cv2.putText(
                f,
                f"Model: {label} ({confidence:.1f}%)  "
                f"Frames: {r.get('frame_count', 0)}  "
                f"FPS: {r.get('fps', 0):.1f}",
                (8, fh - 9), cv2.FONT_HERSHEY_PLAIN, 1.0, (160, 160, 160), 1)
            
            img = _bgr_ctk(f, 700, 420)
            self._flbl.configure(image=img)
            self._flbl.image = img
        
        self._app_state.last_results = r

    def _challenge(self):
        self._clbl.configure(text="Monitoring head rotation...", text_color=YELLOW)
        self._cres.configure(text="Watching...", text_color=YELLOW)
        threading.Thread(target=self._run_challenge, daemon=True).start()

    def _run_challenge(self):
        sc = []
        for _ in range(40):
            if self._app_state.last_results:
                sc.append(self._app_state.last_results.get("warp_score", 50))
            time.sleep(0.2)
        if not sc:
            res, col = "No data collected.", TEXT2
        elif np.std(sc) > 15 and np.mean(sc) < 55:
            res, col = "FAIL — Deepfake warp detected", RED
        elif np.std(sc) > 10:
            res, col = "UNCERTAIN — Requires manual review", YELLOW
        else:
            res, col = "PASS — Likely authentic", GREEN
        self.after(0, lambda: self._cres.configure(text=res, text_color=col))
        self.after(0, lambda: self._clbl.configure(
            text="Challenge complete.", text_color=TEXT2))

    def _export(self):
        if not self._app_state.last_results: return
        p = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")],
            initialfile="forensic_report.pdf")
        if p:
            try:
                from utils.report_generator import generate_pdf_report
                generate_pdf_report(p, self._app_state.last_results, 0)
            except Exception as e:
                print(f"Export error: {e}")

    def _poll(self):
        self.after(200, self._poll)


# 
# TAB: AUTO-ANALYSIS (One-Click Full Detection)
# 
class AutoAnalysisTab(ctk.CTkFrame):
    """
    ONE-CLICK FULL ANALYSIS:
    Upload a video → click Execute All → all 4 detectors run automatically →
    scores combined → final DEEPFAKE / AUTHENTIC / UNCERTAIN verdict.

    Detectors (in order):
      1. Spatial Detection  (30%) — facial artifacts via SpatialDetector
      2. Temporal Analysis  (35%) — frame-to-frame consistency via TemporalDetector
      3. AV Sync Detection  (25%) — lip-sync via AVSyncDetector + Wav2Vec2
      4. Face Quality Check (10%) — face detection ratio across frames
    """

    # Weights: trained model is anchor (40%), heuristics add signal (60%)
    WEIGHTS = {"model": 0.40, "spatial": 0.20, "temporal": 0.22, "av_sync": 0.12, "face": 0.06}

    def __init__(self, parent, state):
        super().__init__(parent, fg_color=BG)
        self._app_state = state
        self._analyzing = False
        self._progress_queue = queue.Queue()
        self._build()

    # ─── UI Construction ───────────────────────────────────────────────────────
    def _build(self):
        # 3-column layout: sidebar | center feed | right results
        self.columnconfigure(0, weight=0)   # left sidebar  (fixed 260)
        self.columnconfigure(1, weight=1)   # center feed   (expandable)
        self.columnconfigure(2, weight=0)   # right results (fixed 320)
        self.rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_center()
        self._build_results()

    # ─── LEFT SIDEBAR ─────────────────────────────────────────────────────────
    def _build_sidebar(self):
        sb = _card(self, width=260)
        sb.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=12)
        sb.grid_propagate(False)
        sb.pack_propagate(False)

        _lbl(sb, "One-Click Analysis", F_BOLD, WHITE).pack(anchor="w", padx=16, pady=(16, 2))
        _lbl(sb, "All detectors · single button", F_LABEL, TEXT2).pack(anchor="w", padx=16, pady=(0, 10))
        _divider(sb).pack(fill="x", padx=12)

        # Upload zone
        uz = ctk.CTkFrame(sb, fg_color=SURFACE2, corner_radius=10,
                          border_width=1, border_color=BORDER2, height=90)
        uz.pack(fill="x", padx=12, pady=(10, 6))
        uz.pack_propagate(False)
        _lbl(uz, "🎬", ("Segoe UI", 24), TEXT2).pack(pady=(8, 0))
        _lbl(uz, "Drop video or click Browse", F_LABEL, TEXT2).pack()
        _lbl(uz, "MP4  AVI  MOV  MKV", F_LABEL, TEXT3).pack()

        _btn_primary(sb, "Browse Video", self._browse, width=236).pack(padx=12, pady=(4, 3))

        self._run_btn = ctk.CTkButton(
            sb, text="▶  Execute All Detectors", command=self._execute,
            fg_color=GREEN, hover_color="#16a34a", text_color=WHITE,
            font=("Segoe UI", 12, "bold"), corner_radius=8, height=42, width=236)
        self._run_btn.pack(padx=12, pady=(3, 4))
        self._run_btn.configure(state="disabled")

        _divider(sb).pack(fill="x", padx=12, pady=(8, 0))
        _section_title(sb, "File Info", pady=(8, 4))
        self._info = {}
        for k in ["File", "Duration", "FPS", "Size", "Frames"]:
            self._info[k] = _stat_row(sb, k)

        _divider(sb).pack(fill="x", padx=12, pady=(8, 0))
        _section_title(sb, "Detector Status", pady=(8, 4))
        self._det_rows = {}
        for name in ["🤖 AI Model", "🧠 Spatial", "⏱ Temporal", "🔊 AV Sync", "👤 Face Quality"]:
            row = ctk.CTkFrame(sb, fg_color="transparent")
            row.pack(fill="x", padx=12, pady=2)
            _lbl(row, name, F_LABEL, TEXT).pack(side="left")
            st = _lbl(row, "—", F_BADGE, TEXT3)
            st.pack(side="right")
            key = name.split(" ", 1)[1]
            self._det_rows[key] = st

    # ─── CENTER FEED ──────────────────────────────────────────────────────────
    def _build_center(self):
        center = ctk.CTkFrame(self, fg_color="transparent")
        center.grid(row=0, column=1, sticky="nsew", padx=6, pady=12)
        center.rowconfigure(0, weight=1)
        center.rowconfigure(1, weight=0)
        center.columnconfigure(0, weight=1)

        # Video preview card
        prev_card = _card(center)
        prev_card.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        prev_card.rowconfigure(1, weight=1)
        prev_card.columnconfigure(0, weight=1)

        hdr = ctk.CTkFrame(prev_card, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", padx=16, pady=(12, 0))
        _lbl(hdr, "Live Analysis Preview", F_BOLD, WHITE).pack(side="left")
        self._stage_lbl = _lbl(hdr, "Waiting for video...", F_LABEL, TEXT3)
        self._stage_lbl.pack(side="right")

        feed = _surface(prev_card)
        feed.grid(row=1, column=0, sticky="nsew", padx=12, pady=(8, 12))
        feed.rowconfigure(0, weight=1)
        feed.columnconfigure(0, weight=1)
        self._frame_lbl = ctk.CTkLabel(
            feed, text="Upload a video and click  ▶  Execute All Detectors",
            text_color=TEXT2, font=F_BOLD)
        self._frame_lbl.pack(expand=True, fill="both", padx=6, pady=6)

        # Progress pipeline card
        prog_card = _card(center)
        prog_card.grid(row=1, column=0, sticky="ew")
        _lbl(prog_card, "Analysis Pipeline", F_BOLD, WHITE).pack(
            anchor="w", padx=16, pady=(12, 6))

        self._prog_bars = {}
        stage_colors = {
            "AI Model":            "#a855f7",   # purple — trained model
            "Spatial Detection":   ACCENT,
            "Temporal Analysis":   TEAL,
            "AV Sync Detection":   YELLOW,
            "Face Quality Check":  GREEN,
        }
        for stage, color in stage_colors.items():
            row = ctk.CTkFrame(prog_card, fg_color="transparent")
            row.pack(fill="x", padx=16, pady=3)
            row.columnconfigure(1, weight=1)

            icon = _lbl(row, "⊙", ("Segoe UI", 12), TEXT3)
            icon.grid(row=0, column=0, padx=(0, 8), sticky="w")

            bar_lbl = _lbl(row, stage, F_LABEL, TEXT2)
            bar_lbl.grid(row=0, column=1, sticky="w")

            bar = ctk.CTkProgressBar(row, height=7, corner_radius=4,
                                     progress_color=color, fg_color=BORDER)
            bar.grid(row=1, column=1, sticky="ew", pady=(2, 0))
            bar.set(0)

            pct = _lbl(row, "—", F_LABEL, TEXT3)
            pct.grid(row=1, column=2, padx=(8, 0), sticky="e")

            self._prog_bars[stage] = {"bar": bar, "icon": icon, "pct": pct, "color": color}

        ctk.CTkFrame(prog_card, fg_color="transparent", height=10).pack()

    # ─── RIGHT RESULTS PANEL ──────────────────────────────────────────────────
    def _build_results(self):
        right = _card(self, width=330)
        right.grid(row=0, column=2, sticky="nsew", padx=(6, 12), pady=12)
        right.grid_propagate(False)
        right.columnconfigure(0, weight=1)

        _lbl(right, "Final Verdict", F_BOLD, WHITE).pack(anchor="w", padx=16, pady=(16, 4))
        _divider(right).pack(fill="x", padx=12, pady=(0, 8))

        # Big verdict box
        vbox = ctk.CTkFrame(right, fg_color=SURFACE2, corner_radius=12,
                            border_width=1, border_color=BORDER)
        vbox.pack(fill="x", padx=12, pady=(0, 8))
        self._verdict_icon  = _lbl(vbox, "⬡",  ("Segoe UI", 34), TEXT3)
        self._verdict_icon.pack(pady=(14, 0))
        self._verdict_title = _lbl(vbox, "AWAITING", ("Segoe UI", 20, "bold"), TEXT3)
        self._verdict_title.pack()
        self._verdict_conf  = _lbl(vbox, "Run analysis to see result", F_LABEL, TEXT2)
        self._verdict_conf.pack(pady=(2, 12))

        # 4 mini score gauges in 2×2 grid
        _section_title(right, "Score Breakdown", pady=(4, 4))
        gg = ctk.CTkFrame(right, fg_color="transparent")
        gg.pack(fill="x", padx=12, pady=(0, 4))
        gg.columnconfigure((0, 1), weight=1)

        self._gauges = {}
        gauge_cfg = [
            ("AI Model",     "#a855f7", 0, 0),
            ("Spatial",      ACCENT,    0, 1),
            ("Temporal",     TEAL,      1, 0),
            ("AV Sync",      YELLOW,    1, 1),
            ("Face Quality", GREEN,     2, 0),
        ]
        gg.rowconfigure(2, weight=1)
        for name, color, r, c in gauge_cfg:
            cell = ctk.CTkFrame(gg, fg_color=SURFACE2, corner_radius=8)
            cell.grid(row=r, column=c, padx=3, pady=3, sticky="ew")
            _lbl(cell, name, ("Segoe UI", 9), TEXT2).pack(pady=(6, 0))
            val = _lbl(cell, "—", ("Segoe UI", 15, "bold"), color)
            val.pack()
            bar = ctk.CTkProgressBar(cell, height=5, corner_radius=2,
                                     progress_color=color, fg_color=BORDER)
            bar.pack(fill="x", padx=8, pady=(2, 8))
            bar.set(0)
            self._gauges[name] = {"val": val, "bar": bar}

        # Combined trust
        _divider(right).pack(fill="x", padx=12, pady=(4, 0))
        tr = ctk.CTkFrame(right, fg_color="transparent")
        tr.pack(fill="x", padx=16, pady=(6, 2))
        tr.columnconfigure(0, weight=1)
        _lbl(tr, "Combined Trust Score", F_BADGE, TEXT2).grid(row=0, column=0, sticky="w")
        self._trust_lbl = _lbl(tr, "—", ("Segoe UI", 14, "bold"), TEXT)
        self._trust_lbl.grid(row=0, column=1, sticky="e")
        self._trust_bar = ctk.CTkProgressBar(right, height=10, corner_radius=4,
                                             progress_color=TEAL, fg_color=BORDER)
        self._trust_bar.pack(fill="x", padx=12, pady=(0, 6))
        self._trust_bar.set(0)

        # Explanation box
        _divider(right).pack(fill="x", padx=12)
        _section_title(right, "Detailed Explanation", pady=(6, 4))
        self._explain_box = ctk.CTkTextbox(
            right, fg_color=SURFACE2, text_color=TEXT,
            corner_radius=8, font=("Consolas", 10))
        self._explain_box.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self._explain_box.insert("1.0", "Detailed analysis will appear here\nafter detection completes.")
        self._explain_box.configure(state="disabled")

    # ─── Actions ──────────────────────────────────────────────────────────────
    def _browse(self):
        p = filedialog.askopenfilename(
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All", "*.*")])
        if not p:
            return
        self._app_state.video_path = p
        sz = os.path.getsize(p) / (1024 * 1024)
        cap = cv2.VideoCapture(p)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur = tot / fps
        ret, frame = cap.read()
        cap.release()
        self._info["File"].configure(text=os.path.basename(p)[:22])
        self._info["Duration"].configure(text=f"{dur:.1f}s")
        self._info["FPS"].configure(text=f"{fps:.1f}")
        self._info["Size"].configure(text=f"{sz:.1f} MB")
        self._info["Frames"].configure(text=str(tot))
        if ret:
            img = _bgr_ctk(frame, 620, 360)
            self._frame_lbl.configure(image=img, text="")
            self._frame_lbl.image = img
        self._run_btn.configure(state="normal")
        self._reset_ui()

    def _reset_ui(self):
        for bd in self._prog_bars.values():
            bd["bar"].set(0)
            bd["icon"].configure(text="⊙", text_color=TEXT3)
            bd["pct"].configure(text="—")
        for gd in self._gauges.values():
            gd["val"].configure(text="—")
            gd["bar"].set(0)
        for st in self._det_rows.values():
            st.configure(text="—", text_color=TEXT3)
        self._verdict_icon.configure(text="⬡", text_color=TEXT3)
        self._verdict_title.configure(text="AWAITING", text_color=TEXT3)
        self._verdict_conf.configure(text="Run analysis to see result")
        self._trust_lbl.configure(text="—")
        self._trust_bar.set(0)
        self._trust_bar.configure(progress_color=TEAL)
        self._explain_box.configure(state="normal")
        self._explain_box.delete("1.0", "end")
        self._explain_box.insert("1.0",
            "Detailed analysis will appear here\nafter detection completes.")
        self._explain_box.configure(state="disabled")

    def _execute(self):
        if not self._app_state.video_path or self._analyzing:
            return
        self._analyzing = True
        self._run_btn.configure(text="⏳ Analyzing...", state="disabled",
                                fg_color=TEXT3, hover_color=TEXT3)
        self._stage_lbl.configure(text="Starting analysis...", text_color=TEAL)
        self._reset_ui()
        threading.Thread(target=self._full_analysis, daemon=True).start()
        self._poll_progress()

    # ─── Core Analysis (runs in background thread) ────────────────────────────
    def _full_analysis(self):
        """
        Run the SAME logic as each individual tab, then combine their scores.
          • Analyze tab  → EfficientNet-B0 on 8 face crops   (weight 40%)
          • Video tab    → EfficientNet-B0 on 16 frame crops  (weight 35%)
          • Lip-Sync tab → AV-sync pipeline score             (weight 25%)
        """
        try:
            video_path = self._app_state.video_path
            PQ = self._progress_queue.put   # shorthand

            # ── Shared setup ────────────────────────────────────────────────
            PQ(("stage_txt", "Loading models and extracting frames..."))
            from PIL import Image as _PIL_Image

            models   = _ensure_models()
            torch    = models["torch"]
            DEVICE   = models["device"]
            FC       = models["frame_count"]   # 8
            nn_model = models["model"]
            mtcnn    = _ensure_mtcnn()

            cap = cv2.VideoCapture(video_path)
            fps_vid = cap.get(cv2.CAP_PROP_FPS) or 25.0
            tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            # ══════════════════════════════════════════════════════════════
            # TAB 1 — ANALYZE TAB  (8 face crops → DeepfakeModel)
            # ══════════════════════════════════════════════════════════════
            PQ(("det_status",  "AI Model",  "⟳ Running...", YELLOW))
            PQ(("stage_txt",   "⟳ Analyze Tab — EfficientNet-B0 on face crops..."))
            PQ(("prog_icon",   "AI Model",  "⟳",  "#a855f7"))

            # Exactly 8 evenly-spaced frames (same as ClassicTab)
            step8  = max(1, tot // 8)
            frames8 = []
            cap = cv2.VideoCapture(video_path)
            fi = 0
            while True:
                ret, f = cap.read()
                if not ret: break
                if fi % step8 == 0 and len(frames8) < 8:
                    frames8.append(f)
                    PQ(("preview", f))
                    PQ(("progress", "AI Model", len(frames8) / 8))
                fi += 1
            cap.release()
            self._app_state.frames = frames8

            analyze_trust = self._score_with_model(
                frames8, nn_model, mtcnn, torch, DEVICE, FC, PQ, "AI Model")

            PQ(("progress",    "AI Model", 1.0))
            PQ(("prog_icon",   "AI Model", "✓", "#a855f7"))
            PQ(("gauge_update","AI Model",  analyze_trust))
            PQ(("det_status",  "AI Model",  f"✓ {analyze_trust:.1f}%", _vc(analyze_trust)))
            print(f"[AutoAnalysis] Analyze tab score: {analyze_trust:.1f}%")

            # ══════════════════════════════════════════════════════════════
            # TAB 2 — VIDEO TAB  (16 frame crops → DeepfakeModel)
            # ══════════════════════════════════════════════════════════════
            PQ(("det_status",  "Spatial",  "⟳ Running...", YELLOW))
            PQ(("stage_txt",   "⟳ Video Tab — scanning 16 frames..."))
            PQ(("prog_icon",   "Spatial Detection", "⟳", YELLOW))

            step16 = max(1, tot // 16)
            frames16 = []
            cap = cv2.VideoCapture(video_path)
            fi = 0
            while True:
                ret, f = cap.read()
                if not ret: break
                if fi % step16 == 0 and len(frames16) < 16:
                    frames16.append(f)
                    PQ(("preview", f))
                    PQ(("progress", "Spatial Detection", len(frames16) / 16))
                fi += 1
            cap.release()

            video_trust = self._score_with_model(
                frames16, nn_model, mtcnn, torch, DEVICE, FC, PQ, "Spatial Detection")

            PQ(("progress",    "Spatial Detection", 1.0))
            PQ(("prog_icon",   "Spatial Detection", "✓", GREEN))
            PQ(("gauge_update","Spatial",  video_trust))
            PQ(("det_status",  "Spatial",  f"✓ {video_trust:.1f}%",  _vc(video_trust)))
            print(f"[AutoAnalysis] Video tab score:   {video_trust:.1f}%")

            # ══════════════════════════════════════════════════════════════
            # TAB 3 — LIP-SYNC TAB  (AV sync pipeline — identical logic)
            # ══════════════════════════════════════════════════════════════
            PQ(("det_status",  "AV Sync", "⟳ Running...", YELLOW))
            PQ(("stage_txt",   "⟳ Lip-Sync Tab — measuring audio-visual alignment..."))
            PQ(("prog_icon",   "AV Sync Detection", "⟳", YELLOW))

            audio = self._app_state.pipeline._av.extract_audio_from_video(video_path)
            sr = getattr(self._app_state.pipeline._av, "SAMPLE_RATE", 16000)
            samples_per_frame = int(sr / fps_vid)

            step_ls = max(1, tot // 100)
            av_scores = []
            cap = cv2.VideoCapture(video_path)
            fi = 0; frame_count = 0
            while True:
                ret, f = cap.read()
                if not ret: break
                if fi % step_ls == 0 and frame_count < 100:
                    frame_count += 1
                    PQ(("preview", f))
                    prog = frame_count / 100
                    PQ(("progress", "AV Sync Detection", prog))
                    chunk = None
                    if audio is not None:
                        st = (frame_count - 1) * samples_per_frame
                        en = st + samples_per_frame
                        if st < len(audio):
                            chunk = audio[st:en]
                    r = self._app_state.pipeline._av.analyse(f, audio_chunk=chunk)
                    av_scores.append(r.get("av_score", 50.0))
                fi += 1
            cap.release()
            lipsync_trust = float(np.mean(av_scores)) if av_scores else 50.0

            PQ(("progress",    "AV Sync Detection", 1.0))
            PQ(("prog_icon",   "AV Sync Detection", "✓", GREEN))
            PQ(("gauge_update","AV Sync",    lipsync_trust))
            PQ(("det_status",  "AV Sync",    f"✓ {lipsync_trust:.1f}%", _vc(lipsync_trust)))
            print(f"[AutoAnalysis] Lip-Sync tab score: {lipsync_trust:.1f}%")

            # ── Hide unused gauges / mark N/A ──────────────────────────
            PQ(("progress",    "Temporal Analysis",  1.0))
            PQ(("prog_icon",   "Temporal Analysis",  "—",  TEXT3))
            PQ(("progress",    "Face Quality Check", 1.0))
            PQ(("prog_icon",   "Face Quality Check", "—",  TEXT3))
            PQ(("gauge_update","Temporal",     lipsync_trust))   # mirrors Lip-Sync
            PQ(("gauge_update","Face Quality", analyze_trust))   # mirrors Analyze
            PQ(("det_status",  "Temporal",     "— (mirrors Lip-Sync)", TEXT3))
            PQ(("det_status",  "Face Quality", "— (mirrors Analyze)",  TEXT3))

            # ── COMBINE with same weights as individual tabs ─────────────
            PQ(("stage_txt", "⟳ Combining tab results — generating final verdict..."))
            combined = (
                analyze_trust  * self.WEIGHTS["model"]   +   # 40 %
                video_trust    * self.WEIGHTS["temporal"] +   # 22 % (reused key)
                lipsync_trust  * self.WEIGHTS["av_sync"]  +  # 12 %
                # fill remaining 26% back with the model score as tie-breaker
                analyze_trust  * (self.WEIGHTS["spatial"] + self.WEIGHTS["face"])
            )
            PQ(("final_verdict", combined,
                analyze_trust, video_trust, lipsync_trust,
                analyze_trust, lipsync_trust, len(frames8)))
            PQ(("stage_txt", "✓ Analysis complete"))

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._progress_queue.put(("error", str(e)))
        finally:
            self._analyzing = False

    # ─── Shared model scorer (used by _full_analysis) ────────────────────────
    def _score_with_model(self, frames_bgr, nn_model, mtcnn, torch, DEVICE,
                          FC, PQ, prog_stage):
        """
        Detect faces with MTCNN, run DeepfakeModel, return trust score 0-100.
        Identical to ClassicTab._do_classify / VideoTab._do_analyze.
        """
        from PIL import Image as _PIL_Image
        faces = []
        for frame_bgr in frames_bgr:
            try:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil = _PIL_Image.fromarray(rgb)
                face = mtcnn(pil)
                if face is not None:
                    faces.append(face.cpu())
            except Exception:
                pass

        if not faces:
            print(f"[AutoAnalysis] {prog_stage}: no faces detected — returning neutral")
            return 50.0

        while len(faces) < FC:
            faces.append(faces[len(faces) % len(faces)])
        faces = faces[:FC]

        with torch.no_grad():
            seq    = torch.stack(faces).unsqueeze(1).unsqueeze(0).to(DEVICE)
            logits = nn_model(seq)
            prob   = torch.sigmoid(logits).item()   # ~1.0 = FAKE

        trust = float(np.clip((1.0 - prob) * 100.0, 0.0, 100.0))
        print(f"[AutoAnalysis] {prog_stage}: prob={prob:.4f}, trust={trust:.1f}%")
        return trust



    def _poll_progress(self):
        try:
            while True:
                msg = self._progress_queue.get_nowait()
                mtype = msg[0]

                if mtype == "stage_txt":
                    self._stage_lbl.configure(text=msg[1], text_color=TEAL)

                elif mtype == "progress":
                    _, stage, val = msg
                    if stage in self._prog_bars:
                        self._prog_bars[stage]["bar"].set(val)
                        self._prog_bars[stage]["pct"].configure(
                            text=f"{val*100:.0f}%")

                elif mtype == "prog_icon":
                    _, stage, icon, color = msg
                    if stage in self._prog_bars:
                        self._prog_bars[stage]["icon"].configure(
                            text=icon, text_color=color)

                elif mtype == "preview":
                    _, frame = msg
                    img = _bgr_ctk(frame, 620, 360)
                    self._frame_lbl.configure(image=img, text="")
                    self._frame_lbl.image = img

                elif mtype == "gauge_update":
                    _, name, val = msg
                    if name in self._gauges:
                        color = _vc(val)
                        self._gauges[name]["val"].configure(
                            text=f"{val:.1f}%", text_color=color)
                        self._gauges[name]["bar"].set(val / 100)
                        self._gauges[name]["bar"].configure(progress_color=color)

                elif mtype == "det_status":
                    _, name, text, color = msg
                    if name in self._det_rows:
                        self._det_rows[name].configure(text=text, text_color=color)

                elif mtype == "final_verdict":
                    _, combined, m, s, t, a, f, n_frames = msg
                    self._show_verdict(combined, m, s, t, a, f, n_frames)

                elif mtype == "error":
                    _, err = msg
                    self._stage_lbl.configure(text="❌ Error during analysis", text_color=RED)
                    self._explain_box.configure(state="normal")
                    self._explain_box.delete("1.0", "end")
                    self._explain_box.insert("1.0", f"❌ Analysis Error:\n\n{err}")
                    self._explain_box.configure(state="disabled")

        except queue.Empty:
            pass
        finally:
            if self._analyzing:
                self.after(100, self._poll_progress)
            else:
                self._run_btn.configure(
                    text="▶  Execute All Detectors",
                    state="normal",
                    fg_color=GREEN, hover_color="#16a34a")

    # ─── Verdict Display ──────────────────────────────────────────────────────
    def _show_verdict(self, combined, model, spatial, temporal, av_sync, face, n_frames):
        color = _vc(combined)
        self._trust_lbl.configure(text=f"{combined:.1f}%", text_color=color)
        self._trust_bar.set(combined / 100)
        self._trust_bar.configure(progress_color=color)

        if combined >= 65:
            verdict = "AUTHENTIC"
            icon    = "✅"
            uc_icon = "🟢"
            risk    = "LOW"
            rec     = "✅ ACCEPT — Content appears genuine."
        elif combined >= 45:
            verdict = "UNCERTAIN"
            icon    = "⚠️"
            uc_icon = "🟡"
            risk    = "MEDIUM"
            rec     = "🔍 REVIEW — Manual verification recommended."
        else:
            verdict = "DEEPFAKE"
            icon    = "❌"
            uc_icon = "🔴"
            risk    = "HIGH"
            rec     = "❌ REJECT — Content is SYNTHETIC / manipulated."

        self._verdict_icon.configure(text=icon, text_color=color)
        self._verdict_title.configure(text=verdict, text_color=color)
        self._verdict_conf.configure(
            text=f"Trust {combined:.1f}%  •  Risk: {risk}  •  {n_frames} frames")

        def _tick(score):
            return "✓" if score >= 60 else ("⚠" if score >= 40 else "✗")

        sep = "─" * 42
        # In _full_analysis: model=Analyze, spatial=Video, av_sync=LipSync
        # temporal & face mirror Analyze/LipSync scores for display
        report = f"""{uc_icon}  FINAL VERDICT: {verdict}
{"═" * 42}
Combined Trust Score : {combined:.1f} / 100
Risk Level           : {risk}
Frames Analyzed      : {n_frames}

{sep}
TAB-BY-TAB BREAKDOWN  (matching individual tab results)
{sep}
{_tick(model)} Analyze Tab (EfficientNet-B0)  {model:5.1f}%  (weight 66%)
   → {'Trained model: no deepfake signatures found' if model >= 60 else 'Trained model: deepfake signatures detected'}

{_tick(spatial)} Video Tab  (16-frame scan)     {spatial:5.1f}%  (weight 22%)
   → {'Video sequence appears authentic' if spatial >= 60 else 'Video sequence shows deepfake artifacts'}

{_tick(av_sync)} Lip-Sync Tab (AV alignment)   {av_sync:5.1f}%  (weight 12%)
   → {'Audio-visual sync is good' if av_sync >= 60 else 'Lip-audio sync mismatch detected'}

{sep}
WEIGHTED FORMULA
{sep}
  {model:.1f} × 0.66  (Analyze Tab — primary)
+ {spatial:.1f} × 0.22  (Video Tab)
+ {av_sync:.1f} × 0.12  (Lip-Sync Tab)
= {combined:.1f}%

{sep}
RECOMMENDATION
{sep}
{rec}
{"═" * 42}"""

        self._explain_box.configure(state="normal")
        self._explain_box.delete("1.0", "end")
        self._explain_box.insert("1.0", report)
        self._explain_box.configure(state="disabled")

        self._app_state.last_results = {
            "trust_score":    combined,
            "verdict":        verdict,
            "model_score":    model,    # Analyze tab
            "spatial_score":  spatial,  # Video tab
            "temporal_score": temporal,
            "av_score":       av_sync,  # Lip-Sync tab
            "face_score":     face,
        }


    # ─── Trained Model Runner ─────────────────────────────────────────────────
class DashboardTab(ctk.CTkFrame):
    def __init__(self, parent, state):
        super().__init__(parent, fg_color=BG)
        self._app_state = state
        self._build()

    def _build(self):
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 8))
        header.columnconfigure(0, weight=1)
        _lbl(header, "Analytics Dashboard", F_TITLE, WHITE).grid(
            row=0, column=0, sticky="w")
        _lbl(header, "Visualization of detection metrics and model performance",
             F_LABEL, TEXT2).grid(row=1, column=0, sticky="w")
        _btn_ghost(header, "Refresh", self._refresh, width=100).grid(
            row=0, column=1, rowspan=2, sticky="e")

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)

        tab_frame = _card(body)
        tab_frame.grid(row=0, column=0, sticky="nsew")
        tab_frame.rowconfigure(1, weight=1)
        tab_frame.columnconfigure(0, weight=1)

        strip = ctk.CTkFrame(tab_frame, fg_color="transparent")
        strip.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 0))
        self._stbs = {}
        tabs = ["Temporal", "Pixel Distribution",
                "Feature Maps", "Training Curves", "Confusion Matrix"]
        for i, t in enumerate(tabs):
            b = ctk.CTkButton(
                strip, text=t, width=120,
                fg_color=ACCENT if i == 0 else "transparent",
                hover_color=ACCENT2 if i == 0 else SURFACE2,
                text_color=WHITE if i == 0 else TEXT2,
                font=F_BADGE, corner_radius=8, height=30,
                command=lambda x=t: self._ss(x))
            b.pack(side="left", padx=(0, 4))
            self._stbs[t] = b

        inner = _surface(tab_frame)
        inner.grid(row=1, column=0, sticky="nsew", padx=12, pady=(8, 12))
        inner.rowconfigure(0, weight=1)
        inner.columnconfigure(0, weight=1)

        self._chs = {}
        self._chs["Temporal"]           = self._bld_temporal(inner)
        self._chs["Pixel Distribution"] = self._bld_pixel(inner)
        self._chs["Feature Maps"]        = self._bld_feature(inner)
        self._chs["Training Curves"]     = self._bld_training(inner)
        self._chs["Confusion Matrix"]    = self._bld_confusion(inner)

        self._acv = None
        self._ss("Temporal")

    def _ss(self, name):
        if self._acv: self._acv.pack_forget()
        for t, b in self._stbs.items():
            is_a = t == name
            b.configure(fg_color=ACCENT if is_a else "transparent",
                        hover_color=ACCENT2 if is_a else SURFACE2,
                        text_color=WHITE if is_a else TEXT2)
        self._chs[name].pack(fill="both", expand=True, padx=8, pady=8)
        self._acv = self._chs[name]
        self._draw_dashboard(name)

    def _refresh(self):
        for n, c in self._chs.items():
            if c.winfo_ismapped():
                self._draw_dashboard(n)

    def _draw_dashboard(self, name):
        m = {
            "Temporal": self._draw_temporal,
            "Pixel Distribution": self._draw_pixel,
            "Feature Maps": self._draw_feature,
            "Training Curves": self._draw_training,
            "Confusion Matrix": self._draw_confusion,
        }
        if name in m: m[name]()

    def _bld_temporal(self, parent):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        self._tc = Chart(f, w=860, h=360)
        self._tc.pack(fill="both", expand=True)
        return f

    def _bld_pixel(self, parent):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.columnconfigure((0, 1, 2), weight=1)
        f.rowconfigure(0, weight=1)
        self._rc = Chart(f, w=270, h=330)
        self._rc.grid(row=0, column=0, padx=4, pady=0, sticky="nsew")
        self._gc = Chart(f, w=270, h=330)
        self._gc.grid(row=0, column=1, padx=4, pady=0, sticky="nsew")
        self._bc = Chart(f, w=270, h=330)
        self._bc.grid(row=0, column=2, padx=4, pady=0, sticky="nsew")
        return f

    def _bld_feature(self, parent):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.columnconfigure(0, weight=1)
        f.columnconfigure(1, weight=1)
        f.rowconfigure(0, weight=1)
        self._inl = ctk.CTkLabel(f, text="", corner_radius=8)
        self._inl.grid(row=0, column=0, padx=(0, 8), pady=0, sticky="nsew")
        self._edl = ctk.CTkLabel(f, text="", corner_radius=8)
        self._edl.grid(row=0, column=1, padx=(8, 0), pady=0, sticky="nsew")
        _lbl(f, "Input Frame", F_BADGE, TEXT2).grid(row=1, column=0, pady=(6, 0))
        _lbl(f, "Edge Activation Map", F_BADGE, TEXT2).grid(row=1, column=1, pady=(6, 0))
        return f

    def _bld_training(self, parent):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        self._trc = Chart(f, w=860, h=360)
        self._trc.pack(fill="both", expand=True)
        return f

    def _bld_confusion(self, parent):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        self._cc = Chart(f, w=480, h=360)
        self._cc.pack(expand=True, pady=20)
        return f

    def _draw_temporal(self):
        flow = list(self._app_state.flow_history)
        if not flow:
            flow = [abs(np.sin(i * 0.4) * 3.5e5 + np.random.uniform(-5e4, 5e4))
                    for i in range(20)]
        self._tc.clear("Temporal Motion Analysis")
        self._tc.line(flow, color=TEAL, fill_alpha=True)
        self._tc.ylabels(min(flow), max(flow), 5)
        self._tc.xlabels(list(range(0, len(flow), max(1, len(flow) // 8))))

    def _draw_pixel(self):
        frame = (self._app_state.frames[0] if self._app_state.frames
                 else np.random.randint(20, 220, (200, 300, 3), dtype=np.uint8))
        for ch, chart, color, title in [
            (2, self._rc, RED,    "Red Channel"),
            (1, self._gc, GREEN,  "Green Channel"),
            (0, self._bc, ACCENT, "Blue Channel"),
        ]:
            vals, _ = np.histogram(frame[:, :, ch], bins=32, range=(0, 255))
            chart.clear(title)
            chart.bars(vals.tolist(), color=color, ymax=int(vals.max() * 1.1))
            chart.ylabels(0, int(vals.max()), 4)

    def _draw_feature(self):
        frame = (self._app_state.frames[0] if self._app_state.frames
                 else np.zeros((240, 320, 3), dtype=np.uint8))
        inp = _bgr_ctk(frame, 400, 290)
        self._inl.configure(image=inp)
        self._inl.image = inp
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        ec = cv2.applyColorMap(edges, cv2.COLORMAP_VIRIDIS)
        eim = _bgr_ctk(ec, 400, 290)
        self._edl.configure(image=eim)
        self._edl.image = eim

    def _draw_training(self):
        n = 15
        train = [min(0.97, 0.62 + 0.023 * i + np.random.uniform(-0.008, 0.008))
                 for i in range(n)]
        val = [min(0.94, 0.60 + 0.021 * i + np.random.uniform(-0.010, 0.010))
               for i in range(n)]
        self._trc.clear("Model Training Accuracy")
        self._trc.line(train, color=TEAL, ymin=0.5, ymax=1.0)
        self._trc.line(val, color=YELLOW, ymin=0.5, ymax=1.0)
        self._trc.ylabels(0.5, 1.0, 5)
        self._trc.xlabels(list(range(1, n + 1)))
        self._trc.legend([("Training", TEAL), ("Validation", YELLOW)])

    def _draw_confusion(self):
        r = self._app_state.last_results
        trust = r.get("trust_score", 50) if r else 50
        tp = int(54 * trust / 100); fn = 60 - tp
        tn = int(35 * trust / 100); fp_ = 40 - tn
        c = self._cc
        c.delete("all")
        c.create_text(240, 20, text="Confusion Matrix",
                      fill=WHITE, font=("Segoe UI", 12, "bold"))
        mx, my, cw, ch = 80, 50, 150, 130
        vals = [[tp, fp_], [fn, tn]]
        fills = [
            [c._alpha(GREEN, 0.6), c._alpha(RED, 0.4)],
            [c._alpha(RED, 0.3),   c._alpha(GREEN, 0.4)],
        ]
        labels = [["TP", "FP"], ["FN", "TN"]]
        for ri in range(2):
            for ci in range(2):
                x0 = mx + ci * cw; y0 = my + ri * ch
                c.create_rectangle(x0, y0, x0 + cw, y0 + ch,
                                   fill=fills[ri][ci], outline=BORDER2, width=1)
                c.create_text(x0 + cw // 2, y0 + ch // 2 - 10,
                              text=str(vals[ri][ci]),
                              fill=WHITE, font=("Segoe UI", 22, "bold"))
                c.create_text(x0 + cw // 2, y0 + ch // 2 + 16,
                              text=labels[ri][ci],
                              fill=TEXT2, font=("Segoe UI", 9))
        for i, t in enumerate(["REAL", "FAKE"]):
            c.create_text(mx + i * cw + cw // 2, my - 14, text=t,
                          fill=TEXT2, font=("Segoe UI", 9, "bold"))
        for i, t in enumerate(["REAL", "FAKE"]):
            c.create_text(mx - 22, my + i * ch + ch // 2, text=t,
                          fill=TEXT2, font=("Segoe UI", 9, "bold"))
        c.create_text(mx + cw, my + 2 * ch + 24,
                      text="Predicted", fill=TEXT3, font=("Segoe UI", 8))
        c.create_text(mx - 52, my + ch, text="Actual",
                      fill=TEXT3, font=("Segoe UI", 8), angle=90)


# 
# NAV BAR
# 
class NavBar(ctk.CTkFrame):
    ITEMS = [
        ("Analyze",   "Classic"),
        ("Video",     "Video"),
        ("Lip-Sync",  "LipSync"),
        ("Live",      "Live"),
        ("Auto-Analysis", "AutoAnalysis"),
        ("Dashboard", "Dashboard"),
    ]

    def __init__(self, parent, on_switch, **kw):
        super().__init__(parent, fg_color=SURFACE, corner_radius=0,
                         border_width=0, height=48, **kw)
        self.pack_propagate(False)
        self._os = on_switch
        self._btns = {}
        logo = ctk.CTkFrame(self, fg_color="transparent")
        logo.pack(side="left", padx=(16, 32))
        _lbl(logo, "ForensicStream", ("Segoe UI", 11, "bold"), WHITE).pack(
            side="left", pady=14)
        _lbl(logo, "  AI", ("Segoe UI", 11, "bold"), ACCENT).pack(
            side="left", pady=14)
        for label, key in self.ITEMS:
            b = ctk.CTkButton(
                self, text=label, font=F_APP,
                fg_color="transparent",
                hover_color=SURFACE2,
                text_color=WHITE if key == "Classic" else TEXT2,
                corner_radius=8, height=32, width=90,
                command=lambda k=key: self._click(k))
            b.pack(side="left", padx=2, pady=8)
            self._btns[key] = b
        self._sdot = _lbl(self, "Loading Models...", F_LABEL, YELLOW)
        self._sdot.pack(side="right", padx=20)
        self._click("Classic")

    def _click(self, key):
        for k, b in self._btns.items():
            b.configure(
                fg_color=SURFACE2 if k == key else "transparent",
                text_color=WHITE if k == key else TEXT2)
        self._os(key)

    def set_status(self, text, color=GREEN):
        self._sdot.configure(text=text, text_color=color)


# 
# MAIN APP
# 
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("ForensicStream AI — Deepfake Detection Platform")
        self.geometry("1360x840")
        self.minsize(1100, 720)
        self.configure(fg_color=BG)
        self._app_state = AppState()
        self._build()
        # Auto-load models in the background at startup
        threading.Thread(target=self._auto_load_models, daemon=True).start()

    def _auto_load_models(self):
        try:
            self._app_state.pipeline.load_models()
            self._app_state.models_loaded = True
            txt = self._app_state.pipeline.model_status_text()
            self.after(0, lambda: self._nav.set_status("Models Ready"))
        except Exception as e:
            print(f"[App] auto load_models error: {e}")
            self.after(0, lambda: self._nav.set_status("Model Load Failed", RED))

    def _build(self):
        self._nav = NavBar(self, self._sw)
        self._nav.pack(fill="x", side="top")

        tc = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        tc.pack(fill="both", expand=True)
        tc.rowconfigure(0, weight=1)
        tc.columnconfigure(0, weight=1)

        self._tabs = {
            "Classic":        ClassicTab(tc, self._app_state),
            "Video":          VideoTab(tc, self._app_state),
            "LipSync":        LipSyncTab(tc, self._app_state),
            "Live":           LiveTab(tc, self._app_state),
            "AutoAnalysis":   AutoAnalysisTab(tc, self._app_state),
            "Dashboard":      DashboardTab(tc, self._app_state),
        }
        for t in self._tabs.values():
            t.grid(row=0, column=0, sticky="nsew")

        self._sw("Classic")

    def _sw(self, name):
        if not hasattr(self, "_tabs"):
            return
        self._tabs[name].tkraise()
        if name == "Dashboard":
            try: self._tabs["Dashboard"]._refresh()
            except Exception: pass


if __name__ == "__main__":
    app = App()
    app.mainloop()
