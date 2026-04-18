"""
ui_components.py — Reusable GUI components for the results section

Components:
  ResultsPanel  — Verdict card with confidence score + stats grid
  TimelineChart — Matplotlib canvas showing lip vs audio over time
  FindingsPanel — List of detection findings with severity indicators
"""

import tkinter as tk
import customtkinter as ctk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np


# ── 1. Results / Verdict Panel ───────────────────────────────
class ResultsPanel(ctk.CTkFrame):
    def __init__(self, parent, result, colors, **kwargs):
        super().__init__(parent, fg_color=colors["surface"], corner_radius=16, **kwargs)
        self.colors = colors
        self._build(result)

    def _build(self, r):
        C = self.colors
        is_fake = r["is_deepfake"]
        conf    = r["confidence"]

        verdict_color = C["red"] if is_fake else C["green"]
        verdict_icon  = "⚠  Likely Deepfake" if is_fake else "✓  Likely Authentic"

        pad = ctk.CTkFrame(self, fg_color="transparent")
        pad.pack(fill="x", padx=24, pady=20)

        # ── Verdict row ──────────────────────────────────────
        top = ctk.CTkFrame(pad, fg_color="transparent")
        top.pack(fill="x", pady=(0, 16))

        ctk.CTkLabel(
            top, text=verdict_icon,
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=verdict_color
        ).pack(side="left")

        conf_frame = ctk.CTkFrame(top, fg_color="transparent")
        conf_frame.pack(side="right")
        ctk.CTkLabel(conf_frame, text="Deepfake Confidence",
                     font=ctk.CTkFont(size=11), text_color=C["text_muted"]).pack(anchor="e")
        ctk.CTkLabel(conf_frame, text=f"{conf}%",
                     font=ctk.CTkFont(size=28, weight="bold"),
                     text_color=verdict_color).pack(anchor="e")

        # ── Progress bar ─────────────────────────────────────
        bar = ctk.CTkProgressBar(pad, height=8, fg_color=C["surface2"],
                                  progress_color=verdict_color)
        bar.pack(fill="x", pady=(0, 20))
        bar.set(conf / 100)

        # ── Stats grid ───────────────────────────────────────
        grid = ctk.CTkFrame(pad, fg_color="transparent")
        grid.pack(fill="x")
        grid.columnconfigure((0, 1, 2, 3), weight=1)

        stats = [
            ("Avg Lip Aperture", f"{r['avg_aperture']*100:.1f}%"),
            ("Sync Score",       f"{r['sync_score']}%"),
            ("Frames Analyzed",  str(r["frames"])),
            ("Temporal Lag",     f"{r['lag_ms']:.0f} ms"),
        ]

        for col, (label, value) in enumerate(stats):
            card = ctk.CTkFrame(grid, fg_color=C["surface2"], corner_radius=10)
            card.grid(row=0, column=col, padx=(0 if col == 0 else 8, 0), sticky="nsew")
            ctk.CTkLabel(card, text=label, font=ctk.CTkFont(size=11),
                         text_color=C["text_muted"]).pack(pady=(12, 2))
            ctk.CTkLabel(card, text=value, font=ctk.CTkFont(size=20, weight="bold"),
                         text_color=C["text"]).pack(pady=(0, 12))


# ── 2. Timeline Chart ────────────────────────────────────────
class TimelineChart(ctk.CTkFrame):
    def __init__(self, parent, lip_data, audio_data, colors, **kwargs):
        super().__init__(parent, fg_color=colors["surface"], corner_radius=16, **kwargs)
        self.colors = colors
        self._build(lip_data, audio_data)

    def _build(self, lip_data, audio_data):
        C = self.colors
        n = min(len(lip_data), len(audio_data))
        t = np.linspace(0, n / 15, n)

        ctk.CTkLabel(self, text="Lip Aperture vs Audio Energy Over Time",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=C["text"]).pack(anchor="w", padx=20, pady=(16, 0))

        # Matplotlib figure
        fig, ax = plt.subplots(figsize=(8, 2.2))
        fig.patch.set_facecolor("#242424")
        ax.set_facecolor("#1e1e1e")

        ax.plot(t, audio_data[:n], color="#378ADD", linewidth=1.5, label="Audio energy", alpha=0.9)
        ax.plot(t, lip_data[:n],   color="#1DB954", linewidth=1.5, label="Lip aperture",  alpha=0.9)

        ax.set_xlim(0, t[-1] if len(t) else 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Time (s)", color="#888888", fontsize=9)
        ax.tick_params(colors="#666666", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        ax.grid(axis="y", color="#333333", linewidth=0.5, alpha=0.6)

        audio_patch = mpatches.Patch(color="#378ADD", label="Audio energy")
        lip_patch   = mpatches.Patch(color="#1DB954", label="Lip aperture")
        ax.legend(handles=[audio_patch, lip_patch], loc="upper right",
                  fontsize=8, facecolor="#2a2a2a", edgecolor="#444", labelcolor="#aaaaaa")

        fig.tight_layout(pad=1.2)

        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="x", padx=16, pady=(8, 16))
        plt.close(fig)


# ── 3. Findings Panel ────────────────────────────────────────
class FindingsPanel(ctk.CTkFrame):
    def __init__(self, parent, findings, colors, **kwargs):
        super().__init__(parent, fg_color=colors["surface"], corner_radius=16, **kwargs)
        self.colors = colors
        self._build(findings)

    def _build(self, findings):
        C = self.colors

        ctk.CTkLabel(self, text="Detection Findings",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=C["text"]).pack(anchor="w", padx=20, pady=(16, 8))

        severity_colors = {
            "high":   C["red"],
            "medium": C["amber"],
            "ok":     C["green"],
        }
        severity_icons = {
            "high":   "✗",
            "medium": "△",
            "ok":     "✓",
        }

        for i, finding in enumerate(findings):
            color = severity_colors.get(finding["severity"], C["text_muted"])
            icon  = severity_icons.get(finding["severity"], "•")

            row = ctk.CTkFrame(self, fg_color="transparent")
            row.pack(fill="x", padx=20, pady=(0, 12 if i < len(findings)-1 else 16))

            # Icon badge
            badge = ctk.CTkFrame(row, fg_color=color, corner_radius=99, width=26, height=26)
            badge.pack(side="left", padx=(0, 12))
            badge.pack_propagate(False)
            ctk.CTkLabel(badge, text=icon, font=ctk.CTkFont(size=11, weight="bold"),
                         text_color="white").place(relx=0.5, rely=0.5, anchor="center")

            # Text
            text_col = ctk.CTkFrame(row, fg_color="transparent")
            text_col.pack(side="left", fill="x", expand=True)
            ctk.CTkLabel(text_col, text=finding["label"],
                         font=ctk.CTkFont(size=13, weight="bold"),
                         text_color=C["text"], anchor="w").pack(fill="x")
            ctk.CTkLabel(text_col, text=finding["detail"],
                         font=ctk.CTkFont(size=11),
                         text_color=C["text_muted"], anchor="w",
                         wraplength=580).pack(fill="x")

            # Divider
            if i < len(findings) - 1:
                div = ctk.CTkFrame(self, fg_color=C["border"], height=1)
                div.pack(fill="x", padx=20, pady=(0, 4))
