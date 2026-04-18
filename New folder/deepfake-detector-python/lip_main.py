"""Deepfake Lip-Sync Detector — Python GUI
Uses MediaPipe FaceMesh + OpenCV + Librosa for detection
Run: python main.py
"""

import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from detector import DeepfakeDetector
from ui_components import ResultsPanel, TimelineChart, FindingsPanel


# ── App Theme ────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

COLORS = {
    "bg":         "#1a1a1a",
    "surface":    "#242424",
    "surface2":   "#2e2e2e",
    "border":     "#3a3a3a",
    "text":       "#f0f0f0",
    "text_muted": "#888888",
    "green":      "#1DB954",
    "red":        "#E24B4A",
    "amber":      "#EF9F27",
    "blue":       "#378ADD",
    "accent":     "#4A9EFF",
}


class DeepfakeApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Deepfake Lip-Sync Detector")
        self.geometry("900x750")
        self.minsize(800, 600)
        self.configure(fg_color=COLORS["bg"])

        self.detector = DeepfakeDetector()
        self.video_path = None
        self.analysis_thread = None

        self._build_ui()

    # ── UI Builder ───────────────────────────────────────────
    def _build_ui(self):
        # Header
        header = ctk.CTkFrame(self, fg_color=COLORS["surface"], corner_radius=0, height=64)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        ctk.CTkLabel(
            header, text="🔍  Deepfake Lip-Sync Detector",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=COLORS["text"]
        ).pack(side="left", padx=24, pady=16)

        ctk.CTkLabel(
            header, text="Analyzes audio-visual sync to detect AI-generated faces",
            font=ctk.CTkFont(size=12),
            text_color=COLORS["text_muted"]
        ).pack(side="left", padx=4, pady=16)

        # Main scroll area
        self.main_scroll = ctk.CTkScrollableFrame(self, fg_color=COLORS["bg"])
        self.main_scroll.pack(fill="both", expand=True, padx=0, pady=0)

        self._build_upload_section()
        self._build_status_section()
        self._build_results_section()

    def _build_upload_section(self):
        # Upload card
        upload_frame = ctk.CTkFrame(self.main_scroll, fg_color=COLORS["surface"], corner_radius=16)
        upload_frame.pack(fill="x", padx=24, pady=(24, 0))

        inner = ctk.CTkFrame(upload_frame, fg_color="transparent")
        inner.pack(fill="x", padx=24, pady=24)

        ctk.CTkLabel(inner, text="📁  Select Video File",
                     font=ctk.CTkFont(size=15, weight="bold"),
                     text_color=COLORS["text"]).pack(anchor="w", pady=(0, 6))

        ctk.CTkLabel(inner, text="Supports MP4, AVI, MOV, MKV, WebM",
                     font=ctk.CTkFont(size=12),
                     text_color=COLORS["text_muted"]).pack(anchor="w", pady=(0, 16))

        file_row = ctk.CTkFrame(inner, fg_color="transparent")
        file_row.pack(fill="x")

        self.file_entry = ctk.CTkEntry(
            file_row, placeholder_text="No file selected…",
            fg_color=COLORS["surface2"], border_color=COLORS["border"],
            text_color=COLORS["text"], height=40, font=ctk.CTkFont(size=12)
        )
        self.file_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))

        ctk.CTkButton(
            file_row, text="Browse", width=100, height=40,
            fg_color=COLORS["surface2"], hover_color=COLORS["border"],
            border_color=COLORS["border"], border_width=1,
            text_color=COLORS["text"], font=ctk.CTkFont(size=12),
            command=self._browse_file
        ).pack(side="left")

        # Analyze button
        self.analyze_btn = ctk.CTkButton(
            inner, text="▶  Analyze Video", height=44,
            fg_color=COLORS["accent"], hover_color="#3A8AE0",
            text_color="white", font=ctk.CTkFont(size=14, weight="bold"),
            corner_radius=10, command=self._start_analysis
        )
        self.analyze_btn.pack(fill="x", pady=(16, 0))

    def _build_status_section(self):
        self.status_frame = ctk.CTkFrame(self.main_scroll, fg_color=COLORS["surface"], corner_radius=12)
        self.status_frame.pack(fill="x", padx=24, pady=(12, 0))
        self.status_frame.pack_forget()

        status_inner = ctk.CTkFrame(self.status_frame, fg_color="transparent")
        status_inner.pack(fill="x", padx=20, pady=14)

        self.status_dot = ctk.CTkLabel(status_inner, text="●", font=ctk.CTkFont(size=14),
                                        text_color=COLORS["amber"], width=20)
        self.status_dot.pack(side="left")

        self.status_label = ctk.CTkLabel(status_inner, text="Processing…",
                                          font=ctk.CTkFont(size=13),
                                          text_color=COLORS["text_muted"])
        self.status_label.pack(side="left", padx=8)

        self.progress_bar = ctk.CTkProgressBar(self.status_frame, height=4,
                                                 fg_color=COLORS["surface2"],
                                                 progress_color=COLORS["accent"])
        self.progress_bar.pack(fill="x", padx=20, pady=(0, 14))
        self.progress_bar.set(0)

    def _build_results_section(self):
        self.results_frame = ctk.CTkFrame(self.main_scroll, fg_color="transparent")
        self.results_frame.pack(fill="x", padx=24, pady=(12, 24))
        self.results_frame.pack_forget()

    # ── Event handlers ───────────────────────────────────────
    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.MP4 *.AVI *.MOV"),
                       ("All files", "*.*")]
        )
        if path:
            self.video_path = path
            self.file_entry.delete(0, "end")
            self.file_entry.insert(0, path)
            # Clear old results
            for w in self.results_frame.winfo_children():
                w.destroy()
            self.results_frame.pack_forget()
            self.status_frame.pack_forget()

    def _start_analysis(self):
        if not self.video_path:
            messagebox.showwarning("No file", "Please select a video file first.")
            return

        self.analyze_btn.configure(state="disabled", text="Analyzing…")
        self.status_frame.pack(fill="x", padx=24, pady=(12, 0))
        self.progress_bar.set(0)
        self._set_status("Starting analysis…", COLORS["amber"])

        self.analysis_thread = threading.Thread(target=self._run_analysis, daemon=True)
        self.analysis_thread.start()

    def _run_analysis(self):
        try:
            result = self.detector.analyze(
                self.video_path,
                progress_callback=self._on_progress,
                status_callback=self._on_status
            )
            self.after(0, lambda: self._show_results(result))
        except Exception as e:
            self.after(0, lambda: self._on_error(str(e)))

    def _on_progress(self, value, message=""):
        self.after(0, lambda: self.progress_bar.set(value))
        if message:
            self.after(0, lambda: self._set_status(message, COLORS["amber"]))

    def _on_status(self, message, color=None):
        self.after(0, lambda: self._set_status(message, color or COLORS["amber"]))

    def _set_status(self, message, color):
        self.status_label.configure(text=message)
        self.status_dot.configure(text_color=color)

    def _on_error(self, error_msg):
        self._set_status(f"Error: {error_msg}", COLORS["red"])
        self.analyze_btn.configure(state="normal", text="▶  Analyze Video")
        messagebox.showerror("Analysis Failed", error_msg)

    def _show_results(self, result):
        self._set_status("Analysis complete ✓", COLORS["green"])
        self.progress_bar.set(1.0)
        self.analyze_btn.configure(state="normal", text="▶  Analyze Video")

        # Clear old results
        for w in self.results_frame.winfo_children():
            w.destroy()

        self.results_frame.pack(fill="x", padx=24, pady=(12, 24))

        # Verdict card
        ResultsPanel(self.results_frame, result, COLORS).pack(fill="x", pady=(0, 12))

        # Timeline chart
        TimelineChart(self.results_frame, result["lip_data"], result["audio_data"], COLORS).pack(fill="x", pady=(0, 12))

        # Findings
        FindingsPanel(self.results_frame, result["findings"], COLORS).pack(fill="x", pady=(0, 0))


# ── Entry point ──────────────────────────────────────────────
if __name__ == "__main__":
    app = DeepfakeApp()
    app.mainloop()
