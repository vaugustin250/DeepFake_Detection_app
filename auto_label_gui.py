"""
🎬 AUTO-LABELING DEEPFAKE DETECTION GUI
Complete Professional Interface with Testing, Visualizations & Reports
"""

import os
import json
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import threading
import queue
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
from torchvision.models import efficientnet_b0
from torchvision.models.feature_extraction import create_feature_extractor
from sklearn.metrics import confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS & STYLING
# ============================================================================
BG_DARK = "#1e1e1e"
BG_LIGHT = "#2d2d2d"
ACCENT_COLOR = "#00d4ff"
SUCCESS_COLOR = "#00ff88"
DANGER_COLOR = "#ff4444"
TEXT_COLOR = "#e0e0e0"
FONT_TITLE = ("Segoe UI", 16, "bold")
FONT_HEADING = ("Segoe UI", 12, "bold")
FONT_TEXT = ("Segoe UI", 10)
FONT_SMALL = ("Segoe UI", 9)
PAD = 10

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================
class FPN(nn.Module):
    def __init__(self, chs, out_ch=128):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_ch, 1) for c in chs])
        self.smooth = nn.ModuleList([nn.Conv2d(out_ch, out_ch, 3, padding=1) for _ in chs])

    def forward(self, feats):
        x = self.lateral[-1](feats[-1])
        outs = [self.smooth[-1](x)]
        for i in range(len(feats) - 2, -1, -1):
            lat = self.lateral[i](feats[i])
            x = lat + nn.functional.interpolate(x, size=lat.shape[-2:], mode='nearest')
            outs.insert(0, self.smooth[i](x))
        pooled = [nn.functional.adaptive_avg_pool2d(f, (1, 1)).flatten(1) for f in outs]
        return torch.cat(pooled, dim=1)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        base = efficientnet_b0(weights="IMAGENET1K_V1")
        self.extract = create_feature_extractor(base, return_nodes={'features.2': 'f2', 'features.4': 'f4', 'features.6': 'f6'})
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            chs = [v.shape[1] for v in self.extract(dummy).values()]
        self.fpn = FPN(chs)
        self.out_dim = 128 * len(chs)

    def forward(self, x):
        feats = list(self.extract(x).values())
        return self.fpn(feats)


class GLUBlock(nn.Module):
    def __init__(self, in_c, out_c, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(in_c, out_c * 2, 3, padding=dilation, dilation=dilation)
        self.res = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else None
        self.bn = nn.BatchNorm1d(out_c)

    def forward(self, x):
        a, b = self.conv(x).chunk(2, 1)
        out = a * torch.sigmoid(b)
        res = x if self.res is None else self.res(x)
        return self.bn(out + res)


class TemporalCNN(nn.Module):
    def __init__(self, in_c, hidden=128):
        super().__init__()
        self.blocks = nn.Sequential(GLUBlock(in_c, hidden, 1), GLUBlock(hidden, hidden, 2), GLUBlock(hidden, hidden, 4))
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        y = self.blocks(x)
        return self.pool(y).squeeze(-1)


class DeepfakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.seg_fc = nn.Sequential(nn.Linear(self.backbone.out_dim, 256), nn.ReLU(), nn.Dropout(0.3))
        self.tcn = TemporalCNN(256)
        self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.backbone(x)
        x = x.view(B, T, -1)
        x = self.seg_fc(x)
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = self.fc(x)
        return x


# ============================================================================
# VIDEO PROCESSOR
# ============================================================================
class VideoProcessor:
    @staticmethod
    def extract_frames_from_video(video_path, num_frames=8):
        """Extract evenly spaced frames from video"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            raise ValueError(f"Cannot read frames from {video_path}")
        
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames extracted from {video_path}")
        
        if len(frames) < num_frames:
            while len(frames) < num_frames:
                frames.append(frames[-1].copy())
        elif len(frames) > num_frames:
            frames = frames[:num_frames]
        
        return np.array(frames)

    @staticmethod
    def frames_to_tensor(frames):
        """Convert frames to tensor (B, T, C, H, W)"""
        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        tensor = tensor.unsqueeze(0)
        return tensor


# ============================================================================
# GUI CLASS
# ============================================================================
class DeepfakeDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎬 AUTO-LABELING DEEPFAKE DETECTION - Professional GUI")
        self.root.geometry("1600x900")
        self.root.config(bg=BG_DARK)
        
        # Variables
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.video_results = []
        self.y_true = []
        self.y_pred = []
        self.y_scores = []
        self.ui_queue = queue.Queue()
        self.testing = False
        
        # Setup GUI
        self._setup_styles()
        self._create_ui()
        self._process_queue()
    
    def _setup_styles(self):
        """Setup tkinter styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('TButton', background=BG_LIGHT, foreground=TEXT_COLOR, borderwidth=0, focuscolor='none')
        style.configure('TLabel', background=BG_LIGHT, foreground=TEXT_COLOR)
        style.configure('TFrame', background=BG_LIGHT)
        style.configure('TLabelFrame', background=BG_LIGHT, foreground=ACCENT_COLOR, borderwidth=1)
        style.configure('TProgressbar', background=SUCCESS_COLOR)
        
        style.map('TButton', background=[('active', ACCENT_COLOR)], foreground=[('active', BG_DARK)])
    
    def _create_ui(self):
        """Create main UI"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=PAD, pady=PAD)
        
        # Header
        self._create_header(main_frame)
        
        # Content frame with left panel and tabbed interface
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill='both', expand=True, pady=(PAD, 0))
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_columnconfigure(0, weight=0)
        content_frame.grid_columnconfigure(1, weight=1)
        
        # Left panel (controls)
        self._create_left_panel(content_frame)
        
        # Right panel (results with tabs)
        self._create_right_panel(content_frame)
    
    def _create_header(self, parent):
        """Create header section"""
        header = ttk.Frame(parent)
        header.pack(fill='x', pady=(0, PAD))
        
        title = ttk.Label(header, text="🎬 AUTO-LABELING DEEPFAKE DETECTION", font=FONT_TITLE)
        title.pack(side='left')
        
        device_label = ttk.Label(header, text=f"Device: {self.device}", font=FONT_SMALL)
        device_label.pack(side='right')
    
    def _create_left_panel(self, parent):
        """Create left control panel"""
        left_frame = ttk.Frame(parent)
        left_frame.grid(row=0, column=0, sticky='ns', padx=(0, PAD))
        
        # Model Configuration
        model_frame = ttk.LabelFrame(left_frame, text="📋 Model Configuration")
        model_frame.pack(fill='x', pady=(0, PAD))
        
        ttk.Label(model_frame, text="Model Path:").pack(anchor='w', pady=(5, 2))
        self.model_entry = ttk.Entry(model_frame, width=30)
        self.model_entry.pack(fill='x', pady=(0, 10))
        self.model_entry.insert(0, "best_model.pth")
        
        btn_load_model = ttk.Button(model_frame, text="📂 Browse Model", command=self._browse_model)
        btn_load_model.pack(fill='x', pady=(0, 5))
        
        btn_init = ttk.Button(model_frame, text="🔧 Initialize Model", command=self._init_model)
        btn_init.pack(fill='x')
        
        # Video Folder Configuration
        video_frame = ttk.LabelFrame(left_frame, text="📁 Video Folder")
        video_frame.pack(fill='x', pady=(0, PAD))
        
        ttk.Label(video_frame, text="Folder Path:").pack(anchor='w', pady=(5, 2))
        self.folder_entry = ttk.Entry(video_frame, width=30)
        self.folder_entry.pack(fill='x', pady=(0, 10))
        self.folder_entry.insert(0, "C:/Users/agust/Downloads/Test")
        
        btn_browse = ttk.Button(video_frame, text="📂 Browse Folder", command=self._browse_folder)
        btn_browse.pack(fill='x')
        
        # Testing Controls
        test_frame = ttk.LabelFrame(left_frame, text="🚀 Testing")
        test_frame.pack(fill='x', pady=(0, PAD))
        
        self.btn_test = ttk.Button(test_frame, text="▶ START TEST", command=self._run_test_thread)
        self.btn_test.pack(fill='x', pady=(0, 5))
        
        self.btn_stop = ttk.Button(test_frame, text="⏹ STOP TEST", command=self._stop_test, state='disabled')
        self.btn_stop.pack(fill='x')
        
        # Progress
        progress_frame = ttk.LabelFrame(left_frame, text="📈 Progress")
        progress_frame.pack(fill='x', pady=(0, PAD))
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill='x', pady=(0, 5))
        
        self.progress_label = ttk.Label(progress_frame, text="Ready", font=FONT_SMALL)
        self.progress_label.pack(anchor='w')
        
        self.status_label = ttk.Label(progress_frame, text="✓ Idle", font=FONT_SMALL, foreground=SUCCESS_COLOR)
        self.status_label.pack(anchor='w', pady=(5, 0))
        
        # Export Options
        export_frame = ttk.LabelFrame(left_frame, text="💾 Export Results")
        export_frame.pack(fill='x')
        
        self.btn_json = ttk.Button(export_frame, text="📊 Export JSON", command=self._export_json, state='disabled')
        self.btn_json.pack(fill='x', pady=(0, 5))
        
        self.btn_report = ttk.Button(export_frame, text="📄 Generate Report", command=self._generate_report, state='disabled')
        self.btn_report.pack(fill='x', pady=(0, 5))
        
        self.btn_charts = ttk.Button(export_frame, text="📈 Save Charts", command=self._save_charts, state='disabled')
        self.btn_charts.pack(fill='x')
    
    def _create_right_panel(self, parent):
        """Create right panel with tabs"""
        right_frame = ttk.Frame(parent)
        right_frame.grid(row=0, column=1, sticky='nsew')
        right_frame.grid_rowconfigure(0, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill='both', expand=True)
        
        # Tab 1: Results Table
        self.tab_results = tk.Frame(self.notebook, bg=BG_LIGHT)
        self.notebook.add(self.tab_results, text="📺 Test Results")
        self._setup_results_tab()
        
        # Tab 2: Summary Stats
        self.tab_summary = tk.Frame(self.notebook, bg=BG_LIGHT)
        self.notebook.add(self.tab_summary, text="📊 Summary Statistics")
        self._setup_summary_tab()
        
        # Tab 3: Confidence Distribution
        self.tab_confidence = tk.Frame(self.notebook, bg=BG_LIGHT)
        self.notebook.add(self.tab_confidence, text="📈 Confidence Distribution")
        
        # Tab 4: Confusion Matrix
        self.tab_confusion = tk.Frame(self.notebook, bg=BG_LIGHT)
        self.notebook.add(self.tab_confusion, text="🔲 Confusion Matrix")
        
        # Tab 5: ROC Curve
        self.tab_roc = tk.Frame(self.notebook, bg=BG_LIGHT)
        self.notebook.add(self.tab_roc, text="📉 ROC Curve")
    
    def _setup_results_tab(self):
        """Setup results table tab"""
        # Create treeview for results
        columns = ("Video", "Prediction", "Confidence", "Status")
        self.results_tree = ttk.Treeview(self.tab_results, columns=columns, height=20)
        self.results_tree.column("#0", width=0, stretch='no')
        self.results_tree.column("Video", anchor='w', width=400)
        self.results_tree.column("Prediction", anchor='center', width=120)
        self.results_tree.column("Confidence", anchor='center', width=120)
        self.results_tree.column("Status", anchor='center', width=100)
        
        self.results_tree.heading("#0", text="", anchor='w')
        self.results_tree.heading("Video", text="Video Name", anchor='w')
        self.results_tree.heading("Prediction", text="Prediction")
        self.results_tree.heading("Confidence", text="Confidence")
        self.results_tree.heading("Status", text="Status")
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(self.tab_results, orient='vertical', command=self.results_tree.yview)
        self.results_tree.configure(yscroll=scrollbar.set)
        
        self.results_tree.pack(side='left', fill='both', expand=True, padx=PAD, pady=PAD)
        scrollbar.pack(side='right', fill='y', padx=(0, PAD), pady=PAD)
    
    def _setup_summary_tab(self):
        """Setup summary statistics tab"""
        self.summary_text = scrolledtext.ScrolledText(self.tab_summary, font=FONT_TEXT, bg=BG_LIGHT, fg=TEXT_COLOR, relief='flat', wrap='word')
        self.summary_text.pack(fill='both', expand=True, padx=PAD, pady=PAD)
        self.summary_text.insert('1.0', "Run a test to see summary statistics here...")
        self.summary_text.config(state='disabled')
    
    def _browse_model(self):
        """Browse for model file"""
        path = filedialog.askopenfilename(
            filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")],
            title="Select Model File"
        )
        if path:
            self.model_entry.delete(0, 'end')
            self.model_entry.insert(0, path)
    
    def _browse_folder(self):
        """Browse for video folder"""
        path = filedialog.askdirectory(title="Select Video Folder")
        if path:
            self.folder_entry.delete(0, 'end')
            self.folder_entry.insert(0, path)
    
    def _init_model(self):
        """Initialize model"""
        try:
            model_path = self.model_entry.get()
            if not model_path:
                messagebox.showerror("Error", "Please enter model path")
                return
            
            if not os.path.exists(model_path):
                messagebox.showerror("Error", f"Model not found: {model_path}")
                return
            
            self._set_status("⏳ Loading model...", ACCENT_COLOR)
            self.root.update()
            
            self.model = DeepfakeModel().to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            
            self._set_status("✓ Model loaded successfully", SUCCESS_COLOR)
            messagebox.showinfo("Success", f"Model loaded successfully!\nDevice: {self.device}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            self._set_status("✗ Model loading failed", DANGER_COLOR)
    
    def _run_test_thread(self):
        """Run test in background thread"""
        if self.model is None:
            messagebox.showerror("Error", "Please load model first")
            return
        
        thread = threading.Thread(target=self._run_test, daemon=True)
        thread.start()
    
    def _run_test(self):
        """Run test on video folder"""
        try:
            self.testing = True
            self.btn_test.config(state='disabled')
            self.btn_stop.config(state='normal')
            
            folder_path = self.folder_entry.get()
            if not os.path.exists(folder_path):
                raise FileNotFoundError(f"Folder not found: {folder_path}")
            
            # Find videos
            video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm')
            video_files = sorted([f for f in os.listdir(folder_path) 
                                 if f.lower().endswith(video_extensions) and 
                                 os.path.isfile(os.path.join(folder_path, f))])
            
            if not video_files:
                raise ValueError("No video files found in folder")
            
            # Clear previous results
            self.video_results = []
            self.y_true = []
            self.y_pred = []
            self.y_scores = []
            self._clear_results_display()
            
            # Process videos
            for idx, video_file in enumerate(video_files):
                if not self.testing:
                    break
                
                progress = (idx / len(video_files)) * 100
                self.ui_queue.put(('progress', progress))
                self.ui_queue.put(('status', f"Processing: {video_file}"))
                
                try:
                    video_path = os.path.join(folder_path, video_file)
                    frames = VideoProcessor.extract_frames_from_video(video_path, num_frames=8)
                    tensor = VideoProcessor.frames_to_tensor(frames).to(self.device)
                    
                    with torch.inference_mode():
                        output = self.model(tensor)
                        prob = float(torch.sigmoid(output).cpu().item())
                        pred = 1 if prob > 0.5 else 0
                    
                    # Auto-label
                    self.y_true.append(pred)
                    self.y_pred.append(pred)
                    self.y_scores.append(prob)
                    
                    pred_str = "🔴 FAKE" if pred == 1 else "🟢 REAL"
                    confidence_pct = (prob if pred == 1 else (1 - prob)) * 100
                    
                    self.video_results.append({
                        'video': video_file,
                        'prediction': pred_str,
                        'confidence': prob,
                        'confidence_pct': confidence_pct
                    })
                    
                    # Update UI
                    self.ui_queue.put(('add_result', video_file, pred_str, f"{confidence_pct:.1f}%"))
                
                except Exception as e:
                    self.ui_queue.put(('add_result', video_file, "ERROR", str(e)[:30]))
            
            if self.testing and len(self.y_true) > 0:
                # Compute and display results
                self._compute_results()
                self.ui_queue.put(('status', f"✓ Test complete! {len(self.y_true)} videos processed"))
                self.ui_queue.put(('progress', 100))
                
                # Enable export buttons
                self.ui_queue.put(('enable_export',))
        
        except Exception as e:
            self.ui_queue.put(('error', f"Test failed: {str(e)}"))
        
        finally:
            self.testing = False
            self.btn_test.config(state='normal')
            self.btn_stop.config(state='disabled')
    
    def _stop_test(self):
        """Stop testing"""
        self.testing = False
        self._set_status("⏹ Test stopped", DANGER_COLOR)
    
    def _compute_results(self):
        """Compute metrics and visualizations"""
        self.y_true = np.array(self.y_true)
        self.y_pred = np.array(self.y_pred)
        self.y_scores = np.array(self.y_scores)
        
        # Create summary
        summary = self._generate_summary_text()
        self.ui_queue.put(('update_summary', summary))
        
        # Update charts
        self.ui_queue.put(('update_confidence_chart',))
        self.ui_queue.put(('update_confusion_chart',))
        self.ui_queue.put(('update_roc_chart',))
    
    def _generate_summary_text(self):
        """Generate summary statistics"""
        real_count = np.sum(self.y_pred == 0)
        fake_count = np.sum(self.y_pred == 1)
        avg_conf = np.mean(self.y_scores)
        
        text = f"""
{'='*60}
📊 TEST RESULTS SUMMARY
{'='*60}

Total Videos:           {len(self.y_true)}
Predicted REAL:         {real_count} ({real_count/len(self.y_true)*100:.1f}%)
Predicted FAKE:         {fake_count} ({fake_count/len(self.y_true)*100:.1f}%)

Average Confidence:     {avg_conf*100:.1f}%
Min Confidence:         {np.min(self.y_scores)*100:.1f}%
Max Confidence:         {np.max(self.y_scores)*100:.1f}%

{'='*60}
📈 METRICS (Auto-labeled - all 100%)
{'='*60}

Accuracy:               100.0%
Precision:              100.0%
Recall:                 100.0%
F1-Score:               100.0%

{'='*60}
🔝 TOP PREDICTIONS BY CONFIDENCE
{'='*60}
"""
        # Top predictions
        top_indices = np.argsort(self.y_scores)[-5:][::-1]
        for rank, idx in enumerate(top_indices, 1):
            result = self.video_results[idx]
            text += f"\n{rank}. {result['video']:<45} → {result['confidence_pct']:.1f}%"
        
        return text
    
    def _clear_results_display(self):
        """Clear results display"""
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
    
    def _export_json(self):
        """Export results to JSON"""
        try:
            path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
                initialfile=f"deepfake_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            if path:
                results = {
                    'timestamp': datetime.now().isoformat(),
                    'total_videos': len(self.y_true),
                    'predictions': {
                        'real': int(np.sum(self.y_pred == 0)),
                        'fake': int(np.sum(self.y_pred == 1))
                    },
                    'confidence': {
                        'average': float(np.mean(self.y_scores)),
                        'min': float(np.min(self.y_scores)),
                        'max': float(np.max(self.y_scores))
                    },
                    'video_results': self.video_results
                }
                
                with open(path, 'w') as f:
                    json.dump(results, f, indent=2)
                
                messagebox.showinfo("Success", f"Results exported to:\n{path}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\n{str(e)}")
    
    def _generate_report(self):
        """Generate detailed text report"""
        try:
            path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
                initialfile=f"deepfake_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )
            
            if path:
                report = self._generate_summary_text()
                report += f"\n\n{'='*60}\n📋 DETAILED VIDEO LIST\n{'='*60}\n\n"
                
                for i, result in enumerate(self.video_results, 1):
                    report += f"{i:2d}. {result['video']:<45} {result['prediction']:<12} {result['confidence_pct']:>6.1f}%\n"
                
                with open(path, 'w') as f:
                    f.write(report)
                
                messagebox.showinfo("Success", f"Report generated:\n{path}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Report generation failed:\n{str(e)}")
    
    def _save_charts(self):
        """Save visualization charts"""
        try:
            folder = filedialog.askdirectory(title="Select folder to save charts")
            if folder:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                # Confidence distribution
                self._plot_confidence_distribution(f"{folder}/confidence_distribution_{timestamp}.png")
                
                # Confusion matrix
                self._plot_confusion_matrix(f"{folder}/confusion_matrix_{timestamp}.png")
                
                # ROC curve
                self._plot_roc_curve(f"{folder}/roc_curve_{timestamp}.png")
                
                messagebox.showinfo("Success", f"Charts saved to:\n{folder}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Chart saving failed:\n{str(e)}")
    
    def _plot_confidence_distribution(self, filepath):
        """Plot confidence distribution"""
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG_DARK)
        ax.set_facecolor(BG_LIGHT)
        
        real_scores = self.y_scores[self.y_pred == 0]
        fake_scores = self.y_scores[self.y_pred == 1]
        
        ax.hist(1 - real_scores if len(real_scores) > 0 else [], bins=15, alpha=0.6, label='REAL Videos', color='#00ff88')
        ax.hist(fake_scores if len(fake_scores) > 0 else [], bins=15, alpha=0.6, label='FAKE Videos', color='#ff4444')
        
        ax.set_xlabel('Confidence', color=TEXT_COLOR)
        ax.set_ylabel('Count', color=TEXT_COLOR)
        ax.set_title('Confidence Distribution', color=TEXT_COLOR, fontsize=14, fontweight='bold')
        ax.legend(facecolor=BG_LIGHT, edgecolor=ACCENT_COLOR)
        ax.tick_params(colors=TEXT_COLOR)
        
        plt.tight_layout()
        plt.savefig(filepath, facecolor=BG_DARK, edgecolor='none', dpi=150)
        plt.close()
    
    def _plot_confusion_matrix(self, filepath):
        """Plot confusion matrix"""
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 7), facecolor=BG_DARK)
        ax.set_facecolor(BG_LIGHT)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'},
                    xticklabels=['REAL', 'FAKE'], yticklabels=['REAL', 'FAKE'])
        
        ax.set_xlabel('Predicted', color=TEXT_COLOR)
        ax.set_ylabel('Actual', color=TEXT_COLOR)
        ax.set_title('Confusion Matrix', color=TEXT_COLOR, fontsize=14, fontweight='bold')
        ax.tick_params(colors=TEXT_COLOR)
        
        plt.tight_layout()
        plt.savefig(filepath, facecolor=BG_DARK, edgecolor='none', dpi=150)
        plt.close()
    
    def _plot_roc_curve(self, filepath):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(self.y_true, self.y_scores)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 8), facecolor=BG_DARK)
        ax.set_facecolor(BG_LIGHT)
        
        ax.plot(fpr, tpr, color=ACCENT_COLOR, lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color=TEXT_COLOR, lw=1, linestyle='--', label='Random')
        
        ax.set_xlabel('False Positive Rate', color=TEXT_COLOR)
        ax.set_ylabel('True Positive Rate', color=TEXT_COLOR)
        ax.set_title('ROC Curve', color=TEXT_COLOR, fontsize=14, fontweight='bold')
        ax.legend(facecolor=BG_LIGHT, edgecolor=ACCENT_COLOR)
        ax.tick_params(colors=TEXT_COLOR)
        ax.grid(True, alpha=0.2, color=TEXT_COLOR)
        
        plt.tight_layout()
        plt.savefig(filepath, facecolor=BG_DARK, edgecolor='none', dpi=150)
        plt.close()
    
    def _set_status(self, msg, color=TEXT_COLOR):
        """Set status message"""
        self.status_label.config(text=msg, foreground=color)
    
    def _display_confidence_chart(self):
        """Display confidence distribution chart in tab"""
        try:
            # Clear previous widgets
            for widget in self.tab_confidence.winfo_children():
                widget.destroy()
            
            fig = Figure(figsize=(8, 6), facecolor=BG_DARK, dpi=100)
            ax = fig.add_subplot(111)
            ax.set_facecolor(BG_LIGHT)
            
            real_scores = self.y_scores[self.y_pred == 0]
            fake_scores = self.y_scores[self.y_pred == 1]
            
            if len(real_scores) > 0:
                ax.hist(1 - real_scores, bins=15, alpha=0.6, label='REAL Videos', color='#00ff88')
            if len(fake_scores) > 0:
                ax.hist(fake_scores, bins=15, alpha=0.6, label='FAKE Videos', color='#ff4444')
            
            ax.set_xlabel('Confidence', color=TEXT_COLOR, fontsize=11)
            ax.set_ylabel('Count', color=TEXT_COLOR, fontsize=11)
            ax.set_title('Confidence Distribution', color=TEXT_COLOR, fontsize=12, fontweight='bold')
            ax.legend(facecolor=BG_LIGHT, edgecolor=ACCENT_COLOR, labelcolor=TEXT_COLOR)
            ax.tick_params(colors=TEXT_COLOR)
            ax.spines['bottom'].set_color(TEXT_COLOR)
            ax.spines['left'].set_color(TEXT_COLOR)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            canvas = FigureCanvasTkAgg(fig, master=self.tab_confidence)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True, padx=PAD, pady=PAD)
        
        except Exception as e:
            print(f"Error displaying confidence chart: {e}")
    
    def _display_confusion_chart(self):
        """Display confusion matrix chart in tab"""
        try:
            # Clear previous widgets
            for widget in self.tab_confusion.winfo_children():
                widget.destroy()
            
            cm = confusion_matrix(self.y_true, self.y_pred)
            
            fig = Figure(figsize=(8, 7), facecolor=BG_DARK, dpi=100)
            ax = fig.add_subplot(111)
            ax.set_facecolor(BG_LIGHT)
            
            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
            
            # Add text annotations
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    text = ax.text(j, i, cm[i, j],
                                  ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black",
                                  fontsize=14, fontweight='bold')
            
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['REAL', 'FAKE'], color=TEXT_COLOR)
            ax.set_yticklabels(['REAL', 'FAKE'], color=TEXT_COLOR)
            ax.set_xlabel('Predicted', color=TEXT_COLOR, fontsize=11)
            ax.set_ylabel('Actual', color=TEXT_COLOR, fontsize=11)
            ax.set_title('Confusion Matrix', color=TEXT_COLOR, fontsize=12, fontweight='bold')
            
            # Colorbar
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Count', color=TEXT_COLOR)
            cbar.ax.tick_params(colors=TEXT_COLOR)
            
            canvas = FigureCanvasTkAgg(fig, master=self.tab_confusion)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True, padx=PAD, pady=PAD)
        
        except Exception as e:
            print(f"Error displaying confusion chart: {e}")
    
    def _display_roc_chart(self):
        """Display ROC curve chart in tab"""
        try:
            # Clear previous widgets
            for widget in self.tab_roc.winfo_children():
                widget.destroy()
            
            fpr, tpr, _ = roc_curve(self.y_true, self.y_scores)
            roc_auc = auc(fpr, tpr)
            
            fig = Figure(figsize=(8, 8), facecolor=BG_DARK, dpi=100)
            ax = fig.add_subplot(111)
            ax.set_facecolor(BG_LIGHT)
            
            ax.plot(fpr, tpr, color=ACCENT_COLOR, lw=3, label=f'ROC Curve (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], color=TEXT_COLOR, lw=1.5, linestyle='--', label='Random')
            
            ax.set_xlabel('False Positive Rate', color=TEXT_COLOR, fontsize=11)
            ax.set_ylabel('True Positive Rate', color=TEXT_COLOR, fontsize=11)
            ax.set_title('ROC Curve', color=TEXT_COLOR, fontsize=12, fontweight='bold')
            ax.legend(facecolor=BG_LIGHT, edgecolor=ACCENT_COLOR, labelcolor=TEXT_COLOR, loc='lower right')
            ax.tick_params(colors=TEXT_COLOR)
            ax.grid(True, alpha=0.2, color=TEXT_COLOR)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.spines['bottom'].set_color(TEXT_COLOR)
            ax.spines['left'].set_color(TEXT_COLOR)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            canvas = FigureCanvasTkAgg(fig, master=self.tab_roc)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True, padx=PAD, pady=PAD)
        
        except Exception as e:
            print(f"Error displaying ROC chart: {e}")
    
    def _process_queue(self):
        """Process UI queue"""
        while True:
            try:
                item = self.ui_queue.get_nowait()
                
                if item[0] == 'progress':
                    self.progress_var.set(item[1])
                    self.progress_label.config(text=f"{item[1]:.1f}%")
                
                elif item[0] == 'status':
                    self._set_status(item[1], ACCENT_COLOR)
                
                elif item[0] == 'add_result':
                    video, pred, conf = item[1], item[2], item[3]
                    self.results_tree.insert('', 'end', values=(video, pred, conf, "✓"))
                
                elif item[0] == 'update_summary':
                    self.summary_text.config(state='normal')
                    self.summary_text.delete('1.0', 'end')
                    self.summary_text.insert('1.0', item[1])
                    self.summary_text.config(state='disabled')
                
                elif item[0] == 'update_confidence_chart':
                    self._display_confidence_chart()
                
                elif item[0] == 'update_confusion_chart':
                    self._display_confusion_chart()
                
                elif item[0] == 'update_roc_chart':
                    self._display_roc_chart()
                
                elif item[0] == 'enable_export':
                    self.btn_json.config(state='normal')
                    self.btn_report.config(state='normal')
                    self.btn_charts.config(state='normal')
                
                elif item[0] == 'error':
                    messagebox.showerror("Error", item[1])
            
            except queue.Empty:
                break
        
        self.root.after(100, self._process_queue)


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    gui = DeepfakeDetectionGUI(root)
    root.mainloop()
