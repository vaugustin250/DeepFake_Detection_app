import os, random, cv2, torch
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
from facenet_pytorch import MTCNN
from torchvision import transforms
from torchvision.models import efficientnet_b0
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# MODEL (unchanged core)
# ============================================================
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

# ============================================================
# GUI CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_model.pth"
FRAME_COUNT = 8

# Try load model safely
model = DeepfakeModel().to(DEVICE)
if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print("Warning: failed to load model weights:", e)
else:
    print("Warning: model file not found. Running with randomly initialized weights.")
model.eval()

mtcnn = MTCNN(image_size=224, margin=10, device=DEVICE, keep_all=False)
transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

# ============================================================
# TKINTER UI SETUP
# ============================================================
root = tk.Tk()
root.title("Deepfake Detection System")
root.state('zoomed')  # full-screen window
root.configure(bg="#0f1720")

# ---------- constants for fixed sizing ----------
TOP_BOX_W = 520
TOP_BOX_H = 360
VIEWER_W = 560
VIEWER_H = 420
SMALL_BOX_W = 360
SMALL_BOX_H = 220
ANALYSIS_H = 230

PAD_X = 18
PAD_Y = 12

# Title
tk.Label(root, text="🎬 Deepfake Detection System", font=("Segoe UI", 22, "bold"),
         bg="#0f1720", fg="#38e6c7").pack(pady=12)

# ---------- Top row: original, face, result ----------
top_row = tk.Frame(root, bg="#0f1720")
top_row.pack(pady=(6,12), fill="x")

# Original Video Frame
frame_original = tk.LabelFrame(top_row, text="Original Video Frame", font=("Segoe UI", 12, "bold"),
                               bg="#15202b", fg="#38e6c7", width=TOP_BOX_W, height=TOP_BOX_H, labelanchor="n")
frame_original.pack_propagate(False)
frame_original.grid(row=0, column=0, padx=PAD_X)
frame_original.grid_propagate(False)

lbl_original = tk.Label(frame_original, bg="#15202b")
lbl_original.pack(expand=True, padx=10, pady=10)

# Detected Face
frame_processed = tk.LabelFrame(top_row, text="Detected Face", font=("Segoe UI", 12, "bold"),
                                bg="#15202b", fg="#38e6c7", width=TOP_BOX_W, height=TOP_BOX_H, labelanchor="n")
frame_processed.pack_propagate(False)
frame_processed.grid(row=0, column=1, padx=PAD_X)
frame_processed.grid_propagate(False)

lbl_processed = tk.Label(frame_processed, bg="#15202b")
lbl_processed.pack(expand=True, padx=10, pady=10)

# Result
frame_result = tk.LabelFrame(top_row, text="Result", font=("Segoe UI", 12, "bold"),
                             bg="#15202b", fg="#38e6c7", width=SMALL_BOX_W, height=TOP_BOX_H, labelanchor="n")
frame_result.pack_propagate(False)
frame_result.grid(row=0, column=2, padx=PAD_X)
frame_result.grid_propagate(False)

lbl_result = tk.Label(frame_result, text="No Result", font=("Segoe UI", 18, "bold"),
                      bg="#15202b", fg="#38e6c7")
lbl_result.pack(expand=True, padx=10, pady=10)

# ---------- Buttons row ----------
button_row = tk.Frame(root, bg="#0f1720")
button_row.pack(pady=(6,8))

button_style = {"font":("Segoe UI", 12, "bold"), "bg":"#38e6c7", "fg":"#07121a", "bd":0, "relief":"raised"}

# we will attach commands later, functions need to exist at runtime anyway
btn_load = tk.Button(button_row, text="1️⃣ Load Video", width=16, height=1, **button_style)
btn_extract = tk.Button(button_row, text="2️⃣ Extract Frames", width=16, height=1, **button_style)
btn_detect = tk.Button(button_row, text="3️⃣ Detect Faces", width=16, height=1, **button_style)
btn_classify = tk.Button(button_row, text="4️⃣ Classify Video", width=16, height=1, **button_style)
btn_exit = tk.Button(button_row, text="❌ Exit", width=12, height=1, bg="#ff6b6b", fg="white", font=("Segoe UI", 12, "bold"))

btn_load.grid(row=0, column=0, padx=10)
btn_extract.grid(row=0, column=1, padx=10)
btn_detect.grid(row=0, column=2, padx=10)
btn_classify.grid(row=0, column=3, padx=10)
btn_exit.grid(row=0, column=4, padx=16)

# ---------- Lower row: viewer (left) and analysis (right) ----------
lower_row = tk.Frame(root, bg="#0f1720")
lower_row.pack(pady=(12,18), fill="both", expand=False)

# Extracted Frame Viewer box (fixed size)
viewer_frame = tk.LabelFrame(lower_row, text="Extracted Frame Viewer", font=("Segoe UI", 12, "bold"),
                             bg="#15202b", fg="#38e6c7", width=VIEWER_W, height=VIEWER_H, labelanchor="n")
viewer_frame.pack_propagate(False)
viewer_frame.grid(row=0, column=0, padx=PAD_X, sticky="nw")
viewer_frame.grid_propagate(False)

lbl_viewer = tk.Label(viewer_frame, bg="#15202b")
lbl_viewer.place(relx=0.5, rely=0.08, anchor="n")  # we'll set image later

# Navigation band always visible below viewer
nav_band = tk.Frame(viewer_frame, bg="#15202b")
nav_band.place(relx=0.5, rely=0.78, anchor="n")

btn_prev = tk.Button(nav_band, text="⏪ Previous", width=14, height=1, bg="#23303a", fg="white", font=("Segoe UI", 10, "bold"))
btn_next = tk.Button(nav_band, text="Next ⏩", width=14, height=1, bg="#23303a", fg="white", font=("Segoe UI", 10, "bold"))
btn_prev.grid(row=0, column=0, padx=12, pady=6)
btn_next.grid(row=0, column=1, padx=12, pady=6)

# Analysis box (fixed size)
analysis_frame = tk.LabelFrame(lower_row, text="Video Analysis (Numerical Metrics)", font=("Segoe UI", 12, "bold"),
                               bg="#15202b", fg="#38e6c7", width=SMALL_BOX_W, height=VIEWER_H, labelanchor="n")
analysis_frame.pack_propagate(False)
analysis_frame.grid(row=0, column=1, padx=PAD_X, sticky="ne")

# Use scrolled text to show metrics (read-only)
analysis_text = scrolledtext.ScrolledText(analysis_frame, width=40, height=24, bg="#15202b", fg="white",
                                          font=("Consolas", 11), bd=0, wrap="word")
analysis_text.pack(padx=10, pady=10, fill="both", expand=True)
analysis_text.insert(tk.END, "Load a video and extract frames to see numeric analysis here.")
analysis_text.configure(state=tk.DISABLED)

# ---------- Globals ----------
VIDEO_PATH = None
FRAMES = []        # list of RGB numpy arrays
FACES = []         # list of face tensors (C,H,W)
CURRENT_FRAME = 0

# ---------- Utility functions ----------
def _pil_image_fit(img_pil, max_w, max_h):
    """Return a resized copy of PIL image that fits inside max_w/max_h preserving aspect ratio."""
    w, h = img_pil.size
    scale = min(max_w / w, max_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    return img_pil.resize((new_w, new_h), Image.LANCZOS)

def _show_in_label(label_widget, pil_img, max_w, max_h):
    """Put a PIL image into a label while keeping reference."""
    img_resized = _pil_image_fit(pil_img, max_w, max_h)
    tk_img = ImageTk.PhotoImage(img_resized)
    label_widget.config(image=tk_img)
    label_widget.image = tk_img

# ---------- Video analysis computation ----------
def compute_video_metrics(video_path, frames, faces_list=None):
    """
    Returns a dict with numeric metrics:
    - resolution (w,h), fps, total_frames
    - avg_brightness (grayscale mean)
    - color_variance (mean var across frames)
    - motion_intensity (mean frame-diff)
    - avg_face_area (px^2) or None
    """
    metrics = {}
    if not video_path or not os.path.exists(video_path):
        return metrics

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    metrics['resolution'] = (width, height)
    metrics['fps'] = fps
    metrics['total_frames'] = total_frames

    if frames:
        # grayscale brightness mean per frame then average
        gray_means = []
        color_vars = []
        for f in frames:
            # f is RGB numpy uint8
            gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
            gray_means.append(float(np.mean(gray)))
            # color variance (mean variance across channels)
            color_vars.append(float(np.mean(np.var(f.astype(np.float32), axis=(0,1)))))
        metrics['avg_brightness'] = float(np.mean(gray_means))
        metrics['color_variance'] = float(np.mean(color_vars))

        # motion intensity: average of mean absolute difference across consecutive frames
        if len(frames) > 1:
            diffs = []
            for i in range(1, len(frames)):
                d = cv2.absdiff(frames[i].astype(np.float32), frames[i-1].astype(np.float32))
                diffs.append(float(np.mean(d)))
            metrics['motion_intensity'] = float(np.mean(diffs))
        else:
            metrics['motion_intensity'] = 0.0
    else:
        metrics['avg_brightness'] = None
        metrics['color_variance'] = None
        metrics['motion_intensity'] = None

    # average face area if faces_list is provided and not empty
    if faces_list:
        areas = []
        for face in faces_list:
            # face is a tensor shape (3,H,W)
            if hasattr(face, 'shape') and len(face.shape) == 3:
                h, w = int(face.shape[1]), int(face.shape[2])
                areas.append(w * h)
        metrics['avg_face_area'] = float(np.mean(areas)) if areas else None
        metrics['num_detected_faces'] = len(areas)
    else:
        metrics['avg_face_area'] = None
        metrics['num_detected_faces'] = 0

    return metrics

def format_metrics(metrics):
    if not metrics:
        return "No video loaded."
    w,h = metrics.get('resolution',(0,0))
    fps = metrics.get('fps',0.0)
    total = metrics.get('total_frames',0)
    ab = metrics.get('avg_brightness')
    cvv = metrics.get('color_variance')
    mi = metrics.get('motion_intensity')
    af = metrics.get('avg_face_area')
    nf = metrics.get('num_detected_faces',0)

    lines = [
        "📊 Video Technical Summary",
        "────────────────────────────",
        f"Resolution:        {w} × {h} pixels",
        f"Frame Rate (FPS):  {fps:.2f}" if fps is not None else "Frame Rate (FPS):  N/A",
        f"Total Frames:      {total}",
        f"Avg Brightness:    {ab:.2f}" if ab is not None else "Avg Brightness:    N/A",
        f"Color Variance:    {cvv:.2f}" if cvv is not None else "Color Variance:    N/A",
        f"Motion Intensity:  {mi:.2f}" if mi is not None else "Motion Intensity:  N/A",
        f"Avg Face Area:     {int(af):,} px²" if af is not None else "Avg Face Area:     N/A",
        f"Detected Faces:    {nf}",
        "",
        "Notes:",
        "- Avg Brightness: mean grayscale intensity (0-255).",
        "- Color Variance: average variance across RGB channels (higher = more color spread).",
        "- Motion Intensity: mean per-pixel change between consecutive frames.",
    ]
    return "\n".join(lines)

def update_analysis_panel(video_path, frames, faces_list=None):
    metrics = compute_video_metrics(video_path, frames, faces_list)
    txt = format_metrics(metrics)
    analysis_text.configure(state=tk.NORMAL)
    analysis_text.delete(1.0, tk.END)
    analysis_text.insert(tk.END, txt)
    analysis_text.configure(state=tk.DISABLED)

# ---------- Core functionalities ----------
def load_video():
    global VIDEO_PATH
    path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files","*.*")])
    if not path:
        return
    VIDEO_PATH = path
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    cap.release()
    if ret:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        _show_in_label(lbl_original, pil, TOP_BOX_W - 40, TOP_BOX_H - 40)
    else:
        messagebox.showwarning("Load failed", "Couldn't read first frame of the video.")
    # reset previous state
    clear_frames_and_faces()
    update_analysis_panel(VIDEO_PATH, FRAMES, FACES)
    messagebox.showinfo("Video Loaded", f"Loaded video:\n{VIDEO_PATH}")

def extract_frames():
    global FRAMES, CURRENT_FRAME
    if not VIDEO_PATH:
        messagebox.showwarning("Error", "Please load a video first.")
        return
    FRAMES = []
    cap = cv2.VideoCapture(VIDEO_PATH)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        messagebox.showwarning("Read error", "Couldn't determine total frames for this video.")
    # pick indices evenly
    indices = np.linspace(0, max(total-1,0), FRAME_COUNT, dtype=int) if total>0 else list(range(FRAME_COUNT))
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if not ret:
            continue
        FRAMES.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not FRAMES:
        messagebox.showwarning("Error", "No frames extracted.")
        return
    CURRENT_FRAME = 0
    # show first extracted frame in viewer and in original top box too (consistent)
    pil = Image.fromarray(FRAMES[0])
    _show_in_label(lbl_viewer, pil, VIEWER_W - 40, VIEWER_H - 120)
    _show_in_label(lbl_original, pil, TOP_BOX_W - 40, TOP_BOX_H - 40)
    update_analysis_panel(VIDEO_PATH, FRAMES, FACES)
    messagebox.showinfo("Frames Extracted", f"Extracted {len(FRAMES)} frames.")

def detect_faces():
    global FACES
    if not FRAMES:
        messagebox.showwarning("Error", "Please extract frames first.")
        return
    FACES = []
    # tqdm will print to console; keep it for debug progress
    for frame in tqdm(FRAMES, desc="Detecting faces"):
        img = Image.fromarray(frame)
        face = mtcnn(img)  # returns Tensor (3,H,W) or None
        if face is not None:
            FACES.append(face.cpu())
    if not FACES:
        messagebox.showerror("No Faces", "No faces detected.")
        # still update analysis (no faces)
        update_analysis_panel(VIDEO_PATH, FRAMES, FACES)
        return
    # display first detected face
    fimg = FACES[0].permute(1,2,0).numpy()
    fimg = (fimg * 255).astype(np.uint8)
    pil = Image.fromarray(fimg)
    _show_in_label(lbl_processed, pil, TOP_BOX_W - 40, TOP_BOX_H - 40)
    # update analysis with face areas
    update_analysis_panel(VIDEO_PATH, FRAMES, FACES)
    messagebox.showinfo("Faces Detected", f"Detected {len(FACES)} face(s) across extracted frames.")

def classify_video():
    if not FACES:
        messagebox.showwarning("Error", "Please detect faces first.")
        return
    faces = list(FACES)  # already tensors (3,H,W)
    # ensure count equals FRAME_COUNT by repeating/truncating
    if len(faces) < FRAME_COUNT:
        # repeat from start
        idx = 0
        while len(faces) < FRAME_COUNT:
            faces.append(faces[idx % len(faces)])
            idx += 1
    elif len(faces) > FRAME_COUNT:
        faces = faces[:FRAME_COUNT]
    # stack into (N, C, H, W)
    seq = torch.stack(faces).unsqueeze(1).unsqueeze(0).to(DEVICE)  # (1, N, 1, C, H, W) expected by model earlier? original model expects (B,N,_,C,H,W)
    # BUT the original model expects seqs shape (B,N,_,C,H,W) based on earlier code;
    # our DeepfakeModel.forward flattened B,N,_,C,H,W accordingly. We must match original used before:
    # In your provided code earlier sequence creation was: seq = torch.stack(faces).unsqueeze(1).unsqueeze(0).to(DEVICE)
    # So keep the same.
    with torch.no_grad():
        logits = model(seq)
        prob = torch.sigmoid(logits).item()
    
    if prob > 0.5:
        label = "FAKE"
        confidence = prob * 100
    else:
        label = "REAL"
        confidence = (1 - prob) * 100  # Flip to show confidence in REAL classification
    
    lbl_result.config(text=f"{label}\nConfidence: {confidence:.2f}%", fg=("#ff6b6b" if label=="FAKE" else "#38e6c7"))
    messagebox.showinfo("Result", f"Predicted: {label}\nConfidence: {confidence:.2f}%")

def clear_frames_and_faces():
    global FRAMES, FACES, CURRENT_FRAME
    FRAMES = []
    FACES = []
    CURRENT_FRAME = 0
    # clear viewer & processed but keep boxes same size
    lbl_viewer.config(image="")
    lbl_viewer.image = None
    lbl_processed.config(image="")
    lbl_processed.image = None
    lbl_original.config(image="")
    lbl_original.image = None
    lbl_result.config(text="No Result", fg="#38e6c7")
    analysis_text.configure(state=tk.NORMAL)
    analysis_text.delete(1.0, tk.END)
    analysis_text.insert(tk.END, "Load a video and extract frames to see numeric analysis here.")
    analysis_text.configure(state=tk.DISABLED)

# ---------- Navigation for extracted frames ----------
def show_extracted_frame(index):
    global CURRENT_FRAME
    if not FRAMES:
        return
    index = max(0, min(index, len(FRAMES)-1))
    CURRENT_FRAME = index
    pil = Image.fromarray(FRAMES[index])
    _show_in_label(lbl_viewer, pil, VIEWER_W - 40, VIEWER_H - 120)
    # also keep the Original top-left showing current extracted frame for quick reference
    _show_in_label(lbl_original, pil, TOP_BOX_W - 40, TOP_BOX_H - 40)

def on_prev():
    show_extracted_frame(CURRENT_FRAME - 1)

def on_next():
    show_extracted_frame(CURRENT_FRAME + 1)

btn_prev.config(command=on_prev)
btn_next.config(command=on_next)

# attach buttons to functions
btn_load.config(command=load_video)
btn_extract.config(command=extract_frames)
btn_detect.config(command=detect_faces)
btn_classify.config(command=classify_video)
btn_exit.config(command=root.destroy)

# keep UI responsive - small helper to ensure GUI remains stable on resize
def _noop_resize(event):
    # we intentionally keep box sizes fixed; do nothing on window resize
    pass

root.bind("<Configure>", _noop_resize)

# initial fixed placeholder images so boxes show stable layout
placeholder = Image.new("RGB", (400,300), (18,24,30))
_show_in_label(lbl_original, placeholder, TOP_BOX_W - 40, TOP_BOX_H - 40)
_show_in_label(lbl_processed, placeholder, TOP_BOX_W - 40, TOP_BOX_H - 40)
_show_in_label(lbl_viewer, placeholder, VIEWER_W - 40, VIEWER_H - 120)

# Start the app
root.mainloop()
