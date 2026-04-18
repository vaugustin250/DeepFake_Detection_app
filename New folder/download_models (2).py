"""
download_models.py
==================
One-shot model downloader for ForensicStream AI.
Run this ONCE before starting the app:

    python download_models.py

Downloads all 3 layer models to ./models/ directory:

  Layer 1 — Spatial:
    • prithivMLmods/Deep-Fake-Detector-v2-Model  (~400 MB, ViT deepfake classifier)
    • Fallback: Wvolf/ViT-Deepfake-Detection      (~330 MB)

  Layer 2 — Temporal:
    • microsoft/xclip-base-patch32               (~600 MB, video understanding)
      Used for: video-level temporal feature extraction
      Fallback: pure OpenCV optical flow (zero download, already works)

  Layer 3 — AV Sync:
    • facebook/wav2vec2-base-960h                (~360 MB, speech audio encoder)
      Used for: phoneme-level audio feature extraction
      Gives HOLA-style audio embedding for lip-sync comparison

Total: ~1.4 GB first run, then permanently cached in ./models/
"""

import os
import sys
import time
import shutil
from pathlib import Path

# ── Colour helpers for terminal output ────────────────────────────────────────
def _c(text, code): return f"\033[{code}m{text}\033[0m"
def cyan(t):   return _c(t, "96")
def green(t):  return _c(t, "92")
def yellow(t): return _c(t, "93")
def red(t):    return _c(t, "91")
def bold(t):   return _c(t, "1")
def dim(t):    return _c(t, "2")

MODELS_DIR = Path("./models")


# ── Pre-flight checks ──────────────────────────────────────────────────────────
def check_dependencies():
    print(bold("\n⬡  ForensicStream AI — Model Downloader"))
    print(dim("─" * 55))
    missing = []
    for pkg, import_name in [
        ("transformers", "transformers"),
        ("torch",        "torch"),
        ("huggingface_hub", "huggingface_hub"),
    ]:
        try:
            __import__(import_name)
            print(f"  {green('✓')}  {pkg}")
        except ImportError:
            print(f"  {red('✗')}  {pkg}  ← MISSING")
            missing.append(pkg)

    if missing:
        print(red(f"\n  Missing packages: {', '.join(missing)}"))
        print(yellow("  Fix with:"))
        print(f"    pip install {' '.join(missing)}\n")
        sys.exit(1)

    print(green("\n  All dependencies OK.\n"))


# ── Single model download ──────────────────────────────────────────────────────
def download_model(model_id: str,
                   save_dir: Path,
                   description: str,
                   model_type: str = "auto") -> bool:
    """
    Downloads a HuggingFace model to save_dir using huggingface_hub.
    Returns True on success.
    """
    print(bold(f"\n  ┌─ {description}"))
    print(dim(f"  │  ID   : {model_id}"))
    print(dim(f"  │  Dest : {save_dir}"))

    marker = save_dir / "DOWNLOAD_COMPLETE"
    if marker.exists():
        print(f"  │  {green('Already downloaded — skipping.')}")
        print("  └─" + green(" ✓ DONE\n"))
        return True

    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
        import transformers

        print(f"  │  {cyan('Downloading...')} (progress shown below)")
        print("  │")

        t0 = time.time()

        local_dir = snapshot_download(
            repo_id=model_id,
            local_dir=str(save_dir),
            local_dir_use_symlinks=False,
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*",
                             "rust_model*", "*.ot", "*.h5"],
        )

        elapsed = time.time() - t0
        size_mb = _dir_size_mb(save_dir)

        # Write completion marker
        marker.write_text(f"Downloaded: {model_id}\nSize: {size_mb:.0f} MB\nTime: {elapsed:.1f}s\n")

        print(f"  │  {green(f'Downloaded {size_mb:.0f} MB in {elapsed:.1f}s')}")
        print("  └─" + green(" ✓ SUCCESS\n"))
        return True

    except Exception as e:
        print(f"  │  {red(f'Error: {e}')}")
        print("  └─" + red(" ✗ FAILED\n"))
        # Clean up partial download
        if save_dir.exists() and not marker.exists():
            shutil.rmtree(save_dir, ignore_errors=True)
        return False


def _dir_size_mb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


# ── Verify model can actually be loaded ───────────────────────────────────────
def verify_model(save_dir: Path, model_type: str, description: str) -> bool:
    print(f"  {dim('Verifying')} {description}...")
    try:
        if model_type == "image-classification":
            from transformers import pipeline
            p = pipeline("image-classification",
                         model=str(save_dir), device=-1)
            del p

        elif model_type == "wav2vec2":
            from transformers import Wav2Vec2Processor, Wav2Vec2Model
            proc  = Wav2Vec2Processor.from_pretrained(str(save_dir))
            model = Wav2Vec2Model.from_pretrained(str(save_dir))
            del proc, model

        elif model_type == "xclip":
            from transformers import XCLIPProcessor, XCLIPModel
            proc  = XCLIPProcessor.from_pretrained(str(save_dir))
            model = XCLIPModel.from_pretrained(str(save_dir))
            del proc, model

        print(f"  {green('✓')} {description} verified OK")
        return True
    except Exception as e:
        print(f"  {yellow('⚠')} Verification warning: {e}")
        return False


# ── Write models/config.json ──────────────────────────────────────────────────
def write_config(results: dict):
    import json
    config = {
        "spatial_model":   str(MODELS_DIR / "spatial"),
        "spatial_ok":      results.get("spatial", False),
        "wav2vec2_model":  str(MODELS_DIR / "wav2vec2"),
        "wav2vec2_ok":     results.get("wav2vec2", False),
        "xclip_model":     str(MODELS_DIR / "xclip"),
        "xclip_ok":        results.get("xclip", False),
    }
    config_path = MODELS_DIR / "config.json"
    MODELS_DIR.mkdir(exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(green(f"\n  Config written → {config_path}"))
    return config_path


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    check_dependencies()

    print(bold("  MODELS TO DOWNLOAD"))
    print(dim("  ─────────────────────────────────────────────────────"))
    print(f"  {cyan('Layer 1 Spatial')}  prithivMLmods/Deep-Fake-Detector-v2-Model  ~400 MB")
    print(f"  {cyan('Layer 2 Temporal')} microsoft/xclip-base-patch32               ~600 MB")
    print(f"  {cyan('Layer 3 AV Sync')}  facebook/wav2vec2-base-960h                ~360 MB")
    print(dim("  ─────────────────────────────────────────────────────"))
    print(f"  {yellow('Total ~1.4 GB')} — cached permanently after first download\n")

    ans = input("  Proceed? [Y/n]: ").strip().lower()
    if ans == "n":
        print(yellow("  Cancelled."))
        sys.exit(0)

    results = {}
    total_t0 = time.time()

    # ── Layer 1: Spatial deepfake classifier ──────────────────────────────────
    print(bold("\n" + "═" * 55))
    print(bold("  LAYER 1 — SPATIAL DEEPFAKE CLASSIFIER"))
    print(bold("═" * 55))

    spatial_ok = download_model(
        model_id    = "prithivMLmods/Deep-Fake-Detector-v2-Model",
        save_dir    = MODELS_DIR / "spatial",
        description = "ViT Deepfake Classifier (DFDC + FF++)",
        model_type  = "image-classification")

    if not spatial_ok:
        print(yellow("  Trying fallback model: Wvolf/ViT-Deepfake-Detection"))
        spatial_ok = download_model(
            model_id    = "Wvolf/ViT-Deepfake-Detection",
            save_dir    = MODELS_DIR / "spatial",
            description = "ViT Deepfake Classifier (fallback)",
            model_type  = "image-classification")

    if spatial_ok:
        verify_model(MODELS_DIR / "spatial", "image-classification",
                     "Spatial classifier")
    results["spatial"] = spatial_ok

    # ── Layer 2: Temporal — X-CLIP video encoder ──────────────────────────────
    print(bold("\n" + "═" * 55))
    print(bold("  LAYER 2 — TEMPORAL VIDEO ENCODER (X-CLIP)"))
    print(bold("═" * 55))
    print(dim("  X-CLIP (Cross-frame Communication Transformers) encodes"))
    print(dim("  multi-frame temporal context — directly implementing the"))
    print(dim("  AltFreezing paper's spatial-temporal feature separation.\n"))

    xclip_ok = download_model(
        model_id    = "microsoft/xclip-base-patch32",
        save_dir    = MODELS_DIR / "xclip",
        description = "X-CLIP Temporal Video Encoder",
        model_type  = "xclip")

    if xclip_ok:
        verify_model(MODELS_DIR / "xclip", "xclip", "X-CLIP encoder")
    results["xclip"] = xclip_ok

    # ── Layer 3: AV Sync — Wav2Vec2 audio encoder ─────────────────────────────
    print(bold("\n" + "═" * 55))
    print(bold("  LAYER 3 — AUDIO ENCODER (WAV2VEC2)"))
    print(bold("═" * 55))
    print(dim("  Wav2Vec2 encodes raw audio into phoneme-level embeddings."))
    print(dim("  We compare audio embeddings against lip movement embeddings"))
    print(dim("  from MediaPipe — directly implementing HOLA's cross-modal"))
    print(dim("  mismatch detection approach.\n"))

    wav2vec_ok = download_model(
        model_id    = "facebook/wav2vec2-base-960h",
        save_dir    = MODELS_DIR / "wav2vec2",
        description = "Wav2Vec2 Speech Audio Encoder",
        model_type  = "wav2vec2")

    if wav2vec_ok:
        verify_model(MODELS_DIR / "wav2vec2", "wav2vec2", "Wav2Vec2 encoder")
    results["wav2vec2"] = wav2vec_ok

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - total_t0
    config_path = write_config(results)

    print(bold("\n" + "═" * 55))
    print(bold("  DOWNLOAD SUMMARY"))
    print(bold("═" * 55))

    all_ok = True
    for name, ok in [("Spatial (Layer 1)",  results["spatial"]),
                     ("X-CLIP  (Layer 2)",   results["xclip"]),
                     ("Wav2Vec2 (Layer 3)",  results["wav2vec2"])]:
        icon = green("✓") if ok else red("✗")
        all_ok = all_ok and ok
        print(f"  {icon}  {name}")

    total_mb = _dir_size_mb(MODELS_DIR) if MODELS_DIR.exists() else 0
    print(dim(f"\n  Total size on disk: {total_mb:.0f} MB"))
    print(dim(f"  Total time: {elapsed:.1f}s"))

    if all_ok:
        print(green(bold("\n  ✓ All models downloaded. Run: python main.py\n")))
    else:
        failed = [k for k, v in results.items() if not v]
        print(yellow(f"\n  ⚠ Some downloads failed: {failed}"))
        print(yellow("  The app will use fallback methods for failed layers."))
        print(yellow("  Run: python main.py  — it will still work.\n"))


if __name__ == "__main__":
    main()
