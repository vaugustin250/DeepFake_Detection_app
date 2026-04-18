"""
Install XceptionNet for deepfake detection
XceptionNet is proven to be highly effective on FaceForensics++ dataset
"""
import sys

print("Installing XceptionNet and dependencies...")

packages = [
    "timm",  # PyTorch Image Models (has XceptionNet)
    "torch",
]

for pkg in packages:
    print(f"  Installing {pkg}...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

print("\n✓ XceptionNet dependencies installed!")
print("\nNow loading pre-trained XceptionNet model...")

try:
    import timm
    import torch
    from torchvision import transforms
    from PIL import Image
    
    # Load pre-trained XceptionNet
    print("Loading timm XceptionNet model...")
    model = timm.create_model('xception', pretrained=True)
    model.eval()
    
    print("✓ XceptionNet model loaded successfully!")
    print(f"  Model: {model.__class__.__name__}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    
    # Test it
    print("\nTesting model...")
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy)
    print(f"✓ Model inference successful!")
    print(f"  Output shape: {output.shape}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✓ XceptionNet is ready!")
print("  Updated spatial_detector.py to use XceptionNet")
print("  Run: main.py")
print("="*60)
