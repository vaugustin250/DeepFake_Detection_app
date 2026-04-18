"""
Download and save XceptionNet model to models/ folder
"""
import os
import sys
from pathlib import Path

# Setup paths
models_dir = Path(__file__).parent / "models" / "xceptionnet"
models_dir.mkdir(parents=True, exist_ok=True)

print(f"Downloading XceptionNet to: {models_dir}")

try:
    import timm
    import torch
    
    print("\nLoading XceptionNet pre-trained weights...")
    model = timm.create_model('xception', pretrained=True)
    
    print(f"Saving model to: {models_dir / 'model.pt'}")
    torch.save(model.state_dict(), models_dir / 'model.pt')
    
    print(f"✓ Model weights saved: {models_dir / 'model.pt'}")
    
    # Also save model config
    config = {
        'model_name': 'xception',
        'pretrained': True,
        'num_classes': 1000,
        'input_size': 224,
    }
    
    import json
    with open(models_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Config saved: {models_dir / 'config.json'}")
    
    # Create DOWNLOAD_COMPLETE marker
    (models_dir / 'DOWNLOAD_COMPLETE').touch()
    print(f"✓ Download complete marker created")
    
    print("\n" + "="*60)
    print("✓ XceptionNet model saved locally!")
    print(f"  Location: {models_dir}")
    print(f"  Size: {(models_dir / 'model.pt').stat().st_size / 1024 / 1024:.1f} MB")
    print("="*60)
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
