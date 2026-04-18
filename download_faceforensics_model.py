"""
Download available deepfake detection models
"""
from transformers import pipeline

print("Downloading deepfake detection models...")
print("This may take 5-10 minutes on first run...")

models_to_try = [
    "prithivMLmods/Deep-Fake-Detector-v2-Model",
    "Wvolf/ViT-Deepfake-Detection"
]

success = False
for idx, model_name in enumerate(models_to_try, 1):
    try:
        print(f"\n[{idx}/{len(models_to_try)}] Downloading: {model_name}")
        model = pipeline("image-classification", model=model_name, device=-1)
        
        # Test it
        from PIL import Image
        dummy = Image.new('RGB', (224, 224), color='red')
        result = model(dummy)
        print(f"✓ Successfully downloaded and tested: {model_name}")
        print(f"  Sample output: {result[0]}")
        success = True
        break
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        continue

if success:
    print("\n" + "="*60)
    print("✓ Download complete! Models cached in HuggingFace cache.")
    print("  You can now run: main.py")
    print("="*60)
else:
    print("\n⚠ Models will be downloaded on first run of main.py")
    print("  Make sure you have internet connectivity.")

