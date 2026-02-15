#!/bin/bash
# Fix common installation issues

echo "Fixing installation issues..."

# Reinstall bitsandbytes (often fixes quantization issues)
pip uninstall -y bitsandbytes
pip install bitsandbytes --no-cache-dir

# Ensure CUDA version matches if using GPU
if command -v nvcc &> /dev/null; then
    echo "CUDA detected. Installing GPU-optimized packages..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
fi

# Upgrade transformers to latest
pip install --upgrade transformers accelerate

echo "✓ Installation fixes applied"
echo ""
echo "Try running: python scripts/download_models.py"