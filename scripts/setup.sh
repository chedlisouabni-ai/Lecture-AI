#!/bin/bash
# Setup script for Linux/macOS

echo "========================================="
echo "Lecture AI - Setup Script"
echo "========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing Python packages..."
pip install -r requirements.txt

# Check for Tesseract
if ! command -v tesseract &> /dev/null
then
    echo "⚠️  Tesseract OCR not found!"
    echo "Please install it:"
    echo "  Ubuntu/Debian: sudo apt-get install tesseract-ocr"
    echo "  macOS: brew install tesseract"
else
    echo "✓ Tesseract OCR found"
fi

# Create .env from template
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
fi

# Create directory structure
echo "Creating directories..."
python3 -c "from config.paths import *"

echo ""
echo "========================================="
echo "✓ Setup complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Download models: python scripts/download_models.py"
echo "2. Run system: python main.py"