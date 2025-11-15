#!/usr/bin/env bash

pip install -r requirements.txt

# Exit on error
set -e

echo "Updating package list..."
apt-get update

echo "Installing Tesseract OCR..."
apt-get install -y tesseract-ocr

echo "Verifying installation..."
tesseract --version

echo "Tesseract installation complete."