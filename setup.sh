#!/bin/bash
# Setup script for the Instacart-LightFM project

echo "Setting up Python environment..."
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
echo "Setup complete. Environment activated."
