#!/bin/bash
# Deep Past GUI Launcher for Linux/macOS

echo ""
echo "============================================================"
echo "   Deep Past - Akkadian Translation GUI"
echo "============================================================"
echo ""

# Check if virtual environment exists
if [ -f ".venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Warning: No virtual environment found. Using system Python."
fi

# Install/update dependencies
echo "Checking dependencies..."
pip install -q gradio>=4.0.0

echo ""
echo "Starting GUI..."
echo ""

python gui/app.py "$@"
