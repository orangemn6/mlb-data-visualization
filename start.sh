#!/bin/bash

# Statcast Spray Chart Pro - Quick Start Script
# This script sets up the environment and runs the application

echo "⚾ Statcast Spray Chart Pro - Quick Start"
echo "========================================"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is required but not found"
    echo "Please install Python 3.9 or higher from https://python.org"
    exit 1
fi

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "🐍 Python version: $python_version"

# Convert version to comparable format (e.g., 3.13 -> 313, 3.9 -> 309)
python_version_numeric=$(echo "$python_version" | awk -F. '{print $1*100 + $2}')
required_version_numeric=309  # 3.9

if [[ $python_version_numeric -lt $required_version_numeric ]]; then
    echo "❌ Error: Python 3.9+ required, found $python_version"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment"
        exit 1
    fi
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies if needed
if [ ! -f "venv/installed.flag" ]; then
    echo "📚 Installing dependencies..."
    pip install -r requirements.txt
    if [ $? -eq 0 ]; then
        touch venv/installed.flag
        echo "✅ Dependencies installed successfully"
    else
        echo "❌ Failed to install dependencies"
        exit 1
    fi
else
    echo "✅ Dependencies already installed"
fi

# Create data directories if they don't exist
mkdir -p data/players
mkdir -p data/stadiums

echo "🚀 Starting Statcast Spray Chart Pro..."
echo ""
echo "The application will open in your default browser at:"
echo "👉 http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

# Run the Streamlit app
streamlit run app.py