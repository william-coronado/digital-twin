#!/bin/bash
# Script to run the Digital Twin local Streamlit dashboard

# Navigate to the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Check if the virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment (.venv) not found. Please run environment setup first."
    exit 1
fi

# Check if Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "⚠️  Ollama does not appear to be running. Attempting to start service..."
    brew services start ollama
    # Give it a second to spin up
    sleep 2
fi

# Activate the virtual environment and launch Streamlit
echo "🚀 Starting Digital Twin local Streamlit dashboard..."
echo "👉 Open your browser to http://localhost:8501"
source .venv/bin/activate
streamlit run app.py --server.port 8501
