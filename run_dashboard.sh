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

# Make sure the Ollama server is answering, starting it if needed
ollama_up() { curl -sf -m 2 http://localhost:11434/api/version > /dev/null; }
if ! ollama_up; then
    echo "⚠️  Ollama is not responding. Attempting to start it..."
    if brew list ollama &> /dev/null; then
        brew services start ollama
    elif [ -d "/Applications/Ollama.app" ]; then
        open -a Ollama
    else
        echo "❌ Ollama not found. Install it with 'brew install ollama' or from https://ollama.com."
        exit 1
    fi
    for _ in $(seq 30); do
        ollama_up && break
        sleep 1
    done
    if ! ollama_up; then
        echo "❌ Ollama did not start within 30 seconds."
        exit 1
    fi
fi

# Activate the virtual environment and launch Streamlit
echo "🚀 Starting Digital Twin local Streamlit dashboard..."
echo "👉 Open your browser to http://localhost:8501"
source .venv/bin/activate
streamlit run app.py --server.port 8501
