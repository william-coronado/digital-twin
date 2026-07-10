# 🧠 Digital Twin: Local-Only Privacy-First RAG System

Digital Twin is a production-grade, 100% local Retrieval-Augmented Generation (RAG) system optimized to run natively on Apple Silicon (tested on Mac Studio/MacBook Pro using MPS acceleration). Your data never leaves your device: all document parsing, vector indexing, embeddings calculation, and LLM completions are executed entirely offline.

> **Platform note:** Digital Twin currently targets **macOS only** — `run_dashboard.sh` manages the local Ollama service via `pgrep`/`brew services`, which have no Linux/Windows equivalent here yet. Cross-platform support isn't implemented, but contributions are welcome.

---

## 🚀 Key Features

* **Dynamic Query Suggestions:** Load query recommendations dynamically from a configuration file and shuffle them interactively directly in the web dashboard.
* **Cloud Sync Safety Configuration:** Parameterize your vector database storage path outside of Synology Drive, Dropbox, or OneDrive folders to prevent index corruption caused by background sync file locks.
* **Parameterized Identity:** Modify the application name in a single configuration file to instantly customize both the web interface titles and the AI model's system prompt instructions.
* **Strict Factual Alignment:** Custom prompt system guarantees the LLM only answers from your documents, responding with *"Information not found locally"* if context is missing.
* **Recursive Folder Watching:** Monitors and indexes files in the document folder **and all nested subfolders** dynamically in real-time.
* **Metadata-Rich Citations:** Dynamic UI citation logs showing relative file paths (preserving your subfolder structure) and page numbers (for PDFs).
* **Multi-Format Ingestion:** Specialized parsers for Text, Markdown, PDF, DOCX, CSV, Excel (multi-sheet), and HTML files.
* **Apple Silicon Optimized:** Configured for low-latency embeddings and unified memory model weights.

---

## 🛠️ Prerequisites

Before starting, ensure you have the following installed on your Mac (macOS only for now — see platform note above):
1. **Homebrew** (Package manager)
2. **Python 3.10+** (Python 3.12 is recommended)
3. **Ollama** (For hosting local weights)

If you need to install Ollama, you can run:
```bash
brew install ollama
brew services start ollama
```

---

## 📥 First-Time Setup

Run the following commands in your terminal to set up the repository:

### 1. Set Up the Python Virtual Environment
Navigate to the project folder and create a virtual environment:
```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install all package dependencies
pip install -r requirements.txt
```

### 2. Download the Default Local Models
Ensure Ollama is running, then pull the default embedding and LLM models configured for this system:
```bash
# Pull the nomic embedding model (~274MB)
ollama pull nomic-embed-text

# Pull the default Qwen 2.5 7B Instruct model (~4.7GB)
ollama pull qwen2.5:7b
```

---

## 🖥️ How to Run the Dashboard

We have included a startup script that handles verifying services, activating your virtual environment, and launching the server on local port `8501`. 

Simply run:
```bash
./run_dashboard.sh
```

Then open your browser to:
👉 **[http://localhost:8501](http://localhost:8501)**

---

## 📂 Data Library & Ingestion

### Document Upload Location
Place all your files inside the documents folder configured in `config.json` (defaults to `./source_documents` in this project workspace). 

### Subfolder Support
You can organize your files in **nested subfolders** (e.g. `./source_documents/policies/health_safety.md`, `./source_documents/reports/q2_finance.csv`).
* The system recursively crawls and watches all subfolders.
* Citation logs will display the full relative path (e.g. `policies/health_safety.md`) so you know exactly which folder the information came from.
* Supports files with identical names placed in different folders without collision.

### Supported Formats
* **`.txt`, `.md`** (Plain Text / Markdown)
* **`.pdf`** (Parsed page-by-page to preserve correct page citations)
* **`.docx`** (Word files, including paragraphs and tables)
* **`.csv`** (Structured row-by-row key-value representation for precise data querying)
* **`.xlsx`** (Excel workbooks, supports multi-sheet structures converted to row-by-row chunks)
* **`.html`** (Cleaned of script/style elements and parsed to text)

### How Ingestion Works
* The application runs a **recursive watchdog observer** that monitors your documents folder while the dashboard is running. When you add, edit, or delete a file (including in subfolders), it will sync automatically.
* You can also force a manual indexing pass at any time by clicking the **🔄 Sync Documents Now** button in the sidebar.
* A JSON manifest (`./metadata_manifest.json` inside the vector database path) tracks timestamps to ensure unchanged files are not re-embedded, keeping ingestion fast.

---

## ⚙️ Configuration & Custom Models

You can manage all RAG pipeline parameters in the centralized **`config.json`** file in the root of the project. `config.default.json` ships with the repo and supplies the working defaults shown below (including `vector_db_dir`); create your own `config.json` (gitignored) to override any of these values — it only needs to include the keys you want to change:

```json
{
  "app_name": "Digital Twin",
  "source_documents_dir": "./source_documents",
  "vector_db_dir": "./vector_store",
  "embedding_model": "nomic-embed-text",
  "default_llm_model": "qwen2.5:7b",
  "available_llm_models": [
    "qwen2.5:7b",
    "llama3:8b-instruct-q8_0",
    "llama3:8b",
    "mistral:7b-instruct-v0.3-q8_0",
    "qwen2:7b",
    "command-r"
  ],
  "suggested_prompts": [
    "What projects are discussed in the indexed documents?",
    "Summarize the main themes found in the document store.",
    "Who is the lead engineer for Project Aurora and what is the budget?",
    "What is the company remote work policy and its reference code?",
    "Who is the designated first aid officer and what should we do in case of fire?",
    "Retrieve the email address of Bob Jones."
  ],
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "temperature": 0.0,
  "top_k": 4
}
```

### Configurable Options:
* `"app_name"`: The name of the application. Changing this dynamically re-brands the Streamlit dashboard tabs, headers, greetings, and formats the LLM's system prompt instructions.
* `"source_documents_dir"`: Path to your documents library. You can redirect this to a local iCloud/Synology folder to sync files outside the workspace directory.
* `"vector_db_dir"`: Path to the Chroma vector database folder. Defaults to `./vector_store` inside the project. **IMPORTANT:** To prevent index file corruption, do not set this path inside folders synced by background engines like Synology Drive, Dropbox, iCloud, or OneDrive — including if your own project checkout itself lives inside one of those synced folders, in which case override this to an absolute path outside it (e.g. `/Users/yourusername/.digital-twin/vector_store`).
* `"suggested_prompts"`: A list of sample queries shown as click-to-submit suggestion buttons in the main chat room.
* `"embedding_model"`: The vector embedding model pulled via Ollama.
* `"default_llm_model"`: The active LLM used for question answering.
* `"chunk_size"` & `"chunk_overlap"`: Chunk size and overlap metrics for text segmentation.
* `"temperature"`: Model temperature (kept at `0.0` to maximize factual adherence).
* `"top_k"`: Number of matched contexts sent to the LLM.

### How to Download and Configure New Models:
1. Open your terminal and pull a model from the Ollama library (e.g. `qwen2.5:7b`):
   ```bash
   ollama pull qwen2.5:7b
   ```
2. Edit **`config.json`** and add your pulled model to the `"available_llm_models"` list:
   ```json
    "available_llm_models": [
      "qwen2.5:7b",
      "llama3:8b-instruct-q8_0",
      "..."
    ]
    ```
3. Set your active default by changing `"default_llm_model"`:
   ```bash
   "default_llm_model": "qwen2.5:7b"
   ```
4. Restart the dashboard using `./run_dashboard.sh`. Your model will automatically be selectable in the sidebar dropdown.

---

## 🧩 Project Structure

* **`config.default.json`** - Tracked template supplying the live-loaded default settings (app name, documents folder, vector store location, suggestions, models).
* **`config.json`** - Optional local overrides (gitignored) for machine-specific/personal values; only needs to include the keys you want to change.
* **`config_loader.py`** - Helper script loading config properties with default fallbacks.
* **`ingest.py`** - Recursive file parser, chunking functions, recursive watchdog, and ChromaDB sync.
* **`llm_client.py`** - Local Ollama Chat wrapper and system prompt definition.
* **`query_engine.py`** - retrieval chains, similarity searches, and context aggregators.
* **`app.py`** - Streamlit dashboard user interface.
* **`run_dashboard.sh`** - Startup shell script.
* **`source_documents/`** - Folder where your knowledge files go (or folder set in `config.json`).
* **`vector_store/`** - Persistent directory holding the ChromaDB SQLite files and manifest.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
