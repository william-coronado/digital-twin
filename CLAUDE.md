# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Digital Twin is a 100% local, privacy-first RAG (Retrieval-Augmented Generation) system: a Streamlit dashboard backed by ChromaDB for vector storage and Ollama for local embeddings/LLM inference. No network calls are made — all parsing, embedding, and completion happen on-device. There is no test suite or linter config in this project. The app targets **macOS only** by design — `run_dashboard.sh` manages Ollama via `pgrep`/`brew services`, with no Linux/Windows equivalent.

## Setup & running

```bash
# One-time setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull qwen2.5:7b

# Run the dashboard (handles venv activation + starting Ollama if needed)
./run_dashboard.sh
# -> http://localhost:8501
```

There is no build step, lint command, or automated test suite in this repo. Manual verification is done by running the dashboard and exercising the chat UI. `requirements.txt` pins upper bounds (below the next major version) to guard against breaking changes from the fast-moving langchain/chromadb ecosystem — bump deliberately, not by dropping the bounds.

To manually trigger a one-off ingestion sync from the command line (without starting the dashboard):
```bash
source .venv/bin/activate
python ingest.py
```

## Architecture

Request flow: `app.py` (Streamlit UI) → `QueryEngine` (`query_engine.py`) → `LocalLLMClient` (`llm_client.py`, wraps `ChatOllama`) + Chroma similarity search, with document ingestion handled independently by `ingest.py`.

- **`config_loader.py`** — `load_config()` loads `config.default.json` (the tracked, live-loaded single source of truth for defaults) and merges `config.json` (gitignored, machine-specific/personal overrides) on top — `config.json` only needs to include the keys it wants to change. A hardcoded Python dict is kept only as an absolute-last-resort fallback if `config.default.json` itself can't be read. Every module re-calls `load_config()` rather than passing config around, so changes to `config.json` take effect on next read without a full app restructure (some values are cached at import time in `ingest.py` and `llm_client.py` module scope — see below).
- **`ingest.py`** — Owns the Chroma collection (`COLLECTION_NAME = "digital_twin_collection"`) and all file parsing. Key pieces:
  - `parse_file()` dispatches by extension (`.txt`/`.md`, `.pdf`, `.docx`, `.csv`, `.xlsx`, `.html`) into a list of LangChain `Document`s, tagging each with `metadata={"source": rel_path, "page": ...}` (page only for PDFs) so citations can show relative paths and page numbers.
  - `sync_vector_store()` is the incremental indexer: it walks `SOURCE_DIR` recursively, diffs against a JSON manifest (`metadata_manifest.json`, stored inside `VECTOR_DB_DIR`) keyed by relative path with `mtime`/`size`, and only re-parses+re-embeds new or modified files. Deletions are detected the same way and removed from Chroma via `db.delete(where={"source": fname})`.
  - `start_folder_watcher()` runs a recursive `watchdog` `Observer` in a background thread with debounce logic (`SourceFolderHandler`), calling back into `sync_vector_store()` on any file change so the dashboard stays in sync without manual action.
  - `get_db()` returns a single process-wide `Chroma` client singleton (guarded by module-level `_db_lock`, a `threading.RLock`), and `sync_vector_store()`/`QueryEngine` both use it rather than each constructing their own client — this avoids two separate client objects hitting the same on-disk SQLite files concurrently from the watcher thread and the query thread. `sync_vector_store()` holds `_db_lock` for its full duration; `QueryEngine.query()` holds it only around the `similarity_search()` call.
  - `SOURCE_DIR`, `VECTOR_DB_DIR`, `MANIFEST_PATH`, and now the shared `Chroma` client's `embedding_model` (since `get_db()` builds it once and caches it) are all fixed at import/first-call time from `load_config()` — changing `source_documents_dir`, `vector_db_dir`, or `embedding_model` in `config.json` requires restarting the process, not just calling `load_config()` again.
- **`llm_client.py`** — `LocalLLMClient` wraps `ChatOllama` (fixed `num_ctx=8192`). The system prompt (`SYSTEM_PROMPT_TEMPLATE`) is the factual-grounding contract: it forces the model to answer only from injected context and to reply with the exact string `"Information not found locally."` when the context doesn't support an answer. `app_name` from config is interpolated into this prompt, so renaming the app in config also changes how the model refers to itself.
- **`query_engine.py`** — `QueryEngine.query()` does similarity search (`top_k` from config), de-dupes retrieved chunks by `(source, page, first-150-chars)`, assembles a labeled context block per chunk (`--- Document: {source} (Page {page}) ---`), and calls `LocalLLMClient.query()`. It then normalizes any close variant of the "not found" response to the exact canonical string and clears citations in that case — this is what the UI relies on to decide whether to show the citations expander.
- **`app.py`** — Streamlit UI. The initial `sync_vector_store()` and folder watcher startup are gated behind `st.cache_resource` (`_init_watcher()`), so they run exactly once per server process — not once per browser session/tab, which would otherwise leak a watcher thread per tab. Chat is modeled as a list of `{role, content, citations?}` dicts in `st.session_state.messages`; a pending user message with no assistant reply yet is detected each rerun and answered synchronously via `QueryEngine.query()`, then `st.rerun()` is called to settle state. Model switching in the sidebar calls `QueryEngine.change_model()`, which re-initializes the underlying `ChatOllama` client only if the model name actually changed. Installed-model detection (`is_model_installed()`) does an exact `(repository, tag)` comparison against `ollama.list()` output, not substring matching. Citation `source`/`content` fields are `html.escape()`-d before being interpolated into `unsafe_allow_html=True` markup, since they come from arbitrary user-ingested documents.

## Adding a new file format to ingestion

Add a branch to `parse_file()` in `ingest.py` keyed on the new extension, returning `Document` objects with at least `metadata={"source": rel_path}`. Also add the extension to the `allowed_exts` set in `sync_vector_store()` and to the extension check in `SourceFolderHandler.on_any_event()` — both must be updated or the watcher/sync will silently ignore the new file type.

## Config-driven behavior

`config.json` is the operator-facing control surface — most product behavior (branding, model choice, retrieval breadth, prompt suggestions) is meant to be changed there rather than in code:
- `app_name` re-brands the UI titles/headers *and* the LLM's self-description in the system prompt.
- `chunk_size`/`chunk_overlap` feed `RecursiveCharacterTextSplitter` at ingest time — changing these only affects newly indexed/re-indexed files, not existing chunks already in the vector store.
- `vector_db_dir` must point outside any cloud-synced folder (Synology Drive/Dropbox/iCloud/OneDrive) — concurrent file locks from sync clients can corrupt the Chroma SQLite files.
