# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Digital Twin is a 100% local, privacy-first RAG (Retrieval-Augmented Generation) system: a Streamlit dashboard backed by ChromaDB for vector storage and Ollama for local embeddings/LLM inference. No network calls are made — all parsing, embedding, and completion happen on-device. There is a pytest suite (`tests/`) but no linter config. The app targets **macOS only** by design — `run_dashboard.sh` health-checks Ollama on `localhost:11434` and starts it via `brew services` or `Ollama.app`, with no Linux/Windows equivalent.

`.streamlit/config.toml` binds the server to `127.0.0.1` and disables Streamlit's usage statistics — Streamlit's defaults (all interfaces, stats on) would break the local-only/privacy guarantee.

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

There is no build step or lint command. Tests run offline in about 2 seconds, with no Ollama needed:
```bash
pip install -r requirements-dev.txt   # once: adds pytest
python -m pytest
```
`tests/conftest.py` keeps tests away from real data: config comes from `config.default.json` only (never `config.json`), the autouse `no_real_store` fixture points `ingest`'s module globals at throwaway paths, and the `store` fixture gives sync/query tests their own temporary source folder and Chroma store with `HashingEmbeddings` (deterministic, unit-length bag-of-words vectors) in place of Ollama. LLM calls are stubbed. Don't call `monkeypatch.undo()` inside a test: it also reverts those fixtures and points `ingest` back at the configured store. UI changes still need a manual check by running the dashboard (or a headless `streamlit.testing.v1.AppTest`). `requirements.txt` pins upper bounds (below the next major version) to guard against breaking changes from the fast-moving langchain/chromadb ecosystem — bump deliberately, not by dropping the bounds.

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
  - `sync_vector_store()` is the incremental indexer: it walks `SOURCE_DIR` recursively, diffs against a JSON manifest (`metadata_manifest.json`, stored inside `VECTOR_DB_DIR`) keyed by relative path with `mtime`/`size`, and only re-parses+re-embeds new or modified files. Deletions are detected the same way and removed from Chroma via `db.delete(where={"source": fname})`. Every (re-)indexed file has its old chunks deleted by `source` first and gets stable chunk IDs (`{rel_path}#{n}`, upserted), so a lost or corrupt manifest costs a re-embed, never duplicate chunks. The manifest is written atomically (temp file + `os.replace`) and also records `index_settings` (embedding model + `chunk_size`/`chunk_overlap`); if those change, the next sync calls `reset_collection()` and rebuilds everything.
  - `start_folder_watcher()` runs a recursive `watchdog` `Observer` in a background thread with a trailing-edge debounce (`SourceFolderHandler`: each event restarts a `WATCHER_QUIET_SECONDS` timer, so a sync runs once activity settles and large copies aren't indexed half-written), calling back into `sync_vector_store()` on any file change so the dashboard stays in sync without manual action.
  - `get_embedding_model()` returns `PrefixedOllamaEmbeddings`, which adds the model's task prefixes (`KNOWN_EMBEDDING_PREFIXES`, e.g. `search_query: `/`search_document: ` for `nomic-embed-text`; override with `embedding_query_prefix`/`embedding_document_prefix`). The document prefix is part of `index_settings`, so changing it rebuilds the index; manifests from before prefixes existed count as unprefixed.
  - `get_db()` returns a single process-wide `Chroma` client singleton (guarded by module-level `_db_lock`, a `threading.RLock`), and `sync_vector_store()`/`QueryEngine` both use it rather than each constructing their own client — this avoids two separate client objects hitting the same on-disk SQLite files concurrently from the watcher thread and the query thread. A separate `_sync_lock` serializes whole syncs (watcher vs. Sync button); `_db_lock` is held only around each Chroma call (one embedding batch at most), never while parsing, so queries keep working during a long sync and may briefly see a file's chunks half-replaced.
  - `SOURCE_DIR`, `VECTOR_DB_DIR`, `MANIFEST_PATH`, and now the shared `Chroma` client's `embedding_model` (since `get_db()` builds it once and caches it) are all fixed at import/first-call time from `load_config()` — changing `source_documents_dir`, `vector_db_dir`, or `embedding_model` in `config.json` requires restarting the process, not just calling `load_config()` again.
- **`llm_client.py`** — `LocalLLMClient` wraps `ChatOllama` (`num_ctx` from config, default 8192). The system prompt (`SYSTEM_PROMPT_TEMPLATE`) is the factual-grounding contract: it forces the model to answer only from injected context and to reply with the exact string `"Information not found locally."` when the context doesn't support an answer. `app_name` from config is interpolated into this prompt, so renaming the app in config also changes how the model refers to itself.
- **`query_engine.py`** — `QueryEngine.retrieve()` (retrieval only, returns `(chunk, cosine similarity)` pairs — use it to evaluate retrieval without the LLM) first calls `scope_sources()`, which matches the question's words against indexed filenames (IDF-weighted, incl. joined adjacent words like `coca cola`→`COCACOLA`); if that picks ≤4 files it searches only those (per-file, `max(top_k, 8 // n_files)` chunks each, interleaved by rank across files), otherwise it falls back to global similarity search. Disable with `"filename_scoping": false`; corpus-specific filename noise words (default: SEC form tokens `10`, `k`, `q`, `8`) are in `filename_scoping_ignore`. It relies on informative filenames (e.g. `3M_2022_10K.pdf`). Ingestion embeds in batches of `embed_batch_size` because one huge request crashes Ollama's runner. Chunks below `min_similarity` (default 0 = off) are dropped. `query()` then de-dupes retrieved chunks by `(source, page, first-150-chars)`, assembles a labeled context block per chunk (`--- Document: {source} (Page {page}) ---`) until a budget of `(num_ctx - RESERVED_TOKENS) * CHARS_PER_TOKEN` characters is used (Ollama silently truncates over-long prompts from the start, dropping the system prompt), and returns `LocalLLMClient.stream()`'s chunk iterator with the citations (`query_stream()`). `finalize_answer()` then normalizes any close variant of the "not found" response to the exact canonical `NOT_FOUND` string and clears citations in that case — this is what the UI relies on to decide whether to show the citations expander. `query()` is the non-streaming wrapper (for CLI/tests) and returns errors as text with no citations; `LocalLLMClient.stream()` itself raises on errors. `standalone_question()` rewrites follow-ups ("what about 2018?") into standalone questions from the last two exchanges via `LocalLLMClient.rewrite_followup()` (one extra short LLM call; disable with `"rewrite_followups": false`), since neither retrieval nor the answer prompt sees the chat history.
- **`app.py`** — Streamlit UI. The initial `sync_vector_store()` and folder watcher startup are gated behind `st.cache_resource` (`_init_watcher()`), so they run exactly once per server process — not once per browser session/tab, which would otherwise leak a watcher thread per tab. Chat is modeled as a list of `{role, content, citations, rewritten?, error?}` dicts in `st.session_state.messages`; a pending user message with no assistant reply yet is detected each rerun, rewritten into a standalone question if there's history, and answered by streaming `QueryEngine.query_stream()` through `st.write_stream`. The finalized reply is stored and `st.rerun()` re-renders it from history (`error` replies render with `st.error`, `rewritten` shows as a "Searched for" caption). Model options are plain model names with install status added by `format_func`, so the selection survives a status change. Model switching in the sidebar calls `QueryEngine.change_model()`, which re-initializes the underlying `ChatOllama` client only if the model name actually changed. Installed-model detection (`is_model_installed()`) does an exact `(repository, tag)` comparison against `ollama.list()` output, not substring matching. Citation `source`/`content` fields are `html.escape()`-d before being interpolated into `unsafe_allow_html=True` markup, since they come from arbitrary user-ingested documents.

## Adding a new file format to ingestion

Add a branch to `parse_file()` in `ingest.py` keyed on the new extension, returning `Document` objects with at least `metadata={"source": rel_path}`. Also add the extension to the `allowed_exts` set in `sync_vector_store()` and to the extension check in `SourceFolderHandler.on_any_event()` — both must be updated or the watcher/sync will silently ignore the new file type.

## Config-driven behavior

`config.json` is the operator-facing control surface — most product behavior (branding, model choice, retrieval breadth, prompt suggestions) is meant to be changed there rather than in code:
- `app_name` re-brands the UI titles/headers *and* the LLM's self-description in the system prompt.
- `chunk_size`/`chunk_overlap` feed `RecursiveCharacterTextSplitter` at ingest time.
- `chunk_size`/`chunk_overlap`/`embedding_model` changes are detected on the next sync and trigger a full rebuild of the vector store (slow on a large corpus).
- `vector_db_dir` must point outside any cloud-synced folder (Synology Drive/Dropbox/iCloud/OneDrive) — concurrent file locks from sync clients can corrupt the Chroma SQLite files.
