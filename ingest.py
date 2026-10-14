import os
import json
import threading
from typing import List, Dict, Any
import pypdf
import docx
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from config_loader import load_config

config = load_config()
SOURCE_DIR = config["source_documents_dir"]
VECTOR_DB_DIR = config["vector_db_dir"]
MANIFEST_PATH = os.path.join(VECTOR_DB_DIR, "metadata_manifest.json")
COLLECTION_NAME = "digital_twin_collection"

# Ensure directories exist
os.makedirs(SOURCE_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)


# Task prefixes some embedding models were trained with; retrieval quality drops
# without them. Keyed by model name without tag: (query prefix, document prefix).
# Override per model with "embedding_query_prefix"/"embedding_document_prefix".
KNOWN_EMBEDDING_PREFIXES = {
    "nomic-embed-text": ("search_query: ", "search_document: "),
    "mxbai-embed-large": ("Represent this sentence for searching relevant passages: ", ""),
}


class PrefixedOllamaEmbeddings(OllamaEmbeddings):
    """OllamaEmbeddings that prepends the model's task prefix. Only the embedded
    text is prefixed; Chroma stores the original chunk text."""
    query_prefix: str = ""
    document_prefix: str = ""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return super().embed_documents([self.document_prefix + t for t in texts])

    def embed_query(self, text: str) -> List[float]:
        # The base embed_query routes through embed_documents, which would add
        # the document prefix as well, so call the parent's directly.
        return super().embed_documents([self.query_prefix + text])[0]


def get_embedding_model() -> PrefixedOllamaEmbeddings:
    """Returns local embedding model wrapper."""
    config = load_config()
    model = config["embedding_model"]
    query_prefix, document_prefix = KNOWN_EMBEDDING_PREFIXES.get(model.split(":")[0], ("", ""))
    return PrefixedOllamaEmbeddings(
        model=model,
        query_prefix=config.get("embedding_query_prefix", query_prefix),
        document_prefix=config.get("embedding_document_prefix", document_prefix),
    )


_db_instance = None
_db_embedding_settings = None  # what the shared client embeds with (fixed until restart)
_db_lock = threading.RLock()  # guards instantiation and each read/write call
_sync_lock = threading.Lock()  # one sync at a time (watcher vs. Sync button)


def get_db() -> Chroma:
    """Single shared Chroma client for this process. Sharing one client
    (rather than one per ingest call and a separate one in QueryEngine)
    avoids two client objects concurrently hitting the same on-disk
    SQLite files from different threads (the folder watcher vs. queries)."""
    global _db_instance, _db_embedding_settings
    with _db_lock:
        if _db_instance is None:
            embeddings = get_embedding_model()
            _db_instance = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=VECTOR_DB_DIR,
            )
            _db_embedding_settings = {
                "embedding_model": embeddings.model,
                "document_prefix": embeddings.document_prefix,
            }
        return _db_instance


def parse_file(file_path: str, rel_path: str) -> List[Document]:
    """Parses a local file (TXT, MD, PDF, DOCX, CSV, Excel, HTML) and returns a list of LangChain Documents.

    Raises on unreadable/corrupt files so callers can tell a parse failure apart from a file with no extractable text.
    """
    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()
    documents = []

    if ext in (".txt", ".md"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        if text.strip():
            documents.append(Document(page_content=text, metadata={"source": rel_path}))

    elif ext == ".pdf":
        reader = pypdf.PdfReader(file_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": rel_path, "page": i + 1}
                    )
                )

    elif ext == ".docx":
        doc = docx.Document(file_path)
        # Combine paragraphs
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        
        # Combine tables
        tables_text = []
        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    tables_text.append(" | ".join(cells))
        
        content = "\n".join(paragraphs)
        if tables_text:
            content += "\n\n=== Tables ===\n" + "\n".join(tables_text)
            
        if content.strip():
            documents.append(Document(page_content=content, metadata={"source": rel_path}))

    elif ext == ".csv":
        import csv
        rows_text = []
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                row_parts = [f"{k}: {v}" for k, v in row.items() if v]
                if row_parts:
                    rows_text.append(f"Row {i+1}: " + ", ".join(row_parts))
        content = "\n".join(rows_text)
        if content.strip():
            documents.append(Document(page_content=content, metadata={"source": rel_path}))

    elif ext == ".xlsx":
        import pandas as pd
        excel_data = pd.read_excel(file_path, sheet_name=None)
        sheets_text = []
        for sheet_name, df in excel_data.items():
            df = df.fillna("")
            rows_text = []
            for i, row in df.iterrows():
                row_parts = [f"{col}: {val}" for col, val in row.items() if str(val).strip()]
                if row_parts:
                    rows_text.append(f"Row {i+1}: " + ", ".join(row_parts))
            if rows_text:
                sheets_text.append(f"=== Sheet: {sheet_name} ===\n" + "\n".join(rows_text))
        content = "\n\n".join(sheets_text)
        if content.strip():
            documents.append(Document(page_content=content, metadata={"source": rel_path}))

    elif ext == ".html":
        from bs4 import BeautifulSoup
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        # Remove scripts and styles
        for elem in soup(["script", "style"]):
            elem.decompose()
        text = soup.get_text(separator="\n")
        lines = (line.strip() for line in text.splitlines())
        chunks_text = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = "\n".join(chunk for chunk in chunks_text if chunk)
        if clean_text.strip():
            documents.append(Document(page_content=clean_text, metadata={"source": rel_path}))

    return documents


def load_manifest() -> Dict[str, Any]:
    """Loads indexing manifest metadata."""
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            # Safe to continue: sync treats every file as new and replaces its
            # chunks by source, so this costs a re-embed, not duplicates.
            print(f"Warning: could not read manifest {MANIFEST_PATH} ({e}); all files will be re-indexed.")
    return {"files": {}, "total_chunks": 0}


def save_manifest(manifest: Dict[str, Any]):
    """Saves indexing manifest metadata atomically, so a crash mid-write (or a
    query reading it concurrently) never sees a truncated file."""
    tmp_path = MANIFEST_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp_path, MANIFEST_PATH)


def sync_vector_store(verbose: bool = True) -> Dict[str, Any]:
    """
    Scans the source directory and performs incremental updates to the Chroma DB.
    Detects deleted, new, and modified files.
    """
    # _sync_lock serializes whole syncs, so two overlapping ones (watcher + Sync
    # button) can't overwrite each other's manifest. _db_lock is taken only
    # around each Chroma call, not while parsing PDFs, so queries keep working
    # during a long sync (waiting at most one embedding batch); they may see a
    # file's chunks half-replaced until its re-index finishes.
    with _sync_lock:
        manifest = load_manifest()
        db = get_db()
        config = load_config()

        # 1. Rebuild from scratch if the embedding model, document prefix or
        # chunking changed: old vectors would be incompatible (or chunked
        # differently) from new ones. Embedding changes take effect after a
        # restart (see get_db()).
        index_settings = {
            **_db_embedding_settings,
            "chunk_size": config["chunk_size"],
            "chunk_overlap": config["chunk_overlap"],
        }
        # Older manifests lack some or all of these keys: assume they matched,
        # except the document prefix, which didn't exist before (so was "").
        previous_settings = {**index_settings, "document_prefix": "", **manifest.get("index_settings", {})}
        if previous_settings != index_settings:
            if manifest["files"]:
                print(f"Index settings changed ({previous_settings} -> {index_settings}); rebuilding the vector store...")
            with _db_lock:
                db.reset_collection()
            manifest["files"] = {}
        manifest["index_settings"] = index_settings

        # 2. Get list of files in source directory recursively
        allowed_exts = {".txt", ".md", ".pdf", ".docx", ".xlsx", ".csv", ".html"}
        current_files = {}
        for root, _, files in os.walk(SOURCE_DIR):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in allowed_exts:
                    full_path = os.path.join(root, file)
                    # Compute path relative to SOURCE_DIR
                    rel_path = os.path.relpath(full_path, SOURCE_DIR)
                    try:
                        stat = os.stat(full_path)
                        current_files[rel_path] = {
                            "path": full_path,
                            "mtime": stat.st_mtime,
                            "size": stat.st_size,
                        }
                    except Exception as e:
                        if verbose:
                            print(f"Error accessing stats for {full_path}: {e}")

        # 3. Detect and handle deletions
        deleted_files = [fname for fname in manifest["files"] if fname not in current_files]
        for fname in deleted_files:
            if verbose:
                print(f"Detected deletion of: {fname}. Removing from vector database...")
            try:
                # Chroma deletes items using metadata filter
                with _db_lock:
                    db.delete(where={"source": fname})
            except Exception as e:
                # If the database or collection is empty, delete can fail; we catch and proceed
                if verbose:
                    print(f"Note: Could not delete {fname} from DB (might not exist yet): {e}")
            del manifest["files"][fname]

        # 4. Detect and handle additions / modifications
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"]
        )

        failed = {}
        for fname, info in current_files.items():
            is_new = fname not in manifest["files"]
            is_modified = False
            if not is_new:
                old_info = manifest["files"][fname]
                # != rather than >: a file replaced by an older copy (backup restore,
                # cp -p, a sync client) still counts as changed
                is_modified = info["mtime"] != old_info.get("mtime", 0) or info["size"] != old_info.get("size", 0)

            if is_new or is_modified:
                if verbose:
                    action = "indexing new file" if is_new else "re-indexing modified file"
                    print(f"Detected change: {action} '{fname}'...")

                # Remove any existing chunks first, even for "new" files: the
                # manifest may have been lost while the vector store survived.
                try:
                    with _db_lock:
                        db.delete(where={"source": fname})
                except Exception as e:
                    if verbose:
                        print(f"Note: Could not delete old chunks for {fname} from DB: {e}")

                try:
                    # Parse and chunk with relative path source
                    raw_docs = parse_file(info["path"], fname)
                    if not raw_docs:
                        print(f"Warning: no extractable text in '{fname}' (scanned/image-only?); indexed with 0 chunks.")
                    chunks = text_splitter.split_documents(raw_docs) if raw_docs else []
                    # Stable IDs make re-adding a file overwrite (upsert) its chunks
                    # instead of duplicating them
                    ids = [f"{fname}#{n}" for n in range(len(chunks))]

                    # Embed in small batches: one huge request crashes Ollama's runner
                    batch_size = max(1, int(config.get("embed_batch_size", 32)))
                    for i in range(0, len(chunks), batch_size):
                        with _db_lock:
                            db.add_documents(chunks[i:i + batch_size], ids=ids[i:i + batch_size])
                except Exception as e:
                    # Leave the file out of the manifest so the next sync retries it,
                    # and drop any partially written chunks so a retry can't duplicate them.
                    print(f"Error indexing '{fname}': {type(e).__name__}: {e}")
                    try:
                        with _db_lock:
                            db.delete(where={"source": fname})
                    except Exception:
                        pass
                    manifest["files"].pop(fname, None)
                    failed[fname] = f"{type(e).__name__}: {e}"
                    continue

                manifest["files"][fname] = {
                    "mtime": info["mtime"],
                    "size": info["size"],
                    "chunks_count": len(chunks)
                }

        # Update total chunk count
        manifest["total_chunks"] = sum(f["chunks_count"] for f in manifest["files"].values())
        manifest["failed"] = failed
        save_manifest(manifest)

    if verbose:
        print(f"Sync complete. Active files: {len(manifest['files'])}, Total Chunks: {manifest['total_chunks']}")
        
    return manifest


# Seconds of no file activity before a sync runs. Long enough for a large copy's
# trailing write events to land, so files aren't indexed half-written.
WATCHER_QUIET_SECONDS = 2.0


# Watchdog Handler
class SourceFolderHandler(FileSystemEventHandler):
    def __init__(self, callback):
        self.callback = callback
        self._timer = None
        self._timer_lock = threading.Lock()

    def on_any_event(self, event):
        # Filter directories and only trigger on documents folder changes
        if event.is_directory:
            return
        
        filename = os.path.basename(event.src_path)
        ext = os.path.splitext(filename)[1].lower()
        if ext in (".txt", ".md", ".pdf", ".docx", ".xlsx", ".csv", ".html", ".tmp", ".crdownload"):
            # Trailing-edge debounce: every event restarts the countdown, so one
            # sync runs after activity settles rather than on the first event of
            # a burst (which would miss the rest of the burst).
            with self._timer_lock:
                if self._timer is not None:
                    self._timer.cancel()
                self._timer = threading.Timer(WATCHER_QUIET_SECONDS, self.callback)
                self._timer.daemon = True
                self._timer.start()


def start_folder_watcher(callback_func) -> Observer:
    """Starts a background thread watching SOURCE_DIR."""
    event_handler = SourceFolderHandler(callback_func)
    observer = Observer()
    observer.schedule(event_handler, path=SOURCE_DIR, recursive=True)
    observer.start()
    return observer


if __name__ == "__main__":
    print(f"Syncing folder: '{SOURCE_DIR}' to vector store: '{VECTOR_DB_DIR}'...")
    sync_vector_store()
