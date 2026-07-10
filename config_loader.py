import os
import json

_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(_DIR, "config.json")
DEFAULT_CONFIG_PATH = os.path.join(_DIR, "config.default.json")

# Absolute last-resort fallback, only used if config.default.json itself
# can't be read. Keep this in sync with config.default.json's contents.
_HARDCODED_FALLBACK = {
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
        "Summarize the main themes found in the document store."
    ],
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "temperature": 0.0,
    "top_k": 4
}


def _load_json(path: str):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not parse {path} ({e}).")
    return None


def load_config() -> dict:
    """Loads runtime config: config.default.json supplies base defaults
    (single source of truth for defaults), config.json overrides with
    machine-specific/personal values. Falls back to a hardcoded dict only
    if config.default.json itself is unavailable."""
    defaults = _load_json(DEFAULT_CONFIG_PATH) or _HARDCODED_FALLBACK
    user_config = _load_json(CONFIG_PATH) or {}
    return {**defaults, **user_config}
