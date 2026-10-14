"""Shared fixtures. Tests never touch config.json, the real vector stores or Ollama:
config comes from config.default.json, and each sync test gets its own temporary
source folder and Chroma store with a small offline embedding model."""
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from types import SimpleNamespace
from typing import List

import pytest
from langchain_core.embeddings import Embeddings

import config_loader

# The app modules read config at import time (ingest also creates the configured
# folders), so before importing them, swap the user's config.json for a test one
# whose folders live in a temporary directory.
_TEST_DIR = tempfile.mkdtemp(prefix="digital-twin-tests-")
config_loader.CONFIG_PATH = os.path.join(_TEST_DIR, "config.json")
with open(config_loader.CONFIG_PATH, "w") as f:
    json.dump({"source_documents_dir": os.path.join(_TEST_DIR, "src"),
               "vector_db_dir": os.path.join(_TEST_DIR, "vs")}, f)

import ingest  # noqa: E402
import llm_client  # noqa: E402
import query_engine  # noqa: E402

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DIR = os.path.join(PROJECT_DIR, "source_documents")


class HashingEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings: texts sharing words get similar
    unit-length vectors, like real embeddings (which the cosine conversion in
    QueryEngine.retrieve() assumes). `broken` simulates Ollama being down."""

    def __init__(self, model: str = "test-embed", size: int = 64):
        self.model = model
        self.document_prefix = ""
        self.size = size
        self.broken = False

    def _embed(self, text: str) -> List[float]:
        if self.broken:
            raise ConnectionError("embedding server unavailable")
        vec = [0.0] * self.size
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.size] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec] if any(vec) else [1.0 / math.sqrt(self.size)] * self.size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


@pytest.fixture(autouse=True)
def no_real_store(tmp_path, monkeypatch):
    """Safety net: ingest's module globals come from config.json and point at the
    real document folder and vector store. Every test gets throwaway paths and no
    shared client, so one that forgets the `store` fixture can't touch real data."""
    unused = tmp_path / "unused"
    monkeypatch.setattr(ingest, "SOURCE_DIR", str(unused / "src"))
    monkeypatch.setattr(ingest, "VECTOR_DB_DIR", str(unused / "vs"))
    monkeypatch.setattr(ingest, "MANIFEST_PATH", str(unused / "vs" / "metadata_manifest.json"))
    monkeypatch.setattr(ingest, "_db_instance", None)


@pytest.fixture
def cfg(monkeypatch):
    """Config from config.default.json only; mutate it in a test to change settings."""
    config = json.load(open(config_loader.DEFAULT_CONFIG_PATH))
    for module in (ingest, query_engine, llm_client):
        monkeypatch.setattr(module, "load_config", lambda: dict(config))
    return config


@pytest.fixture
def store(tmp_path, monkeypatch, cfg):
    """An isolated source folder (seeded with two sample documents) and vector store."""
    src, vs = tmp_path / "src", tmp_path / "vs"
    src.mkdir()
    vs.mkdir()
    for name in ("project_aurora.md", "employees.csv"):
        shutil.copy(os.path.join(SAMPLE_DIR, name), src / name)
    embeddings = HashingEmbeddings()
    monkeypatch.setattr(ingest, "SOURCE_DIR", str(src))
    monkeypatch.setattr(ingest, "VECTOR_DB_DIR", str(vs))
    monkeypatch.setattr(ingest, "MANIFEST_PATH", str(vs / "metadata_manifest.json"))
    monkeypatch.setattr(ingest, "_db_instance", None)
    monkeypatch.setattr(ingest, "_db_embedding_settings", None)
    monkeypatch.setattr(ingest, "get_embedding_model", lambda: embeddings)
    return SimpleNamespace(src=src, embeddings=embeddings, cfg=cfg)


def sources_in_db() -> set:
    return {m["source"] for m in ingest.get_db()._collection.get(include=["metadatas"])["metadatas"]}


def chunk_count() -> int:
    return ingest.get_db()._collection.count()
