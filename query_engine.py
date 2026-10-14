import math
import re
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from ingest import get_db, load_manifest, _db_lock
from llm_client import LocalLLMClient
from config_loader import load_config

_TOKEN_RE = re.compile(r"[a-z]+|[0-9]+")  # letter/digit runs, so "2022Q2" -> 2022, q, 2
# Words that appear in questions and filenames but say nothing about *which* document is meant.
_GENERIC_TOKENS = {"10", "k", "q", "8", "pdf", "txt", "md", "docx", "csv", "xlsx", "html",
                   "the", "of", "a", "an", "and", "in", "for", "what", "was", "is", "were", "how", "much"}
MAX_SCOPED_SOURCES = 4
# Once the search is confined to a few files, look deeper: the answer is often a numeric table
# chunk that ranks just below prose about the same topic.
SCOPED_TOTAL_K = 8


def _tokens(text: str) -> set:
    words = [w for w in _TOKEN_RE.findall(text.lower()) if w not in _GENERIC_TOKENS]
    # Also join adjacent words, so "Lockheed Martin" / "Coca-Cola" match LOCKHEEDMARTIN / COCACOLA filenames
    return set(words) | {a + b for a, b in zip(words, words[1:])}


def scope_sources(query: str, sources: List[str]) -> List[str]:
    """Picks the indexed files whose names best match the query, or [] to search everything.

    Each query token that appears in a filename scores by rarity (IDF), so "3m" counts far more
    than "2022". Files scoring within 10% of the best score are kept, plus the best file(s) for any
    matched token they don't cover (so "compare Nike and Adobe" keeps both companies). If that
    isn't a small, specific set, a vague match (e.g. just a year) falls back to a global search.
    """
    if len(sources) < 2:
        return []
    src_tokens = {src: _tokens(src.rsplit(".", 1)[0]) for src in sources}
    query_tokens = _tokens(query)
    df = {t: sum(1 for toks in src_tokens.values() if t in toks) for t in query_tokens}
    idf = {t: math.log(len(sources) / n) for t, n in df.items() if n}
    if not idf:
        return []

    scores = {src: sum(idf[t] for t in toks & idf.keys()) for src, toks in src_tokens.items()}
    best = max(scores.values())
    picked = {src for src, sc in scores.items() if sc >= best * 0.9}
    for t in idf:
        if not any(t in src_tokens[src] for src in picked):
            with_t = {src: sc for src, sc in scores.items() if t in src_tokens[src]}
            top = max(with_t.values())
            picked |= {src for src, sc in with_t.items() if sc >= top * 0.9}
    return sorted(picked) if len(picked) <= MAX_SCOPED_SOURCES else []


class QueryEngine:
    def __init__(self, model_name: str = None):
        config = load_config()
        active_model = model_name if model_name is not None else config["default_llm_model"]
        self.db = get_db()
        self.llm_client = LocalLLMClient(model_name=active_model)

    def change_model(self, model_name: str):
        """Allows switching models dynamically from the UI."""
        self.llm_client.change_model(model_name)

    def query(self, user_query: str, top_k: int = None) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Executes semantic search, injects context to system prompt,
        calls the local LLM, and returns the response with source citations.
        """
        config = load_config()
        k_val = top_k if top_k is not None else config["top_k"]
        # 1. Retrieve most similar document chunks
        try:
            scoped = scope_sources(user_query, list(load_manifest()["files"])) if config.get("filename_scoping", True) else []
            with _db_lock:
                if scoped:
                    # Search each matching file separately so one file can't crowd out the others
                    per_file_k = max(k_val, SCOPED_TOTAL_K // len(scoped))
                    docs = [d for src in scoped
                            for d in self.db.similarity_search(user_query, k=per_file_k, filter={"source": src})]
                else:
                    docs = self.db.similarity_search(user_query, k=k_val)
        except Exception as e:
            return f"Error querying local vector store: {e}", []

        if not docs:
            return "Information not found locally.", []

        # 2. Assemble context content and build source citations list
        context_parts = []
        citations = []
        seen_chunks = set()

        for doc in docs:
            source = doc.metadata.get("source", "Unknown Document")
            page = doc.metadata.get("page")
            
            # De-duplicate chunks that are identical
            chunk_sig = (source, page, doc.page_content[:150])
            if chunk_sig in seen_chunks:
                continue
            seen_chunks.add(chunk_sig)
            
            page_info = f" (Page {page})" if page else ""
            context_parts.append(f"--- Document: {source}{page_info} ---\n{doc.page_content}")
            
            citations.append({
                "source": source,
                "page": page,
                "content": doc.page_content
            })

        context_text = "\n\n".join(context_parts)

        # 3. Call local LLM
        response_text = self.llm_client.query(user_query, context_text)

        # 4. Strict check for "Information not found locally"
        cleaned_response = response_text.strip()
        # If response indicates empty knowledge, wipe out citations to be clean
        if "information not found locally" in cleaned_response.lower():
            # Standardize exact response
            cleaned_response = "Information not found locally."
            citations = []

        return cleaned_response, citations
