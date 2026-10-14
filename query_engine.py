import math
import re
from itertools import zip_longest
from typing import List, Dict, Any, Iterator, Tuple
from langchain_core.documents import Document
from ingest import get_db, load_manifest, _db_lock
from llm_client import LocalLLMClient
from config_loader import load_config

_TOKEN_RE = re.compile(r"[a-z]+|[0-9]+")  # letter/digit runs, so "2022Q2" -> 2022, q, 2
# Words that appear in questions and filenames but say nothing about *which* document is meant.
# Corpus-specific ones (e.g. SEC form names "10-K" -> 10, k) go in config "filename_scoping_ignore".
_GENERIC_TOKENS = {"pdf", "txt", "md", "docx", "csv", "xlsx", "html",
                   "the", "of", "a", "an", "and", "in", "for", "what", "was", "is", "were", "how", "much"}
MAX_SCOPED_SOURCES = 4
# Once the search is confined to a few files, look deeper: the answer is often a numeric table
# chunk that ranks just below prose about the same topic.
SCOPED_TOTAL_K = 8
# Prompt-size budget: num_ctx minus room for the system prompt, question and
# answer, at a conservative chars-per-token (numeric tables tokenize densely).
# Ollama silently drops the start of an over-long prompt, system prompt included.
RESERVED_TOKENS = 1536
CHARS_PER_TOKEN = 3


NOT_FOUND = "Information not found locally."


def _tokens(text: str, ignore: frozenset = frozenset()) -> set:
    words = [w for w in _TOKEN_RE.findall(text.lower()) if w not in _GENERIC_TOKENS and w not in ignore]
    # Also join adjacent words, so "Lockheed Martin" / "Coca-Cola" match LOCKHEEDMARTIN / COCACOLA filenames
    return set(words) | {a + b for a, b in zip(words, words[1:])}


def scope_sources(query: str, sources: List[str], ignore: frozenset = frozenset()) -> List[str]:
    """Picks the indexed files whose names best match the query, or [] to search everything.

    Each query token that appears in a filename scores by rarity (IDF), so "3m" counts far more
    than "2022". Files scoring within 10% of the best score are kept, plus the best file(s) for any
    matched token they don't cover (so "compare Nike and Adobe" keeps both companies). If that
    isn't a small, specific set, a vague match (e.g. just a year) falls back to a global search.
    """
    if len(sources) < 2:
        return []
    src_tokens = {src: _tokens(src.rsplit(".", 1)[0], ignore) for src in sources}
    query_tokens = _tokens(query, ignore)
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


def finalize_answer(raw_answer: str, citations: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """Normalizes any variant of the "not found" reply to the exact canonical
    string and drops its citations; the UI relies on this to decide whether to
    show the citations expander."""
    answer = raw_answer.strip()
    if NOT_FOUND.lower().rstrip(".") in answer.lower():
        return NOT_FOUND, []
    return answer, citations


class QueryEngine:
    def __init__(self, model_name: str = None):
        config = load_config()
        active_model = model_name if model_name is not None else config["default_llm_model"]
        self.db = get_db()
        self.llm_client = LocalLLMClient(model_name=active_model)

    def change_model(self, model_name: str):
        """Allows switching models dynamically from the UI."""
        self.llm_client.change_model(model_name)

    def retrieve(self, user_query: str, top_k: int = None) -> List[Tuple[Document, float]]:
        """Returns (chunk, cosine similarity) pairs, best first, dropping any
        below config "min_similarity". Scoped searches are interleaved by rank
        across files, so trimming the tail drops each file's weakest chunks
        rather than a whole file."""
        config = load_config()
        k_val = top_k if top_k is not None else config["top_k"]
        scoped = []
        if config.get("filename_scoping", True):
            # Files with no chunks (e.g. scanned PDFs) can't answer anything
            searchable = [f for f, info in load_manifest()["files"].items() if info.get("chunks_count")]
            ignore = frozenset(config.get("filename_scoping_ignore", []))
            scoped = scope_sources(user_query, searchable, ignore)
        hits = []
        with _db_lock:
            if scoped:
                # Search each matching file separately so one file can't crowd out the others
                per_file_k = max(k_val, SCOPED_TOTAL_K // len(scoped))
                per_file = [self.db.similarity_search_with_score(user_query, k=per_file_k, filter={"source": src})
                            for src in scoped]
                hits = [hit for rank in zip_longest(*per_file) for hit in rank if hit]
            if not hits:
                # Unscoped, or the scoped files had nothing (e.g. removed by a sync since the manifest was read)
                hits = self.db.similarity_search_with_score(user_query, k=k_val)
        # The collection uses Chroma's default squared-L2 distance; for the
        # unit-length vectors Ollama returns, cosine similarity = 1 - d / 2.
        min_similarity = config.get("min_similarity", 0.0)  # 0 = no cutoff
        scored = [(doc, 1 - dist / 2) for doc, dist in hits]
        return [(doc, sim) for doc, sim in scored if not min_similarity or sim >= min_similarity]

    def standalone_question(self, user_query: str, history: List[Tuple[str, str]]) -> str:
        """Rewrites a follow-up into a standalone question using the last two
        exchanges of (role, content) history. Returns user_query unchanged when
        there's no history or config "rewrite_followups" is off."""
        if not history or not load_config().get("rewrite_followups", True):
            return user_query
        return self.llm_client.rewrite_followup(user_query, history[-4:])

    def query_stream(self, user_query: str, top_k: int = None) -> Tuple[Iterator[str], List[Dict[str, Any]]]:
        """Retrieves context and returns (answer chunks, citations). The chunks
        stream the model's raw answer: pass the joined text and citations to
        finalize_answer(). Raises if the vector store can't be queried; the
        iterator raises if the LLM can't be reached."""
        # 1. Retrieve most similar document chunks
        hits = self.retrieve(user_query, top_k)
        if not hits:
            return iter([NOT_FOUND]), []

        # 2. Assemble context content and build source citations list, within
        # the context window of the running client (config may have changed since)
        num_ctx = self.llm_client.llm.num_ctx
        budget_chars = (num_ctx - RESERVED_TOKENS) * CHARS_PER_TOKEN - len(user_query)
        context_parts = []
        citations = []
        seen_chunks = set()

        for doc, _ in hits:
            source = doc.metadata.get("source", "Unknown Document")
            page = doc.metadata.get("page")
            
            # De-duplicate chunks that are identical
            chunk_sig = (source, page, doc.page_content[:150])
            if chunk_sig in seen_chunks:
                continue
            seen_chunks.add(chunk_sig)
            
            page_info = f" (Page {page})" if page else ""
            part = f"--- Document: {source}{page_info} ---\n{doc.page_content}"
            if len(part) > budget_chars:
                if context_parts:
                    break
                # Even the best chunk alone doesn't fit (tiny num_ctx or huge
                # chunk_size): send what fits rather than overflow the window
                part = part[:max(budget_chars, 0)]
                if not part:
                    break
            budget_chars -= len(part) + 2  # + the "\n\n" separator
            context_parts.append(part)
            
            citations.append({
                "source": source,
                "page": page,
                "content": doc.page_content
            })

        context_text = "\n\n".join(context_parts)

        # 3. Call local LLM
        return self.llm_client.stream(user_query, context_text), citations

    def query(self, user_query: str, top_k: int = None) -> Tuple[str, List[Dict[str, Any]]]:
        """Non-streaming query_stream(): returns the finalized answer and citations,
        or an error message with no citations."""
        try:
            chunks, citations = self.query_stream(user_query, top_k)
        except Exception as e:
            return f"Error querying local vector store: {e}", []
        try:
            return finalize_answer("".join(chunks), citations)
        except Exception as e:
            return f"Error communicating with local LLM server: {e}", []
