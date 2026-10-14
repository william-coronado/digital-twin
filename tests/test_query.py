"""Retrieval, context assembly and answer handling, with the LLM stubbed out."""
from types import SimpleNamespace

import pytest

import ingest
import query_engine
from llm_client import LocalLLMClient
from query_engine import CHARS_PER_TOKEN, NOT_FOUND, RESERVED_TOKENS, QueryEngine, finalize_answer

CITES = [{"source": "a.md", "page": None, "content": "x"}]


class StubLLM:
    """Stands in for LocalLLMClient: records the context it's given."""

    def __init__(self, num_ctx=8192, answer="Dr. Sarah Jenkins.", error=None):
        self.llm = SimpleNamespace(num_ctx=num_ctx)
        self.answer, self.error, self.context = answer, error, None

    def stream(self, prompt, context_text):
        self.context = context_text
        if self.error:
            raise self.error
        # Token-like pieces that keep their spaces, as a real stream does
        yield from (self.answer[i:i + 4] for i in range(0, len(self.answer), 4))

    def rewrite_followup(self, question, history):
        self.history = history
        return "rewritten: " + question


@pytest.fixture
def engine(store):
    ingest.sync_vector_store(verbose=False)
    eng = QueryEngine.__new__(QueryEngine)
    eng.db = ingest.get_db()
    eng.llm_client = StubLLM()
    return eng


@pytest.mark.parametrize("raw", ["Information not found locally.", '  "information not found locally"  ',
                                 "Sorry. Information not found locally"])
def test_not_found_variants_are_normalized_and_drop_citations(raw):
    assert finalize_answer(raw, CITES) == (NOT_FOUND, [])


def test_real_answer_keeps_citations():
    assert finalize_answer("  $2.4 million USD. ", CITES) == ("$2.4 million USD.", CITES)


def test_retrieve_returns_best_match_first(engine):
    hits = engine.retrieve("Who is the lead engineer of Project Aurora?")
    assert hits[0][0].metadata["source"] == "project_aurora.md"
    assert all(-1 <= sim <= 1 for _, sim in hits)


def test_zero_chunk_files_are_never_scoped(engine, store, monkeypatch):
    (store.src / "aurora_notes.txt").write_text("   \n")
    ingest.sync_vector_store(verbose=False)
    candidates = []
    real_scope = query_engine.scope_sources
    monkeypatch.setattr(query_engine, "scope_sources", lambda q, sources, *a: candidates.extend(sources) or real_scope(q, sources, *a))
    assert engine.retrieve("What do the aurora notes say about the budget?")
    assert "project_aurora.md" in candidates and "aurora_notes.txt" not in candidates


def test_scoped_search_with_no_hits_falls_back_to_global(engine, monkeypatch):
    monkeypatch.setattr(query_engine, "scope_sources", lambda *args: ["deleted_meanwhile.md"])
    assert engine.retrieve("Project Aurora budget")


def test_min_similarity_zero_disables_the_cutoff(engine, store):
    store.cfg["min_similarity"] = 0.0
    everything = engine.retrieve("zzz unrelated words")
    store.cfg["min_similarity"] = 0.99
    assert len(engine.retrieve("zzz unrelated words")) < len(everything)


def test_query_streams_answer_with_citations(engine):
    answer, citations = engine.query("Who is the lead engineer of Project Aurora?")
    assert answer == "Dr. Sarah Jenkins."
    assert citations and "Sarah Jenkins" in engine.llm_client.context


@pytest.mark.parametrize("num_ctx", [8192, 1700, 1000])
def test_context_fits_the_running_clients_window(engine, store, num_ctx):
    store.cfg["num_ctx"] = 99999  # a config change the running client hasn't picked up
    engine.llm_client = StubLLM(num_ctx=num_ctx)
    question = "Who is the lead engineer of Project Aurora?"
    engine.query(question)
    budget = (num_ctx - RESERVED_TOKENS) * CHARS_PER_TOKEN - len(question)
    assert len(engine.llm_client.context) <= max(budget, 0)


def test_llm_error_is_reported_without_citations(engine):
    engine.llm_client = StubLLM(error=ConnectionError("Ollama is not running"))
    answer, citations = engine.query("Who is the lead engineer of Project Aurora?")
    assert answer.startswith("Error communicating with local LLM server") and citations == []


def test_followups_are_rewritten_only_with_history(engine, store):
    history = [("user", "q1"), ("assistant", "a1"), ("user", "q2"), ("assistant", "a2"), ("user", "q3"), ("assistant", "a3")]
    assert engine.standalone_question("What about 2018?", []) == "What about 2018?"
    assert engine.standalone_question("What about 2018?", history) == "rewritten: What about 2018?"
    assert engine.llm_client.history == history[-4:]
    store.cfg["rewrite_followups"] = False
    assert engine.standalone_question("What about 2018?", history) == "What about 2018?"


@pytest.mark.parametrize("model_output, expected", [
    ('"What was 3M\'s total net sales in 2018?"', "What was 3M's total net sales in 2018?"),
    ("Latest question: What was 3M's total net sales in 2018?", "What was 3M's total net sales in 2018?"),
    ("", "What about 2018?"),
    ("3M's total net sales in 2018 were $32,765 million. " * 20, "What about 2018?"),
])
def test_rewrite_output_is_cleaned_or_rejected(model_output, expected):
    client = LocalLLMClient.__new__(LocalLLMClient)
    client.llm = SimpleNamespace(invoke=lambda messages: SimpleNamespace(content=model_output))
    history = [("user", "What was 3M's total net sales in 2022?"), ("assistant", "$34,229 million.")]
    assert client.rewrite_followup("What about 2018?", history) == expected
