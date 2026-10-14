from query_engine import MAX_SCOPED_SOURCES, _tokens, scope_sources

SEC_IGNORE = frozenset({"10", "k", "q", "8"})
FILINGS = [
    "3M_2018_10K.pdf", "3M_2022_10K.pdf", "3M_2023Q2_10Q.pdf",
    "ADOBE_2022_10K.pdf", "NIKE_2022_10K.pdf", "COCACOLA_2022_10K.pdf",
    "LOCKHEEDMARTIN_2022_10K.pdf", "PEPSICO_2022_10K.pdf", "PEPSICO_2021_10K.pdf",
    "AMAZON_2019_10K.pdf", "BOEING_2022_10K.pdf", "VERIZON_2022_10K.pdf",
]


def scope(question):
    return scope_sources(question, FILINGS, SEC_IGNORE)


def test_tokens_split_letters_from_digits_and_drop_generic_words():
    assert {"2022", "q", "2"} <= _tokens("2022Q2")
    assert "the" not in _tokens("What was the revenue")
    assert "pdf" not in _tokens("report.pdf")


def test_tokens_join_adjacent_words_to_match_run_together_filenames():
    assert "cocacola" in _tokens("Coca-Cola")
    assert "lockheedmartin" in _tokens("Lockheed Martin")


def test_ignore_words_are_dropped():
    assert "10" in _tokens("10-K")
    assert not {"10", "k"} & _tokens("10-K", SEC_IGNORE)


def test_company_and_year_pick_one_filing():
    assert scope("What was 3M's capital expenditure in 2018?") == ["3M_2018_10K.pdf"]


def test_company_alone_picks_all_its_filings():
    assert scope("Summarize 3M's segment results") == sorted(f for f in FILINGS if f.startswith("3M"))


def test_joined_words_match_filenames():
    assert scope("What was Coca Cola's revenue in 2022?") == ["COCACOLA_2022_10K.pdf"]
    assert scope("Lockheed Martin backlog in 2022") == ["LOCKHEEDMARTIN_2022_10K.pdf"]


def test_comparison_keeps_every_named_company():
    assert scope("Compare Nike and Adobe revenue growth in 2022") == ["ADOBE_2022_10K.pdf", "NIKE_2022_10K.pdf"]


def test_vague_question_falls_back_to_global_search():
    # A year alone matches far more than MAX_SCOPED_SOURCES files
    assert len([f for f in FILINGS if "2022" in f]) > MAX_SCOPED_SOURCES
    assert scope("What happened in 2022?") == []
    assert scope("What are the main risks?") == []


def test_needs_at_least_two_sources():
    assert scope_sources("3M revenue", ["3M_2022_10K.pdf"]) == []
    assert scope_sources("3M revenue", []) == []
