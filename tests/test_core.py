import json

import pytest

from better_rag import BetterRAG, Document, RagConfig


def test_ingest_search_and_answer_with_citations():
    rag = BetterRAG(RagConfig(min_score=0.18, max_risk=0.85, chunk_size=20))
    rag.add_documents(
        [
            Document(
                id="rag",
                text="Better RAG grounds answers in retrieved evidence. Every answer includes citations and a confidence score.",
                metadata={"source": "docs"},
            ),
            Document(
                id="weather",
                text="Weather reports include temperature, rainfall, and wind speed.",
            ),
        ]
    )

    hits = rag.search("How does Better RAG ground answers?", top_k=2)
    result = rag.ask("How does Better RAG ground answers?")

    assert hits[0].document_id == "rag"
    assert result.passed is True
    assert "retrieved evidence" in result.answer
    assert result.citations[0].document_id == "rag"
    assert result.confidence >= 0.25


def test_refuses_when_retrieval_is_weak():
    rag = BetterRAG(RagConfig(min_score=0.5))
    rag.add_documents([Document(id="a", text="Cats sleep in warm places.")])

    result = rag.ask("How do I rotate cloud database credentials?")

    assert result.passed is False
    assert "not enough evidence" in result.answer.lower()
    assert result.citations == []
    assert result.violations


def test_guard_blocks_unsafe_context():
    rag = BetterRAG(RagConfig(min_score=0.05, max_risk=0.95))
    rag.add_documents([Document(id="unsafe", text="This page describes a dangerous hack.")])

    result = rag.ask("What dangerous hack is described?")

    assert result.passed is False
    assert any("Forbidden content" in item for item in result.violations)


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "index.json"
    rag = BetterRAG(RagConfig(min_score=0.1))
    rag.add_documents([Document(id="release", text="BETTER-RAG can save and load local indexes.")])
    rag.save(path)

    restored = BetterRAG.load(path)
    result = restored.ask("Can BETTER-RAG save indexes?")

    assert result.passed is True
    assert result.citations[0].document_id == "release"


def test_from_jsonl_rejects_bad_records(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text(json.dumps({"id": "missing-text"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="text"):
        BetterRAG.from_jsonl(path)


def test_from_jsonl_accepts_utf8_bom(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text(json.dumps({"id": "bom", "text": "BOM files should load."}) + "\n", encoding="utf-8-sig")

    rag = BetterRAG.from_jsonl(path, config=RagConfig(min_score=0.1))
    result = rag.ask("Should BOM files load?")

    assert result.passed is True
