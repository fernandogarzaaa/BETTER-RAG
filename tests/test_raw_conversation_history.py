from pathlib import Path

from better_rag import BetterRAG

EXAMPLE = Path(__file__).parent.parent / "examples" / "conversation_history.jsonl"


def test_raw_history_answers_from_original_turn():
    rag = BetterRAG.from_jsonl(EXAMPLE)
    result = rag.ask("When is the hotel check-in and late checkout?")

    assert result.passed is True
    assert "April 12" in result.answer
    assert "April 15" in result.answer
    assert result.citations[0].document_id == "turn-03"


def test_raw_history_preserves_original_turn_text():
    rag = BetterRAG.from_jsonl(EXAMPLE)
    hits = rag.search("Maya vegetarian April 13")

    assert any(hit.document_id == "turn-07" for hit in hits)
    turn = next(hit for hit in hits if hit.document_id == "turn-07")
    assert turn.text == (
        "User: My friend Maya is joining on April 13, and she is vegetarian."
    )


def test_raw_history_still_refuses_weak_evidence():
    rag = BetterRAG.from_jsonl(EXAMPLE)
    result = rag.ask("How do I rotate cloud database credentials?")

    assert result.passed is False
