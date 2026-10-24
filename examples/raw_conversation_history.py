"""Raw conversation history example for BETTER-RAG.

Each conversation turn is indexed verbatim, with no summarization and no
fact extraction. That preserves the original wording, the relations between
turns, and the dependency path a later question may need.

This is the dependency-free analogue of the Raw-RAG baseline in Bao et al.,
"Is Complex Agentic Memory Always Needed? Naive RAG Is Stronger Than We
Thought" (arXiv:2610.30755): retrieval over raw history, before any
agentic memory layer. It is not a reproduction of that paper, which uses
dense retrieval, a cross-encoder reranker, and a reader LLM. BETTER-RAG
uses BM25-style scoring, cosine term vectors, and MMR reranking, and
returns extractive answers with citations.

Run from the repo root:

    PYTHONPATH=. python3 examples/raw_conversation_history.py
"""

from pathlib import Path

from better_rag import BetterRAG

QUESTIONS = [
    "When is the hotel check-in and late checkout?",
    "What is the user allergic to?",
]


def main() -> None:
    path = Path(__file__).with_name("conversation_history.jsonl")
    rag = BetterRAG.from_jsonl(path)
    for question in QUESTIONS:
        result = rag.ask(question)
        print(f"Q: {question}")
        print(f"Passed: {result.passed}")
        print(f"A: {result.answer}")
        for citation in result.citations:
            print(f"  [{citation.document_id}] {citation.text}")
        print()


if __name__ == "__main__":
    main()
