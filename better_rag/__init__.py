"""BETTER-RAG: local, cited, guardrailed retrieval-augmented generation."""

from better_rag.core import (
    BetterRAG,
    Citation,
    Document,
    RagConfig,
    RagResult,
    SearchHit,
)

__all__ = [
    "BetterRAG",
    "Citation",
    "Document",
    "RagConfig",
    "RagResult",
    "SearchHit",
]

__version__ = "0.1.0"
