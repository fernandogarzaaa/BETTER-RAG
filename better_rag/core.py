from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for",
    "from", "how", "i", "in", "is", "it", "of", "on", "or", "that", "the",
    "to", "what", "when", "where", "with",
}
_FORBIDDEN = {"weapon", "bomb", "kill", "poison", "hack", "illegal", "dangerous", "harmful"}


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RagConfig:
    chunk_size: int = 120
    chunk_overlap: int = 24
    top_k: int = 4
    min_score: float = 0.22
    max_risk: float = 0.7
    mmr_lambda: float = 0.75

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if self.chunk_overlap >= self.chunk_size:
            object.__setattr__(self, "chunk_overlap", max(0, self.chunk_size // 5))
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError("min_score must be between 0 and 1")
        if not 0.0 <= self.max_risk <= 1.0:
            raise ValueError("max_risk must be between 0 and 1")
        if not 0.0 <= self.mmr_lambda <= 1.0:
            raise ValueError("mmr_lambda must be between 0 and 1")


@dataclass(frozen=True)
class _Chunk:
    id: str
    document_id: str
    text: str
    metadata: dict[str, Any]
    tokens: list[str]
    vector: dict[str, float]


@dataclass(frozen=True)
class SearchHit:
    document_id: str
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": round(self.score, 4),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class Citation:
    document_id: str
    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "score": round(self.score, 4),
            "text": self.text,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class RagResult:
    answer: str
    passed: bool
    confidence: float
    citations: list[Citation] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "passed": self.passed,
            "confidence": round(self.confidence, 4),
            "citations": [citation.to_dict() for citation in self.citations],
            "violations": list(self.violations),
            "trace": list(self.trace),
        }


class BetterRAG:
    """Small, dependency-free RAG engine with hybrid retrieval and citations."""

    def __init__(self, config: RagConfig | None = None) -> None:
        self.config = config or RagConfig()
        self.config.validate()
        self._documents: dict[str, Document] = {}
        self._chunks: list[_Chunk] = []
        self._doc_freq: dict[str, int] = {}

    def add_documents(self, documents: Iterable[Document | dict[str, Any]]) -> None:
        docs = [_coerce_document(item) for item in documents]
        if not docs:
            raise ValueError("at least one document is required")
        for doc in docs:
            _validate_document(doc)
            self._documents[doc.id] = doc
            for chunk_id, text in _chunk_text(doc.id, doc.text, self.config.chunk_size, self.config.chunk_overlap):
                tokens = _tokens(text)
                self._chunks.append(
                    _Chunk(
                        id=chunk_id,
                        document_id=doc.id,
                        text=text,
                        metadata=dict(doc.metadata),
                        tokens=tokens,
                        vector=_term_vector(tokens),
                    )
                )
        self._rebuild_doc_freq()

    def search(self, query: str, *, top_k: int | None = None) -> list[SearchHit]:
        query_tokens = _tokens(query)
        if not query_tokens:
            raise ValueError("query must contain searchable terms")
        if not self._chunks:
            raise ValueError("index is empty")
        k = top_k or self.config.top_k
        query_vector = _term_vector(query_tokens)
        scored = []
        for chunk in self._chunks:
            vector_score = _cosine_sparse(query_vector, chunk.vector)
            bm25_score = self._bm25(query_tokens, chunk.tokens)
            lexical_overlap = _overlap(query_tokens, chunk.tokens)
            score = (0.48 * vector_score) + (0.37 * bm25_score) + (0.15 * lexical_overlap)
            scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = _mmr_select(scored, k, self.config.mmr_lambda)
        return [
            SearchHit(
                document_id=chunk.document_id,
                chunk_id=chunk.id,
                text=chunk.text,
                score=max(0.0, min(1.0, score)),
                metadata=chunk.metadata,
            )
            for score, chunk in selected
        ]

    def ask(self, query: str, *, top_k: int | None = None) -> RagResult:
        hits = self.search(query, top_k=top_k)
        trace = [f"retrieved {len(hits)} hit(s)"]
        grounded = [hit for hit in hits if hit.score >= self.config.min_score]
        if not grounded:
            best = hits[0].score if hits else 0.0
            violation = f"best score {best:.4f} below min_score {self.config.min_score:.4f}"
            return RagResult(
                answer="There is not enough evidence in the index to answer that.",
                passed=False,
                confidence=best,
                violations=[violation],
                trace=trace + [violation],
            )

        answer = self._extract_answer(query, grounded)
        confidence = self._confidence(query, grounded)
        violations = []
        required = 1.0 - self.config.max_risk
        if confidence < required:
            violations.append(f"confidence {confidence:.4f} below required {required:.4f}")
        unsafe = _safety_violations(answer)
        violations.extend(unsafe)
        passed = not violations
        citations = [
            Citation(
                document_id=hit.document_id,
                chunk_id=hit.chunk_id,
                score=hit.score,
                text=hit.text,
                metadata=hit.metadata,
            )
            for hit in grounded
        ]
        trace.append(f"confidence={confidence:.4f}")
        trace.append("guard passed" if passed else "guard failed")
        return RagResult(
            answer=answer if passed else "Retrieved evidence was blocked by safety guards.",
            passed=passed,
            confidence=confidence,
            citations=citations if passed else citations,
            violations=violations,
            trace=trace,
        )

    def save(self, path: str | Path) -> None:
        payload = {
            "config": asdict(self.config),
            "documents": [asdict(doc) for doc in self._documents.values()],
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BetterRAG":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        rag = cls(RagConfig(**payload.get("config", {})))
        rag.add_documents(payload.get("documents", []))
        return rag

    @classmethod
    def from_jsonl(cls, path: str | Path, config: RagConfig | None = None) -> "BetterRAG":
        documents = []
        with Path(path).open(encoding="utf-8-sig") as f:
            for lineno, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"invalid JSON on line {lineno}: {e}") from e
                documents.append(_coerce_document(item))
        rag = cls(config)
        rag.add_documents(documents)
        return rag

    def _rebuild_doc_freq(self) -> None:
        self._doc_freq.clear()
        for chunk in self._chunks:
            for token in set(chunk.tokens):
                self._doc_freq[token] = self._doc_freq.get(token, 0) + 1

    def _bm25(self, query_tokens: list[str], chunk_tokens: list[str]) -> float:
        if not chunk_tokens:
            return 0.0
        n = max(len(self._chunks), 1)
        avgdl = sum(len(chunk.tokens) for chunk in self._chunks) / n
        k1 = 1.5
        b = 0.75
        score = 0.0
        for token in query_tokens:
            freq = chunk_tokens.count(token)
            if freq == 0:
                continue
            df = self._doc_freq.get(token, 0)
            idf = math.log(1 + ((n - df + 0.5) / (df + 0.5)))
            denom = freq + k1 * (1 - b + b * (len(chunk_tokens) / max(avgdl, 1e-9)))
            score += idf * ((freq * (k1 + 1)) / denom)
        return min(score / max(len(set(query_tokens)), 1), 1.0)

    def _extract_answer(self, query: str, hits: list[SearchHit]) -> str:
        query_tokens = _tokens(query)
        ranked = []
        for hit in hits:
            for sentence in _sentences(hit.text):
                ranked.append((_overlap(query_tokens, _tokens(sentence)), hit.score, sentence))
        ranked.sort(key=lambda item: (item[0], item[1], len(item[2])), reverse=True)
        selected = [sentence for overlap, _, sentence in ranked if overlap > 0][:3]
        if not selected:
            selected = [hits[0].text]
        return " ".join(selected)

    @staticmethod
    def _confidence(query: str, hits: list[SearchHit]) -> float:
        query_terms = set(_tokens(query))
        context_terms = set()
        for hit in hits:
            context_terms.update(_tokens(hit.text))
        coverage = len(query_terms.intersection(context_terms)) / max(len(query_terms), 1)
        return max(0.0, min(1.0, (0.6 * hits[0].score) + (0.4 * coverage)))


def _coerce_document(item: Document | dict[str, Any]) -> Document:
    if isinstance(item, Document):
        return item
    return Document(
        id=str(item.get("id", "")),
        text=str(item.get("text", "")),
        metadata=dict(item.get("metadata", {})),
    )


def _validate_document(doc: Document) -> None:
    if not doc.id.strip():
        raise ValueError("document id must not be empty")
    if not doc.text.strip():
        raise ValueError(f"document {doc.id!r} text must not be empty")


def _tokens(text: str) -> list[str]:
    return [token for token in (m.group(0).lower() for m in _TOKEN_RE.finditer(text)) if token not in _STOPWORDS]


def _term_vector(tokens: list[str]) -> dict[str, float]:
    counts: dict[str, float] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0.0) + 1.0
    norm = math.sqrt(sum(value * value for value in counts.values()))
    if norm == 0:
        return counts
    return {token: value / norm for token, value in counts.items()}


def _cosine_sparse(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(token, 0.0) for token, value in left.items())


def _overlap(left: list[str], right: list[str]) -> float:
    if not left:
        return 0.0
    return len(set(left).intersection(right)) / len(set(left))


def _chunk_text(document_id: str, text: str, chunk_size: int, overlap: int) -> Iterable[tuple[str, str]]:
    tokens = text.split()
    if len(tokens) <= chunk_size:
        yield f"{document_id}:0", text.strip()
        return
    step = max(1, chunk_size - overlap)
    for index, start in enumerate(range(0, len(tokens), step)):
        chunk = " ".join(tokens[start:start + chunk_size]).strip()
        if chunk:
            yield f"{document_id}:{index}", chunk
        if start + chunk_size >= len(tokens):
            break


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_RE.split(text.strip()) if part.strip()]


def _mmr_select(scored: list[tuple[float, _Chunk]], top_k: int, lambda_: float) -> list[tuple[float, _Chunk]]:
    selected: list[tuple[float, _Chunk]] = []
    candidates = list(scored)
    while candidates and len(selected) < top_k:
        if not selected:
            selected.append(candidates.pop(0))
            continue
        best_index = 0
        best_score = float("-inf")
        for index, (score, chunk) in enumerate(candidates):
            redundancy = max(_cosine_sparse(chunk.vector, chosen.vector) for _, chosen in selected)
            mmr = (lambda_ * score) - ((1.0 - lambda_) * redundancy)
            if mmr > best_score:
                best_score = mmr
                best_index = index
        selected.append(candidates.pop(best_index))
    return selected


def _safety_violations(text: str) -> list[str]:
    lower = text.lower()
    return [f"Forbidden content '{term}'" for term in sorted(_FORBIDDEN) if term in lower]
