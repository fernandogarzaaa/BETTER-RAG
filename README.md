# BETTER-RAG

BETTER-RAG is a local, dependency-free retrieval-augmented generation engine for grounded answers. It gives developers a complete RAG baseline that is easy to inspect, easy to ship, and safe by default:

- hybrid sparse retrieval with BM25-style scoring plus cosine term vectors
- chunking with overlap and MMR reranking
- extractive answers grounded in retrieved context
- citations for every accepted answer
- refusal when evidence is weak
- simple safety guardrails for unsafe retrieved content
- JSON persistence and a production-friendly CLI

It runs with the Python standard library only.

## Install

```bash
pip install better-rag
```

From source:

```bash
git clone https://github.com/fernandogarzaaa/BETTER-RAG
cd BETTER-RAG
python -m pip install -e ".[dev]"
```

## Quick Start

Create a JSONL corpus:

```jsonl
{"id":"intro","text":"BETTER-RAG builds local indexes and returns cited grounded answers.","metadata":{"source":"docs"}}
{"id":"guards","text":"When retrieval evidence is weak, BETTER-RAG refuses instead of guessing.","metadata":{"source":"docs"}}
```

Build an index:

```bash
better-rag build examples/corpus.jsonl --out .better-rag-index.json
```

Ask a grounded question:

```bash
better-rag ask .better-rag-index.json "How does BETTER-RAG avoid guessing?" --json
```

Search without answer synthesis:

```bash
better-rag search .better-rag-index.json "grounded answers" --json
```

## Python API

```python
from better_rag import BetterRAG, Document, RagConfig

rag = BetterRAG(RagConfig(min_score=0.2, max_risk=0.7))
rag.add_documents([
    Document(
        id="intro",
        text="BETTER-RAG grounds answers in retrieved evidence and returns citations.",
        metadata={"source": "docs"},
    )
])

result = rag.ask("How are answers grounded?")
print(result.answer)
print(result.citations)
```

## Corpus Format

Each JSONL row must contain:

| Field | Required | Description |
|---|---:|---|
| `id` | yes | Stable document identifier |
| `text` | yes | Document text to chunk and index |
| `metadata` | no | Any JSON object copied to search hits and citations |

## Guardrails

BETTER-RAG rejects answers when:

- retrieval scores are below `min_score`
- confidence falls below the risk budget implied by `max_risk`
- extracted evidence contains configured unsafe terms

The default guardrails are intentionally conservative and transparent. They are not a substitute for a full policy engine, but they keep the baseline from silently fabricating unsupported answers.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m build
```

## License

MIT
