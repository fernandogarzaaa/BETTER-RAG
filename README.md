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

From source:

```bash
git clone https://github.com/fernandogarzaaa/BETTER-RAG
cd BETTER-RAG
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
```

Use a virtual environment: on Debian/Ubuntu systems, installing into the
system Python is blocked (`externally-managed-environment`), so the venv
above is required, not optional.

You can also run the CLI without installing, straight from the source tree:

```bash
PYTHONPATH=. python3 -m better_rag --help
```

A PyPI release (`pip install better-rag`) is pending: the repo ships a
trusted-publishing workflow (`.github/workflows/publish.yml`) that publishes
to PyPI on version tags (`v*`), so the first tagged release will land on PyPI
with no further changes. Until then, install from source as above.

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

For compact JSON output, `search` and `ask` accept `--preview-chars N`,
which truncates hit and citation text to N characters (N must be positive).
Shorter texts are left untouched; truncated text ends with a note like
`… [preview of 1200 chars]` so the full length stays visible:

```bash
better-rag search .better-rag-index.json "grounded answers" --json --preview-chars 200
```

## Python API

```python
from better_rag import BetterRAG, Document, RagConfig

rag = BetterRAG(RagConfig(min_score=0.35, max_risk=0.5))
rag.add_documents([
    Document(
        id="intro",
        text="BETTER-RAG returns grounded answers with citations from retrieved evidence.",
        metadata={"source": "docs"},
    )
])

result = rag.ask("How are answers grounded?")
print(result.answer)
print(result.citations)
```

Note: retrieval is exact-term based (no stemming), so the query and the
documents must share the same word forms for the example above to pass.

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

Fail-closed citation behavior: when an answer is blocked by the unsafe-content
guardrail, the citations keep their document ids, chunk ids, scores, and
metadata, but the flagged citation *text* is redacted (empty string). Blocked
evidence is never shipped back inside the response.

Forbidden-term matching is word-boundary aware: "hack" fires on the word
"hack", not inside larger words like "shackles".

## Raw conversation history

BETTER-RAG can index a conversation turn by turn, verbatim, with no
summarization and no fact extraction. Each turn stays a separate document,
so a later question can retrieve the original wording, the relations
between turns, and the dependency path behind an answer, instead of a
rewritten summary that may have dropped those details.

This is the dependency-free analogue of the Raw-RAG baseline in Bao and
others, "Is Complex Agentic Memory Always Needed? Naive RAG Is Stronger
Than We Thought" (arXiv:2610.30755). That paper reports raw retrieval
plus reranking over raw history ranking first in six of seven
benchmark-model settings across LoCoMo, LongMemEval, MEME, and
MemoryAgentBench, and BEAM-1M ingestion in 0.3 hours versus 3.2 to 27
hours for the compared agentic memory systems.

It is an analogue, not a reproduction. The paper uses dense retrieval, a
cross-encoder reranker, and a reader LLM. BETTER-RAG uses BM25-style
scoring, cosine term vectors, and MMR reranking, and returns extractive
answers with citations. No benchmark score from the paper is claimed for
BETTER-RAG itself.

Run the example from the repo root:

```bash
PYTHONPATH=. python3 examples/raw_conversation_history.py
```

Or build the conversation corpus with the CLI and ask it directly:

```bash
better-rag build examples/conversation_history.jsonl --out .conversation-index.json
better-rag ask .conversation-index.json "When is the hotel check-in and late checkout?" --json
```

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
python -m pytest
python -m build
```

## License

MIT
