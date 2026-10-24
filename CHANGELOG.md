# Changelog

## Unreleased

- Added: raw conversation history example (`examples/conversation_history.jsonl`
  and `examples/raw_conversation_history.py`) with tests in
  `tests/test_raw_conversation_history.py`. Each turn is indexed verbatim, with
  no summarization or fact extraction, as a dependency-free analogue of the
  Raw-RAG baseline in Bao et al., arXiv:2610.30755. No new dependencies.
  README documents the analogue honestly: no paper benchmark score is claimed
  for BETTER-RAG, which uses BM25-style scoring, cosine term vectors, and
  MMR reranking, not dense retrieval, a cross-encoder reranker, or a reader LLM.
- Fixed: `add_documents` validates the whole batch before changing the index;
  an invalid document no longer leaves the index half-updated (old chunks of a
  re-added id dropped while document-frequency stats went stale).
- Fixed: `search`/`ask` reject `top_k` values that are not positive integers
  (`0` used to silently mean "default" and `-1` returned no hits, so `ask`
  wrongly reported "not enough evidence"). The CLI prints a clean error.
- Performance: BM25 average document length is computed once per query
  instead of once per chunk (search was O(N^2) in chunk count; about 13x
  faster at 3,000 chunks).
- Fixed: re-adding a document with an existing id now removes its old chunks,
  so stale text is no longer returned by `search` or cited by `ask`.
- CI now runs ruff (lint) and mypy (strict type check); added `uv.lock`
  and weekly Dependabot updates for Python deps and GitHub Actions.
- Added `--preview-chars N` to the `search` and `ask` CLI commands: truncates
  hit and citation text in JSON output to N characters for compact previews
  (N must be positive; truncated text notes its full length).
- Tightened guardrail defaults: `min_score` 0.22 to 0.35, `max_risk` 0.7
  to 0.5. The previous defaults let unrelated questions pass with stitched
  answers; the new defaults refuse properly while real questions still
  answer with citations.
- README install section no longer advertises the unavailable
  `pip install better-rag`; source install is the documented path until
  the first PyPI release. Added a tag-triggered PyPI trusted-publishing
  workflow (OIDC, no long-lived token).

## 0.1.0 - 2026-04-30

- Initial open-source release.
- Added dependency-free `better_rag` Python package.
- Added hybrid sparse retrieval, chunking, MMR reranking, extractive answers, citations, refusal behavior, JSON persistence, and CLI.
- Added tests, example corpus, CI, packaging metadata, and open-source project docs.
