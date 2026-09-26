# Changelog

## Unreleased

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
