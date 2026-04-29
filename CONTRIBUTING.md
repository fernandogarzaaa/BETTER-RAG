# Contributing

BETTER-RAG is intentionally small and dependency-free. Contributions should keep the default path easy to audit and easy to run locally.

## Development Setup

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Pull Request Bar

- Add or update tests for behavior changes.
- Keep public API changes documented in `README.md`.
- Do not add required runtime dependencies without a clear reason.
- Run `python -m pytest` and `python -m build` before submitting.

## Design Principles

- Prefer explicit citations over fluent unsupported answers.
- Refuse weakly grounded answers.
- Keep persistence formats JSON-readable.
- Make failure modes visible in result objects and CLI JSON.
