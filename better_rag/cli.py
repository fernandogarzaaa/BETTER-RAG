from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from better_rag import BetterRAG, RagConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="better-rag",
        description="Local, cited, guardrailed retrieval-augmented generation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Build an index from JSONL documents")
    build.add_argument("docs", help="JSONL file with id, text, optional metadata")
    build.add_argument("--out", required=True, help="Output index JSON path")
    _add_config_args(build)

    search = sub.add_parser("search", help="Search an existing index")
    search.add_argument("index", help="Index JSON path")
    search.add_argument("query", help="Search query")
    search.add_argument("--top-k", type=int, default=None, help="Number of hits")
    search.add_argument("--json", action="store_true", help="Emit JSON")

    ask = sub.add_parser("ask", help="Ask a grounded question")
    ask.add_argument("index", help="Index JSON path")
    ask.add_argument("query", help="Question")
    ask.add_argument("--top-k", type=int, default=None, help="Number of hits")
    ask.add_argument("--min-score", type=float, default=None, help="Override minimum grounding score")
    ask.add_argument("--max-risk", type=float, default=None, help="Override guard risk budget")
    ask.add_argument("--json", action="store_true", help="Emit JSON")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "build":
            return _cmd_build(args)
        if args.command == "search":
            return _cmd_search(args)
        if args.command == "ask":
            return _cmd_ask(args)
    except ValueError as e:
        print(f"better-rag: error: {e}", file=sys.stderr)
        return 1

    parser.error(f"unknown command {args.command}")
    return 1


def _cmd_build(args: argparse.Namespace) -> int:
    config = RagConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        min_score=args.min_score,
        max_risk=args.max_risk,
        mmr_lambda=args.mmr_lambda,
    )
    rag = BetterRAG.from_jsonl(args.docs, config=config)
    rag.save(args.out)
    print(f"indexed {Path(args.docs)} -> {Path(args.out)}")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    rag = BetterRAG.load(args.index)
    hits = rag.search(args.query, top_k=args.top_k)
    if args.json:
        print(json.dumps([hit.to_dict() for hit in hits], indent=2))
    else:
        for hit in hits:
            print(f"{hit.document_id} {hit.chunk_id} score={hit.score:.3f}\n{hit.text}\n")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    rag = BetterRAG.load(args.index)
    if args.min_score is not None or args.max_risk is not None:
        rag.config = replace(
            rag.config,
            min_score=args.min_score if args.min_score is not None else rag.config.min_score,
            max_risk=args.max_risk if args.max_risk is not None else rag.config.max_risk,
        )
        rag.config.validate()
    result = rag.ask(args.query, top_k=args.top_k)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.answer)
        print(f"\nconfidence={result.confidence:.3f} passed={result.passed}")
        if result.citations:
            print("\nCitations:")
            for citation in result.citations:
                print(f"- {citation.document_id} ({citation.chunk_id}) score={citation.score:.3f}")
        if result.violations:
            print("\nViolations:")
            for violation in result.violations:
                print(f"- {violation}")
    return 0 if result.passed else 2


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    defaults = RagConfig()
    parser.add_argument("--chunk-size", type=int, default=defaults.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=defaults.chunk_overlap)
    parser.add_argument("--top-k", type=int, default=defaults.top_k)
    parser.add_argument("--min-score", type=float, default=defaults.min_score)
    parser.add_argument("--max-risk", type=float, default=defaults.max_risk)
    parser.add_argument("--mmr-lambda", type=float, default=defaults.mmr_lambda)


if __name__ == "__main__":
    raise SystemExit(main())
