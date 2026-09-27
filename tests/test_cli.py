import json
import subprocess
import sys


def test_cli_build_search_and_ask(tmp_path):
    docs = tmp_path / "docs.jsonl"
    index = tmp_path / "index.json"
    docs.write_text(
        json.dumps(
            {
                "id": "intro",
                "text": "BETTER-RAG builds local indexes and returns cited grounded answers.",
                "metadata": {"source": "test"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    build = subprocess.run(
        [sys.executable, "-m", "better_rag.cli", "build", str(docs), "--out", str(index)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "indexed" in build.stdout.lower()

    search = subprocess.run(
        [sys.executable, "-m", "better_rag.cli", "search", str(index), "How are answers grounded?", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    hits = json.loads(search.stdout)
    assert hits[0]["document_id"] == "intro"

    ask = subprocess.run(
        [sys.executable, "-m", "better_rag.cli", "ask", str(index), "How are answers grounded?", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(ask.stdout)
    assert payload["passed"] is True
    assert payload["citations"][0]["document_id"] == "intro"


def test_cli_refusal_returns_nonzero(tmp_path):
    docs = tmp_path / "docs.jsonl"
    index = tmp_path / "index.json"
    docs.write_text(json.dumps({"id": "pets", "text": "Dogs like walks."}) + "\n", encoding="utf-8")
    subprocess.run(
        [sys.executable, "-m", "better_rag.cli", "build", str(docs), "--out", str(index)],
        check=True,
        capture_output=True,
        text=True,
    )

    ask = subprocess.run(
        [
            sys.executable,
            "-m",
            "better_rag.cli",
            "ask",
            str(index),
            "How do I configure Kubernetes mTLS?",
            "--min-score",
            "0.7",
            "--json",
        ],
        capture_output=True,
        text=True,
    )

    assert ask.returncode == 2
    assert json.loads(ask.stdout)["passed"] is False


def test_cli_preview_chars_flag(tmp_path):
    docs = tmp_path / "docs.jsonl"
    index = tmp_path / "index.json"
    docs.write_text(
        json.dumps(
            {
                "id": "intro",
                "text": "BETTER-RAG builds local indexes and returns cited grounded answers. " * 20,
                "metadata": {"source": "test"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        [sys.executable, "-m", "better_rag.cli", "build", str(docs), "--out", str(index)],
        check=True, capture_output=True, text=True,
    )
    full = subprocess.run(
        [sys.executable, "-m", "better_rag.cli", "ask", str(index), "grounded answers", "--json"],
        check=True, capture_output=True, text=True,
    )
    preview = subprocess.run(
        [sys.executable, "-m", "better_rag.cli", "ask", str(index), "grounded answers",
         "--json", "--preview-chars", "60"],
        check=True, capture_output=True, text=True,
    )
    full_len = len(json.loads(full.stdout)["citations"][0]["text"])
    prev_text = json.loads(preview.stdout)["citations"][0]["text"]
    assert full_len > 60
    assert len(prev_text) < full_len
    assert "preview of" in prev_text

    search = subprocess.run(
        [sys.executable, "-m", "better_rag.cli", "search", str(index), "grounded answers",
         "--json", "--preview-chars", "60"],
        check=True, capture_output=True, text=True,
    )
    hits = json.loads(search.stdout)
    assert len(hits[0]["text"]) < full_len
