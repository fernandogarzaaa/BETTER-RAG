# Security Policy

BETTER-RAG is a local retrieval and answer synthesis library. It does not call external APIs by default.

## Supported Versions

Security fixes target the latest release on `main`.

## Reporting a Vulnerability

Open a private advisory on GitHub or contact the repository owner directly. Include:

- affected version or commit
- reproduction steps
- expected and actual behavior
- impact assessment

## Current Guardrail Scope

The built-in guardrails are transparent baseline checks for weak retrieval and obviously unsafe retrieved content. They are not a comprehensive policy system. Applications handling sensitive domains should add domain-specific review before returning final answers to users.
