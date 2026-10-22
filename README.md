# carbon-nanotubes-rag-mcp

[![CI](https://github.com/slastrzelec/carbon-nanotubes-rag-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/slastrzelec/carbon-nanotubes-rag-mcp/actions/workflows/ci.yml)

Read-only [Model Context Protocol](https://modelcontextprotocol.io) server that exposes the
[carbon-nanotubes-rag](https://github.com/slastrzelec/carbon-nanotubes-rag) system (hybrid FAISS + BM25
retrieval over papers on Raman spectroscopy of carbon nanotubes) as MCP tools.

> Status: milestones M2-M4 of [SPEC.md](SPEC.md) — stdio and streamable HTTP transports, Docker image, CI,
> LangGraph client agent and tool-choice evaluation harness. The held-out evaluation has not been run yet.

## MCP surface

| Kind | Name | Backend call | Notes |
|---|---|---|---|
| Tool | `search_papers(query, top_k=5)` | `POST /search` | passages only, no LLM cost |
| Tool | `ask_papers(question)` | `POST /query` | paid LLM call, budgeted, never retried |
| Tool | `list_papers()` | `GET /documents` | corpus inventory |
| Resource | `papers://index` | `GET /documents` | same list as JSON |
| Prompt | `literature_summary(topic)` | none | tells the model to search first and cite |

All tools are annotated read-only and closed-world, return structured output with a published schema,
and wrap paper text and generated answers in delimited "untrusted content" blocks.

## Run locally

```powershell
cd "C:\Users\slast\PYTHON\0_projekty do portfolio\carbon-nanotubes-rag-mcp"
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

The server expects the RAG API on `http://127.0.0.1:8000`. To run it from a local clone of
carbon-nanotubes-rag (default path `../13_RAG_raman`, override with `RAG_REPO_PATH`):

```powershell
cd "C:\Users\slast\PYTHON\0_projekty do portfolio\carbon-nanotubes-rag-mcp"
copy .env.example .env   # then set OPENAI_API_KEY in .env (used only by the backend container)
docker compose -f docker-compose.backend.yml up --build
```

Start the MCP server over stdio with `rag-mcp` (or `python -m rag_mcp`).

### Streamable HTTP

```powershell
cd "C:\Users\slast\PYTHON\0_projekty do portfolio\carbon-nanotubes-rag-mcp"
$env:RAG_MCP_HTTP_TOKEN = python -c "import secrets; print(secrets.token_urlsafe(32))"
rag-mcp --transport http   # http://127.0.0.1:8765/mcp, header: Authorization: Bearer <token>
```

Or both containers (RAG API + MCP server; needs `RAG_MCP_HTTP_TOKEN` in `.env`):

```powershell
cd "C:\Users\slast\PYTHON\0_projekty do portfolio\carbon-nanotubes-rag-mcp"
docker compose -f docker-compose.backend.yml --profile http up --build
```

Authentication is a single static bearer token (>= 32 characters, constant-time comparison, never logged).
This is a deliberate simplification for local or single-user use, not the MCP OAuth 2.1 authorization
flow. The server binds to 127.0.0.1 by default and keeps the SDK's DNS-rebinding protection (Host and
Origin allow-lists) enabled.

## Client agent

A LangGraph agent (explicit `StateGraph`: model node, tool node, hard limit of 6 tool calls per question)
that connects to the server over MCP and answers with document citations. It uses a small in-repo
MCP-to-LangChain adapter because `langchain-mcp-adapters` does not yet support `mcp` 2.x.

```powershell
cd "C:\Users\slast\PYTHON\0_projekty do portfolio\carbon-nanotubes-rag-mcp"
pip install -e ".[agent]"
# OPENAI_API_KEY in .env; RAG API running (docker-compose.backend.yml)
rag-agent ask "How is the RBM frequency related to nanotube diameter?"
```

The agent launches the MCP server over stdio with an environment allow-list, so the OpenAI key stays in
the agent process and never reaches the server.

## Results

Tool-choice evaluation (SPEC section 5): 40 hand-written questions in 5 categories (inventory, evidence,
answer, out of corpus, injection), split 15 dev / 25 held-out before any run, the held-out split frozen by
a SHA-256 checksum. Each question runs 3 times with `gpt-4o-mini` at temperature 0 and passes on a 2-of-3
majority; results are reported with Wilson 95% intervals.

**Held-out results: not run yet.** Run with `rag-agent eval --split test`; the report is written to
`eval/results/test-<version>.json` and is never overwritten for the same version.

## Security

- The server holds no LLM key; the backend URL is fixed at start-up and checked against a host
  allow-list; tools accept only declared parameters; redirects and proxy variables are ignored.
- Backend error text is never forwarded to the model (fixed messages only).
- Budgets per server process: `ask_papers` 6/min and 100/day, free tools 30/min (configurable).
- Audit log on stderr: tool, keyed hash of arguments, duration, outcome, backend status. No question,
  passage or answer text. The RAG backend itself logs question text; this repo does not change that.

## Testing

**Verified (no API key, no network, no Docker):**
- Config: backend URL allow-list, rejection of credentials, paths, foreign hosts and bad numbers.
- Backend client: request shapes against fixtures of the real API responses, error mapping, retry policy
  (free endpoints only; `/query` never retried), redirect, oversized and malformed responses.
- MCP contract: in-memory client checks tool names, descriptions, schemas and annotations against a
  reviewed snapshot, and calls every tool, the resource and the prompt.
- Security: oversized or blank input rejected before any backend call, extra arguments cannot change the
  backend host, backend error text never reaches the model, budgets enforced, injected delimiters cannot
  close the untrusted block, audit log contains no request or response text.
- Property-based (Hypothesis): budget limits never exceeded, delimiter neutralisation, deduplication.
- stdio end-to-end: the server runs as a subprocess and degrades to a fixed error without a backend.
- HTTP transport: a real uvicorn server on a loopback port; requests without the exact token (missing,
  wrong, same-length, wrong scheme) get 401 before reaching MCP; forged Host or Origin headers are rejected
  even with a valid token; an authorised client runs a full MCP session; the token never appears in logs.
- CLI: refuses to start with an unsafe configuration (missing or short token, foreign backend host) with
  a one-line error and no traceback.
- Client agent (scripted fake model, real server in memory): the adapter exposes the server's exact
  schemas, returns structured results and error messages; the graph searches and cites returned documents,
  stops a looping model at the tool-call budget, survives backend failures; injected passage text reaches
  the model only inside a fenced tool message; the server process never receives the OpenAI key.
- Evaluation harness: question-set shape and frozen held-out checksum, every scoring rule, Wilson interval
  (known values and a property test), majority and consistency, an end-to-end dev run, refusal to overwrite
  a held-out result or to run with a modified test split.
- Mutation check (`scripts/mutation_check.py`, 18 mutants): re-introducing known bugs must fail the suite.
- Repo hygiene: no `.env`, keys or corpus data tracked; the server package never reads an LLM key.

**CI (GitHub Actions):** ruff, pytest with a 90% coverage gate on Python 3.11-3.13 (Linux) and 3.12
(Windows), mutation check, pip-audit, gitleaks over the full history, and a Docker job that builds the
image and checks that the container refuses to start without a token, answers 401 without it, runs as a
non-root user and does not log the token. Dependabot watches pip, GitHub Actions and the base image.

**Not covered:** the real RAG backend (mocked at HTTP level), answer quality (belongs to the RAG project's
RAGAs evaluation), load and concurrency, budgets across several server processes, the MCP OAuth flow,
the agent's behaviour with a real model (measured separately by the evaluation, not asserted in CI).

## License

MIT
