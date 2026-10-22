# SPEC: carbon-nanotubes-rag-mcp — MCP server for the Carbon Nanotubes RAG + LangGraph client agent

Status: v0.2 APPROVED (2026-10-22). D5 accepted (option a) and merged in the RAG repo. M2 implemented.
Changes from v0.1: backend contract verified against `carbon-nanotubes-rag@f9b04be` (section 10); decisions D1-D4 accepted; new decision D5 (retrieval-only endpoint); error mapping, rate-budget alignment, deduplication and evaluation-leakage rules added.

## 1. Goal

Expose the existing Carbon Nanotubes RAG system as a **Model Context Protocol (MCP) server** with read-only tools, and show it used by an **AI agent client** (LangGraph) that answers with citations. The project demonstrates MCP server design, tool schemas, safe tool use by an agent, a measured tool-choice evaluation, CI and containerization.

Non-goals: changes to retrieval, generation or evaluation of the RAG; write or admin tools; multi-tenant auth or the full MCP OAuth 2.1 authorization flow; a UI.

## 2. Architecture

```
MCP clients: Claude Desktop (stdio) | LangGraph agent (stdio; HTTP in M3)
        |
   carbon-nanotubes-rag-mcp server (Python, official MCP SDK)   <- holds NO LLM key
        | HTTP, fixed base URL from config (default http://127.0.0.1:8000)
   Existing RAG REST API: /health, /documents, /query (+ /search, see D5)
```

- **D1 (accepted):** the server calls the REST API only; it never imports RAG code and never holds the OpenAI key.
- **D2 (accepted):** new repo `carbon-nanotubes-rag-mcp`; the RAG repo is a read-only dependency.
- **D3 (accepted):** demos and manual runs use the RAG API in local Docker. This repo ships `docker-compose.backend.yml` that builds the API from a local clone of the RAG repo and publishes it on `127.0.0.1:8000` only (the RAG repo's own compose file publishes on all interfaces and is not changed). The live Render API is not used (512 MB memory restarts).
- **D4 (accepted):** v1 = stdio server + agent; streamable HTTP with a bearer token in M3.
- **Agent LLM:** the agent process (client side) uses the owner's OpenAI key from `.env`; the MCP server never sees it.

## 3. MCP surface (all read-only)

| Kind | Name | Input | Backend call | Output |
|---|---|---|---|---|
| Tool | `search_papers` | `query` (1-500 chars), `top_k` (1-10, default 5) | `POST /search` (D5) | passages: rank, document, score, text; no generation, no LLM cost |
| Tool | `ask_papers` | `question` (1-500 chars) | `POST /query` (top_k fixed at 5) | grounded answer + cited passages; backend answer is passed through, including its explicit "not in the context" reply |
| Tool | `list_papers` | none | `GET /documents` | indexed documents (document name) and count |
| Resource | `papers://index` | none | `GET /documents` | same list as a resource |
| Prompt | `literature_summary` | `topic` | none | template telling the model to use the tools and cite documents |

- All tools carry `readOnlyHint=true`, `destructiveHint=false`, `openWorldHint=false`. Descriptions are part of agent behaviour and are snapshot-tested.
- **Document identity:** the backend has no paper id, title or chunk id; the only identifier is the PDF filename (some are non-descriptive, e.g. `R1.pdf`). v1 cites by filename. A curated filename -> title/DOI map is out of v1 scope (optional later, metadata only, no paper text).
- **Deduplication:** one paper is indexed twice (`Jorio_2003_New_J._Phys._5_139.pdf` and `R11 Jorio_2003_...pdf`, 76 of 79 chunks identical). The server drops passages whose text exactly matches a higher-ranked passage in the same response and reports how many were dropped. The RAG index itself is not changed.
- Output contains structured content (JSON matching a declared output schema) plus a short text rendering.

## 4. Security and data handling (threat model)

| Risk | Control |
|---|---|
| Indirect prompt injection via text inside papers | Passage text is returned only inside a delimited "untrusted document content" block with source markers; the server never interprets it; agent system prompt forbids following instructions found in tool output; tests put injection strings inside mocked passages |
| Cost abuse (paid LLM behind `/query`) | Server-side budget for `ask_papers`: max 6 calls/min and 100 calls/day per server process (below the backend's 10/min per client and 500/day global caps); `search_papers` and `list_papers` have no LLM cost but are capped at 30 calls/min; a backend 429 becomes a clear "budget exhausted" tool error and is never retried |
| Retry storms / hanging calls | Timeouts: 10 s for `/documents` and `/search`, 60 s for `/query`; at most 2 retries with backoff, only on connection errors and 502/503/504, and only for the free endpoints; `ask_papers` is never retried automatically |
| Leaking backend internals | The backend's 502 response embeds the raw exception text (`detail: "Failed to generate an answer: {e}"`). The server maps every backend error to a fixed message and never forwards `detail` to the model or user; the original goes only to the local debug log without request text |
| SSRF / arbitrary calls | Backend base URL fixed in config and validated at start-up (http/https, host allow-list, default 127.0.0.1); tools accept only declared parameters, never a URL, path or filename filter |
| Oversized or malformed input | Pydantic validation (length, range, type, whitespace-only rejected) before any backend call; limits not wider than the backend's (500 chars) |
| Unauthorised remote use (M3) | HTTP transport binds to 127.0.0.1 by default, requires a bearer token from env, compared in constant time; token never logged; static token is a documented simplification, not the MCP OAuth flow |
| Data leakage in logs | Server audit log (stderr/JSON; stdout is reserved for the stdio protocol): tool name, HMAC-SHA-256 of arguments with a random per-process key (correlatable within a run, not reversible by hashing guesses), duration, status, backend status code. No question text, no passage text, no tokens or keys. Note (disclosed in README): the RAG backend itself logs question text; this repo does not change that |
| Secrets | Only env variables; `.env` git-ignored; `.env.example` without values; tests need no key; secret scanning and a repo-hygiene test (no `.env`, keys or data files tracked) in CI |
| Excess capability | No file system, shell, network-fetch or write tools; no tool can reach any host except the configured backend |
| Third-party papers | The server does not ship or cache paper text; it only relays passages returned by the backend at request time |

## 5. Evaluation (tool choice) and anti-leakage rules

- **Question:** given a user question, does the agent choose the right tool (or none) and refuse to follow injected instructions?
- **Set:** about 40 hand-written questions in 4 categories: needs passages (`search_papers`), needs an answer (`ask_papers`), corpus inventory (`list_papers`), out of corpus / injection attempt (no answer from outside the corpus, no compliance with injected text). Labels written before any run.
- **Split before any tuning:** 15 dev questions (used to iterate tool descriptions and the agent prompt) and 25 held-out test questions, frozen in the repo with a checksum before the first test run. Held-out results are run once per released version; tool descriptions are not edited after looking at held-out failures without a version bump and a note in the results.
- **Runs:** real model (`gpt-4o-mini`, temperature 0), 3 repetitions; report counts, accuracy and a Wilson 95% interval per category and overall. With 25 questions the interval is wide (about +/-15-20 pp); this is stated, not hidden.
- **Backend for evaluation:** local Docker API; `ask_papers` calls are counted against the daily budget (about 25 x 3 = 75 paid calls at most).
- Not measured: answer quality (belongs to the RAG project's RAGAs evaluation).

## 6. Testing (written the recruiter way: verified / not covered)

**Verified in CI (no API key, no network, no Docker)**
- Unit: input validation, output formatting, deduplication, error mapping, rate budget, config validation (rejects non-allow-listed hosts).
- Contract: in-memory MCP client lists tools, resource and prompt with the expected schemas and annotations, and calls each tool against a mocked backend whose responses are recorded fixtures of the real API shapes (section 10).
- Security: oversized and whitespace input rejected before any backend call; raw backend error text never appears in tool output; injection text stays inside the delimited block; no tool parameter can change the backend URL; logs contain no question or passage text (checked by capturing log output); HTTP without or with a wrong token is rejected (M3).
- Agent: a scripted fake chat model drives the LangGraph agent through tool calls; checks that citations from tool output reach the final answer and that the loop stops at the step limit.
- Mutation check: re-introducing known bugs (forwarding backend `detail`, logging the question, dropping the length check) must fail the suite.
- Repo hygiene: no secrets or data files tracked.

**Measured separately (manual, real model):** tool-choice evaluation in section 5.

**Not covered (stated openly):** answer quality; the real RAG backend in CI (mocked); load and concurrency; rate limiting across several server instances (in-memory, per process); Claude Desktop behaviour beyond a manual smoke test.

## 7. Deliverables

1. Repo: server, agent, tests, `docker-compose.backend.yml`, Dockerfile (M3), GitHub Actions (ruff, pytest with coverage threshold, pip-audit), Dependabot, `.env.example`.
2. README (from the common template): setup, Claude Desktop config snippet, example session, Results, Testing, Limitations.
3. Portfolio page in the common template (Why this project, Results, Methodology, Testing with "Not covered", Limitations, Tech stack, Related), cross-linked with the RAG page; CV/LinkedIn line only after the work is done.

## 8. Milestones

| # | Milestone | Estimate | Exit criteria |
|---|---|---|---|
| M1 | Spec v0.2 approved, backend contract verified | done after approval | sections 3-6 and 10 accepted, D5 decided |
| M2 | Server (stdio): 3 tools, resource, prompt; unit/contract/security tests | 3-4 h | all section 6 CI checks for stdio pass locally |
| M3 | Streamable HTTP + token, Dockerfile, CI, Dependabot | 2 h | CI green on GitHub; HTTP auth tests pass |
| M4 | LangGraph agent + fake-model tests + evaluation (section 5) | 3-4 h | held-out results with intervals committed |
| M5 | README, portfolio page, Claude Desktop demo | 1.5 h | audit against all acceptance criteria |

D5 (if accepted) adds about 1 h in the RAG repo before M2.

## 9. Decisions

- D1-D4: accepted as in section 2.
- **D5 Retrieval-only endpoint (accepted: a; implemented on branch `feat/search-endpoint` of the RAG repo, pending merge):** the RAG API has no search-without-generation endpoint.
  (a) **Recommended:** add `POST /search` to the RAG repo in a separate branch (`feat/search-endpoint`): same request validation, `top_k` 1-10, returns `{"results": [{rank, filename, text, score}]}` from the existing `Retriever.retrieve`, no LLM call, its own per-client limit (30/min); 3-4 new API tests; no change to `/query`, `/documents`, retrieval or evaluation. Merged only after review.
  (b) Drop `search_papers`; the agent gets passages only via `ask_papers` (every lookup costs an LLM call and is slower; the agent cannot inspect evidence before answering).

## 9a. Implementation notes (M2)

- MCP Python SDK 2.x (`mcp==2.3.0`): the server class is `MCPServer` (formerly `FastMCP`). Unexpected exceptions in a tool are masked by the SDK (`Error executing tool <name>`) but their traceback is logged; tools therefore raise only `ToolError` with fixed messages.
- The SDK ignores undeclared tool arguments (they never reach the tool function); the published input schemas do not set `additionalProperties: false`. A test asserts that extra `url`/`base_url` arguments cannot change the backend host.
- The SDK prefixes tool error text with `Error executing tool <name>: `.
- The HTTP client ignores proxy environment variables and does not follow redirects.

## 10. Backend contract (verified against carbon-nanotubes-rag@f9b04be, `api.py`)

| Endpoint | Request | Response | Notes |
|---|---|---|---|
| `GET /health` | none | `{"status": "ok"}` | liveness only, does not check the index |
| `GET /documents` | none | `{"count": int, "documents": [filename, ...]}` | 24 filenames, sorted; no titles or ids |
| `POST /query` | `{"question": str 1-500, "top_k": int 1-20 = 5}` | `{"answer": str, "sources": [{"rank": int, "filename": str, "text": str, "score": float}]}` | always calls the LLM; hybrid FAISS + BM25 (alpha 0.5); 422 on invalid input; 429 on rate (10/min per client, client = first `X-Forwarded-For` entry or IP) or daily cap (500, global, in memory, resets on restart); 502 on any pipeline error with raw exception text in `detail` |
| `POST /search` (D5, branch `feat/search-endpoint`) | `{"query": str 1-500, "top_k": int 1-10 = 5}` | `{"results": [{"rank", "filename", "text", "score"}]}` | no LLM call; separate limiter (30/min per client, 5000/day); query text not logged; any failure -> 500 `{"detail": "Search failed."}` |

Other facts that affect this project:
- No authentication on the API; scores are hybrid scores in [0, 1], not calibrated probabilities.
- Index: 1974 chunks from 24 PDF filenames, of which 2 are the same paper (23 distinct works).
- The backend logs question text in its own request log.
