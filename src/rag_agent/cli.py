"""rag-agent: ask one question, or run the tool-choice evaluation.

  rag-agent ask "How is the RBM frequency related to diameter?"
  rag-agent eval --split dev
  rag-agent eval --split test        # held-out; refuses to overwrite an existing result

The OpenAI key is read from the environment or a .env file and stays in this process: the MCP
server is launched with an environment allow-list that excludes it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import anyio
from mcp import Client

from rag_agent.evaluation import evaluate, load_items, split_checksum, summarise
from rag_agent.graph import DEFAULT_MAX_TOOL_CALLS, build_graph
from rag_agent.mcp_tools import load_mcp_tools
from rag_agent.runner import run_question, stdio_server
from rag_mcp import __version__

MODEL = "gpt-4o-mini"
EVAL_DIR = Path("eval")


def make_model():  # pragma: no cover - needs a real API key
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=MODEL, temperature=0, timeout=60, max_retries=2)


@asynccontextmanager
async def agent_session(model, server=None):
    async with Client(server or stdio_server(), read_timeout_seconds=120) as client:
        tools = await load_mcp_tools(client)
        yield build_graph(model, tools)


def _require_key() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # pragma: no cover
        pass
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("rag-agent: OPENAI_API_KEY is not set (environment or .env)")


async def _ask(question: str, model) -> int:
    async with agent_session(model) as graph:
        run = await run_question(graph, question)
    print(run.answer)
    print(f"\n[tools: {', '.join(run.tools_used) or 'none'}]", file=sys.stderr)
    return 0


def result_path(split: str, version: str = __version__, root: Path = EVAL_DIR) -> Path:
    return root / "results" / f"{split}-{version}.json"


async def _eval(split: str, reps: int, pace_s: float, model, root: Path = EVAL_DIR, server=None) -> int:
    items = load_items(root / "questions.jsonl")
    frozen = (root / "test_split.sha256").read_text("utf-8").strip()
    if split_checksum(items, "test") != frozen:
        sys.exit("rag-agent: the frozen test split was modified; refusing to run")
    out = result_path(split, root=root)
    if split == "test" and out.exists():
        sys.exit(f"rag-agent: {out} exists; held-out results are not overwritten (bump the version)")
    chosen = [i for i in items if i.split == split]

    async def pause() -> None:
        await anyio.sleep(pace_s)

    async with agent_session(model, server) as graph:
        results = await evaluate(chosen, lambda q: run_question(graph, q), reps=reps, pause=pause)
    report = {
        "split": split,
        "version": __version__,
        "model": MODEL,
        "temperature": 0,
        "reps": reps,
        "max_tool_calls": DEFAULT_MAX_TOOL_CALLS,
        "test_split_sha256": frozen,
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "summary": summarise(results),
        "items": [
            {"id": r.id, "category": r.category, "outcomes": r.outcomes, "passed": r.passed, "tools": r.tools}
            for r in results
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", "utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"written {out}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None, model_factory=None) -> int:
    parser = argparse.ArgumentParser(prog="rag-agent")
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ev = sub.add_parser("eval")
    ev.add_argument("--split", choices=["dev", "test"], required=True)
    ev.add_argument("--reps", type=int, default=3)
    ev.add_argument("--pace", type=float, default=7.0, help="seconds between calls (stay under budgets)")
    args = parser.parse_args(argv)

    if model_factory is None:  # pragma: no cover - real model path
        _require_key()
        model_factory = make_model
    model = model_factory()
    if args.command == "ask":
        return anyio.run(_ask, args.question, model)
    return anyio.run(_eval, args.split, args.reps, args.pace, model)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
