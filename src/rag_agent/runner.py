"""Running one question and collecting a trace; launching the MCP server without leaking secrets."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from mcp import StdioServerParameters

from rag_agent.graph import DEFAULT_MAX_TOOL_CALLS

# Variables the server process may inherit. Everything else, including OPENAI_API_KEY, is dropped.
_SYSTEM_VARS = ("PATH", "SYSTEMROOT", "TEMP", "TMP", "HOME", "USERPROFILE", "LANG", "PYTHONPATH")


def server_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    env = os.environ if env is None else env
    return {k: v for k, v in env.items() if k in _SYSTEM_VARS or k.startswith("RAG_MCP_")}


def stdio_server(env: Mapping[str, str] | None = None) -> StdioServerParameters:
    return StdioServerParameters(command=sys.executable, args=["-m", "rag_mcp"], env=server_env(env))


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]


@dataclass
class AgentRun:
    answer: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    documents: set[str] = field(default_factory=set)
    tool_errors: int = 0

    @property
    def tools_used(self) -> list[str]:
        return [c.name for c in self.tool_calls]


def _documents(artifact: Any) -> set[str]:
    if not isinstance(artifact, dict):
        return set()
    docs = set(artifact.get("documents", []))
    for key in ("passages", "sources"):
        docs.update(p.get("document", "") for p in artifact.get(key, []) if isinstance(p, dict))
    docs.discard("")
    return docs


async def run_question(graph: Any, question: str, max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS) -> AgentRun:
    state = await graph.ainvoke(
        {"messages": [HumanMessage(question)], "tool_calls_made": 0},
        config={"recursion_limit": 2 * max_tool_calls + 6},
    )
    run = AgentRun(answer="")
    for msg in state["messages"]:
        if isinstance(msg, AIMessage):
            run.tool_calls.extend(ToolCall(c["name"], dict(c.get("args", {}))) for c in msg.tool_calls)
        elif isinstance(msg, ToolMessage):
            run.documents |= _documents(msg.artifact)
            run.tool_errors += msg.status == "error"
    final = state["messages"][-1]
    run.answer = final.content if isinstance(final.content, str) else str(final.content)
    return run
