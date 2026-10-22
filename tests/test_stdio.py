"""End-to-end over the real stdio transport: the server runs as a subprocess, as Claude Desktop runs it.
The backend URL points at a closed local port, so no RAG API is needed."""

import os
import socket
import sys

import pytest
from mcp import Client, StdioServerParameters

pytestmark = pytest.mark.anyio


def _closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def test_stdio_server_lists_tools_and_degrades_cleanly_without_backend():
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": os.pathsep.join(sys.path),
        "RAG_MCP_BACKEND_URL": f"http://127.0.0.1:{_closed_port()}",
    }
    params = StdioServerParameters(command=sys.executable, args=["-m", "rag_mcp"], env=env)
    async with Client(params, read_timeout_seconds=30) as c:
        names = {t.name for t in (await c.list_tools()).tools}
        r = await c.call_tool("list_papers", {})
    assert names == {"search_papers", "ask_papers", "list_papers"}
    assert r.is_error is True
    assert r.content[0].text.endswith("The literature backend is unreachable or timed out.")
