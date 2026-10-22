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


def _base_env() -> dict[str, str]:
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": os.pathsep.join(sys.path)}
    if "SYSTEMROOT" in os.environ:  # needed by Python subprocesses on Windows
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    return env


async def test_stdio_server_lists_tools_and_degrades_cleanly_without_backend():
    env = {**_base_env(), "RAG_MCP_BACKEND_URL": f"http://127.0.0.1:{_closed_port()}"}
    params = StdioServerParameters(command=sys.executable, args=["-m", "rag_mcp"], env=env)
    async with Client(params, read_timeout_seconds=30) as c:
        names = {t.name for t in (await c.list_tools()).tools}
        r = await c.call_tool("list_papers", {})
    assert names == {"search_papers", "ask_papers", "list_papers"}
    assert r.is_error is True
    assert r.content[0].text.endswith("The literature backend is unreachable or timed out.")


@pytest.mark.parametrize(
    ("args", "extra_env"),
    [
        (["--transport", "http"], {}),
        (["--transport", "http"], {"RAG_MCP_HTTP_TOKEN": "too-short"}),
        ([], {"RAG_MCP_BACKEND_URL": "http://evil.example.com"}),
    ],
)
def test_cli_refuses_to_start_with_unsafe_configuration(args, extra_env):
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-m", "rag_mcp", *args],
        env={**_base_env(), **extra_env},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode != 0
    assert "configuration error" in proc.stderr
    assert "Traceback" not in proc.stderr
    assert proc.stdout == ""


def test_cli_http_server_starts_and_requires_the_token():
    import subprocess
    import time

    import httpx

    token = "k" * 40
    port = _closed_port()
    env = {**_base_env(), "RAG_MCP_HTTP_TOKEN": token, "RAG_MCP_HTTP_PORT": str(port)}
    proc = subprocess.Popen(
        [sys.executable, "-m", "rag_mcp", "--transport", "http"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 30
        while True:
            try:
                r = httpx.post(f"http://127.0.0.1:{port}/mcp", json={}, timeout=2)
                break
            except httpx.TransportError:
                if time.monotonic() > deadline or proc.poll() is not None:
                    raise
                time.sleep(0.2)
        assert r.status_code == 401
    finally:
        proc.terminate()
        out, err = proc.communicate(timeout=15)
    assert token.encode() not in out + err
