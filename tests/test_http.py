"""Streamable HTTP transport: a real uvicorn server on a free loopback port, mocked RAG backend."""

from __future__ import annotations

import logging
import socket
import threading
import time
from collections.abc import Iterator

import httpx
import httpx2
import pytest
import uvicorn
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

from rag_mcp.config import ConfigError
from rag_mcp.http_transport import HttpSettings, build_http_app
from rag_mcp.server import build_server

TOKEN = "t" * 40
INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "0"},
    },
}
MCP_HEADERS = {"accept": "application/json, text/event-stream", "content-type": "application/json"}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def http_server(settings, backend) -> Iterator[str]:
    port = _free_port()
    app = build_http_app(build_server(settings, backend), HttpSettings(token=TOKEN, port=port))
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("server did not start")
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/mcp"
    server.should_exit = True
    thread.join(timeout=10)


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"authorization": "Bearer wrong"},
        {"authorization": "Bearer " + "x" * 40},  # same length, wrong value
        {"authorization": "bearer " + TOKEN},  # scheme is case-sensitive here
        {"authorization": TOKEN},
        {"x-api-token": TOKEN},
    ],
)
def test_requests_without_the_exact_token_are_rejected(http_server, fake, headers):
    r = httpx.post(http_server, json=INIT, headers={**MCP_HEADERS, **headers}, timeout=10)
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "Bearer"
    assert TOKEN not in r.text
    assert fake.requests == []


def test_forged_host_header_is_rejected_even_with_token(http_server):
    headers = {**MCP_HEADERS, "authorization": f"Bearer {TOKEN}", "host": "evil.example.com"}
    r = httpx.post(http_server, json=INIT, headers=headers, timeout=10)
    assert r.status_code in (400, 403, 421)


def test_foreign_origin_is_rejected_even_with_token(http_server):
    headers = {**MCP_HEADERS, "authorization": f"Bearer {TOKEN}", "origin": "https://evil.example.com"}
    r = httpx.post(http_server, json=INIT, headers=headers, timeout=10)
    assert r.status_code in (400, 403)


@pytest.mark.anyio
async def test_authorised_client_runs_a_full_session(http_server, fake, caplog):
    with caplog.at_level(logging.DEBUG):
        hc = httpx2.AsyncClient(headers={"authorization": f"Bearer {TOKEN}"})
        async with hc, Client(streamable_http_client(http_server, http_client=hc)) as c:
            names = {t.name for t in (await c.list_tools()).tools}
            r = await c.call_tool("search_papers", {"query": "D band"})
    assert names == {"search_papers", "ask_papers", "list_papers"}
    assert r.is_error is False
    assert fake.calls("/search") == 1
    assert TOKEN not in caplog.text


def test_short_token_is_refused_at_start_up():
    with pytest.raises(ConfigError):
        HttpSettings(token="short")
    with pytest.raises(ConfigError):
        HttpSettings.from_env({})


def test_settings_from_env_and_repr_never_shows_token():
    s = HttpSettings.from_env(
        {
            "RAG_MCP_HTTP_TOKEN": TOKEN,
            "RAG_MCP_HTTP_HOST": "0.0.0.0",
            "RAG_MCP_HTTP_PORT": "9000",
            "RAG_MCP_HTTP_ALLOWED_HOSTS": "127.0.0.1:*",
        }
    )
    assert (s.host, s.port, s.allowed_hosts) == ("0.0.0.0", 9000, ("127.0.0.1:*",))
    assert TOKEN not in repr(s)
    assert TOKEN not in str(s)


@pytest.mark.parametrize(
    "env",
    [
        {"RAG_MCP_HTTP_TOKEN": TOKEN, "RAG_MCP_HTTP_PORT": "http"},
        {"RAG_MCP_HTTP_TOKEN": TOKEN, "RAG_MCP_HTTP_PORT": "70000"},
    ],
)
def test_bad_port_is_refused(env):
    with pytest.raises(ConfigError):
        HttpSettings.from_env(env)


def test_empty_host_allow_list_is_refused():
    with pytest.raises(ConfigError):
        HttpSettings(token=TOKEN, allowed_hosts=())
