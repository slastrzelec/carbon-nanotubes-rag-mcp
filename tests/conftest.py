from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest

from rag_mcp.backend import RagBackend
from rag_mcp.config import Settings

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "backend_responses.json").read_text("utf-8"))

Handler = Callable[[httpx.Request], httpx.Response]


@dataclass
class FakeBackend:
    """Stands in for the RAG API at the HTTP level and records every request."""

    routes: dict[tuple[str, str], list[httpx.Response]] = field(default_factory=dict)
    requests: list[httpx.Request] = field(default_factory=list)
    raise_on: dict[tuple[str, str], Exception] = field(default_factory=dict)

    def respond(self, method: str, path: str, *responses: httpx.Response) -> None:
        self.routes[(method, path)] = list(responses)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = (request.method, request.url.path)
        if key in self.raise_on:
            raise self.raise_on[key]
        queue = self.routes.get(key)
        if not queue:
            return httpx.Response(404, json={"detail": "Not Found"})
        return queue.pop(0) if len(queue) > 1 else queue[0]

    def calls(self, path: str) -> int:
        return sum(1 for r in self.requests if r.url.path == path)


def ok(payload: Any) -> httpx.Response:
    return httpx.Response(200, json=payload)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def settings() -> Settings:
    return Settings(retry_backoff_s=0.0)


@pytest.fixture
def fake() -> FakeBackend:
    f = FakeBackend()
    f.respond("GET", "/documents", ok(FIXTURES["documents"]))
    f.respond("POST", "/search", ok(FIXTURES["search"]))
    f.respond("POST", "/query", ok(FIXTURES["query"]))
    return f


@pytest.fixture
def backend(settings: Settings, fake: FakeBackend) -> RagBackend:
    async def no_sleep(_: float) -> None:
        return None

    return RagBackend(settings, transport=httpx.MockTransport(fake.handler), sleep=no_sleep)
