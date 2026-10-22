"""HTTP client for the RAG REST API.

Error policy: every failure becomes a BackendError subclass with a fixed public message.
The backend's own error text (its 502 `detail` embeds raw exception text) is never put into
a public message; it is kept only in `debug_detail` for local debugging and is not logged.

Retry policy: only the free, idempotent endpoints (/documents, /search) are retried, only on
connection errors and 502/503/504, at most `max_retries` times. /query (paid LLM call) is never
retried automatically.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import anyio
import httpx
from pydantic import BaseModel, ValidationError

from rag_mcp.config import Settings

RETRYABLE_STATUS = frozenset({502, 503, 504})


class RawPassage(BaseModel):
    rank: int
    filename: str
    text: str
    score: float


class DocumentsResponse(BaseModel):
    count: int
    documents: list[str]


class SearchResponse(BaseModel):
    results: list[RawPassage]


class QueryResponse(BaseModel):
    answer: str
    sources: list[RawPassage]


class BackendError(Exception):
    public_message = "The literature backend failed to handle the request."

    def __init__(self, status_code: int | None = None, debug_detail: str = ""):
        super().__init__(self.public_message)
        self.status_code = status_code
        self.debug_detail = debug_detail


class BackendUnavailable(BackendError):
    public_message = "The literature backend is unreachable or timed out."


class BackendBusy(BackendError):
    public_message = (
        "The literature backend refused the request because its rate limit or daily budget "
        "is exhausted. Do not retry now."
    )


class BackendRejected(BackendError):
    public_message = "The literature backend rejected the request as invalid."


class BackendFailure(BackendError):
    public_message = "The literature backend failed to handle the request."


class BackendProtocolError(BackendError):
    public_message = "The literature backend returned an unexpected response."


class RagBackend:
    def __init__(
        self,
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[Any]] = anyio.sleep,
    ):
        self._settings = settings
        self._sleep = sleep
        # trust_env=False: ignore proxy variables from the environment; follow_redirects=False:
        # a redirect must not move a request to another host.
        self._client = httpx.AsyncClient(
            base_url=settings.backend_url,
            transport=transport,
            trust_env=False,
            follow_redirects=False,
            headers={"user-agent": "carbon-nanotubes-rag-mcp"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def list_documents(self) -> DocumentsResponse:
        data = await self._request("GET", "/documents", None, self._settings.free_timeout_s, retry=True)
        return _parse(DocumentsResponse, data)

    async def search(self, query: str, top_k: int) -> SearchResponse:
        body = {"query": query, "top_k": top_k}
        data = await self._request("POST", "/search", body, self._settings.free_timeout_s, retry=True)
        return _parse(SearchResponse, data)

    async def ask(self, question: str) -> QueryResponse:
        body = {"question": question, "top_k": self._settings.ask_top_k}
        data = await self._request("POST", "/query", body, self._settings.ask_timeout_s, retry=False)
        return _parse(QueryResponse, data)

    async def _request(
        self, method: str, path: str, body: dict[str, Any] | None, timeout: float, *, retry: bool
    ) -> Any:
        attempts = 1 + (self._settings.max_retries if retry else 0)
        last_error: BackendError = BackendUnavailable()
        for attempt in range(attempts):
            if attempt:
                await self._sleep(self._settings.retry_backoff_s * 2 ** (attempt - 1))
            try:
                response = await self._client.request(method, path, json=body, timeout=timeout)
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                last_error = BackendUnavailable(debug_detail=type(exc).__name__)
                continue
            except httpx.TimeoutException as exc:
                raise BackendUnavailable(debug_detail=type(exc).__name__) from None
            except httpx.HTTPError as exc:
                raise BackendUnavailable(debug_detail=type(exc).__name__) from None
            if response.status_code in RETRYABLE_STATUS:
                last_error = BackendFailure(response.status_code, _detail(response))
                continue
            return self._decode(response)
        raise last_error

    def _decode(self, response: httpx.Response) -> Any:
        status = response.status_code
        if status == 429:
            raise BackendBusy(status, _detail(response))
        if 400 <= status < 500:
            raise BackendRejected(status, _detail(response))
        if status >= 500:
            raise BackendFailure(status, _detail(response))
        if status != 200:
            raise BackendProtocolError(status)
        if len(response.content) > self._settings.max_response_bytes:
            raise BackendProtocolError(status, "response too large")
        try:
            return response.json()
        except ValueError:
            raise BackendProtocolError(status, "invalid JSON") from None


def _detail(response: httpx.Response) -> str:
    return response.text[:500]


def _parse(model: type[BaseModel], data: Any) -> Any:
    try:
        return model.model_validate(data)
    except ValidationError:
        raise BackendProtocolError(200, f"response does not match {model.__name__}") from None
