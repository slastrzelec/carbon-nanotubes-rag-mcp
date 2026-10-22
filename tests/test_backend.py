import json

import httpx
import pytest

from rag_mcp.backend import (
    BackendBusy,
    BackendFailure,
    BackendProtocolError,
    BackendRejected,
    BackendUnavailable,
)
from tests.conftest import FIXTURES, ok

pytestmark = pytest.mark.anyio


async def test_search_sends_only_declared_fields_to_fixed_host(backend, fake):
    resp = await backend.search("D band", 3)
    assert [p.filename for p in resp.results] == ["paper_a.pdf", "paper_b.pdf", "paper_c.pdf"]
    req = fake.requests[-1]
    assert str(req.url) == "http://127.0.0.1:8000/search"
    assert json.loads(req.content) == {"query": "D band", "top_k": 3}


async def test_ask_uses_fixed_top_k(backend, fake):
    await backend.ask("What is the D band?")
    assert json.loads(fake.requests[-1].content) == {"question": "What is the D band?", "top_k": 5}


async def test_list_documents(backend):
    assert (await backend.list_documents()).count == 3


@pytest.mark.parametrize(
    ("status", "error"),
    [(429, BackendBusy), (422, BackendRejected), (404, BackendRejected), (500, BackendFailure)],
)
async def test_status_codes_map_to_fixed_errors(backend, fake, status, error):
    fake.respond("POST", "/search", httpx.Response(status, json={"detail": "internal secret"}))
    with pytest.raises(error) as info:
        await backend.search("q", 1)
    assert "secret" not in str(info.value)
    assert info.value.status_code == status


async def test_query_502_detail_is_kept_out_of_public_message(backend, fake):
    fake.respond("POST", "/query", httpx.Response(502, json=FIXTURES["error_502"]))
    with pytest.raises(BackendFailure) as info:
        await backend.ask("q")
    assert "secret_path" not in str(info.value)
    assert "secret_path" in info.value.debug_detail


async def test_free_endpoint_retries_on_502_then_succeeds(backend, fake):
    fake.respond("POST", "/search", httpx.Response(502), httpx.Response(503), ok(FIXTURES["search"]))
    resp = await backend.search("q", 1)
    assert len(resp.results) == 3
    assert fake.calls("/search") == 3


async def test_free_endpoint_gives_up_after_max_retries(backend, fake):
    fake.respond("GET", "/documents", httpx.Response(503))
    with pytest.raises(BackendFailure):
        await backend.list_documents()
    assert fake.calls("/documents") == 3  # 1 + max_retries


async def test_paid_endpoint_is_never_retried(backend, fake):
    fake.respond("POST", "/query", httpx.Response(503), ok(FIXTURES["query"]))
    with pytest.raises(BackendFailure):
        await backend.ask("q")
    assert fake.calls("/query") == 1


async def test_connection_error_retries_free_endpoint_only(backend, fake):
    fake.raise_on[("POST", "/search")] = httpx.ConnectError("refused")
    fake.raise_on[("POST", "/query")] = httpx.ConnectError("refused")
    with pytest.raises(BackendUnavailable):
        await backend.search("q", 1)
    with pytest.raises(BackendUnavailable):
        await backend.ask("q")
    assert fake.calls("/search") == 3
    assert fake.calls("/query") == 1


async def test_read_timeout_is_not_retried(backend, fake):
    fake.raise_on[("POST", "/search")] = httpx.ReadTimeout("slow")
    with pytest.raises(BackendUnavailable):
        await backend.search("q", 1)
    assert fake.calls("/search") == 1


async def test_other_transport_error_maps_to_unavailable(backend, fake):
    fake.raise_on[("GET", "/documents")] = httpx.RemoteProtocolError("broken")
    with pytest.raises(BackendUnavailable):
        await backend.list_documents()


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, json={"unexpected": True}),
        httpx.Response(200, content=b"not json"),
        httpx.Response(302, headers={"location": "http://evil.example.com/"}),
        httpx.Response(200, content=b"x" * 2_000_001),
    ],
)
async def test_malformed_redirect_or_oversized_responses_are_rejected(backend, fake, response):
    fake.respond("GET", "/documents", response)
    with pytest.raises(BackendProtocolError):
        await backend.list_documents()
    assert all(r.url.host == "127.0.0.1" for r in fake.requests)


async def test_proxy_environment_is_ignored(settings, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example.com:3128")
    from rag_mcp.backend import RagBackend

    b = RagBackend(settings)
    assert b._client._trust_env is False
    await b.aclose()
