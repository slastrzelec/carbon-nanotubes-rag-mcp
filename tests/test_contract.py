"""MCP contract: what a client sees when it connects (in-memory transport, mocked backend)."""

import json
import os
from pathlib import Path

import pytest
from mcp import Client

from rag_mcp.server import build_server

pytestmark = pytest.mark.anyio
SNAPSHOT = Path(__file__).parent / "snapshots" / "tools.json"


@pytest.fixture
def server(settings, backend):
    return build_server(settings, backend)


def _surface(tools) -> list[dict]:
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.input_schema,
            "output_schema": t.output_schema,
            "annotations": t.annotations.model_dump(exclude_none=True) if t.annotations else None,
        }
        for t in sorted(tools, key=lambda t: t.name)
    ]


async def test_tool_surface_matches_snapshot(server):
    """Tool names, descriptions and schemas drive agent behaviour: any change must be deliberate.
    Regenerate with UPDATE_SNAPSHOTS=1 after reviewing the diff."""
    async with Client(server) as c:
        surface = _surface((await c.list_tools()).tools)
    if os.environ.get("UPDATE_SNAPSHOTS") == "1":
        SNAPSHOT.write_text(json.dumps(surface, indent=2, ensure_ascii=False) + "\n", "utf-8")
    assert surface == json.loads(SNAPSHOT.read_text("utf-8"))


async def test_all_tools_are_read_only_and_closed_world(server):
    async with Client(server) as c:
        tools = (await c.list_tools()).tools
    assert {t.name for t in tools} == {"search_papers", "ask_papers", "list_papers"}
    for t in tools:
        assert t.annotations.read_only_hint is True
        assert t.annotations.destructive_hint is False
        assert t.annotations.open_world_hint is False
        assert t.output_schema is not None


async def test_input_limits_are_published_in_schema(server):
    async with Client(server) as c:
        tools = {t.name: t for t in (await c.list_tools()).tools}
    q = tools["search_papers"].input_schema["properties"]
    assert q["query"]["maxLength"] == 500
    assert q["top_k"]["minimum"] == 1
    assert q["top_k"]["maximum"] == 10
    assert tools["ask_papers"].input_schema["properties"]["question"]["maxLength"] == 500


async def test_search_papers_returns_structured_and_fenced_text(server):
    async with Client(server) as c:
        r = await c.call_tool("search_papers", {"query": "D band", "top_k": 3})
    assert r.is_error is False
    sc = r.structured_content
    assert [p["document"] for p in sc["passages"]] == ["paper_a.pdf", "paper_b.pdf"]
    assert sc["duplicates_removed"] == 1
    assert "data" in sc["notice"]
    assert r.content[0].text.count("<<<UNTRUSTED_CONTENT") == 2


async def test_ask_papers_passes_answer_and_citations_through(server):
    async with Client(server) as c:
        r = await c.call_tool("ask_papers", {"question": "What does the D band indicate?"})
    assert r.is_error is False
    assert r.structured_content["answer"] == "Synthetic answer citing paper_a.pdf."
    assert r.structured_content["sources"][0]["document"] == "paper_a.pdf"
    assert "generated_answer" in r.content[0].text


async def test_list_papers(server):
    async with Client(server) as c:
        r = await c.call_tool("list_papers", {})
    assert r.structured_content == {"count": 3, "documents": ["paper_a.pdf", "paper_b.pdf", "paper_c.pdf"]}


async def test_resource_and_prompt(server):
    async with Client(server) as c:
        resources = (await c.list_resources()).resources
        assert [str(r.uri) for r in resources] == ["papers://index"]
        content = (await c.read_resource("papers://index")).contents[0]
        assert json.loads(content.text)["count"] == 3
        prompts = (await c.list_prompts()).prompts
        assert [p.name for p in prompts] == ["literature_summary"]
        msg = (await c.get_prompt("literature_summary", {"topic": "radial breathing mode"})).messages[0]
        assert "radial breathing mode" in msg.content.text
        assert "search_papers" in msg.content.text


async def test_server_instructions_warn_about_untrusted_content(server):
    async with Client(server) as c:
        assert "untrusted" in (c.instructions or "")
