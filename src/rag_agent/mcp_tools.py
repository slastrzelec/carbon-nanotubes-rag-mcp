"""Minimal MCP -> LangChain tool adapter (langchain-mcp-adapters does not support mcp 2.x yet).

Each MCP tool becomes a StructuredTool whose JSON schema is the server's input schema, so the
model sees exactly what the server publishes; the server remains the place where arguments are
validated. Tool errors are raised as ToolException and reach the model as error tool messages.
The structured result is kept as the tool message artifact, so callers can see which documents
were returned without parsing text.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from mcp import Client


def _text(result: Any) -> str:
    parts = [getattr(block, "text", "") for block in result.content]
    return "\n".join(p for p in parts if p)


async def load_mcp_tools(client: Client) -> list[BaseTool]:
    listed = await client.list_tools()
    return [_wrap(client, tool) for tool in listed.tools]


def _wrap(client: Client, tool: Any) -> BaseTool:
    name = tool.name

    async def call(**arguments: Any) -> tuple[str, dict[str, Any] | None]:
        result = await client.call_tool(name, arguments)
        if result.is_error:
            raise ToolException(_text(result) or f"{name} failed")
        return _text(result), result.structured_content

    return StructuredTool.from_function(
        coroutine=call,
        name=name,
        description=tool.description or "",
        args_schema=tool.input_schema,
        response_format="content_and_artifact",
        handle_tool_error=True,
    )
