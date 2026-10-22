"""Explicit LangGraph agent: model node -> tool node loop with a hard tool-call budget."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from rag_agent.prompts import BUDGET_EXHAUSTED, SYSTEM_PROMPT

DEFAULT_MAX_TOOL_CALLS = 6


class AgentState(MessagesState):
    tool_calls_made: int


def build_graph(model: BaseChatModel, tools: list[BaseTool], max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS):
    if max_tool_calls < 1:
        raise ValueError("max_tool_calls must be positive")
    with_tools = model.bind_tools(tools)
    tool_node = ToolNode(tools, handle_tool_errors=True)

    async def call_model(state: AgentState) -> dict[str, Any]:
        messages = [SystemMessage(SYSTEM_PROMPT), *state["messages"]]
        if state.get("tool_calls_made", 0) >= max_tool_calls:
            # No tools bound: the model has to answer from what it already has.
            response = await model.ainvoke([*messages, SystemMessage(BUDGET_EXHAUSTED)])
        else:
            response = await with_tools.ainvoke(messages)
        return {"messages": [response]}

    async def call_tools(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        result = await tool_node.ainvoke(state)
        made = state.get("tool_calls_made", 0) + len(getattr(last, "tool_calls", []) or [])
        return {"messages": result["messages"], "tool_calls_made": made}

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else END

    graph = StateGraph(AgentState)
    graph.add_node("model", call_model)
    graph.add_node("tools", call_tools)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")
    return graph.compile()
