"""Scripted chat model for agent tests: no network, deterministic tool calls."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


def tool_call(name: str, args: dict[str, Any], call_id: str = "c1") -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}]
    )


class ScriptedModel(BaseChatModel):
    script: list[AIMessage] = Field(default_factory=list)
    seen: list[list[BaseMessage]] = Field(default_factory=list)
    unbound_reply: str = "Answer from gathered results."
    bound: bool = False

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedModel:
        return self.model_copy(update={"bound": True})  # shallow copy: script and seen are shared

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any):
        self.seen.append(list(messages))
        if not self.bound:
            msg = AIMessage(self.unbound_reply)
        else:
            msg = self.script.pop(0) if self.script else AIMessage("done")
        return ChatResult(generations=[ChatGeneration(message=msg)])
