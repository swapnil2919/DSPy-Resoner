"""
OpenAI-compatible request/response models for /v1/chat/completions.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Request Models ───────────────────────────────────────────────────────────


class FunctionDefinition(BaseModel):
    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolDefinition(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class FunctionCall(BaseModel):
    name: str
    arguments: str  # JSON-encoded string


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class ContentBlock(BaseModel):
    """OpenAI structured-content block.

    Accepts the multimodal/structured-content format used by OpenAI clients
    (and tools like LiveMCPBench's LLM-as-judge):
    `[{"type": "text", "text": "..."}, ...]`.
    Unknown fields are preserved so we forward them verbatim to the upstream API.
    """

    type: str
    text: str | None = None
    image_url: dict[str, Any] | None = None

    model_config = {"extra": "allow"}


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ContentBlock] | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # required when role == "tool"


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    tools: list[ToolDefinition] | None = None
    tool_choice: str | dict | None = None
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    # Allow parallel tool calls (forwarded to OpenRouter in proxy mode).
    # Defaults to True when tools are present; set False to force sequential calls.
    parallel_tool_calls: bool | None = None
    # Reasoning Language Model (RLM) controls. Forwarded by the DSPy route
    # to dspy.LM so that reasoning models (o-series, deepseek-r1, gpt-5-thinking,
    # claude-thinking via OpenRouter) can be driven with the right effort level.
    # Ignored by the plain /v1/chat/completions route.
    reasoning_effort: Literal["low", "medium", "high"] | None = None


# ── Response Models ──────────────────────────────────────────────────────────


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[Choice]
    usage: UsageInfo = Field(default_factory=UsageInfo)


# ── Streaming Chunk Models ───────────────────────────────────────────────────


class DeltaContent(BaseModel):
    role: str | None = None
    content: str | None = None
    tool_calls: list[dict] | None = None


class ChunkChoice(BaseModel):
    index: int = 0
    delta: DeltaContent
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[ChunkChoice]
