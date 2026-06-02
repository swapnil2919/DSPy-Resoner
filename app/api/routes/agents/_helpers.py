"""
Shared helpers for DSPy agent routes (ReAct and RLM).

Both endpoints accept the same OpenAI-compatible request shape and return
the same response shape. The helpers own:

  * input validation (last user message extraction, with a 400 on bad input)
  * non-streaming response assembly
  * "stream a fully-computed answer" SSE generator
  * error → HTTPException / SSE-error mapping

Route modules stay thin — they pick the service and pass it in.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any, Awaitable, Callable, Iterable, Optional

from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.config import DSPY_STREAM_CHUNK_WORDS
from app.logger_config import logger
from app.schemas.openai import ChatCompletionRequest
from app.utils.sse import format_chunk, format_done

# Type alias for the service's ``.complete()`` method.
CompleteFunc = Callable[..., Awaitable[str]]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _content_to_text(content: Any) -> str:
    """Flatten ChatMessage.content (str or list[ContentBlock]) into plain text.

    Accepts both pre-validated Pydantic blocks and raw dicts so this works
    whether the caller passes the model instance or a dumped form.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type")
                btext = block.get("text")
            else:
                btype = getattr(block, "type", None)
                btext = getattr(block, "text", None)
            if btype == "text" and btext:
                parts.append(btext)
        return "\n".join(parts)
    return ""


def extract_last_user_question(req: ChatCompletionRequest) -> str:
    """Return the last user message as a plain string, or 400.

    Validates at the route boundary so the service never sees malformed input.
    """
    messages = req.messages or []
    if not messages:
        raise HTTPException(
            status_code=400,
            detail="`messages` must contain at least one entry.",
        )

    for msg in reversed(messages):
        if msg.role != "user":
            continue
        text = _content_to_text(msg.content).strip()
        if text:
            return text

    raise HTTPException(
        status_code=400,
        detail="`messages` must contain a user message with non-empty content.",
    )


# ---------------------------------------------------------------------------
# Response assembly
# ---------------------------------------------------------------------------


def _new_id(label: str) -> str:
    """OpenAI-style chat completion id."""
    return f"chatcmpl-{label}-{uuid.uuid4().hex[:16]}"


def _completion_payload(answer: str, model: str, label: str) -> dict[str, Any]:
    return {
        "id": _new_id(label),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def _iter_chunks(answer: str, words_per_chunk: int) -> Iterable[str]:
    """Yield content deltas, optionally batching multiple words per chunk.

    ``words_per_chunk == 0`` ⇒ emit the whole answer in a single delta.
    """
    if not answer:
        return
    if words_per_chunk <= 0:
        yield answer
        return

    words = answer.split(" ")
    n = len(words)
    if n <= 1:
        yield answer
        return

    for i in range(0, n, words_per_chunk):
        group = words[i : i + words_per_chunk]
        last_in_answer = (i + words_per_chunk) >= n
        chunk = " ".join(group)
        if not last_in_answer:
            chunk += " "
        yield chunk


async def _stream_answer(
    answer: str,
    model: str,
    label: str,
    words_per_chunk: int,
) -> AsyncGenerator[str, None]:
    chunk_id = _new_id(f"{label}-stream")
    created = int(time.time())

    role_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
        ],
    }
    yield format_chunk(role_chunk)

    for piece in _iter_chunks(answer, words_per_chunk):
        chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": {"content": piece}, "finish_reason": None}
            ],
        }
        yield format_chunk(chunk)

    final_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield format_chunk(final_chunk)
    yield format_done()


async def _stream_error(
    message: str, model: str, label: str
) -> AsyncGenerator[str, None]:
    """Surface a service error inside an SSE stream (after headers are sent)."""
    chunk = {
        "id": _new_id(f"{label}-error"),
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "error": {"message": message, "type": "server_error"},
        "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
    }
    yield format_chunk(chunk)
    yield format_done()


# ---------------------------------------------------------------------------
# Public entrypoints — used by the agent route modules
# ---------------------------------------------------------------------------


async def execute(
    *,
    complete_fn: CompleteFunc,
    question: str,
    model: str,
    reasoning_effort: Optional[str],
    timeout: float,
    label: str,
) -> str:
    """Run a DSPy service and convert exceptions into FastAPI HTTPExceptions."""
    try:
        return await complete_fn(
            question,
            model,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.error(f"{label} timed out after {timeout}s")
        raise HTTPException(
            status_code=504,
            detail=f"{label} run exceeded {timeout:.0f}s timeout",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"{label} completion failed", exc_info=True)
        raise HTTPException(status_code=502, detail=str(e))


async def stream_with_error_handling(
    *,
    complete_fn: CompleteFunc,
    question: str,
    model: str,
    reasoning_effort: Optional[str],
    timeout: float,
    label: str,
) -> AsyncGenerator[str, None]:
    """Stream wrapper that maps service errors into in-stream error events.

    By the time we yield to FastAPI, response headers are already flushed,
    so we can't raise HTTPException — surface the error as an SSE chunk.
    """
    try:
        answer = await complete_fn(
            question,
            model,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.error(f"{label} stream timed out after {timeout}s")
        async for ev in _stream_error(
            f"timeout after {timeout:.0f}s", model, label
        ):
            yield ev
        return
    except Exception as e:
        logger.error(f"{label} stream failed", exc_info=True)
        async for ev in _stream_error(str(e), model, label):
            yield ev
        return

    async for ev in _stream_answer(answer, model, label, DSPY_STREAM_CHUNK_WORDS):
        yield ev


async def handle_chat_completion(
    req: ChatCompletionRequest,
    *,
    complete_fn: CompleteFunc,
    timeout: float,
    label: str,
    reasoning_effort: Optional[str],
    extra_headers: Optional[dict[str, str]] = None,
) -> JSONResponse | StreamingResponse:
    """Drive a single chat-completion request end-to-end.

    ``complete_fn`` is the service's ``.complete()`` method — each route
    passes its own service. ``label`` is baked into log lines and response
    IDs (e.g. "react", "rlm").
    """
    from app.core.config import DEFAULT_MODEL

    question = extract_last_user_question(req)
    model = req.model or DEFAULT_MODEL

    logger.info(
        f"{label} request | model={model} stream={req.stream} "
        f"reasoning_effort={reasoning_effort or 'off'} q={question[:80]!r}"
    )

    if req.stream:
        stream_headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        if extra_headers:
            stream_headers.update(extra_headers)
        return StreamingResponse(
            stream_with_error_handling(
                complete_fn=complete_fn,
                question=question,
                model=model,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
                label=label,
            ),
            media_type="text/event-stream",
            headers=stream_headers,
        )

    answer = await execute(
        complete_fn=complete_fn,
        question=question,
        model=model,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
        label=label,
    )
    return JSONResponse(
        content=_completion_payload(answer, model, label),
        headers=extra_headers or None,
    )
