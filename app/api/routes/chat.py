"""
OpenAI-compatible /v1/chat/completions endpoint.
Supports both streaming and non-streaming modes with tool calling.
"""

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.config import DEFAULT_MODEL
from app.schemas.openai import ChatCompletionRequest
from app.services.llm_service import complete, stream_complete
from app.utils.sse import format_chunk, format_done
from app.logger_config import logger

router = APIRouter()


def _strip_spurious_openai_prefix(model: str) -> str:
    """Strip openai/ when it's a routing prefix, not the real provider.

    "openai/minimax/minimax-m2.7" → "minimax/minimax-m2.7"
    "openai/gpt-4o"               → "openai/gpt-4o"  (unchanged, real OpenAI model)
    """
    if model.startswith("openai/") and model.count("/") >= 2:
        return model[len("openai/"):]
    return model


def _build_payload(req: ChatCompletionRequest) -> dict:
    """
    Convert the incoming OpenAI-format request into the payload dict
    to send to OpenRouter (which is also OpenAI-compatible).
    """
    model = _strip_spurious_openai_prefix(req.model or DEFAULT_MODEL)
    payload = {
        "model": model,
        "messages": [m.model_dump(exclude_none=True) for m in req.messages],
    }

    if req.tools:
        payload["tools"] = [t.model_dump() for t in req.tools]
    if req.tool_choice is not None:
        payload["tool_choice"] = req.tool_choice
    if req.temperature is not None:
        payload["temperature"] = req.temperature
    if req.max_tokens is not None:
        payload["max_tokens"] = req.max_tokens
    if req.top_p is not None:
        payload["top_p"] = req.top_p
    if req.stop is not None:
        payload["stop"] = req.stop
    if req.frequency_penalty is not None:
        payload["frequency_penalty"] = req.frequency_penalty
    if req.presence_penalty is not None:
        payload["presence_penalty"] = req.presence_penalty

    return payload

@router.post("", response_model=None)  # POST /v1  — direct alias (e.g. curl, Postman)
@router.post("/chat/completions", response_model=None)  # POST /v1/chat/completions — OpenAI SDK
async def chat_completions(req: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.

    - stream=false → returns ChatCompletion JSON
    - stream=true  → returns SSE stream of ChatCompletionChunk events
    """
    logger.info(
        f"Chat completions request | model={req.model or DEFAULT_MODEL} "
        f"stream={req.stream} tools={len(req.tools) if req.tools else 0}"
    )

    payload = _build_payload(req)

    if req.stream:
        return StreamingResponse(
            _stream_response(payload),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        try:
            response = await complete(payload)
            return JSONResponse(content=response)
        except Exception as e:
            logger.error("Chat completion failed", exc_info=True)
            raise HTTPException(status_code=502, detail=str(e))


async def _stream_response(payload: dict):
    """
    Async generator that yields OpenAI-format SSE events.
    """
    try:
        async for data in stream_complete(payload):
            if data == "[DONE]":
                yield format_done()
            else:
                # data is already a JSON string from OpenRouter
                # Re-parse and re-emit to ensure consistent formatting
                try:
                    chunk = json.loads(data)
                    yield format_chunk(chunk)
                except json.JSONDecodeError:
                    # Pass through as-is if it's not valid JSON
                    yield f"data: {data}\n\n"
    except Exception as e:
        logger.error("Stream error", exc_info=True)
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "server_error",
            }
        }
        yield format_chunk(error_chunk)
        yield format_done()