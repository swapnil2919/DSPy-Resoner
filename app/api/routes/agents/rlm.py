"""
DSPy RLM (Recursive Language Model) — OpenAI-compatible /v1 endpoint.

POST /v1/chat/completions  — run RLM and return an answer
GET  /v1/models            — list available models (benchmark harness probe)

The full conversation (system + history + user) is collapsed into one prompt
so the benchmark harness context is never dropped.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Optional

import dspy
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.config import (
    DEFAULT_MODEL,
    DSPY_RLM_MAX_ITERATIONS,
    DSPY_RLM_MAX_LLM_CALLS,
    DSPY_RLM_TIMEOUT_SECONDS,
    OPENROUTER_API_KEY,
)
from app.core.openrouter_client import call_openrouter, stream_openrouter
from app.logger_config import logger
from app.schemas.openai import ChatCompletionRequest

router = APIRouter()

# Simple caches — no complex classes needed.
_lm_cache: dict[str, dspy.LM] = {}
_rlm: Optional[dspy.RLM] = None


# ---------------------------------------------------------------------------
# LM + RLM setup
# ---------------------------------------------------------------------------


def _get_lm(model: str, effort: Optional[str]) -> dspy.LM:
    """Return a cached DSPy LM for the given model and reasoning effort."""
    key = f"{model}|{effort}"
    if key not in _lm_cache:
        # OpenRouter requires provider-prefixed model names to route correctly.
        # Bare OpenAI model names (gpt-*, o1/o3/o4-series) sent without the
        # "openai/" sub-prefix cause OpenRouter to fall back to the legacy text
        # completions endpoint, which rejects litellm's usage-tracking field.
        if model.startswith("openrouter/"):
            lm_model = model
        elif "/" not in model and (
            model.startswith("gpt-")
            or model.startswith("o1")
            or model.startswith("o3")
            or model.startswith("o4")
        ):
            lm_model = f"openrouter/openai/{model}"
        else:
            lm_model = f"openrouter/{model}"

        kwargs: dict = {
            "model": lm_model,
            "api_key": OPENROUTER_API_KEY,
            "api_base": "https://openrouter.ai/api/v1",
            "model_type": "chat",
            "temperature": 0.1,
            "max_tokens": 4096,
            "cache": False,
        }
        if effort:
            kwargs["extra_body"] = {"reasoning": {"effort": effort}}
        _lm_cache[key] = dspy.LM(**kwargs)
        logger.info(f"DSPy LM created: {model} (effort={effort or 'off'}, temp=0.1)")
    return _lm_cache[key]


def _get_rlm() -> dspy.RLM:
    """Return the shared RLM module, creating it on first call."""
    global _rlm
    if _rlm is None:
        _rlm = dspy.RLM(
            signature="question -> answer",
            max_iterations=DSPY_RLM_MAX_ITERATIONS,
            max_llm_calls=DSPY_RLM_MAX_LLM_CALLS,
            verbose=False,
        )
        logger.info("DSPy RLM ready")
    return _rlm


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def _build_prompt(req: ChatCompletionRequest) -> str:
    """Collapse all messages (system + history + user) into one RLM prompt."""
    labels = {"system": "System", "user": "User", "assistant": "Assistant"}
    parts: list[str] = []

    for msg in req.messages:
        if isinstance(msg.content, list):
            text = "\n".join(
                b.text for b in msg.content
                if getattr(b, "type", None) == "text" and b.text
            ).strip()
        else:
            text = (msg.content or "").strip()

        if text:
            parts.append(f"{labels.get(msg.role, msg.role.capitalize())}: {text}")

    if not parts:
        raise HTTPException(status_code=400, detail="`messages` must have non-empty content.")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# RLM execution (sync, runs in a thread)
# ---------------------------------------------------------------------------


def _run(rlm: dspy.RLM, lm: dspy.LM, question: str) -> str:
    """Run RLM with retry logic for JSON parsing errors."""
    max_retries = 3   # 3 attempts total — weak JSON-followers often miss fields once or twice
    last_error = None

    for attempt in range(max_retries):
        try:
            with dspy.context(lm=lm):
                result = rlm(question=question)
                return getattr(result, "answer", str(result))
        except Exception as e:
            last_error = e
            error_msg = str(e)
            error_type = type(e).__name__
            
            # Check if it's a JSON parsing error
            if "AdapterParseError" in error_msg or "cannot be serialized to a JSON" in error_msg or "json" in error_msg.lower():
                logger.warning(f"RLM attempt {attempt + 1}/{max_retries + 1} failed: {error_type}")
                
                # On last attempt, return a fallback message
                if attempt == max_retries - 1:
                    logger.error(f"RLM failed after {max_retries} attempts. Returning fallback.")
                    return "I apologize, but I encountered a technical issue processing your request. Please try again."
                
                # Wait before retry
                time.sleep(0.5)
            else:
                # Non-parsing error, raise immediately
                logger.error(f"RLM non-parsing error: {error_type}")
                raise
    
    # Fallback
    if last_error:
        raise last_error
    return "An unexpected error occurred."


# ---------------------------------------------------------------------------
# Transparent proxy — used when the caller provides external tool schemas
# ---------------------------------------------------------------------------


def _normalize_for_openrouter(model: str) -> str:
    """Strip routing prefixes that OpenRouter's REST API doesn't accept."""
    if model.startswith("openrouter/"):
        model = model[len("openrouter/"):]
    if model.startswith("openai/") and model.count("/") >= 2:
        model = model[len("openai/"):]
    return model


def _proxy_payload(req: ChatCompletionRequest) -> dict:
    """Build a payload to forward as-is to OpenRouter (transparent-proxy mode)."""
    model = _normalize_for_openrouter(req.model or DEFAULT_MODEL)
    payload: dict = {
        "model": model,
        "messages": [m.model_dump(exclude_none=True) for m in req.messages],
    }
    if req.tools:
        payload["tools"] = [t.model_dump() for t in req.tools]
        parallel = getattr(req, "parallel_tool_calls", None)
        payload["parallel_tool_calls"] = True if parallel is None else parallel
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


async def _proxy_stream(payload: dict):
    """Stream OpenRouter SSE events through unchanged."""
    async for data in stream_openrouter(payload):
        if data == "[DONE]":
            yield "data: [DONE]\n\n"
        else:
            yield f"data: {data}\n\n"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("", response_model=None)
@router.post("/chat/completions", response_model=None)
async def chat_completions(req: ChatCompletionRequest):
    """Run DSPy RLM and return an OpenAI-compatible response."""

    # Proxy mode — when the caller supplies external tool schemas (benchmarks
    # like tau2-airline, BFCL, LiveMCPBench), forward to OpenRouter unchanged
    # so the harness gets native tool_calls back. Falls through to RLM if no
    # tools are sent.
    if req.tools:
        tool_names = [t.function.name for t in req.tools]
        logger.info(f"RLM: external tools {tool_names} — proxy mode")
        payload = _proxy_payload(req)
        if req.stream:
            return StreamingResponse(
                _proxy_stream(payload),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        try:
            response = await call_openrouter(payload)
            return JSONResponse(content=response)
        except Exception as exc:
            logger.error(f"Proxy error: {exc}", exc_info=True)
            raise HTTPException(status_code=502, detail=f"Proxy error: {exc}")

    question = _build_prompt(req)
    model = req.model or DEFAULT_MODEL
    lm = _get_lm(model, req.reasoning_effort)
    rlm = _get_rlm()

    logger.info(f"RLM request | model={model} q={question[:80]!r}")

    answer: str = await asyncio.wait_for(
        asyncio.to_thread(_run, rlm, lm, question),
        timeout=DSPY_RLM_TIMEOUT_SECONDS,
    )

    if req.stream:
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        async def _stream():
            for delta in ({"role": "assistant"}, {"content": answer}, {}):
                finish = "stop" if not delta else None
                yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            _stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })


@router.get("/models")
async def list_models():
    """Benchmark harness probes this to discover the available model."""
    return {
        "object": "list",
        "data": [{"id": DEFAULT_MODEL, "object": "model", "created": int(time.time()), "owned_by": "openrouter"}],
    }
