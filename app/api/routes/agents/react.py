"""
DSPy ReAct — OpenAI-compatible /v1 endpoint.

Mounted at ``/v1``.  Exposes:
  POST /v1/chat/completions  — run ReAct agent, stream real-time thought steps
  POST /v1                   — same (short alias)
  GET  /v1/models            — list available models (benchmark harness probe)

Root-cause fixes in this revision
──────────────────────────────────
1. _ReasoningFallbackLM / _StreamingLM wrapped dspy.BaseLM but newer DSPy
   validates strictly that the LM passed to dspy.context() must be a plain
   dspy.LM instance.  Both wrappers are removed; we use raw dspy.LM directly.

2. _proxy_payload was prepending "openrouter/" to every model name before
   sending it to OpenRouter's REST API.  OpenRouter expects provider/model
   (e.g. "minimax/minimax-m2.5"), NOT "openrouter/minimax/minimax-m2.5".
   The openrouter/ prefix is only meaningful for litellm routing (DSPy path).

3. Streaming no longer relies on a custom LM subclass.  A threading.Event +
   50ms poll loop emits one heartbeat chunk every 3 s while the ReAct thread
   runs, giving live progress without any LM wrapping.

4. Proxy forwards all relevant parameters (top_p, stop, frequency_penalty,
   presence_penalty, parallel_tool_calls) that benchmark harnesses send.

5. temperature / max_tokens from the request are forwarded to the DSPy LM
   so harness-specified generation params are respected on the ReAct path.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import threading
import time
import uuid
from typing import Any, Callable, Optional

import dspy
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.config import (
    DEFAULT_MODEL,
    DSPY_REACT_MAX_ITERS,
    DSPY_REACT_TIMEOUT_SECONDS,
    DSPY_TOOL_OUTPUT_MAX_CHARS,
    OPENROUTER_API_KEY,
)
from app.core.openrouter_client import call_openrouter, stream_openrouter
from app.logger_config import logger
from app.schemas.openai import ChatCompletionRequest
from app.tools.registry import TOOL_MAP, TOOL_SCHEMAS

router = APIRouter()

# ── Heartbeat interval for streaming progress feedback ───────────────────────
_STREAM_HEARTBEAT_SECONDS = 3.0


# ---------------------------------------------------------------------------
# Async bridge — lets DSPy call async tools from its sync thread
# ---------------------------------------------------------------------------

class _AsyncBridge:
    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock = threading.Lock()

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop and self._loop.is_running():
            return self._loop
        with self._lock:
            if self._loop and self._loop.is_running():
                return self._loop
            self._loop = asyncio.new_event_loop()
            threading.Thread(target=self._loop.run_forever, daemon=True).start()
            return self._loop

    def run(self, coro: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._ensure_loop()).result()


_bridge = _AsyncBridge()


# ---------------------------------------------------------------------------
# LM cache — one dspy.LM per (model, effort, temperature, max_tokens) tuple
# ---------------------------------------------------------------------------

_lm_cache: dict[str, dspy.LM] = {}


def _get_lm(
    model: str,
    effort: Optional[str],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> dspy.LM:
    """Return a cached raw dspy.LM — no wrapper classes that break isinstance."""
    key = f"{model}|{effort}|{temperature}|{max_tokens}"
    if key not in _lm_cache:
        # OpenRouter requires provider-prefixed model names to route correctly.
        # Bare OpenAI model names sent without "openai/" sub-prefix cause
        # OpenRouter to fall back to the legacy text completions endpoint.
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
            # Force the chat-completions endpoint. Without this, DSPy can route
            # some models to the legacy Completions.create() path which doesn't
            # accept the `usage` kwarg DSPy 3.x sends → TypeError.
            "model_type": "chat",
            # Disable DSPy's on-disk response cache so benchmark scores reflect
            # real model behaviour, not cached answers from prior runs.
            "cache": False,
        }
        if effort:
            kwargs["extra_body"] = {"reasoning": {"effort": effort}}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        _lm_cache[key] = dspy.LM(**kwargs)
        logger.info(
            f"DSPy LM created: {model} "
            f"(effort={effort or 'off'}, temp={temperature}, max_tokens={max_tokens})"
        )
    return _lm_cache[key]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def _wrap_tool(fn: Callable, name: str, desc: str) -> Callable:
    """Sync-callable wrapper for async tools; truncates oversized output."""
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = _bridge.run(fn(*args, **kwargs))
        text = str(result)
        if len(text) > DSPY_TOOL_OUTPUT_MAX_CHARS:
            text = text[:DSPY_TOOL_OUTPUT_MAX_CHARS] + "\n...[output truncated]"
            logger.warning(f"Tool '{name}' output truncated to {DSPY_TOOL_OUTPUT_MAX_CHARS} chars")
        return text
    wrapper.__name__ = name
    wrapper.__doc__ = desc
    wrapper.__signature__ = inspect.signature(fn)  # type: ignore[attr-defined]
    wrapper.__annotations__ = dict(getattr(fn, "__annotations__", {}))
    return wrapper


def _build_tools() -> list[dspy.Tool]:
    tools = []
    for schema in TOOL_SCHEMAS:
        fn_info = schema["function"]
        name = fn_info["name"]
        if name not in TOOL_MAP:
            continue
        wrapped = _wrap_tool(TOOL_MAP[name], name, fn_info["description"])
        tools.append(dspy.Tool(func=wrapped, name=name, desc=fn_info["description"]))
    logger.info(f"ReAct tools loaded: {[t.name for t in tools]}")
    return tools


# ---------------------------------------------------------------------------
# ReAct signature — fail fast when no tool matches (reduces wasted iterations)
# ---------------------------------------------------------------------------

class _Signature(dspy.Signature):
    """Answer the user's question using the available tools.

    Rules:
    - Call a tool only when it directly helps answer the request.
    - If NO suitable tool exists for the request, answer immediately without
      calling any tool. Do NOT retry or loop with unavailable tools.
    - Cover every part of a multi-part question before writing the final answer.
    """

    question: str = dspy.InputField(desc="User question, possibly with multiple sub-tasks")
    answer: str = dspy.OutputField(desc="Final answer covering ALL parts of the question")


# ---------------------------------------------------------------------------
# ReAct agent singleton
# ---------------------------------------------------------------------------

_react: Optional[dspy.ReAct] = None


def _get_react() -> dspy.ReAct:
    global _react
    if _react is None:
        _react = dspy.ReAct(_Signature, tools=_build_tools(), max_iters=DSPY_REACT_MAX_ITERS)
        logger.info(f"DSPy ReAct ready (max_iters={DSPY_REACT_MAX_ITERS})")
    return _react


# ---------------------------------------------------------------------------
# Model name normalisation for direct OpenRouter REST calls
#
# DSPy/litellm needs "openrouter/provider/model" as a routing prefix.
# OpenRouter's own REST API expects just "provider/model" — no prefix.
# ---------------------------------------------------------------------------

def _normalize_for_openrouter(model: str) -> str:
    # Strip litellm routing prefix
    if model.startswith("openrouter/"):
        model = model[len("openrouter/"):]
    # Strip spurious "openai/" routing prefix (e.g. "openai/minimax/model")
    if model.startswith("openai/") and model.count("/") >= 2:
        model = model[len("openai/"):]
    return model


# ---------------------------------------------------------------------------
# Transparent proxy — used when caller provides external tool schemas
# ---------------------------------------------------------------------------

def _proxy_payload(req: ChatCompletionRequest) -> dict:
    model = _normalize_for_openrouter(req.model or DEFAULT_MODEL)
    payload: dict = {
        "model": model,
        "messages": [m.model_dump(exclude_none=True) for m in req.messages],
    }
    if req.tools:
        payload["tools"] = [t.model_dump() for t in req.tools]
        # Enable parallel tool calls by default; helps multi-tool benchmark tasks.
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
    async for data in stream_openrouter(payload):
        if data == "[DONE]":
            yield "data: [DONE]\n\n"
        else:
            yield f"data: {data}\n\n"


# ---------------------------------------------------------------------------
# Prompt builder — collapses the full message history into one question string
# ---------------------------------------------------------------------------

def _build_prompt(req: ChatCompletionRequest) -> str:
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
# ReAct execution (sync — always called inside a background thread)
# ---------------------------------------------------------------------------

def _run(react: dspy.ReAct, lm: dspy.LM, question: str) -> str:
    # dspy.context requires a plain dspy.LM — never pass a wrapper subclass.
    with dspy.context(lm=lm):
        start = time.perf_counter()
        try:
            result = react(question=question)  # type: ignore[misc]
            logger.info(f"DSPy ReAct finished in {time.perf_counter() - start:.2f}s")
            return getattr(result, "answer", str(result))
        except Exception as exc:
            logger.error(f"DSPy ReAct failed after {time.perf_counter() - start:.2f}s: {exc}")
            raise


# ---------------------------------------------------------------------------
# SSE chunk builder
# ---------------------------------------------------------------------------

def _sse_chunk(
    chunk_id: str,
    created: int,
    model: str,
    delta: dict,
    finish: Optional[str] = None,
) -> str:
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}\n\n"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("")
@router.get("/")
async def root():
    """Health / discovery probe."""
    return JSONResponse({
        "status": "ok",
        "service": "DSPy ReAct agent",
        "endpoints": {
            "chat": "POST /v1/chat/completions",
            "models": "GET /v1/models",
        },
    })


@router.post("", response_model=None)
@router.post("/chat/completions", response_model=None)
async def chat_completions(req: ChatCompletionRequest):
    """Run DSPy ReAct and return an OpenAI-compatible response."""

    # ── Proxy mode ─────────────────────────────────────────────────────────
    # Benchmarks (Toolathlon, BFCL, LiveMCPBench, etc.) send their own tool
    # schemas and expect native tool_calls back.  Pass straight through to
    # OpenRouter so the model can return proper tool_calls.
    if req.tools:
        tool_names = [t.function.name for t in req.tools]
        logger.info(f"ReAct: external tools {tool_names} — proxy mode")
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

    # ── ReAct agent mode ───────────────────────────────────────────────────
    question = _build_prompt(req)
    model = req.model or DEFAULT_MODEL
    react = _get_react()

    # Use a plain dspy.LM — dspy.context() requires exactly dspy.LM, not a subclass.
    # reasoning_effort is ignored here; DSPy ReAct provides its own
    # Thought→Action→Observation loop that replaces model-level chain-of-thought.
    lm = _get_lm(model, effort=None, temperature=req.temperature, max_tokens=req.max_tokens)

    logger.info(f"ReAct request | model={model} q={question[:80]!r}")

    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    # ── Streaming ──────────────────────────────────────────────────────────
    # We no longer wrap the LM to intercept per-step calls (that broke dspy.context).
    # Instead a 50 ms poll loop emits one heartbeat chunk every 3 s while the
    # ReAct thread runs, giving the client live "thinking…" progress.
    if req.stream:
        done_event = threading.Event()
        result_box: list[str] = []
        error_box: list[BaseException] = []

        def _worker() -> None:
            try:
                result_box.append(_run(react, lm, question))
            except Exception as exc:
                error_box.append(exc)
            finally:
                done_event.set()

        threading.Thread(target=_worker, daemon=True).start()

        async def _sse():
            yield _sse_chunk(chunk_id, created, model, {"role": "assistant"})

            deadline = time.monotonic() + DSPY_REACT_TIMEOUT_SECONDS
            next_heartbeat = time.monotonic() + _STREAM_HEARTBEAT_SECONDS
            step = 0

            while not done_event.is_set():
                now = time.monotonic()
                if now > deadline:
                    yield _sse_chunk(
                        chunk_id, created, model,
                        {"content": "\n[timed out]"}, finish="stop",
                    )
                    yield "data: [DONE]\n\n"
                    return
                if now >= next_heartbeat:
                    step += 1
                    yield _sse_chunk(
                        chunk_id, created, model,
                        {"content": f"_(thinking… step {step})_ "},
                    )
                    next_heartbeat = now + _STREAM_HEARTBEAT_SECONDS
                await asyncio.sleep(0.05)

            if error_box:
                err = error_box[0]
                logger.error(f"ReAct streaming error: {err}")
                yield _sse_chunk(
                    chunk_id, created, model,
                    {"content": f"\n[Agent error: {err}]"}, finish="stop",
                )
                yield "data: [DONE]\n\n"
                return

            answer = result_box[0] if result_box else ""
            yield _sse_chunk(chunk_id, created, model, {"content": "\n" + answer})
            yield _sse_chunk(chunk_id, created, model, {}, finish="stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            _sse(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── Non-streaming ──────────────────────────────────────────────────────
    try:
        answer: str = await asyncio.wait_for(
            asyncio.to_thread(_run, react, lm, question),
            timeout=DSPY_REACT_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="ReAct agent timed out.")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Agent error: {exc}")

    return JSONResponse({
        "id": chunk_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })


@router.get("/models")
async def list_models():
    """Benchmark harness probes this to discover the available model."""
    return {
        "object": "list",
        "data": [
            {
                "id": DEFAULT_MODEL,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "openrouter",
            }
        ],
    }
