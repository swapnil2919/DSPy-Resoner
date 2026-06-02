"""
HTTP client for OpenRouter API.
Supports both non-streaming and streaming (SSE) calls.

Uses a single shared httpx.AsyncClient (created at app startup, closed at
shutdown) with a tuned connection pool so the proxy can handle many
concurrent users without exhausting sockets / TLS handshakes.
"""

import httpx
import time
from collections.abc import AsyncGenerator

from app.core.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_URL,
    HTTP_MAX_CONNECTIONS,
    HTTP_MAX_KEEPALIVE,
    HTTP_TIMEOUT_SECONDS,
    HTTP_STREAM_TIMEOUT_SECONDS,
)
from app.logger_config import logger


_client: httpx.AsyncClient | None = None
_stream_client: httpx.AsyncClient | None = None


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }


async def init_clients() -> None:
    """Create the shared httpx clients. Called once from FastAPI lifespan."""
    global _client, _stream_client
    limits = httpx.Limits(
        max_connections=HTTP_MAX_CONNECTIONS,
        max_keepalive_connections=HTTP_MAX_KEEPALIVE,
    )
    _client = httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS, limits=limits)
    _stream_client = httpx.AsyncClient(timeout=HTTP_STREAM_TIMEOUT_SECONDS, limits=limits)
    logger.info(
        f"HTTP pool ready | max_connections={HTTP_MAX_CONNECTIONS} "
        f"keepalive={HTTP_MAX_KEEPALIVE}"
    )


async def close_clients() -> None:
    """Close the shared httpx clients. Called from FastAPI lifespan shutdown."""
    global _client, _stream_client
    if _client is not None:
        await _client.aclose()
        _client = None
    if _stream_client is not None:
        await _stream_client.aclose()
        _stream_client = None


def _get_client() -> httpx.AsyncClient:
    if _client is None:
        # Fallback for scripts/tests that import without lifespan
        return httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS)
    return _client


def _get_stream_client() -> httpx.AsyncClient:
    if _stream_client is None:
        return httpx.AsyncClient(timeout=HTTP_STREAM_TIMEOUT_SECONDS)
    return _stream_client


async def call_openrouter(payload: dict) -> dict:
    """
    Non-streaming call to OpenRouter. Returns the full JSON response.
    """
    logger.info(f"OpenRouter call started | model={payload.get('model')}")

    try:
        start = time.time()
        client = _get_client()
        res = await client.post(OPENROUTER_URL, headers=_headers(), json=payload)

        duration = round(time.time() - start, 3)
        res.raise_for_status()
        data = res.json()
        logger.info(f"OpenRouter call success ({duration}s)")
        return data

    except httpx.HTTPStatusError as e:
        logger.error(
            f"OpenRouter HTTP error: {e.response.status_code} — {e.response.text}",
            exc_info=True,
        )
        raise
    except httpx.RequestError:
        logger.error("OpenRouter request failed", exc_info=True)
        raise
    except Exception:
        logger.error("Unexpected error in OpenRouter call", exc_info=True)
        raise


async def stream_openrouter(payload: dict) -> AsyncGenerator[str, None]:
    """
    Streaming call to OpenRouter. Yields raw SSE `data: ...` lines
    (without the `data: ` prefix stripped — just the JSON payload string).
    The caller gets each line as OpenRouter sends it.
    """
    payload["stream"] = True
    logger.info(f"OpenRouter stream started | model={payload.get('model')}")

    try:
        client = _get_stream_client()
        async with client.stream(
            "POST", OPENROUTER_URL, headers=_headers(), json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:]  # strip "data: " prefix
                    yield data

    except httpx.HTTPStatusError as e:
        logger.error(
            f"OpenRouter stream HTTP error: {e.response.status_code} — {e.response.text}",
            exc_info=True,
        )
        raise
    except httpx.RequestError:
        logger.error("OpenRouter stream request failed", exc_info=True)
        raise
    except Exception:
        logger.error("Unexpected error in OpenRouter stream", exc_info=True)
        raise
