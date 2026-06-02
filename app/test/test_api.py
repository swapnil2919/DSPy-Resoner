"""
API smoke tests. Works against local server OR your ngrok URL.

Usage (local):
    uv run python app/test/test_api.py

Usage (ngrok):
    $env:API_BASE_URL = "https://spearfish-limb-makeover.ngrok-free.dev"
    $env:PROXY_API_KEY = "the-key-you-set-on-the-server"
    uv run python app/test/test_api.py

Env vars:
    API_BASE_URL   default http://localhost:8000   — root URL of the proxy
    PROXY_API_KEY  default ""                      — sent as Bearer token
    TEST_MODEL     default meta-llama/llama-3.1-8b-instruct
    CONCURRENCY    default 10                      — how many parallel requests in load test
"""

import os
import sys
import time
import asyncio
import httpx

BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("PROXY_API_KEY", "")
MODEL = os.getenv("TEST_MODEL", "meta-llama/llama-3.1-8b-instruct")
CONCURRENCY = int(os.getenv("CONCURRENCY", "10"))

HEADERS = {"Content-Type": "application/json"}
if API_KEY:
    HEADERS["Authorization"] = f"Bearer {API_KEY}"
# Skip ngrok's browser interstitial if it ever fires for our user agent.
HEADERS["ngrok-skip-browser-warning"] = "true"

DIVIDER = "\n" + "=" * 60 + "\n"


def _print_config():
    print(f"  Base URL : {BASE_URL}")
    print(f"  Model    : {MODEL}")
    print(f"  Auth     : {'Bearer ****' if API_KEY else '(none)'}")


async def test_health():
    print(DIVIDER + "TEST 1: Health check")
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(f"{BASE_URL}/")
        print(f"  Status: {r.status_code}  Body: {r.text}")
        assert r.status_code == 200
        assert r.json().get("status") == "ok"
    print("  PASSED")


async def test_auth_rejected_without_key():
    """Only meaningful when PROXY_API_KEY is set on the server."""
    print(DIVIDER + "TEST 2: Auth rejects missing/invalid key (only if PROXY_API_KEY is set)")
    if not API_KEY:
        print("  SKIPPED — PROXY_API_KEY not set on the client; can't tell if server enforces auth.")
        return
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            f"{BASE_URL}/v1/chat/completions",
            headers={"Content-Type": "application/json", "ngrok-skip-browser-warning": "true"},
            json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
        )
        print(f"  Status without auth: {r.status_code}")
        assert r.status_code == 401, "expected 401 when PROXY_API_KEY is enforced"
    print("  PASSED")


async def test_non_streaming():
    print(DIVIDER + "TEST 3: Non-streaming chat")
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{BASE_URL}/v1/chat/completions",
            headers=HEADERS,
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
            },
        )
        print(f"  Status: {r.status_code}")
        assert r.status_code == 200, r.text
        body = r.json()
        content = body["choices"][0]["message"]["content"]
        print(f"  Content: {content}")
        assert content
    print("  PASSED")


async def test_streaming():
    print(DIVIDER + "TEST 4: Streaming chat (SSE)")
    chunks = 0
    collected = ""
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            f"{BASE_URL}/v1/chat/completions",
            headers=HEADERS,
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Count from 1 to 3."}],
                "stream": True,
            },
        ) as resp:
            print(f"  Status: {resp.status_code}")
            assert resp.status_code == 200
            print("  Streaming: ", end="", flush=True)
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                chunks += 1
                import json as _json
                try:
                    payload = _json.loads(data)
                    delta = payload["choices"][0]["delta"].get("content")
                    if delta:
                        collected += delta
                        print(delta, end="", flush=True)
                except Exception:
                    pass
    print()
    print(f"  Chunks: {chunks}, total chars: {len(collected)}")
    assert chunks > 0
    print("  PASSED")


async def test_concurrent_load():
    """
    Fires N requests in parallel to confirm the shared connection pool +
    async event loop can handle concurrent users without serializing.
    """
    print(DIVIDER + f"TEST 5: Concurrent load ({CONCURRENCY} parallel requests)")

    async def _one(client: httpx.AsyncClient, i: int) -> tuple[int, float]:
        t0 = time.time()
        r = await client.post(
            f"{BASE_URL}/v1/chat/completions",
            headers=HEADERS,
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": f"Reply with the single word: pong{i}"}],
                "max_tokens": 10,
            },
        )
        return r.status_code, time.time() - t0

    async with httpx.AsyncClient(timeout=120) as client:
        t0 = time.time()
        results = await asyncio.gather(*[_one(client, i) for i in range(CONCURRENCY)])
        wall = time.time() - t0

    ok = sum(1 for s, _ in results if s == 200)
    avg_latency = sum(d for _, d in results) / len(results)
    print(f"  {ok}/{CONCURRENCY} succeeded, wall={wall:.2f}s, avg latency={avg_latency:.2f}s")
    assert ok == CONCURRENCY, f"only {ok}/{CONCURRENCY} succeeded"
    print("  PASSED")


async def main():
    print("Smoke tests for OpenRouter proxy")
    _print_config()
    try:
        await test_health()
        await test_auth_rejected_without_key()
        await test_non_streaming()
        await test_streaming()
        await test_concurrent_load()
    except AssertionError as e:
        print(f"\n  FAILED: {e}")
        sys.exit(1)
    except httpx.HTTPError as e:
        print(f"\n  HTTP ERROR: {e}")
        sys.exit(1)
    print(DIVIDER + "ALL TESTS PASSED" + DIVIDER)


if __name__ == "__main__":
    asyncio.run(main())
