# 🔀 OpenRouter Proxy

An **OpenAI-compatible** FastAPI proxy for [OpenRouter](https://openrouter.ai/) with **server-side tool execution**, **streaming** (SSE), and a built-in **function-calling loop**.

Point any OpenAI SDK client at this proxy and get access to 200+ LLMs through OpenRouter — with your own tools executed automatically on the server.

---

## ✨ Features

| Feature | Description |
|---|---|
| **OpenAI-compatible API** | Drop-in replacement for `POST /v1/chat/completions` — works with any OpenAI SDK |
| **Streaming (SSE)** | Token-by-token `chat.completion.chunk` events for real-time UI / spinner support |
| **Function calling** | Full support for OpenAI's `tools` parameter — LLM decides when to call tools |
| **Server-side tool execution** | Tools are executed on the proxy server; clients receive the final answer directly |
| **Tool-call loop** | Automatic multi-round tool execution (up to 5 rounds) with safety limits |
| **200+ models** | Access any model on OpenRouter — Llama, GPT, Claude, Gemini, Mistral, and more |
| **Low-cost defaults** | Pre-configured with `meta-llama/llama-3.1-8b-instruct` for minimal API costs |

---

## 📐 Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Client (OpenAI SDK / curl / frontend)                   │
│  POST /v1/chat/completions                               │
└──────────────┬───────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│  FastAPI Proxy Server                                    │
│                                                          │
│  1. Receive OpenAI-format request                        │
│  2. Forward to OpenRouter API                            │
│  3. If LLM returns tool_calls:                           │
│     a. Execute tools locally (get_weather, etc.)         │
│     b. Append tool results to messages                   │
│     c. Call OpenRouter again (repeat up to 5 rounds)     │
│  4. Return final response (or stream chunks)             │
└──────────────┬───────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│  OpenRouter API (https://openrouter.ai/api/v1)           │
│  → Routes to: Llama, GPT, Claude, Gemini, etc.          │
└──────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.13+**
- **[uv](https://docs.astral.sh/uv/)** package manager
- An **OpenRouter API key** → [Get one here](https://openrouter.ai/keys)

### 1. Clone & Install

```bash
git clone <your-repo-url>
cd openrouter_proxy

# Install dependencies (creates .venv automatically)
uv sync
```

### 2. Configure

```bash
# Set your API key
cp .env.example .env   # or just edit .env directly
```

Edit `.env`:

```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
DEFAULT_MODEL=meta-llama/llama-3.1-8b-instruct
```

### 3. Run

```bash
uv run uvicorn app.main:app --reload
```

The server starts at **http://localhost:8000**.

### 4. Test

```bash
# Quick health check
curl http://localhost:8000/

# Run the full E2E test suite
uv run python test_e2e.py
```

---

## 📡 API Reference

### `POST /v1/chat/completions`

Fully compatible with the [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat).

#### Request Body

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | `string` | `meta-llama/llama-3.1-8b-instruct` | Any [OpenRouter model ID](https://openrouter.ai/models) |
| `messages` | `array` | *required* | Array of message objects (`role` + `content`) |
| `stream` | `boolean` | `false` | Enable SSE streaming |
| `tools` | `array` | `null` | List of function definitions the model can call |
| `tool_choice` | `string\|object` | `null` | Control tool selection (`auto`, `none`, or specific) |
| `temperature` | `float` | `null` | Sampling temperature (0.0–2.0) |
| `max_tokens` | `integer` | `null` | Maximum tokens in the response |
| `top_p` | `float` | `null` | Nucleus sampling parameter |
| `stop` | `string\|array` | `null` | Stop sequences |
| `frequency_penalty` | `float` | `null` | Frequency penalty (-2.0–2.0) |
| `presence_penalty` | `float` | `null` | Presence penalty (-2.0–2.0) |

#### Non-Streaming Response

```json
{
  "id": "gen-1777470534-ooh3m0ZbnAmfyfGZpXVo",
  "object": "chat.completion",
  "created": 1777470534,
  "model": "meta-llama/llama-3.1-8b-instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 17,
    "completion_tokens": 9,
    "total_tokens": 26
  }
}
```

#### Streaming Response (SSE)

Each event is a `chat.completion.chunk`:

```
data: {"id":"gen-...","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"},"index":0,"finish_reason":null}]}

data: {"id":"gen-...","object":"chat.completion.chunk","choices":[{"delta":{"content":"!"},"index":0,"finish_reason":null}]}

data: {"id":"gen-...","object":"chat.completion.chunk","choices":[{"delta":{},"index":0,"finish_reason":"stop"}]}

data: [DONE]
```

### `GET /tools`

Lists all locally registered tools and their parameter schemas.

```json
{
  "count": 2,
  "tools": [
    {
      "name": "get_weather",
      "description": "Get current weather information for a given city",
      "parameters": { "type": "object", "properties": { "city": { "type": "string" } }, "required": ["city"] }
    },
    {
      "name": "generate_email",
      "description": "Generate a formatted email with the given recipient, subject, and optional tone",
      "parameters": { "..." }
    }
  ]
}
```

### `GET /`

Health check endpoint. Returns `{"status": "ok"}`.

---

## 💡 Usage Examples

### curl — Basic Chat

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/llama-3.1-8b-instruct",
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
  }'
```

### curl — Streaming

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/llama-3.1-8b-instruct",
    "messages": [{"role": "user", "content": "Tell me a short joke."}],
    "stream": true
  }'
```

### curl — Tool Calling

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/llama-3.1-8b-instruct",
    "messages": [{"role": "user", "content": "What is the weather in Mumbai?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }]
  }'
```

### Python — OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # auth is handled by the proxy
)

# Non-streaming
response = client.chat.completions.create(
    model="meta-llama/llama-3.1-8b-instruct",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="meta-llama/llama-3.1-8b-instruct",
    messages=[{"role": "user", "content": "Tell me a story."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### JavaScript — OpenAI SDK

```javascript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:8000/v1",
  apiKey: "not-needed",
});

const response = await client.chat.completions.create({
  model: "meta-llama/llama-3.1-8b-instruct",
  messages: [{ role: "user", content: "Hello!" }],
});

console.log(response.choices[0].message.content);
```

### Frontend — Streaming with Spinner

```javascript
const response = await fetch("http://localhost:8000/v1/chat/completions", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    model: "meta-llama/llama-3.1-8b-instruct",
    messages: [{ role: "user", content: "Hello!" }],
    stream: true,
  }),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

// Hide spinner once first chunk arrives
showSpinner();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  hideSpinner(); // First chunk received — hide loading state

  const text = decoder.decode(value);
  for (const line of text.split("\n")) {
    if (line.startsWith("data: ") && line !== "data: [DONE]") {
      const chunk = JSON.parse(line.slice(6));
      const content = chunk.choices?.[0]?.delta?.content;
      if (content) appendToChat(content);
    }
  }
}
```

---

## 🔧 Adding Custom Tools

### 1. Create the tool function

```python
# app/tools/my_tool.py
import asyncio
from app.logger_config import logger

async def my_custom_tool(param1: str, param2: int = 10):
    logger.info(f"my_custom_tool called with {param1}, {param2}")
    await asyncio.sleep(0.5)  # simulate work
    return {"result": f"Processed {param1} with value {param2}"}
```

### 2. Register it in the registry

```python
# app/tools/registry.py

from app.tools.my_tool import my_custom_tool

# Add to TOOL_MAP
TOOL_MAP = {
    "get_weather": get_weather,
    "generate_email": generate_email,
    "my_custom_tool": my_custom_tool,  # ← add here
}

# Add to TOOL_SCHEMAS
TOOL_SCHEMAS = [
    # ... existing schemas ...
    {
        "type": "function",
        "function": {
            "name": "my_custom_tool",
            "description": "Describe what your tool does",
            "parameters": {
                "type": "object",
                "properties": {
                    "param1": {"type": "string", "description": "First parameter"},
                    "param2": {"type": "integer", "description": "Second parameter", "default": 10}
                },
                "required": ["param1"]
            }
        }
    }
]
```

### 3. Use it

Pass the tool schema in your request's `tools` array. The LLM will decide when to call it, and the proxy will execute it automatically.

---

## 📁 Project Structure

```
openrouter_proxy/
├── .env                           # API key & config (not committed)
├── .gitignore
├── pyproject.toml                 # Dependencies (managed by uv)
├── uv.lock                        # Lockfile for reproducible installs
├── test_e2e.py                    # End-to-end test suite
│
└── app/
    ├── main.py                    # FastAPI app entry point
    ├── logger_config.py           # Logging setup (file + console)
    │
    ├── core/
    │   ├── config.py              # Environment variables & constants
    │   └── openrouter_client.py   # HTTP client (streaming + non-streaming)
    │
    ├── schemas/
    │   └── openai.py              # OpenAI-compatible Pydantic models
    │
    ├── api/
    │   └── routes/
    │       ├── chat.py            # POST /v1/chat/completions
    │       └── tools.py           # GET /tools
    │
    ├── services/
    │   ├── llm_service.py         # Tool-call orchestration loop
    │   └── tool_executor.py       # Executes individual tool calls
    │
    ├── tools/
    │   ├── registry.py            # Tool registry & OpenAI schemas
    │   ├── weather_tool.py        # Mock weather tool
    │   └── email_tool.py          # Mock email generator tool
    │
    └── utils/
        └── sse.py                 # SSE formatting helpers
```

---

## ⚙️ Configuration

All configuration is via environment variables in `.env`:

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENROUTER_API_KEY` | ✅ | — | Your OpenRouter API key |
| `DEFAULT_MODEL` | ❌ | `meta-llama/llama-3.1-8b-instruct` | Fallback model when `model` is not specified in the request |
| `PROXY_API_KEY` | ❌ | — | When set, every protected route requires `Authorization: Bearer <key>`. Leave unset for local dev. **Set this before exposing via ngrok.** |
| `CORS_ALLOW_ORIGINS` | ❌ | `*` | Comma-separated list of allowed origins. |
| `HTTP_MAX_CONNECTIONS` | ❌ | `1000` | Upstream connection pool size (max simultaneous OpenRouter sockets). |
| `HTTP_MAX_KEEPALIVE` | ❌ | `200` | How many idle keep-alive connections to retain. |
| `HTTP_TIMEOUT_SECONDS` | ❌ | `60` | Non-streaming request timeout. |
| `HTTP_STREAM_TIMEOUT_SECONDS` | ❌ | `300` | Streaming connection timeout. |

---

## 🌐 Exposing the proxy via ngrok (e.g. for a remote benchmark harness)

Use this when something off-machine — like the harness lab on a GPU server — needs to call your local proxy.

### One-time setup

1. Install ngrok: `winget install ngrok.ngrok` (or download from [ngrok.com/download](https://ngrok.com/download)).
2. Add your auth token (from [dashboard.ngrok.com](https://dashboard.ngrok.com/get-started/your-authtoken)):
   ```powershell
   ngrok config add-authtoken <your-token>
   ```
3. **Set a `PROXY_API_KEY` in `.env`** — without this, anyone with the ngrok URL can hit your proxy and burn OpenRouter credits.
   ```env
   PROXY_API_KEY=pick-a-long-random-string
   ```

### Run

```powershell
# Terminal 1 — start the API
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000

# Terminal 2 — start the tunnel to your reserved domain
ngrok http --domain=spearfish-limb-makeover.ngrok-free.dev 8000
```

### Configure the harness lab

Point the OpenAI-compatible client at:

| Setting | Value |
|---|---|
| Base URL | `https://spearfish-limb-makeover.ngrok-free.dev/v1` |
| API key  | the same value as `PROXY_API_KEY` |
| Model    | any OpenRouter model id (e.g. `meta-llama/llama-3.1-8b-instruct`) |

### Verify from another machine

```bash
# Health check (no auth required)
curl https://spearfish-limb-makeover.ngrok-free.dev/

# Authenticated chat
curl https://spearfish-limb-makeover.ngrok-free.dev/v1/chat/completions \
  -H "Authorization: Bearer $PROXY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/llama-3.1-8b-instruct","messages":[{"role":"user","content":"hi"}]}'
```

### Notes & gotchas

- **Your laptop must stay awake & online** while the harness is running — if it sleeps, the tunnel dies.
- The free `ngrok-free.dev` domain shows a one-time browser interstitial for browser User-Agents. Real HTTP clients (httpx, requests, curl) don't trigger it. If they ever do, send `ngrok-skip-browser-warning: true` (the included test file does this automatically).
- ngrok free tier has rate limits (currently 120 req/min on the free plan as of writing). For a large benchmark run, consider a paid plan.
- Streaming (SSE) works through the tunnel without any extra config.

---

## 📈 Scaling for many concurrent users

The proxy is async end-to-end and uses a shared `httpx.AsyncClient` with a tuned connection pool, so a single uvicorn process handles hundreds of concurrent in-flight requests.

For 1000+ concurrent users, run multiple worker processes:

```powershell
# Each worker is a separate Python process; pick a number close to your CPU count.
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Each worker has its own pool sized by `HTTP_MAX_CONNECTIONS` (default `1000`). Tune it down if you hit OpenRouter rate limits, or up if you have plenty of upstream headroom and many idle clients.

What this setup does NOT include (deliberate — keep it simple):
- Per-user rate limiting → add a reverse proxy (Caddy/Nginx) or `slowapi` if you need it.
- Per-user keys / billing → the single `PROXY_API_KEY` is shared.

---

## 🧪 Testing

The project includes a comprehensive E2E test suite that validates all 4 core scenarios:

```bash
# Start the server
uv run uvicorn app.main:app --port 8000 &

# Run tests
uv run python test_e2e.py
```

| Test | Description |
|---|---|
| **Test 1** | Non-streaming basic chat — verifies `chat.completion` response format |
| **Test 2** | Streaming chat — verifies token-by-token SSE chunks with `data: [DONE]` |
| **Test 3** | Tool calling (non-streaming) — verifies server-side tool execution |
| **Test 4** | Streaming + tools — verifies buffered tool execution with streamed final response |

All tests use the official **OpenAI Python SDK** (`openai>=2.33.0`) to validate true SDK compatibility.

---

## 🛡️ Safety & Limits

| Protection | Detail |
|---|---|
| **Max tool rounds** | 5 rounds per request — prevents infinite loops from overly eager models |
| **Forced final response** | After exhausting tool rounds, `tools` are removed from the payload and the LLM is forced to produce a text answer |
| **Timeout** | 60s for non-streaming calls, 120s for streaming connections |
| **Error propagation** | Upstream OpenRouter errors (401, 429, 500) are surfaced to the client with proper HTTP status codes |
| **Structured logging** | All requests, tool executions, and errors logged to `logs/app.log` with rotation |

---

## 🤖 Recommended Models

| Model | Tool Calling | Cost | Notes |
|---|---|---|---|
| `meta-llama/llama-3.1-8b-instruct` | ⚡ Basic | 💰 Very low | Default. May not always use tools reliably |
| `meta-llama/llama-3.1-70b-instruct` | ✅ Good | 💰💰 Low | Better tool-calling accuracy |
| `openai/gpt-4o-mini` | ✅ Excellent | 💰💰 Low | Highly reliable function calling |
| `anthropic/claude-3.5-haiku` | ✅ Excellent | 💰💰 Low | Fast with strong tool support |
| `openai/gpt-4o` | ✅ Excellent | 💰💰💰 Medium | Best overall quality |
| `anthropic/claude-3.5-sonnet` | ✅ Excellent | 💰💰💰 Medium | Great for complex tool chains |

Browse all available models at [openrouter.ai/models](https://openrouter.ai/models).

---

## 📝 Logs

Logs are written to both the console and `logs/app.log` (with rotation at 500KB, 3 backups).

Example log output:

```
2026-04-29 19:19:40 - INFO - Chat completions request | model=meta-llama/llama-3.1-8b-instruct stream=False tools=1
2026-04-29 19:19:40 - INFO - OpenRouter call started | model=meta-llama/llama-3.1-8b-instruct
2026-04-29 19:19:41 - INFO - OpenRouter call success (0.686s)
2026-04-29 19:19:41 - INFO - Tool round 1: 1 tool call(s)
2026-04-29 19:19:41 - INFO - Tool execution requested: get_weather (id=call_26ICTYs...)
2026-04-29 19:19:41 - INFO - get_weather called
2026-04-29 19:19:41 - DEBUG - Input -> city: Mumbai
2026-04-29 19:19:42 - INFO - Weather fetched successfully for city: Mumbai
2026-04-29 19:19:42 - INFO - Tool executed successfully: get_weather (1.001s)
2026-04-29 19:19:42 - INFO - OpenRouter call started | model=meta-llama/llama-3.1-8b-instruct
2026-04-29 19:19:43 - INFO - OpenRouter call success (1.049s)
2026-04-29 19:19:43 - INFO - Completion finished after 2 round(s)
```

---

## 📄 License

MIT

---

## 🙏 Acknowledgements

- [OpenRouter](https://openrouter.ai/) — Unified API for LLMs
- [FastAPI](https://fastapi.tiangolo.com/) — High-performance Python web framework
- [uv](https://docs.astral.sh/uv/) — Fast Python package manager