import dspy
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.routes.chat import router as chat_router
from app.api.routes.agents.rlm import router as rlm_router, _get_rlm
from app.core.config import CORS_ALLOW_ORIGINS
from app.core.openrouter_client import init_clients, close_clients
from app.logger_config import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 FastAPI server starting...")
    # Use markdown-style ChatAdapter so weak / open-source models (GLM, Llama,
    # Mistral, Minimax) reliably produce the {reasoning, code} fields RLM
    # expects. JSONAdapter rejects prose-only replies; ChatAdapter recovers.
    # Must be set before _get_rlm() so the RLM module binds to the right adapter.
    dspy.configure(adapter=dspy.ChatAdapter())
    await init_clients()
    # Pre-warm the RLM module so the first benchmark request doesn't pay the
    # ~15s sandbox / DSPy initialisation cost.
    _get_rlm()
    yield
    logger.info("🛑 FastAPI server shutting down...")
    await close_clients()


app = FastAPI(
    title="OpenRouter Proxy",
    description=(
        "POST /v1/chat/completions     — DSPy RLM agent (sandboxed Python REPL),\n"
        "                                auto-falls back to transparent proxy when external tools are sent\n"
        "POST /proxy/chat/completions  — transparent OpenAI-compatible proxy (benchmarks, tool_calls)"
    ),
    lifespan=lifespan,
)

_origins = [o.strip() for o in CORS_ALLOW_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DSPy RLM agent — sandboxed Python REPL reasoning loop, returns final answer.
# Use this for reasoning-capable models with iterative code execution.
# base_url = "https://<cloudflare-host>/v1"
app.include_router(rlm_router, prefix="/v1")

# Transparent OpenAI-compatible proxy — forwards tools/tool_calls as-is.
# Use this for benchmarks (tau2, BFCL, LiveMCPBench) that need native tool_calls.
# base_url = "https://<cloudflare-host>/proxy"
app.include_router(chat_router, prefix="/proxy")


@app.get("/")
async def health_check():
    return {
        "status": "ok",
        "endpoints": {
            "rlm_agent": "POST /v1/chat/completions",
            "proxy": "POST /proxy/chat/completions",
        },
    }
