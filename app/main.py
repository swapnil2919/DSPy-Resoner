import dspy
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.routes.chat import router as chat_router
from app.api.routes.tools import router as tool_router
from app.api.routes.benchmark import router as benchmark_router
from app.api.routes.agents.react import router as react_router
from app.api.routes.agents.rlm import router as rlm_router
from app.core.config import CORS_ALLOW_ORIGINS
from app.core.openrouter_client import init_clients, close_clients

from app.logger_config import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    logger.info("🚀 FastAPI server starting...")
    # Use markdown-style ChatAdapter so weak / open-source models (GLM, Llama,
    # Mistral, Minimax) reliably produce the {reasoning, answer} fields ReAct
    # expects. JSONAdapter rejects prose-only replies; ChatAdapter recovers.
    dspy.configure(adapter=dspy.ChatAdapter())
    await init_clients()

    yield  # <-- app runs here
    # SHUTDOWN
    logger.info("🛑 FastAPI server shutting down...")
    await close_clients()


app = FastAPI(
    title="OpenRouter Proxy",
    description="OpenAI-compatible proxy for OpenRouter with local tool execution",
    lifespan=lifespan,
)

# CORS — comma-separated origins or "*"
_origins = [o.strip() for o in CORS_ALLOW_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI-compatible endpoint: /v1/chat/completions
app.include_router(chat_router, prefix="/v1")

# Utility endpoint: /tools (list available tools)
app.include_router(tool_router, prefix="/tools")

# Benchmark endpoints: /benchmarks/tiny
app.include_router(benchmark_router, prefix="/benchmarks")

# DSPy ReAct agent: /dspy/react/chat/completions
app.include_router(react_router, prefix="/dspy/react")

# DSPy RLM agent: /dspy/rlm/chat/completions
app.include_router(rlm_router, prefix="/dspy/rlm")


# health check (intentionally unauthenticated so ngrok / load-balancers can probe)
@app.get("/")
async def health_check():
    return {"status": "ok"}
