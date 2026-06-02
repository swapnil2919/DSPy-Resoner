import os
from pathlib import Path
from dotenv import load_dotenv

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Load .env from project root
env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)

# --- OpenRouter (upstream) ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "meta-llama/llama-3.1-8b-instruct")

# --- CORS ---
# Comma-separated list, or "*" to allow any origin.
CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*")

# --- Upstream HTTP pool (for handling many concurrent users) ---
HTTP_MAX_CONNECTIONS = int(os.getenv("HTTP_MAX_CONNECTIONS", "1000"))
HTTP_MAX_KEEPALIVE = int(os.getenv("HTTP_MAX_KEEPALIVE", "200"))
HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "60"))
HTTP_STREAM_TIMEOUT_SECONDS = float(os.getenv("HTTP_STREAM_TIMEOUT_SECONDS", "300"))

# --- DSPy / ReAct tunables ---
# Hard cap for a single ReAct run. Prevents a hung upstream from pinning a
# request forever. Cancels the asyncio side; the underlying thread is allowed
# to finish naturally (Python can't kill threads cooperatively).
DSPY_REACT_TIMEOUT_SECONDS = float(os.getenv("DSPY_REACT_TIMEOUT_SECONDS", "120"))
# Max Thought→Action→Observation iterations. Each iteration = 1+ LLM API call.
# 4 is enough for any real task (weather, email, multi-tool). 8 wastes ~24s on
# irrelevant requests where the model keeps retrying unavailable tools.
DSPY_REACT_MAX_ITERS = int(os.getenv("DSPY_REACT_MAX_ITERS", "4"))
# Max characters kept from any single tool output before it is appended to the
# ReAct context. Prevents one large tool response from blowing up the prompt.
DSPY_TOOL_OUTPUT_MAX_CHARS = int(os.getenv("DSPY_TOOL_OUTPUT_MAX_CHARS", "3000"))
# Number of words emitted per SSE chunk when "streaming" a fully-computed
# answer. 1 = word-by-word (lots of tiny chunks, smoother UI). Larger values
# reduce per-event overhead. Set to 0 to emit the entire answer in one chunk.
DSPY_STREAM_CHUNK_WORDS = int(os.getenv("DSPY_STREAM_CHUNK_WORDS", "4"))

# --- DSPy / RLM tunables ---
# RLM uses a sandboxed REPL — longer timeout since code execution is involved.
DSPY_RLM_TIMEOUT_SECONDS = float(os.getenv("DSPY_RLM_TIMEOUT_SECONDS", "300"))
DSPY_RLM_MAX_ITERATIONS = int(os.getenv("DSPY_RLM_MAX_ITERATIONS", "2"))
DSPY_RLM_MAX_LLM_CALLS = int(os.getenv("DSPY_RLM_MAX_LLM_CALLS", "3"))

