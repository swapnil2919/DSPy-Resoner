#!/usr/bin/env bash
# ============================================================================
# Tau2 / τ³-bench Benchmark Runner
# Latest version: airline, retail, telecom, banking_knowledge, voice
# https://github.com/sierra-research/tau2-bench
# ============================================================================

set -euo pipefail

PROXY_URL="${PROXY_URL:-http://localhost:8000/v1}"
MODEL_NAME="${MODEL_NAME:-openai/meta-llama/llama-3.1-8b-instruct}"
USER_MODEL="${USER_MODEL:-gpt-4.1}"
BENCHMARK_DIR="${BENCHMARK_DIR:-$(dirname "$0")/tau2_workspace}"
DOMAIN="${DOMAIN:-airline}"
NUM_TASKS="${NUM_TASKS:-5}"
NUM_TRIALS="${NUM_TRIALS:-1}"

echo "=== Tau2 / τ³-bench Benchmark ==="
echo "Proxy URL:     $PROXY_URL"
echo "Agent Model:   $MODEL_NAME"
echo "User Model:    $USER_MODEL"
echo "Domain:        $DOMAIN"
echo "Num Tasks:     $NUM_TASKS"
echo "Num Trials:    $NUM_TRIALS"
echo ""

# --- Step 1: Setup ---
if [ ! -d "$BENCHMARK_DIR" ]; then
    echo "[1/4] Cloning τ³-bench repository..."
    git clone https://github.com/sierra-research/tau2-bench.git "$BENCHMARK_DIR"
else
    echo "[1/4] τ³-bench workspace already exists."
fi

cd "$BENCHMARK_DIR"

echo "[2/4] Installing with uv..."
uv sync 2>/dev/null || echo "Already synced"

# --- Step 2: Configure ---
echo "[3/4] Configuring environment..."
if [ ! -f .env ]; then
    cp .env.example .env
fi

# Ensure proxy endpoint is configured in .env
grep -q "OPENAI_API_BASE" .env 2>/dev/null || \
    echo "OPENAI_API_BASE=$PROXY_URL" >> .env
grep -q "OPENAI_API_KEY" .env 2>/dev/null || \
    echo "OPENAI_API_KEY=${OPENAI_API_KEY:-dummy}" >> .env

# Also export for LiteLLM
export OPENAI_API_BASE="$PROXY_URL"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

# --- Step 3: Run ---
echo "[4/4] Running τ³-bench on $DOMAIN domain..."
echo ""

uv run tau2 run \
    --domain "$DOMAIN" \
    --agent-llm "$MODEL_NAME" \
    --user-llm "$USER_MODEL" \
    --num-trials "$NUM_TRIALS" \
    --num-tasks "$NUM_TASKS"

# --- Step 4: View results ---
echo ""
echo "--- Results ---"
uv run tau2 view 2>/dev/null || echo "Use 'tau2 view' to browse results interactively."

echo ""
echo "=== τ³-bench Benchmark Complete ==="
echo "Results: $BENCHMARK_DIR/data/simulations/"
echo ""
echo "Available domains: mock, airline, retail, telecom, banking_knowledge"
echo "Re-run with: DOMAIN=retail $0"
