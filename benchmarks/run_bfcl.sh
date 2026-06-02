#!/usr/bin/env bash
# ============================================================================
# BFCL v3 + v4 Benchmark Runner
# Berkeley Function-Calling Leaderboard
# https://github.com/ShishirPatil/gorilla
# ============================================================================

set -euo pipefail

PROXY_URL="${PROXY_URL:-http://localhost:8000/v1}"
MODEL_NAME="${MODEL_NAME:-openrouter-proxy}"
BENCHMARK_DIR="${BENCHMARK_DIR:-$(dirname "$0")/bfcl_workspace}"

echo "=== BFCL v3 + v4 Benchmark ==="
echo "Proxy URL:  $PROXY_URL"
echo "Model Name: $MODEL_NAME"
echo "Workspace:  $BENCHMARK_DIR"
echo ""

# --- Step 1: Setup ---
if [ ! -d "$BENCHMARK_DIR" ]; then
    echo "[1/4] Cloning BFCL repository..."
    git clone https://github.com/ShishirPatil/gorilla.git "$BENCHMARK_DIR"
else
    echo "[1/4] BFCL workspace already exists, skipping clone."
fi

cd "$BENCHMARK_DIR/berkeley-function-call-leaderboard"

if ! command -v bfcl &> /dev/null; then
    echo "[2/4] Installing BFCL..."
    pip install -e .
else
    echo "[2/4] BFCL already installed."
fi

# --- Step 2: Configure environment ---
echo "[3/4] Configuring environment..."
export BFCL_PROJECT_ROOT="$(pwd)"

if [ ! -f .env ]; then
    cp bfcl_eval/.env.example .env
fi

# Ensure proxy endpoint is configured
grep -q "REMOTE_OPENAI_BASE_URL" .env 2>/dev/null || \
    echo "REMOTE_OPENAI_BASE_URL=$PROXY_URL" >> .env
grep -q "REMOTE_OPENAI_API_KEY" .env 2>/dev/null || \
    echo "REMOTE_OPENAI_API_KEY=dummy" >> .env

echo "  REMOTE_OPENAI_BASE_URL=$PROXY_URL"
echo "  BFCL_PROJECT_ROOT=$BFCL_PROJECT_ROOT"

# --- Step 3: Generate responses ---
echo ""
echo "[4/4] Running BFCL generation..."
echo ""

# BFCL v3 categories
echo "--- BFCL v3: simple, multiple, parallel, multi_turn ---"
bfcl generate \
    --model "$MODEL_NAME" \
    --test-category simple,multiple,parallel,multi_turn \
    --skip-server-setup \
    --num-threads 1 || echo "Warning: BFCL v3 generation had errors"

# BFCL v4 categories (live)
echo ""
echo "--- BFCL v4: live_simple, live_multiple, live_parallel, live_relevance ---"
bfcl generate \
    --model "$MODEL_NAME" \
    --test-category live_simple,live_multiple,live_parallel,live_relevance \
    --skip-server-setup \
    --num-threads 1 || echo "Warning: BFCL v4 generation had errors"

# --- Step 4: Evaluate ---
echo ""
echo "--- Evaluating v3 ---"
bfcl evaluate \
    --model "$MODEL_NAME" \
    --test-category simple,multiple,parallel,multi_turn || echo "Warning: v3 evaluation had errors"

echo ""
echo "--- Evaluating v4 ---"
bfcl evaluate \
    --model "$MODEL_NAME" \
    --test-category live_simple,live_multiple,live_parallel,live_relevance || echo "Warning: v4 evaluation had errors"

echo ""
echo "=== BFCL Benchmark Complete ==="
echo "Results: $BFCL_PROJECT_ROOT/result/$MODEL_NAME/"
echo "Scores:  $BFCL_PROJECT_ROOT/score/$MODEL_NAME/"
