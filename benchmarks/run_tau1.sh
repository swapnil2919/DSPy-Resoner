#!/usr/bin/env bash
# ============================================================================
# Tau1 (τ-bench) Benchmark Runner — DEPRECATED
# Original τ-bench: airline + retail domains
# https://github.com/sierra-research/tau-bench
#
# NOTE: This repo is deprecated. Use run_tau2.sh for the latest τ³-bench.
# ============================================================================

set -euo pipefail

PROXY_URL="${PROXY_URL:-http://localhost:8000/v1}"
MODEL_NAME="${MODEL_NAME:-meta-llama/llama-3.1-8b-instruct}"
USER_MODEL="${USER_MODEL:-gpt-4o}"
BENCHMARK_DIR="${BENCHMARK_DIR:-$(dirname "$0")/tau1_workspace}"
DOMAIN="${DOMAIN:-retail}"
NUM_TASKS="${NUM_TASKS:-5}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"

echo "=== Tau1 (τ-bench) Benchmark — DEPRECATED ==="
echo "⚠️  This benchmark is outdated. Consider using run_tau2.sh instead."
echo ""
echo "Proxy URL:     $PROXY_URL"
echo "Agent Model:   $MODEL_NAME"
echo "User Model:    $USER_MODEL"
echo "Domain:        $DOMAIN"
echo "Num Tasks:     $NUM_TASKS"
echo ""

# --- Step 1: Setup ---
if [ ! -d "$BENCHMARK_DIR" ]; then
    echo "[1/3] Cloning tau-bench repository..."
    git clone https://github.com/sierra-research/tau-bench.git "$BENCHMARK_DIR"
else
    echo "[1/3] tau-bench workspace already exists."
fi

cd "$BENCHMARK_DIR"

echo "[2/3] Installing tau-bench..."
pip install -e . 2>/dev/null || echo "Already installed"

# --- Step 2: Configure ---
echo "[3/3] Configuring environment..."
export OPENAI_API_BASE="$PROXY_URL"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

# --- Step 3: Run ---
echo ""
echo "--- Running Tau1 on $DOMAIN domain ---"
python run.py \
    --agent-strategy tool-calling \
    --env "$DOMAIN" \
    --model "$MODEL_NAME" \
    --model-provider openai \
    --user-model "$USER_MODEL" \
    --user-model-provider openai \
    --user-strategy llm \
    --max-concurrency "$MAX_CONCURRENCY"

echo ""
echo "=== Tau1 Benchmark Complete ==="
echo "Results saved in current directory."
