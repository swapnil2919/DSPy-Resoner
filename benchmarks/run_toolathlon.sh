#!/usr/bin/env bash
# ============================================================================
# Toolathlon Benchmark Runner
# 600+ real-world tools, long-horizon tasks (ICLR 2026)
# https://github.com/hkust-nlp/Toolathlon
# ============================================================================

set -euo pipefail

PROXY_URL="${PROXY_URL:-http://localhost:8000/v1}"
MODEL_NAME="${MODEL_NAME:-meta-llama/llama-3.1-8b-instruct}"
API_KEY="${TOOLATHLON_OPENAI_API_KEY:-${OPENAI_API_KEY:-dummy}}"
BENCHMARK_DIR="${BENCHMARK_DIR:-$(dirname "$0")/toolathlon_workspace}"
MODE="${MODE:-public}"
WORKERS="${WORKERS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-./results}"

echo "=== Toolathlon Benchmark ==="
echo "Proxy URL:  $PROXY_URL"
echo "Model:      $MODEL_NAME"
echo "Mode:       $MODE"
echo "Workers:    $WORKERS"
echo ""

# --- Step 1: Setup ---
if [ ! -d "$BENCHMARK_DIR" ]; then
    echo "[1/3] Cloning Toolathlon repository..."
    git clone https://github.com/hkust-nlp/Toolathlon.git "$BENCHMARK_DIR"
else
    echo "[1/3] Toolathlon workspace already exists."
fi

cd "$BENCHMARK_DIR"

echo "[2/3] Installing dependencies..."
bash global_preparation/install_env_minimal.sh false 2>/dev/null || \
    echo "Dependencies may already be installed or require manual setup."

# --- Step 2: Configure ---
echo "[3/3] Configuring environment..."
export TOOLATHLON_OPENAI_API_KEY="$API_KEY"
export TOOLATHLON_OPENAI_BASE_URL="$PROXY_URL"

# --- Step 3: Run ---
echo ""
echo "--- Running Toolathlon ($MODE mode) ---"

if [ "$MODE" = "public" ]; then
    # Create a debug task list for initial testing
    cat > debug_tasks.txt << EOF
find-alita-paper
EOF

    python eval_client.py run \
        --mode public \
        --base-url "$PROXY_URL" \
        --model-name "$MODEL_NAME" \
        --output-dir "$OUTPUT_DIR" \
        --server-host 47.253.6.47 \
        --api-key "$API_KEY" \
        --workers "$WORKERS" \
        --server-port 8080 \
        --ws-proxy-port 8081 \
        --task-list-file ./debug_tasks.txt
else
    echo "Self-hosted mode requires Docker. See Toolathlon docs for setup."
    echo "https://github.com/hkust-nlp/Toolathlon"
fi

echo ""
echo "=== Toolathlon Benchmark Complete ==="
echo "Results: $OUTPUT_DIR/"
