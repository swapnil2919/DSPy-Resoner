#!/usr/bin/env bash
# ============================================================================
# LiveMCPBench Benchmark Runner
# Large-scale MCP toolset navigation benchmark
# https://github.com/icip-cas/LiveMCPBench
#
# REQUIREMENTS:
#   - Docker with GPU support (--gpus all)
#   - npm (for MCP server setup inside container)
#   - ~10GB Docker image
#   - Embedding model API key (e.g., OpenAI text-embedding-3-small)
# ============================================================================

set -euo pipefail

PROXY_URL="${PROXY_URL:-http://localhost:8000/v1}"
MODEL_NAME="${MODEL_NAME:-meta-llama/llama-3.1-8b-instruct}"
API_KEY="${OPENAI_API_KEY:-dummy}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-text-embedding-3-small}"
EMBEDDING_API_KEY="${EMBEDDING_API_KEY:-$API_KEY}"
BENCHMARK_DIR="${BENCHMARK_DIR:-$(dirname "$0")/livemcpbench_workspace}"
CONTAINER_NAME="LiveMCPBench_container"

echo "=== LiveMCPBench Benchmark ==="
echo "Proxy URL:       $PROXY_URL"
echo "Model:           $MODEL_NAME"
echo "Embedding Model: $EMBEDDING_MODEL"
echo "Container:       $CONTAINER_NAME"
echo ""

# --- Pre-flight checks ---
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is required but not installed."
    echo "Install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
    echo "WARNING: NVIDIA Docker runtime not detected."
    echo "LiveMCPBench requires --gpus all for embedding models."
    echo "Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/"
    echo ""
fi

# --- Step 1: Setup ---
if [ ! -d "$BENCHMARK_DIR" ]; then
    echo "[1/6] Cloning LiveMCPBench repository..."
    git clone https://github.com/icip-cas/LiveMCPBench.git "$BENCHMARK_DIR"
else
    echo "[1/6] LiveMCPBench workspace already exists."
fi

cd "$BENCHMARK_DIR"

# --- Step 2: Pull Docker image ---
echo "[2/6] Pulling Docker image..."
docker pull hysdhlx/livemcpbench:latest

# --- Step 3: Start container ---
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[3/6] Container already exists."
    docker start "$CONTAINER_NAME" 2>/dev/null || true
else
    echo "[3/6] Starting Docker container..."
    docker run -itd \
        -v "$(pwd):/outside" \
        --gpus all \
        --ipc=host --net=host \
        --name "$CONTAINER_NAME" \
        hysdhlx/livemcpbench:latest bash
fi

# --- Step 4: Configure .env ---
echo "[4/6] Configuring .env..."
cp .env_template .env

# Use host.docker.internal for macOS/Windows, localhost for Linux with --net=host
cat > .env << EOF
# MCP Copilot Agent Configuration
BASE_URL=$PROXY_URL
OPENAI_API_KEY=$API_KEY
MODEL=$MODEL_NAME

# Tool Retrieval Configuration
EMBEDDING_MODEL=$EMBEDDING_MODEL
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_API_KEY=$EMBEDDING_API_KEY
EMBEDDING_DIMENSIONS=1024
TOP_SERVERS=5
TOP_TOOLS=3
EOF

echo "  .env configured with proxy URL: $PROXY_URL"

# --- Step 5: Run inside container ---
echo "[5/6] Running benchmark inside container..."
echo ""

docker exec -it "$CONTAINER_NAME" bash -c "
    cd /LiveMCPBench/
    bash scripts/env_reset.sh
    bash ./tools/scripts/tool_check.sh
    uv run -m baseline.mcp_copilot.arg_generation
    bash ./baseline/scripts/run_example.sh
"

# --- Step 6: Evaluate ---
echo "[6/6] Running evaluation..."
docker exec -it "$CONTAINER_NAME" bash -c "
    cd /LiveMCPBench/
    bash ./evaluator/scripts/run_baseline.sh
    uv run ./evaluator/stat_success_rate.py --result_path ./evaluator/output/
"

echo ""
echo "=== LiveMCPBench Benchmark Complete ==="
echo "Trajectories: inside container at /LiveMCPBench/baseline/output/"
echo "Evaluation:   inside container at /LiveMCPBench/evaluator/output/"
echo ""
echo "To access results:"
echo "  docker exec -it $CONTAINER_NAME bash"
echo "  cd /LiveMCPBench/"
