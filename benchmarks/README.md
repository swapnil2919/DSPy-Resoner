# Benchmark Integration Guide

This directory contains setup and run scripts for integrating the OpenRouter Proxy with **6 industry-standard tool-calling benchmarks**.

## Prerequisites

- The OpenRouter Proxy must be running at `http://localhost:8000`
- Start it with: `uv run uvicorn app.main:app --port 8000`

## Quick Start

| Script | Framework | Setup Time | Notes |
|---|---|---|---|
| `run_bfcl.sh` | BFCL v3 + v4 | 30 min | Gold standard for function calling |
| `run_tau1.sh` | Tau1 (τ-bench) | 15 min | ⚠️ Deprecated — use τ³-bench |
| `run_tau2.sh` | Tau2 (τ³-bench) | 30 min | Latest: airline, retail, telecom, banking |
| `run_toolathlon.sh` | Toolathlon | 1-2 hrs | 600+ tools, long-horizon tasks |
| `run_livemcpbench.sh` | LiveMCPBench | 2-3 hrs | Docker + GPU required |

## Architecture

All frameworks use the proxy as an **OpenAI-compatible LLM gateway**:

```
Framework CLI → POST /v1/chat/completions → Proxy → OpenRouter → LLM
```

No proxy code changes are needed. Each framework is configured via environment variables to point at `http://localhost:8000/v1`.

## Internal Tiny Benchmark

For quick validation of the proxy's tool-calling pipeline without external dependencies:

```bash
# Run all 8 scenarios
curl -X POST http://localhost:8000/benchmarks/tiny \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/llama-3.1-8b-instruct"}'

# Run specific scenarios
curl -X POST http://localhost:8000/benchmarks/tiny \
  -H "Content-Type: application/json" \
  -d '{"scenario_ids": ["simple_weather", "no_tool_math"]}'

# List available scenarios
curl http://localhost:8000/benchmarks/scenarios
```
