"""
LLM service with tool-call orchestration loop.

Handles:
- Non-streaming completions with automatic tool execution
- Streaming completions with tool-call detection and execution
"""

import asyncio
import json
from collections.abc import AsyncGenerator

from app.core.openrouter_client import call_openrouter, stream_openrouter
from app.services.tool_executor import execute_tool_call
from app.tools.registry import TOOL_MAP
from app.logger_config import logger

MAX_TOOL_ROUNDS = 5


async def _run_one_tool_call(tc: dict) -> dict:
    """Execute a single tool_call, or return a 'tool not available' error message."""
    tc_id = tc.get("id", "")
    fn = tc.get("function") or {}
    fn_name = fn.get("name", "")
    fn_args = fn.get("arguments", "{}")

    if fn_name in TOOL_MAP:
        return await execute_tool_call(tc_id, fn_name, fn_args)

    logger.warning(f"Tool '{fn_name}' not registered locally, skipping")
    return {
        "role": "tool",
        "tool_call_id": tc_id,
        "content": json.dumps({"error": f"Tool '{fn_name}' not available"}),
    }


async def complete(payload: dict) -> dict:
    """
    Non-streaming completion with automatic tool-call loop.

    If the LLM returns tool_calls, executes them locally, appends results
    to messages, and calls the LLM again. Repeats up to MAX_TOOL_ROUNDS.

    Returns the final OpenRouter response dict.
    """
    messages = list(payload.get("messages", []))
    response = None

    for round_num in range(MAX_TOOL_ROUNDS):
        payload["messages"] = messages
        response = await call_openrouter(payload)

        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason", "")

        # Check if LLM wants to call tools
        tool_calls = message.get("tool_calls")
        if not tool_calls or finish_reason != "tool_calls":
            logger.info(f"Completion finished after {round_num + 1} round(s)")
            return response

        # If none of the requested tools are in our registry, return the raw
        # response so external benchmarks (BFCL, etc.) get the unmodified
        # tool_calls back.
        called_names = [tc.get("function", {}).get("name", "") for tc in tool_calls]
        if not any(name in TOOL_MAP for name in called_names):
            logger.info(
                f"No local tools matched ({called_names}), returning raw response"
            )
            return response

        # Append the assistant message with tool_calls
        messages.append(message)

        # Execute all tool calls in this round concurrently
        logger.info(f"Tool round {round_num + 1}: {len(tool_calls)} tool call(s)")
        tool_msgs = await asyncio.gather(*(_run_one_tool_call(tc) for tc in tool_calls))
        messages.extend(tool_msgs)

    logger.warning(f"Reached max tool rounds ({MAX_TOOL_ROUNDS})")

    # After max rounds, do a final call WITHOUT tools to force a text response
    payload["messages"] = messages
    payload.pop("tools", None)
    payload.pop("tool_choice", None)
    response = await call_openrouter(payload)
    logger.info("Final forced response after max tool rounds")
    return response


async def stream_complete(payload: dict) -> AsyncGenerator[str, None]:
    """
    Streaming completion with tool-call detection.

    Strategy:
    - Stream tokens to the client as they arrive
    - If the stream ends with finish_reason="tool_calls", buffer the tool_calls,
      execute them, then start a new streaming round with results appended
    - For the final round (no tool_calls), stream directly to client
    - If max tool rounds exhausted, do a final call without tools
    """
    messages = list(payload.get("messages", []))
    original_tools = payload.get("tools")
    original_tool_choice = payload.get("tool_choice")
    hit_max_rounds = False

    for round_num in range(MAX_TOOL_ROUNDS):
        payload["messages"] = messages
        payload["stream"] = True

        # Accumulate chunks to detect tool_calls
        collected_tool_calls: dict[int, dict] = {}  # index -> {id, function: {name, arguments}}
        collected_content = ""
        finish_reason = None
        is_tool_round = False

        async for data in stream_openrouter(payload):
            if data == "[DONE]":
                break

            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            choice = chunk.get("choices", [{}])[0]
            delta = choice.get("delta", {})
            chunk_finish = choice.get("finish_reason")

            if chunk_finish:
                finish_reason = chunk_finish

            # Check for tool_calls in the delta
            delta_tool_calls = delta.get("tool_calls")
            if delta_tool_calls:
                is_tool_round = True
                for tc_delta in delta_tool_calls:
                    idx = tc_delta.get("index", 0)
                    if idx not in collected_tool_calls:
                        collected_tool_calls[idx] = {
                            "id": tc_delta.get("id", ""),
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    entry = collected_tool_calls[idx]
                    if tc_delta.get("id"):
                        entry["id"] = tc_delta["id"]
                    fn_delta = tc_delta.get("function", {})
                    if fn_delta.get("name"):
                        entry["function"]["name"] += fn_delta["name"]
                    if fn_delta.get("arguments"):
                        entry["function"]["arguments"] += fn_delta["arguments"]

            # Track content
            if delta.get("content"):
                collected_content += delta["content"]

            # If this is NOT a tool round so far, stream the chunk to client
            if not is_tool_round:
                yield data

        # If no tool calls, we're done — stream was already sent to client
        if not is_tool_round or finish_reason != "tool_calls":
            yield "[DONE]"
            logger.info(f"Stream finished after {round_num + 1} round(s)")
            return

        # Execute the tool calls
        tool_calls_list = [collected_tool_calls[i] for i in sorted(collected_tool_calls.keys())]
        logger.info(f"Stream tool round {round_num + 1}: {len(tool_calls_list)} tool call(s)")

        # If none of the requested tools are in our registry, we should have
        # streamed the response as-is. Since we buffered it, re-emit now.
        called_names = [tc["function"]["name"] for tc in tool_calls_list]
        if not any(name in TOOL_MAP for name in called_names):
            logger.info(
                f"No local tools matched in stream ({called_names}), passing through"
            )
            yield "[DONE]"
            return

        # Build the assistant message with tool_calls
        assistant_msg = {"role": "assistant", "content": collected_content or None, "tool_calls": tool_calls_list}
        messages.append(assistant_msg)

        # Execute all tools in this round concurrently
        tool_msgs = await asyncio.gather(*(_run_one_tool_call(tc) for tc in tool_calls_list))
        messages.extend(tool_msgs)

        # Next round will stream the LLM's response after tool execution

    # Max rounds exhausted — do a final streaming call WITHOUT tools to force text response
    logger.warning(f"Stream reached max tool rounds ({MAX_TOOL_ROUNDS}), forcing final response")
    payload["messages"] = messages
    payload["stream"] = True
    payload.pop("tools", None)
    payload.pop("tool_choice", None)

    async for data in stream_openrouter(payload):
        if data == "[DONE]":
            break
        yield data

    yield "[DONE]"
    logger.info("Final forced stream response after max tool rounds")